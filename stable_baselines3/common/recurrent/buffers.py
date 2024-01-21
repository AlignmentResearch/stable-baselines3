import dataclasses
import logging
from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.pytree_dataclass import (
    TensorTree,
    tree_flatten,
    tree_index,
    tree_map,
)
from stable_baselines3.common.recurrent.type_aliases import (
    RecurrentRolloutBufferData,
    RecurrentRolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env.util import as_torch_dtype

log = logging.getLogger(__name__)


def space_to_example(
    batch_shape: Tuple[int, ...],
    space: spaces.Space,
    *,
    device: Optional[th.device] = None,
    ensure_non_batch_dim: bool = False,
) -> TensorTree:
    def _zeros_with_batch(x: np.ndarray) -> th.Tensor:
        shape = x.shape
        if ensure_non_batch_dim and len(shape) == 0:
            shape = (1,)
        return th.zeros((*batch_shape, *shape), device=device, dtype=as_torch_dtype(x.dtype))

    return tree_map(_zeros_with_batch, space.sample())


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the RNN states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_example: Example buffer that will collect RNN states.
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_example: TensorTree,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)
        self.hidden_state_example = hidden_state_example
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        batch_shape = (self.buffer_size, self.n_envs)
        self.device = device = get_device(device)

        self.observation_space_example = space_to_example((), observation_space)

        self.advantages = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.returns = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.data = RecurrentRolloutBufferData(
            observations=space_to_example(batch_shape, self.observation_space, device=device, ensure_non_batch_dim=True),
            actions=th.zeros(
                (*batch_shape, self.action_dim),
                dtype=th.long if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)) else th.float32,
                device=device,
            ),
            rewards=th.zeros(batch_shape, dtype=th.float32, device=device),
            episode_starts=th.zeros(batch_shape, dtype=th.bool, device=device),
            values=th.zeros(batch_shape, dtype=th.float32, device=device),
            log_probs=th.zeros(batch_shape, dtype=th.float32, device=device),
            hidden_states=tree_map(
                lambda x: th.zeros((self.buffer_size, *x.shape), dtype=x.dtype, device=device), hidden_state_example
            ),
        )

    # Expose attributes of the RecurrentRolloutBufferData in the top-level to conform to the RolloutBuffer interface
    @property
    def episode_starts(self) -> th.Tensor:  # type: ignore[override]
        return self.data.episode_starts

    @property
    def values(self) -> th.Tensor:  # type: ignore[override]
        return self.data.values

    @property
    def rewards(self) -> th.Tensor:  # type: ignore[override]
        return self.data.rewards

    def reset(self):
        self.returns.zero_()
        self.advantages.zero_()
        tree_map(lambda x: x.zero_(), self.data)
        super(RolloutBuffer, self).reset()

    def extend(self, data: RecurrentRolloutBufferData) -> None:  # type: ignore[override]
        """
        Add a new batch of transitions to the buffer
        """

        # Do a for loop along the batch axis.
        # Treat lists as leaves to avoid flattening the infos.
        def _is_list(t):
            return isinstance(t, list)

        tensors: list[th.Tensor]
        tensors, _ = tree_flatten(data, is_leaf=_is_list)  # type: ignore
        len_tensors = len(tensors[0])
        assert all(len(t) == len_tensors for t in tensors), "All tensors must have the same batch size"
        for i in range(len_tensors):
            self.add(tree_index(data, i, is_leaf=_is_list))

    def add(self, data: RecurrentRolloutBufferData, **kwargs) -> None:  # type: ignore[override]
        """
        :param hidden_states: Hidden state of the RNN
        """
        new_data = dataclasses.replace(
            data, actions=data.actions.reshape((self.n_envs, self.action_dim))  # type: ignore[misc]
        )

        tree_map(
            lambda buf, x: buf[self.pos].copy_(x if x.ndim + 1 == buf.ndim else x.unsqueeze(-1), non_blocking=True),
            self.data,
            new_data,
        )
        # Increment pos
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self, batch_shape: Optional[tuple[int, int]] = None
    ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Return everything, don't create minibatches
        if batch_shape is None:
            batch_shape = (self.buffer_size, self.n_envs)

        batch_time, batch_envs = batch_shape

        if self.buffer_size % batch_time != 0:
            raise ValueError(f"batch_size[0] must evenly divide sequence length, but {batch_time=} and {self.buffer_size=}")
        if self.n_envs % batch_envs != 0:
            raise ValueError(f"batch_size[1] must evenly divide n_envs, but {batch_envs=} and {self.n_envs=}")

        if batch_envs >= self.n_envs:
            for time_start in range(0, self.buffer_size, batch_time):
                yield self._get_samples(slice(None), slice(time_start, time_start + batch_time))

        else:
            env_indices = th.randperm(self.n_envs)
            for time_start in range(0, self.buffer_size, batch_time):
                for env_start in range(0, self.n_envs, batch_envs):
                    yield self._get_samples(
                        env_indices[env_start : env_start + batch_envs], slice(time_start, time_start + batch_time)
                    )

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: Union[slice, th.Tensor],
        seq_inds: slice,
    ) -> RecurrentRolloutBufferSamples:
        idx = (seq_inds, batch_inds)
        # hidden_states: time, n_layers, batch
        first_hidden_state_idx = (range(seq_inds.start, seq_inds.stop, 1)[0], slice(None), batch_inds)

        return RecurrentRolloutBufferSamples(
            observations=tree_index(self.data.observations, idx),
            actions=self.data.actions[idx],
            old_values=self.data.values[idx],
            old_log_prob=self.data.log_probs[idx],
            advantages=self.advantages[idx],
            returns=self.returns[idx],
            hidden_states=tree_index(self.data.hidden_states, first_hidden_state_idx),
            episode_starts=self.data.episode_starts[idx],
        )


RecurrentDictRolloutBuffer = RecurrentRolloutBuffer
