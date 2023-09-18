import dataclasses
from typing import Any, Callable, Generator, Optional, Union

import optree as ot
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.pytree_dataclass import OT_NAMESPACE as NS
from stable_baselines3.common.recurrent.type_aliases import (
    HiddenState,
    PyTreeGeneric,
    RecurrentRolloutBufferSamples,
    space_to_example,
)
from stable_baselines3.common.vec_env import VecNormalize


def index_into_pytree(
    idx: Any,
    tree: PyTreeGeneric,
    is_leaf: Optional[Union[bool, Callable[[PyTreeGeneric], bool]]] = None,
    none_is_leaf: bool = False,
    namespace: str = NS,
) -> PyTreeGeneric:
    return ot.tree_map(lambda x: x[idx], tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
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
        hidden_state_example: HiddenState,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_example = ot.tree_map(
            lambda x: th.zeros((), dtype=x.dtype, device=device).expand_as(x), hidden_state_example
        )
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

        batch_shape = (self.buffer_size, self.n_envs)
        action_dim = self.action_dim
        observation_space = self.observation_space
        hidden_state_example = self.hidden_state_example
        device = self.device

        self.data = RecurrentRolloutBufferSamples(
            observations=space_to_example(batch_shape, observation_space, device=device, ensure_non_batch_dim=True),
            actions=th.zeros((*batch_shape, action_dim), dtype=th.float32, device=device),
            old_values=th.zeros(batch_shape, dtype=th.float32, device=device),
            old_log_probs=th.zeros(batch_shape, dtype=th.float32, device=device),
            advantages=th.zeros(batch_shape, dtype=th.float32, device=device),
            returns=th.zeros(batch_shape, dtype=th.float32, device=device),
            hidden_states=ot.tree_map(
                lambda x: th.zeros(
                    (*batch_shape[:-1], x.shape[0], batch_shape[-1], x.shape[1:]), dtype=x.dtype, device=device
                ),
                hidden_state_example,
            ),
            episode_starts=th.zeros(batch_shape, dtype=th.float32, device=device),
            rewards=th.zeros(batch_shape, dtype=th.float32, device=device),
        )

    # Expose attributes of the RecurrentRolloutBufferData in the top-level to conform to the RolloutBuffer interface
    @property
    def episode_starts(self) -> th.Tensor:
        return self.data.episode_starts

    @property
    def values(self) -> th.Tensor:
        return self.data.old_values

    @property
    def rewards(self) -> th.Tensor:
        assert self.data.rewards is not None, "RecurrentRolloutBufferData should store rewards"
        return self.data.rewards

    @property
    def advantages(self) -> th.Tensor:
        return self.data.advantages

    @property
    def returns(self) -> th.Tensor:
        return self.data.returns

    @returns.setter
    def _set_returns(self, new_returns: th.Tensor):
        self.data.returns.copy_(new_returns, non_blocking=True)

    def reset(self):
        ot.tree_map(lambda x: x.zero_(), self.data, namespace=NS)
        super(RolloutBuffer, self).reset()

    def extend(self, *args) -> None:
        """
        Add a new batch of transitions to the buffer
        """

        # Do a for loop along the batch axis.
        # Treat lists as leaves to avoid flattening the infos.
        def _is_list(t):
            return isinstance(t, list)

        tensors, _ = ot.tree_flatten(args, is_leaf=_is_list, namespace=NS)
        len_tensors = len(tensors[0])
        assert all(len(t) == len_tensors for t in tensors), "All tensors must have the same batch size"
        for i in range(len_tensors):
            self.add(*index_into_pytree(i, args, is_leaf=_is_list, namespace=NS))

    def add(self, data: RecurrentRolloutBufferSamples, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        if data.rewards is None:
            raise ValueError("Recorded samples must contain a reward")
        new_data = dataclasses.replace(data, actions=data.actions.reshape((self.n_envs, self.action_dim)))
        ot.tree_map(lambda buf, x: buf[self.pos].copy_(x, non_blocking=True), self.data, new_data)
        # Increment pos
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Return everything, don't create minibatches
        if batch_size is None or batch_size == self.buffer_size * self.n_envs:
            yield self._get_samples(slice(None))
            return

        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size):
            yield self._get_samples(slice(start_idx, start_idx + batch_size, None))

    def _get_samples(
        self,
        batch_inds: Union[slice, th.Tensor],
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        data_without_reward: RecurrentRolloutBufferSamples = dataclasses.replace(self.data, rewards=None)
        return ot.tree_map(lambda tens: self.to_device(tens[batch_inds]), data_without_reward)
