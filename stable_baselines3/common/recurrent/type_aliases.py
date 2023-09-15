from typing import Iterable, NamedTuple, Optional, Sequence, Tuple

import torch as th
from gymnasium import spaces
from optree import PyTree
from stable_baselines3.common.pytree_dataclass import dataclass_frozen_pytree
from stable_baselines3.common.type_aliases import TensorDict

HiddenState = PyTree[th.Tensor]


def space_to_example(
    batch_shape: Tuple[int, ...], space: spaces.Space, *, device: Optional[th.device] = None
) -> PyTree[th.Tensor]:
    if isinstance(space, spaces.Box):
        return torch.zeros((*batch_shape, space.shape), dtype=th.float32, device=device)
    elif isinstance(space, spaces.Discrete):
        return torch.zeros((*batch_shape), dtype=th.int64, device=device)
    elif isinstance(space, spaces.Dict):
        return {k: space_to_example(v) for k, v in space.items()}
    elif isinstance(space, spaces.Tuple):
        return tuple(space_to_example(v) for v in space)
    else:
        raise TypeError(f"Unknown space type {type(space)} for {space}")


@dataclass_frozen_pytree
class RecurrentRolloutBufferSamples:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    hidden_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
