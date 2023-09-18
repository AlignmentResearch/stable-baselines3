from typing import Optional, Tuple, TypeVar

import torch as th
from gymnasium import spaces
from optree import PyTree

from stable_baselines3.common.pytree_dataclass import dataclass_frozen_pytree

HiddenState = PyTree[th.Tensor]


PyTreeGeneric = TypeVar("PyTreeGeneric", bound=PyTree)


def space_to_example(
    batch_shape: Tuple[int, ...],
    space: spaces.Space,
    *,
    device: Optional[th.device] = None,
    ensure_non_batch_dim: bool = False,
) -> PyTree[th.Tensor]:
    if isinstance(space, spaces.Dict):
        return {k: space_to_example(v, device=device, ensure_non_batch_dim=ensure_non_batch_dim) for k, v in space.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(space_to_example(v, device=device, ensure_non_batch_dim=ensure_non_batch_dim) for v in space)

    if isinstance(space, spaces.Box):
        space_shape = space.shape
    elif isinstance(space, spaces.Discrete):
        space_shape = ()
    else:
        raise TypeError(f"Unknown space type {type(space)} for {space}")

    if ensure_non_batch_dim and space_shape:
        space_shape = (1,)
    return th.zeros((*batch_shape, *space_shape), dtype=th.float32, device=device)


@functools.partial(dataclass_frozen_pytree, frozen=False)
class RecurrentRolloutBufferSamples:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    hidden_states: HiddenState
    episode_starts: th.Tensor
    rewards: Optional[th.Tensor] = None
