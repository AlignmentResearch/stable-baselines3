from typing import Optional, Tuple, TypeVar

import torch as th
from gymnasium import spaces
from optree import PyTree

from stable_baselines3.common.pytree_dataclass import dataclass_frozen_pytree

HiddenState = PyTree[th.Tensor]


PyTreeGeneric = TypeVar("PyTreeGeneric", bound=PyTree)


@dataclass_frozen_pytree
class RecurrentRolloutBufferData:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    values: th.Tensor
    log_probs: th.Tensor
    hidden_states: HiddenState


@dataclass_frozen_pytree
class RecurrentRolloutBufferSamples:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    episode_starts: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    hidden_states: HiddenState
    advantages: th.Tensor
    returns: th.Tensor
