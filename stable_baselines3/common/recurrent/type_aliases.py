from typing import NamedTuple, Tuple

import torch as th

from stable_baselines3.common.pytree_dataclass import PyTree, dataclass_frozen_pytree


class RNNStates(NamedTuple):
    pi: Tuple[th.Tensor, ...]
    vf: Tuple[th.Tensor, ...]


@dataclass_frozen_pytree
class RecurrentRolloutBufferData:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    values: th.Tensor
    log_probs: th.Tensor
    lstm_states: PyTree[th.Tensor]


@dataclass_frozen_pytree
class RecurrentRolloutBufferSamples:
    observations: PyTree[th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    lstm_states: PyTree[th.Tensor]
    episode_starts: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    mask: th.Tensor


RecurrentDictRolloutBufferSamples = RecurrentRolloutBufferSamples
