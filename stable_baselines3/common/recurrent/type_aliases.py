from typing import NamedTuple, Tuple

import torch as th

from stable_baselines3.common.pytree_dataclass import PyTreeDataclass, TensorTree


class RNNStates(PyTreeDataclass[th.Tensor]):
    pi: TensorTree
    vf: TensorTree


class RecurrentRolloutBufferData(PyTreeDataclass[th.Tensor]):
    observations: TensorTree
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    values: th.Tensor
    log_probs: th.Tensor
    lstm_states: TensorTree


class RecurrentRolloutBufferSamples(PyTreeDataclass[th.Tensor]):
    observations: TensorTree
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    lstm_states: TensorTree
    episode_starts: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    mask: th.Tensor


RecurrentDictRolloutBufferSamples = RecurrentRolloutBufferSamples
