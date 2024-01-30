from typing import Generic, Tuple, TypeVar

import torch as th

from stable_baselines3.common.pytree_dataclass import FrozenPyTreeDataclass, TensorTree

TensorTreeT = TypeVar("TensorTreeT", bound=TensorTree)


LSTMRecurrentState = Tuple[th.Tensor, th.Tensor]
GRURecurrentState = th.Tensor


class ActorCriticStates(FrozenPyTreeDataclass[th.Tensor], Generic[TensorTreeT]):
    pi: TensorTreeT
    vf: TensorTreeT


class RecurrentRolloutBufferData(FrozenPyTreeDataclass[th.Tensor]):
    observations: TensorTree
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    values: th.Tensor
    log_probs: th.Tensor
    hidden_states: TensorTree


class RecurrentRolloutBufferSamples(FrozenPyTreeDataclass[th.Tensor]):
    observations: TensorTree
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    hidden_states: TensorTree
    episode_starts: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
