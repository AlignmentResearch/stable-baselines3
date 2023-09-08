"""Common aliases for type hints"""

from enum import Enum
from typing import Any, Callable, Dict, Generic, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.pytree_dataclass import dataclass_frozen_pytree
import torch as th
from optree import PyTree
import optree as ot

from stable_baselines3.common import callbacks, vec_env

GymEnv = Union[gym.Env, vec_env.VecEnv]
GymObs = Union[Tuple["GymObs", ...], Dict[str, "GymObs"], np.ndarray, int]
TorchGymObs = Union[Tuple["TorchGymObs", ...], Dict[str, "TorchGymObs"], th.Tensor, int]
GymResetReturn = Tuple[GymObs, Dict]
AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
TensorDict = Dict[str, th.Tensor]
TensorIndex = Union[int, slice, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]

EMPTY_PYTREE: PyTree[th.Tensor] = ()  # type: ignore[assignment]

T = TypeVar("T")

@dataclass_frozen_pytree
class OutAndState(Generic[T]):
    out: T
    state: PyTree[th.Tensor]

    def apply(self, func: Callable[[T], T]) -> "OutAndState[T]":
        return OutAndState(func(self.out), self.state)

    def discard_state(self, exception: Exception) -> T:
        def _error(_x):
            raise exception

        ot.tree_map(_error, self.state, namespace="stable-baselines3")
        return self.out


@dataclass_frozen_pytree
class RolloutBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    extractor_states: PyTree[th.Tensor]


@dataclass_frozen_pytree
class DictRolloutBufferSamples:
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    extractor_states: PyTree[th.Tensor]


@dataclass_frozen_pytree
class ReplayBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    extractor_states: PyTree[th.Tensor]


@dataclass_frozen_pytree
class DictReplayBufferSamples:
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    extractor_states: PyTree[th.Tensor]


@dataclass_frozen_pytree
class RolloutReturn:
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: PyTree[th.Tensor],
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> OutAndState[th.Tensor]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

    def initial_state(self, n_envs: Optional[int]) -> PyTree[th.Tensor]:
        """
        Get the initial recurrent states for the model.
        Used in recurrent policies.

        :param n_envs: Batch dimension of the recurrent state. If None, states are not batched.
        :return: the initial recurrent states
        """

def unwrap(x: Optional[T]) -> T:
    if x is None:
        raise ValueError("Expected a value, got None")
    return x
