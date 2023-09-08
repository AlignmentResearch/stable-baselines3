import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan


class NanAndInfEnv(gym.Env):
    """Custom Environment that raised NaNs and Infs"""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    @staticmethod
    def step(action):
        if np.all(np.array(action) > 0):
            obs = float("NaN")
        elif np.all(np.array(action) < 0):
            obs = float("inf")
        else:
            obs = 0
        return [obs], 0.0, False, False, {}

    @staticmethod
    def reset(seed=None):
        return [0.0], {}

    def render(self):
        pass


def test_check_nan():
    """Test VecCheckNan Object"""

    env = DummyVecEnv([NanAndInfEnv])
    env = VecCheckNan(env, raise_exception=True)

    env.step(th.tensor([[0.0]]))

    with pytest.raises(ValueError):
        env.step(th.tensor([[float("NaN")]]))

    with pytest.raises(ValueError):
        env.step(th.tensor([[float("inf")]]))

    with pytest.raises(ValueError):
        env.step(th.tensor([[-1.0]]))

    with pytest.raises(ValueError):
        env.step(th.tensor([[1.0]]))

    env.step(th.tensor([[0.0, 1.0], [0.0, 1.0]]))

    env.reset()
