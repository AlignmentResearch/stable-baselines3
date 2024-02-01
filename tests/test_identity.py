from typing import Any

import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, RecurrentPPO
from stable_baselines3.common.envs import (
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

DIM = 4

# This test used to be flaky because IdentityEnvs weren't properly seeded, but commit 56ba245a solved that.
#
# Now the test is consistent between different runs on the *same* machine, but its results for RecurrentPPO can vary
# quite a lot between machines; presumably because of the LSTM operation.
#
# Here are some example results. You can see they're consistent between runs on the same machine (Apple M2 docker), and
# only the RecurrentPPO results vary between machines.
#
# * Apple M2: Docker on x86_64 VM (Rosetta)
# ** 1
# FAILED tests/test_identity.py::test_discrete[env1-A2C] - AssertionError: Mean reward below threshold: 93.10 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-PPO] - AssertionError: Mean reward below threshold: 71.05 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-RecurrentPPO] - AssertionError: Mean reward below threshold: 12.70 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-A2C] - AssertionError: Mean reward below threshold: 25.50 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-PPO] - AssertionError: Mean reward below threshold: 5.90 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-RecurrentPPO] - AssertionError: Mean reward below threshold: 9.00 < 99.00

# ** 2
# FAILED tests/test_identity.py::test_discrete[env1-A2C] - AssertionError: Mean reward below threshold: 93.10 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-PPO] - AssertionError: Mean reward below threshold: 71.05 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-RecurrentPPO] - AssertionError: Mean reward below threshold: 12.70 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-A2C] - AssertionError: Mean reward below threshold: 25.50 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-PPO] - AssertionError: Mean reward below threshold: 5.90 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-RecurrentPPO] - AssertionError: Mean reward below threshold: 9.00 < 99.00

# ** Apple M2: native, non-virtualized
# FAILED tests/test_identity.py::test_discrete[env1-A2C] - AssertionError: Mean reward below threshold: 93.65 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-A2C] - AssertionError: Mean reward below threshold: 25.50 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-PPO] - AssertionError: Mean reward below threshold: 5.90 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-RecurrentPPO] - AssertionError: Mean reward below threshold: 8.90 < 99.00

# * AMD EPYC: Flamingo
# FAILED tests/test_identity.py::test_discrete[env1-A2C] - AssertionError: Mean reward below threshold: 93.10 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-PPO] - AssertionError: Mean reward below threshold: 71.05 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-RecurrentPPO] - AssertionError: Mean reward below threshold: 36.40 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-A2C] - AssertionError: Mean reward below threshold: 25.50 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-PPO] - AssertionError: Mean reward below threshold: 5.90 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-RecurrentPPO] - AssertionError: Mean reward below threshold: 7.15 < 99.00

# * CircleCI (Intel?)
# FAILED tests/test_identity.py::test_discrete[env1-A2C] - AssertionError: Mean reward below threshold: 93.10 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-PPO] - AssertionError: Mean reward below threshold: 71.05 < 99.00
# FAILED tests/test_identity.py::test_discrete[env1-RecurrentPPO] - AssertionError: Mean reward below threshold: 54.70 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-A2C] - AssertionError: Mean reward below threshold: 25.50 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-PPO] - AssertionError: Mean reward below threshold: 5.90 < 99.00
# FAILED tests/test_identity.py::test_discrete[env2-RecurrentPPO] - AssertionError: Mean reward below threshold: 6.20 < 99.00


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN, RecurrentPPO])
@pytest.mark.parametrize(
    "env_fn", [lambda: IdentityEnv(DIM), lambda: IdentityEnvMultiDiscrete(DIM), lambda: IdentityEnvMultiBinary(DIM)]
)
def test_discrete(model_class, env_fn):
    # Use multiple envs so we can test that batching works correctly
    env_ = DummyVecEnv([env_fn] * 4)
    kwargs: dict[str, Any] = dict()
    total_n_steps = 10000
    if model_class == DQN:
        kwargs = dict(learning_starts=0)
        # DQN only support discrete actions
        if isinstance(env_.envs[0], (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return

    if model_class in (RecurrentPPO, PPO):
        kwargs["target_kl"] = 0.02
        kwargs["n_epochs"] = 30

    if model_class == RecurrentPPO:
        # Ensure that there's not an MLP on top of the LSTM that the default Policy creates.
        kwargs["policy_kwargs"] = dict(net_arch=dict(vf=[], pi=[]))
        kwargs["batch_time"] = 128
        kwargs["batch_envs"] = 4
        kwargs["n_steps"] = 256

    model = model_class("MlpPolicy", env_, gamma=0.4, seed=3, **kwargs).learn(total_n_steps)

    evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=99, warn=False)
    obs, _ = env_.envs[0].reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, DDPG, TD3, RecurrentPPO])
def test_continuous(model_class):
    env = IdentityEnvBox(eps=0.5)

    total_n_steps = 2000 if issubclass(model_class, OnPolicyAlgorithm) else 400

    kwargs: dict[str, Any] = dict(policy_kwargs=dict(net_arch=[64, 64]), seed=0, gamma=0.95)

    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise
    elif model_class in [A2C]:
        kwargs["policy_kwargs"]["log_std_init"] = -0.5
    elif model_class == PPO:
        kwargs = dict(n_steps=512, n_epochs=5)
    elif model_class == RecurrentPPO:
        # Ensure that there's not an MLP on top of the LSTM that the default Policy creates.
        kwargs["policy_kwargs"]["net_arch"] = dict(vf=[], pi=[])

    model = model_class("MlpPolicy", env, learning_rate=1e-3, **kwargs).learn(total_n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)
