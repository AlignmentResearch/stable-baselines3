import copy
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


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN, RecurrentPPO])
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, env):
    if model_class == DQN:
        TOTAL_TIMESTEPS = 2500
        env_ = DummyVecEnv([lambda: copy.deepcopy(env)])
        kwargs = dict(learning_starts=0)
        # DQN only support discrete actions
        if isinstance(env, (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return
    else:
        TOTAL_TIMESTEPS = 2500
        CONCURRENT_ROLLOUT_STEPS = 1 if model_class == A2C else 2
        SEQUENTIAL_ROLLOUT_STEPS = 64 if model_class == A2C else 32
        env_ = DummyVecEnv([lambda: copy.deepcopy(env)] * CONCURRENT_ROLLOUT_STEPS)
        kwargs: dict[str, Any] = dict(
            n_steps=SEQUENTIAL_ROLLOUT_STEPS, learning_rate=1e-3, policy_kwargs=dict(net_arch=dict(pi=[], vf=[]))
        )

        if model_class == PPO:
            kwargs["batch_size"] = CONCURRENT_ROLLOUT_STEPS * SEQUENTIAL_ROLLOUT_STEPS

        if model_class == RecurrentPPO:
            kwargs["batch_size"] = CONCURRENT_ROLLOUT_STEPS * SEQUENTIAL_ROLLOUT_STEPS
            kwargs["policy_kwargs"].update(dict(lstm_hidden_size=32))

    model = model_class("MlpPolicy", env_, gamma=0.4, seed=3, **kwargs).learn(TOTAL_TIMESTEPS)
    mean_reward, std_reward = evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=99, warn=False)
    print(f"{mean_reward=}, {std_reward=}")
    obs, _ = env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, DDPG, TD3, RecurrentPPO])
def test_continuous(model_class):
    env = IdentityEnvBox(eps=0.5)

    n_steps = 2000 if issubclass(model_class, OnPolicyAlgorithm) else 400

    kwargs = dict(policy_kwargs=dict(net_arch=[64, 64]), gamma=0.95)

    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise
    elif model_class in [A2C]:
        kwargs["policy_kwargs"]["log_std_init"] = -0.5
    elif model_class == PPO:
        kwargs = dict(n_steps=512, n_epochs=5)
    elif model_class == RecurrentPPO:
        kwargs = dict(policy_kwargs=dict(net_arch=[], lstm_hidden_size=32), gae_lambda=0.4, n_steps=8)

    model = model_class("MlpPolicy", env, learning_rate=1e-3, seed=1, **kwargs).learn(n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)
