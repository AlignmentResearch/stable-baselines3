import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.vec_env.util import as_torch_dtype, obs_as_tensor

compute_stacking = StackedObservations.compute_stacking
NUM_ENVS = 2
N_STACK = 4
H, W, C = 16, 24, 3


def test_compute_stacking_box():
    space = spaces.Box(-1, 1, (4,))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(N_STACK, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (N_STACK * 4,)
    assert repeat_axis == -1


def test_compute_stacking_multidim_box():
    space = spaces.Box(-1, 1, (4, 5))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(N_STACK, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (4, N_STACK * 5)
    assert repeat_axis == -1


def test_compute_stacking_multidim_box_channel_first():
    space = spaces.Box(-1, 1, (4, 5))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        N_STACK, observation_space=space, channels_order="first"
    )
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (N_STACK * 4, 5)
    assert repeat_axis == 0


def test_compute_stacking_image_channel_first():
    """Detect that image is channel first and stack in that dimension."""
    space = spaces.Box(0, 255, (C, H, W), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(N_STACK, observation_space=space)
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (N_STACK * C, H, W)
    assert repeat_axis == 0


def test_compute_stacking_image_channel_last():
    """Detect that image is channel last and stack in that dimension."""
    space = spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(N_STACK, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (H, W, N_STACK * C)
    assert repeat_axis == -1


def test_compute_stacking_image_channel_first_stack_last():
    """Detect that image is channel first and stack in that dimension."""
    space = spaces.Box(0, 255, (C, H, W), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        N_STACK, observation_space=space, channels_order="last"
    )
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (C, H, N_STACK * W)
    assert repeat_axis == -1


def test_compute_stacking_image_channel_last_stack_first():
    """Detect that image is channel last and stack in that dimension."""
    space = spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        N_STACK, observation_space=space, channels_order="first"
    )
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (N_STACK * H, W, C)
    assert repeat_axis == 0


def test_reset_update_box():
    space = spaces.Box(-1, 1, (4,))
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space)
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * 4)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * 4)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=-1),
    )


def test_reset_update_multidim_box():
    space = spaces.Box(-1, 1, (4, 5))
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space)
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, 4, N_STACK * 5)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, 4, N_STACK * 5)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=-1),
    )


def test_reset_update_multidim_box_channel_first():
    space = spaces.Box(-1, 1, (4, 5))
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space, channels_order="first")
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * 4, 5)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * 4, 5)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_image_channel_first():
    space = spaces.Box(0, 255, (C, H, W), dtype=np.uint8)
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space)
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * C, H, W)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * C, H, W)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_image_channel_last():
    space = spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space)
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, H, W, N_STACK * C)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, H, W, N_STACK * C)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=-1),
    )


def test_reset_update_image_channel_first_stack_last():
    space = spaces.Box(0, 255, (C, H, W), dtype=np.uint8)
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space, channels_order="last")
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, C, H, N_STACK * W)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = np.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, C, H, N_STACK * W)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert np.array_equal(
        stacked_obs,
        np.concatenate(
            (np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=-1
        ),
    )


def test_reset_update_image_channel_last_stack_first():
    space = spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space, channels_order="first")
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * H, W, C)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (NUM_ENVS, N_STACK * H, W, C)
    assert stacked_obs.dtype == as_torch_dtype(space.dtype)
    assert th.equal(
        stacked_obs,
        th.cat((th.zeros_like(observations_1), th.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_dict():
    space = spaces.Dict({"key1": spaces.Box(0, 255, (H, W, C), dtype=np.uint8), "key2": spaces.Box(-1, 1, (4, 5))})
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space, channels_order={"key1": "first", "key2": "last"})
    observations_1 = obs_as_tensor(
        {key: np.stack([subspace.sample() for _ in range(NUM_ENVS)]) for key, subspace in space.spaces.items()}, None
    )
    stacked_obs = stacked_observations.reset(observations_1)
    assert isinstance(stacked_obs, dict)
    assert stacked_obs["key1"].shape == (NUM_ENVS, N_STACK * H, W, C)
    assert stacked_obs["key2"].shape == (NUM_ENVS, 4, N_STACK * 5)
    assert stacked_obs["key1"].dtype == as_torch_dtype(space["key1"].dtype)
    assert stacked_obs["key2"].dtype == as_torch_dtype(space["key2"].dtype)
    observations_2 = obs_as_tensor(
        {key: np.stack([subspace.sample() for _ in range(NUM_ENVS)]) for key, subspace in space.spaces.items()}, None
    )
    dones = np.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs["key1"].shape == (NUM_ENVS, N_STACK * H, W, C)
    assert stacked_obs["key2"].shape == (NUM_ENVS, 4, N_STACK * 5)
    assert stacked_obs["key1"].dtype == as_torch_dtype(space["key1"].dtype)
    assert stacked_obs["key2"].dtype == as_torch_dtype(space["key2"].dtype)

    assert th.equal(
        stacked_obs["key1"],
        th.cat(
            (
                th.zeros_like(observations_1["key1"]),
                th.zeros_like(observations_1["key1"]),
                observations_1["key1"],
                observations_2["key1"],
            ),
            axis=1,
        ),
    )
    assert th.equal(
        stacked_obs["key2"],
        th.cat(
            (
                th.zeros_like(observations_1["key2"]),
                th.zeros_like(observations_1["key2"]),
                observations_1["key2"],
                observations_2["key2"],
            ),
            axis=-1,
        ),
    )


def test_episode_termination_box():
    space = spaces.Box(-1, 1, (4,))
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space)
    observations_1 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_observations.reset(observations_1)
    observations_2 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_observations.update(observations_2, dones, infos)
    terminal_observation = obs_as_tensor(space.sample(), None)
    infos[1]["terminal_observation"] = terminal_observation  # episode termination in env1
    dones[1] = True
    observations_3 = obs_as_tensor(np.stack([space.sample() for _ in range(NUM_ENVS)]), None)
    stacked_obs, infos = stacked_observations.update(observations_3, dones, infos)
    zeros = th.zeros_like(observations_1[0])
    true_stacked_obs_env1 = th.cat((zeros, observations_1[0], observations_2[0], observations_3[0]), axis=-1)
    true_stacked_obs_env2 = th.cat((zeros, zeros, zeros, observations_3[1]), axis=-1)
    true_stacked_obs = th.stack((true_stacked_obs_env1, true_stacked_obs_env2))
    assert th.equal(true_stacked_obs, stacked_obs)


def test_episode_termination_dict():
    space = spaces.Dict({"key1": spaces.Box(0, 255, (H, W, 3), dtype=np.uint8), "key2": spaces.Box(-1, 1, (4, 5))})
    stacked_observations = StackedObservations(NUM_ENVS, N_STACK, space, channels_order={"key1": "first", "key2": "last"})
    observations_1 = obs_as_tensor(
        {key: np.stack([subspace.sample() for _ in range(NUM_ENVS)]) for key, subspace in space.spaces.items()}
    )
    stacked_observations.reset(observations_1)
    observations_2 = obs_as_tensor(
        {key: np.stack([subspace.sample() for _ in range(NUM_ENVS)]) for key, subspace in space.spaces.items()}
    )
    dones = th.zeros((NUM_ENVS,), dtype=bool)
    infos = [{} for _ in range(NUM_ENVS)]
    stacked_observations.update(observations_2, dones, infos)
    terminal_observation = obs_as_tensor(space.sample(), None)
    infos[1]["terminal_observation"] = terminal_observation  # episode termination in env1
    dones[1] = True
    observations_3 = obs_as_tensor(
        {key: np.stack([subspace.sample() for _ in range(NUM_ENVS)]) for key, subspace in space.spaces.items()}
    )
    stacked_obs, infos = stacked_observations.update(observations_3, dones, infos)

    for key, axis in zip(observations_1.keys(), [0, -1]):
        zeros = th.zeros_like(observations_1[key][0])
        true_stacked_obs_env1 = th.cat(
            (zeros, observations_1[key][0], observations_2[key][0], observations_3[key][0]), axis=axis
        )
        true_stacked_obs_env2 = th.cat((zeros, zeros, zeros, observations_3[key][1]), axis=axis)
        true_stacked_obs = th.stack((true_stacked_obs_env1, true_stacked_obs_env2))
        assert th.equal(true_stacked_obs, stacked_obs[key])
