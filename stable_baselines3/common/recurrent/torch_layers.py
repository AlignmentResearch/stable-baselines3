import abc
import functools
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import gymnasium as gym
import torch as th

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.pytree_dataclass import TensorTree, tree_map
from stable_baselines3.common.recurrent.type_aliases import (
    GRURecurrentState,
    LSTMRecurrentState,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import TorchGymObs

RecurrentState = TypeVar("RecurrentState", bound=TensorTree)

RecurrentSubState = TypeVar("RecurrentSubState", bound=TensorTree)

ExtractorInput = TypeVar("ExtractorInput", bound=TorchGymObs)


class RecurrentFeaturesExtractor(BaseFeaturesExtractor, abc.ABC, Generic[ExtractorInput, RecurrentState]):
    @abc.abstractmethod
    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> RecurrentState:
        ...

    @abc.abstractmethod
    def forward(
        self, observations: ExtractorInput, state: RecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, RecurrentState]:
        ...

    @staticmethod
    def _process_sequence(
        rnn: th.nn.RNNBase, inputs: th.Tensor, init_state: RecurrentSubState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, RecurrentSubState]:
        if episode_starts.ndim == 1:
            seq_len = 1
            batch_sz = episode_starts.shape[0]
            inputs = inputs.unsqueeze(0)
            episode_starts = episode_starts.unsqueeze(0)
            squeeze_end = True
        else:
            assert episode_starts.ndim == 2
            seq_len, batch_sz = episode_starts.shape
            squeeze_end = False

        def _reset_state_component(reset_mask: th.Tensor, state: th.Tensor) -> th.Tensor:
            assert state.shape == (rnn.num_layers, batch_sz, rnn.hidden_size)
            return state * reset_mask

        if th.any(episode_starts[1:]):
            state_is_not_reset = (~episode_starts).contiguous().view((seq_len, 1, batch_sz, 1))

            rnn_output_list: list[th.Tensor] = [None] * seq_len  # type: ignore
            end_state = init_state
            for t in range(seq_len):
                end_state = tree_map(functools.partial(_reset_state_component, state_is_not_reset[t]), end_state)
                rnn_output_list[t], end_state = rnn(inputs[t, None], end_state)
            rnn_output = th.cat(rnn_output_list, dim=0)
        else:
            first_state_is_not_reset = (~episode_starts[0]).contiguous().view((1, batch_sz, 1))
            init_state = tree_map(lambda s: _reset_state_component(first_state_is_not_reset, s), init_state)
            rnn_output, end_state = rnn(inputs, init_state)

        if squeeze_end:
            rnn_output = rnn_output.squeeze(0)
        return rnn_output, end_state


class GRUWrappedFeaturesExtractor(RecurrentFeaturesExtractor[ExtractorInput, GRURecurrentState], Generic[ExtractorInput]):
    def __init__(
        self,
        observation_space: gym.Space,
        base_extractor: BaseFeaturesExtractor,
        features_dim: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        if features_dim is None:
            # Ensure features_dim is at least 64 by default so it optimizes fine
            features_dim = max(base_extractor.features_dim, 64)

        assert observation_space == base_extractor._observation_space

        super().__init__(observation_space, features_dim)
        self.base_extractor = base_extractor

        self.rnn = th.nn.GRU(
            input_size=base_extractor.features_dim,
            hidden_size=features_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            dropout=dropout,
            bidirectional=False,
        )

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> GRURecurrentState:
        shape: Tuple[int, ...]
        if n_envs is None:
            shape = (self.rnn.num_layers, self.rnn.hidden_size)
        else:
            shape = (self.rnn.num_layers, n_envs, self.rnn.hidden_size)
        return th.zeros(shape, device=device)

    def forward(
        self, observations: ExtractorInput, state: GRURecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, GRURecurrentState]:
        observations = tree_map(lambda x: x.view(-1, *x.shape[episode_starts.ndim :]), observations)  # type: ignore
        features: th.Tensor = self.base_extractor(observations)
        features = features.view(*episode_starts.shape, -1)
        return self._process_sequence(self.rnn, features, state, episode_starts)

    @property
    def features_dim(self) -> int:
        return self.rnn.hidden_size


class GRUFlattenExtractor(GRUWrappedFeaturesExtractor[th.Tensor]):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = FlattenExtractor(observation_space)
        super().__init__(
            observation_space, base_extractor, features_dim=features_dim, num_layers=num_layers, bias=bias, dropout=dropout
        )


class GRUNatureCNNExtractor(GRUWrappedFeaturesExtractor[th.Tensor]):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = NatureCNN(observation_space, features_dim=features_dim, normalized_image=normalized_image)
        super().__init__(
            observation_space, base_extractor, features_dim=features_dim, num_layers=num_layers, bias=bias, dropout=dropout
        )


class GRUCombinedExtractor(GRUWrappedFeaturesExtractor[Dict[Any, th.Tensor]]):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = CombinedExtractor(observation_space, cnn_output_dim=cnn_output_dim, normalized_image=normalized_image)
        super().__init__(
            observation_space, base_extractor, features_dim=features_dim, num_layers=num_layers, bias=bias, dropout=dropout
        )


class LSTMFlattenExtractor(RecurrentFeaturesExtractor[th.Tensor, LSTMRecurrentState]):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        self.rnn = th.nn.LSTM(
            input_size=get_flattened_obs_dim(self._observation_space),
            hidden_size=features_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            dropout=dropout,
            bidirectional=False,
        )
        self.base_extractor = FlattenExtractor(observation_space)

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> LSTMRecurrentState:
        shape: Tuple[int, ...]
        if n_envs is None:
            shape = (self.rnn.num_layers, self.rnn.hidden_size)
        else:
            shape = (self.rnn.num_layers, n_envs, self.rnn.hidden_size)
        return (th.zeros(shape, device=device), th.zeros(shape, device=device))

    def forward(
        self, observations: th.Tensor, state: LSTMRecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, LSTMRecurrentState]:
        observations = observations.view(-1, *observations.shape[episode_starts.ndim :])
        features: th.Tensor = self.base_extractor(observations)
        features = features.view(*episode_starts.shape, -1)
        return self._process_sequence(self.rnn, features, state, episode_starts)

    @property
    def features_dim(self) -> int:
        return self.rnn.hidden_size
