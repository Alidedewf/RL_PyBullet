import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class SmallConvExtractor(nn.Module):
    """Более глубокий и стабильный conv-extractor для мелких изображений 64x64"""
    def __init__(self, in_channels, features_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute conv output size
        with th.no_grad():
            dummy = th.zeros(1, in_channels, 64, 64)
            n_flat = self.net(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
        self.output_dim = features_dim

    def forward(self, x):
        return self.fc(self.net(x))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Для Dict obs: "pixels" (C,H,W) и "proprioception" (7,)
    Возвращает concatenated features of size features_dim_pixels + proprio_dim
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        # calculate total features dim for BaseFeaturesExtractor ctor
        self._features_dim = None
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat = 0

        # pixels
        if "pixels" in observation_space.spaces:
            px_shape = observation_space.spaces["pixels"].shape  # (C,H,W)
            in_channels = px_shape[0]
            self.cnn = SmallConvExtractor(in_channels, features_dim=features_dim)
            extractors["pixels"] = self.cnn
            total_concat += self.cnn.output_dim

        # proprio
        if "proprioception" in observation_space.spaces:
            proprio_size = observation_space.spaces["proprioception"].shape[0]
            self.mlp = nn.Sequential(
                nn.Linear(proprio_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            extractors["proprioception"] = self.mlp
            total_concat += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations):
        # observations is a dict of tensors, pixels likely in channels_first already
        encoded = []
        # pixels
        if "pixels" in observations:
            x = observations["pixels"]
            # ensure float and proper shape already (B,C,H,W)
            if x.dtype != th.float32:
                x = x.float()
            encoded.append(self.extractors["pixels"](x))
        if "proprioception" in observations:
            p = observations["proprioception"]
            encoded.append(self.extractors["proprioception"](p))
        return th.cat(encoded, dim=1)