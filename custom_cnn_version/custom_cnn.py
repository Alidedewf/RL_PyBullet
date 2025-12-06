import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# NatureCNN Architecture (Требование 4, Вариант А)
class NatureCNN(nn.Module):
    def __init__(self, input_shape, features_dim=512):
        super().__init__()
        
        # Вход: (Batch, Channels=1, Height=64, Width=64)
        n_channels = input_shape[0] # C=1
        
        self.cnn = nn.Sequential(
            # Слой 1: Вход 1@64x64, Выход 32@14x14
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Слой 2: Вход 32@14x14, Выход 64@6x6
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Слой 3: Вход 64@6x6, Выход 64@4x4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Вычисление размера выхода CNN (64 * 4 * 4 = 1024)
        with th.no_grad():
            dummy_input = th.as_tensor(th.zeros(1, n_channels, input_shape[1], input_shape[2])).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.output_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Пользовательский Feature Extractor для Dict Space
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        # --- 1. Обработка Пикселей (CNN) ---
        if "pixels" in observation_space.spaces:
            pixel_shape = observation_space["pixels"].shape
            cnn_input_shape = pixel_shape
            
            self.cnn_extractor = NatureCNN(cnn_input_shape, features_dim=features_dim)
            extractors["pixels"] = self.cnn_extractor
            total_concat_size += self.cnn_extractor.output_dim

        # --- 2. Обработка Проприоцепции (MLP) ---
        if "proprioception" in observation_space.spaces:
            proprio_input_size = observation_space["proprioception"].shape[0]
            # Простой MLP для джоинтов (например, 2 скрытых слоя)
            proprio_extractor = nn.Sequential(
                nn.Linear(proprio_input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            extractors["proprioception"] = proprio_extractor
            total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        # Выходной размер (должен соответствовать features_dim в PPO)
        self._features_dim = total_concat_size 

    def forward(self, observations) -> th.Tensor:
        encoded_tensors = []
        for key, extractor in self.extractors.items():
            tensor = observations[key]
            if key == "pixels":
                # Input is already (Batch, C, H, W)
                # Нормализация: [0, 255] -> [0, 1]
                tensor = tensor.float() / 255.0 
            
            encoded_tensors.append(extractor(tensor))
        
        # Объединение признаков
        return th.cat(encoded_tensors, dim=1)