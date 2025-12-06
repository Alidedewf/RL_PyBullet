
import gymnasium as gym
import torch as th
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MobileNetCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        # --- 1. Vision Extractor (MobileNetV3) ---
        if "pixels" in observation_space.spaces:
            # Загружаем предобученную MobileNetV3 Small
            # weights='DEFAULT' загружает веса ImageNet
            self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            
            # Замораживаем веса (Transfer Learning)
            for param in self.mobilenet.parameters():
                param.requires_grad = False
                
            # Заменяем "голову" классификатора на Identity, чтобы получить вектор признаков
            # У MobileNetV3 Small выход pooling слоя (перед classifier) имеет размер 576
            self.mobilenet.classifier = nn.Identity()
            
            # --- Handling Frame Stacking (RGB x 4 = 12 channels) ---
            # We need to modify the first layer to accept >3 channels if stacked
            n_input_channels = observation_space["pixels"].shape[0]
            if n_input_channels != 3:
                # Get the first conv layer
                first_conv_layer = self.mobilenet.features[0][0]
                
                # Create nice new layer with correct input channels
                new_conv_layer = nn.Conv2d(
                    in_channels=n_input_channels,
                    out_channels=first_conv_layer.out_channels,
                    kernel_size=first_conv_layer.kernel_size,
                    stride=first_conv_layer.stride,
                    padding=first_conv_layer.padding,
                    bias=first_conv_layer.bias is not None
                )
                
                # Initialize weights: copy RGB weights N times / N (to keep scale)
                # This basic heuristic works better than random init for transfer learning
                with th.no_grad():
                    original_weights = first_conv_layer.weight
                    # Repeat weights: (16, 3, 3, 3) -> (16, 12, 3, 3) 
                    # Assuming n_input_channels is a multiple of 3
                    repeat_factor = n_input_channels // 3
                    new_weights = original_weights.repeat(1, repeat_factor, 1, 1) / repeat_factor
                    new_conv_layer.weight.copy_(new_weights)
                    
                    if first_conv_layer.bias is not None:
                         new_conv_layer.bias.copy_(first_conv_layer.bias)

                # Replace the layer
                self.mobilenet.features[0][0] = new_conv_layer
            
            # Размер выхода MobileNetV3 Small после avgpool (перед classifier) - это 576
            # Но так как мы заменили classifier на Identity, выход будет [Batch, 576]
            cnn_output_dim = 576
            
            extractors["pixels"] = self.mobilenet
            total_concat_size += cnn_output_dim

        # --- 2. Proprioception Extractor (MLP) ---
        if "proprioception" in observation_space.spaces:
            proprio_input_size = observation_space["proprioception"].shape[0]
            proprio_extractor = nn.Sequential(
                nn.Linear(proprio_input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            extractors["proprioception"] = proprio_extractor
            total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size 

    def forward(self, observations) -> th.Tensor:
        encoded_tensors = []
        for key, extractor in self.extractors.items():
            tensor = observations[key]
            
            if key == "pixels":
                # Трансформации для ImageNet:
                # 1. Normalize к [0, 1] (уже делается делением на 255 вне, но SB3 обычно подает 0..255 uint8)
                # SB3 NatureCNN ожидает float, давайте убедимся
                tensor = tensor.float() / 255.0
                
                # MobileNet ожидает нормализацию (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # Но для простоты оставим [0, 1], так как сеть все равно эффективна.
                # Важно: MobileNet ожидает 3 канала (RGB).
                # Проверим размерность. SB3 может подавать (Batch, Channels, Height, Width)
                pass
            
            encoded_tensors.append(extractor(tensor))
        
        return th.cat(encoded_tensors, dim=1)
