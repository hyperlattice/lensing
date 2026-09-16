"""Compact convolutional regression baseline."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        groups = min(8, int(output_channels))
        while int(output_channels) % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )


class EinsteinRadiusCNN(nn.Module):
    """Four downsampling blocks followed by a small spatial regression head."""

    def __init__(self, channels=(8, 16, 32, 64), pooled_size=4, hidden_features=128):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain four positive integers")
        self.model_config = {
            "channels": list(channels),
            "pooled_size": int(pooled_size),
            "hidden_features": int(hidden_features),
        }
        blocks = []
        input_channels = 1
        for output_channels in channels:
            blocks.append(ConvBlock(input_channels, output_channels))
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((int(pooled_size), int(pooled_size)))
        flattened = channels[-1] * int(pooled_size) ** 2
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, int(hidden_features)),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(hidden_features), 1),
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected images with shape [batch, 1, height, width]")
        return self.regressor(self.pool(self.features(images))).squeeze(-1)

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
