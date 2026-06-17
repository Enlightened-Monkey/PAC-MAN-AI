"""Shared PPO + PacmanCNN policy components."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PacmanCNN(BaseFeaturesExtractor):
    """CNN feature extractor for PacmanGridEnv (9 or 10, 31, 28) observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


def policy_kwargs(features_dim: int = 256) -> dict:
    return dict(
        features_extractor_class=PacmanCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
