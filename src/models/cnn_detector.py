"""
cnn_detector.py – Convolutional neural network for Pac-Man screen parsing.

Stage 2 of the two-stage training pipeline:
  1. Train a CNN on labelled game screenshots to produce a state vector.
  2. Feed that state vector to the pre-trained RL agent instead of the
     ground-truth observation from the environment.

Classes
-------
PacmanCNN   : Lightweight CNN encoder (feature extractor).
ObjectDetector : Wrapper that maps a raw RGB frame to a state-vector
                 compatible with PacmanEnv.observation_space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PacmanCNN(nn.Module):
    """
    Small CNN that encodes an RGB game frame into a flat feature vector.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3 for RGB).
    out_features : int
        Dimensionality of the output feature vector.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, H, W), values in [0, 1]

        Returns
        -------
        Tensor, shape (B, out_features)
        """
        return self.head(self.features(x))


class ObjectDetector:
    """
    High-level wrapper: RGB frame → state vector.

    The detector uses ``PacmanCNN`` as a backbone and a final linear layer
    to project the CNN features to the exact size expected by PacmanEnv's
    observation_space (``obs_size`` values in [0, 1]).

    Parameters
    ----------
    obs_size : int
        Target observation vector length (must match PacmanEnv._OBS_SIZE).
    device : str
        PyTorch device string ('cpu', 'cuda', etc.).
    """

    def __init__(self, obs_size: int, device: str | torch.device | None = "cpu") -> None:
        self.obs_size = obs_size
        if device is None or device == "auto":
            from src.utils.device_helper import get_best_device
            self.device = get_best_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.cnn = PacmanCNN(out_features=256).to(self.device)
        self.projection = nn.Linear(256, obs_size).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a raw BGR/RGB frame (H, W, 3) uint8 to a state vector.

        Parameters
        ----------
        frame : np.ndarray, shape (H, W, 3), dtype uint8

        Returns
        -------
        np.ndarray, shape (obs_size,), dtype float32, values in [0, 1]
        """
        tensor = self._preprocess(frame)
        with torch.no_grad():
            features = self.cnn(tensor)
            obs = self.sigmoid(self.projection(features))
        return obs.squeeze(0).cpu().numpy().astype(np.float32)

    def _preprocess(self, frame: np.ndarray) -> Tensor:
        """Resize to (84, 84), normalise to [0, 1], add batch dimension."""
        import cv2  # lazy import to keep the module importable without OpenCV

        resized = cv2.resize(frame, (84, 84))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def save(self, path: str) -> None:
        """Save CNN + projection weights."""
        torch.save(
            {
                "cnn": self.cnn.state_dict(),
                "projection": self.projection.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load CNN + projection weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.projection.load_state_dict(checkpoint["projection"])
