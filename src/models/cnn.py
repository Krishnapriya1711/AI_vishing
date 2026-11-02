"""Lightweight CNN for log-Mel spectrogram classification.

Architecture (input: 1 × n_mels × max_frames):

    Conv2d(1,  32, 3, padding=1) + BN + ReLU + MaxPool(2)
    Conv2d(32, 64, 3, padding=1) + BN + ReLU + MaxPool(2)
    Conv2d(64,128, 3, padding=1) + BN + ReLU + MaxPool(2)
    AdaptiveAvgPool(1) → flatten
    Dropout(0.3)
    Linear(128, 64) + ReLU
    Dropout(0.3)
    Linear(64, 2)

The model outputs logits for two classes (bonafide=0, spoof=1).
"""

import torch
import torch.nn as nn


class LightweightCNN(nn.Module):
    """A compact CNN operating on 2-D log-Mel spectrograms."""

    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()

        # --- Convolutional feature extractor ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global average pool → (batch, 128, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape ``(batch, 1, n_mels, n_frames)``

        Returns
        -------
        torch.Tensor  shape ``(batch, num_classes)``  logits
        """
        x = self.features(x)          # (batch, 128, 1, 1)
        x = x.flatten(1)              # (batch, 128)
        x = self.classifier(x)         # (batch, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (column 1 = P(spoof))."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
