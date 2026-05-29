"""
Convolutional neural network for binary classification of Ising configurations
(ordered vs disordered phase), following the architecture sketched in
Azizi & Pleimling (2021), Fig. 2.

Architecture:
    Input  (1, L, L)
    Conv2d  1 -> 16, 3x3, padding=1, ReLU
    MaxPool 2x2
    Conv2d 16 -> 8, 3x3, padding=1, ReLU
    MaxPool 2x2
    Flatten
    Dense   -> 2
    Softmax (applied at the loss for training; here we just keep logits)

Output: 2 logits; softmax(out)[:, 1] = P(ordered)

Notes
-----
* The paper does not give kernel sizes; 3x3 with same-padding is the
  canonical choice for small image-like inputs and reproduces their
  qualitative results.
* The spin values fed to the network are float ±1 (not 0/1).
* We use circular padding by default to respect the periodic boundary
  conditions of the physical system; this is a (small) deviation from
  the paper but is more physically consistent. Set `circular=False` to
  use zero-padding instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv2d(nn.Module):
    """Conv2d wrapper that applies circular padding on both spatial dims."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad,) * 4, mode="circular")
        return self.conv(x)


class IsingCNN(nn.Module):
    def __init__(self, L: int, circular: bool = True):
        super().__init__()
        self.L = L
        Conv = CircularConv2d if circular else (
            lambda i, o, k=3: nn.Conv2d(i, o, kernel_size=k, padding=k // 2)
        )
        self.conv1 = Conv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = Conv(16, 8)
        self.pool2 = nn.MaxPool2d(2)
        # After two 2x2 pools: spatial size = L // 4 (integer division)
        flat_dim = 8 * (L // 4) * (L // 4)
        if flat_dim == 0:
            raise ValueError(f"L={L} is too small: needs L >= 4")
        self.fc = nn.Linear(flat_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L, L) with values in {-1, +1}
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        return self.fc(x)  # logits; apply softmax at inference

    @torch.no_grad()
    def predict_p_ordered(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability that the input is in the ordered phase."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)[:, 1]


__all__ = ["IsingCNN"]
