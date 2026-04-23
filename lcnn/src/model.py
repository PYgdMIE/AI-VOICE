"""
LCNN-style light CNN for binary spoof detection (human vs synthetic).
Input: single-channel time–frequency map (B, 1, F, T), e.g. log-mel or MFCC.
Architecture inspired by light CNNs used in ASVspoof literature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-Feature-Map: split channels, take max along pair (common in light CNNs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(1) // 2
        return torch.max(x[:, :d], x[:, d:])


class LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2) -> None:
        super().__init__()
        # Double out_ch before MFM so after MFM we have out_ch channels
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = MFM()
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        x = self.pool(x)
        return x


class LCNNSpoofDetector(nn.Module):
    """
    Binary classifier: label 0 = human, 1 = AI/synthetic.
    Expects input shape (B, 1, F, T) where F is mel bands or MFCC order.
    """

    def __init__(
        self,
        n_freq_bins: int = 40,
        base: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.features = nn.Sequential(
            LCNNBlock(1, base, kernel=5, pool=2),
            LCNNBlock(base, base, kernel=3, pool=2),
            LCNNBlock(base, base * 2, kernel=3, pool=2),
            LCNNBlock(base * 2, base * 2, kernel=3, pool=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base * 2, base),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.head(x)
