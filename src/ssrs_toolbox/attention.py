"""Attention mechanisms for neural networks.

Includes channel attention (SE-Net style) and layer scaling.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module (SE-Net style).

    Uses global average pooling followed by FC layers to generate
    channel-wise attention weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """Initialize channel attention.

        :param channels: Number of input/output channels
        :param reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        bottleneck = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Attention-weighted tensor
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialChannelAttention(nn.Module):
    """Spatial Channel Attention (SCA) used in NAFNet.

    Simplified attention using global pooling and 1x1 convolution.
    """

    def __init__(self, channels: int):
        """Initialize SCA.

        :param channels: Number of input/output channels
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial channel attention.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Attention-weighted tensor
        """
        return x * self.conv(self.pool(x))


class LayerScaleLayer(nn.Module):
    """Learnable per-channel scaling (used in ConvNeXt).

    Multiplies input by a learnable parameter initialized to a small value.
    """

    def __init__(self, channels: int, init_value: float = 1e-6):
        """Initialize layer scale.

        :param channels: Number of channels
        :param init_value: Initial value for scale parameter
        """
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Scaled tensor
        """
        return x * self.gamma
