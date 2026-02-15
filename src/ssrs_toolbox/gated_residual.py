"""Gated residual block for hierarchical VAE architectures.

HDN-style gated residual block with learned per-channel soft gating.
Follows the pattern of nn_toolbox/blocks.py (NAFBlock3D, ConvNeXtBlock3D).

Citation:
    Prakash et al., "Interpretable Unsupervised Diversity Denoising and
    Artefact Removal", ICLR 2022.
    arXiv: https://arxiv.org/abs/2104.01374
    Source: https://github.com/juglab/HDN (BSD-3-Clause)

See licenses/HDN.txt for full license text.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .activations import get_activation
from .normalization import get_norm_layer


class GatedResidualBlock3D(nn.Module):
    """Gated residual block (HDN-style).

    Pattern: BN -> Act -> Conv3x3 -> Drop -> BN -> Act -> Conv3x3 -> Drop -> Gate -> + res

    The gate is a learned 1x1 conv -> sigmoid that provides per-channel soft gating
    on the residual path, allowing the network to learn how much of each feature
    to let through.
    """

    def __init__(
        self,
        channels: int,
        z_conv: bool = True,
        dropout_p: float = 0.0,
        activation: str = "elu",
        norm_type: str = "batch",
    ):
        """Initialize gated residual block.

        :param channels: Number of input/output channels
        :param z_conv: If True, use 3D kernels; if False, use (1,3,3) kernels
        :param dropout_p: Dropout probability
        :param activation: Activation function name
        :param norm_type: Normalization type
        """
        super().__init__()

        kernel = 3 if z_conv else (1, 3, 3)
        padding = 1 if z_conv else (0, 1, 1)

        self.norm1 = get_norm_layer(norm_type, channels)
        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv3d(channels, channels, kernel, padding=padding)
        self.drop1 = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.norm2 = get_norm_layer(norm_type, channels)
        self.act2 = get_activation(activation)
        self.conv2 = nn.Conv3d(channels, channels, kernel, padding=padding)
        self.drop2 = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

        # Learned gate: 1x1 conv -> sigmoid
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated residual block.

        :param x: Input tensor [B, C, D, H, W]
        :returns: Output tensor [B, C, D, H, W]
        """
        residual = x

        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.drop2(out)

        # Apply learned gate
        out = out * self.gate(out)

        return out + residual
