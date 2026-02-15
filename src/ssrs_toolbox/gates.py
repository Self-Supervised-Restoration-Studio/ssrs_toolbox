"""Gate mechanisms for neural networks.

These modules implement various gating operations used in modern architectures
like NAFNet and other attention-free networks.
"""

import math

import torch
import torch.nn as nn


class SimpleGate3D(nn.Module):
    """Simple element-wise multiplication gate (NAFNet style).

    Splits input channels in half and multiplies them together.
    This serves as an activation-free nonlinearity.

    Input: (B, 2C, D, H, W) -> Output: (B, C, D, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simple gate.

        :param x: Input tensor with even number of channels
        :returns: Gated output with half the channels
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GatedReLUMix(nn.Module):
    """Gated ReLU activation mixing nonlinearity with gating.

    Splits channels and applies ReLU to first half, then multiplies
    with second half: (ReLU(x1), x1 * x2)

    Input: (B, 2C, D, H, W) -> Output: (B, C, D, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated ReLU mix.

        :param x: Input tensor with even number of channels
        :returns: Activated and gated output
        """
        x1, x2 = x.chunk(2, dim=1)
        return torch.relu(x1) * x2


class SinGatedMix(nn.Module):
    """Sinusoidal gated activation.

    Uses sine function as activation with gating mechanism.
    Splits channels and applies: sin(x1) * x2

    Input: (B, 2C, D, H, W) -> Output: (B, C, D, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal gated mix.

        :param x: Input tensor with even number of channels
        :returns: Sinusoidal gated output
        """
        x1, x2 = x.chunk(2, dim=1)
        return torch.sin(x1) * x2


class ScaledSimpleGate3D(nn.Module):
    """Scaled simple gate with normalization factor.

    Same as SimpleGate3D but scales output to prevent numerical overflow.
    Output = (x1 * x2) / sqrt(channels)
    """

    def __init__(self, dim: int):
        """Initialize scaled gate.

        :param dim: Output dimension (half of input channels)
        """
        super().__init__()
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled simple gate.

        :param x: Input tensor with even number of channels
        :returns: Scaled gated output
        """
        x1, x2 = x.chunk(2, dim=1)
        return (x1 * x2) * self.scale
