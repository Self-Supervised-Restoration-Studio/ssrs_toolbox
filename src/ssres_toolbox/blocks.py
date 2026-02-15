"""Special neural network blocks.

Includes NAFBlock3D, ConvNeXtBlock3D, and ResidualBlock.

Citations:
    NAFBlock3D adapted from NAFNet:
        Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
        arXiv: https://arxiv.org/abs/2204.04676
        Source: https://github.com/megvii-research/NAFNet (MIT License)

    ConvNeXtBlock3D adapted from ConvNeXt:
        Liu et al., "A ConvNet for the 2020s", CVPR 2022.
        arXiv: https://arxiv.org/abs/2201.03545
        Source: https://github.com/facebookresearch/ConvNeXt (MIT License)

See licenses/NAFNET.txt and licenses/CONVNEXT.txt for full license texts.
"""

import torch
import torch.nn as nn

from .attention import LayerScaleLayer, SpatialChannelAttention
from .gates import SimpleGate3D
from .normalization import get_norm_layer


class NAFBlock3D(nn.Module):
    """NAFNet-style block (Nonlinear Activation Free).

    Uses depthwise convolutions, simple gates, and spatial channel attention
    instead of traditional activations.

    Reference: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
    License: MIT (https://github.com/megvii-research/NAFNet)
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_p: float = 0.0,
        norm_type: str = "layer",
    ):
        """Initialize NAFBlock3D.

        :param channels: Number of input/output channels
        :param dw_expand: Expansion factor for depthwise conv
        :param ffn_expand: Expansion factor for FFN
        :param dropout_p: Dropout probability
        :param norm_type: Type of normalization
        """
        super().__init__()

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        # First branch: depthwise conv with gate
        self.norm1 = get_norm_layer(norm_type, channels)
        self.conv1 = nn.Conv3d(channels, dw_channels, 1)
        self.conv2 = nn.Conv3d(
            dw_channels,
            dw_channels,
            3,
            padding=1,
            groups=dw_channels,  # Depthwise
        )
        self.gate1 = SimpleGate3D()
        self.conv3 = nn.Conv3d(dw_channels // 2, channels, 1)

        # Spatial channel attention
        self.sca = SpatialChannelAttention(channels)

        # Second branch: FFN with gate
        self.norm2 = get_norm_layer(norm_type, channels)
        self.conv4 = nn.Conv3d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate3D()
        self.conv5 = nn.Conv3d(ffn_channels // 2, channels, 1)

        # Learnable residual scaling
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        # Dropout
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NAF block.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Output tensor
        """
        # First branch
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = self.dropout(y)
        x = x + y * self.beta

        # Second branch
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.gate2(y)
        y = self.conv5(y)
        y = self.dropout(y)
        x = x + y * self.gamma

        return x


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt-style block for 3D data.

    Uses depthwise convolution, layer norm, linear layers, and layer scaling.

    Reference: Liu et al., "A ConvNet for the 2020s", CVPR 2022.
    License: MIT (https://github.com/facebookresearch/ConvNeXt)
    """

    def __init__(
        self,
        channels: int,
        expand_ratio: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
        norm_type: str = "layer",
        activation: str = "gelu",
    ):
        """Initialize ConvNeXtBlock3D.

        :param channels: Number of input/output channels
        :param expand_ratio: Expansion ratio for hidden dimension
        :param kernel_size: Kernel size for depthwise conv
        :param layer_scale_init: Initial value for layer scale
        :param norm_type: Type of normalization
        :param activation: Activation function name
        """
        super().__init__()

        hidden_dim = channels * expand_ratio
        padding = kernel_size // 2

        # Depthwise convolution
        self.dwconv = nn.Conv3d(channels, channels, kernel_size, padding=padding, groups=channels)

        # Normalization
        self.norm = get_norm_layer(norm_type, channels)

        # Pointwise convolutions (linear layers in spatial position)
        self.pwconv1 = nn.Conv3d(channels, hidden_dim, 1)

        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.pwconv2 = nn.Conv3d(hidden_dim, channels, 1)

        # Layer scale
        self.layer_scale = LayerScaleLayer(channels, layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ConvNeXt block.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Output tensor
        """
        shortcut = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)

        return shortcut + x


class ResidualBlock(nn.Module):
    """Simple residual block wrapper.

    Wraps any module with a residual connection.
    """

    def __init__(self, inner: nn.Module):
        """Initialize residual block.

        :param inner: Module to wrap
        """
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection.

        :param x: Input tensor
        :returns: Output with residual: inner(x) + x
        """
        return self.inner(x) + x
