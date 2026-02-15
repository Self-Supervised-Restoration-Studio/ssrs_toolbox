"""Convolution utility functions and shortcuts.

Provides factory functions for creating various convolution configurations.
"""

import torch.nn as nn

from .activations import get_activation
from .normalization import get_norm_layer


def _calculate_padding(kernel_size: int) -> int:
    """Calculate same padding for a kernel size."""
    return kernel_size // 2


def create_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    bias: bool = True,
    groups: int = 1,
    depthwise: bool = False,
    z_conv: bool = True,
) -> nn.Conv3d:
    """Create a 3D convolution with automatic padding.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param kernel_size: Size of the convolution kernel
    :param stride: Stride of the convolution
    :param padding: Padding (auto-calculated if None)
    :param bias: Whether to include bias
    :param groups: Number of groups
    :param depthwise: Whether to use depthwise convolution
    :param z_conv: If True, use 3D conv; if False, use 2D conv (preserve Z)
    :returns: Conv3d layer
    """
    if padding is None:
        padding = _calculate_padding(kernel_size)

    if depthwise:
        groups = in_channels
        out_channels = in_channels

    if not z_conv:
        # Use 2D convolution (1 in Z dimension)
        kernel = (1, kernel_size, kernel_size)
        pad = (0, padding, padding)
        stride_3d = (1, stride, stride)
    else:
        kernel = kernel_size
        pad = padding
        stride_3d = stride

    return nn.Conv3d(
        in_channels, out_channels, kernel, stride=stride_3d, padding=pad, bias=bias, groups=groups
    )


def conv111(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv3d:
    """1x1x1 convolution (channel projection).

    :param in_channels: Input channels
    :param out_channels: Output channels
    :param bias: Whether to include bias
    :returns: 1x1x1 Conv3d
    """
    return nn.Conv3d(in_channels, out_channels, 1, bias=bias)


def conv333(
    in_channels: int, out_channels: int, bias: bool = True, z_conv: bool = True
) -> nn.Conv3d:
    """3x3x3 convolution with same padding.

    :param in_channels: Input channels
    :param out_channels: Output channels
    :param bias: Whether to include bias
    :param z_conv: If True, use 3D; if False, use 2D (preserve Z)
    :returns: 3x3x3 (or 1x3x3) Conv3d
    """
    return create_conv3d(in_channels, out_channels, 3, bias=bias, z_conv=z_conv)


def conv777(
    in_channels: int, out_channels: int, bias: bool = True, z_conv: bool = True
) -> nn.Conv3d:
    """7x7x7 convolution with same padding.

    :param in_channels: Input channels
    :param out_channels: Output channels
    :param bias: Whether to include bias
    :param z_conv: If True, use 3D; if False, use 2D (preserve Z)
    :returns: 7x7x7 (or 1x7x7) Conv3d
    """
    return create_conv3d(in_channels, out_channels, 7, bias=bias, z_conv=z_conv)


def depthwise_conv333(channels: int, bias: bool = True, z_conv: bool = True) -> nn.Conv3d:
    """3x3x3 depthwise convolution.

    :param channels: Number of channels
    :param bias: Whether to include bias
    :param z_conv: If True, use 3D; if False, use 2D (preserve Z)
    :returns: Depthwise Conv3d
    """
    return create_conv3d(channels, channels, 3, bias=bias, depthwise=True, z_conv=z_conv)


def build_conv_unit3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    norm_type: str = "none",
    activation: str = "relu",
    depthwise_separable: bool = False,
    z_conv: bool = True,
) -> nn.Sequential:
    """Build a complete convolution unit with norm and activation.

    :param in_channels: Input channels
    :param out_channels: Output channels
    :param kernel_size: Kernel size
    :param stride: Stride
    :param norm_type: Type of normalization
    :param activation: Activation function
    :param depthwise_separable: Use depthwise separable convolution
    :param z_conv: If True, use 3D; if False, use 2D (preserve Z)
    :returns: Sequential module with conv, norm, activation
    """
    layers = []

    if depthwise_separable:
        # Depthwise
        layers.append(
            create_conv3d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                bias=False,
                depthwise=True,
                z_conv=z_conv,
            )
        )
        if norm_type != "none":
            layers.append(get_norm_layer(norm_type, in_channels))

        # Pointwise
        layers.append(conv111(in_channels, out_channels, bias=norm_type == "none"))
    else:
        # Standard convolution
        layers.append(
            create_conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=norm_type == "none",
                z_conv=z_conv,
            )
        )

    # Normalization
    if norm_type != "none":
        layers.append(get_norm_layer(norm_type, out_channels))

    # Activation
    if activation != "none":
        layers.append(get_activation(activation))

    return nn.Sequential(*layers)
