"""Partial Convolution layers for masked convolutions.

Based on NVIDIA's implementation for image inpainting.
"""

import torch
import torch.nn as nn

EPSILON = 1e-8


class PartialConv3d(nn.Module):
    """3D Partial Convolution layer.

    Performs convolution only on valid (unmasked) regions and updates
    the mask accordingly. Useful for inpainting and handling missing data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        multi_channel: bool = True,
        return_mask: bool = True,
    ):
        """Initialize PartialConv3d.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the convolution kernel
        :param stride: Stride of the convolution
        :param padding: Padding added to input
        :param dilation: Dilation rate
        :param groups: Number of blocked connections
        :param bias: Whether to include bias
        :param multi_channel: Whether mask has multiple channels
        :param return_mask: Whether to return updated mask
        """
        super().__init__()

        self.multi_channel = multi_channel
        self.return_mask = return_mask

        # Main convolution
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Mask convolution (for computing valid regions)
        if self.multi_channel:
            self.mask_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        else:
            self.mask_conv = nn.Conv3d(
                1,
                1,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=False,
            )

        # Initialize mask conv weights to 1
        nn.init.constant_(self.mask_conv.weight, 1.0)

        # Freeze mask conv weights
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Calculate window size for normalization
        self.slide_winsize = kernel_size**3 * (in_channels if multi_channel else 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Apply partial convolution.

        :param x: Input tensor of shape (B, C, D, H, W)
        :param mask: Binary mask (1=valid, 0=invalid)
        :returns: If return_mask: (output, updated_mask), otherwise: output
        """
        # Expand mask to match channels if needed
        if not self.multi_channel and mask.size(1) != 1:
            mask = mask[:, :1, ...]

        # Apply mask to input
        x = x * mask

        # Compute output
        output = self.conv(x)

        # Compute mask update
        with torch.no_grad():
            update_mask = self.mask_conv(mask)

            # Compute normalization ratio
            mask_ratio = self.slide_winsize / (update_mask + EPSILON)

            # Binary update mask
            update_mask = torch.clamp(update_mask, 0, 1)

        # Normalize output by valid region ratio
        output = output * mask_ratio

        # Handle bias
        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1, 1)
            output = (output - bias) * update_mask + bias

        if self.return_mask:
            return output, update_mask
        else:
            return output


class PartialConv2d(nn.Module):
    """2D Partial Convolution layer.

    2D version of PartialConv3d for image data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        multi_channel: bool = True,
        return_mask: bool = True,
    ):
        """Initialize PartialConv2d."""
        super().__init__()

        self.multi_channel = multi_channel
        self.return_mask = return_mask

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.multi_channel:
            self.mask_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        else:
            self.mask_conv = nn.Conv2d(
                1,
                1,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=False,
            )

        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.slide_winsize = kernel_size**2 * (in_channels if multi_channel else 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Apply partial convolution."""
        if not self.multi_channel and mask.size(1) != 1:
            mask = mask[:, :1, ...]

        x = x * mask
        output = self.conv(x)

        with torch.no_grad():
            update_mask = self.mask_conv(mask)
            mask_ratio = self.slide_winsize / (update_mask + EPSILON)
            update_mask = torch.clamp(update_mask, 0, 1)

        output = output * mask_ratio

        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1)
            output = (output - bias) * update_mask + bias

        if self.return_mask:
            return output, update_mask
        else:
            return output
