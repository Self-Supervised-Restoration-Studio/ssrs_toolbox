"""Pooling, upsampling, and merge utilities for U-Net architectures.

Provides functions for downsampling, upsampling, and skip connection merging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixel_shuffle import PixelShuffle3d


def pool(channels: int, method: str = "max", factor: int = 2, z_pool: bool = True) -> nn.Module:
    """Create a pooling/downsampling layer.

    :param channels: Number of input channels (used for strided conv)
    :param method: Pooling method ("max", "avg", "stride")
    :param factor: Downsampling factor
    :param z_pool: Whether to pool in Z dimension
    :returns: Pooling module
    """
    if z_pool:
        kernel = factor
        stride = factor
    else:
        kernel = (1, factor, factor)
        stride = (1, factor, factor)

    if method == "max":
        return nn.MaxPool3d(kernel, stride=stride)
    elif method == "avg":
        return nn.AvgPool3d(kernel, stride=stride)
    elif method == "stride":
        # Strided convolution for learned downsampling
        return nn.Conv3d(channels, channels, kernel, stride=stride, padding=0)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def upconv222(
    in_channels: int,
    out_channels: int,
    method: str = "transpose",
    factor: int = 2,
    z_up: bool = True,
) -> nn.Module:
    """Create an upsampling layer (2x2x2 or configured factor).

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param method: Upsampling method ("transpose", "trilinear", "nearest", "pixel_shuffle")
    :param factor: Upsampling factor
    :param z_up: Whether to upsample in Z dimension
    :returns: Upsampling module
    """
    if z_up:
        kernel = factor
        stride = factor
        scale = factor
    else:
        kernel = (1, factor, factor)
        stride = (1, factor, factor)
        scale = (1, factor, factor)

    if method == "transpose":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride)
    elif method == "trilinear":
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, out_channels, 1),
        )
    elif method == "nearest":
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"), nn.Conv3d(in_channels, out_channels, 1)
        )
    elif method == "pixel_shuffle":
        # Pixel shuffle requires specific channel count
        if z_up:
            shuffle_channels = out_channels * (factor**3)
        else:
            shuffle_channels = out_channels * (factor**2)
        return nn.Sequential(
            nn.Conv3d(in_channels, shuffle_channels, 1),
            PixelShuffle3d(factor) if z_up else _PixelShuffle2D3d(factor),
        )
    else:
        raise ValueError(f"Unknown upsampling method: {method}")


class _PixelShuffle2D3d(nn.Module):
    """Pixel shuffle only in H and W dimensions (not Z)."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        r = self.scale_factor
        out_c = c // (r**2)
        x = x.view(b, out_c, r, r, d, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3)
        return x.contiguous().view(b, out_c, d, h * r, w * r)


def merge(method: str = "concat") -> nn.Module:
    """Create a skip connection merge module.

    :param method: Merge method ("concat", "add", "attention")
    :returns: Merge module that takes (upsampled, skip) as input
    """
    if method == "concat":
        return ConcatMerge()
    elif method == "add":
        return AddMerge()
    elif method == "attention":
        return AttentionMerge()
    else:
        raise ValueError(f"Unknown merge method: {method}")


class ConcatMerge(nn.Module):
    """Merge by concatenation along channel dimension."""

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate upsampled and skip tensors.

        :param upsampled: Upsampled tensor from decoder
        :param skip: Skip connection from encoder
        :returns: Concatenated tensor
        """
        # Handle size mismatches
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = _match_size(upsampled, skip)
        return torch.cat([upsampled, skip], dim=1)


class AddMerge(nn.Module):
    """Merge by element-wise addition."""

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Add upsampled and skip tensors.

        :param upsampled: Upsampled tensor from decoder
        :param skip: Skip connection from encoder
        :returns: Sum tensor
        """
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = _match_size(upsampled, skip)
        return upsampled + skip


class AttentionMerge(nn.Module):
    """Merge with attention gating on skip connection."""

    def __init__(self):
        super().__init__()
        self._gate = None

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply attention gating and concatenate.

        :param upsampled: Upsampled tensor (gating signal)
        :param skip: Skip connection to be gated
        :returns: Concatenated tensor with attention-weighted skip
        """
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = _match_size(upsampled, skip)

        # Lazy initialization of gate
        if self._gate is None:
            channels = skip.shape[1]
            self._gate = nn.Sequential(nn.Conv3d(channels * 2, channels, 1), nn.Sigmoid()).to(
                skip.device
            )

        # Compute attention
        combined = torch.cat([upsampled, skip], dim=1)
        attention = self._gate(combined)
        gated_skip = skip * attention

        return torch.cat([upsampled, gated_skip], dim=1)


def _match_size(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize source to match target spatial dimensions.

    :param source: Tensor to resize
    :param target: Tensor with target size
    :returns: Resized source tensor
    """
    target_size = target.shape[2:]
    if source.shape[2:] == target_size:
        return source

    return F.interpolate(source, size=target_size, mode="trilinear", align_corners=False)


class UpsampleBlock(nn.Module):
    """Complete upsampling block with merge and convolution.

    Combines upsampling, skip connection merging, and convolution
    into a single module for cleaner U-Net decoder construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_method: str = "transpose",
        merge_method: str = "concat",
        factor: int = 2,
        z_up: bool = True,
    ):
        """Initialize UpsampleBlock.

        :param in_channels: Input channels (from lower level)
        :param out_channels: Output channels
        :param up_method: Upsampling method
        :param merge_method: Skip connection merge method
        :param factor: Upsampling factor
        :param z_up: Whether to upsample in Z dimension
        """
        super().__init__()

        self.up = upconv222(in_channels, out_channels, up_method, factor, z_up)
        self.merge = merge(merge_method)

        # Convolution after merge
        if merge_method == "concat":
            conv_in = out_channels * 2  # Concatenated channels
        else:
            conv_in = out_channels

        self.conv = nn.Sequential(
            nn.Conv3d(conv_in, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply upsample block.

        :param x: Input from lower decoder level
        :param skip: Skip connection from encoder
        :returns: Processed output
        """
        x = self.up(x)
        x = self.merge(x, skip)
        x = self.conv(x)
        return x
