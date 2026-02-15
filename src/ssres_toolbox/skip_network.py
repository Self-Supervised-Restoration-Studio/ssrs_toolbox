"""Skip Network - U-Net style encoder-decoder with skip connections.

This module provides a flexible skip network architecture commonly used
for image reconstruction tasks. Supports reflection padding to reduce
boundary artifacts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _get_pad_layer(pad: str, ndim: int = 2) -> type[nn.Module]:
    """Get padding layer class by name and dimensionality."""
    if ndim == 3:
        return {
            "reflection": nn.ReplicationPad3d,  # No ReflectionPad3d in PyTorch
            "replication": nn.ReplicationPad3d,
            "zero": nn.ConstantPad3d,
        }.get(pad, nn.ReplicationPad3d)
    return {
        "reflection": nn.ReflectionPad2d,
        "replication": nn.ReplicationPad2d,
        "zero": nn.ConstantPad2d,
    }.get(pad, nn.ReflectionPad2d)


class SkipNetwork(nn.Module):
    """U-Net style skip network for image reconstruction.

    A flexible encoder-decoder architecture with skip connections,
    suitable for tasks like image restoration, deblurring, and denoising.

    :param input_depth: Number of input channels
    :param output_channels: Number of output channels
    :param num_channels_down: Channels at each encoder level
    :param num_channels_up: Channels at each decoder level
    :param num_channels_skip: Channels for skip connections
    :param activation: Activation type ('leaky_relu', 'relu', 'gelu')
    :param upsample_mode: Interpolation mode for upsampling
    :param need_sigmoid: Whether to apply sigmoid at output
    :param norm_type: Normalization type ('batch', 'instance', 'none')
    :param pad: Padding type ('reflection', 'replication', 'zero')

    Example:
        >>> net = SkipNetwork(
        ...     input_depth=64,
        ...     output_channels=1,
        ...     num_channels_down=[128, 128, 128],
        ...     num_channels_up=[128, 128, 128],
        ...     num_channels_skip=[16, 16, 16],
        ... )
        >>> x = torch.randn(1, 64, 256, 256)
        >>> out = net(x)  # (1, 1, 256, 256)
    """

    def __init__(
        self,
        input_depth: int,
        output_channels: int = 1,
        num_channels_down: list[int] | None = None,
        num_channels_up: list[int] | None = None,
        num_channels_skip: list[int] | None = None,
        activation: str = "leaky_relu",
        upsample_mode: str = "bilinear",
        need_sigmoid: bool = True,
        norm_type: str = "batch",
        pad: str = "reflection",
    ):
        super().__init__()

        if num_channels_down is None:
            num_channels_down = [128, 128, 128, 128, 128]
        if num_channels_up is None:
            num_channels_up = [128, 128, 128, 128, 128]
        if num_channels_skip is None:
            num_channels_skip = [16, 16, 16, 16, 16]

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.depth = len(num_channels_down)
        self.need_sigmoid = need_sigmoid
        self.upsample_mode = upsample_mode
        self._pad_type = pad

        PadLayer = _get_pad_layer(pad, ndim=2)
        self.act = self._get_activation(activation)

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.encoder_skip = nn.ModuleList()

        in_ch = input_depth
        for ch_down, ch_skip in zip(num_channels_down, num_channels_skip, strict=False):
            block = self._make_down_block(in_ch, ch_down, norm_type, PadLayer)
            self.encoder.append(block)
            skip = self._make_skip_block(ch_down, ch_skip, norm_type)
            self.encoder_skip.append(skip)
            in_ch = ch_down

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()

        for i in range(self.depth):
            ch_up = num_channels_up[self.depth - 1 - i]
            ch_skip = num_channels_skip[self.depth - 1 - i]

            if i == 0:
                in_ch = num_channels_down[-1] + ch_skip
            else:
                in_ch = num_channels_up[self.depth - i] + ch_skip

            block = self._make_up_block(in_ch, ch_up, norm_type, PadLayer)
            self.decoder.append(block)

        self.output = nn.Conv2d(num_channels_up[0], output_channels, 1)

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.LeakyReLU(0.2, inplace=True)

    def _get_norm(self, channels: int, norm_type: str) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(channels)
        elif norm_type == "group":
            return nn.GroupNorm(min(32, channels), channels)
        else:
            return nn.Identity()

    def _make_down_block(
        self, in_ch: int, out_ch: int, norm_type: str, PadLayer: type[nn.Module]
    ) -> nn.Sequential:
        """Create encoder (downsampling) block with explicit padding."""
        return nn.Sequential(
            PadLayer(1),
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
            PadLayer(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def _make_skip_block(self, in_ch: int, out_ch: int, norm_type: str) -> nn.Sequential:
        """Create skip connection processing block (1x1 conv, no padding needed)."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def _make_up_block(
        self, in_ch: int, out_ch: int, norm_type: str, PadLayer: type[nn.Module]
    ) -> nn.Sequential:
        """Create decoder (upsampling) block with explicit padding."""
        return nn.Sequential(
            PadLayer(1),
            nn.Conv2d(in_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
            PadLayer(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through skip network.

        :param x: Input features (B, C, H, W)
        :returns: Reconstructed output (B, out_channels, H, W)
        """
        skips = []

        # Encoder path: downsample, then take skip from downsampled features
        for enc, skip_proc in zip(self.encoder, self.encoder_skip, strict=False):
            x = enc(x)
            skips.append(skip_proc(x))

        # Decoder path: upsample, concat with skip, process
        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)

            skip_idx = self.depth - 1 - i
            skip = skips[skip_idx]

            # Adjust sizes if needed (handle odd dimensions)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:], mode=self.upsample_mode, align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.output(x)

        if self.need_sigmoid:
            x = torch.sigmoid(x)

        return x


class SkipNetwork3D(nn.Module):
    """3D variant of SkipNetwork for volumetric data.

    :param input_depth: Number of input channels
    :param output_channels: Number of output channels
    :param num_channels_down: Channels at each encoder level
    :param num_channels_up: Channels at each decoder level
    :param num_channels_skip: Channels for skip connections
    :param activation: Activation type
    :param upsample_mode: Interpolation mode for upsampling
    :param need_sigmoid: Whether to apply sigmoid at output
    :param norm_type: Normalization type
    :param pad: Padding type ('reflection', 'replication', 'zero')
    """

    def __init__(
        self,
        input_depth: int,
        output_channels: int = 1,
        num_channels_down: list[int] | None = None,
        num_channels_up: list[int] | None = None,
        num_channels_skip: list[int] | None = None,
        activation: str = "leaky_relu",
        upsample_mode: str = "trilinear",
        need_sigmoid: bool = True,
        norm_type: str = "batch",
        pad: str = "replication",
    ):
        super().__init__()

        if num_channels_down is None:
            num_channels_down = [64, 64, 64, 64]
        if num_channels_up is None:
            num_channels_up = [64, 64, 64, 64]
        if num_channels_skip is None:
            num_channels_skip = [8, 8, 8, 8]

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.depth = len(num_channels_down)
        self.need_sigmoid = need_sigmoid
        self.upsample_mode = upsample_mode

        PadLayer = _get_pad_layer(pad, ndim=3)
        self.act = self._get_activation(activation)

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_skip = nn.ModuleList()

        in_ch = input_depth
        for ch_down, ch_skip in zip(num_channels_down, num_channels_skip, strict=False):
            block = self._make_down_block(in_ch, ch_down, norm_type, PadLayer)
            self.encoder.append(block)
            skip = self._make_skip_block(ch_down, ch_skip, norm_type)
            self.encoder_skip.append(skip)
            in_ch = ch_down

        # Decoder
        self.decoder = nn.ModuleList()

        for i in range(self.depth):
            ch_up = num_channels_up[self.depth - 1 - i]
            ch_skip = num_channels_skip[self.depth - 1 - i]

            if i == 0:
                in_ch = num_channels_down[-1] + ch_skip
            else:
                in_ch = num_channels_up[self.depth - i] + ch_skip

            block = self._make_up_block(in_ch, ch_up, norm_type, PadLayer)
            self.decoder.append(block)

        self.output = nn.Conv3d(num_channels_up[0], output_channels, 1)

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            return nn.LeakyReLU(0.2, inplace=True)

    def _get_norm(self, channels: int, norm_type: str) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm3d(channels)
        elif norm_type == "instance":
            return nn.InstanceNorm3d(channels)
        elif norm_type == "group":
            return nn.GroupNorm(min(32, channels), channels)
        else:
            return nn.Identity()

    def _make_down_block(
        self, in_ch: int, out_ch: int, norm_type: str, PadLayer: type[nn.Module]
    ) -> nn.Sequential:
        return nn.Sequential(
            PadLayer(1),
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
            PadLayer(1),
            nn.Conv3d(out_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def _make_skip_block(self, in_ch: int, out_ch: int, norm_type: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def _make_up_block(
        self, in_ch: int, out_ch: int, norm_type: str, PadLayer: type[nn.Module]
    ) -> nn.Sequential:
        return nn.Sequential(
            PadLayer(1),
            nn.Conv3d(in_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
            PadLayer(1),
            nn.Conv3d(out_ch, out_ch, 3, padding=0),
            self._get_norm(out_ch, norm_type),
            self.act,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through 3D skip network."""
        skips = []

        for enc, skip_proc in zip(self.encoder, self.encoder_skip, strict=False):
            x = enc(x)
            skips.append(skip_proc(x))

        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
            skip_idx = self.depth - 1 - i
            skip = skips[skip_idx]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode=self.upsample_mode, align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.output(x)

        if self.need_sigmoid:
            x = torch.sigmoid(x)

        return x
