"""3D and 2D Pixel Shuffle/Unshuffle operations.

These modules rearrange tensor elements between channels and spatial dimensions,
commonly used for upsampling in neural networks.
"""

import torch
import torch.nn as nn


class PixelShuffle3d(nn.Module):
    """3D Pixel Shuffle for upsampling.

    Rearranges elements from channels to spatial dimensions.
    Input: (B, C * r^3, D, H, W) -> Output: (B, C, D*r, H*r, W*r)
    """

    def __init__(self, scale_factor: int = 2, *, scale: int | None = None):
        """Initialize PixelShuffle3d.

        :param scale_factor: Upsampling factor for each spatial dimension
        :param scale: Alias for scale_factor (for backward compatibility)
        """
        super().__init__()
        self.scale_factor = scale if scale is not None else scale_factor

    @property
    def scale(self) -> int:
        """Alias for scale_factor (backward compatibility)."""
        return self.scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D pixel shuffle.

        :param x: Input tensor of shape (B, C*r^3, D, H, W)
        :returns: Output tensor of shape (B, C, D*r, H*r, W*r)
        """
        r = self.scale_factor
        b, c, d, h, w = x.shape
        if c % (r**3) != 0:
            msg = f"PixelShuffle3d: channels ({c}) must be divisible by scale_factor^3 ({r**3})"
            raise ValueError(msg)
        out_c = c // (r**3)

        x = x.view(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.contiguous().view(b, out_c, d * r, h * r, w * r)

        return x


class PixelUnshuffle3d(nn.Module):
    """3D Pixel Unshuffle for downsampling.

    Rearranges elements from spatial dimensions to channels.
    Input: (B, C, D, H, W) -> Output: (B, C * r^3, D/r, H/r, W/r)
    """

    def __init__(self, scale_factor: int = 2, *, scale: int | None = None):
        """Initialize PixelUnshuffle3d.

        :param scale_factor: Downsampling factor for each spatial dimension
        :param scale: Alias for scale_factor (for backward compatibility)
        """
        super().__init__()
        self.scale_factor = scale if scale is not None else scale_factor

    @property
    def scale(self) -> int:
        """Alias for scale_factor (backward compatibility)."""
        return self.scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D pixel unshuffle.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Output tensor of shape (B, C*r^3, D/r, H/r, W/r)
        """
        r = self.scale_factor
        b, c, d, h, w = x.shape
        if d % r != 0 or h % r != 0 or w % r != 0:
            msg = (
                f"PixelUnshuffle3d: spatial dims ({d}, {h}, {w}) must be "
                f"divisible by scale_factor ({r})"
            )
            raise ValueError(msg)

        x = x.view(b, c, d // r, r, h // r, r, w // r, r)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.contiguous().view(b, c * (r**3), d // r, h // r, w // r)

        return x


class PixelShuffle2d(nn.Module):
    """2D Pixel Shuffle that preserves Z dimension.

    Useful for 3D volumes where Z-axis has different sampling.
    Input: (B, C * r^2, Z, H, W) -> Output: (B, C, Z, H*r, W*r)
    """

    def __init__(self, scale_factor: int = 2, *, scale: int | None = None):
        """Initialize PixelShuffle2d.

        :param scale_factor: Upsampling factor for H and W dimensions
        :param scale: Alias for scale_factor (for backward compatibility)
        """
        super().__init__()
        self.scale_factor = scale if scale is not None else scale_factor

    @property
    def scale(self) -> int:
        """Alias for scale_factor (backward compatibility)."""
        return self.scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D pixel shuffle (preserving Z).

        :param x: Input tensor of shape (B, C*r^2, Z, H, W)
        :returns: Output tensor of shape (B, C, Z, H*r, W*r)
        """
        r = self.scale_factor
        b, c, z, h, w = x.shape
        if c % (r**2) != 0:
            msg = f"PixelShuffle2d: channels ({c}) must be divisible by scale_factor^2 ({r**2})"
            raise ValueError(msg)
        out_c = c // (r**2)

        x = x.view(b, out_c, r, r, z, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3)
        x = x.contiguous().view(b, out_c, z, h * r, w * r)

        return x


class PixelUnshuffle2d(nn.Module):
    """2D Pixel Unshuffle that preserves Z dimension.

    Useful for 3D volumes where Z-axis has different sampling.
    Input: (B, C, Z, H, W) -> Output: (B, C * r^2, Z, H/r, W/r)
    """

    def __init__(self, scale_factor: int = 2, *, scale: int | None = None):
        """Initialize PixelUnshuffle2d.

        :param scale_factor: Downsampling factor for H and W dimensions
        :param scale: Alias for scale_factor (for backward compatibility)
        """
        super().__init__()
        self.scale_factor = scale if scale is not None else scale_factor

    @property
    def scale(self) -> int:
        """Alias for scale_factor (backward compatibility)."""
        return self.scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D pixel unshuffle (preserving Z).

        :param x: Input tensor of shape (B, C, Z, H, W)
        :returns: Output tensor of shape (B, C*r^2, Z, H/r, W/r)
        """
        r = self.scale_factor
        b, c, z, h, w = x.shape
        if h % r != 0 or w % r != 0:
            msg = (
                f"PixelUnshuffle2d: spatial dims H={h}, W={w} must be "
                f"divisible by scale_factor ({r})"
            )
            raise ValueError(msg)

        x = x.view(b, c, z, h // r, r, w // r, r)
        x = x.permute(0, 1, 4, 6, 2, 3, 5)
        x = x.contiguous().view(b, c * (r**2), z, h // r, w // r)

        return x
