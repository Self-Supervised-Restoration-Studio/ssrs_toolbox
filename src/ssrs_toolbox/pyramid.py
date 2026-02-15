"""Multi-Scale Pyramid Utilities.

This module provides utilities for building and working with
multi-scale image pyramids for coarse-to-fine optimization.
Supports both 4D (B, C, H, W) and 5D (B, C, D, H, W) inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


class MultiScalePyramid:
    """Multi-scale image pyramid for coarse-to-fine optimization.

    Creates a pyramid of downsampled images for progressive optimization,
    commonly used in blind deconvolution and image restoration tasks.
    Supports both 4D (B, C, H, W) and 5D (B, C, D, H, W) inputs.

    :param scales: List of scale factors (e.g., [0.25, 0.5, 1.0])
    :param mode: Interpolation mode for downsampling

    Example:
        >>> pyramid = MultiScalePyramid(scales=[0.25, 0.5, 1.0])
        >>> image = torch.randn(1, 1, 256, 256)
        >>> levels = pyramid.build(image)
        >>> print(levels[0.25].shape)  # (1, 1, 64, 64)
        >>> print(levels[1.0].shape)   # (1, 1, 256, 256)
    """

    def __init__(
        self,
        scales: list[float] | None = None,
        mode: str = "bilinear",
    ):
        if scales is None:
            scales = [0.25, 0.5, 1.0]
        self.scales = sorted(scales)
        self.mode = mode

    def _get_interp_mode(self, ndim: int) -> str:
        """Get appropriate interpolation mode for tensor dimensionality."""
        if ndim == 5:
            return "trilinear" if self.mode == "bilinear" else self.mode
        return self.mode

    def build(self, image: Tensor) -> dict[float, Tensor]:
        """Build image pyramid.

        :param image: Input image (B, C, H, W), (C, H, W), (B, C, D, H, W), or (C, D, H, W)
        :returns: Dictionary mapping scale to downsampled image
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() == 4 and self._looks_like_5d_without_batch(image):
            # Ambiguous: 4D could be (B,C,H,W) or (C,D,H,W) — treat as (B,C,H,W)
            pass

        interp_mode = self._get_interp_mode(image.ndim)

        pyramid = {}
        for scale in self.scales:
            if scale == 1.0:
                pyramid[scale] = image
            else:
                size = tuple(int(s * scale) for s in image.shape[2:])
                pyramid[scale] = F.interpolate(
                    image, size=size, mode=interp_mode, align_corners=False
                )

        return pyramid

    @staticmethod
    def _looks_like_5d_without_batch(_image: Tensor) -> bool:
        """Heuristic — always returns False; 4D tensors are (B,C,H,W)."""
        return False

    def build_with_blur(
        self,
        image: Tensor,
        blur_sigma: float = 1.0,
    ) -> dict[float, Tensor]:
        """Build pyramid with Gaussian blur at each scale.

        :param image: Input image (B, C, H, W) or (B, C, D, H, W)
        :param blur_sigma: Blur sigma relative to downsampling factor
        :returns: Dictionary mapping scale to blurred+downsampled image
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        interp_mode = self._get_interp_mode(image.ndim)

        pyramid = {}
        for scale in self.scales:
            if scale == 1.0:
                pyramid[scale] = image
            else:
                # Apply blur before downsampling to avoid aliasing
                sigma = blur_sigma / scale
                blurred = self._gaussian_blur(image, sigma)
                size = tuple(int(s * scale) for s in image.shape[2:])
                pyramid[scale] = F.interpolate(
                    blurred, size=size, mode=interp_mode, align_corners=False
                )

        return pyramid

    def _gaussian_blur(self, image: Tensor, sigma: float) -> Tensor:
        """Apply Gaussian blur to image. Supports 4D and 5D tensors."""
        if sigma <= 0:
            return image

        # Create 1D Gaussian kernel
        kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        padding = kernel_size // 2

        if image.ndim == 5:
            # 3D Gaussian kernel via outer products
            B, C, D, H, W = image.shape
            kernel_3d = (
                kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            )
            kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
            kernel_3d = kernel_3d.expand(C, 1, -1, -1, -1)
            return F.conv3d(image, kernel_3d, padding=padding, groups=C)

        # 2D path
        B, C, H, W = image.shape
        kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
        kernel_2d = kernel_2d.expand(C, 1, -1, -1)
        return F.conv2d(image, kernel_2d, padding=padding, groups=C)

    def get_kernel_size(self, base_size: int, scale: float) -> int:
        """Get kernel size for a given scale.

        :param base_size: Full resolution kernel size
        :param scale: Current scale factor
        :returns: Scaled kernel size (minimum 3)
        """
        return max(3, math.ceil(base_size * scale))

    def interpolate_to_scale(
        self, tensor: Tensor, target_scale: float, reference: Tensor
    ) -> Tensor:
        """Interpolate tensor to match a target scale's size.

        :param tensor: Tensor to interpolate
        :param target_scale: Target scale factor
        :param reference: Reference tensor at full scale for size calculation
        :returns: Interpolated tensor
        """
        interp_mode = self._get_interp_mode(tensor.ndim)

        if target_scale == 1.0:
            target_size = reference.shape[2:]
        else:
            target_size = tuple(int(s * target_scale) for s in reference.shape[2:])

        return F.interpolate(tensor, size=target_size, mode=interp_mode, align_corners=False)


class GaussianPyramid(MultiScalePyramid):
    """Gaussian pyramid with proper anti-aliasing.

    Similar to MultiScalePyramid but applies Gaussian blur before
    each downsampling step to prevent aliasing artifacts.

    :param num_levels: Number of pyramid levels
    :param downsample_factor: Factor to downsample at each level
    :param blur_sigma: Blur sigma for anti-aliasing
    """

    def __init__(
        self,
        num_levels: int = 4,
        downsample_factor: float = 0.5,
        blur_sigma: float = 1.0,
    ):
        scales = [downsample_factor**i for i in range(num_levels - 1, -1, -1)]
        super().__init__(scales=scales)
        self.blur_sigma = blur_sigma

    def build(self, image: Tensor) -> dict[float, Tensor]:
        """Build Gaussian pyramid with anti-aliasing."""
        return self.build_with_blur(image, self.blur_sigma)


class LaplacianPyramid:
    """Laplacian pyramid for multi-scale image decomposition.

    Decomposes an image into multiple frequency bands, useful for
    multi-scale blending and editing operations.
    Supports both 4D (B, C, H, W) and 5D (B, C, D, H, W) inputs.

    :param num_levels: Number of pyramid levels
    :param mode: Interpolation mode

    Example:
        >>> lap_pyr = LaplacianPyramid(num_levels=4)
        >>> image = torch.randn(1, 1, 256, 256)
        >>> levels = lap_pyr.decompose(image)
        >>> reconstructed = lap_pyr.reconstruct(levels)
    """

    def __init__(
        self,
        num_levels: int = 4,
        mode: str = "bilinear",
    ):
        self.num_levels = num_levels
        self.mode = mode

    def _get_interp_mode(self, ndim: int) -> str:
        """Get appropriate interpolation mode for tensor dimensionality."""
        if ndim == 5:
            return "trilinear" if self.mode == "bilinear" else self.mode
        return self.mode

    def decompose(self, image: Tensor) -> list[Tensor]:
        """Decompose image into Laplacian pyramid.

        :param image: Input image (B, C, H, W) or (B, C, D, H, W)
        :returns: List of Laplacian levels (high to low frequency)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        interp_mode = self._get_interp_mode(image.ndim)
        levels = []
        current = image

        for _i in range(self.num_levels - 1):
            # Downsample
            down_size = tuple(s // 2 for s in current.shape[2:])
            down = F.interpolate(current, size=down_size, mode=interp_mode, align_corners=False)

            # Upsample back
            up = F.interpolate(down, size=current.shape[2:], mode=interp_mode, align_corners=False)

            # Laplacian = original - upsampled(downsampled)
            laplacian = current - up
            levels.append(laplacian)

            current = down

        # Final level is the low-frequency residual
        levels.append(current)

        return levels

    def reconstruct(self, levels: list[Tensor]) -> Tensor:
        """Reconstruct image from Laplacian pyramid.

        :param levels: List of Laplacian levels
        :returns: Reconstructed image
        """
        current = levels[-1]  # Start with low-frequency residual
        interp_mode = self._get_interp_mode(current.ndim)

        for laplacian in reversed(levels[:-1]):
            # Upsample
            current = F.interpolate(
                current, size=laplacian.shape[2:], mode=interp_mode, align_corners=False
            )
            # Add Laplacian
            current = current + laplacian

        return current


def create_scale_space(
    image: Tensor,
    sigma_start: float = 0.5,
    sigma_end: float = 8.0,
    num_scales: int = 8,
) -> tuple[list[Tensor], list[float]]:
    """Create scale-space representation of an image.

    Supports both 4D (B, C, H, W) and 5D (B, C, D, H, W) inputs.

    :param image: Input image (B, C, H, W) or (B, C, D, H, W)
    :param sigma_start: Starting sigma value
    :param sigma_end: Ending sigma value
    :param num_scales: Number of scales
    :returns: Tuple of (blurred images, sigma values)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Log-spaced sigma values
    sigmas = torch.logspace(
        math.log10(sigma_start),
        math.log10(sigma_end),
        num_scales,
    ).tolist()

    blurred_images = []

    for sigma in sigmas:
        # Create 1D Gaussian kernel
        kernel_size = int(6 * sigma + 1) | 1
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-(x.float() ** 2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        padding = kernel_size // 2

        if image.ndim == 5:
            B, C, D, H, W = image.shape
            kernel_3d = (
                kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            )
            kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
            kernel_3d = kernel_3d.expand(C, 1, -1, -1, -1)
            blurred = F.conv3d(image, kernel_3d, padding=padding, groups=C)
        else:
            B, C, H, W = image.shape
            kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
            kernel_2d = kernel_2d.expand(C, 1, -1, -1)
            blurred = F.conv2d(image, kernel_2d, padding=padding, groups=C)

        blurred_images.append(blurred)

    return blurred_images, sigmas
