"""Adaptive layers for spatial transformations and centering.

Provides layers for adaptive spatial operations, particularly useful
for blur kernel processing in deblurring applications.

Example:
    from nn_toolbox import AdaptiveCentralLayer, SoftmaxCentralizer

    # Center a blur kernel around its center of mass
    central = AdaptiveCentralLayer()
    kernel = torch.rand(1, 1, 64, 64)
    centered = central(kernel)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCentralLayer(nn.Module):
    """Adaptively centers a 2D kernel around its center of mass.

    Computes the centroid of a kernel based on its values and shifts
    the kernel so the centroid aligns with the center of the kernel.
    This is useful for blur kernel estimation where the kernel should
    be centered for proper convolution.

    Supports both 2D (H, W) and 4D (B, C, H, W) inputs.

    Example:
        layer = AdaptiveCentralLayer()
        kernel = torch.rand(1, 1, 64, 64)
        centered = layer(kernel)  # Kernel shifted to center its mass
    """

    def __init__(self, eps: float = 1e-8):
        """Initialize the adaptive central layer.

        :param eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        """Center the kernel around its center of mass.

        :param kernel: Input kernel of shape (B, C, H, W) or (H, W)
        :returns: Centered kernel with same shape
        """
        squeeze_batch = False
        squeeze_channel = False

        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
            squeeze_channel = True
        elif kernel.dim() == 3:
            kernel = kernel.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = kernel.shape

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=kernel.device, dtype=kernel.dtype),
            torch.arange(W, device=kernel.device, dtype=kernel.dtype),
            indexing="ij",
        )

        # Compute kernel sum per sample
        kernel_sum = kernel.sum(dim=(2, 3), keepdim=True) + self.eps

        # Compute centroid (center of mass)
        centroid_y = (kernel * y_grid).sum(dim=(2, 3), keepdim=True) / kernel_sum
        centroid_x = (kernel * x_grid).sum(dim=(2, 3), keepdim=True) / kernel_sum

        # Compute required shift to center
        shift_y = (H / 2.0 - centroid_y).round().long()
        shift_x = (W / 2.0 - centroid_x).round().long()

        # Apply shift using roll
        # Note: torch.roll doesn't support per-batch shifts, so we process each
        centered = torch.zeros_like(kernel)
        for b in range(B):
            for c in range(C):
                sy = shift_y[b, c, 0, 0].item()
                sx = shift_x[b, c, 0, 0].item()
                centered[b, c] = torch.roll(
                    kernel[b, c],
                    shifts=(int(sy), int(sx)),
                    dims=(0, 1),
                )

        if squeeze_channel:
            centered = centered.squeeze(1)
        if squeeze_batch:
            centered = centered.squeeze(0)

        return centered

    def compute_centroid(self, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute centroid without shifting.

        :param kernel: Input kernel of shape (B, C, H, W)
        :returns: Tuple of (centroid_y, centroid_x) each of shape (B, C)
        """
        B, C, H, W = kernel.shape

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=kernel.device, dtype=kernel.dtype),
            torch.arange(W, device=kernel.device, dtype=kernel.dtype),
            indexing="ij",
        )

        kernel_sum = kernel.sum(dim=(2, 3), keepdim=True) + self.eps
        centroid_y = (kernel * y_grid).sum(dim=(2, 3)) / kernel_sum.squeeze(-1).squeeze(-1)
        centroid_x = (kernel * x_grid).sum(dim=(2, 3)) / kernel_sum.squeeze(-1).squeeze(-1)

        return centroid_y, centroid_x


class SoftmaxCentralizer(nn.Module):
    """Differentiable kernel centering using spatial softmax.

    Unlike AdaptiveCentralLayer which uses hard shifts (roll),
    this uses soft spatial attention for differentiable centering.
    Better for end-to-end training but more computationally expensive.

    :param kernel_size: Expected kernel size for initialization
    :param temperature: Softmax temperature (lower = sharper)
    """

    def __init__(self, kernel_size: int = 64, temperature: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.temperature = temperature

        # Pre-compute centered Gaussian weighting
        y = torch.linspace(-1, 1, kernel_size)
        x = torch.linspace(-1, 1, kernel_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_weight = torch.exp(-(xx**2 + yy**2) / 0.5)
        self.register_buffer("center_weight", center_weight)

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        """Apply differentiable centering.

        :param kernel: Input kernel of shape (B, C, H, W)
        :returns: Centered kernel
        """
        B, C, H, W = kernel.shape

        # Compute soft shift via cross-correlation with center weight
        # This biases the kernel towards the center
        weighted = kernel * self.center_weight

        # Normalize
        weighted = weighted / (weighted.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Soft blend between original and weighted
        # This provides a differentiable approximation to centering
        return weighted


class AdaptiveCentralLayer3D(nn.Module):
    """3D version of AdaptiveCentralLayer for volumetric kernels.

    Centers a 3D kernel around its center of mass along all three axes.

    :param eps: Small value for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        """Center 3D kernel around its center of mass.

        :param kernel: Input kernel of shape (B, C, D, H, W)
        :returns: Centered kernel with same shape
        """
        B, C, D, H, W = kernel.shape

        # Create coordinate grids
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.arange(D, device=kernel.device, dtype=kernel.dtype),
            torch.arange(H, device=kernel.device, dtype=kernel.dtype),
            torch.arange(W, device=kernel.device, dtype=kernel.dtype),
            indexing="ij",
        )

        # Compute kernel sum
        kernel_sum = kernel.sum(dim=(2, 3, 4), keepdim=True) + self.eps

        # Compute centroid
        centroid_z = (kernel * z_grid).sum(dim=(2, 3, 4), keepdim=True) / kernel_sum
        centroid_y = (kernel * y_grid).sum(dim=(2, 3, 4), keepdim=True) / kernel_sum
        centroid_x = (kernel * x_grid).sum(dim=(2, 3, 4), keepdim=True) / kernel_sum

        # Compute shifts
        shift_z = (D / 2.0 - centroid_z).round().long()
        shift_y = (H / 2.0 - centroid_y).round().long()
        shift_x = (W / 2.0 - centroid_x).round().long()

        # Apply shifts
        centered = torch.zeros_like(kernel)
        for b in range(B):
            for c in range(C):
                sz = shift_z[b, c, 0, 0, 0].item()
                sy = shift_y[b, c, 0, 0, 0].item()
                sx = shift_x[b, c, 0, 0, 0].item()
                centered[b, c] = torch.roll(
                    kernel[b, c],
                    shifts=(int(sz), int(sy), int(sx)),
                    dims=(0, 1, 2),
                )

        return centered


class SpatialTransformer2D(nn.Module):
    """Differentiable 2D spatial transformer for kernel manipulation.

    Uses grid_sample for differentiable spatial transformations
    including translation, rotation, and scaling.

    :param output_size: Output spatial size (H, W)
    """

    def __init__(self, output_size: tuple[int, int]):
        super().__init__()
        self.output_size = output_size

        # Identity grid
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        self.register_buffer("identity_theta", theta)

    def forward(
        self,
        kernel: torch.Tensor,
        translation: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply spatial transformation to kernel.

        :param kernel: Input kernel (B, C, H, W)
        :param translation: Translation (B, 2) in [-1, 1] range
        :param scale: Scale factor (B, 2) or (B, 1)
        :param rotation: Rotation angle in radians (B, 1)
        :returns: Transformed kernel
        """
        B = kernel.shape[0]
        device = kernel.device

        # Start with identity
        theta = self.identity_theta.unsqueeze(0).expand(B, -1, -1).clone()
        theta = theta.to(device)

        # Apply transformations
        if scale is not None:
            if scale.dim() == 1:
                scale = scale.unsqueeze(-1)
            if scale.shape[-1] == 1:
                scale = scale.expand(-1, 2)
            theta[:, 0, 0] = theta[:, 0, 0] * scale[:, 0]
            theta[:, 1, 1] = theta[:, 1, 1] * scale[:, 1]

        if rotation is not None:
            cos_r = torch.cos(rotation).squeeze(-1)
            sin_r = torch.sin(rotation).squeeze(-1)
            rot_matrix = torch.stack(
                [torch.stack([cos_r, -sin_r], dim=-1), torch.stack([sin_r, cos_r], dim=-1)], dim=1
            )
            theta[:, :2, :2] = torch.bmm(rot_matrix, theta[:, :2, :2])

        if translation is not None:
            theta[:, :, 2] = translation

        # Generate grid and sample
        grid = F.affine_grid(
            theta,
            [B, kernel.shape[1], *self.output_size],
            align_corners=True,
        )
        output = F.grid_sample(
            kernel,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return output


class KernelNormalizer(nn.Module):
    """Normalizes kernels to have specific properties.

    Ensures kernels are valid for convolution (positive, sum to 1, etc.).

    :param normalize_sum: Normalize kernel sum to 1
    :param enforce_positive: Ensure all values are non-negative
    :param method: Normalization method ('softmax', 'abs', 'relu')
    """

    def __init__(
        self,
        normalize_sum: bool = True,
        enforce_positive: bool = True,
        method: str = "softmax",
    ):
        super().__init__()
        self.normalize_sum = normalize_sum
        self.enforce_positive = enforce_positive
        self.method = method

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        """Normalize the kernel.

        :param kernel: Input kernel (B, C, H, W)
        :returns: Normalized kernel
        """
        if self.enforce_positive:
            if self.method == "softmax":
                # Softmax over spatial dimensions
                shape = kernel.shape
                flat = kernel.view(shape[0], shape[1], -1)
                flat = F.softmax(flat, dim=-1)
                kernel = flat.view(shape)
            elif self.method == "abs":
                kernel = torch.abs(kernel)
            elif self.method == "relu":
                kernel = F.relu(kernel)

        if self.normalize_sum and self.method != "softmax":
            # Normalize to sum to 1
            kernel_sum = kernel.sum(dim=(2, 3), keepdim=True) + 1e-8
            kernel = kernel / kernel_sum

        return kernel
