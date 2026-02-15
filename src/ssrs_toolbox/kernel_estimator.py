"""Kernel Estimation Networks - SIREN-based blur kernel estimators.

This module provides neural network architectures for estimating
blur kernels using implicit neural representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .adaptive_layers import AdaptiveCentralLayer, AdaptiveCentralLayer3D
from .siren import SirenNetwork, get_mgrid


class SIRENKernelEstimator(nn.Module):
    """SIREN-based blur kernel estimation network.

    Uses a SIREN (Sinusoidal Representation Network) to map coordinates
    to kernel values, then applies softmax normalization and center-of-mass
    centering for stable optimization.

    :param kernel_size: Size of the blur kernel to estimate
    :param hidden_features: Hidden layer size for SIREN
    :param hidden_layers: Number of hidden layers in SIREN
    :param omega_0: SIREN frequency factor (higher = finer details)
    :param use_centering: Whether to center kernel around center of mass
    :param spatial_dims: Number of spatial dimensions (2 or 3)

    Example:
        >>> estimator = SIRENKernelEstimator(kernel_size=64)
        >>> kernel = estimator()  # (1, 1, 64, 64)
        >>> print(kernel.sum())  # ~1.0 (normalized)

        >>> estimator3d = SIRENKernelEstimator(kernel_size=16, spatial_dims=3)
        >>> kernel = estimator3d()  # (1, 1, 16, 16, 16)
    """

    def __init__(
        self,
        kernel_size: int = 64,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        use_centering: bool = True,
        spatial_dims: int = 2,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.use_centering = use_centering
        self.spatial_dims = spatial_dims

        self.siren = SirenNetwork(
            in_features=spatial_dims,
            out_features=1,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            outermost_linear=True,
        )

        if use_centering:
            self.central_layer = (
                AdaptiveCentralLayer3D() if spatial_dims == 3 else AdaptiveCentralLayer()
            )
        else:
            self.central_layer = None

        # Pre-compute coordinate grid
        self.register_buffer(
            "coords",
            get_mgrid(kernel_size, dim=spatial_dims),
            persistent=False,
        )

    def forward(
        self,
        coords: Tensor | None = None,
        kernel_size: int | None = None,
    ) -> Tensor:
        """Estimate blur kernel.

        :param coords: Optional coordinate grid (uses pre-computed if None)
        :param kernel_size: Optional kernel size (uses self.kernel_size if None)
        :returns: Normalized, optionally centered kernel
                  (1, 1, K, K) for 2D or (1, 1, K, K, K) for 3D
        """
        if kernel_size is None:
            kernel_size = self.kernel_size

        if coords is None:
            if kernel_size == self.kernel_size:
                coords = self.coords
            else:
                coords = get_mgrid(kernel_size, dim=self.spatial_dims).to(self.coords.device)

        # Get raw kernel values
        raw = self.siren(coords)  # (N, 1)

        # Reshape to spatial dims
        spatial_shape = [kernel_size] * self.spatial_dims
        raw = raw.view(1, 1, *spatial_shape)

        # Normalize with softmax
        raw_flat = raw.view(1, -1)
        kernel_flat = F.softmax(raw_flat, dim=-1)
        kernel = kernel_flat.view(1, 1, *spatial_shape)

        # Center around center of mass
        if self.central_layer is not None:
            kernel = self.central_layer(kernel)

        return kernel

    def get_multi_scale_kernel(self, base_size: int, scales: list[float]) -> dict[float, Tensor]:
        """Get kernels at multiple scales.

        :param base_size: Full resolution kernel size
        :param scales: List of scale factors (e.g., [0.25, 0.5, 1.0])
        :returns: Dictionary mapping scale to kernel
        """
        kernels = {}
        for scale in scales:
            k_size = max(3, int(base_size * scale))
            kernels[scale] = self(kernel_size=k_size)
        return kernels


class LearnableKernel(nn.Module):
    """Direct learnable blur kernel without neural network.

    A simpler alternative to SIREN-based estimation that directly
    learns kernel values as parameters.

    :param kernel_size: Size of the blur kernel
    :param init_mode: Initialization mode ('gaussian', 'uniform', 'delta')
    :param use_centering: Whether to center kernel around center of mass
    :param spatial_dims: Number of spatial dimensions (2 or 3)

    Example:
        >>> kernel = LearnableKernel(kernel_size=31, init_mode='gaussian')
        >>> k = kernel()  # (1, 1, 31, 31)

        >>> kernel3d = LearnableKernel(kernel_size=15, spatial_dims=3)
        >>> k = kernel3d()  # (1, 1, 15, 15, 15)
    """

    def __init__(
        self,
        kernel_size: int = 31,
        init_mode: str = "gaussian",
        use_centering: bool = True,
        spatial_dims: int = 2,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.use_centering = use_centering
        self.spatial_dims = spatial_dims

        # Initialize kernel values
        kernel = self._init_kernel(kernel_size, init_mode)
        self.kernel_logits = nn.Parameter(kernel)

        if use_centering:
            self.central_layer = (
                AdaptiveCentralLayer3D() if spatial_dims == 3 else AdaptiveCentralLayer()
            )
        else:
            self.central_layer = None

    def _init_kernel(self, size: int, mode: str) -> Tensor:
        """Initialize kernel values."""
        spatial_shape = [size] * self.spatial_dims

        if mode == "gaussian":
            center = size // 2
            coords = [torch.arange(size) - center for _ in range(self.spatial_dims)]
            grids = torch.meshgrid(*coords, indexing="ij")
            sigma = size / 6
            dist_sq = sum(g**2 for g in grids)
            kernel = torch.exp(-dist_sq / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel = torch.log(kernel + 1e-10)
        elif mode == "delta":
            kernel = torch.zeros(*spatial_shape)
            center_idx = tuple(size // 2 for _ in range(self.spatial_dims))
            kernel[center_idx] = 1.0
            kernel = torch.log(kernel + 1e-10)
        else:
            # Uniform initialization
            kernel = torch.zeros(*spatial_shape)

        return kernel.view(1, 1, *spatial_shape)

    def forward(self) -> Tensor:
        """Get normalized kernel.

        :returns: Normalized kernel (1, 1, K, K) for 2D or (1, 1, K, K, K) for 3D
        """
        spatial_shape = [self.kernel_size] * self.spatial_dims

        # Softmax normalization
        kernel_flat = self.kernel_logits.view(1, -1)
        kernel_flat = F.softmax(kernel_flat, dim=-1)
        kernel = kernel_flat.view(1, 1, *spatial_shape)

        # Center around center of mass
        if self.central_layer is not None:
            kernel = self.central_layer(kernel)

        return kernel


class GaussianKernelEstimator(nn.Module):
    """Parametric Gaussian kernel estimator.

    Estimates an anisotropic Gaussian blur kernel with learnable
    parameters (sigma_x, sigma_y, rotation angle).

    Note: This class is inherently 2D due to its parametric formulation
    (sigma_x, sigma_y, theta). 3D would require a different parameterization.

    :param kernel_size: Size of the output kernel
    :param init_sigma: Initial sigma value

    Example:
        >>> estimator = GaussianKernelEstimator(kernel_size=31)
        >>> kernel = estimator()  # (1, 1, 31, 31)
    """

    def __init__(
        self,
        kernel_size: int = 31,
        init_sigma: float = 3.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        # Learnable parameters
        self.log_sigma_x = nn.Parameter(torch.tensor(init_sigma).log())
        self.log_sigma_y = nn.Parameter(torch.tensor(init_sigma).log())
        self.theta = nn.Parameter(torch.tensor(0.0))

        # Pre-compute coordinate grid
        center = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32) - center,
            torch.arange(kernel_size, dtype=torch.float32) - center,
            indexing="ij",
        )
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def forward(self) -> Tensor:
        """Compute Gaussian kernel.

        :returns: Normalized Gaussian kernel (1, 1, K, K)
        """
        sigma_x = self.log_sigma_x.exp()
        sigma_y = self.log_sigma_y.exp()
        theta = self.theta

        # Rotation
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x_rot = cos_t * self.x + sin_t * self.y
        y_rot = -sin_t * self.x + cos_t * self.y

        # Gaussian
        kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))

        # Normalize
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, self.kernel_size, self.kernel_size)
