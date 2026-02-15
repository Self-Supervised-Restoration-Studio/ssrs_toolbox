"""SIREN (Sinusoidal Representation Networks) layers.

Implements SIREN layers from "Implicit Neural Representations with Periodic
Activation Functions" (Sitzmann et al., NeurIPS 2020).

SIREN uses sin(omega * Wx + b) as activation, with special initialization
to maintain signal magnitude across layers.

Example:
    from nn_toolbox import SineLayer, SirenNetwork

    # Single SIREN layer
    layer = SineLayer(in_features=2, out_features=64, omega_0=30.0)

    # Full SIREN network for coordinate-based representations
    network = SirenNetwork(
        in_features=2,
        out_features=1,
        hidden_features=256,
        hidden_layers=5,
        omega_0=30.0,
    )
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    """SIREN layer: Linear followed by sinusoidal activation.

    The key insight of SIREN is that using sin(omega * Wx + b) as activation
    enables networks to represent signals with fine detail, as the sinusoidal
    activation can represent arbitrary frequency content.

    Initialization is crucial:
    - For the first layer: weights uniform in [-1/in, 1/in]
    - For hidden layers: weights uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]

    :param in_features: Number of input features
    :param out_features: Number of output features
    :param bias: Whether to include bias term
    :param is_first: Whether this is the first layer (affects initialization)
    :param omega_0: Frequency multiplier (typically 30 for first layer)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation followed by sine activation.

        :param x: Input tensor of shape (..., in_features)
        :returns: Output tensor of shape (..., out_features)
        """
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_preactivation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both output and pre-activation.

        Useful for visualization and analysis of activation distributions.

        :param x: Input tensor
        :returns: Tuple of (activated output, pre-activation values)
        """
        preact = self.omega_0 * self.linear(x)
        return torch.sin(preact), preact

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"omega_0={self.omega_0}, "
            f"is_first={self.is_first}"
        )


class SirenNetwork(nn.Module):
    """Full SIREN network for implicit neural representations.

    Creates a multi-layer network with SIREN layers, suitable for:
    - 2D/3D coordinate to RGB/density (NeRF-style)
    - Coordinate to signed distance (SDF)
    - Coordinate to blur kernel value (deblurring)

    :param in_features: Input dimension (e.g., 2 for 2D coords, 3 for 3D)
    :param out_features: Output dimension (e.g., 1 for grayscale, 3 for RGB)
    :param hidden_features: Hidden layer width
    :param hidden_layers: Number of hidden layers
    :param first_omega_0: Omega for first layer (typically 30)
    :param hidden_omega_0: Omega for hidden layers (typically 30)
    :param outermost_linear: If True, final layer is linear (no sine)
    :param final_activation: Optional activation for final layer

    Example:
        # Network for 2D blur kernel estimation
        net = SirenNetwork(
            in_features=2,  # (x, y) coordinates
            out_features=1,  # kernel value
            hidden_features=64,
            hidden_layers=3,
            outermost_linear=True,
        )

        # Generate kernel from coordinate grid
        coords = get_mgrid(kernel_size, dim=2)  # (N, 2)
        kernel = net(coords)  # (N, 1)
        kernel = kernel.view(kernel_size, kernel_size)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = False,
        final_activation: str | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        layers = []

        # First layer
        layers.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        # Output layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            # Initialize output layer similarly to hidden layers
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

        # Optional final activation
        self.final_activation: nn.Module | None = None
        if final_activation == "softmax":
            self.final_activation = nn.Softmax(dim=-1)
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN network.

        :param coords: Input coordinates of shape (..., in_features)
        :returns: Output values of shape (..., out_features)
        """
        output = self.net(coords)

        if self.final_activation is not None:
            output = self.final_activation(output)

        return output

    def forward_with_coords(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns coords with grad enabled.

        Useful for computing gradients of output w.r.t. input coordinates.

        :param coords: Input coordinates
        :returns: Tuple of (output, coords_with_grad)
        """
        coords = coords.clone().detach().requires_grad_(True)
        output = self.forward(coords)
        return output, coords


class SirenKernelNet(nn.Module):
    """SIREN network specialized for blur kernel estimation.

    Outputs a normalized (softmax) 2D kernel from coordinate inputs.
    Designed for use in blind deblurring applications.

    :param kernel_size: Output kernel size (kernel_size x kernel_size)
    :param hidden_features: Hidden layer width
    :param hidden_layers: Number of hidden layers
    :param omega_0: Frequency multiplier

    Example:
        net = SirenKernelNet(kernel_size=64, hidden_features=64, hidden_layers=3)
        kernel = net()  # (1, 1, 64, 64) normalized kernel
    """

    def __init__(
        self,
        kernel_size: int = 64,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        self.siren = SirenNetwork(
            in_features=2,
            out_features=1,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            outermost_linear=True,
        )

        # Pre-compute coordinate grid
        self.register_buffer(
            "coords",
            get_mgrid(kernel_size, dim=2),
            persistent=False,
        )

    def forward(self) -> torch.Tensor:
        """Generate normalized blur kernel.

        :returns: Kernel tensor of shape (1, 1, kernel_size, kernel_size)
        """
        # Get raw kernel values
        raw = self.siren(self.coords)  # (N, 1)

        # Reshape to 2D
        raw = raw.view(1, 1, self.kernel_size, self.kernel_size)

        # Normalize with softmax to ensure positive values summing to 1
        raw_flat = raw.view(1, -1)
        kernel_flat = F.softmax(raw_flat, dim=-1)
        kernel = kernel_flat.view(1, 1, self.kernel_size, self.kernel_size)

        return kernel

    def forward_with_coords(
        self, coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with explicit coordinates.

        :param coords: Optional coordinate grid (uses internal if None)
        :returns: Tuple of (kernel, coords_with_grad)
        """
        if coords is None:
            coords = self.coords

        coords = coords.clone().detach().requires_grad_(True)
        raw = self.siren(coords)
        raw = raw.view(1, 1, self.kernel_size, self.kernel_size)

        raw_flat = raw.view(1, -1)
        kernel_flat = F.softmax(raw_flat, dim=-1)
        kernel = kernel_flat.view(1, 1, self.kernel_size, self.kernel_size)

        return kernel, coords


def get_mgrid(sidelen: int, dim: int = 2) -> torch.Tensor:
    """Generate a coordinate grid for SIREN input.

    Creates a grid of coordinates in [-1, 1] for use with implicit
    neural representations.

    :param sidelen: Number of points per dimension
    :param dim: Number of dimensions (2 for 2D, 3 for 3D)
    :returns: Coordinate tensor of shape (sidelen^dim, dim)

    Example:
        # 2D grid for 64x64 image
        coords = get_mgrid(64, dim=2)  # (4096, 2)

        # 3D grid for 32x32x32 volume
        coords = get_mgrid(32, dim=3)  # (32768, 3)
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_mgrid_asymmetric(shape: tuple[int, ...]) -> torch.Tensor:
    """Generate coordinate grid for non-square/non-cubic shapes.

    :param shape: Tuple of dimensions (e.g., (H, W) or (D, H, W))
    :returns: Coordinate tensor of shape (prod(shape), len(shape))

    Example:
        # Non-square 2D grid
        coords = get_mgrid_asymmetric((64, 128))  # (8192, 2)
    """
    dim = len(shape)
    tensors = tuple(torch.linspace(-1, 1, steps=s) for s in shape)
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


# Differential operators for SIREN (useful for physics-informed losses)


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute gradient of y with respect to x.

    :param y: Output tensor
    :param x: Input tensor (must have requires_grad=True)
    :returns: Gradient tensor
    """
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute divergence of vector field y with respect to x.

    :param y: Vector field tensor of shape (..., dim)
    :param x: Coordinate tensor of shape (..., dim)
    :returns: Divergence tensor
    """
    div = torch.zeros_like(y[..., 0:1])
    for i in range(y.shape[-1]):
        div = (
            div
            + torch.autograd.grad(
                y[..., i],
                x,
                torch.ones_like(y[..., i]),
                create_graph=True,
                retain_graph=True,
            )[0][..., i : i + 1]
        )
    return div


def laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian of scalar field y.

    :param y: Scalar field tensor
    :param x: Coordinate tensor
    :returns: Laplacian tensor
    """
    grad = gradient(y, x)
    return divergence(grad, x)
