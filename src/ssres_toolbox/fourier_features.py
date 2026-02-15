"""Fourier Feature Encoding for Positional Encoding.

Implements Fourier feature mappings for coordinate-based neural networks.
Based on "Fourier Features Let Networks Learn High Frequency Functions
in Low Dimensional Domains" (Tancik et al., NeurIPS 2020).

Fourier features help networks learn high-frequency functions by mapping
low-dimensional inputs to a higher-dimensional space using sinusoidal functions.

Example:
    from nn_toolbox import FourierFeatureEncoding

    # Standard positional encoding (like in NeRF)
    encoding = FourierFeatureEncoding(
        in_features=3,
        num_frequencies=10,
        include_input=True,
    )
    coords = torch.randn(1000, 3)  # 3D coordinates
    features = encoding(coords)  # (1000, 3 + 3*10*2) = (1000, 63)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    """Deterministic Fourier feature encoding (positional encoding).

    Maps input coordinates to higher dimensions using sinusoidal functions
    at multiple frequencies. This is the standard positional encoding
    used in NeRF and transformers.

    For input x, outputs:
    [x, sin(base^0 * pi * x), cos(base^0 * pi * x), ...,
         sin(base^(L-1) * pi * x), cos(base^(L-1) * pi * x)]

    :param in_features: Input dimension
    :param num_frequencies: Number of frequency bands (L)
    :param max_frequency: Maximum frequency exponent (base^max_freq * pi)
    :param include_input: Whether to include original input in output
    :param log_sampling: If True, frequencies are log-spaced; else linear
    :param base: Base for exponential frequency spacing (default 2.0)

    Output dimension: in_features * (1 + 2 * num_frequencies) if include_input
                     else in_features * 2 * num_frequencies
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 10,
        max_frequency: float | None = None,
        include_input: bool = True,
        log_sampling: bool = True,
        base: float = 2.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Compute output dimension
        self.out_features = in_features * 2 * num_frequencies
        if include_input:
            self.out_features += in_features

        # Generate frequency bands
        if max_frequency is None:
            max_frequency = num_frequencies - 1

        if log_sampling:
            # Frequencies: base^0, base^1, ..., base^(max_freq)
            freq_bands = base ** torch.linspace(0.0, max_frequency, num_frequencies)
        else:
            # Linear spacing
            freq_bands = torch.linspace(1.0, base**max_frequency, num_frequencies)

        # Store as buffer (not a parameter)
        self.register_buffer("freq_bands", freq_bands * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding.

        :param x: Input tensor of shape (..., in_features)
        :returns: Encoded tensor of shape (..., out_features)
        """
        # x: (..., in_features)
        # freq_bands: (num_frequencies,)

        # Compute scaled inputs for each frequency
        # Result: (..., in_features, num_frequencies)
        scaled = x.unsqueeze(-1) * self.freq_bands

        # Apply sin and cos
        # Result: (..., in_features, num_frequencies, 2)
        encoded = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        # Flatten frequency and sin/cos dimensions
        # Result: (..., in_features * num_frequencies * 2)
        encoded = encoded.view(*x.shape[:-1], -1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"num_frequencies={self.num_frequencies}, "
            f"out_features={self.out_features}, "
            f"include_input={self.include_input}"
        )


class GaussianFourierFeatures(nn.Module):
    """Random Fourier features with Gaussian-sampled frequencies.

    Instead of deterministic frequency bands, uses randomly sampled
    frequencies from a Gaussian distribution. This can better approximate
    certain kernels and has been shown to improve performance in some cases.

    For input x, computes: [sin(Bx), cos(Bx)]
    where B is a random matrix sampled from N(0, sigma^2).

    :param in_features: Input dimension
    :param out_features: Output dimension (must be even)
    :param sigma: Standard deviation of frequency distribution
    :param learnable: If True, frequencies are learnable parameters

    Reference:
        "Random Features for Large-Scale Kernel Machines" (Rahimi & Recht, 2007)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 256,
        sigma: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if out_features % 2 != 0:
            raise ValueError("out_features must be even for sin/cos pairs")

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        # Random frequency matrix: (in_features, out_features // 2)
        B = torch.randn(in_features, out_features // 2) * sigma * 2 * math.pi

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian Fourier feature encoding.

        :param x: Input tensor of shape (..., in_features)
        :returns: Encoded tensor of shape (..., out_features)
        """
        # x @ B: (..., out_features // 2)
        projected = torch.matmul(x, self.B)

        # Concatenate sin and cos
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, sigma={self.sigma}"
        )


class LearnableFourierFeatures(nn.Module):
    """Fully learnable Fourier feature encoding.

    Both frequencies and phases are learnable parameters, allowing the
    network to adapt the encoding to the specific task.

    :param in_features: Input dimension
    :param num_frequencies: Number of frequency bands per input dimension
    :param include_input: Whether to include original input
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 16,
        include_input: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        self.out_features = in_features * 2 * num_frequencies
        if include_input:
            self.out_features += in_features

        # Learnable frequencies and phases
        # Initialize frequencies similarly to standard positional encoding
        freq_init = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        freq_init = freq_init.unsqueeze(0).expand(in_features, -1) * math.pi

        self.frequencies = nn.Parameter(freq_init)
        self.phases = nn.Parameter(torch.zeros(in_features, num_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable Fourier feature encoding.

        :param x: Input tensor of shape (..., in_features)
        :returns: Encoded tensor of shape (..., out_features)
        """
        # x: (..., in_features)
        # frequencies: (in_features, num_frequencies)

        # Compute phase: x * freq + phase
        # Result: (..., in_features, num_frequencies)
        phase = x.unsqueeze(-1) * self.frequencies + self.phases

        # Apply sin and cos
        encoded = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)

        # Flatten
        encoded = encoded.view(*x.shape[:-1], -1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"num_frequencies={self.num_frequencies}, "
            f"out_features={self.out_features}"
        )


class IntegratedPositionalEncoding(nn.Module):
    """Integrated Positional Encoding (IPE) from Mip-NeRF.

    For anti-aliased rendering, encodes intervals rather than points.
    Takes mean and variance of a distribution and produces expected
    value of positional encoding over that distribution.

    :param in_features: Input dimension
    :param num_frequencies: Number of frequency bands
    :param max_frequency: Maximum frequency exponent
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 10,
        max_frequency: float = 9.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_features = in_features * 2 * num_frequencies

        # Frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, max_frequency, num_frequencies)
        self.register_buffer("freq_bands", freq_bands * math.pi)

    def forward(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """Apply integrated positional encoding.

        :param mean: Mean of coordinate distribution (..., in_features)
        :param var: Variance of coordinate distribution (..., in_features)
        :returns: Encoded tensor (..., out_features)
        """
        # Scale mean and variance by frequencies
        # (..., in_features, num_frequencies)
        scaled_mean = mean.unsqueeze(-1) * self.freq_bands
        scaled_var = var.unsqueeze(-1) * (self.freq_bands**2)

        # Expected value of sin/cos under Gaussian:
        # E[sin(x)] where x ~ N(mu, sigma^2) = sin(mu) * exp(-sigma^2/2)
        # E[cos(x)] where x ~ N(mu, sigma^2) = cos(mu) * exp(-sigma^2/2)
        decay = torch.exp(-0.5 * scaled_var)

        sin_enc = torch.sin(scaled_mean) * decay
        cos_enc = torch.cos(scaled_mean) * decay

        # Stack and flatten
        encoded = torch.stack([sin_enc, cos_enc], dim=-1)
        return encoded.view(*mean.shape[:-1], -1)


def get_positional_encoding(
    encoding_type: Literal["fourier", "gaussian", "learnable", "integrated"] = "fourier",
    **kwargs,
) -> nn.Module:
    """Factory function for positional encodings.

    :param encoding_type: Type of encoding to create
    :param kwargs: Arguments passed to encoding constructor
    :returns: Positional encoding module
    """
    if encoding_type == "fourier":
        return FourierFeatureEncoding(**kwargs)
    elif encoding_type == "gaussian":
        return GaussianFourierFeatures(**kwargs)
    elif encoding_type == "learnable":
        return LearnableFourierFeatures(**kwargs)
    elif encoding_type == "integrated":
        return IntegratedPositionalEncoding(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
