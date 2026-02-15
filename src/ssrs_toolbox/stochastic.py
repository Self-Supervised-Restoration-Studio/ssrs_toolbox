"""Stochastic building blocks for variational autoencoders.

Standalone components with no internal dependencies (PyTorch only).

Provides:
- StochasticOutput: Container for latent samples and KL divergence
- StochasticLayer: Single-level stochastic bottleneck (DivNoising-style)
- MergeLayer: Combines bottom-up and top-down features (Ladder VAE-style)

Citations:
    StochasticLayer based on DivNoising:
        Prakash et al., "Fully Unsupervised Diversity Denoising with Convolutional
        Variational Autoencoders", ICLR 2021.
        arXiv: https://arxiv.org/abs/2006.06072
        Source: https://github.com/juglab/DivNoising (BSD-3-Clause)

    MergeLayer based on Ladder VAE / HDN:
        Sonderby et al., "Ladder Variational Autoencoders", NeurIPS 2016.
        arXiv: https://arxiv.org/abs/1602.02282

        Prakash et al., "Interpretable Unsupervised Diversity Denoising and
        Artefact Removal", ICLR 2022.
        arXiv: https://arxiv.org/abs/2104.01374
        Source: https://github.com/juglab/HDN (BSD-3-Clause)

See licenses/DIVNOISING.txt and licenses/HDN.txt for full license texts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


@dataclass
class StochasticOutput:
    """Output from a stochastic layer.

    :param z: Sampled latent tensor
    :param mu: Mean of the distribution
    :param logvar: Log-variance of the distribution
    :param kl: KL divergence (scalar, summed over spatial dims, averaged over batch)
    """

    z: Tensor
    mu: Tensor
    logvar: Tensor
    kl: Tensor


class StochasticLayer(nn.Module):
    """Single-level stochastic layer for VAE bottleneck.

    Learns a spatial latent distribution q(z|x) = N(mu, sigma^2).
    During training, samples via reparameterization trick.
    During eval, returns the mean (no sampling).

    KL is computed against the standard normal prior N(0, I).
    """

    def __init__(self, in_channels: int, latent_channels: int):
        """Initialize stochastic layer.

        :param in_channels: Input feature channels
        :param latent_channels: Latent space channels
        """
        super().__init__()
        self.mu_head = nn.Conv3d(in_channels, latent_channels, 3, padding=1)
        self.logvar_head = nn.Conv3d(in_channels, latent_channels, 3, padding=1)
        self.proj_back = nn.Conv3d(latent_channels, in_channels, 1)

    def forward(self, x: Tensor) -> StochasticOutput:
        """Encode features to latent space and sample.

        :param x: Input features [B, C, D, H, W]
        :returns: StochasticOutput with z projected back to in_channels
        """
        mu = self.mu_head(x)
        logvar = torch.clamp(self.logvar_head(x), -10.0, 10.0)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL vs N(0,1): -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_element = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        # Sum over C, D, H, W; mean over batch
        kl = kl_per_element.sum(dim=[1, 2, 3, 4]).mean()

        z_proj = self.proj_back(z)
        return StochasticOutput(z=z_proj, mu=mu, logvar=logvar, kl=kl)


class MergeLayer(nn.Module):
    """Combines bottom-up and top-down features for hierarchical VAE.

    Implements the merge operation in a Ladder VAE (HDN):
    - Prior p(z|top_down) from top-down features alone
    - Posterior q(z|bottom_up, top_down) from combined features
    - During training: sample from posterior, KL(q || p)
    - During eval: sample from prior
    """

    def __init__(self, channels: int, latent_channels: int):
        """Initialize merge layer.

        :param channels: Feature channels (same for bottom-up and top-down)
        :param latent_channels: Latent space channels
        """
        super().__init__()
        # Combine bottom-up + top-down → channels
        self.combine = nn.Conv3d(channels * 2, channels, 1)

        # Prior from top-down only
        self.prior_head = nn.Conv3d(channels, latent_channels * 2, 3, padding=1)

        # Posterior from combined features
        self.posterior_head = nn.Conv3d(channels, latent_channels * 2, 3, padding=1)

        # Project z back to feature space
        self.proj_back = nn.Conv3d(latent_channels, channels, 1)

    def forward(
        self, top_down: Tensor, bottom_up: Tensor | None = None
    ) -> tuple[Tensor, StochasticOutput]:
        """Merge top-down and bottom-up features, sample latent.

        :param top_down: Top-down features [B, C, D, H, W]
        :param bottom_up: Bottom-up features [B, C, D, H, W] (None during generation)
        :returns: Tuple of (updated features, StochasticOutput)
        """
        # Prior from top-down
        prior_params = self.prior_head(top_down)
        mu_p, logvar_p = prior_params.chunk(2, dim=1)
        logvar_p = torch.clamp(logvar_p, -10.0, 10.0)

        if self.training and bottom_up is not None:
            # Posterior from combined features
            # Handle spatial size mismatch
            if bottom_up.shape[2:] != top_down.shape[2:]:
                bottom_up = functional.interpolate(
                    bottom_up, size=top_down.shape[2:], mode="trilinear", align_corners=False
                )
            combined = self.combine(torch.cat([top_down, bottom_up], dim=1))
            posterior_params = self.posterior_head(combined)
            mu_q, logvar_q = posterior_params.chunk(2, dim=1)
            logvar_q = torch.clamp(logvar_q, -10.0, 10.0)

            # Sample from posterior
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + std_q * eps

            # Analytic KL(q || p)
            kl_per_element = 0.5 * (
                logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1.0
            )
            kl = kl_per_element.sum(dim=[1, 2, 3, 4]).mean()
            mu_out, logvar_out = mu_q, logvar_q
        else:
            # Sample from prior
            std_p = torch.exp(0.5 * logvar_p)
            if self.training:
                eps = torch.randn_like(std_p)
                z = mu_p + std_p * eps
            else:
                z = mu_p
            kl = top_down.new_tensor(0.0)
            mu_out, logvar_out = mu_p, logvar_p

        z_proj = self.proj_back(z)
        features = top_down + z_proj

        return features, StochasticOutput(z=z_proj, mu=mu_out, logvar=logvar_out, kl=kl)
