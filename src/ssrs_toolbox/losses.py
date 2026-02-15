"""Standalone loss functions for SSRS projects.

Self-contained PyTorch modules with no internal project dependencies.
Includes kernel smoothness, masked MSE, and normalized (scale-invariant) losses.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Numerical stability utilities
# ---------------------------------------------------------------------------

EPSILON: Final[float] = 1e-8
LOG_EPSILON: Final[float] = -18.42  # log(1e-8)
MAX_EXP: Final[float] = 88.0  # Maximum safe value for exp() to avoid overflow
MIN_EXP: Final[float] = -88.0  # Minimum safe value for exp() to avoid underflow


def safe_exp(x: Tensor, max_val: float = MAX_EXP) -> Tensor:
    """Numerically stable exponential with clamping.

    :param x: Input tensor
    :param max_val: Maximum input value before clamping
    :returns: Clamped exponential
    """
    return torch.exp(torch.clamp(x, min=MIN_EXP, max=max_val))


def safe_log(x: Tensor, eps: float = EPSILON) -> Tensor:
    """Numerically stable logarithm.

    :param x: Input tensor
    :param eps: Minimum clamp value to avoid log(0)
    :returns: Clamped logarithm
    """
    return torch.log(torch.clamp(x, min=eps))


def logsumexp_mean(x: Tensor, dims: tuple[int, ...], keepdim: bool = True) -> Tensor:
    """Compute log(mean(exp(x))) using numerically stable log-sum-exp.

    Equivalent to ``torch.log(torch.mean(torch.exp(x)))`` but stable.

    :param x: Input tensor
    :param dims: Dimensions to reduce
    :param keepdim: Whether to keep reduced dimensions
    :returns: log-mean-exp result
    """
    n_elements = 1
    for d in dims:
        n_elements *= x.shape[d]
    log_n = math.log(n_elements)
    return torch.logsumexp(x, dim=dims, keepdim=keepdim) - log_n


# ---------------------------------------------------------------------------
# Kernel smoothness
# ---------------------------------------------------------------------------


class SmoothnessLoss(nn.Module):
    """Kernel smoothness loss using gradient penalties.

    Penalizes sharp transitions in a kernel, encouraging smooth blur estimates.
    Auto-detects 2D (4D tensor) vs 3D (5D tensor) inputs.

    :param mode: Penalty mode ('l1' or 'l2')
    """

    def __init__(self, mode: str = "l2"):
        super().__init__()
        self.mode = mode

    def _penalty(self, grad: Tensor) -> Tensor:
        if self.mode == "l1":
            return torch.abs(grad).mean()
        return (grad**2).mean()

    def forward(self, kernel: Tensor) -> Tensor:
        """Compute smoothness loss on kernel tensor.

        :param kernel: Kernel tensor (B, C, H, W) or (B, C, D, H, W)
        :returns: Scalar smoothness penalty
        """
        if kernel.ndim == 5:
            grad_d = kernel[:, :, 1:, :, :] - kernel[:, :, :-1, :, :]
            grad_h = kernel[:, :, :, 1:, :] - kernel[:, :, :, :-1, :]
            grad_w = kernel[:, :, :, :, 1:] - kernel[:, :, :, :, :-1]
            return self._penalty(grad_d) + self._penalty(grad_h) + self._penalty(grad_w)

        grad_x = kernel[:, :, :, 1:] - kernel[:, :, :, :-1]
        grad_y = kernel[:, :, 1:, :] - kernel[:, :, :-1, :]
        return self._penalty(grad_x) + self._penalty(grad_y)


# ---------------------------------------------------------------------------
# Masked MSE
# ---------------------------------------------------------------------------


class MaskedMSELoss(nn.Module):
    """MSE loss with explicit mask support.

    When a mask is provided, only masked pixels contribute to the loss.

    :param reduction: Reduction mode ('none', 'mean', 'sum')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute masked MSE loss.

        :param prediction: Model output
        :param target: Ground truth
        :param mask: Binary mask — 1 = valid, 0 = ignore
        :returns: Loss tensor
        """
        diff = prediction - target
        squared = diff * diff

        if mask is not None:
            squared = squared * mask
            if self.reduction == "mean":
                return squared.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return squared.sum()
            else:
                return squared
        else:
            if self.reduction == "mean":
                return squared.mean()
            elif self.reduction == "sum":
                return squared.sum()
            else:
                return squared


# ---------------------------------------------------------------------------
# Normalized (scale-invariant) losses
# ---------------------------------------------------------------------------


class NormalizedMSELoss(nn.Module):
    """Normalized MSE loss with exponential transformation.

    Applies exp to the prediction (assuming log-space output), normalizes
    both prediction and target by their respective means, then computes MSE.
    This makes the loss scale-invariant.

    :param epsilon: Small constant for numerical stability
    :param reduce_dims: Dimensions to reduce over for mean computation
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        reduce_dims: tuple[int, ...] = (-1, -2, -3, -4),
    ):
        super().__init__()
        self.epsilon = epsilon
        self._reduce_dims = reduce_dims

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute normalized MSE loss.

        :param prediction: Model output in log-space
        :param target: Ground truth
        :param mask: Optional binary mask
        :returns: Scalar loss tensor
        """
        result = prediction * mask if mask is not None else prediction
        target_masked = target * mask if mask is not None else target

        exp_energy = safe_exp(result)

        exp_mean = torch.mean(exp_energy, dim=self._reduce_dims, keepdim=True)
        target_mean = torch.mean(target_masked, dim=self._reduce_dims, keepdim=True)

        exp_energy_norm = exp_energy / torch.clamp(exp_mean, min=self.epsilon)
        target_norm = target_masked / torch.clamp(target_mean, min=self.epsilon)

        diff = exp_energy_norm - target_norm
        return torch.mean(diff * diff)


class NormalizedL1Loss(nn.Module):
    """Normalized L1 loss with exponential transformation.

    Same as NormalizedMSELoss but uses L1 distance. More robust to outliers.

    :param epsilon: Small constant for numerical stability
    :param reduce_dims: Dimensions to reduce over for mean computation
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        reduce_dims: tuple[int, ...] = (-1, -2, -3, -4),
    ):
        super().__init__()
        self.epsilon = epsilon
        self._reduce_dims = reduce_dims

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute normalized L1 loss.

        :param prediction: Model output in log-space
        :param target: Ground truth
        :param mask: Optional binary mask
        :returns: Scalar loss tensor
        """
        result = prediction * mask if mask is not None else prediction
        target_masked = target * mask if mask is not None else target

        exp_energy = safe_exp(result)

        exp_mean = torch.mean(exp_energy, dim=self._reduce_dims, keepdim=True)
        target_mean = torch.mean(target_masked, dim=self._reduce_dims, keepdim=True)

        exp_energy_norm = exp_energy / torch.clamp(exp_mean, min=self.epsilon)
        target_norm = target_masked / torch.clamp(target_mean, min=self.epsilon)

        return torch.mean(torch.abs(exp_energy_norm - target_norm))
