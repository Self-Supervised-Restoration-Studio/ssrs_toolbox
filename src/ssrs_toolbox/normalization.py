"""Custom normalization layers.

Includes LayerNorm3D and a factory function for creating normalization layers.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LayerNorm3D(nn.Module):
    """Layer normalization for 3D convolution outputs.

    Normalizes over the channel dimension for (B, C, D, H, W) tensors.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True):
        """Initialize LayerNorm3D.

        :param num_features: Number of channels
        :param eps: Small constant for numerical stability
        :param elementwise_affine: Whether to learn scale and bias
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        :param x: Input tensor of shape (B, C, D, H, W)
        :returns: Normalized tensor
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got {x.dim()}D")

        # Permute to (B, D, H, W, C) for layer_norm
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)
        # Permute back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        return x


def get_norm_layer(norm_type: str, num_features: int, num_groups: int = 8) -> nn.Module:
    """Factory function for creating normalization layers.

    :param norm_type: Type of normalization ("layer", "batch", "group", "instance", "none")
    :param num_features: Number of channels
    :param num_groups: Number of groups for GroupNorm
    :returns: Normalization layer
    """
    norm_type = norm_type.lower()

    if norm_type == "layer":
        return LayerNorm3D(num_features)

    elif norm_type == "batch":
        return nn.BatchNorm3d(num_features)

    elif norm_type == "group":
        # Ensure num_groups divides num_features
        while num_features % num_groups != 0 and num_groups > 1:
            num_groups //= 2

        if num_groups < 1:
            logger.warning(
                f"Cannot create GroupNorm for {num_features} features, using Identity as fallback"
            )
            return nn.Identity()

        return nn.GroupNorm(num_groups, num_features)

    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_features)

    elif norm_type in ("none", "identity"):
        return nn.Identity()

    else:
        logger.warning(f"Unknown norm type '{norm_type}', using Identity as fallback")
        return nn.Identity()
