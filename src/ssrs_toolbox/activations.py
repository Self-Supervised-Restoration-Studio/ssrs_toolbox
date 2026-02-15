"""Activation function utilities and factory.

Provides a factory function for creating various activation functions.
"""

import torch
import torch.nn as nn


def get_activation(name: str, inplace: bool = True, **kwargs) -> nn.Module:
    """Factory function for creating activation layers.

    :param name: Activation function name
    :param inplace: Whether to perform operation in-place (where supported)
    :param kwargs: Additional arguments for specific activations
    :returns: Activation module
    """
    name = name.lower().strip()

    if name == "relu":
        return nn.ReLU(inplace=inplace)

    elif name == "leaky_relu":
        negative_slope = kwargs.get("negative_slope", 0.2)
        return nn.LeakyReLU(negative_slope, inplace=inplace)

    elif name == "prelu":
        num_parameters = kwargs.get("num_parameters", 1)
        init = kwargs.get("init", 0.25)
        return nn.PReLU(num_parameters, init)

    elif name == "elu":
        alpha = kwargs.get("alpha", 1.0)
        return nn.ELU(alpha, inplace=inplace)

    elif name == "selu":
        return nn.SELU(inplace=inplace)

    elif name == "gelu":
        approximate = kwargs.get("approximate", "none")
        return nn.GELU(approximate=approximate)

    elif name in ("silu", "swish"):
        return nn.SiLU(inplace=inplace)

    elif name == "mish":
        return nn.Mish(inplace=inplace)

    elif name == "softplus":
        beta = kwargs.get("beta", 1)
        threshold = kwargs.get("threshold", 20)
        return nn.Softplus(beta, threshold)

    elif name == "tanh":
        return nn.Tanh()

    elif name == "sigmoid":
        return nn.Sigmoid()

    elif name == "hardswish":
        return nn.Hardswish(inplace=inplace)

    elif name == "hardsigmoid":
        return nn.Hardsigmoid(inplace=inplace)

    elif name in ("none", "identity", "linear"):
        return nn.Identity()

    else:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            "Supported: relu, leaky_relu, prelu, elu, selu, gelu, "
            "silu/swish, mish, softplus, tanh, sigmoid, hardswish, "
            "hardsigmoid, none/identity"
        )


# Alias for backward compatibility
activation_function = get_activation


class ActivationLayer(nn.Module):
    """Wrapper for activation functions with consistent interface.

    Useful when you need to pass activation as a module but want
    the flexibility of string-based configuration.
    """

    def __init__(self, name: str, **kwargs):
        """Initialize activation layer.

        :param name: Activation function name
        :param kwargs: Arguments passed to get_activation
        """
        super().__init__()
        self.activation = get_activation(name, **kwargs)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation.

        :param x: Input tensor
        :returns: Activated tensor
        """
        return self.activation(x)

    def __repr__(self) -> str:
        return f"ActivationLayer({self.name})"


class SquaredReLU(nn.Module):
    """Squared ReLU activation: max(0, x)^2.

    Used in some efficient architectures for better gradient flow.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squared ReLU.

        :param x: Input tensor
        :returns: Squared ReLU output
        """
        return torch.pow(torch.relu(x), 2)


class StarReLU(nn.Module):
    """Star ReLU: s * relu(x)^2 + b.

    Learnable scaled squared ReLU with bias, from MetaFormer.
    """

    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        scale_learnable: bool = True,
        bias_learnable: bool = True,
    ):
        """Initialize StarReLU.

        :param scale_value: Initial scale value
        :param bias_value: Initial bias value
        :param scale_learnable: Whether scale is learnable
        :param bias_learnable: Whether bias is learnable
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value), requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.tensor(bias_value), requires_grad=bias_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Star ReLU.

        :param x: Input tensor
        :returns: Star ReLU output
        """
        return self.scale * torch.pow(torch.relu(x), 2) + self.bias


class GEGLU(nn.Module):
    """GEGLU activation: x * GELU(gate).

    Gated activation used in transformer FFN layers.
    Input is split into two halves along channel dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GEGLU.

        :param x: Input tensor with even number of channels
        :returns: GEGLU output with half the channels
        """
        x, gate = x.chunk(2, dim=1)
        return x * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(gate).

    Gated activation variant using SiLU/Swish instead of GELU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU.

        :param x: Input tensor with even number of channels
        :returns: SwiGLU output with half the channels
        """
        x, gate = x.chunk(2, dim=1)
        return x * torch.nn.functional.silu(gate)
