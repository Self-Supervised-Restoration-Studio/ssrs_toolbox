"""SSRES Toolbox - Standalone neural network building blocks.

Provides specialized neural network components for SSRES projects.
All components are self-contained with minimal dependencies (PyTorch only).

Components:
- Pixel Shuffle/Unshuffle (3D and 2D variants)
- NAF (Nonlinear Activation Free) blocks
- ConvNeXt blocks
- Special activations (GatedReLU, SinGated, SimpleGate)
- Partial convolutions
- Channel/Spatial attention
- Custom normalizations
- Utility functions

Example:
    from ssres_toolbox import (
        PixelShuffle3d,
        NAFBlock3D,
        SimpleGate3D,
        ChannelAttention,
        LayerNorm3D,
    )

    # Use in your model
    self.upsample = PixelShuffle3d(scale_factor=2)
    self.block = NAFBlock3D(channels=64)
"""

# Pixel Shuffle operations
# Activation factory
from .activations import (
    GEGLU,
    ActivationLayer,
    SquaredReLU,
    StarReLU,
    SwiGLU,
    activation_function,
    get_activation,
)

# Adaptive layers for spatial operations
from .adaptive_layers import (
    AdaptiveCentralLayer,
    AdaptiveCentralLayer3D,
    KernelNormalizer,
    SoftmaxCentralizer,
    SpatialTransformer2D,
)

# Attention mechanisms
from .attention import (
    ChannelAttention,
    LayerScaleLayer,
    SpatialChannelAttention,
)

# Special blocks
from .blocks import (
    ConvNeXtBlock3D,
    NAFBlock3D,
    ResidualBlock,
)

# Convolution utilities
from .conv_utils import (
    build_conv_unit3d,
    conv111,
    conv333,
    conv777,
    create_conv3d,
    depthwise_conv333,
)

# Fourier Feature Encoding
from .fourier_features import (
    FourierFeatureEncoding,
    GaussianFourierFeatures,
    IntegratedPositionalEncoding,
    LearnableFourierFeatures,
    get_positional_encoding,
)

# Gated residual block (HDN-style)
from .gated_residual import GatedResidualBlock3D

# Gate mechanisms
from .gates import (
    GatedReLUMix,
    ScaledSimpleGate3D,
    SimpleGate3D,
    SinGatedMix,
)

# Kernel Estimation
from .kernel_estimator import (
    GaussianKernelEstimator,
    LearnableKernel,
    SIRENKernelEstimator,
)

# Losses
from .losses import (
    EPSILON,
    LOG_EPSILON,
    MAX_EXP,
    MIN_EXP,
    MaskedMSELoss,
    NormalizedL1Loss,
    NormalizedMSELoss,
    SmoothnessLoss,
    logsumexp_mean,
    safe_exp,
    safe_log,
)

# Normalization
from .normalization import (
    LayerNorm3D,
    get_norm_layer,
)

# Partial convolutions
from .partial_conv import (
    PartialConv2d,
    PartialConv3d,
)
from .pixel_shuffle import (
    PixelShuffle2d,
    PixelShuffle3d,
    PixelUnshuffle2d,
    PixelUnshuffle3d,
)

# Pooling and upsampling
from .pool_upsample import (
    AddMerge,
    AttentionMerge,
    ConcatMerge,
    UpsampleBlock,
    merge,
    pool,
    upconv222,
)

# Multi-Scale Pyramids
from .pyramid import (
    GaussianPyramid,
    LaplacianPyramid,
    MultiScalePyramid,
    create_scale_space,
)

# SIREN (Sinusoidal Representation Networks)
from .siren import (
    SineLayer,
    SirenKernelNet,
    SirenNetwork,
    divergence,
    get_mgrid,
    get_mgrid_asymmetric,
    gradient,
    laplacian,
)

# Skip Networks (U-Net style encoder-decoder)
from .skip_network import (
    SkipNetwork,
    SkipNetwork3D,
)

# Stochastic layers (VAE)
from .stochastic import MergeLayer, StochasticLayer, StochasticOutput

__all__ = [
    # Pixel Shuffle
    "PixelShuffle3d",
    "PixelUnshuffle3d",
    "PixelShuffle2d",
    "PixelUnshuffle2d",
    # Gates
    "SimpleGate3D",
    "GatedReLUMix",
    "SinGatedMix",
    "ScaledSimpleGate3D",
    # Blocks
    "NAFBlock3D",
    "ConvNeXtBlock3D",
    "ResidualBlock",
    # Attention
    "ChannelAttention",
    "SpatialChannelAttention",
    "LayerScaleLayer",
    # Normalization
    "LayerNorm3D",
    "get_norm_layer",
    # Partial Conv
    "PartialConv3d",
    "PartialConv2d",
    # Conv utilities
    "create_conv3d",
    "conv111",
    "conv333",
    "conv777",
    "depthwise_conv333",
    "build_conv_unit3d",
    # Pool/Upsample
    "pool",
    "upconv222",
    "merge",
    "ConcatMerge",
    "AddMerge",
    "AttentionMerge",
    "UpsampleBlock",
    # Activations
    "activation_function",
    "get_activation",
    "ActivationLayer",
    "SquaredReLU",
    "StarReLU",
    "GEGLU",
    "SwiGLU",
    # SIREN
    "SineLayer",
    "SirenNetwork",
    "SirenKernelNet",
    "get_mgrid",
    "get_mgrid_asymmetric",
    "gradient",
    "divergence",
    "laplacian",
    # Fourier Features
    "FourierFeatureEncoding",
    "GaussianFourierFeatures",
    "LearnableFourierFeatures",
    "IntegratedPositionalEncoding",
    "get_positional_encoding",
    # Adaptive Layers
    "AdaptiveCentralLayer",
    "AdaptiveCentralLayer3D",
    "SoftmaxCentralizer",
    "SpatialTransformer2D",
    "KernelNormalizer",
    # Skip Networks
    "SkipNetwork",
    "SkipNetwork3D",
    # Kernel Estimation
    "SIRENKernelEstimator",
    "LearnableKernel",
    "GaussianKernelEstimator",
    # Losses & numerical utilities
    "EPSILON",
    "LOG_EPSILON",
    "MAX_EXP",
    "MIN_EXP",
    "SmoothnessLoss",
    "MaskedMSELoss",
    "NormalizedMSELoss",
    "NormalizedL1Loss",
    "safe_exp",
    "safe_log",
    "logsumexp_mean",
    # Pyramids
    "MultiScalePyramid",
    "GaussianPyramid",
    "LaplacianPyramid",
    "create_scale_space",
    # Gated Residual
    "GatedResidualBlock3D",
    # Stochastic (VAE)
    "StochasticLayer",
    "StochasticOutput",
    "MergeLayer",
]
