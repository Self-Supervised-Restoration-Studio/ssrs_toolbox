# SSRS Toolbox

Standalone neural network building blocks for [Self-Supervised Restoration Studio](https://github.com/Self-Supervised-Restoration-Studio) projects. PyTorch-only, no framework lock-in.

## Install

```bash
uv add git+https://github.com/Self-Supervised-Restoration-Studio/ssrs_toolbox.git
```

Or for development:

```bash
git clone https://github.com/Self-Supervised-Restoration-Studio/ssrs_toolbox.git
cd ssrs_toolbox
uv sync --extra dev
```

## Module Catalogue

### Blocks & Architectures

| Module | Key exports | Description |
|--------|------------|-------------|
| `blocks` | `NAFBlock3D`, `ConvNeXtBlock3D`, `ResidualBlock` | Core 3D network blocks. NAFBlock from [NAFNet](https://arxiv.org/abs/2204.04676), ConvNeXt from [ConvNeXt](https://arxiv.org/abs/2201.03545) |
| `gated_residual` | `GatedResidualBlock3D` | HDN-style gated residual block with learned per-channel soft gating ([Prakash et al., ICLR 2022](https://arxiv.org/abs/2104.10526)) |
| `skip_network` | `SkipNetwork`, `SkipNetwork3D` | U-Net style encoder-decoder with skip connections and reflection padding |
| `stochastic` | `StochasticLayer`, `StochasticOutput`, `MergeLayer` | VAE sampling layers for hierarchical variational autoencoders (Ladder VAE / HDN / DivNoising) |

### Spatial Operations

| Module | Key exports | Description |
|--------|------------|-------------|
| `pixel_shuffle` | `PixelShuffle3d`, `PixelUnshuffle3d`, `PixelShuffle2d`, `PixelUnshuffle2d` | Sub-pixel up/downsampling for 2D and 3D tensors |
| `pool_upsample` | `pool`, `upconv222`, `ConcatMerge`, `AddMerge`, `AttentionMerge`, `UpsampleBlock` | Pooling, upsampling, and skip-connection merging strategies |
| `partial_conv` | `PartialConv2d`, `PartialConv3d` | Partial convolutions for masked inputs (NVIDIA inpainting style) |
| `conv_utils` | `conv111`, `conv333`, `conv777`, `depthwise_conv333`, `create_conv3d` | Convolution shorthands and factory functions |

### Attention & Gating

| Module | Key exports | Description |
|--------|------------|-------------|
| `attention` | `ChannelAttention`, `SpatialChannelAttention`, `LayerScaleLayer` | SE-Net style channel attention and layer scaling |
| `gates` | `SimpleGate3D`, `GatedReLUMix`, `SinGatedMix`, `ScaledSimpleGate3D` | Gating mechanisms for NAF-style activation-free networks |
| `activations` | `GEGLU`, `SwiGLU`, `SquaredReLU`, `StarReLU`, `get_activation` | Activation functions and factory |
| `normalization` | `LayerNorm3D`, `get_norm_layer` | 3D LayerNorm and normalization factory |

### Implicit Neural Representations

| Module | Key exports | Description |
|--------|------------|-------------|
| `siren` | `SirenNetwork`, `SirenKernelNet`, `SineLayer`, `gradient`, `laplacian`, `divergence` | SIREN networks ([Sitzmann et al., NeurIPS 2020](https://arxiv.org/abs/2006.09661)) with differential operators |
| `fourier_features` | `FourierFeatureEncoding`, `GaussianFourierFeatures`, `LearnableFourierFeatures`, `IntegratedPositionalEncoding` | Fourier feature mappings ([Tancik et al., NeurIPS 2020](https://arxiv.org/abs/2006.10739)) |
| `kernel_estimator` | `SIRENKernelEstimator`, `GaussianKernelEstimator`, `LearnableKernel` | PSF/blur kernel estimation via implicit neural representations |
| `adaptive_layers` | `SpatialTransformer2D`, `KernelNormalizer`, `SoftmaxCentralizer`, `AdaptiveCentralLayer` | Adaptive spatial operations for kernel processing |

### Multi-Scale

| Module | Key exports | Description |
|--------|------------|-------------|
| `pyramid` | `GaussianPyramid`, `LaplacianPyramid`, `MultiScalePyramid`, `create_scale_space` | Multi-scale image pyramids for coarse-to-fine optimization. Supports 4D and 5D tensors |

### Losses

| Module | Key exports | Description |
|--------|------------|-------------|
| `losses` | `SmoothnessLoss`, `MaskedMSELoss`, `NormalizedMSELoss`, `NormalizedL1Loss`, `safe_exp`, `safe_log`, `logsumexp_mean` | Losses and numerical stability utilities (`EPSILON`, `LOG_EPSILON`, `MAX_EXP`, `MIN_EXP`) |

## Usage

```python
from ssrs_toolbox import (
    NAFBlock3D,
    PixelShuffle3d,
    SimpleGate3D,
    ChannelAttention,
    LayerNorm3D,
)

self.upsample = PixelShuffle3d(scale_factor=2)
self.block = NAFBlock3D(channels=64)
```

## Ecosystem

`ssrs_toolbox` is the shared foundation for the Self-Supervised Restoration Studio:

```
ssrs_toolbox              ← shared neural-network building blocks (this repo)
 ├── gap_bit2bit_ssrs     ← photon-splitting strategies & losses (plugin)
 ├── deblur_inr_ssrs      ← blind deblurring via INR (plugin)
 └── ssrs                 ← framework: config, training, export, plugin discovery
```

Plugins depend on `ssrs_toolbox` for reusable components and declare
`[project.entry-points."ssrs.plugins"]` so `ssrs` can discover them at runtime.

| Repository | Description | Link |
|-----------|-------------|------|
| **ssrs_toolbox** | PyTorch building blocks | [GitHub](https://github.com/Self-Supervised-Restoration-Studio/ssrs_toolbox) |
| **gap_bit2bit_ssrs** | GAP/Bit2Bit photon denoising | [GitHub](https://github.com/Self-Supervised-Restoration-Studio/gap_bit2bit_ssrs) |
| **deblur_inr_ssrs** | Blind deblurring via implicit neural representations | [GitHub](https://github.com/Self-Supervised-Restoration-Studio/deblur_inr_ssrs) |
| **ssrs** | Framework: training, export, CLI | [GitHub](https://github.com/Self-Supervised-Restoration-Studio/ssrs) |

## Requirements

- Python >= 3.12
- PyTorch >= 2.7.0

## License

[MIT](LICENSE)
