"""Tests for SIREN (Sinusoidal Representation Networks) layers."""

import math

import torch

from ssrs_toolbox.siren import (
    SineLayer,
    SirenKernelNet,
    SirenNetwork,
    get_mgrid,
    get_mgrid_asymmetric,
    gradient,
)


class TestSineLayer:
    """Tests for SineLayer."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        layer = SineLayer(in_features=2, out_features=64)
        x = torch.randn(10, 2)
        y = layer(x)

        assert y.shape == (10, 64)

    def test_first_layer_initialization(self):
        """Test weight initialization for first layer."""
        layer = SineLayer(in_features=3, out_features=32, is_first=True)

        # First layer weights should be uniform in [-1/in, 1/in]
        bound = 1.0 / 3
        assert layer.linear.weight.min() >= -bound - 0.01
        assert layer.linear.weight.max() <= bound + 0.01

    def test_hidden_layer_initialization(self):
        """Test weight initialization for hidden layers."""
        omega_0 = 30.0
        in_features = 64
        layer = SineLayer(
            in_features=in_features,
            out_features=64,
            is_first=False,
            omega_0=omega_0,
        )

        # Hidden layer weights should be uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
        bound = math.sqrt(6.0 / in_features) / omega_0
        assert layer.linear.weight.min() >= -bound - 0.01
        assert layer.linear.weight.max() <= bound + 0.01

    def test_output_range(self):
        """Test that output is bounded by [-1, 1] (sine function range)."""
        layer = SineLayer(in_features=2, out_features=32)
        x = torch.randn(100, 2)
        y = layer(x)

        assert y.min() >= -1.0
        assert y.max() <= 1.0

    def test_with_preactivation(self):
        """Test forward_with_preactivation method."""
        layer = SineLayer(in_features=2, out_features=32)
        x = torch.randn(10, 2)
        y, preact = layer.forward_with_preactivation(x)

        assert y.shape == (10, 32)
        assert preact.shape == (10, 32)
        assert torch.allclose(y, torch.sin(preact))


class TestSirenNetwork:
    """Tests for SirenNetwork."""

    def test_basic_construction(self):
        """Test basic network construction."""
        net = SirenNetwork(
            in_features=2,
            out_features=1,
            hidden_features=64,
            hidden_layers=3,
        )

        x = torch.randn(100, 2)
        y = net(x)

        assert y.shape == (100, 1)

    def test_outermost_linear(self):
        """Test network with linear output layer."""
        net = SirenNetwork(
            in_features=2,
            out_features=3,
            hidden_features=32,
            hidden_layers=2,
            outermost_linear=True,
        )

        x = torch.randn(50, 2)
        y = net(x)

        # With linear output, values can exceed [-1, 1]
        assert y.shape == (50, 3)

    def test_with_final_activation(self):
        """Test network with final activation."""
        net = SirenNetwork(
            in_features=2,
            out_features=10,
            hidden_features=32,
            hidden_layers=2,
            outermost_linear=True,
            final_activation="softmax",
        )

        x = torch.randn(20, 2)
        y = net(x)

        # Softmax output should sum to 1
        assert y.shape == (20, 10)
        assert torch.allclose(y.sum(dim=-1), torch.ones(20), atol=1e-5)

    def test_forward_with_coords(self):
        """Test forward_with_coords method."""
        net = SirenNetwork(in_features=2, out_features=1, hidden_features=32, hidden_layers=2)
        x = torch.randn(10, 2)
        y, coords = net.forward_with_coords(x)

        assert y.shape == (10, 1)
        assert coords.shape == (10, 2)
        assert coords.requires_grad


class TestSirenKernelNet:
    """Tests for SirenKernelNet."""

    def test_basic_kernel_generation(self):
        """Test basic kernel generation."""
        net = SirenKernelNet(kernel_size=16)
        kernel = net()

        assert kernel.shape == (1, 1, 16, 16)
        # Kernel should be normalized (sum to 1)
        assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)
        # All values should be positive (softmax output)
        assert kernel.min() >= 0

    def test_different_kernel_sizes(self):
        """Test different kernel sizes."""
        for size in [8, 32, 64]:
            net = SirenKernelNet(kernel_size=size)
            kernel = net()

            assert kernel.shape == (1, 1, size, size)
            assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        net = SirenKernelNet(kernel_size=16)
        kernel = net()

        loss = kernel.sum()
        loss.backward()

        # Check gradients exist
        for param in net.parameters():
            assert param.grad is not None


class TestMgrid:
    """Tests for coordinate grid functions."""

    def test_get_mgrid_2d(self):
        """Test 2D coordinate grid."""
        grid = get_mgrid(8, dim=2)

        assert grid.shape == (64, 2)  # 8*8 = 64 points
        assert grid.min() >= -1.0
        assert grid.max() <= 1.0

    def test_get_mgrid_3d(self):
        """Test 3D coordinate grid."""
        grid = get_mgrid(4, dim=3)

        assert grid.shape == (64, 3)  # 4*4*4 = 64 points
        assert grid.min() >= -1.0
        assert grid.max() <= 1.0

    def test_get_mgrid_asymmetric(self):
        """Test asymmetric coordinate grid."""
        grid = get_mgrid_asymmetric((8, 16))

        assert grid.shape == (128, 2)  # 8*16 = 128 points
        assert grid.min() >= -1.0
        assert grid.max() <= 1.0


class TestGradientOperators:
    """Tests for differential operators."""

    def test_gradient_computation(self):
        """Test gradient computation."""
        net = SirenNetwork(in_features=2, out_features=1, hidden_features=32, hidden_layers=2)
        coords = torch.randn(10, 2, requires_grad=True)
        y = net(coords)

        grad = gradient(y, coords)

        assert grad.shape == (10, 2)
