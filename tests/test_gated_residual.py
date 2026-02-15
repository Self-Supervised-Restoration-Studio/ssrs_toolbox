"""Tests for gated residual block."""

import torch

from ssrs_toolbox.gated_residual import GatedResidualBlock3D


class TestGatedResidualBlock3D:
    """Tests for GatedResidualBlock3D."""

    def test_forward_shape(self):
        """Output should match input shape (residual connection)."""
        block = GatedResidualBlock3D(channels=32, z_conv=True)
        x = torch.randn(2, 32, 4, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_forward_shape_2d_conv(self):
        """Should work with z_conv=False (1,3,3) kernels."""
        block = GatedResidualBlock3D(channels=32, z_conv=False)
        x = torch.randn(2, 32, 4, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (processing happened)."""
        block = GatedResidualBlock3D(channels=32, z_conv=True)
        x = torch.randn(2, 32, 4, 8, 8)
        out = block(x)
        assert not torch.equal(out, x)

    def test_with_dropout(self):
        """Should work with dropout enabled."""
        block = GatedResidualBlock3D(channels=32, z_conv=True, dropout_p=0.2)
        block.train()
        x = torch.randn(2, 32, 4, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_different_activations(self):
        """Should work with different activation functions."""
        for act in ["elu", "relu", "gelu", "silu"]:
            block = GatedResidualBlock3D(channels=16, z_conv=True, activation=act)
            x = torch.randn(1, 16, 2, 4, 4)
            out = block(x)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        block = GatedResidualBlock3D(channels=16, z_conv=True)
        x = torch.randn(1, 16, 2, 4, 4, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
