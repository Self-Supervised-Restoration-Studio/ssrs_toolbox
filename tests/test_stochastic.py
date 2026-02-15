"""Tests for stochastic VAE building blocks."""

import torch

from ssrs_toolbox.stochastic import MergeLayer, StochasticLayer, StochasticOutput


class TestStochasticLayer:
    """Tests for StochasticLayer."""

    def test_forward_shape(self):
        """Output z should have same shape as input (due to proj_back)."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        x = torch.randn(2, 32, 4, 8, 8)
        out = layer(x)
        assert isinstance(out, StochasticOutput)
        assert out.z.shape == x.shape

    def test_mu_logvar_shape(self):
        """mu and logvar should have latent_channels."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        x = torch.randn(2, 32, 4, 8, 8)
        out = layer(x)
        assert out.mu.shape == (2, 16, 4, 8, 8)
        assert out.logvar.shape == (2, 16, 4, 8, 8)

    def test_kl_positive_training(self):
        """KL should be non-negative during training."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        layer.train()
        x = torch.randn(2, 32, 4, 8, 8)
        out = layer(x)
        assert out.kl.item() >= 0.0

    def test_kl_scalar(self):
        """KL should be a scalar tensor."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        x = torch.randn(2, 32, 4, 8, 8)
        out = layer(x)
        assert out.kl.dim() == 0

    def test_eval_returns_mu(self):
        """In eval mode, z should equal mu (no sampling)."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        layer.eval()
        x = torch.randn(2, 32, 4, 8, 8)
        with torch.no_grad():
            out = layer(x)
        # proj_back(mu) should be deterministic
        with torch.no_grad():
            out2 = layer(x)
        torch.testing.assert_close(out.z, out2.z)

    def test_training_is_stochastic(self):
        """In training mode, z should differ across calls (stochastic sampling)."""
        layer = StochasticLayer(in_channels=32, latent_channels=16)
        layer.train()
        x = torch.randn(2, 32, 4, 8, 8)
        out1 = layer(x)
        out2 = layer(x)
        # Very unlikely to be exactly equal
        assert not torch.equal(out1.mu, out1.z) or not torch.equal(out2.mu, out2.z)


class TestMergeLayer:
    """Tests for MergeLayer."""

    def test_forward_shape(self):
        """Output features should have same shape as top_down."""
        layer = MergeLayer(channels=32, latent_channels=16)
        td = torch.randn(2, 32, 4, 8, 8)
        bu = torch.randn(2, 32, 4, 8, 8)
        layer.train()
        features, stoch_out = layer(td, bu)
        assert features.shape == td.shape
        assert isinstance(stoch_out, StochasticOutput)

    def test_kl_positive_with_posterior(self):
        """KL(q||p) should be non-negative when using posterior."""
        layer = MergeLayer(channels=32, latent_channels=16)
        layer.train()
        td = torch.randn(2, 32, 4, 8, 8)
        bu = torch.randn(2, 32, 4, 8, 8)
        _, stoch_out = layer(td, bu)
        assert stoch_out.kl.item() >= 0.0

    def test_eval_uses_prior(self):
        """In eval mode, should use prior (no bottom_up needed)."""
        layer = MergeLayer(channels=32, latent_channels=16)
        layer.eval()
        td = torch.randn(2, 32, 4, 8, 8)
        with torch.no_grad():
            features, stoch_out = layer(td, None)
        assert features.shape == td.shape
        # KL should be 0 when using prior
        assert stoch_out.kl.item() == 0.0

    def test_spatial_mismatch_interpolation(self):
        """Should handle spatial size mismatch between top_down and bottom_up."""
        layer = MergeLayer(channels=32, latent_channels=16)
        layer.train()
        td = torch.randn(2, 32, 4, 8, 8)
        bu = torch.randn(2, 32, 8, 16, 16)  # Different spatial size
        features, stoch_out = layer(td, bu)
        assert features.shape == td.shape
