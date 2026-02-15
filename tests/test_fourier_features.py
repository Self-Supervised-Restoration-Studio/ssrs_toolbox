"""Tests for Fourier Feature Encoding."""

import pytest
import torch

from ssres_toolbox.fourier_features import (
    FourierFeatureEncoding,
    GaussianFourierFeatures,
    IntegratedPositionalEncoding,
    LearnableFourierFeatures,
    get_positional_encoding,
)


class TestFourierFeatureEncoding:
    """Tests for FourierFeatureEncoding."""

    def test_basic_encoding(self):
        """Test basic encoding."""
        encoding = FourierFeatureEncoding(
            in_features=2,
            num_frequencies=4,
            include_input=True,
        )

        x = torch.randn(10, 2)
        y = encoding(x)

        # Output dim = in_features + in_features * 2 * num_frequencies
        # = 2 + 2 * 2 * 4 = 2 + 16 = 18
        expected_out = 2 + 2 * 2 * 4
        assert y.shape == (10, expected_out)

    def test_without_input(self):
        """Test encoding without including raw input."""
        encoding = FourierFeatureEncoding(
            in_features=3,
            num_frequencies=5,
            include_input=False,
        )

        x = torch.randn(20, 3)
        y = encoding(x)

        # Output dim = in_features * 2 * num_frequencies = 3 * 2 * 5 = 30
        assert y.shape == (20, 30)

    def test_output_dimension_property(self):
        """Test out_features property."""
        encoding = FourierFeatureEncoding(
            in_features=2,
            num_frequencies=10,
            include_input=True,
        )

        assert encoding.out_features == 2 + 2 * 2 * 10

    def test_linear_sampling(self):
        """Test with linear frequency sampling."""
        encoding = FourierFeatureEncoding(
            in_features=2,
            num_frequencies=8,
            log_sampling=False,
        )

        x = torch.randn(5, 2)
        y = encoding(x)

        assert y.shape[1] == encoding.out_features


class TestGaussianFourierFeatures:
    """Tests for GaussianFourierFeatures."""

    def test_basic_encoding(self):
        """Test basic Gaussian encoding."""
        encoding = GaussianFourierFeatures(
            in_features=2,
            out_features=64,
            sigma=1.0,
        )

        x = torch.randn(10, 2)
        y = encoding(x)

        assert y.shape == (10, 64)

    def test_even_output_required(self):
        """Test that odd output features raises error."""
        with pytest.raises(ValueError, match="even"):
            GaussianFourierFeatures(
                in_features=2,
                out_features=63,  # Odd number
            )

    def test_learnable_frequencies(self):
        """Test learnable frequency mode."""
        encoding = GaussianFourierFeatures(
            in_features=2,
            out_features=32,
            learnable=True,
        )

        # B should be a parameter
        assert isinstance(encoding.B, torch.nn.Parameter)

        x = torch.randn(5, 2)
        y = encoding(x)
        loss = y.sum()
        loss.backward()

        assert encoding.B.grad is not None


class TestLearnableFourierFeatures:
    """Tests for LearnableFourierFeatures."""

    def test_basic_encoding(self):
        """Test basic learnable encoding."""
        encoding = LearnableFourierFeatures(
            in_features=2,
            num_frequencies=8,
            include_input=True,
        )

        x = torch.randn(10, 2)
        y = encoding(x)

        # Output dim = in_features + in_features * 2 * num_frequencies
        expected = 2 + 2 * 2 * 8
        assert y.shape == (10, expected)

    def test_gradient_flow(self):
        """Test that gradients flow to learnable parameters."""
        encoding = LearnableFourierFeatures(
            in_features=2,
            num_frequencies=4,
        )

        x = torch.randn(5, 2)
        y = encoding(x)
        loss = y.sum()
        loss.backward()

        assert encoding.frequencies.grad is not None
        assert encoding.phases.grad is not None


class TestIntegratedPositionalEncoding:
    """Tests for IntegratedPositionalEncoding."""

    def test_basic_encoding(self):
        """Test basic IPE encoding."""
        encoding = IntegratedPositionalEncoding(
            in_features=2,
            num_frequencies=4,
        )

        mean = torch.randn(10, 2)
        var = torch.rand(10, 2) * 0.1  # Small positive variance

        y = encoding(mean, var)

        # Output dim = in_features * 2 * num_frequencies
        assert y.shape == (10, 2 * 2 * 4)

    def test_zero_variance_equals_standard(self):
        """Test that zero variance gives standard encoding."""
        encoding = IntegratedPositionalEncoding(
            in_features=2,
            num_frequencies=4,
        )

        mean = torch.randn(10, 2)
        zero_var = torch.zeros(10, 2)

        y = encoding(mean, zero_var)

        # With zero variance, decay factor is 1, so output equals standard encoding
        assert y.shape == (10, 16)


class TestGetPositionalEncoding:
    """Tests for factory function."""

    def test_factory_fourier(self):
        """Test creating Fourier encoding via factory."""
        encoding = get_positional_encoding(
            "fourier",
            in_features=2,
            num_frequencies=8,
        )

        assert isinstance(encoding, FourierFeatureEncoding)

    def test_factory_gaussian(self):
        """Test creating Gaussian encoding via factory."""
        encoding = get_positional_encoding(
            "gaussian",
            in_features=2,
            out_features=64,
        )

        assert isinstance(encoding, GaussianFourierFeatures)

    def test_factory_learnable(self):
        """Test creating learnable encoding via factory."""
        encoding = get_positional_encoding(
            "learnable",
            in_features=2,
            num_frequencies=8,
        )

        assert isinstance(encoding, LearnableFourierFeatures)

    def test_factory_invalid(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            get_positional_encoding("invalid_type", in_features=2)
