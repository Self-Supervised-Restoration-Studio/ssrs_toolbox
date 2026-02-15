"""Tests for 3D support in kernel estimator modules."""

import pytest

from ssrs_toolbox import LearnableKernel, SIRENKernelEstimator


class TestSIRENKernelEstimator3D:
    """Tests for SIRENKernelEstimator with spatial_dims=3."""

    def test_output_shape(self):
        estimator = SIRENKernelEstimator(kernel_size=8, spatial_dims=3, hidden_features=16)
        kernel = estimator()
        assert kernel.shape == (1, 1, 8, 8, 8)

    def test_kernel_sums_to_one(self):
        estimator = SIRENKernelEstimator(kernel_size=8, spatial_dims=3, hidden_features=16)
        kernel = estimator()
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_kernel_size_override(self):
        estimator = SIRENKernelEstimator(kernel_size=8, spatial_dims=3, hidden_features=16)
        kernel = estimator(kernel_size=4)
        assert kernel.shape == (1, 1, 4, 4, 4)
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_centering(self):
        estimator = SIRENKernelEstimator(
            kernel_size=8, spatial_dims=3, hidden_features=16, use_centering=False
        )
        kernel = estimator()
        assert kernel.shape == (1, 1, 8, 8, 8)
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_backward_compat_2d(self):
        estimator = SIRENKernelEstimator(kernel_size=16, hidden_features=16)
        kernel = estimator()
        assert kernel.shape == (1, 1, 16, 16)
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestLearnableKernel3D:
    """Tests for LearnableKernel with spatial_dims=3."""

    def test_output_shape_gaussian(self):
        kernel_mod = LearnableKernel(kernel_size=7, spatial_dims=3, init_mode="gaussian")
        kernel = kernel_mod()
        assert kernel.shape == (1, 1, 7, 7, 7)

    def test_output_shape_delta(self):
        kernel_mod = LearnableKernel(kernel_size=7, spatial_dims=3, init_mode="delta")
        kernel = kernel_mod()
        assert kernel.shape == (1, 1, 7, 7, 7)

    def test_output_shape_uniform(self):
        kernel_mod = LearnableKernel(kernel_size=7, spatial_dims=3, init_mode="uniform")
        kernel = kernel_mod()
        assert kernel.shape == (1, 1, 7, 7, 7)

    def test_kernel_sums_to_one(self):
        kernel_mod = LearnableKernel(kernel_size=7, spatial_dims=3)
        kernel = kernel_mod()
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_backward_compat_2d(self):
        kernel_mod = LearnableKernel(kernel_size=15)
        kernel = kernel_mod()
        assert kernel.shape == (1, 1, 15, 15)
        assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_gradient_flows(self):
        kernel_mod = LearnableKernel(kernel_size=5, spatial_dims=3)
        kernel = kernel_mod()
        loss = kernel.sum()
        loss.backward()
        assert kernel_mod.kernel_logits.grad is not None
