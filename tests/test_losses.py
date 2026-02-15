"""Tests for ssrs_toolbox.losses."""

import pytest
import torch

from ssrs_toolbox.losses import (
    MaskedMSELoss,
    NormalizedL1Loss,
    NormalizedMSELoss,
    SmoothnessLoss,
    safe_exp,
)


class TestSmoothnessLoss:
    def test_2d_output_scalar(self):
        loss_fn = SmoothnessLoss(mode="l2")
        kernel = torch.randn(1, 1, 7, 7)
        loss = loss_fn(kernel)
        assert loss.dim() == 0

    def test_3d_output_scalar(self):
        loss_fn = SmoothnessLoss(mode="l2")
        kernel = torch.randn(1, 1, 5, 5, 5)
        loss = loss_fn(kernel)
        assert loss.dim() == 0

    def test_constant_kernel_zero_loss(self):
        loss_fn = SmoothnessLoss(mode="l2")
        kernel = torch.ones(1, 1, 7, 7)
        loss = loss_fn(kernel)
        assert loss.item() == 0.0

    def test_l1_mode(self):
        loss_fn = SmoothnessLoss(mode="l1")
        kernel = torch.randn(1, 1, 7, 7)
        loss = loss_fn(kernel)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        loss_fn = SmoothnessLoss(mode="l2")
        kernel = torch.randn(1, 1, 7, 7, requires_grad=True)
        loss = loss_fn(kernel)
        loss.backward()
        assert kernel.grad is not None

    def test_3d_constant_zero(self):
        loss_fn = SmoothnessLoss(mode="l2")
        kernel = torch.ones(1, 1, 5, 5, 5)
        loss = loss_fn(kernel)
        assert loss.item() == 0.0


class TestSafeExp:
    def test_normal_values(self):
        x = torch.tensor([0.0, 1.0, -1.0])
        result = safe_exp(x)
        expected = torch.exp(x)
        torch.testing.assert_close(result, expected)

    def test_clamping_large(self):
        result = safe_exp(torch.tensor([200.0]))
        assert torch.isfinite(result).all()

    def test_clamping_small(self):
        result = safe_exp(torch.tensor([-200.0]))
        assert torch.isfinite(result).all()


class TestMaskedMSELoss:
    def test_without_mask_equals_mse(self):
        loss_fn = MaskedMSELoss(reduction="mean")
        pred = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, target)
        expected = torch.nn.functional.mse_loss(pred, target)
        torch.testing.assert_close(loss, expected)

    def test_with_mask_only_valid_pixels(self):
        loss_fn = MaskedMSELoss(reduction="mean")
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(pred, target, mask)
        # Only first pixel: (1 - 0)^2 / 1 = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_reduction_sum(self):
        loss_fn = MaskedMSELoss(reduction="sum")
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[0.0, 0.0]])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(5.0, abs=1e-6)

    def test_reduction_none(self):
        loss_fn = MaskedMSELoss(reduction="none")
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[0.0, 0.0]])
        loss = loss_fn(pred, target)
        assert loss.shape == pred.shape

    def test_gradient_flow(self):
        loss_fn = MaskedMSELoss()
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        loss = loss_fn(pred, target, mask)
        loss.backward()
        assert pred.grad is not None


class TestNormalizedMSELoss:
    def test_scalar_output(self):
        loss_fn = NormalizedMSELoss()
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_finite_output(self):
        loss_fn = NormalizedMSELoss()
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        assert torch.isfinite(loss_fn(pred, target))

    def test_gradient_flow(self):
        loss_fn = NormalizedMSELoss()
        pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestNormalizedL1Loss:
    def test_scalar_output(self):
        loss_fn = NormalizedL1Loss()
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_finite_output(self):
        loss_fn = NormalizedL1Loss()
        pred = torch.randn(2, 1, 4, 8, 8)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        assert torch.isfinite(loss_fn(pred, target))

    def test_gradient_flow(self):
        loss_fn = NormalizedL1Loss()
        pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 4, 8, 8) + 0.1
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
