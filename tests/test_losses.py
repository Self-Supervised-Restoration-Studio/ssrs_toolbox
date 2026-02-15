"""Tests for ssres_toolbox.losses."""

import pytest
import torch

from ssres_toolbox.losses import (
    MaskedMSELoss,
    NormalizedL1Loss,
    NormalizedMSELoss,
    SmoothnessLoss,
    logsumexp_mean,
    safe_exp,
    safe_log,
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


class TestSafeLog:
    def test_normal_values(self):
        x = torch.tensor([1.0, 2.0, 10.0])
        result = safe_log(x)
        expected = torch.log(x)
        torch.testing.assert_close(result, expected)

    def test_zero_input_finite(self):
        result = safe_log(torch.tensor([0.0]))
        assert torch.isfinite(result).all()

    def test_negative_input_finite(self):
        result = safe_log(torch.tensor([-1.0]))
        assert torch.isfinite(result).all()

    def test_gradient_flow(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = safe_log(x).sum()
        result.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestLogsumexpMean:
    def test_matches_naive(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp_mean(x, dims=(0, 1), keepdim=False)
        expected = torch.log(torch.mean(torch.exp(x)))
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_values_stable(self):
        x = torch.tensor([[80.0, 85.0], [82.0, 81.0]])
        result = logsumexp_mean(x, dims=(0, 1), keepdim=False)
        assert torch.isfinite(result).all()
        # Should be close to max(x) + log(mean(exp(x - max(x))))
        assert result.item() > 80.0

    def test_keepdim(self):
        x = torch.randn(2, 3, 4)
        result = logsumexp_mean(x, dims=(1, 2), keepdim=True)
        assert result.shape == (2, 1, 1)

    def test_no_keepdim(self):
        x = torch.randn(2, 3, 4)
        result = logsumexp_mean(x, dims=(1, 2), keepdim=False)
        assert result.shape == (2,)

    def test_single_element(self):
        x = torch.tensor([[5.0]])
        result = logsumexp_mean(x, dims=(0, 1), keepdim=False)
        torch.testing.assert_close(result, torch.tensor(5.0), atol=1e-5, rtol=1e-5)
