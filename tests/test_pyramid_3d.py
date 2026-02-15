"""Tests for 3D support in pyramid modules."""

import torch

from ssrs_toolbox import LaplacianPyramid, MultiScalePyramid, create_scale_space


class TestMultiScalePyramid5D:
    """Tests for MultiScalePyramid with 5D (B, C, D, H, W) inputs."""

    def test_build_5d(self):
        pyramid = MultiScalePyramid(scales=[0.5, 1.0])
        image = torch.randn(1, 1, 16, 32, 32)
        levels = pyramid.build(image)
        assert levels[1.0].shape == (1, 1, 16, 32, 32)
        assert levels[0.5].shape == (1, 1, 8, 16, 16)

    def test_build_with_blur_5d(self):
        pyramid = MultiScalePyramid(scales=[0.5, 1.0])
        image = torch.randn(1, 1, 16, 32, 32)
        levels = pyramid.build_with_blur(image, blur_sigma=1.0)
        assert levels[1.0].shape == (1, 1, 16, 32, 32)
        assert levels[0.5].shape == (1, 1, 8, 16, 16)

    def test_interpolate_to_scale_5d(self):
        pyramid = MultiScalePyramid(scales=[0.5, 1.0])
        reference = torch.randn(1, 1, 16, 32, 32)
        tensor = torch.randn(1, 1, 8, 16, 16)
        result = pyramid.interpolate_to_scale(tensor, 0.5, reference)
        assert result.shape == (1, 1, 8, 16, 16)

    def test_backward_compat_4d(self):
        pyramid = MultiScalePyramid(scales=[0.5, 1.0])
        image = torch.randn(1, 1, 64, 64)
        levels = pyramid.build(image)
        assert levels[1.0].shape == (1, 1, 64, 64)
        assert levels[0.5].shape == (1, 1, 32, 32)

    def test_multiple_scales_5d(self):
        pyramid = MultiScalePyramid(scales=[0.25, 0.5, 1.0])
        image = torch.randn(1, 2, 16, 32, 32)
        levels = pyramid.build(image)
        assert levels[0.25].shape == (1, 2, 4, 8, 8)
        assert levels[0.5].shape == (1, 2, 8, 16, 16)
        assert levels[1.0].shape == (1, 2, 16, 32, 32)


class TestLaplacianPyramid5D:
    """Tests for LaplacianPyramid with 5D inputs."""

    def test_decompose_5d(self):
        lap = LaplacianPyramid(num_levels=3)
        image = torch.randn(1, 1, 16, 32, 32)
        levels = lap.decompose(image)
        assert len(levels) == 3
        assert levels[0].shape == (1, 1, 16, 32, 32)
        assert levels[1].shape == (1, 1, 8, 16, 16)
        assert levels[2].shape == (1, 1, 4, 8, 8)

    def test_reconstruct_roundtrip_5d(self):
        lap = LaplacianPyramid(num_levels=3)
        image = torch.randn(1, 1, 16, 32, 32)
        levels = lap.decompose(image)
        reconstructed = lap.reconstruct(levels)
        assert reconstructed.shape == image.shape
        torch.testing.assert_close(reconstructed, image, atol=1e-5, rtol=1e-5)

    def test_backward_compat_4d(self):
        lap = LaplacianPyramid(num_levels=3)
        image = torch.randn(1, 1, 64, 64)
        levels = lap.decompose(image)
        reconstructed = lap.reconstruct(levels)
        assert reconstructed.shape == image.shape
        torch.testing.assert_close(reconstructed, image, atol=1e-5, rtol=1e-5)


class TestCreateScaleSpace5D:
    """Tests for create_scale_space with 5D inputs."""

    def test_5d_input(self):
        image = torch.randn(1, 1, 16, 32, 32)
        blurred, sigmas = create_scale_space(image, num_scales=3)
        assert len(blurred) == 3
        assert len(sigmas) == 3
        for b in blurred:
            assert b.shape == image.shape

    def test_backward_compat_4d(self):
        image = torch.randn(1, 1, 64, 64)
        blurred, sigmas = create_scale_space(image, num_scales=3)
        assert len(blurred) == 3
        for b in blurred:
            assert b.shape == image.shape
