import numpy as np
import pytest

from openvision_lab.image_ops import (
    adaptive_threshold,
    binary_threshold,
    canny,
    gaussian_blur,
    to_grayscale,
)


def test_to_grayscale_returns_single_channel(bgr_image):
    gray = to_grayscale(bgr_image)

    assert gray.shape == (8, 8)
    assert gray.dtype == np.uint8


def test_to_grayscale_uses_luminance_weights():
    red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    blue = np.array([[[255, 0, 0]]], dtype=np.uint8)

    assert to_grayscale(red)[0, 0] == 76
    assert to_grayscale(green)[0, 0] == 150
    assert to_grayscale(blue)[0, 0] == 29


def test_to_grayscale_rejects_grayscale_input(gray_image):
    with pytest.raises(ValueError):
        to_grayscale(gray_image)


def test_gaussian_blur_preserves_shape_and_dtype(gray_image):
    blurred = gaussian_blur(gray_image)

    assert blurred.shape == gray_image.shape
    assert blurred.dtype == np.uint8


def test_gaussian_blur_keeps_constant_image():
    image = np.full((6, 6), 100, dtype=np.uint8)

    blurred = gaussian_blur(image)

    np.testing.assert_array_equal(blurred, image)


def test_gaussian_blur_rejects_color_input(bgr_image):
    with pytest.raises(ValueError):
        gaussian_blur(bgr_image)


def test_gaussian_blur_accepts_custom_kernel_size():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 255

    blurred = gaussian_blur(image, kernel_size=3)

    assert blurred.shape == (7, 7)
    assert blurred.dtype == np.uint8


@pytest.mark.parametrize("kernel_size", [0, 2, 4, -1])
def test_gaussian_blur_rejects_invalid_kernel_size(kernel_size):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_blur(image, kernel_size=kernel_size)


def test_gaussian_blur_rejects_negative_sigma():
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_blur(image, sigma=-1.0)


def test_binary_threshold_outputs_only_black_and_white(gray_image):
    result = binary_threshold(gray_image)

    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_binary_threshold_applies_threshold_127(gray_image):
    result = binary_threshold(gray_image)

    expected = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_binary_threshold_rejects_color_input(bgr_image):
    with pytest.raises(ValueError):
        binary_threshold(bgr_image)


def test_binary_threshold_accepts_custom_threshold():
    image = np.array([[100, 130]], dtype=np.uint8)

    result = binary_threshold(image, threshold=120)

    np.testing.assert_array_equal(result, np.array([[0, 255]], dtype=np.uint8))


@pytest.mark.parametrize("threshold", [-1, 256])
def test_binary_threshold_rejects_out_of_range_threshold(threshold):
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        binary_threshold(image, threshold=threshold)


def test_canny_preserves_shape_and_dtype(gray_image):
    edges = canny(gray_image)

    assert edges.shape == gray_image.shape
    assert edges.dtype == np.uint8


def test_canny_outputs_only_black_and_white(gray_image):
    edges = canny(gray_image)

    assert set(np.unique(edges)).issubset({0, 255})


def test_canny_constant_image_has_no_edges():
    image = np.full((6, 6), 100, dtype=np.uint8)

    edges = canny(image)

    assert not np.any(edges)


def test_canny_detects_a_brightness_step():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 255

    edges = canny(image)

    assert np.any(edges)


def test_canny_rejects_color_input(bgr_image):
    with pytest.raises(ValueError):
        canny(bgr_image)


def test_canny_accepts_custom_parameters():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255

    edges = canny(image, low_threshold=50, high_threshold=150, aperture_size=5)

    assert edges.shape == (8, 8)
    assert edges.dtype == np.uint8


@pytest.mark.parametrize("aperture_size", [1, 2, 4, 9, 5.5, "3"])
def test_canny_rejects_invalid_aperture_size(aperture_size):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        canny(image, aperture_size=aperture_size)


@pytest.mark.parametrize("value", [-1, 256, 1.5, "100"])
def test_canny_rejects_invalid_thresholds(value):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        canny(image, low_threshold=value)
    with pytest.raises(ValueError):
        canny(image, high_threshold=value)


def test_canny_rejects_low_threshold_above_high():
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        canny(image, low_threshold=200, high_threshold=100)


def test_adaptive_threshold_preserves_shape_and_dtype(gray_image):
    result = adaptive_threshold(gray_image)

    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8


def test_adaptive_threshold_outputs_only_black_and_white(gray_image):
    result = adaptive_threshold(gray_image)

    assert set(np.unique(result)).issubset({0, 255})


def test_adaptive_threshold_rejects_color_input(bgr_image):
    with pytest.raises(ValueError):
        adaptive_threshold(bgr_image)


def test_adaptive_threshold_constant_image_with_positive_constant():
    image = np.full((8, 8), 100, dtype=np.uint8)

    result = adaptive_threshold(image, constant=2.0)

    np.testing.assert_array_equal(result, np.full((8, 8), 255, dtype=np.uint8))


def test_adaptive_threshold_constant_image_with_zero_constant():
    image = np.full((8, 8), 100, dtype=np.uint8)

    result = adaptive_threshold(image, constant=0.0)

    np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.uint8))


def test_adaptive_threshold_accepts_gaussian_method():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255

    result = adaptive_threshold(image, use_gaussian=True)

    assert result.shape == (8, 8)
    assert result.dtype == np.uint8


def test_adaptive_threshold_accepts_negative_constant():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255

    result = adaptive_threshold(image, constant=-5.0)

    assert result.shape == (8, 8)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("block_size", [1, 2, 4, 0, -3, True, 5.5, "7"])
def test_adaptive_threshold_rejects_invalid_block_size(block_size):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        adaptive_threshold(image, block_size=block_size)


@pytest.mark.parametrize("constant", ["2", True, None])
def test_adaptive_threshold_rejects_non_numeric_constant(constant):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        adaptive_threshold(image, constant=constant)


@pytest.mark.parametrize("use_gaussian", [0, 1, "mean", None])
def test_adaptive_threshold_rejects_non_boolean_method(use_gaussian):
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        adaptive_threshold(image, use_gaussian=use_gaussian)
