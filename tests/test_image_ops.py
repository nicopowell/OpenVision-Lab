import cv2
import numpy as np
import pytest

from openvision_lab.image_ops import (
    ImageLoadError,
    binary_threshold,
    gaussian_blur,
    load_image,
    run_pipeline,
    to_grayscale,
)


def test_load_image_returns_bgr_array(tmp_path):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)

    loaded = load_image(str(path))

    assert loaded.shape == (4, 5, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, image)


def test_load_image_missing_file_raises():
    with pytest.raises(ImageLoadError):
        load_image("this_file_does_not_exist.png")


def test_load_image_invalid_file_raises(tmp_path):
    path = tmp_path / "not_an_image.png"
    path.write_text("this is not an image")

    with pytest.raises(ImageLoadError):
        load_image(str(path))


def test_to_grayscale_returns_single_channel():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    gray = to_grayscale(image)

    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8


def test_to_grayscale_uses_luminance_weights():
    red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    blue = np.array([[[255, 0, 0]]], dtype=np.uint8)

    assert to_grayscale(red)[0, 0] == 76
    assert to_grayscale(green)[0, 0] == 150
    assert to_grayscale(blue)[0, 0] == 29


def test_to_grayscale_rejects_grayscale_input():
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_grayscale(image)


def test_gaussian_blur_preserves_shape_and_dtype():
    image = np.zeros((5, 7), dtype=np.uint8)
    image[2, 3] = 255

    blurred = gaussian_blur(image)

    assert blurred.shape == (5, 7)
    assert blurred.dtype == np.uint8


def test_gaussian_blur_keeps_constant_image():
    image = np.full((6, 6), 100, dtype=np.uint8)

    blurred = gaussian_blur(image)

    assert np.all(blurred == 100)


def test_gaussian_blur_rejects_color_input():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_blur(image)


def test_binary_threshold_outputs_only_black_and_white():
    image = np.array([[0, 60, 200, 255]], dtype=np.uint8)

    result = binary_threshold(image)

    assert result.shape == (1, 4)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_binary_threshold_applies_threshold_127():
    image = np.array([[0, 127, 128, 255]], dtype=np.uint8)

    result = binary_threshold(image)

    assert result.tolist() == [[0, 0, 255, 255]]


def test_binary_threshold_rejects_color_input():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        binary_threshold(image)


def test_run_pipeline_returns_binary_grayscale():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 255

    result = run_pipeline(image)

    assert result.shape == (8, 8)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_run_pipeline_matches_manual_composition():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3, 3] = 255

    expected = binary_threshold(gaussian_blur(to_grayscale(image)))

    assert np.array_equal(run_pipeline(image), expected)


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


def test_binary_threshold_accepts_custom_threshold():
    image = np.array([[100, 130]], dtype=np.uint8)

    result = binary_threshold(image, threshold=120)

    assert result.tolist() == [[0, 255]]


@pytest.mark.parametrize("threshold", [-1, 256])
def test_binary_threshold_rejects_out_of_range_threshold(threshold):
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        binary_threshold(image, threshold=threshold)


def test_run_pipeline_uses_custom_parameters():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3, 3] = 255

    expected = binary_threshold(
        gaussian_blur(to_grayscale(image), kernel_size=3), threshold=100
    )

    result = run_pipeline(image, kernel_size=3, threshold=100)

    assert np.array_equal(result, expected)


