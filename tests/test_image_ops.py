import cv2
import numpy as np
import pytest

from openvision_lab.image_ops import ImageLoadError, load_image, to_grayscale


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
