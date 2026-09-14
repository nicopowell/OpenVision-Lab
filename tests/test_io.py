import numpy as np
import pytest

from openvision_lab.image_ops import (
    ImageLoadError,
    ImageSaveError,
    load_image,
    save_image,
)


def test_load_image_returns_bgr_array(image_file, bgr_image):
    loaded = load_image(image_file)

    assert loaded.shape == (8, 8, 3)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, bgr_image)


def test_load_image_missing_file_raises():
    with pytest.raises(ImageLoadError):
        load_image("this_file_does_not_exist.png")


def test_load_image_invalid_file_raises(tmp_path):
    path = tmp_path / "not_an_image.png"
    path.write_text("this is not an image")

    with pytest.raises(ImageLoadError):
        load_image(str(path))


def test_save_image_round_trip_color(tmp_path, bgr_image):
    path = tmp_path / "out.png"

    save_image(str(path), bgr_image)

    np.testing.assert_array_equal(load_image(str(path)), bgr_image)


def test_save_image_round_trip_grayscale(tmp_path, gray_image):
    path = tmp_path / "out.bmp"

    save_image(str(path), gray_image)

    # load_image always returns 3-channel BGR, so a grayscale file is read with
    # three identical channels.
    loaded = load_image(str(path))
    assert loaded.shape == (2, 2, 3)
    np.testing.assert_array_equal(loaded[:, :, 0], gray_image)
    np.testing.assert_array_equal(loaded[:, :, 1], gray_image)
    np.testing.assert_array_equal(loaded[:, :, 2], gray_image)


def test_save_image_invalid_path_raises(tmp_path, bgr_image):
    with pytest.raises(ImageSaveError):
        save_image(str(tmp_path / "missing_dir" / "out.png"), bgr_image)
