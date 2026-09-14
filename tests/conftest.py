import cv2
import numpy as np
import pytest


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Small BGR image with a known left/right split."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 200
    return image


@pytest.fixture
def gray_image() -> np.ndarray:
    """Small grayscale image spanning the 0-255 range."""
    return np.array([[0, 127], [128, 255]], dtype=np.uint8)


@pytest.fixture
def image_file(tmp_path, bgr_image) -> str:
    """Write the sample BGR image to a temporary PNG and return its path."""
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), bgr_image)
    return str(path)
