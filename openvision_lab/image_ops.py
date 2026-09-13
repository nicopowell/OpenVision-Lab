import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image file cannot be loaded."""


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not load image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "to_grayscale expects a BGR image with shape (H, W, 3); "
            f"got shape {image.shape}."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
