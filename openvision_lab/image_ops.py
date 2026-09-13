"""Image loading and processing operations.

Images are handled as NumPy arrays following OpenCV's conventions:

- Color images are ``uint8`` with shape ``(height, width, 3)`` and channels in
  BGR order (not RGB).
- Grayscale images are ``uint8`` with shape ``(height, width)``.

This module contains no Qt code so the processing logic stays independent from
the user interface and can be tested on its own.
"""

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image file cannot be loaded.

    ``cv2.imread`` returns ``None`` instead of raising when a path does not
    exist, is unreadable, or has an unsupported format. This exception makes
    that silent failure explicit for callers.
    """


def load_image(path: str) -> np.ndarray:
    """Load an image file as a BGR color image.

    Args:
        path: Path to the image file. OpenCV detects the format from the file
            contents, so the extension is not strictly required.

    Returns:
        A ``uint8`` array of shape ``(height, width, 3)`` with BGR channels.

    Raises:
        ImageLoadError: If the file is missing, unreadable, or not a supported
            image format.
    """
    # IMREAD_COLOR always returns a 3-channel BGR image, even when the file is
    # stored as grayscale, so callers can rely on the same shape.
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not load image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR color image to grayscale.

    Args:
        image: BGR image of shape ``(height, width, 3)`` and dtype ``uint8``.

    Returns:
        A ``uint8`` array of shape ``(height, width)``.

    Raises:
        ValueError: If ``image`` is not a 3-channel image. Already grayscale
            input is rejected on purpose so the M1 contract stays explicit:
            a BGR image goes in, a grayscale image comes out.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "to_grayscale expects a BGR image with shape (H, W, 3); "
            f"got shape {image.shape}."
        )
    # cvtColor uses the BT.601 luminance weights: ~0.299 R + 0.587 G + 0.114 B.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
