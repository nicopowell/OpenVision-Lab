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


def _require_grayscale(image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image with shape (H, W); got shape {image.shape}."
        )


def gaussian_blur(
    image: np.ndarray, *, kernel_size: int = 5, sigma: float = 0.0
) -> np.ndarray:
    """Smooth a grayscale image with a Gaussian blur.

    Args:
        image: Grayscale image of shape ``(height, width)`` and dtype ``uint8``.
        kernel_size: Size of the square Gaussian kernel. It must be a positive
            odd integer. Larger values blur more.
        sigma: Standard deviation of the Gaussian kernel. It must be
            non-negative. A value of 0 lets OpenCV derive it from
            ``kernel_size``.

    Returns:
        A blurred ``uint8`` array with the same shape as the input.

    Raises:
        ValueError: If ``image`` is not a single-channel grayscale image, if
            ``kernel_size`` is not a positive odd integer, or if ``sigma`` is
            negative.
    """
    _require_grayscale(image)
    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise ValueError(f"kernel_size must be an integer; got {kernel_size!r}.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be positive and odd; got {kernel_size}.")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}.")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def binary_threshold(image: np.ndarray, *, threshold: int = 127) -> np.ndarray:
    """Convert a grayscale image into a binary black-and-white image.

    Pixels greater than the threshold become 255 and the rest become 0,
    following OpenCV's ``THRESH_BINARY`` rule. The maximum value is fixed at
    255 for now.

    Args:
        image: Grayscale image of shape ``(height, width)`` and dtype ``uint8``.
        threshold: Pixel value that separates black from white. It must be an
            integer between 0 and 255.

    Returns:
        A ``uint8`` array with the same shape, containing only 0 or 255.

    Raises:
        ValueError: If ``image`` is not a single-channel grayscale image, or if
            ``threshold`` is not an integer between 0 and 255.
    """
    _require_grayscale(image)
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise ValueError(f"threshold must be an integer; got {threshold!r}.")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be between 0 and 255; got {threshold}.")

    # cv2.threshold returns (retval, dst); only the thresholded image matters.
    _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return result


def run_pipeline(
    image: np.ndarray, *, kernel_size: int = 5, sigma: float = 0.0, threshold: int = 127
) -> np.ndarray:
    """Run the fixed processing sequence on a color image.

    The steps are applied in order: grayscale, Gaussian blur, and binary
    threshold. Each step receives the output of the previous one.

    Args:
        image: BGR image of shape ``(height, width, 3)`` and dtype ``uint8``.
        kernel_size: Gaussian blur kernel size, passed to :func:`gaussian_blur`.
        sigma: Gaussian blur sigma, passed to :func:`gaussian_blur`.
        threshold: Binary threshold value, passed to :func:`binary_threshold`.

    Returns:
        A binary ``uint8`` array of shape ``(height, width)``.

    Raises:
        ValueError: If a step receives an image that does not match its
            precondition, or if a parameter is outside its valid range.
    """
    result = to_grayscale(image)
    result = gaussian_blur(result, kernel_size=kernel_size, sigma=sigma)
    result = binary_threshold(result, threshold=threshold)
    return result
