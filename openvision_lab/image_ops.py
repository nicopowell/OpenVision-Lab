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


class ImageSaveError(Exception):
    """Raised when an image cannot be written to a file.

    ``cv2.imwrite`` returns ``False`` instead of raising on failure, so this
    exception makes that silent failure explicit for callers.
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


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image array to a file.

    OpenCV chooses the output format from the file extension.

    Args:
        path: Destination path including the file extension, for example
            ``result.png`` or ``result.jpg``.
        image: Image array to write. Grayscale ``(height, width)`` and BGR
            ``(height, width, 3)`` ``uint8`` images are supported.

    Raises:
        ImageSaveError: If OpenCV cannot write the file, for example because
            the extension is unsupported or the parent directory does not
            exist.
    """
    if not cv2.imwrite(path, image):
        raise ImageSaveError(f"Could not save image: {path}")


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


def canny(
    image: np.ndarray,
    *,
    low_threshold: int = 100,
    high_threshold: int = 200,
    aperture_size: int = 3,
) -> np.ndarray:
    """Detect edges in a grayscale image with the Canny algorithm.

    Canny finds locations where the brightness changes sharply. It computes the
    image gradient, keeps only local maxima (thin edges), and uses two
    thresholds for hysteresis: strong edges above ``high_threshold`` are kept,
    and weak edges above ``low_threshold`` are kept only when they connect to a
    strong edge.

    Args:
        image: Grayscale image of shape ``(height, width)`` and dtype ``uint8``.
        low_threshold: Lower hysteresis threshold. It must be an integer between
            0 and 255. Lower values keep more weak edges.
        high_threshold: Upper hysteresis threshold. It must be an integer between
            0 and 255 and greater than or equal to ``low_threshold``. Higher
            values keep fewer strong edges.
        aperture_size: Size of the Sobel kernel used for the gradient. OpenCV
            only accepts 3, 5, or 7.

    Returns:
        A ``uint8`` array with the same shape, where edge pixels are 255 and the
        rest are 0.

    Raises:
        ValueError: If ``image`` is not a single-channel grayscale image, if a
            threshold is not an integer between 0 and 255, if ``low_threshold``
            is greater than ``high_threshold``, or if ``aperture_size`` is not
            3, 5, or 7.
    """
    _require_grayscale(image)
    for name, value in (
        ("low_threshold", low_threshold),
        ("high_threshold", high_threshold),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer; got {value!r}.")
        if not 0 <= value <= 255:
            raise ValueError(f"{name} must be between 0 and 255; got {value}.")
    if low_threshold > high_threshold:
        raise ValueError(
            "low_threshold must be less than or equal to high_threshold; "
            f"got {low_threshold} > {high_threshold}."
        )
    if not isinstance(aperture_size, int) or isinstance(aperture_size, bool):
        raise ValueError(f"aperture_size must be an integer; got {aperture_size!r}.")
    if aperture_size not in (3, 5, 7):
        raise ValueError(f"aperture_size must be 3, 5, or 7; got {aperture_size}.")

    # OpenCV names the hysteresis thresholds threshold1 (low) and threshold2
    # (high), so the public names are mapped to those keyword arguments here.
    return cv2.Canny(
        image,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
    )


def adaptive_threshold(
    image: np.ndarray,
    *,
    block_size: int = 11,
    constant: float = 2.0,
    use_gaussian: bool = False,
) -> np.ndarray:
    """Threshold a grayscale image using a threshold computed per pixel.

    Unlike :func:`binary_threshold`, which uses one global threshold for the
    whole image, this processor compares each pixel with the mean of its local
    neighborhood minus ``constant``. That makes it robust to uneven
    illumination, where a single global threshold fails.

    Args:
        image: Grayscale image of shape ``(height, width)`` and dtype ``uint8``.
        block_size: Side length of the square neighborhood used to compute the
            local threshold. It must be an odd integer greater than or equal to
            3. Larger values consider a wider area around each pixel.
        constant: Value subtracted from the local mean before comparing. It can
            be any real number, including 0 and negative values. Larger positive
            values lower the local threshold and therefore tend to classify more
            pixels as foreground.
        use_gaussian: When ``True``, the local mean is weighted by a Gaussian
            window instead of a flat average.

    Returns:
        A ``uint8`` array with the same shape, containing only 0 or 255, where
        255 marks pixels above their local threshold.

    Raises:
        ValueError: If ``image`` is not a single-channel grayscale image, if
            ``block_size`` is not an odd integer of at least 3, if ``constant``
            is not a number, or if ``use_gaussian`` is not a boolean.
    """
    _require_grayscale(image)
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError(f"block_size must be an integer; got {block_size!r}.")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(f"block_size must be odd and at least 3; got {block_size}.")
    if not isinstance(constant, (int, float)) or isinstance(constant, bool):
        raise ValueError(f"constant must be a number; got {constant!r}.")
    if not isinstance(use_gaussian, bool):
        raise ValueError(f"use_gaussian must be a boolean; got {use_gaussian!r}.")

    method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if use_gaussian else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    # The maximum value is fixed at 255 and the type at THRESH_BINARY, matching
    # binary_threshold. OpenCV calls the subtracted constant C, so the clearer
    # public name is mapped to that positional argument here.
    return cv2.adaptiveThreshold(
        image,
        255,
        method,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )


def morphology(
    image: np.ndarray,
    *,
    kernel_size: int = 5,
    use_closing: bool = False,
) -> np.ndarray:
    """Apply a morphological opening or closing to a grayscale image.

    Morphology slides a structuring element over the image. Opening erodes and
    then dilates, which removes small bright spots and thin protrusions from the
    foreground. Closing dilates and then erodes, which fills small dark holes
    and gaps in the foreground. Both keep the overall size of larger regions.

    Args:
        image: Grayscale image of shape ``(height, width)`` and dtype ``uint8``.
        kernel_size: Side length of the square structuring element. It must be
            an odd integer greater than or equal to 3. Larger values remove
            bigger spots or fill bigger holes.
        use_closing: When ``True``, apply a closing instead of an opening.

    Returns:
        A ``uint8`` array with the same shape as the input.

    Raises:
        ValueError: If ``image`` is not a single-channel grayscale image, if
            ``kernel_size`` is not an odd integer of at least 3, or if
            ``use_closing`` is not a boolean.
    """
    _require_grayscale(image)
    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise ValueError(f"kernel_size must be an integer; got {kernel_size!r}.")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and at least 3; got {kernel_size}.")
    if not isinstance(use_closing, bool):
        raise ValueError(f"use_closing must be a boolean; got {use_closing!r}.")

    # The shape and iterations are fixed for now: an elliptical element is a
    # good default, and a single pass keeps the effect easy to reason about.
    operation = cv2.MORPH_CLOSE if use_closing else cv2.MORPH_OPEN
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, operation, kernel, iterations=1)
