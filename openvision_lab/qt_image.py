"""Adapters that turn OpenCV/NumPy images into Qt objects for display.

Kept separate from ``image_ops`` so the processing code does not depend on Qt.
"""

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def array_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a NumPy image array into a ``QPixmap`` ready to be shown.

    Args:
        image: Either a grayscale ``uint8`` array of shape ``(height, width)``
            or a BGR ``uint8`` array of shape ``(height, width, 3)``.

    Returns:
        A ``QPixmap`` that owns its own pixel data.

    Raises:
        ValueError: If ``image`` has an unsupported shape.
    """
    # QImage wraps the array's memory instead of copying it, so the buffer has
    # to be contiguous. ascontiguousarray guarantees the layout assumed by the
    # bytesPerLine argument (array.strides[0], the bytes in one image row).
    array = np.ascontiguousarray(image)
    height, width = array.shape[:2]
    if array.ndim == 2:
        qimage = QImage(
            array.data, width, height, array.strides[0], QImage.Format_Grayscale8
        )
    elif array.ndim == 3 and array.shape[2] == 3:
        # Format_BGR888 tells Qt the bytes are ordered B, G, R, which matches
        # OpenCV's default. Naming the right format avoids converting BGR->RGB.
        qimage = QImage(
            array.data, width, height, array.strides[0], QImage.Format_BGR888
        )
    else:
        raise ValueError(f"Unsupported image shape for display: {array.shape}")
    # QImage does not own the buffer, so copy() makes the pixmap independent of
    # the NumPy array's lifetime before the array can be garbage collected.
    return QPixmap.fromImage(qimage.copy())
