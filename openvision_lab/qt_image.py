import numpy as np
from PySide6.QtGui import QImage, QPixmap


def array_to_qpixmap(image: np.ndarray) -> QPixmap:
    array = np.ascontiguousarray(image)
    height, width = array.shape[:2]
    if array.ndim == 2:
        qimage = QImage(
            array.data, width, height, array.strides[0], QImage.Format_Grayscale8
        )
    elif array.ndim == 3 and array.shape[2] == 3:
        qimage = QImage(
            array.data, width, height, array.strides[0], QImage.Format_BGR888
        )
    else:
        raise ValueError(f"Unsupported image shape for display: {array.shape}")
    return QPixmap.fromImage(qimage.copy())
