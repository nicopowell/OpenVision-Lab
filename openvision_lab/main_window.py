from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from openvision_lab.image_ops import ImageLoadError, load_image, to_grayscale
from openvision_lab.qt_image import array_to_qpixmap

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenVision Lab")
        self.resize(900, 600)

        self.original_label = self._create_image_label()
        self.grayscale_label = self._create_image_label()

        layout = QHBoxLayout()
        layout.addWidget(self._create_panel("Original", self.original_label))
        layout.addWidget(self._create_panel("Grayscale", self.grayscale_label))

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self.open_image)
        self.menuBar().addMenu("File").addAction(open_action)

    def _create_image_label(self) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setScaledContents(True)
        return label

    def _create_panel(self, title: str, label: QLabel) -> QGroupBox:
        panel = QGroupBox(title)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(label)
        return panel

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", IMAGE_FILTER)
        if not path:
            return
        try:
            image = load_image(path)
            grayscale = to_grayscale(image)
        except (ImageLoadError, ValueError) as error:
            QMessageBox.warning(self, "Open Image", str(error))
            return
        self.original_label.setPixmap(array_to_qpixmap(image))
        self.grayscale_label.setPixmap(array_to_qpixmap(grayscale))
