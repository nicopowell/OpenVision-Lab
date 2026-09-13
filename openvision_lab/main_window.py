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

# Qt file dialog filter syntax: a description followed by space-separated glob
# patterns in parentheses.
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenVision Lab")
        self.resize(900, 600)

        # Keep the labels as attributes so open_image() can replace their
        # pixmaps later, when the user picks a file.
        self.original_label = self._create_image_label()
        self.grayscale_label = self._create_image_label()

        layout = QHBoxLayout()
        layout.addWidget(self._create_panel("Original", self.original_label))
        layout.addWidget(self._create_panel("Grayscale", self.grayscale_label))

        # A QMainWindow shows one central widget, so the layout lives inside a
        # plain QWidget that is set as that central widget.
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # The action is parented to the window so Qt keeps it alive as long as
        # the menu exists.
        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self.open_image)
        self.menuBar().addMenu("File").addAction(open_action)

    def _create_image_label(self) -> QLabel:
        label = QLabel()
        # Center the pixmap inside the label and keep a minimum panel size.
        # Contents stay unscaled so the image is not stretched to fit.
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setScaledContents(False)
        return label

    def _create_panel(self, title: str, label: QLabel) -> QGroupBox:
        panel = QGroupBox(title)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(label)
        return panel

    def open_image(self) -> None:
        """Ask the user for an image file and update both panels.

        Does nothing when the dialog is cancelled. Load and conversion errors
        are shown in a message box instead of being raised.
        """
        # getOpenFileName returns (path, selected_filter); the path is empty
        # when the user cancels.
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
