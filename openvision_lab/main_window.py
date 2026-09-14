import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from openvision_lab.image_ops import ImageLoadError, load_image, run_pipeline
from openvision_lab.qt_image import array_to_qpixmap

# Qt file dialog filter syntax: a description followed by space-separated glob
# patterns in parentheses.
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenVision Lab")
        self.resize(900, 600)

        # The loaded source image is kept so the pipeline can be re-run with
        # new parameters without reloading the file.
        self.current_image: np.ndarray | None = None

        # Keep the labels as attributes so open_image() and _process() can
        # replace their pixmaps later.
        self.original_label = self._create_image_label()
        self.result_label = self._create_image_label()

        images_layout = QHBoxLayout()
        images_layout.addWidget(self._create_panel("Original", self.original_label))
        images_layout.addWidget(self._create_panel("Result", self.result_label))

        central_layout = QVBoxLayout()
        central_layout.addWidget(self._create_parameters_group())
        central_layout.addLayout(images_layout)

        # A QMainWindow shows one central widget, so the layout lives inside a
        # plain QWidget that is set as that central widget.
        container = QWidget()
        container.setLayout(central_layout)
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

    def _create_parameters_group(self) -> QGroupBox:
        # Only odd kernel sizes are valid for Gaussian blur, so the spin box
        # steps by 2. gaussian_blur() still validates the value defensively.
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 31)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(5)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(127)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._process)

        form_layout = QFormLayout()
        form_layout.addRow("Blur kernel size", self.kernel_spin)
        form_layout.addRow("Threshold", self.threshold_spin)

        group = QGroupBox("Parameters")
        group_layout = QVBoxLayout(group)
        group_layout.addLayout(form_layout)
        group_layout.addWidget(self.apply_button)
        return group

    def open_image(self) -> None:
        """Ask the user for an image file and update both panels.

        Does nothing when the dialog is cancelled. Load errors are shown in a
        message box instead of being raised. The parameters are applied with
        their current values.
        """
        # getOpenFileName returns (path, selected_filter); the path is empty
        # when the user cancels.
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", IMAGE_FILTER)
        if not path:
            return
        try:
            image = load_image(path)
        except ImageLoadError as error:
            QMessageBox.warning(self, "Open Image", str(error))
            return
        self.current_image = image
        self.original_label.setPixmap(array_to_qpixmap(image))
        self.apply_button.setEnabled(True)
        self._process()

    def _process(self) -> None:
        """Re-run the pipeline on the loaded image with the current parameters."""
        if self.current_image is None:
            return
        try:
            result = run_pipeline(
                self.current_image,
                kernel_size=self.kernel_spin.value(),
                threshold=self.threshold_spin.value(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Processing", str(error))
            return
        self.result_label.setPixmap(array_to_qpixmap(result))
