import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from openvision_lab.history import PipelineHistory
from openvision_lab.image_ops import (
    ImageLoadError,
    ImageSaveError,
    load_image,
    save_image,
)
from openvision_lab.pipeline import (
    PipelineStep,
    Processor,
    default_pipeline,
    run_pipeline_with_intermediates,
)
from openvision_lab.pipeline_io import (
    PipelineFileError,
    load_pipeline,
    save_pipeline,
)
from openvision_lab.qt_image import array_to_qpixmap

# Qt file dialog filter syntax: a description followed by space-separated glob
# patterns in parentheses.
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
SAVE_FILTER = (
    "PNG image (*.png);;"
    "JPEG image (*.jpg *.jpeg);;"
    "BMP image (*.bmp);;"
    "TIFF image (*.tif *.tiff)"
)
PIPELINE_FILTER = "Pipeline files (*.json);;All files (*)"


class ImageLabel(QLabel):
    """Label that shows an image scaled to fit without distorting it.

    The image is scaled inside ``paintEvent`` using the current widget size, so
    storing a new image never changes the label's size hint and cannot feed
    back into the window layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)

    def set_array(self, image: np.ndarray) -> None:
        """Store the image to display and request a repaint."""
        self._pixmap = array_to_qpixmap(image)
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(self)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenVision Lab")

        # The loaded source image is kept so the pipeline can be re-run without
        # reloading the file.
        self.current_image: np.ndarray | None = None

        # The pipeline is the application state the UI edits. It starts with
        # the default steps but is a plain list that can be changed.
        self.steps = default_pipeline()

        # Undo/redo history of the pipeline (structure and parameters only).
        self.history = PipelineHistory(self.steps)

        # Continuous parameter edits are coalesced into a single history entry:
        # the timer restarts on every change and commits once editing pauses.
        self._parameter_commit_pending = False
        self._parameter_timer = QTimer(self)
        self._parameter_timer.setSingleShot(True)
        self._parameter_timer.setInterval(400)
        self._parameter_timer.timeout.connect(self._commit_parameters)

        # results[0] is the original image and results[k] is the output after
        # the first k steps, so the last element is the final result.
        self.intermediates: list[np.ndarray] = []

        # Image currently shown in the preview panel (a selected stage).
        self._preview_array: np.ndarray | None = None

        # Guard set while the parameter widgets are being populated, so loading
        # a step does not write its values back into the step.
        self._loading_step = False

        # Keep the labels as attributes so they can be refreshed later.
        self.original_label = ImageLabel()
        self.preview_label = ImageLabel()

        self.view_combo = QComboBox()
        # Widen the combo to fit the longest stage name instead of truncating.
        self.view_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View"))
        view_layout.addWidget(self.view_combo)
        view_layout.addStretch(1)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self._create_panel("Original", self.original_label))
        images_layout.addWidget(self._create_panel("Preview", self.preview_label))

        images_column = QVBoxLayout()
        images_column.addLayout(view_layout)
        images_column.addLayout(images_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self._create_pipeline_group())
        central_layout.addLayout(images_column, stretch=1)

        # A QMainWindow shows one central widget, so the layout lives inside a
        # plain QWidget that is set as that central widget.
        container = QWidget()
        container.setLayout(central_layout)
        self.setCentralWidget(container)

        # The actions are parented to the window so Qt keeps them alive as long
        # as the menu exists.
        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self.open_image)

        self.save_action = QAction("Save Result As...", self)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_result)

        load_pipeline_action = QAction("Load Pipeline...", self)
        load_pipeline_action.triggered.connect(self.load_pipeline_file)

        save_pipeline_action = QAction("Save Pipeline...", self)
        save_pipeline_action.triggered.connect(self.save_pipeline_file)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(load_pipeline_action)
        file_menu.addAction(save_pipeline_action)

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        self.resize(1000, 600)

    def _create_panel(self, title: str, label: QLabel) -> QGroupBox:
        panel = QGroupBox(title)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(label)
        return panel

    def _create_pipeline_group(self) -> QGroupBox:
        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self._on_step_selected)

        self.processor_combo = QComboBox()
        for processor in Processor:
            # Store the enum member as item data and show its display name.
            self.processor_combo.addItem(processor.value, processor)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_step)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_step)

        self.up_button = QPushButton("Move Up")
        self.up_button.clicked.connect(lambda: self._move_step(-1))

        self.down_button = QPushButton("Move Down")
        self.down_button.clicked.connect(lambda: self._move_step(1))

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.processor_combo)
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(self.remove_button)
        buttons_layout.addWidget(self.up_button)
        buttons_layout.addWidget(self.down_button)

        # Only odd kernel sizes are valid for Gaussian blur, so the spin box
        # steps by 2. gaussian_blur() still validates the value defensively.
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 31)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.valueChanged.connect(self._on_parameters_changed)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 10.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.valueChanged.connect(self._on_parameters_changed)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.valueChanged.connect(self._on_parameters_changed)

        # Canny uses the same 0-255 range for both hysteresis thresholds.
        self.low_threshold_spin = QSpinBox()
        self.low_threshold_spin.setRange(0, 255)
        self.low_threshold_spin.valueChanged.connect(self._on_parameters_changed)

        self.high_threshold_spin = QSpinBox()
        self.high_threshold_spin.setRange(0, 255)
        self.high_threshold_spin.valueChanged.connect(self._on_parameters_changed)

        # Only 3, 5, or 7 are valid Sobel aperture sizes, but the range is left
        # wide (like the blur kernel) so a value typed by the user is not
        # silently clamped. canny() rejects it and _process shows the reason.
        self.aperture_spin = QSpinBox()
        self.aperture_spin.setRange(1, 31)
        self.aperture_spin.setSingleStep(2)
        self.aperture_spin.valueChanged.connect(self._on_parameters_changed)

        # The same wide-range rule applies to the adaptive threshold block size:
        # only odd values of at least 3 are valid, but an invalid typed value
        # should reach the processor instead of being clamped.
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(1, 31)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.valueChanged.connect(self._on_parameters_changed)

        # The -50..50 range is only a UI convenience; the processor accepts any
        # real constant, including negative values.
        self.constant_spin = QDoubleSpinBox()
        self.constant_spin.setRange(-50.0, 50.0)
        self.constant_spin.setSingleStep(0.5)
        self.constant_spin.setDecimals(2)
        self.constant_spin.valueChanged.connect(self._on_parameters_changed)

        self.method_combo = QComboBox()
        # Store the boolean as item data and show the method name.
        self.method_combo.addItem("Mean", False)
        self.method_combo.addItem("Gaussian", True)
        self.method_combo.currentIndexChanged.connect(self._on_parameters_changed)

        self.parameters_form = QFormLayout()
        self.parameters_form.addRow("Blur kernel size", self.kernel_spin)
        self.parameters_form.addRow("Blur sigma", self.sigma_spin)
        self.parameters_form.addRow("Threshold", self.threshold_spin)
        self.parameters_form.addRow("Adaptive block size", self.block_size_spin)
        self.parameters_form.addRow("Adaptive constant", self.constant_spin)
        self.parameters_form.addRow("Adaptive method", self.method_combo)
        self.parameters_form.addRow("Canny low threshold", self.low_threshold_spin)
        self.parameters_form.addRow("Canny high threshold", self.high_threshold_spin)
        self.parameters_form.addRow("Canny aperture", self.aperture_spin)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._process)

        group = QGroupBox("Pipeline")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(self.step_list)
        group_layout.addLayout(buttons_layout)
        group_layout.addLayout(self.parameters_form)
        group_layout.addWidget(self.apply_button)

        self._rebuild_step_list(0)
        return group

    def _describe_step(self, step: PipelineStep) -> str:
        """Return the short text shown for a step in the list."""
        if step.processor is Processor.GAUSSIAN_BLUR:
            return f"{step.processor.value} (k={step.kernel_size}, sigma={step.sigma})"
        if step.processor is Processor.BINARY_THRESHOLD:
            return f"{step.processor.value} (t={step.threshold})"
        if step.processor is Processor.ADAPTIVE_THRESHOLD:
            method = "gaussian" if step.use_gaussian else "mean"
            return (
                f"{step.processor.value} "
                f"(block={step.block_size}, C={step.constant}, {method})"
            )
        if step.processor is Processor.CANNY:
            return (
                f"{step.processor.value} "
                f"(low={step.low_threshold}, high={step.high_threshold})"
            )
        return step.processor.value

    def _selected_step(self) -> PipelineStep | None:
        row = self.step_list.currentRow()
        if 0 <= row < len(self.steps):
            return self.steps[row]
        return None

    def _rebuild_step_list(self, select_row: int) -> None:
        """Rebuild the list widget from ``self.steps`` and select a row."""
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for step in self.steps:
            self.step_list.addItem(self._describe_step(step))
        self.step_list.blockSignals(False)

        if self.steps:
            self.step_list.setCurrentRow(select_row)
        else:
            # No steps: clear the editor and refresh the buttons explicitly,
            # because no selection change signal is emitted.
            self._load_selected_step()
            self._update_buttons()

    def _on_step_selected(self, _row: int) -> None:
        self._load_selected_step()
        self._update_buttons()

    def _update_buttons(self) -> None:
        row = self.step_list.currentRow()
        has_selection = 0 <= row < len(self.steps)
        self.remove_button.setEnabled(has_selection)
        self.up_button.setEnabled(has_selection and row > 0)
        self.down_button.setEnabled(has_selection and row < len(self.steps) - 1)

    def _load_selected_step(self) -> None:
        """Show the selected step's parameters in the editor widgets."""
        step = self._selected_step()
        self._loading_step = True
        try:
            is_blur = step is not None and step.processor is Processor.GAUSSIAN_BLUR
            is_threshold = (
                step is not None and step.processor is Processor.BINARY_THRESHOLD
            )
            is_canny = step is not None and step.processor is Processor.CANNY
            is_adaptive = (
                step is not None and step.processor is Processor.ADAPTIVE_THRESHOLD
            )
            # Rows that do not apply to the selected processor are hidden.
            self.parameters_form.setRowVisible(self.kernel_spin, is_blur)
            self.parameters_form.setRowVisible(self.sigma_spin, is_blur)
            self.parameters_form.setRowVisible(self.threshold_spin, is_threshold)
            self.parameters_form.setRowVisible(self.block_size_spin, is_adaptive)
            self.parameters_form.setRowVisible(self.constant_spin, is_adaptive)
            self.parameters_form.setRowVisible(self.method_combo, is_adaptive)
            self.parameters_form.setRowVisible(self.low_threshold_spin, is_canny)
            self.parameters_form.setRowVisible(self.high_threshold_spin, is_canny)
            self.parameters_form.setRowVisible(self.aperture_spin, is_canny)

            if step is None:
                return
            self.kernel_spin.setValue(step.kernel_size)
            self.sigma_spin.setValue(step.sigma)
            self.threshold_spin.setValue(step.threshold)
            self.block_size_spin.setValue(step.block_size)
            self.constant_spin.setValue(step.constant)
            self.method_combo.setCurrentIndex(1 if step.use_gaussian else 0)
            self.low_threshold_spin.setValue(step.low_threshold)
            self.high_threshold_spin.setValue(step.high_threshold)
            self.aperture_spin.setValue(step.aperture_size)
        finally:
            self._loading_step = False

    def _on_parameters_changed(self, _value: float) -> None:
        """Write edited parameters back into the selected step."""
        if self._loading_step:
            return
        step = self._selected_step()
        if step is None:
            return
        if step.processor is Processor.GAUSSIAN_BLUR:
            step.kernel_size = self.kernel_spin.value()
            step.sigma = self.sigma_spin.value()
        elif step.processor is Processor.BINARY_THRESHOLD:
            step.threshold = self.threshold_spin.value()
        elif step.processor is Processor.ADAPTIVE_THRESHOLD:
            step.block_size = self.block_size_spin.value()
            step.constant = self.constant_spin.value()
            step.use_gaussian = bool(self.method_combo.currentData())
        elif step.processor is Processor.CANNY:
            step.low_threshold = self.low_threshold_spin.value()
            step.high_threshold = self.high_threshold_spin.value()
            step.aperture_size = self.aperture_spin.value()

        row = self.step_list.currentRow()
        self.step_list.blockSignals(True)
        self.step_list.item(row).setText(self._describe_step(step))
        self.step_list.blockSignals(False)
        self._schedule_parameter_commit()

    def _add_step(self) -> None:
        self._flush_parameter_commit()
        processor = self.processor_combo.currentData()
        self.steps.append(PipelineStep(processor))
        self._rebuild_step_list(len(self.steps) - 1)
        self._commit_history()

    def _remove_step(self) -> None:
        row = self.step_list.currentRow()
        if not 0 <= row < len(self.steps):
            return
        self._flush_parameter_commit()
        del self.steps[row]
        self._rebuild_step_list(min(row, len(self.steps) - 1))
        self._commit_history()

    def _move_step(self, delta: int) -> None:
        row = self.step_list.currentRow()
        target = row + delta
        if not (0 <= row < len(self.steps) and 0 <= target < len(self.steps)):
            return
        self._flush_parameter_commit()
        self.steps[row], self.steps[target] = self.steps[target], self.steps[row]
        # Keep the moved step selected at its new position.
        self._rebuild_step_list(target)
        self._commit_history()

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(self.history.can_undo)
        self.redo_action.setEnabled(self.history.can_redo)

    def _commit_history(self) -> None:
        """Record the current pipeline state as a new history snapshot."""
        self._parameter_commit_pending = False
        self._parameter_timer.stop()
        self.history.record(self.steps)
        self._update_history_actions()

    def _schedule_parameter_commit(self) -> None:
        """Coalesce continuous parameter edits into one history entry."""
        self._parameter_commit_pending = True
        self._parameter_timer.start()

    def _flush_parameter_commit(self) -> None:
        """Record a pending parameter edit before another action happens."""
        if self._parameter_commit_pending:
            self._commit_history()

    def _commit_parameters(self) -> None:
        if self._parameter_commit_pending:
            self._commit_history()

    def _restore_steps(self, steps: list[PipelineStep]) -> None:
        """Replace the pipeline with a snapshot and refresh the UI."""
        previous_row = self.step_list.currentRow()
        self.steps = steps
        if self.steps:
            row = previous_row if 0 <= previous_row < len(self.steps) else 0
        else:
            row = 0
        self._rebuild_step_list(row)
        self._process()
        self._update_history_actions()

    def undo(self) -> None:
        """Restore the previous pipeline state."""
        self._flush_parameter_commit()
        steps = self.history.undo()
        if steps is not None:
            self._restore_steps(steps)

    def redo(self) -> None:
        """Restore the next pipeline state."""
        self._flush_parameter_commit()
        steps = self.history.redo()
        if steps is not None:
            self._restore_steps(steps)

    def _update_view_combo(self) -> None:
        """Fill the view combo with the available stages.

        Item 0 is the final result. Item ``k`` shows the output after the first
        ``k`` steps, which is exactly ``self.intermediates[k]``.
        """
        previous = self.view_combo.currentIndex()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("Final result")
        for step in self.steps:
            self.view_combo.addItem(f"After {self._describe_step(step)}")
        if 0 <= previous < self.view_combo.count():
            self.view_combo.setCurrentIndex(previous)
        else:
            self.view_combo.setCurrentIndex(0)
        self.view_combo.blockSignals(False)

    def _on_view_changed(self, index: int) -> None:
        if not self.intermediates:
            self._preview_array = None
        elif index == 0:
            self._preview_array = self.intermediates[-1]
        elif index < len(self.intermediates):
            self._preview_array = self.intermediates[index]
        else:
            self._preview_array = self.intermediates[-1]
        self._render_images()

    def _render_images(self) -> None:
        """Refresh the original and preview panels from the stored arrays."""
        if self.current_image is not None:
            self.original_label.set_array(self.current_image)
        if self._preview_array is not None:
            self.preview_label.set_array(self._preview_array)

    def open_image(self) -> None:
        """Ask the user for an image file and update both panels.

        Does nothing when the dialog is cancelled. Load errors are shown in a
        message box instead of being raised. The pipeline is applied with its
        current steps.
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
        self.apply_button.setEnabled(True)
        self.save_action.setEnabled(True)
        self._process()

    def _process(self) -> None:
        """Run the pipeline and refresh the preview and the stage selector."""
        if self.current_image is None:
            return
        try:
            self.intermediates = run_pipeline_with_intermediates(
                self.current_image, self.steps
            )
        except ValueError as error:
            QMessageBox.warning(self, "Processing", str(error))
            return
        self._update_view_combo()
        self._on_view_changed(self.view_combo.currentIndex())

    def save_result(self) -> None:
        """Save the final processed image to a file chosen by the user.

        The pipeline result is written at full resolution, not the scaled
        preview. Save errors are shown in a message box instead of being
        raised.
        """
        if not self.intermediates:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "result.png", SAVE_FILTER
        )
        if not path:
            return
        try:
            save_image(path, self.intermediates[-1])
        except ImageSaveError as error:
            QMessageBox.warning(self, "Save Result", str(error))

    def save_pipeline_file(self) -> None:
        """Save the current pipeline configuration to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "pipeline.json", PIPELINE_FILTER
        )
        if not path:
            return
        try:
            save_pipeline(path, self.steps)
        except PipelineFileError as error:
            QMessageBox.warning(self, "Save Pipeline", str(error))

    def load_pipeline_file(self) -> None:
        """Load a pipeline configuration and make it the new history baseline.

        Loading is not an undoable edit: the history is reset to the loaded
        pipeline. Errors are shown in a message box instead of being raised.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", PIPELINE_FILTER
        )
        if not path:
            return
        try:
            steps = load_pipeline(path)
        except PipelineFileError as error:
            QMessageBox.warning(self, "Load Pipeline", str(error))
            return

        # Discard any pending parameter edit before replacing the pipeline.
        self._parameter_timer.stop()
        self._parameter_commit_pending = False
        self.steps = steps
        self.history.reset(self.steps)
        self._rebuild_step_list(0)
        self._process()
        self._update_history_actions()
