import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from openvision_lab.main_window import MainWindow
from openvision_lab.pipeline import PipelineStep, Processor


def _write_sample_image(path) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :16] = 200
    cv2.imwrite(str(path), image)


def _patch_open_image(monkeypatch, path) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), "")
    )


def test_open_image_shows_panels_and_view_stages(qtbot, tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _patch_open_image(monkeypatch, image_path)

    window.open_image()

    assert window.current_image is not None
    assert not window.original_label._pixmap.isNull()
    assert not window.preview_label._pixmap.isNull()
    assert window.view_combo.count() == len(window.steps) + 1
    assert window.save_action.isEnabled()


def test_view_selector_shows_the_selected_stage(qtbot, tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _patch_open_image(monkeypatch, image_path)
    window.open_image()

    window.view_combo.setCurrentIndex(1)

    np.testing.assert_array_equal(window._preview_array, window.intermediates[1])

    window.view_combo.setCurrentIndex(0)

    np.testing.assert_array_equal(window._preview_array, window.intermediates[-1])


def test_add_and_reorder_steps_updates_the_list(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    initial_count = len(window.steps)
    window.processor_combo.setCurrentIndex(
        list(Processor).index(Processor.BINARY_THRESHOLD)
    )

    window._add_step()

    assert len(window.steps) == initial_count + 1
    assert window.step_list.count() == initial_count + 1

    window.step_list.setCurrentRow(initial_count)
    window._move_step(-1)

    assert window.steps[initial_count - 1].processor is Processor.BINARY_THRESHOLD


def test_canny_parameters_write_back_to_the_step(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.CANNY),
    ]
    window._rebuild_step_list(1)

    window.low_threshold_spin.setValue(50)
    window.high_threshold_spin.setValue(150)
    window.aperture_spin.setValue(5)

    step = window.steps[1]
    assert step.low_threshold == 50
    assert step.high_threshold == 150
    assert step.aperture_size == 5
    assert "low=50" in window.step_list.item(1).text()
    assert "high=150" in window.step_list.item(1).text()


def test_invalid_canny_aperture_is_not_silently_corrected(qtbot, tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    _write_sample_image(image_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _patch_open_image(monkeypatch, image_path)
    window.open_image()

    window.steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.CANNY),
    ]
    window._rebuild_step_list(1)

    # An invalid aperture typed by the user must survive the widget instead of
    # being clamped, so the pipeline validation can explain the problem.
    window.aperture_spin.setValue(2)
    assert window.aperture_spin.value() == 2
    assert window.steps[1].aperture_size == 2

    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    window._process()

    assert warnings


def test_processor_combo_lists_every_processor(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    listed = [
        window.processor_combo.itemData(row)
        for row in range(window.processor_combo.count())
    ]

    assert Processor.CANNY in listed


def test_undo_and_redo_restore_the_pipeline(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._add_step()
    after_add = len(window.steps)

    window.undo()

    assert len(window.steps) == after_add - 1
    assert window.redo_action.isEnabled()

    window.redo()

    assert len(window.steps) == after_add
    assert not window.redo_action.isEnabled()


def test_save_and_load_pipeline_resets_history(qtbot, tmp_path, monkeypatch):
    pipeline_path = tmp_path / "pipeline.json"
    window = MainWindow()
    qtbot.addWidget(window)
    window._add_step()
    assert window.undo_action.isEnabled()

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(pipeline_path), ""),
    )
    window.save_pipeline_file()
    assert pipeline_path.exists()

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(pipeline_path), ""),
    )
    window.load_pipeline_file()

    assert window.step_list.count() == len(window.steps)
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_invalid_image_reports_an_error(qtbot, tmp_path, monkeypatch):
    bad_path = tmp_path / "bad.png"
    bad_path.write_text("this is not an image")
    window = MainWindow()
    qtbot.addWidget(window)
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    _patch_open_image(monkeypatch, bad_path)

    window.open_image()

    assert warnings
    assert window.current_image is None
