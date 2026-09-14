import numpy as np

from openvision_lab.image_ops import (
    binary_threshold,
    load_image,
    save_image,
    to_grayscale,
)
from openvision_lab.pipeline import (
    PipelineStep,
    Processor,
    default_pipeline,
    run_pipeline,
    run_pipeline_with_intermediates,
)


def test_file_to_file_flow_with_default_pipeline(tmp_path):
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    source[:, :8] = 220
    source_path = tmp_path / "source.png"
    save_image(str(source_path), source)

    loaded = load_image(str(source_path))
    np.testing.assert_array_equal(loaded, source)

    results = run_pipeline_with_intermediates(loaded, default_pipeline())
    final = results[-1]
    assert final.shape == (16, 16)
    assert set(np.unique(final)).issubset({0, 255})

    result_path = tmp_path / "result.png"
    save_image(str(result_path), final)
    reloaded = load_image(str(result_path))
    np.testing.assert_array_equal(reloaded[:, :, 0], final)
    np.testing.assert_array_equal(reloaded[:, :, 1], final)
    np.testing.assert_array_equal(reloaded[:, :, 2], final)


def test_file_to_file_flow_with_custom_pipeline(tmp_path, bgr_image):
    source_path = tmp_path / "source.png"
    save_image(str(source_path), bgr_image)

    loaded = load_image(str(source_path))
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=150),
    ]

    expected = binary_threshold(to_grayscale(loaded), threshold=150)
    np.testing.assert_array_equal(run_pipeline(loaded, steps), expected)


def test_processing_does_not_modify_the_loaded_image(tmp_path, bgr_image):
    source_path = tmp_path / "source.png"
    save_image(str(source_path), bgr_image)

    loaded = load_image(str(source_path))
    loaded_copy = loaded.copy()

    run_pipeline(loaded, default_pipeline())

    np.testing.assert_array_equal(loaded, loaded_copy)
    np.testing.assert_array_equal(load_image(str(source_path)), bgr_image)
