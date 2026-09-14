import numpy as np
import pytest

from openvision_lab.image_ops import binary_threshold, gaussian_blur, to_grayscale
from openvision_lab.pipeline import (
    PipelineStep,
    Processor,
    default_pipeline,
    run_pipeline,
    run_pipeline_with_intermediates,
)


def _sample_image() -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3, 3] = 255
    return image


def test_empty_pipeline_returns_image_unchanged():
    image = _sample_image()

    result = run_pipeline(image, [])

    assert np.array_equal(result, image)


def test_run_pipeline_matches_manual_composition():
    image = _sample_image()

    expected = binary_threshold(
        gaussian_blur(to_grayscale(image), kernel_size=5), threshold=127
    )

    assert np.array_equal(run_pipeline(image, default_pipeline()), expected)


def test_step_order_changes_result():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[:, :4] = 255

    blur_then_threshold = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.GAUSSIAN_BLUR, kernel_size=9),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=127),
    ]
    threshold_then_blur = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=127),
        PipelineStep(Processor.GAUSSIAN_BLUR, kernel_size=9),
    ]

    assert not np.array_equal(
        run_pipeline(image, blur_then_threshold),
        run_pipeline(image, threshold_then_blur),
    )


def test_removing_a_step_matches_composition_without_it():
    image = _sample_image()
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=100),
    ]

    expected = binary_threshold(to_grayscale(image), threshold=100)

    assert np.array_equal(run_pipeline(image, steps), expected)


def test_step_uses_its_own_parameters():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = 100
    low_threshold = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=50),
    ]
    high_threshold = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=150),
    ]

    assert not np.array_equal(
        run_pipeline(image, low_threshold),
        run_pipeline(image, high_threshold),
    )


def test_incompatible_order_raises_value_error():
    image = _sample_image()
    steps = [
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=127),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError):
        run_pipeline(image, steps)


def test_default_pipeline_returns_independent_lists():
    first = default_pipeline()
    second = default_pipeline()

    first[1].kernel_size = 3

    assert first is not second
    assert second[1].kernel_size == 5


def test_intermediates_include_input_and_final_result():
    image = _sample_image()
    steps = default_pipeline()

    results = run_pipeline_with_intermediates(image, steps)

    assert len(results) == len(steps) + 1
    assert results[0] is image
    assert np.array_equal(results[-1], run_pipeline(image, steps))


def test_intermediates_match_prefix_composition():
    image = _sample_image()
    steps = default_pipeline()

    results = run_pipeline_with_intermediates(image, steps)

    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, kernel_size=5, sigma=0.0)
    expected = binary_threshold(blurred, threshold=127)
    assert np.array_equal(results[1], gray)
    assert np.array_equal(results[2], blurred)
    assert np.array_equal(results[3], expected)


def test_intermediates_empty_pipeline_returns_only_input():
    image = _sample_image()

    results = run_pipeline_with_intermediates(image, [])

    assert len(results) == 1
    assert results[0] is image


def test_intermediates_do_not_modify_original():
    image = _sample_image()
    original_copy = image.copy()

    run_pipeline_with_intermediates(image, default_pipeline())

    assert np.array_equal(image, original_copy)
