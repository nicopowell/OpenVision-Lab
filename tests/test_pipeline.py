import numpy as np
import pytest

from openvision_lab.image_ops import (
    adaptive_threshold,
    binary_threshold,
    canny,
    gaussian_blur,
    morphology,
    to_grayscale,
)
from openvision_lab.pipeline import (
    PipelineStep,
    Processor,
    default_pipeline,
    run_pipeline,
    run_pipeline_with_intermediates,
)


def test_empty_pipeline_returns_image_unchanged(bgr_image):
    np.testing.assert_array_equal(run_pipeline(bgr_image, []), bgr_image)


def test_run_pipeline_matches_manual_composition(bgr_image):
    expected = binary_threshold(
        gaussian_blur(to_grayscale(bgr_image), kernel_size=5), threshold=127
    )

    np.testing.assert_array_equal(run_pipeline(bgr_image, default_pipeline()), expected)


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


def test_removing_a_step_matches_composition_without_it(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=100),
    ]

    expected = binary_threshold(to_grayscale(bgr_image), threshold=100)

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


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


def test_incompatible_order_raises_value_error(bgr_image):
    steps = [
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=127),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError):
        run_pipeline(bgr_image, steps)


def test_canny_step_runs_after_grayscale(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.CANNY),
    ]

    expected = canny(to_grayscale(bgr_image))

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_canny_step_uses_its_own_parameters(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(
            Processor.CANNY,
            low_threshold=50,
            high_threshold=150,
            aperture_size=5,
        ),
    ]

    expected = canny(
        to_grayscale(bgr_image), low_threshold=50, high_threshold=150, aperture_size=5
    )

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_canny_before_grayscale_raises_value_error(bgr_image):
    steps = [
        PipelineStep(Processor.CANNY),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError):
        run_pipeline(bgr_image, steps)


def test_adaptive_threshold_step_runs_after_grayscale(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.ADAPTIVE_THRESHOLD),
    ]

    expected = adaptive_threshold(to_grayscale(bgr_image))

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_adaptive_threshold_step_uses_its_own_parameters(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(
            Processor.ADAPTIVE_THRESHOLD,
            block_size=3,
            constant=1.0,
            use_gaussian=True,
        ),
    ]

    expected = adaptive_threshold(
        to_grayscale(bgr_image), block_size=3, constant=1.0, use_gaussian=True
    )

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_adaptive_threshold_before_grayscale_raises_value_error(bgr_image):
    steps = [
        PipelineStep(Processor.ADAPTIVE_THRESHOLD),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError):
        run_pipeline(bgr_image, steps)


def test_error_message_names_the_failing_processor(bgr_image):
    steps = [
        PipelineStep(Processor.ADAPTIVE_THRESHOLD),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError) as excinfo:
        run_pipeline(bgr_image, steps)

    message = str(excinfo.value)
    assert "Processor 'Adaptive threshold' failed:" in message
    assert "grayscale image with shape" in message


def test_morphology_step_runs_after_grayscale(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.MORPHOLOGY),
    ]

    expected = morphology(to_grayscale(bgr_image), kernel_size=5)

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_morphology_step_uses_its_own_parameters(bgr_image):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.MORPHOLOGY, kernel_size=7, use_closing=True),
    ]

    expected = morphology(to_grayscale(bgr_image), kernel_size=7, use_closing=True)

    np.testing.assert_array_equal(run_pipeline(bgr_image, steps), expected)


def test_morphology_before_grayscale_raises_value_error(bgr_image):
    steps = [
        PipelineStep(Processor.MORPHOLOGY),
        PipelineStep(Processor.GRAYSCALE),
    ]

    with pytest.raises(ValueError) as excinfo:
        run_pipeline(bgr_image, steps)

    assert "Processor 'Morphology' failed:" in str(excinfo.value)


def test_default_pipeline_returns_independent_lists():
    first = default_pipeline()
    second = default_pipeline()

    first[1].kernel_size = 3

    assert first is not second
    assert second[1].kernel_size == 5


def test_intermediates_include_input_and_final_result(bgr_image):
    steps = default_pipeline()

    results = run_pipeline_with_intermediates(bgr_image, steps)

    assert len(results) == len(steps) + 1
    assert results[0] is bgr_image
    np.testing.assert_array_equal(results[-1], run_pipeline(bgr_image, steps))


def test_intermediates_match_prefix_composition(bgr_image):
    steps = default_pipeline()

    results = run_pipeline_with_intermediates(bgr_image, steps)

    gray = to_grayscale(bgr_image)
    blurred = gaussian_blur(gray, kernel_size=5, sigma=0.0)
    expected = binary_threshold(blurred, threshold=127)
    np.testing.assert_array_equal(results[1], gray)
    np.testing.assert_array_equal(results[2], blurred)
    np.testing.assert_array_equal(results[3], expected)


def test_intermediates_empty_pipeline_returns_only_input(bgr_image):
    results = run_pipeline_with_intermediates(bgr_image, [])

    assert len(results) == 1
    assert results[0] is bgr_image


def test_intermediates_do_not_modify_original(bgr_image):
    original_copy = bgr_image.copy()

    run_pipeline_with_intermediates(bgr_image, default_pipeline())

    np.testing.assert_array_equal(bgr_image, original_copy)
