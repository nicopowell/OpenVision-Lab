import json

import numpy as np
import pytest

from openvision_lab.pipeline import (
    PipelineStep,
    Processor,
    default_pipeline,
    run_pipeline,
)
from openvision_lab.pipeline_io import (
    PipelineFileError,
    load_pipeline,
    save_pipeline,
)


def _write_json(tmp_path, data) -> str:
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_round_trip_preserves_steps(tmp_path):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.GAUSSIAN_BLUR, kernel_size=7, sigma=1.5),
        PipelineStep(Processor.BINARY_THRESHOLD, threshold=200),
    ]
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), steps)

    assert load_pipeline(str(path)) == steps


def test_round_trip_empty_pipeline(tmp_path):
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), [])

    assert load_pipeline(str(path)) == []


def test_round_trip_preserves_canny_step(tmp_path):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(
            Processor.CANNY,
            low_threshold=50,
            high_threshold=150,
            aperture_size=5,
        ),
    ]
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), steps)

    assert load_pipeline(str(path)) == steps


def test_canny_step_is_readable_json(tmp_path):
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), [PipelineStep(Processor.CANNY)])

    data = json.loads(path.read_text())
    assert data["steps"][0] == {
        "processor": "CANNY",
        "low_threshold": 100,
        "high_threshold": 200,
        "aperture_size": 3,
    }


def test_round_trip_preserves_adaptive_threshold_step(tmp_path):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(
            Processor.ADAPTIVE_THRESHOLD,
            block_size=3,
            constant=-5.0,
            use_gaussian=True,
        ),
    ]
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), steps)

    assert load_pipeline(str(path)) == steps


def test_adaptive_threshold_step_is_readable_json(tmp_path):
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), [PipelineStep(Processor.ADAPTIVE_THRESHOLD)])

    data = json.loads(path.read_text())
    assert data["steps"][0] == {
        "processor": "ADAPTIVE_THRESHOLD",
        "block_size": 11,
        "constant": 2.0,
        "use_gaussian": False,
    }


def test_load_adaptive_threshold_uses_defaults_for_missing_parameters(tmp_path):
    path = _write_json(
        tmp_path,
        {"version": 1, "steps": [{"processor": "ADAPTIVE_THRESHOLD"}]},
    )

    loaded = load_pipeline(path)

    assert loaded[0].block_size == 11
    assert loaded[0].constant == 2.0
    assert loaded[0].use_gaussian is False


@pytest.mark.parametrize("block_size", [1, 2, 4, 0, -3, True, 5.5, "7"])
def test_load_invalid_adaptive_block_size_raises(tmp_path, block_size):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "ADAPTIVE_THRESHOLD", "block_size": block_size}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("constant", ["2", True, None])
def test_load_invalid_adaptive_constant_raises(tmp_path, constant):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "ADAPTIVE_THRESHOLD", "constant": constant}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("use_gaussian", [0, 1, "yes", None])
def test_load_invalid_adaptive_method_raises(tmp_path, use_gaussian):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [
                {"processor": "ADAPTIVE_THRESHOLD", "use_gaussian": use_gaussian}
            ],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_round_trip_preserves_morphology_step(tmp_path):
    steps = [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.MORPHOLOGY, kernel_size=7, use_closing=True),
    ]
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), steps)

    assert load_pipeline(str(path)) == steps


def test_morphology_step_is_readable_json(tmp_path):
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), [PipelineStep(Processor.MORPHOLOGY)])

    data = json.loads(path.read_text())
    assert data["steps"][0] == {
        "processor": "MORPHOLOGY",
        "kernel_size": 5,
        "use_closing": False,
    }


def test_load_morphology_uses_defaults_for_missing_parameters(tmp_path):
    path = _write_json(
        tmp_path,
        {"version": 1, "steps": [{"processor": "MORPHOLOGY"}]},
    )

    loaded = load_pipeline(path)

    assert loaded[0].kernel_size == 5
    assert loaded[0].use_closing is False


@pytest.mark.parametrize("kernel_size", [1, 0, -3, 2, 4, True, 5.5, "5"])
def test_load_invalid_morphology_kernel_size_raises(tmp_path, kernel_size):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "MORPHOLOGY", "kernel_size": kernel_size}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("use_closing", [0, 1, "close", None])
def test_load_invalid_morphology_operation_raises(tmp_path, use_closing):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "MORPHOLOGY", "use_closing": use_closing}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_saved_file_is_readable_json(tmp_path):
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), default_pipeline())

    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert data["steps"][0] == {"processor": "GRAYSCALE"}
    assert data["steps"][1]["processor"] == "GAUSSIAN_BLUR"
    assert data["steps"][1]["kernel_size"] == 5
    assert data["steps"][1]["sigma"] == 0.0
    assert data["steps"][2] == {"processor": "BINARY_THRESHOLD", "threshold": 127}


def test_load_missing_file_raises():
    with pytest.raises(PipelineFileError):
        load_pipeline("this_file_does_not_exist.json")


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text("this is not json")

    with pytest.raises(PipelineFileError):
        load_pipeline(str(path))


def test_load_unsupported_version_raises(tmp_path):
    path = _write_json(tmp_path, {"version": 99, "steps": []})

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_steps_not_a_list_raises(tmp_path):
    path = _write_json(tmp_path, {"version": 1, "steps": {"processor": "GRAYSCALE"}})

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_unknown_processor_raises(tmp_path):
    path = _write_json(tmp_path, {"version": 1, "steps": [{"processor": "EDGE"}]})

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("kernel_size", [0, -1, 4, 5.5, "5"])
def test_load_invalid_kernel_size_raises(tmp_path, kernel_size):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [
                {"processor": "GAUSSIAN_BLUR", "kernel_size": kernel_size, "sigma": 0.0}
            ],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_negative_sigma_raises(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "GAUSSIAN_BLUR", "kernel_size": 5, "sigma": -1.0}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("threshold", [-1, 256, 1.5, "127"])
def test_load_invalid_threshold_raises(tmp_path, threshold):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "BINARY_THRESHOLD", "threshold": threshold}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_uses_defaults_for_missing_parameters(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [
                {"processor": "GAUSSIAN_BLUR"},
                {"processor": "BINARY_THRESHOLD"},
                {"processor": "CANNY"},
            ],
        },
    )

    loaded = load_pipeline(path)

    assert loaded[0].kernel_size == 5
    assert loaded[0].sigma == 0.0
    assert loaded[1].threshold == 127
    assert loaded[2].low_threshold == 100
    assert loaded[2].high_threshold == 200
    assert loaded[2].aperture_size == 3


@pytest.mark.parametrize("aperture_size", [1, 2, 4, 9, 5.5, "3"])
def test_load_invalid_canny_aperture_size_raises(tmp_path, aperture_size):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "CANNY", "aperture_size": aperture_size}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


@pytest.mark.parametrize("value", [-1, 256, 1.5, "100"])
def test_load_invalid_canny_threshold_raises(tmp_path, value):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [{"processor": "CANNY", "low_threshold": value}],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_canny_low_above_high_raises(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "version": 1,
            "steps": [
                {"processor": "CANNY", "low_threshold": 200, "high_threshold": 100}
            ],
        },
    )

    with pytest.raises(PipelineFileError):
        load_pipeline(path)


def test_load_ignores_unknown_keys(tmp_path):
    path = _write_json(
        tmp_path,
        {"version": 1, "steps": [{"processor": "GRAYSCALE", "extra": 123}]},
    )

    assert load_pipeline(path) == [PipelineStep(Processor.GRAYSCALE)]


def test_save_to_invalid_path_raises(tmp_path):
    with pytest.raises(PipelineFileError):
        save_pipeline(
            str(tmp_path / "missing_dir" / "pipeline.json"), default_pipeline()
        )


def test_saved_pipeline_runs_the_same(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 220
    steps = default_pipeline()
    path = tmp_path / "pipeline.json"

    save_pipeline(str(path), steps)
    loaded = load_pipeline(str(path))

    np.testing.assert_array_equal(
        run_pipeline(image, loaded), run_pipeline(image, steps)
    )
