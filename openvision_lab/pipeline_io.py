"""Save and load pipeline configurations as JSON.

The format is intentionally simple and readable: a version and a list of steps,
where each step names its processor and only the parameters that apply to it.
Missing parameters fall back to the ``PipelineStep`` defaults when loading.
"""

import json

from openvision_lab.pipeline import PipelineStep, Processor

SCHEMA_VERSION = 1


class PipelineFileError(Exception):
    """Raised when a pipeline file cannot be saved or loaded."""


def save_pipeline(path: str, steps: list[PipelineStep]) -> None:
    """Write a pipeline configuration to a JSON file.

    Args:
        path: Destination path for the JSON file.
        steps: Pipeline steps to save.

    Raises:
        PipelineFileError: If the file cannot be written.
    """
    data = {
        "version": SCHEMA_VERSION,
        "steps": [_step_to_dict(step) for step in steps],
    }
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    except OSError as error:
        raise PipelineFileError(f"Could not save pipeline: {path}") from error


def load_pipeline(path: str) -> list[PipelineStep]:
    """Load a pipeline configuration from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The loaded pipeline steps.

    Raises:
        PipelineFileError: If the file is missing or unreadable, is not valid
            JSON, or does not match the expected schema.
    """
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except OSError as error:
        raise PipelineFileError(f"Could not load pipeline: {path}") from error
    except json.JSONDecodeError as error:
        raise PipelineFileError(f"Invalid pipeline file: {path}") from error

    return _parse_pipeline(data)


def _step_to_dict(step: PipelineStep) -> dict:
    """Convert a step to its JSON representation with only relevant parameters."""
    data: dict = {"processor": step.processor.name}
    if step.processor is Processor.GAUSSIAN_BLUR:
        data["kernel_size"] = step.kernel_size
        data["sigma"] = step.sigma
    elif step.processor is Processor.BINARY_THRESHOLD:
        data["threshold"] = step.threshold
    elif step.processor is Processor.ADAPTIVE_THRESHOLD:
        data["block_size"] = step.block_size
        data["constant"] = step.constant
        data["use_gaussian"] = step.use_gaussian
    elif step.processor is Processor.CANNY:
        data["low_threshold"] = step.low_threshold
        data["high_threshold"] = step.high_threshold
        data["aperture_size"] = step.aperture_size
    return data


def _parse_pipeline(data: object) -> list[PipelineStep]:
    if not isinstance(data, dict):
        raise PipelineFileError("Invalid pipeline: the root must be a JSON object.")

    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise PipelineFileError(
            f"Unsupported pipeline version: {version!r}. Expected {SCHEMA_VERSION}."
        )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise PipelineFileError("Invalid pipeline: 'steps' must be a list.")

    return [_parse_step(raw_step) for raw_step in raw_steps]


def _parse_step(data: object) -> PipelineStep:
    if not isinstance(data, dict):
        raise PipelineFileError("Invalid pipeline: each step must be a JSON object.")

    name = data.get("processor")
    if not isinstance(name, str) or name not in Processor.__members__:
        raise PipelineFileError(f"Unknown processor: {name!r}.")
    processor = Processor[name]

    if processor is Processor.GAUSSIAN_BLUR:
        kernel_size = _parse_int(data.get("kernel_size", 5), "kernel_size")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise PipelineFileError(
                f"kernel_size must be positive and odd; got {kernel_size}."
            )
        sigma = _parse_number(data.get("sigma", 0.0), "sigma")
        if sigma < 0:
            raise PipelineFileError(f"sigma must be non-negative; got {sigma}.")
        return PipelineStep(processor, kernel_size=kernel_size, sigma=sigma)

    if processor is Processor.BINARY_THRESHOLD:
        threshold = _parse_int(data.get("threshold", 127), "threshold")
        if not 0 <= threshold <= 255:
            raise PipelineFileError(
                f"threshold must be between 0 and 255; got {threshold}."
            )
        return PipelineStep(processor, threshold=threshold)

    if processor is Processor.ADAPTIVE_THRESHOLD:
        block_size = _parse_int(data.get("block_size", 11), "block_size")
        if block_size < 3 or block_size % 2 == 0:
            raise PipelineFileError(
                f"block_size must be odd and at least 3; got {block_size}."
            )
        constant = _parse_number(data.get("constant", 2.0), "constant")
        use_gaussian = data.get("use_gaussian", False)
        if not isinstance(use_gaussian, bool):
            raise PipelineFileError(
                f"use_gaussian must be a boolean; got {use_gaussian!r}."
            )
        return PipelineStep(
            processor,
            block_size=block_size,
            constant=constant,
            use_gaussian=use_gaussian,
        )

    if processor is Processor.CANNY:
        low_threshold = _parse_int(data.get("low_threshold", 100), "low_threshold")
        high_threshold = _parse_int(data.get("high_threshold", 200), "high_threshold")
        for name, value in (
            ("low_threshold", low_threshold),
            ("high_threshold", high_threshold),
        ):
            if not 0 <= value <= 255:
                raise PipelineFileError(
                    f"{name} must be between 0 and 255; got {value}."
                )
        if low_threshold > high_threshold:
            raise PipelineFileError(
                "low_threshold must be less than or equal to high_threshold; "
                f"got {low_threshold} > {high_threshold}."
            )
        aperture_size = _parse_int(data.get("aperture_size", 3), "aperture_size")
        if aperture_size not in (3, 5, 7):
            raise PipelineFileError(
                f"aperture_size must be 3, 5, or 7; got {aperture_size}."
            )
        return PipelineStep(
            processor,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            aperture_size=aperture_size,
        )

    return PipelineStep(processor)


def _parse_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PipelineFileError(f"{field} must be an integer; got {value!r}.")
    return value


def _parse_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PipelineFileError(f"{field} must be a number; got {value!r}.")
    return float(value)
