"""Pipeline model: an ordered list of processing steps.

A step pairs a processor type with its parameters. Running a pipeline simply
applies each configured step to the image in order.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from openvision_lab.image_ops import (
    adaptive_threshold,
    binary_threshold,
    canny,
    gaussian_blur,
    to_grayscale,
)


class Processor(Enum):
    """Supported image processors.

    The value is the human-readable name shown in the user interface.
    """

    GRAYSCALE = "Grayscale"
    GAUSSIAN_BLUR = "Gaussian blur"
    BINARY_THRESHOLD = "Binary threshold"
    ADAPTIVE_THRESHOLD = "Adaptive threshold"
    CANNY = "Canny edges"


@dataclass
class PipelineStep:
    """One configured processor in the pipeline.

    Attributes:
        processor: Which operation to apply.
        kernel_size: Gaussian blur kernel size. Ignored by other processors.
        sigma: Gaussian blur sigma. Ignored by other processors.
        threshold: Binary threshold value. Ignored by other processors.
        block_size: Adaptive threshold neighborhood size. Ignored by other
            processors.
        constant: Adaptive threshold value subtracted from the local mean.
            Ignored by other processors.
        use_gaussian: Whether adaptive threshold uses a Gaussian window instead
            of a flat mean. Ignored by other processors.
        low_threshold: Canny lower hysteresis threshold. Ignored by other
            processors.
        high_threshold: Canny upper hysteresis threshold. Ignored by other
            processors.
        aperture_size: Canny Sobel aperture size. Ignored by other processors.
    """

    processor: Processor
    kernel_size: int = 5
    sigma: float = 0.0
    threshold: int = 127
    block_size: int = 11
    constant: float = 2.0
    use_gaussian: bool = False
    low_threshold: int = 100
    high_threshold: int = 200
    aperture_size: int = 3


def apply_step(image: np.ndarray, step: PipelineStep) -> np.ndarray:
    """Apply a single pipeline step to an image.

    Args:
        image: Input array. Each processor validates its own expected shape:
            grayscale expects a BGR ``(height, width, 3)`` image and produces a
            grayscale ``(height, width)`` one, while Gaussian blur, binary
            threshold, adaptive threshold, and Canny expect a grayscale
            ``(height, width)`` image.
        step: The configured step to apply.

    Returns:
        The processed array.

    Raises:
        ValueError: If the processor is unknown, or if the image does not
            satisfy the processor's precondition. The message names the
            processor that failed.
    """
    try:
        if step.processor is Processor.GRAYSCALE:
            return to_grayscale(image)
        if step.processor is Processor.GAUSSIAN_BLUR:
            return gaussian_blur(image, kernel_size=step.kernel_size, sigma=step.sigma)
        if step.processor is Processor.BINARY_THRESHOLD:
            return binary_threshold(image, threshold=step.threshold)
        if step.processor is Processor.ADAPTIVE_THRESHOLD:
            return adaptive_threshold(
                image,
                block_size=step.block_size,
                constant=step.constant,
                use_gaussian=step.use_gaussian,
            )
        if step.processor is Processor.CANNY:
            return canny(
                image,
                low_threshold=step.low_threshold,
                high_threshold=step.high_threshold,
                aperture_size=step.aperture_size,
            )
        raise ValueError(f"Unsupported processor: {step.processor!r}")
    except ValueError as error:
        # image_ops describes the specific failure; the pipeline adds which
        # step caused it, so the UI can show a useful message.
        raise ValueError(
            f"Processor {step.processor.value!r} failed: {error}"
        ) from error


def run_pipeline_with_intermediates(
    image: np.ndarray, steps: list[PipelineStep]
) -> list[np.ndarray]:
    """Apply a list of steps and keep the result after each step.

    Args:
        image: BGR image of shape ``(height, width, 3)`` and dtype ``uint8``.
        steps: Ordered steps to apply.

    Returns:
        A list of ``len(steps) + 1`` arrays. Index 0 is the input image and
        index ``k`` is the output after the first ``k`` steps. The arrays are
        kept by reference instead of copied, so the original image is part of
        the list but is never modified.

    Raises:
        ValueError: If a step receives an image that does not match its
            precondition, for example a blur step placed before grayscale.
    """
    results = [image]
    for step in steps:
        results.append(apply_step(results[-1], step))
    return results


def run_pipeline(image: np.ndarray, steps: list[PipelineStep]) -> np.ndarray:
    """Apply a list of steps to an image, in order.

    Args:
        image: BGR image of shape ``(height, width, 3)`` and dtype ``uint8``.
        steps: Ordered steps to apply. An empty list returns the image
            unchanged.

    Returns:
        The image produced by the last step.

    Raises:
        ValueError: If a step receives an image that does not match its
            precondition, for example a blur step placed before grayscale.
    """
    return run_pipeline_with_intermediates(image, steps)[-1]


def default_pipeline() -> list[PipelineStep]:
    """Return a new pipeline with the default steps.

    The default sequence is grayscale, Gaussian blur, and binary threshold.
    A new list is returned on every call so callers never share mutable steps.
    """
    return [
        PipelineStep(Processor.GRAYSCALE),
        PipelineStep(Processor.GAUSSIAN_BLUR),
        PipelineStep(Processor.BINARY_THRESHOLD),
    ]
