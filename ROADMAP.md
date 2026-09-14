# OpenVision Lab Roadmap

## General Objective

OpenVision Lab is a desktop playground for learning Computer Vision with
Python.

The initial workflow is:

1. Load an image.
2. Apply a linear sequence of image-processing operations.
3. Visualize the results.

The project prioritizes learning Computer Vision while building a functional
and understandable application.

## Current Status

M9 (classical Computer Vision processors) increment 1 is complete: the pipeline
supports Canny edge detection. Further classical processors are not implemented
yet. M10 (batch processing and video) and M11 (extensibility and plugins) remain
conditional.

## Guiding Principles

- Keep every milestone as small as possible.
- Prefer functional and reviewable increments.
- Learn and validate one concept at a time.
- Avoid overengineering and speculative features.
- Do not introduce abstractions before a concrete need appears.
- Let the architecture evolve with the project.
- Keep image-processing logic separate from the user interface.
- Introduce tests progressively from M1 onward.
- M10 and M11 are optional extensions, not current commitments. M9 is the
  active milestone.
- Keep cross-platform compatibility when it does not add unnecessary complexity.

## M0: Bootstrap

M0 must remain very small. Its purpose is to establish a reproducible
development environment without building application features prematurely.

### M0.1: Prepare the Python Environment

**Objective:** Prepare a reproducible Python development environment and
document the development commands.

**Learning focus:**

- Python environments and dependencies.
- Basic project setup.
- Running development commands consistently.

**Expected result:**

- The project can be set up consistently on Linux.
- The selected dependencies are documented.
- The commands for development tasks are documented.

**Current status:** Complete.

### M0.2: Minimal PySide6 Application

**Objective:** Create a minimal PySide6 application that starts successfully.

**Learning focus:**

- Qt application lifecycle.
- Event loops.
- Basic windows and widgets.

**Expected result:**

- The application opens a window and exits cleanly.
- No image-processing functionality is required yet.

**New decisions:**

- Minimal application entry point.
- Basic UI organization.
- Development command used to launch the application.

**Current status:** Complete.

## M1: Load Image and Apply Grayscale

**Objective:** Load one image, display the original, convert it to grayscale,
and display the processed result.

**Learning focus:**

- Separating UI code from processing logic.
- NumPy arrays, shapes, channels, and data types.
- OpenCV image loading and grayscale conversion.
- Converting images for display in Qt.
- Basic error handling.

**Expected result:**

- The user can select a supported image.
- The original image is displayed.
- The grayscale result is displayed.
- Invalid image files produce a useful error.
- The grayscale logic has focused tests.

**New decisions:**

- Initially supported image formats.
- Internal image representation.
- BGR/RGB conversion boundaries.
- Basic error behavior.

A generic processor abstraction is not needed at this stage.

**Current status:** Complete.

## M2: Multiple Processors in a Fixed Sequence

**Objective:** Execute multiple processors in a fixed linear order.

The initial processors may include:

- Grayscale.
- Gaussian blur.
- Binary threshold.

**Learning focus:**

- Function composition.
- Sequential data flow.
- Processor preconditions.
- Image filtering and thresholding.
- The effect of operation order.

**Expected result:**

- An image can pass through multiple operations.
- The final result is displayed.
- The sequence is still fixed and does not need to be edited by the user.

**New decisions:**

- How to represent a sequence temporarily.
- Which processors accept color or grayscale input.
- How incompatible operations are reported.

**Implemented:**

- A fixed pipeline: grayscale -> Gaussian blur -> binary threshold.
- `PIPELINE`, a tuple of plain processing functions.
- `run_pipeline()`, which applies the steps in order.
- Fixed parameters for blur (`ksize=(5, 5)`, `sigma=0`) and threshold
  (`thresh=127`, `maxval=255`, `THRESH_BINARY`).
- The final result is displayed in the UI next to the original image.
- Focused tests for blur, threshold, and the pipeline.

**Current status:** Complete.

Do not introduce plugins, registries, factories, or dynamic discovery.

## M3: Configurable Parameters

**Objective:** Allow the user to configure basic processor parameters.

Possible parameters include:

- Blur kernel size.
- Blur sigma.
- Threshold value.

**Learning focus:**

- UI state and validation.
- Qt signals and controls.
- Image value ranges.
- The visual effect of processing parameters.

**Expected result:**

- The user can change supported parameters.
- Invalid values are rejected clearly.
- Processing uses the selected values.

**New decisions:**

- Parameter representation.
- Valid ranges and defaults.
- Explicit processing versus automatic previews.

**Implemented:**

- `gaussian_blur()` accepts a configurable `kernel_size` and `sigma`.
- `binary_threshold()` accepts a configurable `threshold` (maximum value fixed
  at 255).
- `run_pipeline()` receives those parameters and applies the fixed steps in
  explicit order. The M2 `PIPELINE` tuple was replaced by these calls.
- Parameters are validated: `ValueError` on even or non-positive kernel sizes,
  negative sigma, or thresholds outside 0-255.
- The UI has a "Parameters" group with a kernel-size spin box, a threshold spin
  box, and an Apply button that reprocesses the loaded image.
- Focused tests cover custom parameters and their valid ranges.

## M4: Editable Pipeline

**Objective:** Allow the user to add, remove, and reorder supported processors.

**Learning focus:**

- Application state.
- Synchronizing a data model and UI.
- Lists of configured operations.
- Validation of pipeline state.

**Expected result:**

- The user can build a linear pipeline from the available processors.
- Each step keeps its own parameters.
- The pipeline executes in the selected order.

This is the first milestone where a small processor-step abstraction may be
justified. It should represent only the concrete needs of the UI, such as
processor type and parameters. It should not become a plugin architecture.

**Implemented:**

- A `PipelineStep` dataclass pairs a `Processor` enum with its parameters, and
  `run_pipeline()` applies a list of steps in order.
- The pipeline is a plain list that the user can edit.
- The UI lists the steps and provides Add, Remove, Move Up, and Move Down
  controls, plus a combo box to choose the processor.
- Each step keeps its own parameters: kernel size and sigma for Gaussian blur,
  and the threshold value for binary threshold. The editor shows only the
  fields relevant to the selected step.
- The available processors are grayscale, Gaussian blur, and binary threshold.
- `Apply` reprocesses the loaded image with the current steps.
- Focused tests cover step ordering, removing steps, per-step parameters, and
  incompatible order.

**Current status:** Complete.

Intermediate results and exporting the final image remain for M5.

## M5: Intermediate Results and Export

**Objective:** Inspect the result after each processor and optionally save the
final processed image.

**Learning focus:**

- Intermediate application state.
- Image copies and memory usage.
- Image scaling for visualization.
- File output.

**Expected result:**

- The user can inspect intermediate results.
- The final result can be exported.
- The original image remains unchanged.

**New decisions:**

- How intermediate results are selected or displayed.
- Which output formats are supported.
- How large images are handled.

**Implemented:**

- `run_pipeline_with_intermediates()` returns the result of every stage:
  index 0 is the original image and index `k` is the output after the first `k`
  steps. Arrays are kept by reference, not copied, and the original is never
  modified. `run_pipeline()` delegates to it.
- A `View` selector above the preview panel shows the final result or the
  output after any step.
- The preview is scaled to fit its panel while preserving the aspect ratio;
  exporting always writes the full-resolution result.
- `File > Save Result As...` exports the final processed image as PNG, JPEG,
  BMP, or TIFF through `save_image()`, which raises `ImageSaveError` on failure.
- Focused tests cover the intermediate list and image saving.

**Current status:** Complete.

## M6: Testing, Quality, and Refactoring

**Objective:** Consolidate the implemented behavior without adding major new
features.

**Learning focus:**

- pytest fixtures and test organization.
- NumPy array comparisons.
- Integration tests.
- Refactoring based on real duplication or complexity.
- Consistent error handling.

**Expected result:**

- Core processors have deterministic tests.
- Pipeline ordering and parameter validation are tested.
- Important image loading behavior is covered.
- The code remains understandable and maintainable.

Tests must be introduced progressively from M1. M6 is a quality milestone,
not the first testing milestone.

Refactor only problems demonstrated by the existing implementation. Avoid a
large rewrite for architectural purity.

**Implemented:**

- Tests are organized by concern: `test_io.py` (load/save), `test_processors.py`
  (grayscale, blur, threshold), `test_pipeline.py`, and `test_integration.py`
  (file-to-file flows), with shared fixtures in `conftest.py`.
- Array comparisons use `np.testing.assert_array_equal` for detailed failures.
- Integration tests cover loading, running the pipeline, and saving end to end,
  and verify that the original image is never modified.
- No production refactor was needed: the existing structure was found
  understandable and maintainable, so no changes were made to `openvision_lab/`.

**Current status:** Complete.

## M7: Undo and Redo

**Objective:** Allow users to undo and redo pipeline and parameter changes.

This milestone is conditional. It should only be implemented if editing the
pipeline becomes complex enough that users need history.

It should initially store pipeline state rather than full image copies.

**Implemented:**

- `PipelineHistory` keeps snapshots of the pipeline (steps and parameters),
  never image arrays, and drops the redo branch after a new edit.
- The `Edit` menu provides `Undo` and `Redo` with the standard shortcuts, and
  they are enabled only when there is a state to return to.
- Structural edits (add, remove, move) are recorded immediately. Continuous
  parameter edits are coalesced into a single entry with a short timer.
- Undo/redo restore the pipeline and re-run it, so the preview and stage
  selector stay in sync. Loading an image does not reset the history.
- `tests/test_history.py` covers recording, undo/redo, redo invalidation,
  snapshot independence, and reset.

**Current status:** Complete.

## M8: Save and Load Pipelines

**Objective:** Save and load pipeline configurations.

This milestone depends on a stable representation of processor steps and
parameters.

A simple, readable format such as JSON may be considered. Versioning and
validation should be added only as needed.

**Implemented:**

- Pipelines are stored as readable JSON with a `version` field and one entry
  per step. Each step names its processor by the stable `Processor` name and
  includes only the parameters that apply to it.
- `save_pipeline()` and `load_pipeline()` live in `pipeline_io.py` and raise
  `PipelineFileError` on unreadable, malformed, or out-of-range files.
- Loading validates the file before changing the UI, uses the `PipelineStep`
  defaults for missing parameters, and ignores unknown keys.
- The `File` menu provides `Save Pipeline...` and `Load Pipeline...`. Loading a
  pipeline resets the undo history, so it becomes the new baseline.
- `tests/test_pipeline_io.py` covers round-trips, the JSON structure, validation
  errors, defaults, and an end-to-end save/load/run check.

**Current status:** Complete.

## M9: Classical Computer Vision Processors

**Objective:** Add classic Computer Vision processors that go beyond basic
filtering and thresholding, while keeping the single-image, grayscale pipeline.

**Learning focus:**

- Edge detection and image gradients.
- Hysteresis thresholds and how their values affect the result.
- The Sobel aperture size and its effect on edge detection.
- How operation order shapes the result.

**Expected result:**

- The pipeline can include a Canny edge detection step.
- The step keeps its own parameters.
- Results are deterministic and covered by tests.

**New decisions:**

- Public parameter names `low_threshold` and `high_threshold` map to OpenCV's
  `threshold1` and `threshold2`.
- `PipelineStep` stays flat. A per-processor parameter map is reconsidered when
  a second parameterized processor appears.

**Implemented (increment 1):**

- `canny()` in `image_ops.py` validates grayscale input, integer thresholds in
  0-255, `low_threshold` less than or equal to `high_threshold`, and an
  aperture size of 3, 5, or 7.
- `Processor.CANNY`, the flat `low_threshold`, `high_threshold`, and
  `aperture_size` fields on `PipelineStep`, and a branch in `apply_step`.
- Pipelines are saved and loaded with the Canny parameters, using the same
  validation and defaults.
- The UI shows spin boxes for both thresholds and the aperture only for a
  selected Canny step.
- Tests cover the processor, pipeline order, persistence, and UI parameters.

Further processors in this milestone (adaptive threshold, histogram
equalization, morphology) remain to be added one at a time. Processors that
output color (contours, features, Hough) need a separate architecture decision.

## M10: Batch Processing and Video

**Objective:** Apply stable image pipelines to multiple images and,
eventually, video frames.

Batch processing should be considered before video because it is closer to the
existing image workflow.

Video introduces additional concerns:

- Frames and frame rate.
- Codecs.
- Long-running processing.
- Progress and cancellation.
- Background work.
- Processors with temporal state.

This milestone is conditional and should not influence the initial image
architecture prematurely.

## M11: Extensibility and Plugins

**Objective:** Support external processors only if a concrete need appears.

Possible reasons to consider this milestone include:

- A large number of processors.
- External contributions.
- Separately distributed processors.
- A stable need for user-installed extensions.

Plugins, dynamic loading, registries, factories, and similar systems should
not be designed before that need exists. They introduce additional concerns
around contracts, compatibility, packaging, errors, and security.

## Deferred Features

The following features remain intentionally deferred:

- Machine learning models.
- Real-time camera processing.
- Advanced node-based graphs.
- GPU acceleration.
- Project collaboration or backend services.
- Advanced packaging and installers.
- Performance optimization without measurements.
