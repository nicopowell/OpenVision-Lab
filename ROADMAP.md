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

M0.2 (minimal PySide6 application) is complete. The next milestone is M1:
loading an image and applying grayscale.

## Guiding Principles

- Keep every milestone as small as possible.
- Prefer functional and reviewable increments.
- Learn and validate one concept at a time.
- Avoid overengineering and speculative features.
- Do not introduce abstractions before a concrete need appears.
- Let the architecture evolve with the project.
- Keep image-processing logic separate from the user interface.
- Introduce tests progressively from M1 onward.
- M7 through M10 are optional extensions, not current commitments.
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

## M7: Undo and Redo

**Objective:** Allow users to undo and redo pipeline and parameter changes.

This milestone is conditional. It should only be implemented if editing the
pipeline becomes complex enough that users need history.

It should initially store pipeline state rather than full image copies.

## M8: Save and Load Pipelines

**Objective:** Save and load pipeline configurations.

This milestone depends on a stable representation of processor steps and
parameters.

A simple, readable format such as JSON may be considered. Versioning and
validation should be added only as needed.

## M9: Batch Processing and Video

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

## M10: Extensibility and Plugins

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
