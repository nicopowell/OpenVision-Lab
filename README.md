# OpenVision Lab

A desktop playground for learning and experimenting with Computer Vision.

The initial workflow is:

1. Load an image.
2. Apply image-processing operations.
3. Display the result.

The application loads an image, applies an editable pipeline of image
operations, and shows the original image next to a preview. Steps can be added,
removed, and reordered, and each step keeps its own parameters. The available
processors are grayscale, Gaussian blur, and binary threshold. The preview can
show the final result or any intermediate stage, and the result can be
exported.

## Usage

1. Start the application.
2. Open `File > Open Image...` and choose a PNG, JPEG, BMP, or TIFF file.
3. The original image and the preview are shown side by side.
4. In the left panel, choose a processor and click `Add` to append a step.
5. Select a step to edit its parameters, and use `Remove`, `Move Up`, or
   `Move Down` to change the pipeline.
6. Click `Apply` to reprocess with the current steps.
7. Use the `View` selector to preview the final result or the output after any
   step.
8. Use `File > Save Result As...` to export the final result as PNG, JPEG, BMP,
   or TIFF.
9. Use `Edit > Undo` (`Ctrl+Z`) and `Edit > Redo` (`Ctrl+Shift+Z`) to step back
   and forward through pipeline changes.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Setup

Install the runtime and development dependencies and create the virtual
environment (`.venv/`) in one step:

```bash
uv sync
```

## Development Commands

Run the application:

```bash
uv run python -m openvision_lab
```

Run a command inside the project environment:

```bash
uv run python -c "import PySide6, cv2, numpy; print(PySide6.__version__, cv2.__version__, numpy.__version__)"
```

Run the test suite:

```bash
uv run pytest
```

## Project Status

See `ROADMAP.md` for the current milestone and planned work.
