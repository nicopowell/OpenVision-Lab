# OpenVision Lab

A desktop playground for learning and experimenting with Computer Vision.

The initial workflow is:

1. Load an image.
2. Apply image-processing operations.
3. Display the result.

The initial goal is intentionally small: load one image, convert it to
grayscale, and display both the original and processed images.

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
