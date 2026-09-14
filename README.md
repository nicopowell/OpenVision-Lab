# OpenVision Lab

A desktop playground for learning and experimenting with Computer Vision.

The initial workflow is:

1. Load an image.
2. Apply image-processing operations.
3. Display the result.

The application loads an image, runs a fixed pipeline (grayscale, Gaussian
blur, binary threshold), and shows the original and processed images. The blur
kernel size and threshold value can be adjusted.

## Usage

1. Start the application.
2. Open `File > Open Image...` and choose a PNG, JPEG, BMP, or TIFF file.
3. The original and processed images are shown side by side.
4. Adjust `Blur kernel size` or `Threshold` and click `Apply` to reprocess.

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
