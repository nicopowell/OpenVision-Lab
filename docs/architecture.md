# Architecture

OpenVision Lab is a desktop Computer Vision playground. It loads one image,
runs an ordered pipeline of operations, and shows the original next to a
preview of the result or any intermediate stage. Processing is kept separate
from the Qt interface so it can be understood and tested on its own.

## Modules

| Module | Responsibility |
| --- | --- |
| `image_ops.py` | OpenCV primitives: load, save, grayscale, Gaussian blur, binary threshold. No Qt. |
| `pipeline.py` | Pipeline model: `Processor`, `PipelineStep`, `apply_step`, `run_pipeline`, `run_pipeline_with_intermediates`, `default_pipeline`. |
| `pipeline_io.py` | Save and load pipelines as JSON, with validation. |
| `history.py` | Snapshot-based undo and redo of the pipeline. |
| `qt_image.py` | Convert NumPy arrays to `QPixmap` for display. |
| `main_window.py` | Qt window: pipeline editor, previews, and dialogs. Delegates all processing. |
| `app.py` / `__main__.py` | Application entry point and Qt lifecycle. |

## Data flow

`load_image` returns a BGR `uint8` array of shape `(H, W, 3)`. Each step takes
an image and returns one; grayscale and binary threshold output `(H, W)`.
`run_pipeline_with_intermediates` keeps the input plus the output of every
step, so index 0 is the original and index `k` is the result after `k` steps.

## Key decisions

- **Internal representation.** Color images are BGR `uint8` `(H, W, 3)`;
  grayscale is `(H, W)`. The BGR/RGB boundary is resolved at display time by
  naming Qt's `QImage.Format_BGR888`, so no channel swap is needed.
- **Pipeline as data.** A pipeline is a plain list of `PipelineStep`
  dataclasses, and `apply_step` dispatches with explicit branches. No
  registries, factories, or plugins, because there is no concrete need yet.
- **Intermediates by reference.** Step outputs are new arrays, so the original
  image is never modified and no copies are needed to inspect results.
- **Errors.** File problems raise `ImageLoadError`, `ImageSaveError`, or
  `PipelineFileError`; invalid shapes or parameters raise `ValueError`.
- **History stores state, not images.** Undo and redo snapshot the pipeline and
  re-run it. Loading a pipeline resets the history to the loaded state.
- **Scaled preview.** `ImageLabel` scales the image in `paintEvent`, which
  keeps the pixmap size from feeding back into the window layout.

## Pipeline file format

Pipelines are saved as readable JSON. Each step names its processor and only
the parameters that apply to it:

```json
{
  "version": 1,
  "steps": [
    {"processor": "GRAYSCALE"},
    {"processor": "GAUSSIAN_BLUR", "kernel_size": 5, "sigma": 0.0},
    {"processor": "BINARY_THRESHOLD", "threshold": 127}
  ]
}
```

## Tests

Processing, pipeline, IO, and history logic have deterministic tests. The Qt
interface is covered with `pytest-qt` in offscreen mode.

## Not in scope

Plugins, video, machine learning, and GPU acceleration are intentionally
deferred. See the "Deferred Features" section of `ROADMAP.md`.
