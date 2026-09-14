"""Generate the README screenshot of the application.

The window is rendered offscreen, so the screenshot can be regenerated without
a display. Run `uv run python scripts/generate_screenshot.py` to write
`docs/images/overview.png`.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "images" / "overview.png"


def _sample_image() -> np.ndarray:
    """Return a synthetic BGR image with simple shapes."""
    height, width = 360, 480
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = horizontal
    image[:, :, 1] = horizontal[::-1]
    image[:, :, 2] = 128
    cv2.rectangle(image, (60, 80), (220, 280), (255, 255, 255), -1)
    cv2.circle(image, (330, 180), 80, (0, 0, 0), -1)
    cv2.putText(
        image,
        "OpenVision",
        (70, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    # Select the offscreen platform before importing PySide6, so the screenshot
    # works on machines without a display.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Running this file directly puts `scripts/` on sys.path, so add the project
    # root before importing the package.
    sys.path.insert(0, str(PROJECT_ROOT))
    from PySide6.QtWidgets import QApplication

    from openvision_lab.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.current_image = _sample_image()
    window.apply_button.setEnabled(True)
    window.save_action.setEnabled(True)
    window._process()
    window.resize(1000, 600)
    window.show()
    app.processEvents()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
