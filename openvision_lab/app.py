import sys

from PySide6.QtWidgets import QApplication

from openvision_lab.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    """Start the Qt application and run its event loop.

    Args:
        argv: Command-line arguments handed to ``QApplication``. Defaults to
            ``sys.argv`` when omitted.

    Returns:
        The exit code from the event loop (``0`` on a normal close).
    """
    if argv is None:
        argv = sys.argv
    # QApplication must exist before any widget and must stay alive for the
    # whole run. Qt expects exactly one instance per application.
    app = QApplication(argv)
    # Keep a reference to the window so it survives while exec() blocks.
    window = MainWindow()
    window.show()
    # exec() starts the event loop and returns only when the application quits.
    return app.exec()
