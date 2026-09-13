import sys

from PySide6.QtWidgets import QApplication

from openvision_lab.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    return app.exec()
