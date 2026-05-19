"""Application entry point for QuickFixTranscription."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.transcription_window import TranscriptionWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("QuickFixTranscription")
    window = TranscriptionWindow()
    window.resize(1180, 760)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
