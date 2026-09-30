"""Application entry point for QuickFixTranscription."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def configure_qt_runtime() -> Path | None:
    """Point Qt at the platform plugins bundled with this Python environment."""
    if os.name != "nt":
        return None

    spec = importlib.util.find_spec("PySide6")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if not locations:
        raise RuntimeError("PySide6 is not installed in the QuickFixTranscription environment.")

    pyside_root = Path(locations[0]).resolve()
    plugins_root = pyside_root / "plugins"
    platforms_root = plugins_root / "platforms"
    windows_plugin = platforms_root / "qwindows.dll"
    if not windows_plugin.exists():
        raise RuntimeError(
            "The PySide6 Windows platform plugin is missing. "
            "Run the QuickFixTranscription launcher again to repair local dependencies."
        )

    os.environ["QT_PLUGIN_PATH"] = str(plugins_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_root)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(pyside_root).lower() not in {entry.lower() for entry in path_entries if entry}:
        os.environ["PATH"] = str(pyside_root) + os.pathsep + os.environ.get("PATH", "")
    return platforms_root


configure_qt_runtime()

from PySide6.QtWidgets import QApplication

from ui.transcription_window import TranscriptionWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("QuickFixTranscription")
    if "--startup-check" in sys.argv:
        # Construct the actual UI as well as QApplication so launcher checks
        # catch missing plugins and widget/signal wiring failures.
        window = TranscriptionWindow()
        window.resize(1180, 760)
        window.close()
        print("QuickFixTranscription Qt startup check passed.")
        return 0
    window = TranscriptionWindow()
    window.resize(1180, 760)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
