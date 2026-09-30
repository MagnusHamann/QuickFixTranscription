from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ui.dote_integration_panel import (
    dote_version_from_path,
    find_dote_executable,
    find_dote_installer,
    find_installed_dote_version,
)


class DoteIntegrationTests(unittest.TestCase):
    def test_newest_dote_installer_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older = root / "DOTE-Whisper-win-x64-1.0.1.exe"
            newer = root / "DOTE-Whisper-win-x64-1.2.0.exe"
            older.touch()
            newer.touch()

            self.assertEqual(find_dote_installer(root), newer)

    def test_dote_installer_version_ignores_x64_in_filename(self) -> None:
        installer = Path("DOTE-Whisper-win-x64-1.0.2.exe")
        self.assertEqual(dote_version_from_path(installer), (1, 0, 2))

    def test_automatically_cached_installer_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            installer = (
                root
                / "QuickFixAppDependencies"
                / ".tools"
                / "dote-whisper"
                / "installers"
                / "DOTE-Whisper-win-x64-1.0.2.exe"
            )
            installer.parent.mkdir(parents=True)
            installer.touch()

            self.assertEqual(find_dote_installer(root), installer)

    def test_installed_dote_whisper_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            local_app_data = Path(folder)
            executable = local_app_data / "DOTE-whisper-1.0.1" / "app-1.0.1" / "DOTE-whisper.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()

            self.assertEqual(find_dote_executable(local_app_data), executable)

    def test_newest_installed_dote_version_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            local_app_data = Path(folder)
            install_root = local_app_data / "DOTE-whisper"
            (install_root / "app-1.0.1").mkdir(parents=True)
            (install_root / "app-1.0.2").mkdir()

            self.assertEqual(find_installed_dote_version(local_app_data), (1, 0, 2))

    def test_separate_dote_application_is_not_mistaken_for_dote_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            local_app_data = Path(folder)
            executable = local_app_data / "DOTE" / "DOTE.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()

            self.assertIsNone(find_dote_executable(local_app_data))


if __name__ == "__main__":
    unittest.main()
