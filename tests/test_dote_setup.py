from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcription.dote_setup import (
    DOTE_INSTALLER_NAME,
    DOTE_INSTALLER_URL,
    DOTE_VERSION,
    dote_installation_is_ready,
    find_dote_whisper_executable,
    find_installed_dote_version,
    installer_is_valid,
    setup_dote_whisper,
)


class DoteSetupTests(unittest.TestCase):
    @staticmethod
    def _create_installed_app(local_app_data: Path) -> Path:
        app_dir = local_app_data / "DOTE-whisper" / "app-1.0.2"
        whisper = app_dir / "resources" / "bin" / "win32-x64" / "whisper-cli.exe"
        whisper.parent.mkdir(parents=True)
        (app_dir / "DOTE-whisper.exe").touch()
        whisper.touch()
        return whisper

    def test_installed_version_is_found_without_network_or_installer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local_app_data = root / "LocalAppData"
            whisper = self._create_installed_app(local_app_data)
            download_called = False

            def downloader(_url, _destination, _progress):
                nonlocal download_called
                download_called = True
                return False

            with patch("transcription.dote_setup.platform.system", return_value="Windows"), patch(
                "transcription.dote_setup.ensure_dote_models",
                return_value=(root / "segmentation.onnx", root / "embedding.onnx"),
            ):
                ready = setup_dote_whisper(
                    lambda _message: None,
                    dependency_root=root / "Dependencies",
                    quickfix_root=root,
                    local_app_data=local_app_data,
                    downloader=downloader,
                )

            self.assertTrue(ready)
            self.assertFalse(download_called)
            self.assertEqual(find_dote_whisper_executable(local_app_data), whisper)

    def test_missing_installer_is_downloaded_and_installed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependencies = root / "Dependencies"
            local_app_data = root / "LocalAppData"
            calls: list[tuple[str, Path]] = []

            def downloader(url, destination, _progress):
                destination.parent.mkdir(parents=True)
                destination.write_bytes(b"test installer")
                calls.append((url, destination))
                return True

            def installer_runner(_installer, _progress):
                self._create_installed_app(local_app_data)
                return True

            with (
                patch("transcription.dote_setup.platform.system", return_value="Windows"),
                patch("transcription.dote_setup.installer_is_valid", return_value=True),
                patch(
                    "transcription.dote_setup.ensure_dote_models",
                    return_value=(root / "segmentation.onnx", root / "embedding.onnx"),
                ),
            ):
                ready = setup_dote_whisper(
                    lambda _message: None,
                    dependency_root=dependencies,
                    quickfix_root=root,
                    local_app_data=local_app_data,
                    downloader=downloader,
                    installer_runner=installer_runner,
                    poll_timeout=0,
                )

            expected = dependencies / ".tools" / "dote-whisper" / "installers" / DOTE_INSTALLER_NAME
            self.assertTrue(ready)
            self.assertEqual(calls, [(DOTE_INSTALLER_URL, expected)])
            self.assertEqual(find_installed_dote_version(local_app_data), DOTE_VERSION)
            self.assertTrue(dote_installation_is_ready(local_app_data))

    def test_version_directory_without_executable_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            local_app_data = Path(folder) / "LocalAppData"
            (local_app_data / "DOTE-whisper" / "app-1.0.2").mkdir(parents=True)

            self.assertEqual(find_installed_dote_version(local_app_data), DOTE_VERSION)
            self.assertFalse(dote_installation_is_ready(local_app_data))

    def test_invalid_installer_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            installer = Path(folder) / DOTE_INSTALLER_NAME
            installer.write_bytes(b"not the official package")
            self.assertFalse(installer_is_valid(installer))


if __name__ == "__main__":
    unittest.main()
