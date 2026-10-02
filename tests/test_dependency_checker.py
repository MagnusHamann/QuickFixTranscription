from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from transcription.runtime_check import unsatisfied_requirements
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QuickFixDependencyCheck.checks import (  # noqa: E402
    DependencyCheckResult,
    DependencyCheckSpec,
    build_dependency_checks,
    check_ffmpeg_runtime,
    check_python_runtime,
    check_danish_whisper_runtime,
    check_whisper_model,
    command_line_report,
)


class DependencyCheckerTests(unittest.TestCase):
    def test_local_runtime_check_accepts_installed_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("PySide6>=6.7\n", encoding="utf-8")
            self.assertEqual(unsatisfied_requirements(requirements), ())

    def test_runtime_only_imports_are_declared_as_direct_requirements(self) -> None:
        requirements = ROOT / "QuickFixTranscription" / "requirements.txt"
        requirement_names = {
            line.split("=", 1)[0].split(">", 1)[0].strip().lower()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("packaging", requirement_names)
        self.assertIn("numpy", requirement_names)

    def test_local_runtime_check_reports_missing_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("quickfix-package-that-does-not-exist>=1\n", encoding="utf-8")
            failures = unsatisfied_requirements(requirements)
            self.assertEqual(len(failures), 1)
            self.assertIn("not installed", failures[0])

    def test_build_dependency_checks_includes_the_expected_rows(self) -> None:
        checks = build_dependency_checks()
        self.assertGreaterEqual(len(checks), 10)
        self.assertEqual(checks[0].key, "python")
        self.assertEqual(checks[-1].key, "sherpa_runtime")

    def test_python_runtime_check_reports_the_current_interpreter(self) -> None:
        result = check_python_runtime()
        self.assertIsInstance(result, DependencyCheckResult)
        self.assertEqual(result.key, "python")
        self.assertIn(sys.executable, "\n".join(result.details))
        self.assertIn("Version:", "\n".join(result.details))

    def test_ffmpeg_runtime_check_uses_local_paths(self) -> None:
        with patch("QuickFixDependencyCheck.checks.find_ffmpeg_tools", return_value=("C:/local/ffmpeg.exe", "C:/local/ffprobe.exe")):
            result = check_ffmpeg_runtime()
        self.assertTrue(result.ok)
        self.assertIn("C:/local/ffmpeg.exe", "\n".join(result.details))
        self.assertIn("C:/local/ffprobe.exe", "\n".join(result.details))

    def test_whisper_model_check_accepts_a_local_custom_model(self) -> None:
        with patch("QuickFixDependencyCheck.checks.find_model_file", return_value="C:/models/custom.gguf"), patch(
            "QuickFixDependencyCheck.checks.model_is_valid", return_value=False
        ):
            result = check_whisper_model()
        self.assertTrue(result.ok)
        self.assertIn("whisper model", result.headline.lower())
        self.assertIn("C:/models/custom.gguf", "\n".join(result.details).replace("\\", "/"))

    def test_optional_danish_whisper_check_does_not_fail_when_not_installed(self) -> None:
        with patch("QuickFixDependencyCheck.checks.find_danish_whisper_model", return_value=None), patch(
            "QuickFixDependencyCheck.checks.danish_whisper_runtime_is_ready", return_value=False
        ):
            result = check_danish_whisper_runtime()
        self.assertEqual(result.severity, "warn")
        self.assertIn("DOTE Whisper remains available", result.headline)

    def test_command_line_report_includes_each_selected_section(self) -> None:
        fake_result = DependencyCheckResult("fake", "Fake dependency", "pass", "It is ready.", ("local only",))
        fake_spec = DependencyCheckSpec("fake", "Fake dependency", "Description", lambda: fake_result)
        with patch("QuickFixDependencyCheck.checks.build_dependency_checks", return_value=(fake_spec,)):
            report = command_line_report()
        self.assertIn("Fake dependency: PASS - It is ready.", report)
        self.assertIn("local only", report)


if __name__ == "__main__":
    unittest.main()
