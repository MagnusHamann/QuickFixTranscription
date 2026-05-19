"""Core tests for QuickFixTranscription."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transcription.engine import parse_whisper_json
from transcription.file_utils import collect_media_files, output_directory_for, unique_output_path
from transcription.jeffersonian import format_simple_jeffersonian
from transcription.media import build_audio_extract_command
from transcription.models import TranscriptResult, TranscriptSegment
from transcription.model_setup import sha1_file
from transcription.rtf_exporter import rtf_escape
from transcription.time_utils import validate_time_range


class TimeRangeTests(unittest.TestCase):
    def test_optional_start_and_finish_are_validated(self) -> None:
        self.assertEqual(validate_time_range("01:00", "02:30"), (60, 150))
        self.assertEqual(validate_time_range("", "02:30"), (None, 150))
        self.assertEqual(validate_time_range("01:00", ""), (60, None))

    def test_finish_must_be_after_start(self) -> None:
        with self.assertRaises(ValueError):
            validate_time_range("02:00", "01:59")


class MediaCommandTests(unittest.TestCase):
    def test_build_extract_command_uses_duration_for_start_and_finish(self) -> None:
        command = build_audio_extract_command(
            "ffmpeg",
            Path("input.mp4"),
            Path("output.wav"),
            start_seconds=70,
            finish_seconds=100,
        )
        self.assertIn("-ss", command)
        self.assertIn("00:01:10", command)
        self.assertIn("-t", command)
        self.assertIn("00:00:30", command)
        self.assertIn("-ar", command)
        self.assertIn("16000", command)
        self.assertEqual(command[-1], "output.wav")


class OutputPathTests(unittest.TestCase):
    def test_transcription_output_folder_and_unique_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "meeting.mp4"
            source.write_bytes(b"media")
            first = unique_output_path(source, "transcript", ".rtf")
            self.assertEqual(output_directory_for(source), Path(folder) / "QuickFixTranscription")
            first.write_text("existing", encoding="utf-8")
            second = unique_output_path(source, "transcript", ".rtf")
            self.assertEqual(second.name, "meeting_transcript_2.rtf")

    def test_collect_media_files_skips_transcription_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            keep = root / "keep.wav"
            keep.write_bytes(b"audio")
            ignored_dir = root / "QuickFixTranscription"
            ignored_dir.mkdir()
            ignored = ignored_dir / "ignored.wav"
            ignored.write_bytes(b"audio")
            self.assertEqual(collect_media_files([root]), [keep])


class RtfTests(unittest.TestCase):
    def test_rtf_escape_handles_control_chars_and_unicode(self) -> None:
        escaped = rtf_escape(r"Use {braces} \ and café")
        self.assertIn(r"\{braces\}", escaped)
        self.assertIn(r"\\", escaped)
        self.assertIn(r"\u233?", escaped)


class JeffersonianTests(unittest.TestCase):
    def test_simple_formatter_marks_silence_and_overlap(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("hello", start=0.0, end=0.5, speaker="one"),
                TranscriptSegment("again", start=0.7, end=1.0, speaker="two"),
                TranscriptSegment("same time", start=0.8, end=1.2, speaker="one"),
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result),
            ["A: hello", "(0.2)", "B: again", "A: [same time]"],
        )


class WhisperJsonTests(unittest.TestCase):
    def test_parse_whisper_cpp_json(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:01,000", "to": "00:00:02,500"},
                                "text": " hello ",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_whisper_json(json_path, Path("source.wav"))
            self.assertEqual(parsed.language, "en")
            self.assertEqual(parsed.segments[0].text, "hello")
            self.assertEqual(parsed.segments[0].start, 1.0)
            self.assertEqual(parsed.segments[0].end, 2.5)


class ModelSetupTests(unittest.TestCase):
    def test_sha1_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha1_file(path), "a9993e364706816aba3e25717850c26c9cd0d89d")


if __name__ == "__main__":
    unittest.main()
