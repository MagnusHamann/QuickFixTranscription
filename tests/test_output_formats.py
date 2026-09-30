"""Tests for selectable RTF and structured JSON transcript output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcription import cache as transcription_cache
from transcription.file_utils import output_directory_for
from transcription.json_exporter import has_current_json_output, transcript_json_payload
from transcription.models import (
    BROAD_JEFFERSONIAN_TRANSCRIPTION,
    VERBATIM_TRANSCRIPTION,
    MediaRecord,
    TranscriptResult,
    TranscriptSegment,
    TranscriptionOptions,
    WordToken,
)


def _result(source: Path) -> TranscriptResult:
    return TranscriptResult(
        source_path=source,
        language="en",
        segments=[
            TranscriptSegment(
                "Hello there",
                start=0.1,
                end=0.9,
                speaker="dote-speaker-a",
                words=(WordToken("Hello", 0.1, 0.4, "dote-speaker-a", 0.91),),
            ),
            TranscriptSegment(
                "Hi",
                start=1.0,
                end=1.3,
                speaker="dote-speaker-b",
                words=(WordToken("Hi", 1.0, 1.3, "dote-speaker-b", 0.82),),
                speaker_uncertain=True,
            ),
        ],
    )


class JsonExporterTests(unittest.TestCase):
    def test_payload_uses_dote_style_transcription_and_stable_speaker_labels(self) -> None:
        source = Path("interview.wav")
        payload = transcript_json_payload(
            _result(source),
            VERBATIM_TRANSCRIPTION,
            ["SP1: Hello there", "SP2 (?): Hi"],
            output_key="output-key",
        )

        self.assertEqual(payload["result"], {"language": "en"})
        self.assertEqual(payload["source_file"], "interview.wav")
        self.assertEqual(payload["transcription"][0]["speaker"], "SP1")
        self.assertEqual(payload["transcription"][1]["speaker"], "SP2")
        self.assertTrue(payload["transcription"][1]["speaker_uncertain"])
        self.assertEqual(payload["transcription"][0]["words"][0]["probability"], 0.91)
        self.assertEqual(payload["transcription"][0]["offsets"], {"from": 100, "to": 900})
        self.assertEqual(payload["rendered_lines"], ["SP1: Hello there", "SP2 (?): Hi"])

    def test_current_json_output_requires_matching_key(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transcript.json"
            payload = transcript_json_payload(_result(Path("source.wav")), VERBATIM_TRANSCRIPTION, [], "right-key")
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(has_current_json_output(path, "right-key"))
            self.assertFalse(has_current_json_output(path, "wrong-key"))


class OutputSelectionTests(unittest.TestCase):
    def test_options_require_at_least_one_output_format(self) -> None:
        options = TranscriptionOptions(
            whisper_executable="missing-whisper",
            model_path="missing-model",
            output_rtf=False,
            output_json=False,
        )
        with self.assertRaisesRegex(ValueError, "at least one output format"):
            options.validate()

    def _run_batch(
        self,
        root: Path,
        *,
        output_rtf: bool,
        output_json: bool,
        mode: str = VERBATIM_TRANSCRIPTION,
    ) -> tuple[Path, type]:
        from transcription.batch_processor import TranscriptionBatchProcessor

        source = root / "sample.wav"
        whisper = root / "whisper.exe"
        model = root / "model.bin"
        for path in (source, whisper, model):
            path.write_bytes(b"local")

        class FakeRunner:
            ffmpeg_path = "ffmpeg"

            def run(self, command, _log_callback, _cancelled) -> int:
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            calls = 0

            def __init__(self, *_args, **_kwargs) -> None:
                return None

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio_path, source_path, _work_dir, _language_code, _log_callback, _cancelled):
                type(self).calls += 1
                return _result(source_path)

        options = TranscriptionOptions(
            whisper_executable=str(whisper),
            model_path=str(model),
            transcription_mode=mode,
            output_rtf=output_rtf,
            output_json=output_json,
            use_cache=False,
            resume_completed=False,
        )
        record = MediaRecord(source, "00:02", "Audio", source.stat().st_size)
        with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch(
            "transcription.batch_processor.DoteWhisperEngine", FakeEngine
        ):
            TranscriptionBatchProcessor([record], options, FakeRunner()).run()
        return output_directory_for(source), FakeEngine

    def test_json_only_creates_json_without_rtf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output_dir, fake_engine = self._run_batch(Path(folder), output_rtf=False, output_json=True)

            self.assertEqual(fake_engine.calls, 1)
            self.assertTrue((output_dir / "sample_verbatim.json").exists())
            self.assertFalse((output_dir / "sample_verbatim.rtf").exists())

    def test_selecting_both_formats_runs_transcription_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output_dir, fake_engine = self._run_batch(Path(folder), output_rtf=True, output_json=True)

            self.assertEqual(fake_engine.calls, 1)
            self.assertTrue((output_dir / "sample_verbatim.rtf").exists())
            json_path = output_dir / "sample_verbatim.json"
            self.assertTrue(json_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual([segment["speaker"] for segment in payload["transcription"]], ["SP1", "SP2"])

    def test_broad_jeffersonian_json_contains_final_rendered_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output_dir, fake_engine = self._run_batch(
                Path(folder),
                output_rtf=False,
                output_json=True,
                mode=BROAD_JEFFERSONIAN_TRANSCRIPTION,
            )

            self.assertEqual(fake_engine.calls, 1)
            payload = json.loads((output_dir / "sample_broad_jeffersonian.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["transcription_type"], BROAD_JEFFERSONIAN_TRANSCRIPTION)
            self.assertTrue(payload["rendered_lines"])
            self.assertTrue(any("SP1:" in line for line in payload["rendered_lines"]))


if __name__ == "__main__":
    unittest.main()
