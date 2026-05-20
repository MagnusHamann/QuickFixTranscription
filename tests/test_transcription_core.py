"""Core tests for QuickFixTranscription."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.acoustic_analysis import _merge_wrapped_stretches
from transcription.engine import parse_whisper_json
from transcription.file_utils import collect_media_files, output_directory_for, unique_output_path
from transcription.languages import LANGUAGE_CHOICES
from transcription.jeffersonian import format_simple_jeffersonian
from transcription.media import build_audio_extract_command
from transcription.models import TranscriptResult, TranscriptSegment, TranscriptionOptions, WordToken
from transcription.model_setup import sha1_file, sha256_file
from transcription.rtf_exporter import rtf_escape, transcript_lines, write_rtf
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


class LanguageChoiceTests(unittest.TestCase):
    def test_danish_and_mandarin_are_available_for_regular_transcription(self) -> None:
        choices = dict(LANGUAGE_CHOICES)
        self.assertEqual(choices["da"], "Danish")
        self.assertEqual(choices["zh"], "Mandarin Chinese")


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
            [
                "1   SP1:        hello",
                "2               (0.2)",
                "3   SP2:        [again]",
                "4   SP1:        [same time]",
            ],
        )

    def test_overlap_brackets_align_across_speaker_lines(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "What a good idea to mark overlapping speech",
                    start=0.0,
                    end=4.0,
                    speaker="one",
                ),
                TranscriptSegment(
                    "this type of speech",
                    start=2.6,
                    end=4.0,
                    speaker="two",
                ),
            ],
        )
        lines = format_simple_jeffersonian(result)
        self.assertEqual(lines[0], "1   SP1:        What a good idea to mark [overlapping speech]")
        self.assertEqual(lines[1], "2   SP2:                                 [this type of speech]")
        self.assertEqual(lines[0].index("["), lines[1].index("["))

    def test_inline_word_silence_is_inserted_inside_turn(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "What a nice idea to mark silence in talk",
                    start=0.0,
                    end=2.2,
                    speaker="one",
                    words=(
                        WordToken("What", 0.0, 0.2, "one"),
                        WordToken("a", 0.2, 0.3, "one"),
                        WordToken("nice", 0.3, 0.5, "one"),
                        WordToken("idea", 0.5, 0.8, "one"),
                        WordToken("to", 0.8, 0.9, "one"),
                        WordToken("mark", 0.9, 1.0, "one"),
                        WordToken("silence", 1.2, 1.5, "one"),
                        WordToken("in", 1.5, 1.6, "one"),
                        WordToken("talk", 1.6, 1.9, "one"),
                    ),
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result),
            ["1   SP1:        What a nice idea to mark (0.2) silence in talk"],
        )

    def test_jeffersonian_lines_wrap_at_50_text_characters(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "one two three four five six seven eight nine ten eleven twelve thirteen",
                    start=0.0,
                    end=4.0,
                    speaker="one",
                )
            ],
        )
        lines = format_simple_jeffersonian(result)
        self.assertEqual(
            lines,
            [
                "1   SP1:        one two three four five six seven eight nine ten",
                "2   SP1:        eleven twelve thirteen",
            ],
        )
        for line in lines:
            text = line[16:]
            self.assertLessEqual(len(text), 50)

    def test_jeffersonian_line_width_can_be_customized(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "one two three four five six seven eight",
                    start=0.0,
                    end=3.0,
                    speaker="one",
                )
            ],
        )
        lines = format_simple_jeffersonian(result, max_text_columns=25)
        self.assertEqual(
            lines,
            [
                "1   SP1:        one two three four five",
                "2   SP1:        six seven eight",
            ],
        )
        for line in lines:
            self.assertLessEqual(len(line[16:]), 25)

    def test_jeffersonian_strips_asr_punctuation(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("Hello, what happened? Let's see.", start=0.0, end=2.0, speaker="one")
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result),
            ["1   SP1:        Hello what happened Let's see"],
        )

    def test_mandarin_jeffersonian_uses_pinyin(self) -> None:
        fake_pypinyin = types.ModuleType("pypinyin")
        fake_pypinyin.Style = types.SimpleNamespace(TONE3="tone3")

        def fake_lazy_pinyin(text: str, **_kwargs: object) -> list[str]:
            lookup = {
                "\u4f60": "ni3",
                "\u597d": "hao3",
                "\u4e16": "shi4",
                "\u754c": "jie4",
            }
            return [lookup.get(char, char) for char in text]

        fake_pypinyin.lazy_pinyin = fake_lazy_pinyin

        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="zh",
            segments=[
                TranscriptSegment("\u4f60\u597d\uff0c\u4e16\u754c\u3002", start=0.0, end=2.0, speaker="one")
            ],
        )

        with patch.dict(sys.modules, {"pypinyin": fake_pypinyin}):
            self.assertEqual(
                format_simple_jeffersonian(result),
                ["1   SP1:        ni3 hao3 shi4 jie4"],
            )

    def test_mandarin_manual_language_override_uses_pinyin(self) -> None:
        fake_pypinyin = types.ModuleType("pypinyin")
        fake_pypinyin.Style = types.SimpleNamespace(TONE3="tone3")
        fake_pypinyin.lazy_pinyin = lambda text, **_kwargs: ["ni3" if char == "\u4f60" else "hao3" for char in text]

        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language=None,
            segments=[TranscriptSegment("\u4f60\u597d", start=0.0, end=1.0, speaker="one")],
        )

        with patch.dict(sys.modules, {"pypinyin": fake_pypinyin}):
            self.assertEqual(
                format_simple_jeffersonian(result, language_code="zh"),
                ["1   SP1:        ni3 hao3"],
            )

    def test_jeffersonian_rtf_can_skip_title_and_use_monospace(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.rtf"
            write_rtf(path, "Ignored title", ["1   SP1:        hello"], font_name="Courier New", include_title=False)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Courier New", content)
            self.assertNotIn("Ignored title", content)
            self.assertIn("1   SP1:        hello", content)

    def test_regular_transcript_lines_use_sp_speaker_labels(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("first", start=0.0, end=0.5, speaker="alpha"),
                TranscriptSegment("second", start=0.7, end=1.0, speaker="beta"),
                TranscriptSegment("third", start=1.2, end=1.5, speaker="alpha"),
            ],
        )
        self.assertEqual(
            transcript_lines(result),
            [
                "[00:00] SP1: first",
                "[00:01] SP2: second",
                "[00:01] SP1: third",
            ],
        )

    def test_regular_transcript_lines_default_to_sp1_without_diarization(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[TranscriptSegment("single speaker text")],
        )
        self.assertEqual(transcript_lines(result), ["SP1: single speaker text"])


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
                                "words": [
                                    {
                                        "timestamps": {"from": "00:00:01,000", "to": "00:00:01,400"},
                                        "word": "hello",
                                        "probability": 0.9,
                                    }
                                ],
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
            self.assertEqual(parsed.segments[0].words[0].text, "hello")
            self.assertEqual(parsed.segments[0].words[0].start, 1.0)
            self.assertEqual(parsed.segments[0].words[0].end, 1.4)


class AcousticAnalysisTests(unittest.TestCase):
    def test_continuous_quiet_and_pitch_stretches_are_merged(self) -> None:
        degree = "\N{DEGREE SIGN}"
        up = "\N{UPWARDS ARROW}"
        self.assertEqual(
            _merge_wrapped_stretches([f"{degree}Let's{degree}", f"{degree}see{degree}", "now"]),
            [f"{degree}Let's see{degree}", "now"],
        )
        self.assertEqual(
            _merge_wrapped_stretches([f"{up}Let's", f"{up}see", "now"]),
            [f"{up}Let's see", "now"],
        )

    def test_local_acoustic_annotations_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            wav_path = Path(folder) / "sample.wav"
            sample_rate = 16000
            samples: list[int] = []
            specs = [
                (0.15, 220.0, 0.08),
                (0.60, 220.0, 0.08),
                (0.15, 220.0, 0.08),
                (0.15, 220.0, 0.015),
                (0.15, 180.0, 0.08),
                (0.15, 300.0, 0.08),
                (0.15, 220.0, 0.25),
            ]
            for duration, frequency, amplitude in specs:
                for index in range(int(duration * sample_rate)):
                    sample = amplitude * math.sin(2 * math.pi * frequency * (index / sample_rate))
                    samples.append(int(sample * 32767))

            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))

            result = TranscriptResult(
                source_path=wav_path,
                language="en",
                segments=[
                    TranscriptSegment(
                        "one long normal quiet rise loud",
                        start=0.0,
                        end=1.5,
                        speaker="one",
                        words=(
                            WordToken("one", 0.0, 0.15, "one"),
                            WordToken("long", 0.15, 0.75, "one"),
                            WordToken("normal", 0.75, 0.90, "one"),
                            WordToken("quiet", 0.90, 1.05, "one"),
                            WordToken("rise", 1.05, 1.35, "one"),
                            WordToken("loud", 1.35, 1.50, "one"),
                        ),
                    )
                ],
            )
            annotated = apply_local_acoustic_annotations(wav_path, result)
            text = annotated.segments[0].text
            self.assertIn("lo::ng", text)
            self.assertIn("°quiet°", text)
            self.assertIn("↑rise", text)
            self.assertIn("LOUD", text)


class ModelSetupTests(unittest.TestCase):
    def test_transcription_options_reject_out_of_range_jeffersonian_width(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            whisper.write_text("exe", encoding="utf-8")
            model.write_text("model", encoding="utf-8")
            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                jeffersonian_line_width=10,
            )
            with self.assertRaises(ValueError):
                options.validate()

    def test_sha1_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha1_file(path), "a9993e364706816aba3e25717850c26c9cd0d89d")

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
