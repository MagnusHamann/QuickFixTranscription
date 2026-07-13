"""Core tests for QuickFixTranscription."""

from __future__ import annotations

import json
import math
import socket
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from quickfix_sibling_apps import quickfix_app_roots
from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.acoustic_analysis import _merge_wrapped_stretches
from transcription.batch_processor import TranscriptionBatchProcessor
from transcription.cpu.vad import energy_vad
from transcription.engine import parse_whisper_json
from transcription.event_engine import suggestions_for_transcript
from transcription.file_utils import collect_media_files, output_directory_for, unique_output_path
from transcription.languages import LANGUAGE_CHOICES
from transcription.jeffersonian import BROAD_JEFFERSONIAN_PROFILE, NARROW_JEFFERSONIAN_PROFILE, format_simple_jeffersonian
from transcription.media import build_audio_extract_command, build_channel_extract_command
from transcription.mfa_alignment import AlignedInterval, apply_mfa_word_alignment, parse_textgrid, phone_tier_lines
from transcription.mfa_presets import DEFAULT_SETUP_MFA_PRESET_IDS, MFA_PRESETS, mfa_preset_by_id, preset_for_language_code
from transcription.models import (
    BROAD_JEFFERSONIAN_TRANSCRIPTION,
    NARROW_JEFFERSONIAN_TRANSCRIPTION,
    VERBATIM_TRANSCRIPTION,
    MediaRecord,
    TranscriptResult,
    TranscriptSegment,
    TranscriptionOptions,
    WordToken,
)
from transcription.model_setup import sha1_file, sha256_file
from transcription.offline_guard import offline_processing_guard, reject_remote_source
from transcription.project_builder import build_project_from_transcript
from transcription.project_exports import export_project_docx, export_project_pdf, export_project_txt
from transcription.project_renderer import iter_project_suggestions, render_confirmed_jefferson_lines
from transcription.project_review import set_candidate_status
from transcription.project_store import load_project, save_project
from transcription.overlap_analysis import channels_are_distinct, has_cross_speaker_overlap, merge_channel_results
from transcription import dependencies as transcription_dependencies
from transcription import mfa_alignment as transcription_mfa_alignment
from transcription import mfa_setup as transcription_mfa_setup
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

    def test_build_channel_extract_command_uses_pan_filter(self) -> None:
        command = build_channel_extract_command("ffmpeg", Path("input.wav"), Path("right.wav"), 1)
        self.assertIn("-af", command)
        self.assertIn("pan=mono|c0=c1", command)
        self.assertEqual(command[-1], "right.wav")


class OverlapAnalysisTests(unittest.TestCase):
    def _write_mono_wav(self, path: Path, values: list[int], sample_rate: int = 16000) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"".join(struct.pack("<h", value) for value in values))

    def test_channel_distinct_detection_rejects_duplicate_channels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample_rate = 16000
            values = [
                int(0.2 * math.sin(2 * math.pi * 220.0 * (index / sample_rate)) * 32767)
                for index in range(sample_rate // 4)
            ]
            left = root / "left.wav"
            right = root / "right.wav"
            self._write_mono_wav(left, values)
            self._write_mono_wav(right, values)
            self.assertFalse(channels_are_distinct(left, right))

    def test_channel_distinct_detection_accepts_different_channels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample_rate = 16000
            left_values = [
                int(0.2 * math.sin(2 * math.pi * 220.0 * (index / sample_rate)) * 32767)
                for index in range(sample_rate // 4)
            ]
            right_values = [
                int(0.2 * math.sin(2 * math.pi * 440.0 * (index / sample_rate)) * 32767)
                for index in range(sample_rate // 4)
            ]
            left = root / "left.wav"
            right = root / "right.wav"
            self._write_mono_wav(left, left_values)
            self._write_mono_wav(right, right_values)
            self.assertTrue(channels_are_distinct(left, right))

    def test_merge_channel_results_preserves_cross_speaker_overlap(self) -> None:
        source = Path("sample.wav")
        fallback = TranscriptResult(source_path=source, language="en", segments=[])
        left = TranscriptResult(
            source_path=source,
            language="en",
            segments=[TranscriptSegment("left speaker", start=0.0, end=1.0, speaker="old")],
        )
        right = TranscriptResult(
            source_path=source,
            language="en",
            segments=[TranscriptSegment("right speaker", start=0.5, end=1.2, speaker="old")],
        )
        merged = merge_channel_results(source, fallback, [("channel_1", left), ("channel_2", right)])
        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertTrue(has_cross_speaker_overlap(merged))
        self.assertEqual([segment.speaker for segment in merged.segments], ["channel_1", "channel_2"])
        lines = format_simple_jeffersonian(merged, profile=BROAD_JEFFERSONIAN_PROFILE)
        self.assertEqual(lines[0].index("["), lines[1].index("["))


class SiblingAppDiscoveryTests(unittest.TestCase):
    def test_quickfix_app_roots_prefers_shared_dependencies_then_current_app(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "QuickFixTranscription"
            editing = root / "QuickFixEditing"
            shared = root / "QuickFixAppDependencies"
            unrelated = root / "OtherApp"
            current.mkdir()
            editing.mkdir()
            unrelated.mkdir()

            self.assertEqual(quickfix_app_roots(current), [shared.resolve(), current.resolve()])


class LanguageChoiceTests(unittest.TestCase):
    def test_danish_and_mandarin_are_available_for_regular_transcription(self) -> None:
        choices = dict(LANGUAGE_CHOICES)
        self.assertEqual(choices["da"], "Danish")
        self.assertEqual(choices["zh"], "Mandarin Chinese")

    def test_mfa_presets_include_uk_english_and_multilingual_pairs(self) -> None:
        presets = {preset.id: preset for preset in MFA_PRESETS}
        self.assertIn("english_uk_mfa", presets)
        self.assertEqual(presets["english_uk_mfa"].acoustic_model, "english_mfa")
        self.assertEqual(presets["english_uk_mfa"].dictionary_model, "english_uk_mfa")
        self.assertIn("english_us_arpa", presets)
        self.assertIn("mandarin_china_mfa", presets)
        self.assertIn("french_mfa", presets)
        self.assertIn("german_mfa", presets)
        self.assertIn("japanese_mfa", presets)
        self.assertIn("korean_mfa", presets)
        self.assertIn("english_uk_mfa", DEFAULT_SETUP_MFA_PRESET_IDS)
        self.assertEqual(preset_for_language_code("en"), mfa_preset_by_id("english_uk_mfa"))


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

    def test_collect_media_files_accepts_aac(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "interview.aac"
            source.write_bytes(b"audio")

            self.assertEqual(collect_media_files([root]), [source])


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
                "2               eleven twelve thirteen",
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
                "2               six seven eight",
            ],
        )
        for line in lines:
            self.assertLessEqual(len(line[16:]), 25)

    def test_jeffersonian_repeats_speaker_only_for_new_turns_not_wrapped_lines(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "one two three four five six",
                    start=0.0,
                    end=1.0,
                    speaker="one",
                ),
                TranscriptSegment(
                    "new turn same speaker",
                    start=1.4,
                    end=2.0,
                    speaker="one",
                ),
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, max_text_columns=18),
            [
                "1   SP1:        one two three four",
                "2               five six",
                "3   SP1:        (0.4) new turn",
                "4               same speaker",
            ],
        )

    def test_jeffersonian_marks_latching_on_no_gap_speaker_change(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("right", start=0.0, end=0.5, speaker="one"),
                TranscriptSegment("yes", start=0.5, end=0.9, speaker="two"),
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result),
            [
                "1   SP1:        right=",
                "2   SP2:        =yes",
            ],
        )

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

    def test_jeffersonian_preserves_disfluencies_and_word_order(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "I I was going to to say uh no I mean wo- we should",
                    start=0.0,
                    end=3.0,
                    speaker="one",
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, max_text_columns=200),
            ["1   SP1:        I I was going to to say eh no I mean wo- we should"],
        )

    def test_jeffersonian_maps_non_word_sounds(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "[cough] throat clear sniff sigh bilabial click click inbreath outbreath \u0259 uh um hm mhm unclear",
                    start=0.0,
                    end=2.0,
                    speaker="one",
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, max_text_columns=200),
            [
                "1   SP1:        ((cough)) ((clears throat)) .snih. ((sigh)) .mt. .dt. .hhh hhh eh eh uhm hm mhm (     )"
            ],
        )

    def test_jeffersonian_marks_low_confidence_words_as_best_guess(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "maybe clear",
                    start=0.0,
                    end=1.0,
                    speaker="one",
                    words=(
                        WordToken("maybe", 0.0, 0.4, "one", confidence=0.12),
                        WordToken("clear", 0.4, 0.8, "one", confidence=0.92),
                    ),
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result),
            ["1   SP1:        (maybe) clear"],
        )

    def test_broad_jeffersonian_skips_voice_quality_annotations(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "\N{DEGREE SIGN}wo::rd\N{DEGREE SIGN} >fast< [cough]",
                    start=0.0,
                    end=1.0,
                    speaker="one",
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, profile=BROAD_JEFFERSONIAN_PROFILE),
            ["1   SP1:        word fast cough"],
        )

    def test_broad_jeffersonian_does_not_mark_low_confidence_words(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "maybe clear",
                    start=0.0,
                    end=1.0,
                    speaker="one",
                    words=(
                        WordToken("maybe", 0.0, 0.4, "one", confidence=0.12),
                        WordToken("clear", 0.4, 0.8, "one", confidence=0.92),
                    ),
                )
            ],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, profile=BROAD_JEFFERSONIAN_PROFILE),
            ["1   SP1:        maybe clear"],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, profile=NARROW_JEFFERSONIAN_PROFILE),
            ["1   SP1:        (maybe) clear"],
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

    def test_rtf_can_use_ipa_font_family(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ipa.rtf"
            write_rtf(path, "IPA", ["SP1: \u0259 \u0283 \u014b"], font_name="Charis")
            content = path.read_text(encoding="utf-8")
            self.assertIn("Charis", content)
            self.assertIn(r"\u601?", content)
            self.assertIn(r"\u643?", content)
            self.assertIn(r"\u331?", content)

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
                                    },
                                    {
                                        "timestamps": {"from": "00:00:01,400", "to": "00:00:01,600"},
                                        "word": "[cough]",
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
            self.assertEqual(parsed.segments[0].words[1].text, "[cough]")


class MfaAlignmentTests(unittest.TestCase):
    def test_mfa_launch_prefers_executable_over_python_module(self) -> None:
        executable = r"C:\QuickFixAppDependencies\.tools\mfa\env\Scripts\mfa.exe"
        self.assertEqual(transcription_mfa_alignment._mfa_command_prefix(executable), [executable])
        self.assertEqual(transcription_mfa_setup._mfa_command_prefix(executable), [executable])

    def test_mfa_launcher_failure_message_triggers_retry_path(self) -> None:
        self.assertTrue(transcription_mfa_alignment._mfa_launcher_failed(["failed to create process."]))
        self.assertTrue(transcription_mfa_setup._mfa_launcher_failed_line("failed to create process."))
        self.assertFalse(transcription_mfa_alignment._mfa_launcher_failed(["Alignment complete."]))
        self.assertFalse(transcription_mfa_setup._mfa_launcher_failed_line("Alignment complete."))

    def test_mfa_alignment_retries_module_when_direct_launcher_fails_silently(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "sample.wav"
            audio.write_bytes(b"local wav placeholder")
            env_dir = root / "mfa-env"
            scripts_dir = env_dir / "Scripts"
            scripts_dir.mkdir(parents=True)
            mfa = scripts_dir / "mfa.exe"
            python = env_dir / "python.exe"
            dictionary = root / "english.dict"
            acoustic = root / "english.zip"
            for path in (mfa, python, dictionary, acoustic):
                path.write_text("placeholder", encoding="utf-8")

            transcript = TranscriptResult(
                source_path=audio,
                language="en",
                segments=[TranscriptSegment("hello", start=0.0, end=0.6, speaker="one")],
            )
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                mfa_executable=str(mfa),
                mfa_dictionary=str(dictionary),
                mfa_acoustic_model=str(acoustic),
            )
            calls: list[list[str]] = []

            def fake_run(command: list[str], _env: dict[str, str], _log, _cancelled) -> tuple[int, list[str]]:
                calls.append(command)
                if len(calls) == 1:
                    return 0, ["failed to create process."]
                align_index = command.index("align")
                output_dir = Path(command[align_index + 4])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "sample.TextGrid").write_text(
                    """
File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0.1
        xmax = 0.5
        text = "hello"
""",
                    encoding="utf-8",
                )
                return 0, ["done"]

            with patch.object(transcription_mfa_alignment, "_run_streamed_mfa_process", side_effect=fake_run):
                aligned = transcription_mfa_alignment.run_mfa_alignment(
                    audio,
                    transcript,
                    root / "work",
                    options,
                    lambda _message: None,
                    lambda: False,
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], str(mfa))
            self.assertEqual(calls[1][:3], [str(python), "-m", "montreal_forced_aligner.command_line.mfa"])
            self.assertEqual(aligned.words, (AlignedInterval("hello", 0.1, 0.5),))

    def test_parse_textgrid_word_and_phone_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.TextGrid"
            path.write_text(
                """
File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0.1
        xmax = 0.5
        text = "hello"
    intervals [2]:
        xmin = 0.5
        xmax = 0.9
        text = ""
item [2]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0.1
        xmax = 0.2
        text = "HH"
""",
                encoding="utf-8",
            )
            parsed = parse_textgrid(path)
            self.assertEqual(parsed.words, (AlignedInterval("hello", 0.1, 0.5),))
            self.assertEqual(parsed.phones, (AlignedInterval("HH", 0.1, 0.2),))

    def test_apply_mfa_word_alignment_updates_segment_and_word_times(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[TranscriptSegment("hello world", speaker="one")],
        )
        aligned = apply_mfa_word_alignment(
            result,
            (
                AlignedInterval("hello", 0.1, 0.4),
                AlignedInterval("world", 0.6, 1.0),
            ),
        )
        segment = aligned.segments[0]
        self.assertEqual(segment.start, 0.1)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.words[0].text, "hello")
        self.assertEqual(segment.words[0].start, 0.1)
        self.assertEqual(segment.words[1].end, 1.0)

    def test_apply_mfa_word_alignment_keeps_unaligned_broad_words(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "okay let us see you can hear it",
                    start=0.0,
                    end=3.0,
                    speaker="one",
                    words=(
                        WordToken("okay", 0.0, 0.4, "one"),
                        WordToken("let", 0.4, 0.7, "one"),
                        WordToken("us", 0.7, 0.9, "one"),
                        WordToken("see", 0.9, 1.1, "one"),
                        WordToken("you", 1.1, 1.3, "one"),
                        WordToken("can", 1.3, 1.5, "one"),
                        WordToken("hear", 1.5, 1.8, "one"),
                        WordToken("it", 1.8, 2.0, "one"),
                    ),
                )
            ],
        )
        aligned = apply_mfa_word_alignment(
            result,
            (
                AlignedInterval("okay", 0.05, 0.35),
                AlignedInterval("let", 0.45, 0.65),
                AlignedInterval("us", 0.70, 0.85),
            ),
        )
        segment = aligned.segments[0]
        self.assertEqual([word.text for word in segment.words], ["okay", "let", "us", "see", "you", "can", "hear", "it"])
        self.assertEqual(segment.words[0].start, 0.05)
        self.assertEqual(segment.words[3].text, "see")
        self.assertEqual(segment.words[3].start, 0.9)
        self.assertEqual(segment.end, 3.0)

    def test_phone_tier_lines_use_sp_labels_and_phone_symbols(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("hello", start=0.0, end=0.8, speaker="one"),
                TranscriptSegment("world", start=1.0, end=1.6, speaker="two"),
            ],
        )
        self.assertEqual(
            phone_tier_lines(
                result,
                (
                    AlignedInterval("h", 0.0, 0.1),
                    AlignedInterval("\u0259", 0.1, 0.3),
                    AlignedInterval("w", 1.0, 1.1),
                    AlignedInterval("\u025d", 1.1, 1.4),
                ),
            ),
            [
                "[00:00] SP1: h \u0259",
                "[00:01] SP2: w \u025d",
            ],
        )


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


class ProjectWorkflowTests(unittest.TestCase):
    def _sample_result(self, source: Path) -> TranscriptResult:
        return TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "maybe pause",
                    start=0.0,
                    end=0.8,
                    speaker="one",
                    words=(
                        WordToken("maybe", 0.0, 0.2, "one", confidence=0.1),
                        WordToken("pause", 0.4, 0.6, "one", confidence=0.9),
                    ),
                ),
                TranscriptSegment(
                    "same time",
                    start=0.7,
                    end=1.2,
                    speaker="two",
                    words=(
                        WordToken("same", 0.7, 0.9, "two", confidence=0.8),
                        WordToken("time", 0.9, 1.2, "two", confidence=0.8),
                    ),
                ),
            ],
        )

    def test_project_save_load_preserves_nested_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            project = build_project_from_transcript(source, self._sample_result(source), workspace=root / "project", copy_source=False)
            path = save_project(project, root / "project" / "sample.qftproj")

            loaded = load_project(path)
            self.assertEqual(loaded.schema_version, 1)
            self.assertEqual(loaded.recordings[0].segments[0].speaker_label, "SP1")
            self.assertEqual(loaded.recordings[0].segments[0].tokens[0].corrected_text, "maybe")
            self.assertTrue(loaded.recordings[0].segments[0].candidate_annotations)

    def test_project_candidates_include_uncertainty_pause_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "sample.wav"
            source.write_bytes(b"audio")
            result = self._sample_result(source)
            project = build_project_from_transcript(source, result, workspace=Path(folder) / "project", copy_source=False)

            kinds = {suggestion.kind for suggestion in iter_project_suggestions(project, status=None)}
            self.assertIn("uncertain_word", kinds)
            self.assertIn("inline_pause", kinds)
            self.assertIn("overlap", kinds)
            self.assertIn("overlap", {suggestion.kind for suggestion in suggestions_for_transcript(result)})

    def test_pending_candidates_are_not_final_until_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "sample.wav"
            source.write_bytes(b"audio")
            project = build_project_from_transcript(source, self._sample_result(source), workspace=Path(folder) / "project", copy_source=False)
            overlap = next(suggestion for suggestion in iter_project_suggestions(project, status=None) if suggestion.kind == "overlap")

            self.assertFalse(any("[ ]" in line for line in render_confirmed_jefferson_lines(project)))
            confirmed = set_candidate_status(project, overlap.id, "confirmed")
            self.assertTrue(any("[ ]" in line for line in render_confirmed_jefferson_lines(confirmed)))
            rejected = set_candidate_status(confirmed, overlap.id, "rejected")
            self.assertFalse(any("[ ]" in line for line in render_confirmed_jefferson_lines(rejected)))
            self.assertGreaterEqual(len(rejected.edit_history), 2)

    def test_project_exports_are_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            project = build_project_from_transcript(source, self._sample_result(source), workspace=root / "project", copy_source=False)

            txt = export_project_txt(project, root / "out.txt")
            docx = export_project_docx(project, root / "out.docx")
            pdf = export_project_pdf(project, root / "out.pdf")
            self.assertTrue(txt.read_text(encoding="utf-8").startswith("1"))
            self.assertEqual(docx.read_bytes()[:2], b"PK")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))

    def test_offline_guard_rejects_remote_sources_and_blocks_python_sockets(self) -> None:
        reject_remote_source(Path(r"C:\local\sample.wav"))
        reject_remote_source(r"C:\local\sample.wav")
        with self.assertRaises(ValueError):
            reject_remote_source("https://example.com/audio.wav")
        with offline_processing_guard():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    def test_energy_vad_detects_local_speech_region(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            wav_path = Path(folder) / "vad.wav"
            sample_rate = 16000
            samples: list[int] = []
            for _ in range(int(0.2 * sample_rate)):
                samples.append(0)
            for index in range(int(0.3 * sample_rate)):
                sample = 0.15 * math.sin(2 * math.pi * 220.0 * (index / sample_rate))
                samples.append(int(sample * 32767))
            for _ in range(int(0.2 * sample_rate)):
                samples.append(0)
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))

            regions = energy_vad(wav_path)
            self.assertEqual(len(regions), 1)
            self.assertGreaterEqual(regions[0].start, 0.18)
            self.assertLessEqual(regions[0].end, 0.55)


class BatchOutputTests(unittest.TestCase):
    def test_batch_processor_writes_only_selected_visible_rtf(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"

            def run(self, command, _log_callback, _cancelled) -> int:
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str) -> None:
                return None

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio_path, source_path, _work_dir, _language_code, _log_callback, _cancelled):
                return TranscriptResult(
                    source_path=source_path,
                    language="en",
                    segments=[
                        TranscriptSegment(
                            "hello there",
                            start=0.0,
                            end=0.8,
                            speaker="one",
                            words=(WordToken("hello", 0.0, 0.3, "one"), WordToken("there", 0.3, 0.8, "one")),
                        )
                    ],
                )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.wav"
            whisper = root / "whisper"
            model = root / "model.bin"
            for path in (source, whisper, model):
                path.write_bytes(b"local")
            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=BROAD_JEFFERSONIAN_TRANSCRIPTION,
            )
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)

            with patch("transcription.batch_processor.WhisperCppEngine", FakeEngine), patch(
                "transcription.batch_processor.apply_local_acoustic_annotations",
                side_effect=AssertionError("Broad Jeffersonian should skip acoustic voice-quality analysis."),
            ), patch(
                "transcription.batch_processor.run_mfa_alignment",
                side_effect=AssertionError("Broad Jeffersonian should skip MFA alignment."),
            ):
                worker = TranscriptionBatchProcessor([record], options, FakeRunner())
                worker.run()

            output_dir = root / "QuickFixTranscription"
            visible_outputs = sorted(path.name for path in output_dir.iterdir() if path.is_file())
            self.assertEqual(visible_outputs, ["sample_broad_jeffersonian.rtf"])
            self.assertFalse((output_dir / ".tmp").exists())


class ModelSetupTests(unittest.TestCase):
    def test_mfa_preset_path_lookup_uses_selected_model_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            acoustic_dir = root / ".tools" / "mfa" / "root" / "pretrained_models" / "acoustic"
            dictionary_dir = root / ".tools" / "mfa" / "root" / "pretrained_models" / "dictionary"
            acoustic_dir.mkdir(parents=True)
            dictionary_dir.mkdir(parents=True)
            acoustic = acoustic_dir / "english_mfa.zip"
            dictionary = dictionary_dir / "english_uk_mfa.dict"
            acoustic.write_text("model", encoding="utf-8")
            dictionary.write_text("dictionary", encoding="utf-8")

            with patch.object(transcription_dependencies, "quickfix_app_roots", return_value=[root]):
                self.assertEqual(transcription_dependencies.find_mfa_acoustic_model("english_mfa"), str(acoustic))
                self.assertEqual(transcription_dependencies.find_mfa_dictionary("english_uk_mfa"), str(dictionary))

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

    def test_transcription_options_modes_select_broad_and_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            for path in (whisper, model):
                path.write_text("local", encoding="utf-8")

            broad = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=BROAD_JEFFERSONIAN_TRANSCRIPTION,
            )
            self.assertEqual(broad.selected_mode(), BROAD_JEFFERSONIAN_TRANSCRIPTION)
            self.assertFalse(broad.needs_mfa_alignment)

            narrow = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=NARROW_JEFFERSONIAN_TRANSCRIPTION,
            )
            self.assertEqual(narrow.selected_mode(), NARROW_JEFFERSONIAN_TRANSCRIPTION)
            self.assertTrue(narrow.needs_mfa_alignment)
            with self.assertRaises(ValueError):
                narrow.validate()

    def test_transcription_options_validate_local_mfa_assets_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            mfa = root / "mfa"
            acoustic_model = root / "english_mfa.zip"
            dictionary = root / "english.dict"
            for path in (whisper, model, mfa, acoustic_model, dictionary):
                path.write_text("local", encoding="utf-8")

            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                use_mfa_alignment=True,
                mfa_executable=str(mfa),
                mfa_acoustic_model=str(acoustic_model),
                mfa_dictionary=str(dictionary),
            )
            self.assertEqual(options.validate(), (None, None))

    def test_transcription_options_reject_missing_mfa_dictionary_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            mfa = root / "mfa"
            acoustic_model = root / "english_mfa.zip"
            for path in (whisper, model, mfa, acoustic_model):
                path.write_text("local", encoding="utf-8")

            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                use_mfa_alignment=True,
                mfa_executable=str(mfa),
                mfa_acoustic_model=str(acoustic_model),
                mfa_dictionary=str(root / "missing.dict"),
            )
            with self.assertRaises(ValueError):
                options.validate()

    def test_transcription_options_require_mfa_for_phone_tier_export(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            whisper.write_text("exe", encoding="utf-8")
            model.write_text("model", encoding="utf-8")
            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                export_mfa_phone_transcript=True,
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
