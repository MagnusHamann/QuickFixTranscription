"""Core tests for QuickFixTranscription."""

from __future__ import annotations

import json
import importlib.util
import math
import io
import os
import socket
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from main import configure_qt_runtime

from ffmpeg.ffmpeg_runner import FFmpegRunner
from quickfix_sibling_apps import quickfix_app_roots
from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.acoustic_analysis import _merge_wrapped_stretches
from transcription.batch_processor import TranscriptionBatchProcessor
from transcription.diagnostics.diarization import diagnose_diarization
from transcription.cpu.vad import energy_vad
from transcription.engine import WhisperCppEngine, parse_whisper_json
from transcription.event_engine import suggestions_for_transcript
from transcription.file_utils import collect_media_files, output_directory_for, temp_directory_for, unique_output_path
from transcription.languages import LANGUAGE_CHOICES
from transcription.jeffersonian import BROAD_JEFFERSONIAN_PROFILE, NARROW_JEFFERSONIAN_PROFILE, format_simple_jeffersonian
from transcription.media import build_audio_extract_command, build_channel_extract_command
from transcription.mfa_alignment import AlignedInterval, apply_mfa_word_alignment, parse_textgrid, phone_tier_lines
from transcription.mfa_presets import DEFAULT_SETUP_MFA_PRESET_IDS, MFA_PRESETS, mfa_preset_by_id, preset_for_language_code
from transcription.models import (
    ASR_BACKEND_SAGA_2_M,
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
from transcription.pyannote_diarization import (
    DiarizationResult,
    DiarizationTurn,
    _UtteranceSpeakerCandidate,
    _resolve_utterance_speaker_candidates,
    apply_diarization_to_transcript,
    run_pyannote_diarization,
    run_pyannote_diarization_isolated,
)
from transcription.preflight import build_preflight_report, cloud_sync_label, model_profile_for_path
from transcription import cache as transcription_cache
from transcription import dependencies as transcription_dependencies
from transcription import mfa_alignment as transcription_mfa_alignment
from transcription import mfa_setup as transcription_mfa_setup
from transcription.dependencies import DependencyStatus
from transcription.rtf_exporter import RTF_OUTPUT_VERSION_MARKER, rtf_escape, transcript_lines, write_rtf
from transcription.time_utils import validate_time_range


PYANNOTE_RUNTIME_AVAILABLE = bool(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("pyannote") is not None
)
requires_pyannote_runtime = unittest.skipUnless(
    PYANNOTE_RUNTIME_AVAILABLE,
    "optional legacy pyannote/Torch runtime is not installed",
)


class TimeRangeTests(unittest.TestCase):
    def test_optional_start_and_finish_are_validated(self) -> None:
        self.assertEqual(validate_time_range("01:00", "02:30"), (60, 150))
        self.assertEqual(validate_time_range("", "02:30"), (None, 150))
        self.assertEqual(validate_time_range("01:00", ""), (60, None))

    def test_finish_must_be_after_start(self) -> None:
        with self.assertRaises(ValueError):
            validate_time_range("02:00", "01:59")


class SagaOptionsTests(unittest.TestCase):
    def test_saga_options_require_danish_and_a_ready_local_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper.exe"
            model = root / "model.bin"
            saga = root / "saga"
            whisper.write_bytes(b"local")
            model.write_bytes(b"local")
            saga.mkdir()
            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                asr_backend=ASR_BACKEND_SAGA_2_M,
                saga_model_path=str(saga),
                language_code="da",
            )

            with patch("transcription.saga_setup.saga_model_is_ready", return_value=True), patch(
                "transcription.saga_setup.saga_runtime_is_ready", return_value=True
            ):
                self.assertEqual(options.validate(), (None, None))

            wrong_language = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                asr_backend=ASR_BACKEND_SAGA_2_M,
                saga_model_path=str(saga),
                language_code="en",
            )
            with self.assertRaisesRegex(ValueError, "Danish only"):
                wrong_language.validate()

    def test_cache_key_changes_between_dote_and_saga(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.wav"
            whisper = root / "whisper.exe"
            model = root / "model.bin"
            saga = root / "saga"
            saga.mkdir()
            for path in (source, whisper, model, saga / "model.safetensors"):
                path.write_bytes(b"local")
            dote = TranscriptionOptions(str(whisper), str(model))
            saga_options = TranscriptionOptions(
                str(whisper),
                str(model),
                asr_backend=ASR_BACKEND_SAGA_2_M,
                saga_model_path=str(saga),
                language_code="da",
            )

            self.assertNotEqual(
                transcription_cache.cache_key("dote_base", source, dote),
                transcription_cache.cache_key("dote_base", source, saga_options),
            )


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


def _create_valid_pyannote_pipeline(root: Path) -> Path:
    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "config.yaml").write_text("pipeline: local", encoding="utf-8")
    for child_name in ("embedding", "plda", "segmentation"):
        (pipeline_dir / child_name).mkdir()
    return pipeline_dir


def _write_valid_test_wav(path: Path, duration_seconds: float = 0.1, sample_rate: int = 16000) -> None:
    frame_count = max(1, int(sample_rate * duration_seconds))
    frames = bytearray()
    for index in range(frame_count):
        sample = int(0.15 * math.sin(2 * math.pi * 220.0 * (index / sample_rate)) * 32767)
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


class PyannoteDiarizationTests(unittest.TestCase):
    def test_pyannote_missing_pipeline_is_a_hard_failure(self) -> None:
        options = TranscriptionOptions(whisper_executable="whisper", model_path="model.bin")
        with patch("transcription.pyannote_diarization._resolve_pipeline_path", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "No complete local pyannote pipeline"):
                run_pyannote_diarization(Path("sample.wav"), options, lambda _message: None, lambda: False)

    @requires_pyannote_runtime
    def test_pyannote_diarization_targets_two_speakers_by_default(self) -> None:
        class FakePyannoteResult:
            def __init__(self, call_kwargs: dict[str, object]) -> None:
                self.call_kwargs = call_kwargs

            def itertracks(self, yield_label: bool = True):
                del yield_label
                yield types.SimpleNamespace(start=0.0, end=1.0), None, "speaker_a"
                yield types.SimpleNamespace(start=1.0, end=2.0), None, "speaker_b"

        class FakePipeline:
            last_call_kwargs: dict[str, object] = {}
            last_audio_input: object = None
            last_device: object = None

            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, device: object) -> None:
                if isinstance(device, str):
                    raise TypeError("pyannote requires torch.device, not str")
                FakePipeline.last_device = device

            def __call__(self, audio_input: object, **kwargs):
                FakePipeline.last_audio_input = audio_input
                FakePipeline.last_call_kwargs = dict(kwargs)
                return FakePyannoteResult(dict(kwargs))

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            result = TranscriptResult(
                source_path=wav,
                language="en",
                segments=[
                    TranscriptSegment(
                        "hello there thanks",
                        start=0.0,
                        end=2.0,
                        speaker="fallback",
                        words=(
                            WordToken("hello", 0.0, 0.5, "fallback"),
                            WordToken("there", 0.5, 1.1, "fallback"),
                            WordToken("thanks", 1.1, 1.8, "fallback"),
                        ),
                    )
                ],
            )
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
            )

            pyannote_module = types.ModuleType("pyannote")
            pyannote_audio = types.ModuleType("pyannote.audio")
            pyannote_audio.Pipeline = FakePipeline
            pyannote_module.audio = pyannote_audio

            with patch("transcription.pyannote_diarization._load_pyannote_pipeline", return_value=FakePipeline):
                turns = run_pyannote_diarization(wav, options, logs.append, lambda: False)

        self.assertEqual(len(turns), 2)
        self.assertEqual(sorted({turn.speaker for turn in turns}), ["pyannote_1", "pyannote_2"])
        self.assertIsInstance(FakePipeline.last_audio_input, dict)
        self.assertIn("waveform", FakePipeline.last_audio_input)
        self.assertIn("sample_rate", FakePipeline.last_audio_input)
        self.assertEqual(FakePipeline.last_call_kwargs.get("num_speakers"), 2)
        self.assertEqual(str(FakePipeline.last_device), "cpu")
        self.assertTrue(any("Loading local pyannote.audio module." in line for line in logs))
        self.assertTrue(any("Loading local pyannote pipeline from" in line for line in logs))
        self.assertTrue(any("Canonical pyannote audio:" in line for line in logs))
        self.assertTrue(any("Exact pyannote input:" in line for line in logs))
        self.assertTrue(any("num_speakers=2" in line for line in logs))

    @requires_pyannote_runtime
    def test_pyannote_diarization_honors_known_speakers_hint(self) -> None:
        class FakePyannoteResult:
            def __init__(self, call_kwargs: dict[str, object]) -> None:
                self.call_kwargs = call_kwargs

            def itertracks(self, yield_label: bool = True):
                del yield_label
                yield types.SimpleNamespace(start=0.0, end=1.0), None, "speaker_a"
                yield types.SimpleNamespace(start=1.0, end=2.0), None, "speaker_b"

        class FakePipeline:
            last_call_kwargs: dict[str, object] = {}

            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                FakePipeline.last_call_kwargs = dict(kwargs)
                assert isinstance(audio_input, dict)
                return FakePyannoteResult(dict(kwargs))

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            result = TranscriptResult(
                source_path=wav,
                language="en",
                segments=[
                    TranscriptSegment(
                        "hello there thanks",
                        start=0.0,
                        end=2.0,
                        speaker="fallback",
                        words=(
                            WordToken("hello", 0.0, 0.5, "fallback"),
                            WordToken("there", 0.5, 1.1, "fallback"),
                            WordToken("thanks", 1.1, 1.8, "fallback"),
                        ),
                    )
                ],
            )
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
                known_speakers=3,
            )

            pyannote_module = types.ModuleType("pyannote")
            pyannote_audio = types.ModuleType("pyannote.audio")
            pyannote_audio.Pipeline = FakePipeline
            pyannote_module.audio = pyannote_audio

            with patch("transcription.pyannote_diarization._load_pyannote_pipeline", return_value=FakePipeline):
                turns = run_pyannote_diarization(wav, options, logs.append, lambda: False)

        self.assertEqual(len(turns), 2)
        self.assertEqual(sorted({turn.speaker for turn in turns}), ["pyannote_1", "pyannote_2"])
        self.assertEqual(FakePipeline.last_call_kwargs.get("num_speakers"), 3)
        self.assertTrue(any("num_speakers=3" in line for line in logs))

    def test_pyannote_isolated_worker_forwards_stderr_trace(self) -> None:
        class FakeProcess:
            def __init__(self, command: list[str], **_kwargs) -> None:
                self.returncode = 0
                self.stderr = io.StringIO("Canonical pyannote audio:\nExact pyannote input:\n")
                result_path = Path(command[command.index("--result-path") + 1])
                result_path.write_text(
                    json.dumps(
                        {
                            "attribution_turns": [
                                {"start": 0.0, "end": 1.0, "speaker": "pyannote_1"}
                            ],
                            "overlap_turns": [],
                            "utterance_turns": [
                                {
                                    "start": 0.0,
                                    "end": 1.0,
                                    "speaker": "pyannote_1",
                                    "uncertain": True,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            def poll(self) -> int:
                return 0

            def terminate(self) -> None:
                return None

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
            )

            with patch("transcription.pyannote_diarization.subprocess.Popen", side_effect=lambda command, **kwargs: FakeProcess(command, **kwargs)):
                turns = run_pyannote_diarization_isolated(wav, options, logs.append, lambda: False)

        self.assertEqual([turn.speaker for turn in turns], ["pyannote_1"])
        self.assertEqual(len(turns.utterance_turns), 1)
        self.assertTrue(turns.utterance_turns[0].uncertain)
        self.assertTrue(any("Canonical pyannote audio:" in line for line in logs))
        self.assertTrue(any("Exact pyannote input:" in line for line in logs))

    @requires_pyannote_runtime
    def test_pyannote_diarization_retries_with_wider_speaker_range_when_one_speaker_found(self) -> None:
        class FakePyannoteResult:
            def __init__(self, speaker_labels: tuple[str, ...]) -> None:
                self.speaker_labels = speaker_labels

            def itertracks(self, yield_label: bool = True):
                del yield_label
                if len(self.speaker_labels) == 1:
                    yield types.SimpleNamespace(start=0.0, end=3.0), None, self.speaker_labels[0]
                    return
                yield types.SimpleNamespace(start=0.0, end=1.6), None, self.speaker_labels[0]
                yield types.SimpleNamespace(start=1.6, end=3.2), None, self.speaker_labels[1]

        class FakePipeline:
            call_kwargs_history: list[dict[str, object]] = []

            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                FakePipeline.call_kwargs_history.append(dict(kwargs))
                assert isinstance(audio_input, dict)
                if kwargs.get("num_speakers") == 2:
                    return FakePyannoteResult(("speaker_a",))
                if kwargs.get("min_speakers") == 2 and kwargs.get("max_speakers") == 3:
                    return FakePyannoteResult(("speaker_a", "speaker_b"))
                return FakePyannoteResult(("speaker_a",))

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            result = TranscriptResult(
                source_path=wav,
                language="en",
                segments=[
                    TranscriptSegment(
                        "hello there thanks",
                        start=0.0,
                        end=3.2,
                        speaker="fallback",
                        words=(
                            WordToken("hello", 0.0, 0.5, "fallback"),
                            WordToken("there", 1.1, 1.5, "fallback"),
                            WordToken("thanks", 2.0, 2.6, "fallback"),
                        ),
                    )
                ],
            )
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
            )

            pyannote_module = types.ModuleType("pyannote")
            pyannote_audio = types.ModuleType("pyannote.audio")
            pyannote_audio.Pipeline = FakePipeline
            pyannote_module.audio = pyannote_audio

            with patch("transcription.pyannote_diarization._load_pyannote_pipeline", return_value=FakePipeline):
                turns = run_pyannote_diarization(wav, options, logs.append, lambda: False)
                diarized = apply_diarization_to_transcript(result, turns)

        self.assertEqual(sorted({segment.speaker for segment in diarized.segments if segment.speaker}), ["pyannote_1", "pyannote_2"])
        self.assertGreaterEqual(len(FakePipeline.call_kwargs_history), 2)
        self.assertEqual(FakePipeline.call_kwargs_history[0].get("num_speakers"), 2)
        self.assertEqual(FakePipeline.call_kwargs_history[1].get("min_speakers"), 2)
        self.assertEqual(FakePipeline.call_kwargs_history[1].get("max_speakers"), 3)
        self.assertTrue(any("wider speaker range" in line for line in logs))

    def test_pyannote_smooths_isolated_speaker_blips_into_surrounding_turn(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "hello there can you hear me",
                    start=0.0,
                    end=1.5,
                    speaker="fallback",
                    words=(
                        WordToken("hello", 0.0, 0.2, "fallback"),
                        WordToken("there", 0.2, 0.5, "fallback"),
                        WordToken("can", 0.5, 0.7, "fallback"),
                        WordToken("you", 0.7, 0.9, "fallback"),
                        WordToken("hear", 0.9, 1.2, "fallback"),
                        WordToken("me", 1.2, 1.5, "fallback"),
                    ),
                )
            ],
        )

        turns = (
            DiarizationTurn(0.0, 0.55, "pyannote_1"),
            DiarizationTurn(0.55, 0.68, "pyannote_2"),
            DiarizationTurn(0.68, 1.5, "pyannote_1"),
        )

        diarized = apply_diarization_to_transcript(result, turns)

        speakers = [segment.speaker for segment in diarized.segments if segment.speaker]
        self.assertEqual(speakers, ["pyannote_1"])
        self.assertTrue(all(word.speaker == "pyannote_1" for segment in diarized.segments for word in segment.words))

    def test_pyannote_preserves_a_real_speaker_change_when_it_is_sustained(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "yes no maybe right",
                    start=0.0,
                    end=1.2,
                    speaker="fallback",
                    words=(
                        WordToken("yes", 0.0, 0.3, "fallback"),
                        WordToken("no", 0.3, 0.55, "fallback"),
                        WordToken("maybe", 0.55, 0.85, "fallback"),
                        WordToken("right", 0.85, 1.2, "fallback"),
                    ),
                )
            ],
        )

        turns = (
            DiarizationTurn(0.0, 0.55, "pyannote_1"),
            DiarizationTurn(0.55, 1.2, "pyannote_2"),
        )

        diarized = apply_diarization_to_transcript(result, turns)

        speakers = [segment.speaker for segment in diarized.segments if segment.speaker]
        self.assertEqual(speakers, ["pyannote_1", "pyannote_2"])

    def test_pyannote_preserves_whisper_utterances_and_bridges_short_unlabelled_gaps(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "how are you",
                    start=0.0,
                    end=1.0,
                    words=(
                        WordToken("how", 0.0, 0.3),
                        WordToken("are", 0.3, 0.6),
                        WordToken("you", 0.6, 1.0),
                    ),
                ),
                TranscriptSegment(
                    "fine thank you",
                    start=1.1,
                    end=2.0,
                    words=(
                        WordToken("fine", 1.1, 1.35),
                        WordToken("thank", 1.35, 1.7),
                        WordToken("you", 1.7, 2.0),
                    ),
                ),
            ],
        )
        turns = (
            DiarizationTurn(0.0, 0.85, "pyannote_1"),
            DiarizationTurn(1.3, 2.0, "pyannote_2"),
        )

        diarized = apply_diarization_to_transcript(result, turns)

        spoken = [segment for segment in diarized.segments if segment.text != "(     )"]
        self.assertEqual([segment.text for segment in spoken], ["how are you", "fine thank you"])
        self.assertEqual([segment.speaker for segment in spoken], ["pyannote_1", "pyannote_2"])
        self.assertTrue(all(word.speaker for segment in spoken for word in segment.words))

    def test_pyannote_uses_utterance_consensus_for_boundary_jitter(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "you are doing really well",
                    start=0.0,
                    end=3.0,
                    words=(
                        WordToken("you", 0.0, 0.5),
                        WordToken("are", 0.5, 1.0),
                        WordToken("doing", 1.0, 1.5),
                        WordToken("really", 1.5, 2.2),
                        WordToken("well", 2.2, 3.0),
                    ),
                )
            ],
        )
        turns = (
            DiarizationTurn(0.0, 0.9, "pyannote_1"),
            DiarizationTurn(0.9, 1.15, "pyannote_2"),
            DiarizationTurn(1.15, 3.0, "pyannote_1"),
        )

        diarized = apply_diarization_to_transcript(result, turns)

        spoken = [segment for segment in diarized.segments if segment.text != "(     )"]
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0].speaker, "pyannote_1")
        self.assertTrue(all(word.speaker == "pyannote_1" for word in spoken[0].words))

    def test_acoustic_utterance_assignment_refines_coarse_turn_timeline(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "hello",
                    start=0.0,
                    end=1.0,
                    words=(WordToken("hello", 0.0, 1.0),),
                ),
                TranscriptSegment(
                    "how are you",
                    start=1.0,
                    end=2.0,
                    words=(
                        WordToken("how", 1.0, 1.3),
                        WordToken("are", 1.3, 1.6),
                        WordToken("you", 1.6, 2.0),
                    ),
                ),
            ],
        )
        evidence = DiarizationResult(
            attribution_turns=(DiarizationTurn(0.0, 2.0, "pyannote_1"),),
            overlap_turns=(DiarizationTurn(0.0, 2.0, "pyannote_1"),),
            utterance_turns=(
                DiarizationTurn(0.0, 1.0, "pyannote_1"),
                DiarizationTurn(1.0, 2.0, "pyannote_2"),
            ),
        )

        diarized = apply_diarization_to_transcript(result, evidence)

        self.assertEqual(
            [segment.speaker for segment in diarized.segments],
            ["pyannote_1", "pyannote_2"],
        )

    def test_borderline_utterance_uses_matching_confident_neighbours_and_stays_uncertain(self) -> None:
        candidates = [
            _UtteranceSpeakerCandidate(0.0, 1.0, "pyannote_1", 0.8, 0.5, True),
            _UtteranceSpeakerCandidate(1.1, 2.0, "pyannote_2", 0.31, 0.02, False),
            _UtteranceSpeakerCandidate(2.1, 3.0, "pyannote_1", 0.75, 0.4, True),
        ]
        timeline = (
            DiarizationTurn(0.0, 1.0, "pyannote_1"),
            DiarizationTurn(1.1, 2.0, "pyannote_2"),
            DiarizationTurn(2.1, 3.0, "pyannote_1"),
        )

        resolved = _resolve_utterance_speaker_candidates(candidates, timeline)

        self.assertEqual([turn.speaker for turn in resolved], ["pyannote_1", "pyannote_1", "pyannote_1"])
        self.assertFalse(resolved[0].uncertain)
        self.assertTrue(resolved[1].uncertain)
        self.assertFalse(resolved[2].uncertain)

    def test_borderline_utterance_does_not_alternate_when_neighbours_disagree(self) -> None:
        candidates = [
            _UtteranceSpeakerCandidate(0.0, 1.0, "pyannote_1", 0.8, 0.5, True),
            _UtteranceSpeakerCandidate(1.1, 2.0, "pyannote_2", 0.31, 0.02, False),
            _UtteranceSpeakerCandidate(2.1, 3.0, "pyannote_3", 0.75, 0.4, True),
        ]
        timeline = (DiarizationTurn(1.1, 2.0, "pyannote_2"),)

        resolved = _resolve_utterance_speaker_candidates(candidates, timeline)

        self.assertEqual(resolved[1].speaker, "pyannote_2")
        self.assertTrue(resolved[1].uncertain)

    def test_overlap_contaminated_embedding_uses_clean_neighbours_and_stays_uncertain(self) -> None:
        candidates = [
            _UtteranceSpeakerCandidate(0.0, 1.0, "pyannote_1", 0.8, 0.5, True),
            _UtteranceSpeakerCandidate(1.0, 2.0, "pyannote_2", 0.9, 0.6, True, overlapping=True),
            _UtteranceSpeakerCandidate(2.0, 3.0, "pyannote_1", 0.75, 0.4, True),
        ]
        timeline = (
            DiarizationTurn(0.0, 1.0, "pyannote_1"),
            DiarizationTurn(1.0, 2.0, "pyannote_2"),
            DiarizationTurn(2.0, 3.0, "pyannote_1"),
        )

        resolved = _resolve_utterance_speaker_candidates(candidates, timeline)

        self.assertEqual([turn.speaker for turn in resolved], ["pyannote_1", "pyannote_1", "pyannote_1"])
        self.assertTrue(resolved[1].uncertain)

    def test_uncertain_utterance_assignment_marks_transcript_segment(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "maybe this speaker",
                    start=0.0,
                    end=1.0,
                    words=(
                        WordToken("maybe", 0.0, 0.4),
                        WordToken("this", 0.4, 0.7),
                        WordToken("speaker", 0.7, 1.0),
                    ),
                )
            ],
        )
        evidence = DiarizationResult(
            attribution_turns=(DiarizationTurn(0.0, 1.0, "pyannote_1"),),
            overlap_turns=(DiarizationTurn(0.0, 1.0, "pyannote_1"),),
            utterance_turns=(DiarizationTurn(0.0, 1.0, "pyannote_1", uncertain=True),),
        )

        diarized = apply_diarization_to_transcript(result, evidence)

        self.assertEqual(diarized.segments[0].speaker, "pyannote_1")
        self.assertTrue(diarized.segments[0].speaker_uncertain)

    def test_overlap_affected_utterance_uses_word_timeline_instead_of_flattening_speakers(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "hello yes",
                    start=0.0,
                    end=2.0,
                    words=(
                        WordToken("hello", 0.0, 0.8),
                        WordToken("yes", 1.2, 2.0),
                    ),
                )
            ],
        )
        evidence = DiarizationResult(
            attribution_turns=(
                DiarizationTurn(0.0, 1.0, "pyannote_1"),
                DiarizationTurn(1.0, 2.0, "pyannote_2"),
            ),
            overlap_turns=(
                DiarizationTurn(0.0, 1.3, "pyannote_1"),
                DiarizationTurn(0.8, 2.0, "pyannote_2"),
            ),
            utterance_turns=(DiarizationTurn(0.0, 2.0, "pyannote_1", uncertain=True),),
        )

        diarized = apply_diarization_to_transcript(result, evidence)
        spoken = [segment for segment in diarized.segments if segment.text != "(     )"]

        self.assertEqual([segment.speaker for segment in spoken], ["pyannote_1", "pyannote_2"])
        self.assertTrue(all(segment.speaker_uncertain for segment in spoken))

    def test_confident_utterance_embedding_does_not_flatten_a_sustained_speaker_change(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "anything worse Oh yeah I've probably",
                    start=7.0,
                    end=15.1,
                    words=(
                        WordToken("anything", 7.37, 11.0),
                        WordToken("worse", 11.17, 11.63),
                        WordToken("Oh", 12.0, 12.48),
                        WordToken("yeah", 12.62, 13.42),
                        WordToken("I've", 13.92, 14.0),
                        WordToken("probably", 14.05, 15.02),
                    ),
                )
            ],
        )
        evidence = DiarizationResult(
            attribution_turns=(
                DiarizationTurn(7.0, 11.8, "pyannote_1"),
                DiarizationTurn(11.8, 15.1, "pyannote_2"),
            ),
            overlap_turns=(
                DiarizationTurn(7.0, 11.8, "pyannote_1"),
                DiarizationTurn(11.8, 15.1, "pyannote_2"),
            ),
            utterance_turns=(DiarizationTurn(7.0, 15.1, "pyannote_1"),),
        )

        diarized = apply_diarization_to_transcript(result, evidence)

        self.assertEqual([segment.speaker for segment in diarized.segments], ["pyannote_1", "pyannote_2"])
        self.assertEqual([segment.text for segment in diarized.segments], ["anything worse", "Oh yeah I've probably"])
        self.assertTrue(all(segment.speaker_uncertain for segment in diarized.segments))

    def test_pyannote_overlay_retags_words_and_adds_missing_overlap_placeholder(self) -> None:
        source = Path("sample.wav")
        result = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "hello there",
                    start=0.0,
                    end=0.9,
                    speaker="asr",
                    words=(
                        WordToken("hello", 0.0, 0.4, "asr"),
                        WordToken("there", 0.5, 0.9, "asr"),
                    ),
                )
            ],
        )
        diarized = apply_diarization_to_transcript(
            result,
            (
                DiarizationTurn(0.0, 0.9, "pyannote_1"),
                DiarizationTurn(0.3, 0.8, "pyannote_2"),
            ),
        )

        self.assertTrue(has_cross_speaker_overlap(diarized))
        self.assertEqual(diarized.segments[0].speaker, "pyannote_1")
        self.assertIn(
            TranscriptSegment(
                "(     )",
                start=0.3,
                end=0.8,
                speaker="pyannote_2",
                speaker_uncertain=True,
            ),
            diarized.segments,
        )
        lines = format_simple_jeffersonian(diarized, profile=BROAD_JEFFERSONIAN_PROFILE)
        self.assertEqual(lines[0].index("["), lines[1].index("["))

    @requires_pyannote_runtime
    def test_pyannote_diarization_unwraps_diarizeoutput_wrapper(self) -> None:
        class FakeAnnotation:
            def __init__(self, turns: list[tuple[float, float, str]]) -> None:
                self.turns = turns

            def __len__(self) -> int:
                return len(self.turns)

            def itertracks(self, yield_label: bool = True):
                del yield_label
                for start, end, speaker in self.turns:
                    yield types.SimpleNamespace(start=start, end=end), None, speaker

        class FakeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = FakeAnnotation([(0.0, 1.0, "speaker_a")])
                self.exclusive_speaker_diarization = FakeAnnotation(
                    [(0.0, 0.8, "speaker_a"), (0.8, 1.6, "speaker_b")]
                )
                self.speaker_embeddings = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                assert isinstance(audio_input, dict)
                self.call_kwargs = dict(kwargs)
                return FakeOutput()

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
            )

            with patch("transcription.pyannote_diarization._load_pyannote_pipeline", return_value=FakePipeline):
                turns = run_pyannote_diarization(wav, options, logs.append, lambda: False)

        self.assertEqual([turn.speaker for turn in turns], ["pyannote_1", "pyannote_2"])
        self.assertEqual(len(turns.overlap_turns), 1)
        self.assertTrue(any("pyannote raw output: FakeOutput" in line for line in logs))

    @requires_pyannote_runtime
    def test_pyannote_diarization_falls_back_to_speaker_annotation_when_exclusive_is_empty(self) -> None:
        class FakeAnnotation:
            def __init__(self, turns: list[tuple[float, float, str]]) -> None:
                self.turns = turns

            def __len__(self) -> int:
                return len(self.turns)

            def itertracks(self, yield_label: bool = True):
                del yield_label
                for start, end, speaker in self.turns:
                    yield types.SimpleNamespace(start=start, end=end), None, speaker

        class FakeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = FakeAnnotation([(0.0, 1.0, "speaker_a"), (1.0, 2.0, "speaker_b")])
                self.exclusive_speaker_diarization = FakeAnnotation([])
                self.speaker_embeddings = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                assert isinstance(audio_input, dict)
                self.call_kwargs = dict(kwargs)
                return FakeOutput()

        logs: list[str] = []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "sample.wav"
            _write_valid_test_wav(wav)
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            options = TranscriptionOptions(
                whisper_executable=str(root / "whisper"),
                model_path=str(root / "model.bin"),
                pyannote_pipeline_path=str(pipeline_dir),
            )

            with patch("transcription.pyannote_diarization._load_pyannote_pipeline", return_value=FakePipeline):
                turns = run_pyannote_diarization(wav, options, logs.append, lambda: False)

        self.assertEqual([turn.speaker for turn in turns], ["pyannote_1", "pyannote_2"])
        self.assertEqual(turns.attribution_turns, turns.overlap_turns)

    def test_pyannote_uses_exclusive_turns_for_words_and_regular_turns_for_overlap(self) -> None:
        source = Path("sample.wav")
        transcript = TranscriptResult(
            source_path=source,
            language="en",
            segments=[
                TranscriptSegment(
                    "hello there",
                    start=0.0,
                    end=1.0,
                    words=(WordToken("hello", 0.0, 0.4), WordToken("there", 0.5, 1.0)),
                )
            ],
        )
        evidence = DiarizationResult(
            attribution_turns=(DiarizationTurn(0.0, 1.0, "pyannote_1"),),
            overlap_turns=(
                DiarizationTurn(0.0, 1.0, "pyannote_1"),
                DiarizationTurn(0.4, 0.8, "pyannote_2"),
            ),
        )

        diarized = apply_diarization_to_transcript(transcript, evidence)

        spoken = [segment for segment in diarized.segments if segment.text != "(     )"]
        self.assertTrue(all(segment.speaker == "pyannote_1" for segment in spoken))
        self.assertIn(
            TranscriptSegment(
                "(     )",
                start=0.4,
                end=0.8,
                speaker="pyannote_2",
                speaker_uncertain=True,
            ),
            diarized.segments,
        )


class WhisperCppEngineTests(unittest.TestCase):
    def test_gpu_preference_adds_no_gpu_flag_only_when_disabled(self) -> None:
        commands: list[list[str]] = []

        class FakeProcess:
            returncode = 0

            def __init__(self, command: list[str], **_kwargs) -> None:
                commands.append(command)
                output_base = Path(command[command.index("-of") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "result": {"language": "en"},
                            "transcription": [
                                {
                                    "text": "hello",
                                    "timestamps": {"from": "00:00:00.000", "to": "00:00:00.500"},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            def poll(self) -> int:
                return 0

            def wait(self) -> None:
                return None

            def terminate(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / "whisper"
            model = root / "model.bin"
            audio = root / "sample.wav"
            work_dir = root / "work"
            work_dir.mkdir()
            for path in (executable, model, audio):
                path.write_bytes(b"local")

            with patch("transcription.engine.subprocess.Popen", FakeProcess):
                cpu_engine = WhisperCppEngine(str(executable), str(model), prefer_gpu=False)
                cpu_engine.transcribe(audio, audio, work_dir, "", lambda _text: None, lambda: False)
                gpu_engine = WhisperCppEngine(str(executable), str(model), prefer_gpu=True)
                gpu_engine.transcribe(audio, audio, work_dir, "", lambda _text: None, lambda: False)

        self.assertIn("-ng", commands[0])
        self.assertNotIn("-ng", commands[1])


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

    def test_temp_directory_uses_local_app_temp_not_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "meeting.wav"
            source.write_bytes(b"audio")
            local_temp = root / "local-temp"
            with patch.object(transcription_cache, "local_temp_root", return_value=local_temp):
                temp_dir = temp_directory_for(source)
            self.assertEqual(temp_dir.parent, local_temp)
            self.assertNotEqual(temp_dir.parent, output_directory_for(source) / ".tmp")


class CacheTests(unittest.TestCase):
    def test_transcript_cache_roundtrip_is_local_and_structured(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache_root = root / "cache"
            source = root / "sample.wav"
            whisper = root / "whisper"
            model = root / "ggml-base.bin"
            for path in (source, whisper, model):
                path.write_bytes(b"local")
            options = TranscriptionOptions(whisper_executable=str(whisper), model_path=str(model))
            result = TranscriptResult(
                source_path=source,
                language="en",
                segments=[
                    TranscriptSegment(
                        "hello",
                        start=0.0,
                        end=0.5,
                        speaker="one",
                        words=(WordToken("hello", 0.0, 0.5, "one", 0.9),),
                    )
                ],
            )
            with patch.object(transcription_cache, "local_cache_root", return_value=cache_root):
                key = transcription_cache.cache_key("whisper", source, options)
                transcription_cache.save_cached_transcript("whisper", key, result)
                loaded = transcription_cache.load_cached_transcript("whisper", key, source)

            self.assertEqual(loaded, result)
            self.assertTrue(any(cache_root.rglob("*.json")))

    def test_transcript_cache_preserves_speaker_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache_root = root / "cache"
            source = root / "sample.wav"
            whisper = root / "whisper"
            model = root / "ggml-base.bin"
            for path in (source, whisper, model):
                path.write_bytes(b"local")
            options = TranscriptionOptions(whisper_executable=str(whisper), model_path=str(model))
            result = TranscriptResult(
                source_path=source,
                language="en",
                segments=[
                    TranscriptSegment(
                        "please review",
                        start=0.0,
                        end=1.0,
                        speaker="one",
                        speaker_uncertain=True,
                    )
                ],
            )
            with patch.object(transcription_cache, "local_cache_root", return_value=cache_root):
                key = transcription_cache.cache_key("broad", source, options)
                transcription_cache.save_cached_transcript("broad", key, result)
                loaded = transcription_cache.load_cached_transcript("broad", key, source)

            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.segments[0].speaker_uncertain)

    def test_cached_whisper_pause_marker_is_removed_without_retranscribing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache_root = root / "cache"
            source = root / "sample.wav"
            whisper = root / "whisper"
            model = root / "ggml-base.bin"
            for path in (source, whisper, model):
                path.write_bytes(b"local")
            options = TranscriptionOptions(whisper_executable=str(whisper), model_path=str(model))
            result = TranscriptResult(
                source_path=source,
                language="en",
                segments=[
                    TranscriptSegment(
                        "[Pause]",
                        start=0.0,
                        end=10.0,
                        words=(WordToken("P", 1.4, 1.4), WordToken("ause", 1.4, 8.5)),
                    ),
                    TranscriptSegment("Hello", start=10.0, end=11.0, words=(WordToken("Hello", 10.0, 11.0),)),
                ],
            )
            with patch.object(transcription_cache, "local_cache_root", return_value=cache_root):
                key = transcription_cache.cache_key("whisper", source, options)
                transcription_cache.save_cached_transcript("whisper", key, result)
                loaded = transcription_cache.load_cached_transcript("whisper", key, source)

            self.assertIsNotNone(loaded)
            self.assertEqual([segment.text for segment in loaded.segments], ["Hello"])


class PreflightTests(unittest.TestCase):
    class FakeRunner:
        ready = True

    def _ready_status(self, root: Path) -> tuple[DependencyStatus, TranscriptionOptions]:
        ffmpeg = root / "ffmpeg"
        ffprobe = root / "ffprobe"
        whisper = root / "whisper"
        model = root / "ggml-base.bin"
        dote = root / "DOTE-whisper.exe"
        segmentation = root / "segmentation.onnx"
        embedding = root / "embedding.onnx"
        pipeline = _create_valid_pyannote_pipeline(root)
        for path in (ffmpeg, ffprobe, whisper, model, dote, segmentation, embedding):
            path.write_bytes(b"local")
        status = DependencyStatus(
            ffmpeg_path=str(ffmpeg),
            ffprobe_path=str(ffprobe),
            whisper_path=str(whisper),
            model_path=str(model),
            pyannote_pipeline_path=str(pipeline),
            dote_application_path=str(dote),
            dote_segmentation_model_path=str(segmentation),
            dote_embedding_model_path=str(embedding),
            sherpa_onnx_ready=True,
        )
        options = TranscriptionOptions(
            whisper_executable=str(whisper),
            model_path=str(model),
            pyannote_pipeline_path=str(pipeline),
        )
        return status, options

    def test_cloud_synced_paths_warn_or_error_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status, options = self._ready_status(root)
            source = root / "OneDrive" / "sample.wav"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)

            warning_report = build_preflight_report([record], options, self.FakeRunner(), status)
            self.assertTrue(any(issue.severity == "warning" and "OneDrive" in issue.message for issue in warning_report.issues))

            strict_options = TranscriptionOptions(
                whisper_executable=options.whisper_executable,
                model_path=options.model_path,
                pyannote_pipeline_path=options.pyannote_pipeline_path,
                block_cloud_synced_paths=True,
            )
            error_report = build_preflight_report([record], strict_options, self.FakeRunner(), status)
            self.assertTrue(any(issue.severity == "error" and "OneDrive" in issue.message for issue in error_report.issues))
            self.assertEqual(cloud_sync_label(source), "OneDrive")

    def test_model_profile_detection_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status, options = self._ready_status(root)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)
            options = TranscriptionOptions(
                whisper_executable=options.whisper_executable,
                model_path=options.model_path,
                model_profile="highest accuracy",
            )

            self.assertEqual(model_profile_for_path(options.model_path), "fast draft")
            report = build_preflight_report([record], options, self.FakeRunner(), status)
            self.assertTrue(any("Requested 'highest accuracy'" in issue.message for issue in report.warnings))


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
                "SP1:        hello",
                "            (0.2)",
                "SP2:        [again]",
                "SP1:        [same time]",
            ],
        )

    def test_simple_formatter_only_shows_speaker_tags_on_changes(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("alpha", start=0.0, end=0.4, speaker="one"),
                TranscriptSegment("beta", start=0.5, end=0.8, speaker="one"),
                TranscriptSegment("gamma", start=0.9, end=1.2, speaker="two"),
                TranscriptSegment("delta", start=1.3, end=1.6, speaker="two"),
                TranscriptSegment("epsilon", start=1.7, end=2.0, speaker="one"),
            ],
        )
        lines = format_simple_jeffersonian(result)
        self.assertTrue(lines[0].startswith("SP1:"))
        self.assertTrue(lines[1].startswith("            "))
        self.assertTrue(lines[2].startswith("SP2:"))
        self.assertTrue(lines[3].startswith("            "))
        self.assertTrue(lines[4].startswith("SP1:"))

    def test_jeffersonian_marks_uncertain_speaker_even_when_speaker_is_repeated(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("certain", start=0.0, end=0.5, speaker="one"),
                TranscriptSegment(
                    "please review",
                    start=0.6,
                    end=1.0,
                    speaker="one",
                    speaker_uncertain=True,
                ),
            ],
        )

        lines = format_simple_jeffersonian(result)

        self.assertTrue(lines[0].startswith("SP1:"))
        self.assertTrue(lines[1].startswith("SP1 (?):"))

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
        self.assertEqual(lines[0], "SP1:        What a good idea to mark [overlapping speech]")
        self.assertEqual(lines[1], "SP2:                                 [this type of speech]")
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
            ["SP1:        What a nice idea to mark (0.2) silence in talk"],
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
                "SP1:        one two three four five six seven eight nine ten",
                "            eleven twelve thirteen",
            ],
        )
        for line in lines:
            text = line[12:]
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
                "SP1:        one two three four five",
                "            six seven eight",
            ],
        )
        for line in lines:
            self.assertLessEqual(len(line[12:]), 25)

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
                "SP1:        one two three four",
                "            five six",
                "            (0.4) new turn",
                "            same speaker",
            ],
        )

    def test_jeffersonian_skips_placeholder_and_punctuation_only_segments(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("[pause]", start=0.0, end=0.2, speaker="one"),
                TranscriptSegment(".", start=0.2, end=0.3, speaker="two"),
                TranscriptSegment("Hello", start=0.3, end=0.8, speaker="two"),
            ],
        )
        self.assertEqual(format_simple_jeffersonian(result), ["SP1:        Hello"])

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
                "SP1:        right=",
                "SP2:        =yes",
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
            ["SP1:        Hello what happened Let's see"],
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
            ["SP1:        I I was going to to say eh no I mean wo- we should"],
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
                "SP1:        ((cough)) ((clears throat)) .snih. ((sigh)) .mt. .dt. .hhh hhh eh eh uhm hm mhm (     )"
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
            ["SP1:        (maybe) clear"],
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
            ["SP1:        word fast cough"],
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
            ["SP1:        maybe clear"],
        )
        self.assertEqual(
            format_simple_jeffersonian(result, profile=NARROW_JEFFERSONIAN_PROFILE),
            ["SP1:        (maybe) clear"],
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
                ["SP1:        ni3 hao3 shi4 jie4"],
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
                ["SP1:        ni3 hao3"],
            )

    def test_jeffersonian_rtf_can_skip_title_and_use_monospace(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.rtf"
            write_rtf(path, "Ignored title", ["SP1:        hello"], font_name="Courier New", include_title=False)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Courier New", content)
            self.assertNotIn("Ignored title", content)
            self.assertIn("SP1:        hello", content)

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
                "SP1: first",
                "SP2: second",
                "SP1: third",
            ],
        )

    def test_regular_transcript_lines_only_show_speaker_tags_when_the_speaker_changes(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("first", start=0.0, end=0.5, speaker="alpha"),
                TranscriptSegment("second", start=0.6, end=1.0, speaker="alpha"),
                TranscriptSegment("third", start=1.1, end=1.5, speaker="beta"),
                TranscriptSegment("fourth", start=1.6, end=2.0, speaker="beta"),
                TranscriptSegment("fifth", start=2.1, end=2.5, speaker="alpha"),
            ],
        )
        lines = transcript_lines(result)
        self.assertIn("SP1: first", lines[0])
        self.assertNotIn("SP1:", lines[1])
        self.assertNotIn("SP2:", lines[1])
        self.assertIn("SP2: third", lines[2])
        self.assertNotIn("SP2:", lines[3])
        self.assertIn("SP1: fifth", lines[4])

    def test_regular_transcript_marks_uncertain_speaker_even_when_speaker_is_repeated(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("certain", start=0.0, end=0.5, speaker="alpha"),
                TranscriptSegment(
                    "please review",
                    start=0.6,
                    end=1.0,
                    speaker="alpha",
                    speaker_uncertain=True,
                ),
            ],
        )

        self.assertEqual(
            transcript_lines(result),
            ["SP1: certain", "SP1 (?): please review"],
        )

    def test_regular_transcript_marks_untranscribed_speech_as_unclear(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "I heard [inaudible] there",
                    start=0.0,
                    end=1.0,
                    speaker="alpha",
                ),
                TranscriptSegment(
                    "(     )",
                    start=1.0,
                    end=1.5,
                    speaker="beta",
                    speaker_uncertain=True,
                ),
            ],
        )

        self.assertEqual(
            transcript_lines(result),
            ["SP1: I heard (unclear) there", "SP2 (?): (unclear)"],
        )

    def test_regular_transcript_keeps_low_confidence_best_guess_visible(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "example",
                    start=0.0,
                    end=0.5,
                    speaker="alpha",
                    words=(WordToken("example", 0.0, 0.5, confidence=0.1),),
                )
            ],
        )

        self.assertEqual(transcript_lines(result), ["SP1: example"])

    def test_regular_transcript_lines_default_to_sp1_without_diarization(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[TranscriptSegment("single speaker text")],
        )
        self.assertEqual(transcript_lines(result), ["SP1: single speaker text"])

    def test_regular_transcript_lines_insert_30_second_timepoint_markers(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("first", start=0.0, end=2.0, speaker="alpha"),
                TranscriptSegment("second", start=31.0, end=32.0, speaker="beta"),
                TranscriptSegment("third", start=61.0, end=62.0, speaker="alpha"),
            ],
        )
        self.assertEqual(
            transcript_lines(result),
            [
                "SP1: first",
                "0.00.30",
                "SP2: second",
                "0.01.00",
                "SP1: third",
            ],
        )

    def test_regular_transcript_lines_render_unclear_placeholder_and_keep_timepoints(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment(
                    "( )",
                    start=29.8,
                    end=31.2,
                    speaker="pyannote_2",
                    speaker_uncertain=True,
                ),
                TranscriptSegment("after", start=31.2, end=32.0, speaker="alpha"),
            ],
        )
        self.assertEqual(
            transcript_lines(result),
            [
                "SP1 (?): (unclear)",
                "0.00.30",
                "SP2: after",
            ],
        )

    def test_regular_transcript_lines_skip_punctuation_only_and_bracketed_placeholder_segments(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("[pause]", start=0.0, end=0.2, speaker="alpha"),
                TranscriptSegment("Hello", start=29.8, end=30.2, speaker="alpha"),
                TranscriptSegment(".", start=30.2, end=31.0, speaker="beta"),
                TranscriptSegment("there", start=31.0, end=31.5, speaker="beta"),
            ],
        )
        self.assertEqual(
            transcript_lines(result),
            [
                "SP1: Hello",
                "0.00.30",
                "SP2: there",
            ],
        )

    def test_regular_transcript_lines_skip_split_blank_audio_placeholders(self) -> None:
        result = TranscriptResult(
            source_path=Path("sample.wav"),
            language="en",
            segments=[
                TranscriptSegment("BLANK", start=0.0, end=0.2, speaker="alpha", words=(WordToken("BLANK", 0.0, 0.2),)),
                TranscriptSegment("AUDIO", start=0.2, end=0.4, speaker="beta", words=(WordToken("AUDIO", 0.2, 0.4),)),
                TranscriptSegment("Hello", start=0.5, end=1.0, speaker="alpha", words=(WordToken("Hello", 0.5, 1.0),)),
            ],
        )
        self.assertEqual(transcript_lines(result), ["SP1: Hello"])


class WhisperJsonTests(unittest.TestCase):
    def test_parse_whisper_cpp_json_reconstructs_bpe_subwords_and_contractions(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:15.000", "to": "00:00:18.000"},
                                "text": "peppermint capsules, but I've",
                                "tokens": [
                                    {"text": " pepp", "offsets": {"from": 15000, "to": 15100}, "p": 0.25},
                                    {"text": "erm", "offsets": {"from": 15100, "to": 15300}, "p": 0.40},
                                    {"text": "int", "offsets": {"from": 15300, "to": 15600}, "p": 0.96},
                                    {"text": " caps", "offsets": {"from": 15600, "to": 16000}, "p": 0.79},
                                    {"text": "ules", "offsets": {"from": 16000, "to": 16400}, "p": 0.99},
                                    {"text": ",", "offsets": {"from": 16400, "to": 16500}, "p": 0.99},
                                    {"text": " but", "offsets": {"from": 16500, "to": 16800}, "p": 0.90},
                                    {"text": " I", "offsets": {"from": 16800, "to": 17000}, "p": 0.95},
                                    {"text": "'ve", "offsets": {"from": 17000, "to": 17300}, "p": 0.80},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_whisper_json(json_path, Path("source.wav"))

        words = parsed.segments[0].words
        self.assertEqual([word.text for word in words], ["peppermint", "capsules", "but", "I've"])
        self.assertEqual((words[0].start, words[0].end), (15.0, 15.6))
        self.assertEqual(words[0].confidence, 0.25)
        self.assertEqual(transcript_lines(parsed), ["SP1: peppermint capsules but I've"])

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

    def test_parse_whisper_cpp_json_skips_punctuation_only_word_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:01,000", "to": "00:00:02,500"},
                                "text": "hello .",
                                "words": [
                                    {
                                        "timestamps": {"from": "00:00:01,000", "to": "00:00:01,400"},
                                        "word": "hello",
                                        "probability": 0.9,
                                    },
                                    {
                                        "timestamps": {"from": "00:00:01,400", "to": "00:00:01,600"},
                                        "word": ".",
                                        "probability": 0.9,
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_whisper_json(json_path, Path("source.wav"))
            self.assertEqual([word.text for word in parsed.segments[0].words], ["hello"])

    def test_parse_whisper_cpp_json_skips_blank_audio_marker_segments(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                                "text": "[BLANK_AUDIO_ON_SCREEN]",
                                "words": [
                                    {"word": "[", "offsets": {"from": 0, "to": 10}},
                                    {"word": "BLANK", "offsets": {"from": 10, "to": 300}},
                                    {"word": "_", "offsets": {"from": 300, "to": 320}},
                                    {"word": "AUDIO", "offsets": {"from": 320, "to": 600}},
                                    {"word": "_", "offsets": {"from": 600, "to": 620}},
                                    {"word": "on", "offsets": {"from": 620, "to": 760}},
                                    {"word": "screen", "offsets": {"from": 760, "to": 980}},
                                    {"word": "]", "offsets": {"from": 980, "to": 1000}},
                                ],
                            },
                            {
                                "timestamps": {"from": "00:00:01,000", "to": "00:00:02,000"},
                                "text": "Hello there",
                                "words": [
                                    {"word": "Hello", "offsets": {"from": 1000, "to": 1400}},
                                    {"word": "there", "offsets": {"from": 1400, "to": 2000}},
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_whisper_json(json_path, Path("source.wav"))
            self.assertEqual([segment.text for segment in parsed.segments], ["Hello there"])

    def test_parse_whisper_cpp_json_skips_pause_marker_before_split_tokens_render(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:00,000", "to": "00:00:10,000"},
                                "text": "[Pause]",
                                "words": [
                                    {"word": "P", "offsets": {"from": 1440, "to": 1440}},
                                    {"word": "ause", "offsets": {"from": 1440, "to": 8530}},
                                ],
                            },
                            {
                                "timestamps": {"from": "00:00:10,000", "to": "00:00:11,000"},
                                "text": "Hello",
                                "words": [{"word": "Hello", "offsets": {"from": 10000, "to": 11000}}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_whisper_json(json_path, Path("source.wav"))

            self.assertEqual([segment.text for segment in parsed.segments], ["Hello"])

    def test_parse_whisper_cpp_json_removes_inline_blank_audio_word_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            json_path.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:00,000", "to": "00:00:02,000"},
                                "text": "Hello [BLANK_AUDIO] there",
                                "words": [
                                    {"word": "Hello", "offsets": {"from": 0, "to": 400}},
                                    {"word": "BLANK", "offsets": {"from": 400, "to": 700}},
                                    {"word": "AUDIO", "offsets": {"from": 700, "to": 1000}},
                                    {"word": "there", "offsets": {"from": 1000, "to": 1600}},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_whisper_json(json_path, Path("source.wav"))
            self.assertEqual([word.text for word in parsed.segments[0].words], ["Hello", "there"])
            self.assertEqual(transcript_lines(parsed), ["SP1: Hello there"])


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
            self.assertTrue(txt.read_text(encoding="utf-8").startswith("SP1"))
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

    def test_offline_guard_disables_pyannote_and_huggingface_telemetry(self) -> None:
        with offline_processing_guard():
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
            self.assertEqual(os.environ["PYANNOTE_METRICS_ENABLED"], "0")
            self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")


class StartupTests(unittest.TestCase):
    def test_qt_runtime_uses_bundled_windows_platform_plugin(self) -> None:
        plugin_dir = configure_qt_runtime()
        if os.name != "nt":
            self.assertIsNone(plugin_dir)
            return
        assert plugin_dir is not None
        self.assertTrue((plugin_dir / "qwindows.dll").exists())
        self.assertEqual(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"], str(plugin_dir))


class BatchOutputTests(unittest.TestCase):
    def test_batch_processor_writes_only_selected_visible_rtf(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def probe(self, _path: Path) -> dict[str, object]:
                return {"audio_channels": 2}

            def run(self, command, _log_callback, _cancelled) -> int:
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
                return None

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio_path, source_path, _work_dir, _language_code, _log_callback, _cancelled):
                return TranscriptResult(
                    source_path=source_path,
                    language="en",
                    segments=[
                        TranscriptSegment(
                            "hello",
                            start=0.0,
                            end=0.4,
                            speaker="one",
                            words=(WordToken("hello", 0.0, 0.4, "one"),),
                        ),
                        TranscriptSegment(
                            "there",
                            start=0.4,
                            end=0.8,
                            speaker="two",
                            words=(WordToken("there", 0.4, 0.8, "two"),),
                        ),
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

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine), patch(
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

    def test_batch_processor_uses_dote_base_for_mono_broad_audio(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def probe(self, _path: Path) -> dict[str, object]:
                return {"audio_channels": 1}

            def run(self, command, _log_callback, _cancelled) -> int:
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
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

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine):
                worker = TranscriptionBatchProcessor([record], options, FakeRunner())
                worker.run()

            output_dir = root / "QuickFixTranscription"
            self.assertTrue((output_dir / "sample_broad_jeffersonian.rtf").exists())

    def test_batch_processor_uses_dote_for_stereo_speaker_separation(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def __init__(self) -> None:
                self.run_calls = 0

            def probe(self, _path: Path) -> dict[str, object]:
                return {"audio_channels": 2}

            def run(self, command, _log_callback, _cancelled) -> int:
                self.run_calls += 1
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
                self.transcribe_calls: list[Path] = []

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio_path, source_path, _work_dir, _language_code, _log_callback, _cancelled):
                return TranscriptResult(
                    source_path=source_path,
                    language="en",
                    segments=[
                        TranscriptSegment(
                            "hello",
                            start=0.0,
                            end=0.4,
                            speaker="one",
                            words=(WordToken("hello", 0.0, 0.4, "one"),),
                        ),
                        TranscriptSegment(
                            "there",
                            start=0.4,
                            end=0.8,
                            speaker="two",
                            words=(WordToken("there", 0.4, 0.8, "two"),),
                        ),
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
            runner = FakeRunner()

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine):
                worker = TranscriptionBatchProcessor([record], options, runner)
                worker.run()

            output_dir = root / "QuickFixTranscription"
            content = (output_dir / "sample_broad_jeffersonian.rtf").read_text(encoding="utf-8")
            self.assertEqual(runner.run_calls, 1)
            self.assertIn("SP1:", content)
            self.assertIn("SP2:", content)

    def test_batch_processor_verbatim_separates_speakers_with_local_diarization(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def probe(self, _path: Path) -> dict[str, object]:
                return {"audio_channels": 1}

            def run(self, command, _log_callback, _cancelled) -> int:
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
                return None

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio_path, source_path, _work_dir, _language_code, _log_callback, _cancelled):
                return TranscriptResult(
                    source_path=source_path,
                    language="en",
                    segments=[
                        TranscriptSegment(
                            "Leanne, how are",
                            start=0.0,
                            end=1.0,
                            speaker="one",
                            words=(
                                WordToken("Leanne,", 0.0, 0.4, "one"),
                                WordToken("how", 0.4, 0.8, "one"),
                                WordToken("are", 0.8, 1.0, "one"),
                            ),
                        ),
                        TranscriptSegment(
                            "you? I'm okay, thanks.",
                            start=1.0,
                            end=2.0,
                            speaker="two",
                            words=(
                                WordToken("you?", 1.0, 1.4, "two"),
                                WordToken("I'm", 1.4, 1.6, "two"),
                                WordToken("okay,", 1.6, 1.8, "two"),
                                WordToken("thanks.", 1.8, 2.0, "two"),
                            ),
                        ),
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
                transcription_mode=VERBATIM_TRANSCRIPTION,
            )
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine):
                worker = TranscriptionBatchProcessor([record], options, FakeRunner())
                worker.run()

            content = (root / "QuickFixTranscription" / "sample_verbatim.rtf").read_text(encoding="utf-8")
            self.assertIn("SP1: Leanne, how are", content)
            self.assertIn("SP2: you? I'm okay, thanks.", content)
            self.assertNotIn("SP1: I'm okay, thanks.", content)

    def test_batch_processor_resume_skips_existing_selected_output(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def run(self, *_args, **_kwargs) -> int:
                raise AssertionError("Resume should skip FFmpeg when selected output already exists.")

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
                return None

            def terminate(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.wav"
            whisper = root / "whisper"
            model = root / "model.bin"
            for path in (source, whisper, model):
                path.write_bytes(b"local")
            output_dir = output_directory_for(source)
            output_dir.mkdir()
            existing = output_dir / "sample_verbatim.rtf"
            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=VERBATIM_TRANSCRIPTION,
                resume_completed=True,
            )
            selected_output_key = transcription_cache.cache_key(
                "selected_output",
                source,
                options,
                {
                    "mode": VERBATIM_TRANSCRIPTION,
                    "jeffersonian_line_width": options.jeffersonian_line_width,
                    "known_speakers": options.known_speakers,
                    "base_pipeline": "dote-1.0.2",
                    "use_ipa_font_regular": options.use_ipa_font_regular,
                    "use_ipa_font_jeffersonian": options.use_ipa_font_jeffersonian,
                },
            )
            write_rtf(existing, "Existing transcript", ["SP1: done"], include_title=False, output_key=selected_output_key)
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine):
                worker = TranscriptionBatchProcessor([record], options, FakeRunner())
                worker.run()

            self.assertIn(selected_output_key, existing.read_text(encoding="utf-8"))
            self.assertIn(RTF_OUTPUT_VERSION_MARKER, existing.read_text(encoding="utf-8"))

    def test_batch_processor_refreshes_existing_output_when_output_settings_change(self) -> None:
        class FakeRunner:
            ffmpeg_path = "ffmpeg"
            ffprobe_path = "ffprobe"

            def __init__(self) -> None:
                self.run_calls = 0

            def probe(self, _path: Path) -> dict[str, object]:
                return {"audio_channels": 1}

            def run(self, command, _log_callback, _cancelled) -> int:
                self.run_calls += 1
                Path(command[-1]).write_bytes(b"wav")
                return 0

            def terminate(self) -> None:
                return None

        class FakeEngine:
            def __init__(self, _executable: str, _model_path: str, _prefer_gpu: bool = True, **_kwargs) -> None:
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

            output_dir = output_directory_for(source)
            output_dir.mkdir()
            existing = output_dir / "sample_verbatim.rtf"

            old_options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=VERBATIM_TRANSCRIPTION,
                resume_completed=True,
                use_ipa_font_regular=False,
            )
            old_key = transcription_cache.cache_key(
                "selected_output",
                source,
                old_options,
                {
                    "mode": VERBATIM_TRANSCRIPTION,
                    "jeffersonian_line_width": old_options.jeffersonian_line_width,
                    "known_speakers": old_options.known_speakers,
                    "base_pipeline": "dote-1.0.2",
                    "use_ipa_font_regular": old_options.use_ipa_font_regular,
                    "use_ipa_font_jeffersonian": old_options.use_ipa_font_jeffersonian,
                },
            )
            write_rtf(existing, "Existing transcript", ["SP1: done"], include_title=False, output_key=old_key)

            new_options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=VERBATIM_TRANSCRIPTION,
                resume_completed=True,
                use_ipa_font_regular=True,
            )
            runner = FakeRunner()
            record = MediaRecord(source, "00:01", "Audio", source.stat().st_size)

            with patch.object(transcription_cache, "local_temp_root", return_value=root / "local-temp"), patch.object(
                transcription_cache,
                "local_cache_root",
                return_value=root / "local-cache",
            ), patch("transcription.batch_processor.DoteWhisperEngine", FakeEngine):
                worker = TranscriptionBatchProcessor([record], new_options, runner)
                worker.run()

            self.assertGreater(runner.run_calls, 0)
            new_key = transcription_cache.cache_key(
                "selected_output",
                source,
                new_options,
                {
                    "mode": VERBATIM_TRANSCRIPTION,
                    "jeffersonian_line_width": new_options.jeffersonian_line_width,
                    "known_speakers": new_options.known_speakers,
                    "base_pipeline": "dote-1.0.2",
                    "use_ipa_font_regular": new_options.use_ipa_font_regular,
                    "use_ipa_font_jeffersonian": new_options.use_ipa_font_jeffersonian,
                },
            )
            self.assertIn(new_key, existing.read_text(encoding="utf-8"))
            self.assertIn(
                RTF_OUTPUT_VERSION_MARKER,
                existing.read_text(encoding="utf-8"),
            )


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

    def test_transcription_options_allow_optional_pyannote_pipeline_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            whisper = root / "whisper"
            model = root / "model.bin"
            for path in (whisper, model):
                path.write_text("local", encoding="utf-8")
            pipeline = _create_valid_pyannote_pipeline(root)

            options = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=BROAD_JEFFERSONIAN_TRANSCRIPTION,
                pyannote_pipeline_path=str(pipeline),
                known_speakers=2,
            )
            self.assertEqual(options.validate(), (None, None))

            verbatim = TranscriptionOptions(
                whisper_executable=str(whisper),
                model_path=str(model),
                transcription_mode=VERBATIM_TRANSCRIPTION,
                pyannote_pipeline_path=str(pipeline),
            )
            self.assertEqual(verbatim.validate(), (None, None))

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


class FFmpegRunnerProbeTests(unittest.TestCase):
    def test_probe_reports_audio_sample_rate_and_codec_name(self) -> None:
        runner = FFmpegRunner.__new__(FFmpegRunner)
        runner.ffprobe_path = "ffprobe"
        payload = {
            "format": {"duration": "12.5"},
            "streams": [
                {
                    "codec_type": "audio",
                    "channels": "2",
                    "sample_rate": "48000",
                    "codec_name": "aac",
                }
            ],
        }

        with patch("ffmpeg.ffmpeg_runner.subprocess.run") as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
            info = FFmpegRunner.probe(runner, Path("sample.mp4"))

        self.assertEqual(info["audio_channels"], 2)
        self.assertEqual(info["audio_sample_rate"], 48000)
        self.assertEqual(info["audio_codec_name"], "aac")


class DiarizationDiagnosticTests(unittest.TestCase):
    @requires_pyannote_runtime
    def test_diagnostic_reports_canonical_audio_and_raw_speakers(self) -> None:
        class FakeAnnotation:
            def __init__(self, turns: list[tuple[float, float, str]]) -> None:
                self.turns = turns

            def __len__(self) -> int:
                return len(self.turns)

            def itertracks(self, yield_label: bool = True):
                del yield_label
                for start, end, speaker in self.turns:
                    yield types.SimpleNamespace(start=start, end=end), None, speaker

        class FakeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = FakeAnnotation([(0.0, 1.0, "SPEAKER_00")])
                self.exclusive_speaker_diarization = FakeAnnotation(
                    [(0.0, 0.8, "SPEAKER_00"), (0.8, 1.6, "SPEAKER_01")]
                )
                self.speaker_embeddings = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                assert isinstance(audio_input, dict)
                self.call_kwargs = dict(kwargs)
                return FakeOutput()

        class FakeRunner:
            def __init__(self, source_probe: dict[str, object]) -> None:
                self.ffmpeg_path = "ffmpeg"
                self.ffprobe_path = "ffprobe"
                self.source_probe = source_probe
                self.commands: list[list[str]] = []

            @property
            def ready(self) -> bool:
                return True

            def probe(self, _path: Path) -> dict[str, object]:
                return dict(self.source_probe)

            def run(self, command: list[str], _log_callback, _cancelled) -> int:
                self.commands.append(command)
                output_path = Path(command[-1])
                with wave.open(str(output_path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x00\x00" * 1600)
                return 0

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.mp3"
            source.write_bytes(b"local")
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            runner = FakeRunner(
                {
                    "duration": 12.5,
                    "audio_channels": 2,
                    "audio_sample_rate": 48000,
                    "audio_codec_name": "mp3",
                }
            )

            with patch("transcription.diagnostics.diarization._load_pyannote_pipeline", return_value=FakePipeline):
                report = diagnose_diarization(source, pipeline_path=str(pipeline_dir), runner=runner)

        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.detected_speakers, ("SPEAKER_00", "SPEAKER_01"))
        self.assertEqual(report.speech_segments, 2)
        self.assertTrue(any(line.startswith("Original:") for line in report.lines))
        self.assertTrue(any("sample_rate=48000" in line for line in report.lines))
        self.assertTrue(any(line.startswith("Canonical diarization audio:") for line in report.lines))
        self.assertTrue(any("Detected speakers: 2" in line for line in report.lines))
        self.assertTrue(any("SPEAKER_00" in line for line in report.lines))
        self.assertTrue(any("SPEAKER_01" in line for line in report.lines))

    @requires_pyannote_runtime
    def test_diagnostic_reports_failure_when_no_speaker_turns_returned(self) -> None:
        class EmptyAnnotation:
            def __len__(self) -> int:
                return 0

            def itertracks(self, yield_label: bool = True):
                del yield_label
                if False:
                    yield None  # pragma: no cover

        class FakeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = EmptyAnnotation()
                self.exclusive_speaker_diarization = EmptyAnnotation()
                self.speaker_embeddings = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _path: str) -> "FakePipeline":
                return cls()

            def to(self, _device: str) -> None:
                return None

            def __call__(self, audio_input: object, **kwargs):
                assert isinstance(audio_input, dict)
                self.call_kwargs = dict(kwargs)
                return FakeOutput()

        class FakeRunner:
            def __init__(self) -> None:
                self.ffmpeg_path = "ffmpeg"
                self.ffprobe_path = "ffprobe"
                self.commands: list[list[str]] = []

            @property
            def ready(self) -> bool:
                return True

            def probe(self, _path: Path) -> dict[str, object]:
                return {
                    "duration": 4.0,
                    "audio_channels": 1,
                    "audio_sample_rate": 16000,
                    "audio_codec_name": "wav",
                }

            def run(self, command: list[str], _log_callback, _cancelled) -> int:
                self.commands.append(command)
                output_path = Path(command[-1])
                with wave.open(str(output_path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x00\x00" * 1600)
                return 0

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.m4a"
            source.write_bytes(b"local")
            pipeline_dir = _create_valid_pyannote_pipeline(root)
            runner = FakeRunner()

            with patch("transcription.diagnostics.diarization._load_pyannote_pipeline", return_value=FakePipeline):
                report = diagnose_diarization(source, pipeline_path=str(pipeline_dir), runner=runner)

        self.assertNotEqual(report.exit_code, 0)
        self.assertEqual(report.speech_segments, 0)
        self.assertEqual(report.detected_speakers, ())
        self.assertTrue(any("✗ Diarization failure" in line for line in report.lines))
        self.assertTrue(any("Speech segments: 0" in line for line in report.lines))


if __name__ == "__main__":
    unittest.main()
