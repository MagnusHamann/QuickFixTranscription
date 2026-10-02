from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from transcription.danish_whisper_engine import DanishWhisperBatchTranscriber
from transcription.danish_whisper_setup import DANISH_WHISPER_REQUIRED_PATHS, danish_whisper_model_is_ready
from transcription.dote_diarization import _map_and_merge_turns, _raw_segments, _read_canonical_wav
from transcription.dote_engine import DoteWhisperEngine, _segments_for_turn
from transcription.models import ASR_BACKEND_DANISH_WHISPER, TranscriptResult, TranscriptSegment, WordToken
from transcription.offline_guard import OFFLINE_ENVIRONMENT
from transcription.pyannote_diarization import DiarizationResult, DiarizationTurn


class DotePipelineTests(unittest.TestCase):
    def test_danish_whisper_model_readiness_requires_every_local_asset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in DANISH_WHISPER_REQUIRED_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"local")
            self.assertTrue(danish_whisper_model_is_ready(root))
            (root / DANISH_WHISPER_REQUIRED_PATHS[-1]).unlink()
            self.assertFalse(danish_whisper_model_is_ready(root))

    def test_danish_whisper_worker_uses_isolated_python_and_offline_cpu_environment(self) -> None:
        class FakeProcess:
            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

            def terminate(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "turn.wav"
            audio.write_bytes(b"local audio")
            isolated_python = root / "danish-python.exe"
            isolated_python.write_bytes(b"local executable")
            (root / "danish_whisper_output.json").write_text(
                '{"device":"cpu","results":[{"language":"da","segments":[{"text":"hej","start":0.1,"end":0.4,"words":[{"text":"hej","start":0.1,"end":0.4,"confidence":0.9}]}]}]}',
                encoding="utf-8",
            )
            transcriber = DanishWhisperBatchTranscriber(str(root / "model"), prefer_gpu=False)

            with patch("transcription.danish_whisper_engine.danish_whisper_python_executable", return_value=isolated_python), patch(
                "transcription.danish_whisper_engine.subprocess.Popen", return_value=FakeProcess()
            ) as popen:
                result = transcriber.transcribe([audio], root, lambda _message: None, lambda: False)

        self.assertEqual(result[0].text, "hej")
        self.assertEqual(result[0].segments[0].words[0].confidence, 0.9)
        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(command[0], str(isolated_python))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        for key, value in OFFLINE_ENVIRONMENT.items():
            self.assertEqual(environment[key], value)

    def test_canonical_wav_loader_has_its_numpy_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            wav_path = Path(folder) / "canonical.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x01\x00" * 1600)

            samples, duration = _read_canonical_wav(wav_path)

        self.assertEqual(samples.shape, (1600,))
        self.assertAlmostEqual(duration, 0.1)

    def test_installed_sherpa_result_shape_uses_sorted_segment_return_value(self) -> None:
        expected = ({"start": 0.0, "end": 1.0, "speaker": 0},)

        class SherpaResultShape:
            def sort_by_start_time(self):
                return list(expected)

        self.assertEqual(tuple(_raw_segments(SherpaResultShape())), expected)

    def test_speaker_mapping_is_stable_by_first_appearance(self) -> None:
        turns = _map_and_merge_turns(
            (
                {"start": 0.0, "end": 0.4, "speaker": "speaker-b"},
                {"start": 0.2, "end": 0.6, "speaker": "speaker-a"},
                {"start": 0.7, "end": 1.0, "speaker": "speaker-b"},
            )
        )

        self.assertEqual([turn.speaker for turn in turns], ["SP1", "SP2", "SP1"])
        self.assertEqual((turns[1].start, turns[1].end), (0.2, 0.6))

    def test_turn_words_are_shifted_clipped_and_speaker_labelled(self) -> None:
        raw = TranscriptResult(
            Path("source.wav"),
            "en",
            [
                TranscriptSegment(
                    "hello there",
                    0.0,
                    1.0,
                    words=(
                        WordToken("hello", 0.0, 0.4, confidence=0.9),
                        WordToken("there", 0.4, 1.0, confidence=0.8),
                    ),
                )
            ],
        )

        segments = _segments_for_turn(raw, 4.7, 5.0, 5.5, "SP2")

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker, "SP2")
        self.assertEqual([word.speaker for word in segments[0].words], ["SP2", "SP2"])
        self.assertEqual((segments[0].words[0].start, segments[0].words[-1].end), (5.0, 5.5))

    def test_engine_uses_one_diarization_result_and_no_context_whisper(self) -> None:
        class FakeDiarizer:
            def terminate(self) -> None:
                return None

            def diarize(self, *_args):
                turns = (
                    DiarizationTurn(0.0, 0.8, "SP1"),
                    DiarizationTurn(0.8, 1.6, "SP2"),
                )
                return DiarizationResult(turns, turns, turns)

        class FakeWhisper:
            no_context_values: list[bool] = []
            calls = 0

            def __init__(self, _executable, _model, _gpu, *, no_context=False):
                self.no_context_values.append(no_context)

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio, source, _work, _language, _log, _cancelled):
                self.__class__.calls += 1
                return TranscriptResult(
                    source,
                    "en",
                    [TranscriptSegment("hello", 0.3, 0.5, words=(WordToken("hello", 0.3, 0.5),))],
                )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav_path = root / "canonical.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 32000)

            engine = DoteWhisperEngine("whisper.exe", "model.bin")
            engine.diarizer = FakeDiarizer()
            with patch("transcription.dote_engine.WhisperCppEngine", FakeWhisper):
                result = engine.transcribe(wav_path, wav_path, root, "en", lambda _message: None, lambda: False)

        self.assertEqual(FakeWhisper.no_context_values, [True])
        self.assertEqual(FakeWhisper.calls, 2)
        self.assertEqual([segment.speaker for segment in result.segments], ["SP1", "SP2"])

    def test_danish_engine_batches_turns_once_and_uses_native_word_timing(self) -> None:
        class FakeDiarizer:
            def terminate(self) -> None:
                return None

            def diarize(self, *_args):
                turns = (
                    DiarizationTurn(0.0, 0.8, "SP1"),
                    DiarizationTurn(0.8, 1.6, "SP2"),
                )
                return DiarizationResult(turns, turns, turns)

        class FakeDanishWhisper:
            calls = 0
            audio_count = 0

            def terminate(self) -> None:
                return None

            def transcribe(self, audio_paths, *_args):
                type(self).calls += 1
                type(self).audio_count = len(audio_paths)
                return [
                    TranscriptResult(
                        audio_paths[0],
                        "da",
                        [TranscriptSegment("Goddag", 0.3, 0.5, words=(WordToken("Goddag", 0.3, 0.5),))],
                    ),
                    TranscriptResult(
                        audio_paths[1],
                        "da",
                        [TranscriptSegment("Tak", 0.3, 0.5, words=(WordToken("Tak", 0.3, 0.5),))],
                    ),
                ]

        class FakeWhisper:
            calls = 0
            languages: list[str] = []

            def __init__(self, *_args, **_kwargs):
                raise AssertionError("whisper.cpp must not run for the Røst v3 backend")

            def terminate(self) -> None:
                return None

            def transcribe(self, _audio, source, _work, language, _log, _cancelled):
                type(self).calls += 1
                type(self).languages.append(language)
                return TranscriptResult(
                    source,
                    language,
                    [TranscriptSegment("timing", 0.3, 0.5, words=(WordToken("timing", 0.3, 0.5),))],
                )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav_path = root / "canonical.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 32000)

            engine = DoteWhisperEngine(
                "whisper.exe",
                "model.bin",
                asr_backend=ASR_BACKEND_DANISH_WHISPER,
                danish_model_path=str(root / "roest"),
            )
            engine.diarizer = FakeDiarizer()
            engine.danish_whisper = FakeDanishWhisper()
            with patch("transcription.dote_engine.WhisperCppEngine", FakeWhisper):
                result = engine.transcribe(wav_path, wav_path, root, "", lambda _message: None, lambda: False)

        self.assertEqual(FakeDanishWhisper.calls, 1)
        self.assertEqual(FakeDanishWhisper.audio_count, 2)
        self.assertEqual(FakeWhisper.calls, 0)
        self.assertEqual(FakeWhisper.languages, [])
        self.assertEqual([segment.text for segment in result.segments], ["Goddag", "Tak"])
        self.assertEqual([segment.speaker for segment in result.segments], ["SP1", "SP2"])
        self.assertEqual(result.language, "da")


if __name__ == "__main__":
    unittest.main()
