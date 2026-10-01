from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from transcription.dote_diarization import _map_and_merge_turns, _raw_segments, _read_canonical_wav
from transcription.dote_engine import DoteWhisperEngine, _segments_for_turn
from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.pyannote_diarization import DiarizationResult, DiarizationTurn


class DotePipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
