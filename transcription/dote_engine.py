"""Integrated DOTE base transcription: diarization first, then per-turn Whisper."""

from __future__ import annotations

from pathlib import Path
import shutil
import wave
from collections.abc import Callable

from transcription.dote_diarization import DoteDiarizer
from transcription.engine import WhisperCppEngine
from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.overlap_analysis import has_cross_speaker_overlap


DOTE_TURN_PADDING_SECONDS = 0.3


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    if rate <= 0 or frames <= 0:
        raise RuntimeError("The canonical DOTE WAV has zero duration.")
    return frames / rate


def _write_wav_slice(source: Path, destination: Path, start: float, end: float) -> None:
    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        if channels != 1 or sample_width != 2 or sample_rate != 16000:
            raise RuntimeError("DOTE turn slicing received a non-canonical WAV.")
        start_frame = max(0, min(reader.getnframes(), int(round(start * sample_rate))))
        end_frame = max(start_frame, min(reader.getnframes(), int(round(end * sample_rate))))
        reader.setpos(start_frame)
        frames = reader.readframes(end_frame - start_frame)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def _clip_word(word: WordToken, offset: float, turn_start: float, turn_end: float, speaker: str) -> WordToken | None:
    start = word.start + offset if word.start is not None else None
    end = word.end + offset if word.end is not None else None
    if start is not None and end is not None:
        if end <= turn_start or start >= turn_end:
            return None
        start = max(turn_start, start)
        end = min(turn_end, end)
        if end <= start:
            return None
    return WordToken(word.text, start, end, speaker, word.confidence)


def _segments_for_turn(
    result: TranscriptResult,
    slice_start: float,
    turn_start: float,
    turn_end: float,
    speaker: str,
) -> list[TranscriptSegment]:
    output: list[TranscriptSegment] = []
    for segment in result.segments:
        words = tuple(
            clipped
            for word in segment.words
            if (clipped := _clip_word(word, slice_start, turn_start, turn_end, speaker)) is not None
        )
        if words:
            output.append(
                TranscriptSegment(
                    text=" ".join(word.text for word in words),
                    start=words[0].start,
                    end=words[-1].end,
                    speaker=speaker,
                    words=words,
                )
            )
            continue

        segment_start = segment.start + slice_start if segment.start is not None else turn_start
        segment_end = segment.end + slice_start if segment.end is not None else turn_end
        clipped_start = max(turn_start, segment_start)
        clipped_end = min(turn_end, segment_end)
        if segment.text.strip() and clipped_end > clipped_start:
            output.append(
                TranscriptSegment(
                    text=segment.text.strip(),
                    start=clipped_start,
                    end=clipped_end,
                    speaker=speaker,
                )
            )
    return output


class DoteWhisperEngine:
    """Produces the shared base transcript consumed by every QuickFix output mode."""

    def __init__(
        self,
        executable: str,
        model_path: str,
        prefer_gpu: bool = True,
        *,
        known_speakers: int = 0,
    ) -> None:
        self.executable = executable
        self.model_path = model_path
        self.prefer_gpu = prefer_gpu
        self.diarizer = DoteDiarizer(known_speakers=known_speakers)
        self.whisper: WhisperCppEngine | None = None

    def terminate(self) -> None:
        self.diarizer.terminate()
        if self.whisper is not None:
            self.whisper.terminate()

    def transcribe(
        self,
        audio_path: Path,
        source_path: Path,
        work_dir: Path,
        language_code: str,
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> TranscriptResult:
        duration = _wav_duration(audio_path)
        evidence = self.diarizer.diarize(audio_path, work_dir, log_callback, cancelled)
        turns = evidence.overlap_turns
        if not turns:
            raise RuntimeError("DOTE did not produce speaker turns; ASR was not started.")

        self.whisper = WhisperCppEngine(
            self.executable,
            self.model_path,
            self.prefer_gpu,
            no_context=True,
        )
        turn_root = work_dir / "dote_turns"
        if turn_root.exists():
            shutil.rmtree(turn_root, ignore_errors=True)
        turn_root.mkdir(parents=True, exist_ok=True)

        output_segments: list[TranscriptSegment] = []
        detected_language: str | None = language_code or None
        log_callback(
            "DOTE base pipeline: transcribing each diarized speaker turn with local whisper.cpp "
            "and no cross-turn context."
        )
        for index, turn in enumerate(turns, start=1):
            if cancelled():
                raise RuntimeError("DOTE transcription stopped by user.")
            slice_start = max(0.0, turn.start - DOTE_TURN_PADDING_SECONDS)
            slice_end = min(duration, turn.end + DOTE_TURN_PADDING_SECONDS)
            if slice_end <= slice_start:
                continue
            slice_path = turn_root / f"turn_{index:05d}_{turn.speaker}.wav"
            turn_work = turn_root / f"turn_{index:05d}_output"
            turn_work.mkdir(parents=True, exist_ok=True)
            _write_wav_slice(audio_path, slice_path, slice_start, slice_end)
            log_callback(
                f"DOTE ASR turn {index}/{len(turns)}: {turn.speaker} "
                f"{turn.start:.2f}-{turn.end:.2f}s"
            )
            result = self.whisper.transcribe(
                slice_path,
                source_path,
                turn_work,
                language_code,
                log_callback,
                cancelled,
            )
            if result.language and not detected_language:
                detected_language = result.language
            output_segments.extend(
                _segments_for_turn(result, slice_start, turn.start, turn.end, turn.speaker)
            )

        output_segments.sort(
            key=lambda segment: (
                segment.start is None,
                segment.start or 0.0,
                segment.end or 0.0,
                segment.speaker or "",
            )
        )
        if not output_segments:
            raise RuntimeError("DOTE diarization found speech, but whisper.cpp returned no usable transcript text.")

        transcript = TranscriptResult(source_path, detected_language, output_segments)
        speakers = sorted({segment.speaker for segment in output_segments if segment.speaker})
        log_callback(
            f"DOTE base transcript ready: {len(output_segments)} segment(s), {len(speakers)} speaker(s)."
        )
        if has_cross_speaker_overlap(transcript):
            log_callback("DOTE retained simultaneous speaker turns for Broad Jeffersonian overlap notation.")
        return transcript
