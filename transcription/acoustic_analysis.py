"""Local acoustic heuristics for Jeffersonian-style annotation.

These annotations are deliberately conservative. They are not a substitute for
human conversation-analysis judgement, but they add useful local cues when word
timings are available.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.nonword_sounds import normalize_nonword_token


LOUD_RATIO = 1.75
QUIET_RATIO = 0.45
FAST_WORDS_PER_SECOND = 4.2
SLOW_WORDS_PER_SECOND = 1.6
PITCH_RATIO = 1.25
MIN_PITCH_HZ = 70.0
MAX_PITCH_HZ = 500.0
ASR_PUNCTUATION_PATTERN = re.compile(r'(?<!\d)\.(?!\d)|[,?!;"“”‘’…]')
DEGREE = "\N{DEGREE SIGN}"
UP_ARROW = "\N{UPWARDS ARROW}"
DOWN_ARROW = "\N{DOWNWARDS ARROW}"


@dataclass(frozen=True)
class PcmAudio:
    sample_rate: int
    samples: tuple[float, ...]

    @classmethod
    def from_wav(cls, path: Path) -> "PcmAudio":
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if sample_width != 2:
            raise ValueError("Expected 16-bit PCM WAV for acoustic analysis.")

        pcm = array("h")
        pcm.frombytes(frames)
        if sys.byteorder != "little":
            pcm.byteswap()

        if channels > 1:
            mono: list[float] = []
            for index in range(0, len(pcm), channels):
                mono.append(sum(pcm[index : index + channels]) / (channels * 32768.0))
            samples = tuple(mono)
        else:
            samples = tuple(value / 32768.0 for value in pcm)

        return cls(sample_rate=sample_rate, samples=samples)

    def slice(self, start: float | None, end: float | None) -> tuple[float, ...]:
        start_index = 0 if start is None else max(0, int(start * self.sample_rate))
        end_index = len(self.samples) if end is None else min(len(self.samples), int(end * self.sample_rate))
        if end_index <= start_index:
            return ()
        return self.samples[start_index:end_index]


def rms(samples: tuple[float, ...]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def estimate_pitch_hz(samples: tuple[float, ...], sample_rate: int) -> float | None:
    if len(samples) < sample_rate * 0.05:
        return None

    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous <= 0 < sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample

    duration = len(samples) / sample_rate
    if duration <= 0:
        return None

    hz = crossings / (2.0 * duration)
    if MIN_PITCH_HZ <= hz <= MAX_PITCH_HZ:
        return hz
    return None


def _split_text_words(text: str) -> list[str]:
    return [part for part in ASR_PUNCTUATION_PATTERN.sub("", text).split() if part.strip()]


def _ensure_word_tokens(segment: TranscriptSegment) -> tuple[WordToken, ...]:
    if segment.words:
        return segment.words

    parts = _split_text_words(segment.text)
    if not parts:
        return ()

    if segment.start is None or segment.end is None or segment.end <= segment.start:
        return tuple(WordToken(text=part, speaker=segment.speaker) for part in parts)

    duration = (segment.end - segment.start) / len(parts)
    return tuple(
        WordToken(
            text=part,
            start=segment.start + index * duration,
            end=segment.start + (index + 1) * duration,
            speaker=segment.speaker,
        )
        for index, part in enumerate(parts)
    )


def _first_vowel_index(text: str) -> int | None:
    for index, char in enumerate(text):
        if char.lower() in "aeiouy":
            return index
    return None


def _mark_prolonged(text: str) -> str:
    index = _first_vowel_index(text)
    if index is None:
        return text
    return f"{text[: index + 1]}::{text[index + 1 :]}"


def _has_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith((".", ",", "?", "!", ":", ";", "-", "=", "]", ")"))


def _annotate_word(
    word: WordToken,
    word_rms: float,
    median_rms: float,
    duration: float | None,
    median_duration: float | None,
    pitch_start: float | None,
    pitch_end: float | None,
    cutoff: bool,
) -> str:
    nonword = normalize_nonword_token(word.text, word.confidence)
    if nonword is not None:
        return nonword

    text = ASR_PUNCTUATION_PATTERN.sub("", word.text).strip()

    if duration is not None and median_duration is not None:
        if duration >= max(0.55, median_duration * 1.8):
            text = _mark_prolonged(text)

    if median_rms > 0:
        if word_rms >= median_rms * LOUD_RATIO:
            text = text.upper()
        elif 0 < word_rms <= median_rms * QUIET_RATIO:
            text = f"{DEGREE}{text}{DEGREE}"

    if pitch_start and pitch_end:
        if pitch_end / pitch_start >= PITCH_RATIO:
            text = f"{UP_ARROW}{text}"
        elif pitch_start / pitch_end >= PITCH_RATIO:
            text = f"{DOWN_ARROW}{text}"

    if cutoff and not _has_terminal_punctuation(text):
        text = f"{text}-"

    return text


def _merge_wrapped_stretches(words: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word.startswith(DEGREE) and word.endswith(DEGREE) and len(word) >= 2:
            stretch = [word[1:-1]]
            index += 1
            while index < len(words) and words[index].startswith(DEGREE) and words[index].endswith(DEGREE):
                stretch.append(words[index][1:-1])
                index += 1
            merged.append(f"{DEGREE}{' '.join(stretch)}{DEGREE}")
            continue

        if word.startswith(UP_ARROW) or word.startswith(DOWN_ARROW):
            marker = word[0]
            stretch = [word[1:]]
            index += 1
            while index < len(words) and words[index].startswith(marker):
                stretch.append(words[index][1:])
                index += 1
            merged.append(f"{marker}{' '.join(stretch)}")
            continue

        merged.append(word)
        index += 1

    return merged


def _segment_rate(words: tuple[WordToken, ...]) -> float | None:
    timed = [word for word in words if word.start is not None and word.end is not None]
    if len(timed) < 2:
        return None
    start = min(word.start for word in timed if word.start is not None)
    end = max(word.end for word in timed if word.end is not None)
    if end <= start:
        return None
    return len(timed) / (end - start)


def _apply_rate_markers(words: list[str], rate: float | None) -> list[str]:
    if not words or rate is None:
        return words
    marked = list(words)
    if rate >= FAST_WORDS_PER_SECOND:
        marked[0] = f">{marked[0]}"
        marked[-1] = f"{marked[-1]}<"
    elif rate <= SLOW_WORDS_PER_SECOND:
        marked[0] = f"<{marked[0]}"
        marked[-1] = f"{marked[-1]}>"
    return marked


def apply_local_acoustic_annotations(
    audio_path: Path,
    result: TranscriptResult,
    log_callback: Callable[[str], None] | None = None,
) -> TranscriptResult:
    """Return a transcript with local acoustic Jeffersonian cues added."""
    try:
        audio = PcmAudio.from_wav(audio_path)
    except Exception as exc:
        if log_callback:
            log_callback(f"Local acoustic annotation skipped: {exc}")
        return result

    segment_words = [_ensure_word_tokens(segment) for segment in result.segments]
    timed_words = [
        word
        for words in segment_words
        for word in words
        if word.start is not None and word.end is not None and word.end > word.start
    ]
    if not timed_words:
        if log_callback:
            log_callback("Local acoustic annotation skipped: no timed words available.")
        return result

    rms_values = [rms(audio.slice(word.start, word.end)) for word in timed_words]
    duration_values = [word.end - word.start for word in timed_words if word.start is not None and word.end is not None]
    median_rms = statistics.median(value for value in rms_values if value > 0) if any(value > 0 for value in rms_values) else 0.0
    median_duration = statistics.median(duration_values) if duration_values else None

    new_segments: list[TranscriptSegment] = []
    for segment, words in zip(result.segments, segment_words):
        annotated_words: list[str] = []
        for index, word in enumerate(words):
            word_duration = word.end - word.start if word.start is not None and word.end is not None else None
            word_samples = audio.slice(word.start, word.end)
            word_rms = rms(word_samples)
            halfway = None
            if word.start is not None and word.end is not None:
                halfway = word.start + ((word.end - word.start) / 2.0)
            pitch_start = estimate_pitch_hz(audio.slice(word.start, halfway), audio.sample_rate) if halfway else None
            pitch_end = estimate_pitch_hz(audio.slice(halfway, word.end), audio.sample_rate) if halfway else None

            cutoff = False
            if index == len(words) - 1 and word.end is not None and word_duration is not None and median_duration is not None:
                tail_rms = rms(audio.slice(max(word.start or 0.0, word.end - 0.04), word.end))
                after_rms = rms(audio.slice(word.end, word.end + 0.08))
                cutoff = word_duration <= min(0.25, median_duration * 0.75) and tail_rms > median_rms * 0.75 and after_rms < tail_rms * 0.25

            annotated_words.append(
                _annotate_word(
                    word,
                    word_rms=word_rms,
                    median_rms=median_rms,
                    duration=word_duration,
                    median_duration=median_duration,
                    pitch_start=pitch_start,
                    pitch_end=pitch_end,
                    cutoff=cutoff,
                )
            )

        annotated_words = _merge_wrapped_stretches(annotated_words)
        annotated_words = _apply_rate_markers(annotated_words, _segment_rate(words))
        new_word_tokens = tuple(
            WordToken(
                text=text,
                start=word.start,
                end=word.end,
                speaker=word.speaker,
                confidence=word.confidence,
            )
            for text, word in zip(annotated_words, words)
        )
        new_text = " ".join(annotated_words) if annotated_words else segment.text
        new_segments.append(
            TranscriptSegment(
                text=new_text,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                words=new_word_tokens,
            )
        )

    if log_callback:
        log_callback("Applied local acoustic Jeffersonian cues.")
    return TranscriptResult(source_path=result.source_path, language=result.language, segments=new_segments)
