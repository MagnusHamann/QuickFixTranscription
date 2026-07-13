"""Local broad-overlap helpers."""

from __future__ import annotations

import math
from pathlib import Path

from transcription.acoustic_analysis import PcmAudio, rms
from transcription.models import TranscriptResult, TranscriptSegment, WordToken


MIN_CHANNEL_RMS = 0.002
MIN_CHANNEL_DIFFERENCE_RATIO = 0.08
MAX_DUPLICATE_CORRELATION = 0.985


def channels_are_distinct(left_path: Path, right_path: Path) -> bool:
    """Return True when two extracted channels contain genuinely different audio."""
    try:
        left = PcmAudio.from_wav(left_path)
        right = PcmAudio.from_wav(right_path)
    except Exception:
        return False

    sample_count = min(len(left.samples), len(right.samples))
    if sample_count == 0:
        return False

    left_samples = left.samples[:sample_count]
    right_samples = right.samples[:sample_count]
    left_rms = rms(left_samples)
    right_rms = rms(right_samples)
    if max(left_rms, right_rms) < MIN_CHANNEL_RMS:
        return False

    difference = tuple(left_samples[index] - right_samples[index] for index in range(sample_count))
    difference_ratio = rms(difference) / max(left_rms, right_rms, 1e-9)
    if difference_ratio < MIN_CHANNEL_DIFFERENCE_RATIO:
        return False

    correlation = _pearson_correlation(left_samples, right_samples)
    if correlation is not None and correlation > MAX_DUPLICATE_CORRELATION:
        return False

    return True


def merge_channel_results(
    source_path: Path,
    fallback: TranscriptResult,
    channel_results: list[tuple[str, TranscriptResult]],
) -> TranscriptResult | None:
    """Merge local per-channel transcripts into a broad transcript with speaker IDs."""
    merged_segments: list[TranscriptSegment] = []
    languages = [result.language for _speaker, result in channel_results if result.language]
    language = fallback.language or (languages[0] if languages else None)

    for speaker, result in channel_results:
        for segment in result.segments:
            if not segment.text.strip():
                continue
            merged_segments.append(_retag_segment(segment, speaker))

    speakers = {segment.speaker for segment in merged_segments if segment.speaker}
    if len(speakers) < 2:
        return None

    merged_segments.sort(key=lambda segment: (segment.start is None, segment.start or 0.0, segment.end or 0.0, segment.speaker or ""))
    return TranscriptResult(source_path=source_path, language=language, segments=merged_segments)


def has_cross_speaker_overlap(result: TranscriptResult) -> bool:
    """Return True when timed segments from different speakers overlap."""
    timed = [
        segment
        for segment in result.segments
        if segment.start is not None and segment.end is not None and segment.speaker
    ]
    for index, segment in enumerate(timed):
        assert segment.start is not None and segment.end is not None
        for other in timed[index + 1 :]:
            assert other.start is not None and other.end is not None
            if segment.speaker == other.speaker:
                continue
            if max(segment.start, other.start) < min(segment.end, other.end):
                return True
    return False


def _retag_segment(segment: TranscriptSegment, speaker: str) -> TranscriptSegment:
    return TranscriptSegment(
        text=segment.text,
        start=segment.start,
        end=segment.end,
        speaker=speaker,
        words=tuple(_retag_word(word, speaker) for word in segment.words),
    )


def _retag_word(word: WordToken, speaker: str) -> WordToken:
    return WordToken(
        text=word.text,
        start=word.start,
        end=word.end,
        speaker=speaker,
        confidence=word.confidence,
    )


def _pearson_correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_total = 0.0
    right_total = 0.0
    for left_value, right_value in zip(left, right):
        left_delta = left_value - left_mean
        right_delta = right_value - right_mean
        numerator += left_delta * right_delta
        left_total += left_delta * left_delta
        right_total += right_delta * right_delta
    denominator = math.sqrt(left_total * right_total)
    if denominator <= 0:
        return None
    return numerator / denominator
