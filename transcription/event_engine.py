"""Deterministic Jefferson candidate suggestion engine."""

from __future__ import annotations

import hashlib
import re

from transcription.jeffersonian import MIN_TIMED_SILENCE_SECONDS
from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.project_schema import AnnotationSpan, CandidateSuggestion, SegmentState


DEGREE = "\N{DEGREE SIGN}"
UP_ARROW = "\N{UPWARDS ARROW}"
DOWN_ARROW = "\N{DOWNWARDS ARROW}"
LOW_CONFIDENCE_THRESHOLD = 0.25
ACOUSTIC_MARKER_PATTERN = re.compile(r"::|^>|<$|^<|>$|-$")


def suggestions_for_project_segments(
    segments: tuple[SegmentState, ...],
    annotated_segments: tuple[TranscriptSegment, ...] = (),
) -> tuple[CandidateSuggestion, ...]:
    """Generate stable candidate annotations from structured segment state."""
    suggestions: list[CandidateSuggestion] = []
    previous: SegmentState | None = None
    for index, segment in enumerate(segments):
        suggestions.extend(_word_level_suggestions(segment))
        if index < len(annotated_segments):
            suggestions.extend(_project_acoustic_suggestions(segment, annotated_segments[index]))
        if previous is not None:
            suggestions.extend(_boundary_suggestions(previous, segment))
        previous = segment
    return tuple(sorted(suggestions, key=_suggestion_sort_key))


def suggestions_for_transcript(
    result: TranscriptResult,
    annotated_result: TranscriptResult | None = None,
) -> tuple[CandidateSuggestion, ...]:
    """Generate suggestions directly from transcript objects.

    This is used by the legacy batch path while the app transitions to full
    project-state editing.
    """
    suggestions: list[CandidateSuggestion] = []
    annotated_by_index = annotated_result.segments if annotated_result else []
    previous: TranscriptSegment | None = None
    for index, segment in enumerate(result.segments):
        annotated = annotated_by_index[index] if index < len(annotated_by_index) else None
        segment_id = _id("segment", index, segment.start, segment.end, segment.text)
        suggestions.extend(_transcript_word_suggestions(segment_id, segment, annotated))
        if previous is not None:
            suggestions.extend(_transcript_boundary_suggestions(index, previous, segment))
        previous = segment
    return tuple(sorted(suggestions, key=_suggestion_sort_key))


def attach_suggestions_to_segments(
    segments: tuple[SegmentState, ...],
    suggestions: tuple[CandidateSuggestion, ...],
) -> tuple[SegmentState, ...]:
    by_segment: dict[str, list[CandidateSuggestion]] = {}
    for suggestion in suggestions:
        if suggestion.span.segment_id:
            by_segment.setdefault(suggestion.span.segment_id, []).append(suggestion)

    updated: list[SegmentState] = []
    for segment in segments:
        segment_suggestions = tuple(sorted(by_segment.get(segment.id, ()), key=_suggestion_sort_key))
        updated.append(
            SegmentState(
                id=segment.id,
                recording_id=segment.recording_id,
                start=segment.start,
                end=segment.end,
                speaker_label=segment.speaker_label,
                raw_asr_text=segment.raw_asr_text,
                corrected_text=segment.corrected_text,
                tokens=segment.tokens,
                candidate_annotations=segment_suggestions,
                confirmed_annotations=segment.confirmed_annotations,
                review_flags=segment.review_flags,
                confidence=segment.confidence,
                notes=segment.notes,
            )
        )
    return tuple(updated)


def _word_level_suggestions(segment: SegmentState) -> list[CandidateSuggestion]:
    suggestions: list[CandidateSuggestion] = []
    previous = None
    for token in segment.tokens:
        if token.confidence is not None and token.confidence < LOW_CONFIDENCE_THRESHOLD:
            suggestions.append(
                _suggestion(
                    "uncertain_word",
                    "Uncertain word",
                    f"Review low-confidence word '{token.text}'.",
                    segment.id,
                    token.start,
                    token.end,
                    (token.id,),
                    f"({token.text})",
                    token.confidence,
                )
            )
        if previous and previous.end is not None and token.start is not None:
            gap = token.start - previous.end
            rounded = round(gap, 1)
            if gap >= MIN_TIMED_SILENCE_SECONDS and rounded >= MIN_TIMED_SILENCE_SECONDS:
                suggestions.append(
                    _suggestion(
                        "inline_pause",
                        "Possible inline pause",
                        f"Review within-turn pause of {rounded:.1f}s.",
                        segment.id,
                        previous.end,
                        token.start,
                        (previous.id, token.id),
                        f"({rounded:.1f})",
                    )
                )
        previous = token
    return suggestions


def _project_acoustic_suggestions(segment: SegmentState, annotated: TranscriptSegment) -> list[CandidateSuggestion]:
    suggestions: list[CandidateSuggestion] = []
    for index, token in enumerate(segment.tokens):
        if index >= len(annotated.words):
            break
        raw_word = WordToken(
            text=token.text,
            start=token.start,
            end=token.end,
            speaker=token.speaker_label,
            confidence=token.confidence,
        )
        suggestions.extend(_acoustic_marker_suggestions(segment.id, token.id, raw_word, annotated.words[index]))
    return suggestions


def _boundary_suggestions(previous: SegmentState, current: SegmentState) -> list[CandidateSuggestion]:
    suggestions: list[CandidateSuggestion] = []
    if previous.end is None or current.start is None:
        return suggestions
    if current.start < previous.end:
        suggestions.append(
            _suggestion(
                "overlap",
                "Possible overlap",
                "Review overlap boundary and bracket placement.",
                current.id,
                current.start,
                min(previous.end, current.end or previous.end),
                (),
                "[ ]",
                source="local_timing",
            )
        )
        return suggestions
    gap = current.start - previous.end
    rounded = round(gap, 1)
    if gap >= MIN_TIMED_SILENCE_SECONDS and rounded >= MIN_TIMED_SILENCE_SECONDS:
        suggestions.append(
            _suggestion(
                "turn_pause",
                "Possible between-turn pause",
                f"Review between-turn pause of {rounded:.1f}s.",
                current.id,
                previous.end,
                current.start,
                (),
                f"({rounded:.1f})",
                source="local_timing",
            )
        )
    return suggestions


def _transcript_word_suggestions(
    segment_id: str,
    segment: TranscriptSegment,
    annotated: TranscriptSegment | None,
) -> list[CandidateSuggestion]:
    suggestions: list[CandidateSuggestion] = []
    words = segment.words
    annotated_words = annotated.words if annotated else ()
    previous = None
    for index, word in enumerate(words):
        word_id = _id("word", segment_id, index, word.text, word.start, word.end)
        if word.confidence is not None and word.confidence < LOW_CONFIDENCE_THRESHOLD:
            suggestions.append(
                _suggestion(
                    "uncertain_word",
                    "Uncertain word",
                    f"Review low-confidence word '{word.text}'.",
                    segment_id,
                    word.start,
                    word.end,
                    (word_id,),
                    f"({word.text})",
                    word.confidence,
                )
            )
        if index < len(annotated_words):
            suggestions.extend(_acoustic_marker_suggestions(segment_id, word_id, word, annotated_words[index]))
        if previous and previous.end is not None and word.start is not None:
            gap = word.start - previous.end
            rounded = round(gap, 1)
            if gap >= MIN_TIMED_SILENCE_SECONDS and rounded >= MIN_TIMED_SILENCE_SECONDS:
                previous_id = _id("word", segment_id, index - 1, previous.text, previous.start, previous.end)
                suggestions.append(
                    _suggestion(
                        "inline_pause",
                        "Possible inline pause",
                        f"Review within-turn pause of {rounded:.1f}s.",
                        segment_id,
                        previous.end,
                        word.start,
                        (previous_id, word_id),
                        f"({rounded:.1f})",
                    )
                )
        previous = word
    return suggestions


def _transcript_boundary_suggestions(
    index: int,
    previous: TranscriptSegment,
    current: TranscriptSegment,
) -> list[CandidateSuggestion]:
    current_id = _id("segment", index, current.start, current.end, current.text)
    previous_state = SegmentState(
        id=_id("segment", index - 1, previous.start, previous.end, previous.text),
        recording_id="transcript",
        start=previous.start,
        end=previous.end,
        speaker_label=previous.speaker or "SP1",
        raw_asr_text=previous.text,
        corrected_text=previous.text,
    )
    current_state = SegmentState(
        id=current_id,
        recording_id="transcript",
        start=current.start,
        end=current.end,
        speaker_label=current.speaker or "SP1",
        raw_asr_text=current.text,
        corrected_text=current.text,
    )
    return _boundary_suggestions(previous_state, current_state)


def _acoustic_marker_suggestions(
    segment_id: str,
    word_id: str,
    raw_word: WordToken,
    annotated_word: WordToken,
) -> list[CandidateSuggestion]:
    raw = raw_word.text.strip()
    marked = annotated_word.text.strip()
    if not raw or not marked or raw == marked:
        return []

    suggestions: list[CandidateSuggestion] = []
    if DEGREE in marked:
        suggestions.append(
            _suggestion(
                "quiet",
                "Possible quiet speech",
                f"Review quietness marking for '{raw}'.",
                segment_id,
                raw_word.start,
                raw_word.end,
                (word_id,),
                f"{DEGREE}{raw}{DEGREE}",
                source="local_acoustic",
            )
        )
    if marked.startswith(UP_ARROW) or marked.startswith(DOWN_ARROW):
        symbol = marked[0]
        label = "Possible pitch rise" if symbol == UP_ARROW else "Possible pitch fall"
        suggestions.append(
            _suggestion("pitch", label, f"Review pitch movement for '{raw}'.", segment_id, raw_word.start, raw_word.end, (word_id,), symbol, source="local_acoustic")
        )
    if "::" in marked:
        suggestions.append(
            _suggestion(
                "elongation",
                "Possible elongation",
                f"Review elongation marking for '{raw}'.",
                segment_id,
                raw_word.start,
                raw_word.end,
                (word_id,),
                "::",
                source="local_acoustic",
            )
        )
    if marked.endswith("-") and not raw.endswith("-"):
        suggestions.append(
            _suggestion(
                "cutoff",
                "Possible cut-off",
                f"Review cut-off marking for '{raw}'.",
                segment_id,
                raw_word.start,
                raw_word.end,
                (word_id,),
                "-",
                source="local_acoustic",
            )
        )
    if raw.lower() != marked.lower() and marked.upper() == marked and any(char.isalpha() for char in marked):
        suggestions.append(
            _suggestion(
                "loud",
                "Possible loud speech",
                f"Review loudness marking for '{raw}'.",
                segment_id,
                raw_word.start,
                raw_word.end,
                (word_id,),
                raw.upper(),
                source="local_acoustic",
            )
        )
    if ACOUSTIC_MARKER_PATTERN.search(marked) and not suggestions:
        suggestions.append(
            _suggestion(
                "acoustic_marker",
                "Possible acoustic cue",
                f"Review acoustic cue '{marked}' for '{raw}'.",
                segment_id,
                raw_word.start,
                raw_word.end,
                (word_id,),
                marked,
                source="local_acoustic",
            )
        )
    return suggestions


def _suggestion(
    kind: str,
    label: str,
    detail: str,
    segment_id: str,
    start: float | None,
    end: float | None,
    word_ids: tuple[str, ...],
    symbol: str,
    confidence: float | None = None,
    source: str = "local_rule_engine",
) -> CandidateSuggestion:
    span = AnnotationSpan(
        id=_id("span", kind, segment_id, start, end, *word_ids),
        start=start,
        end=end,
        segment_id=segment_id,
        word_ids=word_ids,
    )
    return CandidateSuggestion(
        id=_id("candidate", kind, segment_id, start, end, *word_ids, symbol),
        kind=kind,
        label=label,
        detail=detail,
        span=span,
        confidence=confidence,
        suggested_symbol=symbol,
        source=source,
    )


def _suggestion_sort_key(suggestion: CandidateSuggestion) -> tuple[float, str, str]:
    start = suggestion.span.start if suggestion.span.start is not None else float("inf")
    return start, suggestion.kind, suggestion.id


def _id(*parts: object) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
