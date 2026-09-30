"""Deterministic rendering from structured project state."""

from __future__ import annotations

from transcription.project_schema import CandidateSuggestion, Project, SegmentState
from transcription.speakers import speaker_display_prefix
from transcription.text_display import clean_transcript_text, is_placeholder_display_text


def render_project_draft_lines(project: Project) -> list[str]:
    rows: list[tuple[float, str, str]] = []
    previous_displayed_speaker: str | None = None
    for recording in project.recordings:
        for segment in recording.segments:
            start = segment.start if segment.start is not None else float("inf")
            text = clean_transcript_text(segment.corrected_text)
            if is_placeholder_display_text(text):
                continue
            speaker_display, previous_displayed_speaker = speaker_display_prefix(
                segment.speaker_label,
                previous_displayed_speaker,
            )
            speaker_prefix = f"{speaker_display}: " if speaker_display else " " * len(f"{segment.speaker_label}: ")
            rows.append((start, segment.id, f"{speaker_prefix}{text}"))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [line for _, __, line in rows]


def render_candidate_review_lines(project: Project) -> list[str]:
    suggestions = list(iter_project_suggestions(project))
    if not suggestions:
        return ["No pending machine-generated annotation suggestions."]
    lines = ["Pending machine-generated annotation suggestions:", ""]
    for index, suggestion in enumerate(suggestions, start=1):
        start = "" if suggestion.span.start is None else f"{suggestion.span.start:.2f}s"
        end = "" if suggestion.span.end is None else f"{suggestion.span.end:.2f}s"
        time_range = start if not end else f"{start}-{end}"
        symbol = f" [{suggestion.suggested_symbol}]" if suggestion.suggested_symbol else ""
        lines.append(f"{index}. {suggestion.label}{symbol} at {time_range}: {suggestion.detail} ({suggestion.status})")
    return lines


def render_confirmed_jefferson_lines(project: Project) -> list[str]:
    """Render text from confirmed annotations only.

    Pending machine suggestions are deliberately excluded from final rendering.
    """
    rows: list[tuple[float, str, str]] = []
    previous_displayed_speaker: str | None = None
    for recording in project.recordings:
        for segment in recording.segments:
            text = clean_transcript_text(segment.corrected_text)
            if is_placeholder_display_text(text):
                continue
            confirmed = sorted(segment.confirmed_annotations, key=lambda item: (item.span.start or 0.0, item.kind, item.id))
            if confirmed:
                suffix = " ".join(f"{{{annotation.kind}:{annotation.symbol}}}" for annotation in confirmed)
                text = f"{text} {suffix}".strip()
            start = segment.start if segment.start is not None else float("inf")
            speaker_display, previous_displayed_speaker = speaker_display_prefix(
                segment.speaker_label,
                previous_displayed_speaker,
            )
            speaker_prefix = f"{speaker_display}: " if speaker_display else " " * len(f"{segment.speaker_label}: ")
            rows.append((start, segment.id, f"{speaker_prefix}{text}"))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [line for _, __, line in rows]


def iter_project_segments(project: Project) -> tuple[SegmentState, ...]:
    segments: list[SegmentState] = []
    for recording in project.recordings:
        segments.extend(recording.segments)
    return tuple(sorted(segments, key=lambda item: (item.start if item.start is not None else float("inf"), item.id)))


def iter_project_suggestions(project: Project, status: str | None = "pending") -> tuple[CandidateSuggestion, ...]:
    suggestions: list[CandidateSuggestion] = []
    for segment in iter_project_segments(project):
        for suggestion in segment.candidate_annotations:
            if status is None or suggestion.status == status:
                suggestions.append(suggestion)
    return tuple(sorted(suggestions, key=lambda item: (item.span.start if item.span.start is not None else float("inf"), item.kind, item.id)))
