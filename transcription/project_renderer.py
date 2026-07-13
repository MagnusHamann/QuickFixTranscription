"""Deterministic rendering from structured project state."""

from __future__ import annotations

from transcription.project_schema import CandidateSuggestion, Project, SegmentState


def render_project_draft_lines(project: Project) -> list[str]:
    rows: list[tuple[float, str, str]] = []
    for recording in project.recordings:
        for segment in recording.segments:
            start = segment.start if segment.start is not None else float("inf")
            rows.append((start, segment.id, f"{segment.speaker_label}: {segment.corrected_text}"))
    rows.sort(key=lambda row: (row[0], row[1]))
    width = max(4, len(str(len(rows))) + 3)
    return [f"{index:<{width}}{line}" for index, (_, __, line) in enumerate(rows, start=1)]


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
    for recording in project.recordings:
        for segment in recording.segments:
            text = segment.corrected_text
            confirmed = sorted(segment.confirmed_annotations, key=lambda item: (item.span.start or 0.0, item.kind, item.id))
            if confirmed:
                suffix = " ".join(f"{{{annotation.kind}:{annotation.symbol}}}" for annotation in confirmed)
                text = f"{text} {suffix}".strip()
            start = segment.start if segment.start is not None else float("inf")
            rows.append((start, segment.id, f"{segment.speaker_label}: {text}"))
    rows.sort(key=lambda row: (row[0], row[1]))
    width = max(4, len(str(len(rows))) + 3)
    return [f"{index:<{width}}{line}" for index, (_, __, line) in enumerate(rows, start=1)]


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
