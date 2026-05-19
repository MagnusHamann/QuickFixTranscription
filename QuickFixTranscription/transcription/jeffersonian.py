"""Simple Jeffersonian-style transcript formatter."""

from __future__ import annotations

from transcription.models import TranscriptResult, TranscriptSegment


def speaker_label(segment: TranscriptSegment, labels: dict[str, str]) -> str:
    key = segment.speaker or "speaker_1"
    if key not in labels:
        labels[key] = chr(ord("A") + len(labels)) if len(labels) < 26 else f"S{len(labels) + 1}"
    return labels[key]


def format_simple_jeffersonian(result: TranscriptResult) -> list[str]:
    """Format segments using a deliberately small Jeffersonian subset."""
    labels: dict[str, str] = {}
    lines: list[str] = []
    previous_end: float | None = None

    for segment in result.segments:
        text = " ".join(segment.text.split())
        if not text:
            continue

        overlap = False
        if previous_end is not None and segment.start is not None:
            gap = segment.start - previous_end
            if gap >= 0.05:
                rounded = round(gap, 1)
                if rounded >= 0.1:
                    lines.append(f"({rounded:.1f})")
            elif gap < -0.05:
                overlap = True

        if overlap:
            text = f"[{text}]"

        lines.append(f"{speaker_label(segment, labels)}: {text}")

        if segment.end is not None:
            previous_end = max(previous_end or segment.end, segment.end)

    return lines

