"""Speaker label helpers shared by transcript exporters."""

from __future__ import annotations

from transcription.models import TranscriptSegment


def speaker_label(segment: TranscriptSegment, labels: dict[str, str]) -> str:
    key = segment.speaker or "speaker_1"
    if key not in labels:
        labels[key] = f"SP{len(labels) + 1}"
    return labels[key]


def speaker_display_prefix(
    speaker: str,
    previous_speaker: str | None,
    uncertain: bool = False,
) -> tuple[str, str | None]:
    """Return the visible speaker prefix for the current row.

    The first row for a speaker keeps the full tag. Repeated consecutive rows
    for the same speaker keep the label mapping but suppress the visible tag so
    that output only shows the speaker when it changes.
    """
    if not speaker:
        return "", previous_speaker
    if uncertain:
        return f"{speaker} (?)", speaker
    if speaker == previous_speaker:
        return "", previous_speaker
    return speaker, speaker
