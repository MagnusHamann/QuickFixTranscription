"""Speaker label helpers shared by transcript exporters."""

from __future__ import annotations

from transcription.models import TranscriptSegment


def speaker_label(segment: TranscriptSegment, labels: dict[str, str]) -> str:
    key = segment.speaker or "speaker_1"
    if key not in labels:
        labels[key] = f"SP{len(labels) + 1}"
    return labels[key]
