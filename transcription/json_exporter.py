"""Structured JSON export for integrated DOTE transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.speakers import speaker_label
from transcription.text_display import verbatim_segment_display_text


JSON_OUTPUT_SCHEMA = "quickfix-dote-transcript"
JSON_OUTPUT_SCHEMA_VERSION = 1


def _timestamp(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _offset(milliseconds_from_seconds: float | None) -> int | None:
    return None if milliseconds_from_seconds is None else int(round(milliseconds_from_seconds * 1000))


def _mapped_speaker(raw_speaker: str | None, labels: dict[str, str]) -> str:
    return speaker_label(TranscriptSegment(text="", speaker=raw_speaker), labels)


def _word_payload(word: WordToken, fallback_speaker: str, labels: dict[str, str]) -> dict[str, Any]:
    speaker = _mapped_speaker(word.speaker, labels) if word.speaker else fallback_speaker
    payload: dict[str, Any] = {
        "word": word.text,
        "timestamps": {"from": _timestamp(word.start), "to": _timestamp(word.end)},
        "offsets": {"from": _offset(word.start), "to": _offset(word.end)},
        "speaker": speaker,
    }
    if word.confidence is not None:
        payload["probability"] = word.confidence
    return payload


def transcript_json_payload(
    result: TranscriptResult,
    transcription_type: str,
    rendered_lines: list[str],
    output_key: str | None = None,
) -> dict[str, Any]:
    """Return a portable, DOTE/whisper-style structured transcript payload."""
    labels: dict[str, str] = {}
    transcription: list[dict[str, Any]] = []
    for segment in result.segments:
        speaker = speaker_label(segment, labels)
        display_text = verbatim_segment_display_text(segment)
        item: dict[str, Any] = {
            "timestamps": {"from": _timestamp(segment.start), "to": _timestamp(segment.end)},
            "offsets": {"from": _offset(segment.start), "to": _offset(segment.end)},
            "text": segment.text,
            "display_text": display_text,
            "speaker": speaker,
            "speaker_uncertain": segment.speaker_uncertain,
            "words": [_word_payload(word, speaker, labels) for word in segment.words],
        }
        transcription.append(item)

    return {
        "schema": JSON_OUTPUT_SCHEMA,
        "schema_version": JSON_OUTPUT_SCHEMA_VERSION,
        "source_file": result.source_path.name,
        "result": {"language": result.language},
        "transcription_type": transcription_type,
        "text": result.text,
        "rendered_lines": rendered_lines,
        "transcription": transcription,
        "quickfix": {"output_key": output_key},
    }


def write_json_transcript(
    path: Path,
    result: TranscriptResult,
    transcription_type: str,
    rendered_lines: list[str],
    output_key: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = transcript_json_payload(result, transcription_type, rendered_lines, output_key)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def has_current_json_output(path: Path, output_key: str | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("schema") != JSON_OUTPUT_SCHEMA or payload.get("schema_version") != JSON_OUTPUT_SCHEMA_VERSION:
        return False
    if output_key is None:
        return True
    quickfix = payload.get("quickfix")
    return isinstance(quickfix, dict) and quickfix.get("output_key") == output_key
