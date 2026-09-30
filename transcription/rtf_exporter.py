"""RTF export helpers."""

from __future__ import annotations

from pathlib import Path

from transcription.models import TranscriptResult
from transcription.text_display import verbatim_segment_display_text
from transcription.speakers import speaker_display_prefix, speaker_label


RTF_OUTPUT_VERSION = 7
RTF_OUTPUT_VERSION_MARKER = f"QuickFixTranscriptionOutputVersion={RTF_OUTPUT_VERSION}"
RTF_OUTPUT_KEY_MARKER_PREFIX = "QuickFixTranscriptionOutputKey="


def _escape_code_unit(unit: int) -> str:
    signed = unit if unit < 32768 else unit - 65536
    return f"\\u{signed}?"


def rtf_escape(text: str) -> str:
    parts: list[str] = []
    for char in text:
        if char == "\\":
            parts.append("\\\\")
        elif char == "{":
            parts.append("\\{")
        elif char == "}":
            parts.append("\\}")
        elif char == "\n":
            parts.append("\\par\n")
        elif char == "\t":
            parts.append("\\tab ")
        elif ord(char) < 128:
            parts.append(char)
        else:
            encoded = char.encode("utf-16le")
            units = [encoded[index] + (encoded[index + 1] << 8) for index in range(0, len(encoded), 2)]
            parts.extend(_escape_code_unit(unit) for unit in units)
    return "".join(parts)


def write_rtf(
    path: Path,
    title: str,
    lines: list[str],
    font_name: str = "Calibri",
    include_title: bool = True,
    output_key: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\\par\n".join(rtf_escape(line) for line in lines)
    title_block = f"\\b {rtf_escape(title)}\\b0\\par\n\\par\n" if include_title else ""
    version_block = f"{{\\*\\comment {RTF_OUTPUT_VERSION_MARKER}}}\n"
    key_block = f"{{\\*\\comment {RTF_OUTPUT_KEY_MARKER_PREFIX}{output_key}}}\n" if output_key else ""
    content = (
        "{\\rtf1\\ansi\\deff0\n"
        f"{{\\fonttbl{{\\f0 {font_name};}}}}\n"
        "\\fs24\n"
        f"{version_block}"
        f"{key_block}"
        f"{title_block}"
        f"{body}\n"
        "}\n"
    )
    path.write_text(content, encoding="utf-8")


def _format_timepoint_marker(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}.{minutes:02d}.{secs:02d}"


def _advance_timepoint_markers(
    rows: list[str],
    next_marker: float | None,
    start: float | None,
    end: float | None,
    timepoint_interval_seconds: float,
) -> float | None:
    if next_marker is None:
        return None
    if start is not None:
        while start >= next_marker - 1e-9:
            rows.append(_format_timepoint_marker(next_marker))
            next_marker += timepoint_interval_seconds
    if end is not None:
        while end >= next_marker - 1e-9:
            rows.append(_format_timepoint_marker(next_marker))
            next_marker += timepoint_interval_seconds
    return next_marker


def transcript_lines(result: TranscriptResult, timepoint_interval_seconds: float = 30.0) -> list[str]:
    """Render plain verbatim transcript lines with speaker labels and time markers.

    The transcript keeps speaker separation, drops per-line timestamps, and
    inserts a separate line every 30 seconds by default so readers can orient
    themselves in the recording without cluttering each transcript line.
    """
    rows: list[str] = []
    speaker_labels: dict[str, str] = {}
    next_marker = timepoint_interval_seconds if timepoint_interval_seconds > 0 else None
    previous_displayed_speaker: str | None = None

    for segment in result.segments:
        text = verbatim_segment_display_text(segment)
        if not text:
            next_marker = _advance_timepoint_markers(rows, next_marker, segment.start, segment.end, timepoint_interval_seconds)
            continue
        next_marker = _advance_timepoint_markers(rows, next_marker, segment.start, None, timepoint_interval_seconds)
        speaker_name = speaker_label(segment, speaker_labels)
        speaker_display, previous_displayed_speaker = speaker_display_prefix(
            speaker_name,
            previous_displayed_speaker,
            uncertain=segment.speaker_uncertain,
        )
        speaker_prefix = f"{speaker_display}: " if speaker_display else " " * len(f"{speaker_name}: ")
        rows.append(f"{speaker_prefix}{text}")
        next_marker = _advance_timepoint_markers(rows, next_marker, None, segment.end, timepoint_interval_seconds)

    return rows


def has_current_output_version(path: Path, output_key: str | None = None) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if RTF_OUTPUT_VERSION_MARKER not in content:
        return False
    if output_key is None:
        return True
    return f"{RTF_OUTPUT_KEY_MARKER_PREFIX}{output_key}" in content
