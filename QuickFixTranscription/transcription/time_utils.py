"""Time parsing and formatting helpers for transcription segments."""

from __future__ import annotations

import re


TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def parse_time_to_seconds(value: str, label: str) -> int:
    """Parse mm:ss or hh:mm:ss into whole seconds."""
    clean = value.strip()
    if not TIME_PATTERN.fullmatch(clean):
        raise ValueError(f"{label} must use mm:ss or hh:mm:ss.")

    parts = [int(part) for part in clean.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    else:
        hours, minutes, seconds = parts

    if minutes > 59 or seconds > 59:
        raise ValueError(f"{label} has minutes or seconds outside the valid range.")

    return hours * 3600 + minutes * 60 + seconds


def parse_optional_time(value: str, label: str) -> int | None:
    """Parse an optional mm:ss or hh:mm:ss value."""
    clean = value.strip()
    if not clean:
        return None
    return parse_time_to_seconds(clean, label)


def validate_time_range(start: str, end: str) -> tuple[int | None, int | None]:
    """Validate optional start/end values and return seconds."""
    start_seconds = parse_optional_time(start, "Start time")
    end_seconds = parse_optional_time(end, "Finish time")

    if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("Finish time must be greater than start time.")

    return start_seconds, end_seconds


def format_seconds_for_ffmpeg(seconds: int) -> str:
    """Format seconds as hh:mm:ss for FFmpeg arguments."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float | None) -> str:
    """Format a transcript timestamp for display."""
    if seconds is None:
        return "--:--"
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

