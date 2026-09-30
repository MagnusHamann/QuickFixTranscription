"""Helpers for rendering transcript text cleanly."""

from __future__ import annotations

import re

from transcription.models import TranscriptSegment


PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)
BLANK_AUDIO_MARKER_COMPACTS = {
    "blankaudio",
    "blankaudioonscreen",
    "noaudio",
    "noaudioonscreen",
}
VERBATIM_UNCLEAR_MARKER = "(unclear)"
EXPLICIT_UNCLEAR_TERMS = {
    "inaudible",
    "incomprehensible",
    "uncertain",
    "unclear",
    "unintelligible",
    "unknown",
}
EXPLICIT_UNCLEAR_PATTERN = re.compile(
    r"(?:\[\s*(?:inaudible|incomprehensible|uncertain|unclear|unintelligible|unknown|\?+)\s*\]"
    r"|\(\s*(?:inaudible|incomprehensible|uncertain|unclear|unintelligible|unknown|\?+)\s*\)"
    r"|\{\s*(?:inaudible|incomprehensible|uncertain|unclear|unintelligible|unknown|\?+)\s*\}"
    r"|<\s*(?:inaudible|incomprehensible|uncertain|unclear|unintelligible|unknown|\?+)\s*>)",
    re.IGNORECASE,
)


def clean_transcript_text(text: str) -> str:
    """Collapse whitespace and strip obviously empty display text."""
    return " ".join(text.split()).strip()


def is_punctuation_only(text: str) -> bool:
    cleaned = clean_transcript_text(text)
    return bool(cleaned) and bool(PUNCTUATION_ONLY_PATTERN.fullmatch(cleaned))


def _compact_placeholder_text(text: str) -> str:
    return re.sub(r"[\s\[\]\(\)\{\}<>_]+", "", clean_transcript_text(text).lower())


def _placeholder_word_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_transcript_text(text).lower())


def is_blank_audio_marker_text(text: str) -> bool:
    return _compact_placeholder_text(text) in BLANK_AUDIO_MARKER_COMPACTS


def blank_audio_word_run_length(words: list[str] | tuple[str, ...], start_index: int) -> int:
    keys = [_placeholder_word_key(word) for word in words]
    if start_index < 0 or start_index >= len(keys):
        return 0

    current = keys[start_index]
    if current in {"blankaudio", "noaudio"}:
        return 1

    remaining = keys[start_index:]
    for prefix in (("blank", "audio", "on", "screen"), ("no", "audio", "on", "screen")):
        if tuple(remaining[: len(prefix)]) == prefix:
            return len(prefix)
    for prefix in (("blank", "audio"), ("no", "audio")):
        if tuple(remaining[: len(prefix)]) == prefix:
            return len(prefix)
    return 0


def remove_blank_audio_word_runs(words: list[str] | tuple[str, ...]) -> list[str]:
    cleaned = [clean_transcript_text(word) for word in words if clean_transcript_text(word)]
    if not cleaned:
        return []
    if is_blank_audio_marker_text(" ".join(cleaned)):
        return []

    filtered: list[str] = []
    index = 0
    while index < len(cleaned):
        run_length = blank_audio_word_run_length(cleaned, index)
        if run_length:
            index += run_length
            continue
        filtered.append(cleaned[index])
        index += 1
    return filtered


def is_overlap_placeholder_text(text: str) -> bool:
    cleaned = clean_transcript_text(text)
    return cleaned.replace(" ", "") == "()"


def is_placeholder_display_text(text: str, preserve_overlap_placeholder: bool = False) -> bool:
    """Return True for rows that should not be rendered as transcript speech."""
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return True
    if is_overlap_placeholder_text(cleaned):
        return not preserve_overlap_placeholder
    if is_punctuation_only(cleaned):
        return True

    compact = _compact_placeholder_text(cleaned)
    if not compact:
        return True
    if compact in BLANK_AUDIO_MARKER_COMPACTS:
        return True
    if compact in {"blank", "audio"} and cleaned.isupper():
        return True
    if compact in {"pause", "silence"} and (cleaned.startswith("[") or cleaned.startswith("(") or cleaned.isupper()):
        return True
    return False


def segment_display_text(segment: TranscriptSegment, preserve_overlap_placeholder: bool = False) -> str:
    """Return a cleaned display string for a transcript segment.

    When word tokens are available, punctuation-only tokens are suppressed so
    they do not become standalone rows after diarization splitting.
    """
    if segment.words:
        words = remove_blank_audio_word_runs([
            clean_transcript_text(word.text)
            for word in segment.words
            if clean_transcript_text(word.text) and not is_punctuation_only(word.text)
        ])
        text = " ".join(words)
    else:
        text = clean_transcript_text(segment.text)

    if is_placeholder_display_text(text, preserve_overlap_placeholder=preserve_overlap_placeholder):
        return ""
    return text


def verbatim_segment_display_text(segment: TranscriptSegment) -> str:
    """Render speech omissions explicitly without applying Jefferson notation."""
    text = segment_display_text(segment, preserve_overlap_placeholder=True)
    if not text:
        return ""
    if is_overlap_placeholder_text(text):
        return VERBATIM_UNCLEAR_MARKER

    cleaned = clean_transcript_text(text)
    if _compact_placeholder_text(cleaned) in EXPLICIT_UNCLEAR_TERMS:
        return VERBATIM_UNCLEAR_MARKER
    return EXPLICIT_UNCLEAR_PATTERN.sub(VERBATIM_UNCLEAR_MARKER, cleaned)
