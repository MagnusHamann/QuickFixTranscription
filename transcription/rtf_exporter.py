"""RTF export helpers."""

from __future__ import annotations

from pathlib import Path

from transcription.models import TranscriptResult
from transcription.speakers import speaker_label
from transcription.time_utils import format_timestamp


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


def write_rtf(path: Path, title: str, lines: list[str], font_name: str = "Calibri", include_title: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\\par\n".join(rtf_escape(line) for line in lines)
    title_block = f"\\b {rtf_escape(title)}\\b0\\par\n\\par\n" if include_title else ""
    content = (
        "{\\rtf1\\ansi\\deff0\n"
        f"{{\\fonttbl{{\\f0 {font_name};}}}}\n"
        "\\fs24\n"
        f"{title_block}"
        f"{body}\n"
        "}\n"
    )
    path.write_text(content, encoding="utf-8")


def transcript_lines(result: TranscriptResult) -> list[str]:
    lines: list[str] = []
    speaker_labels: dict[str, str] = {}
    for segment in result.segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        labelled_text = f"{speaker_label(segment, speaker_labels)}: {text}"
        if segment.start is not None:
            lines.append(f"[{format_timestamp(segment.start)}] {labelled_text}")
        else:
            lines.append(labelled_text)
    return lines
