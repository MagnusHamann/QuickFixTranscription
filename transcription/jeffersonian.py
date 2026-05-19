"""Simple Jeffersonian-style transcript formatter."""

from __future__ import annotations

from dataclasses import dataclass
import re

from transcription.models import TranscriptResult, TranscriptSegment, WordToken

MIN_TIMED_SILENCE_SECONDS = 0.2
LINE_NUMBER_WIDTH = 4
SPEAKER_WIDTH = 12
MAX_TEXT_COLUMNS = 50
ASR_PUNCTUATION_PATTERN = re.compile(r'(?<!\d)\.(?!\d)|[,?!;"“”‘’…]')


@dataclass
class JeffersonianRow:
    speaker: str
    text: str
    segment: TranscriptSegment | None = None


def speaker_label(segment: TranscriptSegment, labels: dict[str, str]) -> str:
    key = segment.speaker or "speaker_1"
    if key not in labels:
        labels[key] = chr(ord("A") + len(labels)) if len(labels) < 26 else f"S{len(labels) + 1}"
    return labels[key]


def _format_rows(rows: list[JeffersonianRow]) -> list[str]:
    line_number_width = max(LINE_NUMBER_WIDTH, len(str(len(rows))) + 3)
    return [
        f"{index:<{line_number_width}}{_speaker_cell(row.speaker):<{SPEAKER_WIDTH}}{row.text}"
        for index, row in enumerate(rows, start=1)
    ]


def _speaker_cell(speaker: str) -> str:
    return f"{speaker}:" if speaker else ""


def _normalize_text(text: str) -> str:
    return " ".join(ASR_PUNCTUATION_PATTERN.sub("", text).split())


def _segment_text_and_spans(segment: TranscriptSegment) -> tuple[str, list[tuple[WordToken, int, int]]]:
    if not segment.words:
        return _normalize_text(segment.text), []

    parts: list[str] = []
    spans: list[tuple[WordToken, int, int]] = []
    cursor = 0
    previous: WordToken | None = None

    for word in segment.words:
        text = _normalize_text(word.text)
        if not text:
            continue

        if parts:
            separator = " "
            if previous and previous.end is not None and word.start is not None:
                gap = word.start - previous.end
                rounded = round(gap, 1)
                if gap >= MIN_TIMED_SILENCE_SECONDS - 1e-9 and rounded >= MIN_TIMED_SILENCE_SECONDS:
                    separator += f"({rounded:.1f}) "
            parts.append(separator)
            cursor += len(separator)

        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((word, start, cursor))
        previous = word

    return "".join(parts), spans


def _nearest_word_boundary(text: str, index: int) -> int:
    if index <= 0 or not text:
        return 0
    if index >= len(text):
        return len(text)

    boundaries = {0, len(text)}
    for position, char in enumerate(text):
        if char == " ":
            boundaries.add(position)
            boundaries.add(position + 1)

    return min(boundaries, key=lambda boundary: (abs(boundary - index), boundary))


def _time_to_text_column(segment: TranscriptSegment, when: float) -> int:
    text, spans = _segment_text_and_spans(segment)
    if spans:
        nearest = spans[0]
        nearest_distance = float("inf")
        for word, start_col, end_col in spans:
            if word.start is not None and word.end is not None and word.start <= when <= word.end:
                return start_col
            if word.start is not None:
                distance = abs(word.start - when)
                if distance < nearest_distance:
                    nearest = (word, start_col, end_col)
                    nearest_distance = distance
        return nearest[1]

    if not text or segment.start is None or segment.end is None or segment.end <= segment.start:
        return 0

    ratio = (when - segment.start) / (segment.end - segment.start)
    rough_index = int(round(max(0.0, min(1.0, ratio)) * len(text)))
    return _nearest_word_boundary(text, rough_index)


def _insert_overlap_brackets(text: str, start_column: int, end_column: int | None = None) -> str:
    clean = _normalize_text(text)
    if not clean:
        return clean

    start = max(0, min(start_column, len(clean)))
    if end_column is None:
        end = len(clean)
    else:
        end = max(start, min(end_column, len(clean)))

    return f"{clean[:start]}[{clean[start:end]}]{clean[end:]}"


def _find_overlap_row(rows: list[JeffersonianRow], current: TranscriptSegment, current_speaker: str) -> JeffersonianRow | None:
    if current.start is None:
        return None
    for row in reversed(rows):
        previous = row.segment
        if previous is None or previous.end is None:
            continue
        if row.speaker == current_speaker:
            continue
        if current.start < previous.end:
            return row
    return None


def _apply_overlap_alignment(row: JeffersonianRow, current: TranscriptSegment) -> tuple[JeffersonianRow, int]:
    previous = row.segment
    if previous is None or current.start is None:
        return row, 0

    overlap_start = current.start
    start_column = _time_to_text_column(previous, overlap_start)
    end_column = None
    if previous.end is not None and current.end is not None:
        overlap_end = min(previous.end, current.end)
        end_column = _time_to_text_column(previous, overlap_end)
        if end_column <= start_column:
            end_column = None

    if "[" not in row.text:
        row.text = _insert_overlap_brackets(row.text, start_column, end_column)
    else:
        start_column = row.text.index("[")

    return row, start_column


def _wrap_text(text: str, max_columns: int = MAX_TEXT_COLUMNS) -> list[str]:
    max_columns = max(1, int(max_columns))
    if len(text) <= max_columns:
        return [text]

    leading_spaces = len(text) - len(text.lstrip(" "))
    if leading_spaces:
        prefix = text[:leading_spaces]
        body = text[leading_spaces:]
        available = max_columns - leading_spaces
        if available < 10:
            return [text]
        body_lines = _wrap_text(body, available)
        return [prefix + body_lines[0], *body_lines[1:]]

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if not word:
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_columns:
            current = candidate
            continue
        if current:
            lines.append(current)
        while len(word) > max_columns:
            lines.append(word[:max_columns])
            word = word[max_columns:]
        current = word

    if current:
        lines.append(current)

    return lines or [text[:max_columns]]


def _wrap_rows(rows: list[JeffersonianRow], max_text_columns: int = MAX_TEXT_COLUMNS) -> list[JeffersonianRow]:
    wrapped: list[JeffersonianRow] = []
    for row in rows:
        for text in _wrap_text(row.text, max_text_columns):
            wrapped.append(JeffersonianRow(row.speaker, text, row.segment))
    return wrapped


def format_simple_jeffersonian(result: TranscriptResult, max_text_columns: int = MAX_TEXT_COLUMNS) -> list[str]:
    """Format segments as a simple numbered Jeffersonian transcript."""
    labels: dict[str, str] = {}
    rows: list[JeffersonianRow] = []
    previous_end: float | None = None
    previous_speaker: str | None = None

    for segment in result.segments:
        text, _ = _segment_text_and_spans(segment)
        text = _normalize_text(text)
        if not text:
            continue

        current_speaker = speaker_label(segment, labels)
        pause = ""
        if previous_end is not None and segment.start is not None:
            gap = segment.start - previous_end
            rounded = round(gap, 1)
            if gap >= MIN_TIMED_SILENCE_SECONDS - 1e-9:
                if rounded >= MIN_TIMED_SILENCE_SECONDS:
                    pause = f"({rounded:.1f}) "
                    if previous_speaker is not None and previous_speaker != current_speaker:
                        rows.append(JeffersonianRow("", pause.strip()))
                        pause = ""

        overlap_row = _find_overlap_row(rows, segment, current_speaker)
        if overlap_row:
            _, bracket_column = _apply_overlap_alignment(overlap_row, segment)
            text = f"{' ' * bracket_column}[{text}]"
        else:
            text = f"{pause}{text}"

        rows.append(JeffersonianRow(current_speaker, text, segment))

        if segment.end is not None:
            previous_end = max(previous_end or segment.end, segment.end)
        previous_speaker = current_speaker

    return _format_rows(_wrap_rows(rows, max_text_columns))
