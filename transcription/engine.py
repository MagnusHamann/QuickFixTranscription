"""Local transcription engine adapters."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable

from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.text_display import (
    blank_audio_word_run_length,
    is_blank_audio_marker_text,
    is_placeholder_display_text,
    is_punctuation_only,
    segment_display_text,
)
from transcription.nonword_sounds import is_supported_nonword_source


class MissingDependencyError(RuntimeError):
    """Raised when a required local transcription dependency is missing."""


def _parse_timestamp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except ValueError:
        return None
    return None


def _segment_time(segment: dict, key: str) -> float | None:
    timestamps = segment.get("timestamps") or {}
    parsed = _parse_timestamp(timestamps.get(key))
    if parsed is not None:
        return parsed

    direct = segment.get(key) or segment.get("t0" if key == "from" else "t1")
    parsed = _parse_timestamp(direct)
    if parsed is not None:
        return parsed

    offsets = segment.get("offsets") or {}
    offset = offsets.get(key)
    if isinstance(offset, (int, float)):
        return float(offset) / 1000.0
    return None


def _confidence(raw: dict) -> float | None:
    for key in ("confidence", "probability", "p", "prob"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _parse_word_tokens(segment: dict, speaker: str | None) -> tuple[WordToken, ...]:
    raw_words = segment.get("words")
    if isinstance(raw_words, list):
        return _parse_timestamp_units(raw_words, speaker)

    raw_tokens = segment.get("tokens")
    if not isinstance(raw_tokens, list):
        return ()

    return _merge_whisper_subword_tokens(raw_tokens, speaker)


def _parse_timestamp_units(raw_units: list[object], speaker: str | None) -> tuple[WordToken, ...]:
    words: list[WordToken] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word") or raw.get("text") or raw.get("token") or "").strip()
        if not _is_usable_timestamp_text(text):
            continue
        words.append(_word_token_from_raw(text, raw, speaker))
    return _drop_blank_audio_word_runs(words)


def _merge_whisper_subword_tokens(raw_tokens: list[object], speaker: str | None) -> tuple[WordToken, ...]:
    """Rebuild words from whisper.cpp BPE tokens using their leading-space boundaries."""
    usable = [raw for raw in raw_tokens if isinstance(raw, dict)]
    raw_texts = [str(raw.get("word") or raw.get("text") or raw.get("token") or "") for raw in usable]
    has_word_boundaries = any(text[:1].isspace() for text in raw_texts[1:] if text)
    if not has_word_boundaries:
        return _parse_timestamp_units(usable, speaker)

    words: list[WordToken] = []
    current_parts: list[str] = []
    current_units: list[dict] = []

    def flush() -> None:
        if not current_parts or not current_units:
            current_parts.clear()
            current_units.clear()
            return
        text = "".join(current_parts).strip()
        if _is_usable_timestamp_text(text):
            starts = [_segment_time(raw, "from") for raw in current_units]
            ends = [_segment_time(raw, "to") for raw in current_units]
            confidences = [_confidence(raw) for raw in current_units]
            words.append(
                WordToken(
                    text=text,
                    start=min((value for value in starts if value is not None), default=None),
                    end=max((value for value in ends if value is not None), default=None),
                    speaker=speaker,
                    confidence=min((value for value in confidences if value is not None), default=None),
                )
            )
        current_parts.clear()
        current_units.clear()

    for raw, raw_text in zip(usable, raw_texts):
        if not raw_text:
            continue
        stripped = raw_text.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]") and not is_supported_nonword_source(stripped):
            flush()
            continue
        if raw_text[:1].isspace() and current_parts:
            flush()
        if is_punctuation_only(stripped):
            continue
        current_parts.append(stripped)
        current_units.append(raw)
    flush()
    return _drop_blank_audio_word_runs(words)


def _is_usable_timestamp_text(text: str) -> bool:
    if not text:
        return False
    if text.startswith("[") and text.endswith("]") and not is_supported_nonword_source(text):
        return False
    return not is_punctuation_only(text)


def _word_token_from_raw(text: str, raw: dict, speaker: str | None) -> WordToken:
    return WordToken(
        text=text,
        start=_segment_time(raw, "from"),
        end=_segment_time(raw, "to"),
        speaker=speaker,
        confidence=_confidence(raw),
    )


def _drop_blank_audio_word_runs(words: list[WordToken]) -> tuple[WordToken, ...]:
    if not words:
        return ()
    texts = [word.text for word in words]
    filtered: list[WordToken] = []
    index = 0
    while index < len(words):
        run_length = blank_audio_word_run_length(texts, index)
        if run_length:
            index += run_length
            continue
        filtered.append(words[index])
        index += 1
    return tuple(filtered)


class WhisperCppEngine:
    """Adapter for a local whisper.cpp command-line executable."""

    def __init__(
        self,
        executable: str,
        model_path: str,
        prefer_gpu: bool = True,
        *,
        no_context: bool = False,
    ) -> None:
        self.executable = Path(executable).expanduser()
        self.model_path = Path(model_path).expanduser()
        self.prefer_gpu = prefer_gpu
        self.no_context = no_context
        self.current_process: subprocess.Popen[str] | None = None

    def validate(self) -> None:
        if not self.executable.exists():
            raise MissingDependencyError("The whisper.cpp executable was not found.")
        if not self.model_path.exists():
            raise MissingDependencyError("The Whisper model file was not found.")

    def terminate(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()

    def transcribe(
        self,
        audio_path: Path,
        source_path: Path,
        work_dir: Path,
        language_code: str,
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> TranscriptResult:
        self.validate()
        output_base = work_dir / f"{audio_path.stem}_whisper"
        json_path = output_base.with_suffix(".json")
        txt_path = output_base.with_suffix(".txt")

        command = [
            str(self.executable),
            "-m",
            str(self.model_path),
            "-f",
            str(audio_path),
            "-oj",
            "-ojf",
            "-otxt",
            "-of",
            str(output_base),
        ]
        if language_code:
            command.extend(["-l", language_code])
        if self.no_context:
            command.extend(["-mc", "0"])
        if self.prefer_gpu:
            log_callback("GPU preference enabled; whisper.cpp will use local GPU acceleration if this binary supports it.")
        else:
            command.append("-ng")
            log_callback("GPU preference disabled; running whisper.cpp in CPU mode.")

        log_callback("Running local whisper.cpp transcription.")
        self.current_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        while self.current_process.poll() is None:
            if cancelled():
                self.terminate()
                raise RuntimeError("Transcription stopped by user.")
            time.sleep(0.1)

        self.current_process.wait()
        return_code = self.current_process.returncode
        self.current_process = None

        if return_code != 0:
            raise RuntimeError(f"whisper.cpp exited with code {return_code}.")

        if json_path.exists():
            return parse_whisper_json(json_path, source_path)
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            return TranscriptResult(source_path=source_path, language=language_code or None, segments=[TranscriptSegment(text=text)])

        raise RuntimeError("whisper.cpp did not create a transcript output file.")


def parse_whisper_json(path: Path, source_path: Path) -> TranscriptResult:
    parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    language = None
    result = parsed.get("result")
    if isinstance(result, dict):
        language = result.get("language")

    raw_segments = parsed.get("transcription") or parsed.get("segments") or []
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        if is_blank_audio_marker_text(text) or is_placeholder_display_text(text):
            continue
        speaker = raw.get("speaker") or raw.get("speaker_id")
        speaker_text = str(speaker) if speaker is not None else None
        words = _parse_word_tokens(raw, speaker_text)
        segment = TranscriptSegment(
            text=text,
            start=_segment_time(raw, "from"),
            end=_segment_time(raw, "to"),
            speaker=speaker_text,
            words=words,
        )
        if not segment_display_text(segment):
            continue
        segments.append(segment)

    return TranscriptResult(source_path=source_path, language=language, segments=segments)
