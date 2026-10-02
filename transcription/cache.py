"""Local cache and temp storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from transcription.models import TranscriptResult, TranscriptSegment, TranscriptionOptions, WordToken
from transcription.text_display import is_placeholder_display_text


CACHE_SCHEMA_VERSION = 7
WHISPER_TOKENIZATION_VERSION = 2
SPEAKER_RECONCILIATION_VERSION = 6


def local_data_root() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif system == "Darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "QuickFixTranscription"


def local_cache_root() -> Path:
    return local_data_root() / "cache"


def local_temp_root() -> Path:
    return local_data_root() / "temp"


def temp_directory_for_source(source_path: Path) -> Path:
    token = _path_token(source_path)
    return local_temp_root() / token


def clear_local_cache() -> int:
    """Remove generated local cache/temp data and return removed byte count."""
    total = _directory_size(local_cache_root()) + _directory_size(local_temp_root())
    for root in (local_cache_root(), local_temp_root()):
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    return total


def cache_key(stage: str, source_path: Path, options: TranscriptionOptions, extra: dict[str, Any] | None = None) -> str:
    source = source_path.expanduser()
    payload: dict[str, Any] = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "source": str(source.resolve()) if source.exists() else str(source),
        "source_size": source.stat().st_size if source.exists() else None,
        "source_mtime_ns": source.stat().st_mtime_ns if source.exists() else None,
        "language_code": options.language_code,
        "transcribe_section": options.transcribe_section,
        "start_time": options.start_time,
        "finish_time": options.finish_time,
        "model_path": _file_fingerprint(options.model_path),
        "asr_backend": options.asr_backend,
        "danish_whisper_model": _file_fingerprint(
            str(Path(options.danish_model_path).expanduser() / "model.bin")
        ) if options.danish_model_path.strip() else None,
        "whisper_executable": _file_fingerprint(options.whisper_executable),
        "whisper_tokenization_version": WHISPER_TOKENIZATION_VERSION,
    }
    if stage == "broad":
        payload.update(
            {
                "pyannote_pipeline_path": _file_fingerprint(options.pyannote_pipeline_path),
                "known_speakers": options.known_speakers,
                "speaker_reconciliation_version": SPEAKER_RECONCILIATION_VERSION,
            }
        )
    elif stage == "selected_output":
        payload["speaker_reconciliation_version"] = SPEAKER_RECONCILIATION_VERSION
    if extra:
        payload.update(extra)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_transcript(stage: str, key: str, source_path: Path) -> TranscriptResult | None:
    path = _cache_path(stage, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CACHE_SCHEMA_VERSION:
            return None
        return _transcript_from_json(payload["transcript"], source_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_cached_transcript(stage: str, key: str, result: TranscriptResult) -> Path:
    path = _cache_path(stage, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "transcript": _transcript_to_json(result),
    }
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    part_path.replace(path)
    return path


def _cache_path(stage: str, key: str) -> Path:
    safe_stage = "".join(char for char in stage.lower() if char.isalnum() or char in {"-", "_"}) or "stage"
    return local_cache_root() / safe_stage / f"{key}.json"


def _path_token(path: Path) -> str:
    text = str(path.expanduser().resolve() if path.exists() else path.expanduser())
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def _file_fingerprint(raw_path: str) -> dict[str, Any] | None:
    if not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _transcript_to_json(result: TranscriptResult) -> dict[str, Any]:
    return {
        "language": result.language,
        "segments": [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "speaker_uncertain": segment.speaker_uncertain,
                "words": [
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "speaker": word.speaker,
                        "confidence": word.confidence,
                    }
                    for word in segment.words
                ],
            }
            for segment in result.segments
        ],
    }


def _transcript_from_json(payload: dict[str, Any], source_path: Path) -> TranscriptResult:
    segments = []
    for raw_segment in payload.get("segments") or []:
        raw_text = str(raw_segment.get("text") or "")
        if is_placeholder_display_text(raw_text):
            continue
        words = tuple(
            WordToken(
                text=str(raw_word.get("text") or ""),
                start=_optional_float(raw_word.get("start")),
                end=_optional_float(raw_word.get("end")),
                speaker=str(raw_word["speaker"]) if raw_word.get("speaker") is not None else None,
                confidence=_optional_float(raw_word.get("confidence")),
            )
            for raw_word in raw_segment.get("words") or []
            if str(raw_word.get("text") or "").strip()
        )
        segments.append(
            TranscriptSegment(
                text=raw_text,
                start=_optional_float(raw_segment.get("start")),
                end=_optional_float(raw_segment.get("end")),
                speaker=str(raw_segment["speaker"]) if raw_segment.get("speaker") is not None else None,
                words=words,
                speaker_uncertain=bool(raw_segment.get("speaker_uncertain", False)),
            )
        )
    return TranscriptResult(source_path=source_path, language=payload.get("language"), segments=segments)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total
