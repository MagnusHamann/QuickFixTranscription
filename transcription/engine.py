"""Local transcription engine adapters."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable

from transcription.models import TranscriptResult, TranscriptSegment, WordToken
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
    if not isinstance(raw_words, list):
        raw_words = segment.get("tokens")
    if not isinstance(raw_words, list):
        return ()

    words: list[WordToken] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word") or raw.get("text") or raw.get("token") or "").strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]") and not is_supported_nonword_source(text):
            continue
        start = _segment_time(raw, "from")
        end = _segment_time(raw, "to")
        words.append(
            WordToken(
                text=text,
                start=start,
                end=end,
                speaker=speaker,
                confidence=_confidence(raw),
            )
        )
    return tuple(words)


class WhisperCppEngine:
    """Adapter for a local whisper.cpp command-line executable."""

    def __init__(self, executable: str, model_path: str, prefer_gpu: bool = True) -> None:
        self.executable = Path(executable).expanduser()
        self.model_path = Path(model_path).expanduser()
        self.prefer_gpu = prefer_gpu
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
        speaker = raw.get("speaker") or raw.get("speaker_id")
        speaker_text = str(speaker) if speaker is not None else None
        segments.append(
            TranscriptSegment(
                text=text,
                start=_segment_time(raw, "from"),
                end=_segment_time(raw, "to"),
                speaker=speaker_text,
                words=_parse_word_tokens(raw, speaker_text),
            )
        )

    return TranscriptResult(source_path=source_path, language=language, segments=segments)
