"""Local dependency discovery for QuickFixTranscription."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from ffmpeg.ffmpeg_runner import PROJECT_ROOT, find_ffmpeg_tools
from transcription.model_setup import DEFAULT_MODEL_PATH


WHISPER_EXECUTABLE_NAMES = (
    "whisper-cli.exe",
    "main.exe",
    "whisper.exe",
    "whisper-cli",
    "main",
    "whisper",
)
MODEL_EXTENSIONS = {".bin", ".gguf"}


@dataclass(frozen=True)
class DependencyStatus:
    ffmpeg_path: str | None
    ffprobe_path: str | None
    whisper_path: str | None
    model_path: str | None

    @property
    def ready_for_transcription(self) -> bool:
        return bool(self.ffmpeg_path and self.ffprobe_path and self.whisper_path and self.model_path)

    @property
    def missing_labels(self) -> list[str]:
        missing: list[str] = []
        if not self.ffmpeg_path:
            missing.append("FFmpeg")
        if not self.ffprobe_path:
            missing.append("FFprobe")
        if not self.whisper_path:
            missing.append("whisper.cpp executable")
        if not self.model_path:
            missing.append("Whisper model")
        return missing


def find_whisper_executable() -> str | None:
    search_roots = [
        PROJECT_ROOT / ".tools" / "whisper",
        PROJECT_ROOT / ".tools" / "whisper" / "bin",
        PROJECT_ROOT / ".tools" / "whisper" / "build" / "bin",
        PROJECT_ROOT / ".tools" / "whisper" / "build" / "bin" / "Release",
    ]
    if platform.system() == "Windows":
        names = WHISPER_EXECUTABLE_NAMES
    else:
        names = tuple(name for name in WHISPER_EXECUTABLE_NAMES if not name.endswith(".exe"))

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)

    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_model_file() -> str | None:
    roots = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / ".tools" / "models",
        PROJECT_ROOT / ".tools" / "whisper" / "models",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.suffix.lower() in MODEL_EXTENSIONS)
    if not candidates:
        if DEFAULT_MODEL_PATH.exists():
            return str(DEFAULT_MODEL_PATH)
        return None
    candidates.sort(key=lambda path: (path.name.lower(), len(str(path))))
    return str(candidates[0])


def dependency_status() -> DependencyStatus:
    ffmpeg, ffprobe = find_ffmpeg_tools()
    return DependencyStatus(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        whisper_path=find_whisper_executable(),
        model_path=find_model_file(),
    )
