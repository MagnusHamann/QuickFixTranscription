"""Setup and local discovery for the ungated Danish Røst v3 Whisper model."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from ffmpeg.ffmpeg_runner import PROJECT_ROOT
from quickfix_sibling_apps import quickfix_app_roots
from transcription.offline_guard import OFFLINE_ENVIRONMENT


DANISH_WHISPER_REPOSITORY_ID = "pluttodk/roest-v3-whisper-1.5b-ct2"
# Pin downloads so a future repository update cannot silently change local inference.
DANISH_WHISPER_MODEL_REVISION = "4ddeb1a4fb829260da60088303832e0b947f3114"
DANISH_WHISPER_MODEL_DIR = quickfix_app_roots(PROJECT_ROOT)[0] / ".tools" / "roest-v3-whisper-1.5b-ct2"
DANISH_WHISPER_ENV_DIR = quickfix_app_roots(PROJECT_ROOT)[0] / ".venvs" / "QuickFixDanishWhisper"
DANISH_WHISPER_REQUIRED_PATHS = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
)
DANISH_WHISPER_RUNTIME_REQUIREMENTS = (
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.2",
    "av==18.1.0",
    "huggingface-hub==1.33.0",
)


def danish_whisper_model_is_ready(path: str | Path = DANISH_WHISPER_MODEL_DIR) -> bool:
    root = Path(path).expanduser()
    return root.is_dir() and all((root / relative).is_file() for relative in DANISH_WHISPER_REQUIRED_PATHS)


def find_danish_whisper_model() -> str | None:
    for app_root in quickfix_app_roots(PROJECT_ROOT):
        for candidate in (
            app_root / ".tools" / "roest-v3-whisper-1.5b-ct2",
            app_root / "models" / "roest-v3-whisper-1.5b-ct2",
        ):
            if danish_whisper_model_is_ready(candidate):
                return str(candidate)
    return None


def danish_whisper_python_executable() -> Path:
    return DANISH_WHISPER_ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@lru_cache(maxsize=1)
def danish_whisper_runtime_is_ready() -> bool:
    python = danish_whisper_python_executable()
    if not python.is_file():
        return False
    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from importlib.metadata import version; "
                    "import ctranslate2, faster_whisper; "
                    "assert version('faster-whisper') == '1.2.1'; "
                    "assert version('ctranslate2') == '4.8.2'; "
                    "assert version('av') == '18.1.0'"
                ),
            ],
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _run(command: list[str], progress: Callable[[str], None]) -> bool:
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        for line in completed.stdout.splitlines():
            progress(line)
    return completed.returncode == 0


def setup_danish_whisper(progress: Callable[[str], None] = print) -> Path | None:
    """Download the ungated model and isolated runtime during setup only."""
    python = danish_whisper_python_executable()
    if not python.is_file():
        progress(f"Creating an isolated Danish Whisper runtime: {DANISH_WHISPER_ENV_DIR}")
        if not _run([sys.executable, "-m", "venv", str(DANISH_WHISPER_ENV_DIR)], progress):
            progress("Could not create the isolated Danish Whisper Python environment.")
            return None

    progress("Installing the pinned local Danish Whisper inference runtime.")
    if not _run([str(python), "-m", "pip", "install", *DANISH_WHISPER_RUNTIME_REQUIREMENTS], progress):
        progress("Danish Whisper Python runtime installation failed.")
        return None

    download_script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={DANISH_WHISPER_REPOSITORY_ID!r}, "
        f"revision={DANISH_WHISPER_MODEL_REVISION!r}, local_dir={str(DANISH_WHISPER_MODEL_DIR)!r})"
    )
    DANISH_WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    progress("Downloading Røst v3 Danish Whisper into QuickFixAppDependencies.")
    progress("This setup download contains model files only; no recordings or transcripts are involved.")
    if not _run([str(python), "-c", download_script], progress):
        progress("Røst v3 Danish Whisper download failed.")
        return None

    if not danish_whisper_model_is_ready(DANISH_WHISPER_MODEL_DIR):
        progress("The downloaded Røst v3 Danish Whisper folder is incomplete.")
        return None

    danish_whisper_runtime_is_ready.cache_clear()
    if not danish_whisper_runtime_is_ready():
        progress("Røst v3 files were downloaded, but its local Python runtime is incomplete.")
        return None
    progress(f"Røst v3 is ready for offline Danish transcription: {DANISH_WHISPER_MODEL_DIR}")
    return DANISH_WHISPER_MODEL_DIR
