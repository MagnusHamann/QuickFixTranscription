"""Setup helper for optional local sherpa-onnx diarization assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from transcription.dependencies import SHERPA_TOOLS_DIR, find_sherpa_embedding_model, find_sherpa_segmentation_model


SEGMENTATION_ARCHIVE_NAME = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-segmentation-models/{SEGMENTATION_ARCHIVE_NAME}"
)
SEGMENTATION_MODEL_DIR = SHERPA_TOOLS_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_MODEL_PATH = SEGMENTATION_MODEL_DIR / "model.int8.onnx"

EMBEDDING_MODEL_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{EMBEDDING_MODEL_NAME}"
)
EMBEDDING_MODEL_PATH = SHERPA_TOOLS_DIR / EMBEDDING_MODEL_NAME


def sherpa_python_package_available() -> bool:
    try:
        import numpy  # noqa: F401
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def install_sherpa_python_package(progress: Callable[[str], None] = print) -> bool:
    """Install optional sherpa Python wheels into the current local Python environment."""
    if sherpa_python_package_available():
        progress("sherpa-onnx Python package already available.")
        return True

    command = [sys.executable, "-m", "pip", "install", "numpy>=1.26", "sherpa-onnx>=1.10.28"]
    progress("Installing optional local sherpa-onnx Python package.")
    progress("Running: " + subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in completed.stdout.splitlines():
        if line.strip():
            progress(line.rstrip())
    if completed.returncode != 0:
        progress(f"sherpa-onnx package install failed with exit code {completed.returncode}.")
        return False
    return sherpa_python_package_available()


def download_sherpa_models(progress: Callable[[str], None] = print) -> bool:
    """Download optional local sherpa-onnx diarization models."""
    SHERPA_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    segmentation_ok = _ensure_segmentation_model(progress)
    embedding_ok = _ensure_embedding_model(progress)
    return segmentation_ok and embedding_ok


def setup_sherpa(progress: Callable[[str], None] = print) -> bool:
    """Install/check optional local sherpa-onnx package and diarization models."""
    package_ok = install_sherpa_python_package(progress)
    models_ok = download_sherpa_models(progress)
    segmentation = find_sherpa_segmentation_model()
    embedding = find_sherpa_embedding_model()
    if segmentation:
        progress(f"sherpa-onnx segmentation model ready: {segmentation}")
    if embedding:
        progress(f"sherpa-onnx embedding model ready: {embedding}")
    return bool(package_ok and models_ok and segmentation and embedding)


def _ensure_segmentation_model(progress: Callable[[str], None]) -> bool:
    existing = find_sherpa_segmentation_model()
    if existing:
        progress(f"sherpa-onnx segmentation model already available: {existing}")
        return True

    archive_path = SHERPA_TOOLS_DIR / SEGMENTATION_ARCHIVE_NAME
    progress("Downloading optional sherpa-onnx speaker segmentation model.")
    if not _download_file(SEGMENTATION_ARCHIVE_URL, archive_path, progress):
        return False

    try:
        _safe_extract_tar(archive_path, SHERPA_TOOLS_DIR)
    except tarfile.TarError as exc:
        progress(f"sherpa-onnx segmentation extraction failed: {exc}")
        archive_path.unlink(missing_ok=True)
        return False
    archive_path.unlink(missing_ok=True)

    if find_sherpa_segmentation_model():
        return True
    progress("sherpa-onnx segmentation archive did not contain the expected ONNX model.")
    return False


def _ensure_embedding_model(progress: Callable[[str], None]) -> bool:
    existing = find_sherpa_embedding_model()
    if existing:
        progress(f"sherpa-onnx embedding model already available: {existing}")
        return True
    progress("Downloading optional sherpa-onnx speaker embedding model.")
    return _download_file(EMBEDDING_MODEL_URL, EMBEDDING_MODEL_PATH, progress)


def _download_file(url: str, destination: Path, progress: Callable[[str], None]) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.unlink(missing_ok=True)
    progress(f"Source: {url}")
    progress(f"Destination: {destination}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part_path.open("wb") as output:
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None
            downloaded = 0
            next_report = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if expected:
                    percent = int((downloaded / expected) * 100)
                    if percent >= next_report:
                        progress(f"Download progress: {percent}%")
                        next_report = percent + 10
    except (OSError, urllib.error.URLError) as exc:
        part_path.unlink(missing_ok=True)
        progress(f"Download failed: {exc}")
        return False
    if part_path.stat().st_size == 0:
        part_path.unlink(missing_ok=True)
        progress("Download produced an empty file.")
        return False
    part_path.replace(destination)
    return True


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination_root)):
                raise tarfile.TarError(f"Unsafe tar member path: {member.name}")
        archive.extractall(destination)


def clear_sherpa_download_cache(progress: Callable[[str], None] = print) -> None:
    """Remove incomplete sherpa downloads without touching completed models."""
    for part in (SHERPA_TOOLS_DIR.rglob("*.part") if SHERPA_TOOLS_DIR.exists() else ()):
        progress(f"Removing incomplete sherpa download: {part}")
        part.unlink(missing_ok=True)
    archive = SHERPA_TOOLS_DIR / SEGMENTATION_ARCHIVE_NAME
    if archive.exists():
        progress(f"Removing downloaded sherpa archive: {archive}")
        archive.unlink(missing_ok=True)
    empty_dirs = [path for path in SHERPA_TOOLS_DIR.rglob("*") if path.is_dir()] if SHERPA_TOOLS_DIR.exists() else []
    for path in sorted(empty_dirs, key=lambda item: len(str(item)), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    if SHERPA_TOOLS_DIR.exists() and not any(SHERPA_TOOLS_DIR.iterdir()):
        shutil.rmtree(SHERPA_TOOLS_DIR, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/check optional local sherpa-onnx diarization assets.")
    parser.add_argument("--yes", action="store_true", help="Accepted for setup-script compatibility.")
    parser.add_argument("--check-only", action="store_true", help="Only inspect existing local sherpa assets.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when sherpa assets are not ready.")
    args = parser.parse_args(argv)

    ready = bool(sherpa_python_package_available() and find_sherpa_segmentation_model() and find_sherpa_embedding_model())
    if not ready and not args.check_only:
        ready = setup_sherpa()
    return 0 if ready else (1 if args.strict else 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
