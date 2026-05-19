"""Setup helper for downloading the default local Whisper model."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "ggml-base.bin"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
DEFAULT_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
DEFAULT_MODEL_SHA1 = "465707469ff3a37a2b9b8d8f89f2f99de7299dac"
DEFAULT_MODEL_LABEL = "Whisper base multilingual GGML model"

TOOLS_DIR = PROJECT_ROOT / ".tools"
WHISPER_DIR = TOOLS_DIR / "whisper"
WHISPER_EXECUTABLE_NAMES = (
    "whisper-cli.exe",
    "main.exe",
    "whisper.exe",
)
WHISPER_CPP_VERSION = "v1.8.4"
WINDOWS_WHISPER_ZIP_NAME = "whisper-bin-x64.zip"
WINDOWS_WHISPER_ZIP_URL = (
    f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_CPP_VERSION}/{WINDOWS_WHISPER_ZIP_NAME}"
)
WINDOWS_WHISPER_ZIP_SHA256 = "74f973345cb52ef5ba3ec9e7e7af8e48cc8c71722d1528603b80588a11f82e3e"


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_is_valid(path: Path = DEFAULT_MODEL_PATH) -> bool:
    return path.exists() and sha1_file(path).lower() == DEFAULT_MODEL_SHA1


def _download_file(url: str, destination: Path, progress: Callable[[str], None]) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    progress(f"Source: {url}")
    progress(f"Destination: {destination}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response, part_path.open("wb") as output:
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
        if part_path.exists():
            part_path.unlink()
        progress(f"Download failed: {exc}")
        return False

    part_path.replace(destination)
    return True


def download_default_model(progress: Callable[[str], None] = print) -> Path | None:
    """Download the default model if it is missing.

    Returns the model path on success, or None if setup could not complete.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if model_is_valid(DEFAULT_MODEL_PATH):
        progress(f"Default Whisper model already available: {DEFAULT_MODEL_PATH}")
        return DEFAULT_MODEL_PATH

    if DEFAULT_MODEL_PATH.exists():
        existing_hash = sha1_file(DEFAULT_MODEL_PATH)
        progress(f"Existing model checksum did not match expected SHA1: {existing_hash}")
        progress("Keeping existing file untouched. Choose it manually if you trust it.")
        return None

    progress(f"Downloading {DEFAULT_MODEL_LABEL}.")
    if not _download_file(DEFAULT_MODEL_URL, DEFAULT_MODEL_PATH, progress):
        return None

    actual_hash = sha1_file(DEFAULT_MODEL_PATH)
    if actual_hash.lower() != DEFAULT_MODEL_SHA1:
        DEFAULT_MODEL_PATH.unlink(missing_ok=True)
        progress(f"Downloaded model checksum mismatch: {actual_hash}")
        progress("The model was deleted and will not be used.")
        return None

    progress(f"Model ready: {DEFAULT_MODEL_PATH}")
    return DEFAULT_MODEL_PATH


def _find_local_windows_whisper() -> Path | None:
    if platform.system() != "Windows":
        return None
    if WHISPER_DIR.exists():
        for name in WHISPER_EXECUTABLE_NAMES:
            found = next(WHISPER_DIR.rglob(name), None)
            if found and found.exists():
                return found
    return None


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination_root)):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(destination)


def download_windows_whisper_cpp(progress: Callable[[str], None] = print) -> Path | None:
    """Download the official Windows x64 whisper.cpp release zip if needed."""
    existing = _find_local_windows_whisper()
    if existing:
        progress(f"whisper.cpp executable already available: {existing}")
        return existing

    if platform.system() != "Windows":
        progress("Automatic whisper.cpp binary download is currently Windows-only.")
        progress("On macOS/Linux, setup can install whisper-cpp via Homebrew where available.")
        return None

    zip_path = TOOLS_DIR / WINDOWS_WHISPER_ZIP_NAME
    progress(f"Downloading whisper.cpp {WHISPER_CPP_VERSION} Windows x64 binary.")
    if not _download_file(WINDOWS_WHISPER_ZIP_URL, zip_path, progress):
        return None

    actual_hash = sha256_file(zip_path)
    if actual_hash.lower() != WINDOWS_WHISPER_ZIP_SHA256:
        zip_path.unlink(missing_ok=True)
        progress(f"Downloaded whisper.cpp zip checksum mismatch: {actual_hash}")
        progress("The zip was deleted and will not be used.")
        return None

    _safe_extract_zip(zip_path, WHISPER_DIR)
    zip_path.unlink(missing_ok=True)

    executable = _find_local_windows_whisper()
    if executable:
        progress(f"whisper.cpp ready: {executable}")
        return executable

    progress("whisper.cpp zip was extracted, but no executable was found.")
    return None


def setup_runtime(progress: Callable[[str], None] = print) -> bool:
    """Download large runtime assets that are not committed to GitHub."""
    model_path = download_default_model(progress)
    whisper_path = download_windows_whisper_cpp(progress)
    if platform.system() == "Windows":
        return bool(model_path and whisper_path)
    return bool(model_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download local QuickFixTranscription runtime assets.")
    parser.add_argument("--yes", action="store_true", help="Accepted for setup-script compatibility.")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code if runtime setup is not ready.")
    args = parser.parse_args(argv)

    ready = setup_runtime()
    if ready:
        return 0
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
