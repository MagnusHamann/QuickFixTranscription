"""Setup helper for downloading the default local Whisper model."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "ggml-base.bin"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
DEFAULT_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
DEFAULT_MODEL_SHA1 = "465707469ff3a37a2b9b8d8f89f2f99de7299dac"
DEFAULT_MODEL_LABEL = "Whisper base multilingual GGML model"


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_is_valid(path: Path = DEFAULT_MODEL_PATH) -> bool:
    return path.exists() and sha1_file(path).lower() == DEFAULT_MODEL_SHA1


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

    part_path = DEFAULT_MODEL_PATH.with_suffix(DEFAULT_MODEL_PATH.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    progress(f"Downloading {DEFAULT_MODEL_LABEL}.")
    progress(f"Source: {DEFAULT_MODEL_URL}")
    progress(f"Destination: {DEFAULT_MODEL_PATH}")

    try:
        with urllib.request.urlopen(DEFAULT_MODEL_URL, timeout=30) as response, part_path.open("wb") as output:
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
                        progress(f"Model download progress: {percent}%")
                        next_report = percent + 10
    except (OSError, urllib.error.URLError) as exc:
        if part_path.exists():
            part_path.unlink()
        progress(f"Model download failed: {exc}")
        return None

    actual_hash = sha1_file(part_path)
    if actual_hash.lower() != DEFAULT_MODEL_SHA1:
        part_path.unlink(missing_ok=True)
        progress(f"Downloaded model checksum mismatch: {actual_hash}")
        progress("The model was deleted and will not be used.")
        return None

    part_path.replace(DEFAULT_MODEL_PATH)
    progress(f"Model ready: {DEFAULT_MODEL_PATH}")
    return DEFAULT_MODEL_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the default local Whisper model.")
    parser.add_argument("--yes", action="store_true", help="Accepted for setup-script compatibility.")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code if the model is not ready.")
    args = parser.parse_args(argv)

    model_path = download_default_model()
    if model_path:
        return 0
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
