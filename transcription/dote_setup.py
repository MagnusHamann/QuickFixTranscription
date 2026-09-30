"""Download and install the pinned DOTE Whisper desktop package on Windows."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
import urllib.error
import urllib.request

from quickfix_sibling_apps import quickfix_dependency_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ROOT = quickfix_dependency_root(PROJECT_ROOT)
DOTE_VERSION = (1, 0, 2)
DOTE_VERSION_TEXT = ".".join(str(part) for part in DOTE_VERSION)
DOTE_INSTALLER_NAME = f"DOTE-Whisper-win-x64-{DOTE_VERSION_TEXT}.exe"
DOTE_INSTALLER_URL = (
    "https://github.com/BigSoftVideo/DOTEwhisper/releases/download/"
    f"v.{DOTE_VERSION_TEXT}/{DOTE_INSTALLER_NAME}"
)
DOTE_INSTALLER_SHA256 = "b460353917b40f97b738a9cdcb124e89cbdb0e4cf52d30984e1ffa9b44e47407"
DOTE_INSTALLERS_DIR = DEPENDENCY_ROOT / ".tools" / "dote-whisper" / "installers"
DOTE_MODELS_DIR = DEPENDENCY_ROOT / ".tools" / "dote-whisper" / "models" / "diarization"
DOTE_SEGMENTATION_MODEL_NAME = "sherpa-onnx-pyannote-segmentation-3-0.onnx"
DOTE_SEGMENTATION_MODEL_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/"
    "resolve/main/model.onnx"
)
DOTE_SEGMENTATION_MODEL_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
DOTE_EMBEDDING_MODEL_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
DOTE_EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    f"{DOTE_EMBEDDING_MODEL_NAME}"
)
DOTE_EMBEDDING_MODEL_SHA256 = "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
_SEMANTIC_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_DOTE_EXECUTABLE_NAMES = {"dote-whisper.exe", "dote whisper.exe"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installer_is_valid(path: Path) -> bool:
    return path.is_file() and sha256_file(path).lower() == DOTE_INSTALLER_SHA256


def _version_from_name(path: Path) -> tuple[int, int, int] | None:
    match = _SEMANTIC_VERSION_PATTERN.search(path.name)
    return tuple(int(part) for part in match.groups()) if match else None


def _candidate_install_roots(local_app_data: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for base in (local_app_data, local_app_data / "Programs"):
        if not base.is_dir():
            continue
        try:
            children = tuple(base.iterdir())
        except OSError:
            continue
        roots.extend(
            child
            for child in children
            if child.is_dir() and child.name.lower().startswith("dote-whisper")
        )
    return tuple(roots)


def find_installed_dote_version(local_app_data: Path | None = None) -> tuple[int, int, int] | None:
    """Return the newest installed DOTE Whisper Squirrel app version."""

    configured_base = os.environ.get("LOCALAPPDATA")
    if local_app_data is None and not configured_base:
        return None
    base = local_app_data or Path(configured_base or "")
    versions: list[tuple[int, int, int]] = []
    for root in _candidate_install_roots(base):
        try:
            children = tuple(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and child.name.lower().startswith("app-"):
                version = _version_from_name(child)
                if version is not None:
                    versions.append(version)
    return max(versions) if versions else None


def find_dote_executable(local_app_data: Path | None = None) -> Path | None:
    """Locate an executable in a completed local DOTE Whisper installation."""

    configured_base = os.environ.get("LOCALAPPDATA")
    if local_app_data is None and not configured_base:
        return None
    base = local_app_data or Path(configured_base or "")
    candidates: list[Path] = []
    for root in _candidate_install_roots(base):
        try:
            children = tuple(root.iterdir())
        except OSError:
            continue
        locations = (root, *(child for child in children if child.is_dir() and child.name.lower().startswith("app-")))
        for location in locations:
            try:
                items = tuple(location.iterdir())
            except OSError:
                continue
            candidates.extend(
                item
                for item in items
                if item.is_file() and item.name.lower() in _DOTE_EXECUTABLE_NAMES
            )
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.parent.name.lower().startswith("app-"), _version_from_name(path.parent) or (0, 0, 0)))
    return candidates[0]


def find_dote_app_directory(local_app_data: Path | None = None) -> Path | None:
    """Return the newest installed DOTE application directory."""

    configured_base = os.environ.get("LOCALAPPDATA")
    if local_app_data is None and not configured_base:
        return None
    base = local_app_data or Path(configured_base or "")
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for root in _candidate_install_roots(base):
        try:
            children = tuple(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or not child.name.lower().startswith("app-"):
                continue
            version = _version_from_name(child)
            if version is not None:
                candidates.append((version, child))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def find_dote_whisper_executable(
    local_app_data: Path | None = None,
    *,
    prefer_gpu: bool = True,
) -> Path | None:
    """Locate DOTE's bundled whisper.cpp binary, preferring a GPU build."""

    app_dir = find_dote_app_directory(local_app_data)
    if app_dir is None:
        return None
    bin_root = app_dir / "resources" / "bin"
    platform_dirs = ["win32-x64-cuda", "win32-x64"] if prefer_gpu else ["win32-x64", "win32-x64-cuda"]
    for folder in platform_dirs:
        candidate = bin_root / folder / "whisper-cli.exe"
        if candidate.is_file():
            return candidate
    try:
        return next(path for path in bin_root.rglob("whisper-cli.exe") if path.is_file())
    except (OSError, StopIteration):
        return None


def dote_model_paths(dependency_root: Path | None = None) -> tuple[Path, Path]:
    root = (dependency_root or DEPENDENCY_ROOT).resolve()
    model_dir = root / ".tools" / "dote-whisper" / "models" / "diarization"
    return model_dir / DOTE_SEGMENTATION_MODEL_NAME, model_dir / DOTE_EMBEDDING_MODEL_NAME


def _asset_is_valid(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and sha256_file(path).lower() == expected_sha256


def dote_models_are_ready(dependency_root: Path | None = None) -> bool:
    segmentation, embedding = dote_model_paths(dependency_root)
    return _asset_is_valid(segmentation, DOTE_SEGMENTATION_MODEL_SHA256) and _asset_is_valid(
        embedding,
        DOTE_EMBEDDING_MODEL_SHA256,
    )


def ensure_dote_models(
    progress: Callable[[str], None] = print,
    *,
    dependency_root: Path | None = None,
    downloader: Callable[[str, Path, Callable[[str], None]], bool] | None = None,
) -> tuple[Path, Path] | None:
    """Install the two pinned Sherpa-ONNX models used by the DOTE base pipeline."""

    dependencies = (dependency_root or DEPENDENCY_ROOT).resolve()
    download = downloader or _download_file
    assets = (
        (
            DOTE_SEGMENTATION_MODEL_NAME,
            DOTE_SEGMENTATION_MODEL_URL,
            DOTE_SEGMENTATION_MODEL_SHA256,
        ),
        (
            DOTE_EMBEDDING_MODEL_NAME,
            DOTE_EMBEDDING_MODEL_URL,
            DOTE_EMBEDDING_MODEL_SHA256,
        ),
    )
    targets = dote_model_paths(dependencies)
    profile_dir = dependencies / ".tools" / "dote-whisper" / "profile" / "models" / "diarization"

    for (name, url, expected_hash), target in zip(assets, targets):
        if _asset_is_valid(target, expected_hash):
            progress(f"DOTE diarization model already available: {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        seed = profile_dir / name
        if _asset_is_valid(seed, expected_hash):
            shutil.copy2(seed, target)
            progress(f"Copied verified DOTE diarization model into the shared runtime: {target}")
            continue

        target.unlink(missing_ok=True)
        progress(f"Downloading pinned DOTE diarization model: {name}")
        if not download(url, target, progress):
            return None
        if not _asset_is_valid(target, expected_hash):
            actual_hash = sha256_file(target) if target.is_file() else "missing"
            target.unlink(missing_ok=True)
            progress(f"DOTE diarization model checksum mismatch for {name}: {actual_hash}")
            return None
        progress(f"DOTE diarization model verified: {target}")

    return targets


def dote_installation_is_ready(local_app_data: Path | None = None) -> bool:
    version = find_installed_dote_version(local_app_data)
    return bool(
        version is not None
        and version >= DOTE_VERSION
        and find_dote_executable(local_app_data) is not None
    )


def dote_runtime_is_ready(
    local_app_data: Path | None = None,
    dependency_root: Path | None = None,
) -> bool:
    return bool(
        dote_installation_is_ready(local_app_data)
        and find_dote_whisper_executable(local_app_data) is not None
        and dote_models_are_ready(dependency_root)
    )


def _download_file(url: str, destination: Path, progress: Callable[[str], None]) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.unlink(missing_ok=True)
    progress("Downloading a pinned local DOTE runtime asset from its official release source.")
    progress(f"Destination: {destination}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response, part_path.open("wb") as output:
            total_header = response.headers.get("Content-Length")
            expected = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            next_report = 10
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if expected:
                    percent = int(downloaded * 100 / expected)
                    if percent >= next_report:
                        progress(f"DOTE Whisper download progress: {percent}%")
                        next_report = percent + 10
    except (OSError, urllib.error.URLError) as exc:
        part_path.unlink(missing_ok=True)
        progress(f"DOTE Whisper download failed: {exc}")
        return False

    part_path.replace(destination)
    return True


def _run_installer(installer: Path, progress: Callable[[str], None]) -> bool:
    progress("Installing DOTE Whisper locally. This may take a few minutes.")
    kwargs: dict[str, object] = {
        "check": False,
        "timeout": 600,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run([str(installer), "--silent"], **kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        progress(f"DOTE Whisper installer failed: {exc}")
        return False
    if result.returncode != 0:
        progress(f"DOTE Whisper installer exited with code {result.returncode}.")
        return False
    return True


def setup_dote_whisper(
    progress: Callable[[str], None] = print,
    *,
    dependency_root: Path | None = None,
    quickfix_root: Path | None = None,
    local_app_data: Path | None = None,
    downloader: Callable[[str, Path, Callable[[str], None]], bool] | None = None,
    installer_runner: Callable[[Path, Callable[[str], None]], bool] | None = None,
    model_downloader: Callable[[str, Path, Callable[[str], None]], bool] | None = None,
    poll_timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Ensure DOTE and its pinned local inference assets are fully installed."""

    if platform.system() != "Windows":
        progress("Automatic DOTE Whisper installation is available on Windows only.")
        return False

    dependencies = (dependency_root or DEPENDENCY_ROOT).resolve()
    app_parent = (quickfix_root or PROJECT_ROOT.parent).resolve()
    installed_version = find_installed_dote_version(local_app_data)

    if dote_installation_is_ready(local_app_data):
        progress(f"DOTE Whisper {'.'.join(map(str, installed_version))} is already installed.")
    else:
        if installed_version is not None and installed_version >= DOTE_VERSION:
            progress("The DOTE Whisper installation is incomplete; setup will repair it.")

        cached_installer = dependencies / ".tools" / "dote-whisper" / "installers" / DOTE_INSTALLER_NAME
        adjacent_installer = app_parent / DOTE_INSTALLER_NAME
        installer: Path | None = None
        for candidate in (cached_installer, adjacent_installer):
            if not candidate.is_file():
                continue
            if installer_is_valid(candidate):
                installer = candidate
                progress(f"Verified DOTE Whisper installer: {candidate}")
                break
            progress(f"Ignoring DOTE Whisper installer with an invalid checksum: {candidate}")
            if candidate == cached_installer:
                candidate.unlink(missing_ok=True)

        if installer is None:
            download = downloader or _download_file
            if not download(DOTE_INSTALLER_URL, cached_installer, progress):
                return False
            if not installer_is_valid(cached_installer):
                actual_hash = sha256_file(cached_installer) if cached_installer.is_file() else "missing"
                cached_installer.unlink(missing_ok=True)
                progress(f"Downloaded DOTE Whisper checksum mismatch: {actual_hash}")
                progress("The invalid installer was deleted and was not executed.")
                return False
            progress("DOTE Whisper installer checksum verified.")
            installer = cached_installer

        run_installer = installer_runner or _run_installer
        if not run_installer(installer, progress):
            return False

        deadline = time.monotonic() + max(0.0, poll_timeout)
        while not dote_installation_is_ready(local_app_data):
            if time.monotonic() >= deadline:
                progress("DOTE Whisper installer completed, but the installed application was not detected.")
                return False
            sleep(1.0)
        installed_version = find_installed_dote_version(local_app_data)
        progress(f"DOTE Whisper {'.'.join(map(str, installed_version))} is installed.")

    whisper_path = find_dote_whisper_executable(local_app_data)
    if whisper_path is None:
        progress("DOTE Whisper is installed, but its bundled whisper.cpp executable is missing.")
        return False
    progress(f"DOTE whisper.cpp ready: {whisper_path}")

    models = ensure_dote_models(
        progress,
        dependency_root=dependencies,
        downloader=model_downloader,
    )
    if models is None:
        progress("DOTE's local speaker-diarization models are incomplete.")
        return False

    progress("DOTE base transcription runtime is fully ready for offline processing.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the pinned local DOTE Whisper desktop package.")
    parser.add_argument("--yes", action="store_true", help="Accepted for setup-script compatibility.")
    parser.parse_args(argv)
    return 0 if setup_dote_whisper() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
