"""Setup helper for local Montreal Forced Aligner presets."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from transcription.dependencies import (
    MFA_ENV_DIR,
    MFA_ROOT_DIR,
    MFA_TOOLS_DIR,
    dependency_status,
    find_mfa_acoustic_model,
    find_mfa_dictionary,
    find_mfa_executable,
)
from transcription.mfa_presets import DEFAULT_SETUP_MFA_PRESET_IDS, MFA_PRESETS, mfa_preset_by_id


MICROMAMBA_DIR = MFA_TOOLS_DIR / "micromamba"


def micromamba_root_dir() -> Path:
    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "qfmfa-py311"
    return MFA_TOOLS_DIR / "micromamba-root-py311"


MICROMAMBA_ROOT_DIR = micromamba_root_dir()
MICROMAMBA_PACKAGE_CACHE_DIR = MICROMAMBA_ROOT_DIR / "pkgs"


def micromamba_platform() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return "win-64"
    if system == "Darwin":
        return "osx-arm64" if machine in {"arm64", "aarch64"} else "osx-64"
    if system == "Linux":
        return "linux-aarch64" if machine in {"aarch64", "arm64"} else "linux-64"
    raise RuntimeError(f"Unsupported platform for micromamba: {system} {machine}")


def micromamba_executable_path() -> Path:
    if platform.system() == "Windows":
        return MICROMAMBA_DIR / "Library" / "bin" / "micromamba.exe"
    return MICROMAMBA_DIR / "bin" / "micromamba"


def install_micromamba(progress: Callable[[str], None] = print) -> Path | None:
    existing = shutil.which("micromamba")
    if existing:
        return Path(existing)

    target = micromamba_executable_path()
    if target.exists():
        return target

    platform_name = micromamba_platform()
    url = f"https://micro.mamba.pm/api/micromamba/{platform_name}/latest"
    MFA_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MFA_TOOLS_DIR / "micromamba.tar.bz2"
    progress(f"Downloading micromamba for {platform_name}.")
    progress(f"Source: {url}")
    progress(f"Destination: {archive}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        archive.unlink(missing_ok=True)
        progress(f"Micromamba download failed: {exc}")
        return None

    try:
        with tarfile.open(archive, "r:bz2") as handle:
            handle.extractall(MICROMAMBA_DIR)
    except tarfile.TarError:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(MICROMAMBA_DIR)
    archive.unlink(missing_ok=True)

    if not target.exists():
        progress("Micromamba archive did not contain the expected executable.")
        return None

    target.chmod(target.stat().st_mode | 0o111)
    return target


def install_mfa(progress: Callable[[str], None] = print) -> Path | None:
    existing = find_mfa_executable()
    if existing:
        progress(f"MFA executable already available: {existing}")
        return Path(existing)

    micromamba = install_micromamba(progress)
    if not micromamba:
        return None

    if MFA_ENV_DIR.exists():
        progress(f"Removing incomplete MFA environment: {MFA_ENV_DIR}")
        if not _remove_tree(MFA_ENV_DIR):
            progress("Could not remove incomplete MFA environment.")
            return None

    progress(f"Installing Montreal Forced Aligner into {MFA_ENV_DIR}")
    command = [
        str(micromamba),
        "create",
        "-y",
        "-p",
        str(MFA_ENV_DIR),
        "-c",
        "conda-forge",
        "python=3.11",
        "montreal-forced-aligner",
        "kalpy",
        "kaldi=*=cpu*",
    ]
    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(MICROMAMBA_ROOT_DIR)
    if not _run(command, progress, env=env):
        progress("MFA install failed. Clearing generated micromamba package cache for next retry.")
        _remove_tree(MFA_ENV_DIR)
        _remove_tree(MICROMAMBA_PACKAGE_CACHE_DIR)
        return None

    installed = find_mfa_executable()
    if installed:
        progress(f"MFA ready: {installed}")
        return Path(installed)
    progress("MFA install completed, but the executable was not found.")
    return None


def download_mfa_preset(preset_id: str, progress: Callable[[str], None] = print) -> bool:
    preset = mfa_preset_by_id(preset_id)
    if preset is None:
        progress(f"Unknown MFA preset: {preset_id}")
        return False

    mfa = find_mfa_executable()
    if not mfa:
        progress("Cannot download MFA models before MFA is installed.")
        return False

    MFA_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    env = _mfa_runtime_env()
    env["MFA_ROOT_DIR"] = str(MFA_ROOT_DIR)
    mfa_command = _mfa_command_prefix(mfa)
    commands: list[list[str]] = []
    existing_acoustic = find_mfa_acoustic_model(preset.acoustic_model)
    existing_dictionary = find_mfa_dictionary(preset.dictionary_model)
    if existing_acoustic:
        progress(f"MFA acoustic model already available for {preset.label}: {existing_acoustic}")
    else:
        commands.append([*mfa_command, "model", "download", "acoustic", preset.acoustic_model])
    if existing_dictionary:
        progress(f"MFA dictionary already available for {preset.label}: {existing_dictionary}")
    else:
        commands.append([*mfa_command, "model", "download", "dictionary", preset.dictionary_model])

    if not commands:
        return True

    ok = True
    for command in commands:
        progress("Running: " + subprocess.list2cmdline(command))
        if not _run(command, progress, env=env):
            ok = False
    return ok


def download_default_mfa_models(progress: Callable[[str], None] = print) -> bool:
    ok = True
    for preset_id in DEFAULT_SETUP_MFA_PRESET_IDS:
        if not download_mfa_preset(preset_id, progress):
            ok = False
    return ok


def setup_mfa(progress: Callable[[str], None] = print) -> bool:
    status = dependency_status()
    if status.mfa_path and status.mfa_acoustic_model_path and status.mfa_dictionary_path:
        progress(f"MFA executable already available: {status.mfa_path}")
        progress(f"MFA acoustic model already available: {status.mfa_acoustic_model_path}")
        progress(f"MFA dictionary already available: {status.mfa_dictionary_path}")
        return True

    install_mfa(progress)
    download_default_mfa_models(progress)
    status = dependency_status()
    if status.mfa_path and status.mfa_acoustic_model_path and status.mfa_dictionary_path:
        progress(f"MFA ready: {status.mfa_path}")
        progress(f"MFA acoustic model: {status.mfa_acoustic_model_path}")
        progress(f"MFA dictionary: {status.mfa_dictionary_path}")
        return True

    progress("MFA preset not fully available yet. You can still transcribe without HEAVY MFA alignment.")
    return False


def _mfa_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    path_parts = [
        MFA_ENV_DIR / "Library" / "bin",
        MFA_ENV_DIR / "Scripts",
        MFA_ENV_DIR,
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(str(path) for path in path_parts) + os.pathsep + existing_path
    return env


def _mfa_command_prefix(mfa_executable: str) -> list[str]:
    path = Path(mfa_executable).expanduser()
    if platform.system() == "Windows" and path.name.lower() == "mfa.exe" and path.parent.name.lower() == "scripts":
        python = path.parent.parent / "python.exe"
        if python.exists():
            return [str(python), "-m", "montreal_forced_aligner.command_line.mfa"]
    return [mfa_executable]


def _run(command: list[str], progress: Callable[[str], None], env: dict[str, str] | None = None) -> bool:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in completed.stdout.splitlines():
        if line.strip():
            progress(_safe_output(line.rstrip()))
    if completed.returncode != 0:
        progress(f"Command failed with exit code {completed.returncode}.")
        return False
    return True


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return True

    def onerror(function, item_path, _exc_info) -> None:
        try:
            item = Path(item_path)
            item.chmod(0o700)
            function(item_path)
        except OSError:
            pass

    try:
        shutil.rmtree(path, onerror=onerror)
    except OSError:
        return False
    return True


def _safe_output(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/check local Montreal Forced Aligner presets.")
    parser.add_argument("--yes", action="store_true", help="Accepted for setup-script compatibility.")
    parser.add_argument("--check-only", action="store_true", help="Only inspect existing local MFA assets.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when MFA is not ready.")
    parser.add_argument("--preset", action="append", help="Download one MFA preset id. May be specified more than once.")
    parser.add_argument("--list-presets", action="store_true", help="List bundled MFA preset ids and exit.")
    args = parser.parse_args(argv)

    if args.list_presets:
        for preset in MFA_PRESETS:
            print(f"{preset.id}\t{preset.label}\tacoustic={preset.acoustic_model}\tdictionary={preset.dictionary_model}")
        return 0

    status = dependency_status()
    if args.preset:
        install_mfa()
        ok = True
        for preset_id in args.preset:
            if not download_mfa_preset(preset_id):
                ok = False
        return 0 if ok else (1 if args.strict else 0)

    if status.mfa_path and status.mfa_acoustic_model_path and status.mfa_dictionary_path:
        print(f"MFA ready: {status.mfa_path}")
        print(f"MFA acoustic model: {status.mfa_acoustic_model_path}")
        print(f"MFA dictionary: {status.mfa_dictionary_path}")
        return 0

    ready = False if args.check_only else setup_mfa()
    return 0 if ready else (1 if args.strict else 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
