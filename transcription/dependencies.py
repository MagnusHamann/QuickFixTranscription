"""Local dependency discovery for QuickFixTranscription."""

from __future__ import annotations

import platform
import shutil
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from ffmpeg.ffmpeg_runner import PROJECT_ROOT, find_ffmpeg_tools
from transcription.font_assets import find_ipa_font_file
from transcription.dote_setup import (
    dote_model_paths,
    dote_models_are_ready,
    find_dote_executable,
    find_dote_whisper_executable,
)
from transcription.mfa_presets import DEFAULT_MFA_PRESET_ID, mfa_preset_by_id
from quickfix_sibling_apps import quickfix_app_roots

try:
    from transcription.danish_whisper_setup import find_danish_whisper_model, danish_whisper_runtime_is_ready
except ModuleNotFoundError:
    # A partial optional-backend update must not prevent the standard DOTE app from starting.
    def find_danish_whisper_model() -> str | None:
        return None

    def danish_whisper_runtime_is_ready() -> bool:
        return False


DEPENDENCY_ROOT = quickfix_app_roots(PROJECT_ROOT)[0]
WHISPER_EXECUTABLE_NAMES = (
    "whisper-cli.exe",
    "main.exe",
    "whisper.exe",
    "whisper-cli",
    "main",
    "whisper",
)
MODEL_EXTENSIONS = {".bin", ".gguf"}
MFA_EXECUTABLE_NAMES = ("mfa.exe", "mfa")
MFA_MODEL_EXTENSIONS = {".zip"}
MFA_MODEL_MARKERS = {"meta.yaml", "final.mdl", "phones.txt"}
MFA_DICTIONARY_EXTENSIONS = {".dict", ".txt", ".yaml", ".yml", ".zip"}
MFA_TOOLS_DIR = DEPENDENCY_ROOT / ".tools" / "mfa"
MFA_ENV_DIR = MFA_TOOLS_DIR / "env"
MFA_ROOT_DIR = MFA_TOOLS_DIR / "root"
PYANNOTE_TOOLS_DIR = DEPENDENCY_ROOT / ".tools" / "pyannote"
PYANNOTE_PIPELINE_CONFIG_MARKERS = {"config.yaml", "config.yml", "pipeline.yaml", "pipeline.yml"}
PYANNOTE_PIPELINE_REQUIRED_CHILDREN = {"embedding", "plda", "segmentation"}
DEFAULT_MFA_PRESET = mfa_preset_by_id(DEFAULT_MFA_PRESET_ID)
if DEFAULT_MFA_PRESET is None:
    raise RuntimeError(f"Unknown default MFA preset: {DEFAULT_MFA_PRESET_ID}")
DEFAULT_MFA_ACOUSTIC_MODEL = DEFAULT_MFA_PRESET.acoustic_model
DEFAULT_MFA_DICTIONARY_MODEL = DEFAULT_MFA_PRESET.dictionary_model


@dataclass(frozen=True)
class DependencyStatus:
    ffmpeg_path: str | None
    ffprobe_path: str | None
    whisper_path: str | None
    model_path: str | None
    ipa_font_path: str | None = None
    mfa_path: str | None = None
    mfa_acoustic_model_path: str | None = None
    mfa_dictionary_path: str | None = None
    pyannote_pipeline_path: str | None = None
    dote_application_path: str | None = None
    dote_segmentation_model_path: str | None = None
    dote_embedding_model_path: str | None = None
    sherpa_onnx_ready: bool = False
    danish_whisper_model_path: str | None = None
    danish_whisper_runtime_ready: bool = False

    @property
    def ready_for_transcription(self) -> bool:
        return bool(
            self.ffmpeg_path
            and self.ffprobe_path
            and self.whisper_path
            and self.model_path
            and self.dote_application_path
            and self.dote_segmentation_model_path
            and self.dote_embedding_model_path
            and self.sherpa_onnx_ready
        )

    @property
    def fully_ready(self) -> bool:
        return bool(
            self.ready_for_transcription
            and self.ipa_font_path
        )

    @property
    def missing_labels(self) -> list[str]:
        missing: list[str] = []
        if not self.ffmpeg_path:
            missing.append("FFmpeg")
        if not self.ffprobe_path:
            missing.append("FFprobe")
        if not self.dote_application_path:
            missing.append("DOTE Whisper 1.0.2")
        if not self.whisper_path:
            missing.append("DOTE whisper.cpp executable")
        if not self.model_path:
            missing.append("Whisper model")
        if not self.dote_segmentation_model_path:
            missing.append("DOTE segmentation model")
        if not self.dote_embedding_model_path:
            missing.append("DOTE speaker embedding model")
        if not self.sherpa_onnx_ready:
            missing.append("sherpa-onnx runtime")
        if not self.ipa_font_path:
            missing.append("IPA font")
        return missing


def find_whisper_executable() -> str | None:
    dote_whisper = find_dote_whisper_executable()
    if dote_whisper is not None:
        return str(dote_whisper)
    search_roots = []
    for root in quickfix_app_roots(PROJECT_ROOT):
        search_roots.extend(
            [
                root / ".tools" / "whisper",
                root / ".tools" / "whisper" / "Release",
                root / ".tools" / "whisper" / "bin",
                root / ".tools" / "whisper" / "build" / "bin",
                root / ".tools" / "whisper" / "build" / "bin" / "Release",
            ]
        )
    if platform.system() == "Windows":
        names = WHISPER_EXECUTABLE_NAMES
    else:
        names = tuple(name for name in WHISPER_EXECUTABLE_NAMES if not name.endswith(".exe"))

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
        if root.exists():
            for name in names:
                found = next(root.rglob(name), None)
                if found and found.exists():
                    return str(found)

    for name in names:
        found = shutil.which(name)
        if found and _looks_like_whisper_executable(Path(found), names):
            return found
    return None


def _looks_like_whisper_executable(path: Path, allowed_names: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    return path.name.lower() in {name.lower() for name in allowed_names}


def find_model_file() -> str | None:
    from transcription.model_setup import DEFAULT_MODEL_PATH

    roots = []
    for root in quickfix_app_roots(PROJECT_ROOT):
        roots.extend(
            [
                root / "models",
                root / ".tools" / "models",
                root / ".tools" / "whisper" / "models",
            ]
        )
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


def find_mfa_executable() -> str | None:
    names = MFA_EXECUTABLE_NAMES if platform.system() == "Windows" else tuple(
        name for name in MFA_EXECUTABLE_NAMES if not name.endswith(".exe")
    )
    search_roots = []
    for root in quickfix_app_roots(PROJECT_ROOT):
        mfa_tools = root / ".tools" / "mfa"
        mfa_env = mfa_tools / "env"
        search_roots.extend(
            [
                mfa_env / "Scripts",
                mfa_env / "bin",
                mfa_tools,
                mfa_tools / "bin",
                mfa_tools / "Scripts",
            ]
        )
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


def find_mfa_acoustic_model(model_name: str | None = None) -> str | None:
    target_model = model_name or DEFAULT_MFA_ACOUSTIC_MODEL
    saved = _saved_mfa_model_path("acoustic", target_model)
    if saved:
        return str(saved)
    if model_name:
        return None

    candidates: list[Path] = []
    for app_root in quickfix_app_roots(PROJECT_ROOT):
        roots = [
            app_root / "models" / "mfa",
            app_root / ".tools" / "mfa" / "models",
            app_root / ".tools" / "mfa" / "root" / "pretrained_models" / "acoustic",
        ]
        for root in roots:
            if root.exists():
                candidates.extend(path for path in root.rglob("*") if _looks_like_acoustic_model(path))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.name.lower(), len(str(path))))
    return str(candidates[0])


def find_mfa_dictionary(model_name: str | None = None) -> str | None:
    target_model = model_name or DEFAULT_MFA_DICTIONARY_MODEL
    saved = _saved_mfa_model_path("dictionary", target_model)
    if saved:
        return str(saved)
    if model_name:
        return None

    candidates: list[Path] = []
    for app_root in quickfix_app_roots(PROJECT_ROOT):
        roots = [
            app_root / "models" / "mfa" / "dictionaries",
            app_root / ".tools" / "mfa" / "models" / "dictionaries",
            app_root / ".tools" / "mfa" / "root" / "pretrained_models" / "dictionary",
        ]
        for root in roots:
            if root.exists():
                candidates.extend(
                    path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MFA_DICTIONARY_EXTENSIONS
                )
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.name.lower(), len(str(path))))
    return str(candidates[0])


def _looks_like_acoustic_model(path: Path) -> bool:
    if path.is_file() and path.suffix.lower() in MFA_MODEL_EXTENSIONS:
        return True
    if path.is_dir():
        try:
            child_names = {child.name for child in path.iterdir()}
        except OSError:
            return False
        return bool(child_names & MFA_MODEL_MARKERS)
    return False


def _saved_mfa_model_path(model_type: str, model_name: str) -> Path | None:
    if model_type == "dictionary":
        patterns = (f"{model_name}.dict", f"{model_name}.zip", model_name)
    else:
        patterns = (f"{model_name}.zip", model_name)
    for app_root in quickfix_app_roots(PROJECT_ROOT):
        root = app_root / ".tools" / "mfa" / "root" / "pretrained_models" / model_type
        if not root.exists():
            continue
        for pattern in patterns:
            candidate = root / pattern
            if candidate.exists():
                return candidate
    return None


def mfa_preset_paths(preset_id: str) -> tuple[str | None, str | None]:
    preset = mfa_preset_by_id(preset_id)
    if preset is None:
        return None, None
    return (
        find_mfa_acoustic_model(preset.acoustic_model),
        find_mfa_dictionary(preset.dictionary_model),
    )


def find_pyannote_pipeline() -> str | None:
    candidates: list[Path] = []
    for app_root in quickfix_app_roots(PROJECT_ROOT):
        roots = [
            app_root / ".tools" / "pyannote",
            app_root / "models" / "pyannote",
        ]
        for root in roots:
            if root.exists():
                if _looks_like_pyannote_pipeline(root):
                    candidates.append(root)
                for marker in PYANNOTE_PIPELINE_CONFIG_MARKERS:
                    for path in root.rglob(marker):
                        if path.is_file():
                            candidate = path.parent
                            if _looks_like_pyannote_pipeline(candidate):
                                candidates.append(candidate)
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    unique_candidates.sort(key=lambda path: (path.name.lower(), len(str(path))))
    return str(unique_candidates[0]) if unique_candidates else None


def _looks_like_pyannote_pipeline(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        child_names = {child.name.lower() for child in path.iterdir()}
    except OSError:
        return False
    return bool(child_names & PYANNOTE_PIPELINE_CONFIG_MARKERS) and PYANNOTE_PIPELINE_REQUIRED_CHILDREN.issubset(child_names)


def is_complete_pyannote_pipeline(path: str | Path) -> bool:
    """Return whether a local folder contains the required pipeline assets."""
    return _looks_like_pyannote_pipeline(Path(path).expanduser())


def dependency_status() -> DependencyStatus:
    ffmpeg, ffprobe = find_ffmpeg_tools()
    segmentation, embedding = dote_model_paths()
    models_ready = dote_models_are_ready()
    dote_app = find_dote_executable()
    return DependencyStatus(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        whisper_path=find_whisper_executable(),
        model_path=find_model_file(),
        ipa_font_path=str(font_path) if (font_path := find_ipa_font_file()) else None,
        mfa_path=find_mfa_executable(),
        mfa_acoustic_model_path=find_mfa_acoustic_model(),
        mfa_dictionary_path=find_mfa_dictionary(),
        pyannote_pipeline_path=find_pyannote_pipeline(),
        dote_application_path=str(dote_app) if dote_app else None,
        dote_segmentation_model_path=str(segmentation) if models_ready else None,
        dote_embedding_model_path=str(embedding) if models_ready else None,
        sherpa_onnx_ready=importlib.util.find_spec("sherpa_onnx") is not None,
        danish_whisper_model_path=find_danish_whisper_model(),
        danish_whisper_runtime_ready=danish_whisper_runtime_is_ready(),
    )
