"""Preflight checks for local transcription jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from ffmpeg.ffmpeg_runner import FFmpegRunner
from transcription.cache import local_cache_root, local_temp_root
from transcription.dependencies import DependencyStatus, dependency_status
from transcription.file_utils import output_directory_for
from transcription.models import (
    ASR_BACKEND_DANISH_WHISPER,
    BROAD_JEFFERSONIAN_TRANSCRIPTION,
    VERBATIM_TRANSCRIPTION,
    MediaRecord,
    TranscriptionOptions,
)
from transcription.danish_whisper_setup import danish_whisper_model_is_ready, danish_whisper_runtime_is_ready


CLOUD_SYNC_MARKERS = {
    "onedrive": "OneDrive",
    "dropbox": "Dropbox",
    "icloud": "iCloud",
    "iclouddrive": "iCloud",
    "google drive": "Google Drive",
    "googledrive": "Google Drive",
    "box": "Box",
    "mega": "MEGA",
    "synologydrive": "Synology Drive",
}
MIN_FREE_SPACE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def infos(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "info")

    def summary(self) -> str:
        if not self.issues:
            return "Preflight check passed."
        lines: list[str] = []
        for severity in ("error", "warning", "info"):
            for issue in self.issues:
                if issue.severity == severity:
                    lines.append(f"{severity.upper()}: {issue.message}")
        return "\n".join(lines)


def build_preflight_report(
    records: list[MediaRecord],
    options: TranscriptionOptions,
    runner: FFmpegRunner,
    status: DependencyStatus | None = None,
) -> PreflightReport:
    status = status or dependency_status()
    issues: list[PreflightIssue] = []

    _check_dependencies(issues, status, options, runner)
    _check_cloud_locations(issues, records, options)
    _check_disk_space(issues, records)
    _check_gpu_preference(issues, options)
    _check_model_profile(issues, options)
    _check_pipeline_cost(issues, records, options)

    if options.use_cache:
        issues.append(PreflightIssue("info", f"Local cache enabled: {local_cache_root()}"))
        issues.append(PreflightIssue("info", f"Local temp workspace: {local_temp_root()}"))
    else:
        issues.append(PreflightIssue("info", "Local transcript cache disabled for this run."))

    return PreflightReport(tuple(issues))


def cloud_sync_label(path: Path) -> str | None:
    text = str(path.expanduser()).lower()
    parts = [part.lower().replace(" ", "") for part in path.expanduser().parts]
    for marker, label in CLOUD_SYNC_MARKERS.items():
        normalized_marker = marker.replace(" ", "")
        if normalized_marker in parts:
            return label
        if len(normalized_marker) > 4 and marker in text:
            return label
    return None


def model_profile_for_path(path: str) -> str:
    name = Path(path).name.lower()
    if any(token in name for token in ("tiny", "ggml-tiny")):
        return "fast draft"
    if any(token in name for token in ("base", "ggml-base")):
        return "fast draft"
    if "small" in name:
        return "balanced"
    if "medium" in name:
        return "better accuracy"
    if "large" in name or "turbo" in name:
        return "highest accuracy"
    return "unknown"


def detect_whisper_gpu_hint(executable: str) -> str:
    path = Path(executable).expanduser()
    if not executable.strip() or not path.exists():
        return "unknown"
    try:
        completed = subprocess.run(
            [str(path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    output = completed.stdout.lower()
    if "-ng" in output or "--no-gpu" in output or "no gpu" in output or "gpu" in output:
        return "gpu-option-present"
    return "unknown"


def _check_dependencies(
    issues: list[PreflightIssue],
    status: DependencyStatus,
    options: TranscriptionOptions,
    runner: FFmpegRunner,
) -> None:
    if not runner.ready or not status.ffmpeg_path or not status.ffprobe_path:
        issues.append(PreflightIssue("error", "Local FFmpeg/FFprobe is not ready."))
    if not status.dote_application_path:
        issues.append(PreflightIssue("error", "The pinned local DOTE Whisper application is missing."))
    if not status.whisper_path and not Path(options.whisper_executable).expanduser().exists():
        issues.append(PreflightIssue("error", "DOTE's local whisper.cpp executable is missing."))
    if not status.model_path and not Path(options.model_path).expanduser().exists():
        issues.append(PreflightIssue("error", "Local Whisper model is missing."))
    if not status.dote_segmentation_model_path or not status.dote_embedding_model_path:
        issues.append(PreflightIssue("error", "DOTE's pinned local speaker-diarization models are missing."))
    if not status.sherpa_onnx_ready:
        issues.append(PreflightIssue("error", "The pinned local sherpa-onnx runtime is missing."))
    if options.asr_backend == ASR_BACKEND_DANISH_WHISPER:
        if not options.danish_model_path.strip() or not danish_whisper_model_is_ready(options.danish_model_path):
            issues.append(PreflightIssue("error", "The selected local Røst v3 model folder is missing or incomplete."))
        if not danish_whisper_runtime_is_ready():
            issues.append(PreflightIssue("error", "The local Røst v3 Python runtime is incomplete."))


def _check_cloud_locations(
    issues: list[PreflightIssue],
    records: list[MediaRecord],
    options: TranscriptionOptions,
) -> None:
    reported: set[tuple[str, str]] = set()
    for record in records:
        for path in (record.path, output_directory_for(record.path)):
            label = cloud_sync_label(path)
            if not label:
                continue
            key = (label, str(path.parent if path.suffix else path))
            if key in reported:
                continue
            reported.add(key)
            message = (
                f"{label} appears in the source/output path. The app processes locally, "
                "but that folder is not local and may sync files outside the computer."
            )
            issues.append(PreflightIssue("error" if options.block_cloud_synced_paths else "warning", message))


def _check_disk_space(issues: list[PreflightIssue], records: list[MediaRecord]) -> None:
    total_source_bytes = sum(max(0, record.size_bytes) for record in records)
    estimated_bytes = max(MIN_FREE_SPACE_BYTES, total_source_bytes * 3)
    roots = {local_temp_root().anchor or str(local_temp_root())}
    roots.update((output_directory_for(record.path).anchor or str(output_directory_for(record.path))) for record in records)
    for root in sorted(roots):
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            continue
        if free < estimated_bytes:
            issues.append(PreflightIssue("error", f"Low disk space on {root}: free space is below the estimated working need."))
        elif free < estimated_bytes * 2:
            issues.append(PreflightIssue("warning", f"Disk space on {root} is tight for this batch."))


def _check_gpu_preference(issues: list[PreflightIssue], options: TranscriptionOptions) -> None:
    if not options.prefer_gpu:
        engine = "Røst v3 Faster-Whisper" if options.asr_backend == ASR_BACKEND_DANISH_WHISPER else "whisper.cpp"
        issues.append(PreflightIssue("info", f"GPU preference is off; {engine} will be forced into CPU mode."))
        return
    hint = detect_whisper_gpu_hint(options.whisper_executable)
    if hint == "gpu-option-present":
        engine = "Røst v3 Faster-Whisper" if options.asr_backend == ASR_BACKEND_DANISH_WHISPER else "the selected whisper.cpp binary"
        issues.append(PreflightIssue("info", f"GPU preference is on; {engine} can use available local GPU support."))
    else:
        issues.append(
            PreflightIssue(
                "warning",
                "GPU preference is on, but the selected whisper.cpp binary did not clearly report GPU support. It may still run on CPU.",
            )
        )


def _check_model_profile(issues: list[PreflightIssue], options: TranscriptionOptions) -> None:
    actual = model_profile_for_path(options.model_path)
    requested = options.model_profile
    if requested == "auto":
        issues.append(PreflightIssue("info", f"Selected model profile appears to be: {actual}."))
        return
    if actual == "unknown":
        issues.append(PreflightIssue("warning", f"Could not infer the selected model profile for requested '{requested}' mode."))
    elif actual != requested:
        issues.append(PreflightIssue("warning", f"Requested '{requested}' model profile, but selected model looks like '{actual}'."))
    else:
        issues.append(PreflightIssue("info", f"Selected model matches requested '{requested}' profile."))


def _check_pipeline_cost(issues: list[PreflightIssue], records: list[MediaRecord], options: TranscriptionOptions) -> None:
    mode = options.selected_mode()
    if mode == VERBATIM_TRANSCRIPTION:
        engine = (
            "local Røst v3 Faster-Whisper with native word timestamps"
            if options.asr_backend == ASR_BACKEND_DANISH_WHISPER
            else "whisper.cpp per speaker turn"
        )
        issues.append(
            PreflightIssue(
                "info",
                f"Verbatim mode uses the integrated local DOTE pipeline: FFmpeg, Sherpa-ONNX diarization, then {engine}.",
            )
        )
    elif mode == BROAD_JEFFERSONIAN_TRANSCRIPTION:
        issues.append(
            PreflightIssue(
                "info",
                "Broad Jeffersonian mode builds on the same DOTE verbatim result, then adds QuickFix temporal, sequential, pause, and overlap notation.",
            )
        )
    if len(records) > 1 and options.resume_completed:
        issues.append(PreflightIssue("info", "Resume mode is on; existing selected transcript outputs will be skipped."))
