"""File discovery and safe output paths for QuickFixTranscription."""

from __future__ import annotations

from pathlib import Path

from transcription.models import MediaRecord


SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
OUTPUT_FOLDER_NAME = "QuickFixTranscription"


def collect_media_files(paths: list[Path]) -> list[Path]:
    """Collect supported local media files from files or folders."""
    files: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file():
            if expanded.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
                files.append(expanded)
            continue
        if expanded.is_dir():
            folder_files = sorted(
                child
                for child in expanded.rglob("*")
                if child.is_file()
                and child.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
                and OUTPUT_FOLDER_NAME not in child.parts
            )
            files.extend(folder_files)
            continue
        raise ValueError(f"Path not found: {path}")
    return files


def human_size(size_bytes: int) -> str:
    amount = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{size_bytes} B"


def duration_label(seconds: float | None) -> str:
    if seconds is None:
        return "Unknown"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def output_directory_for(input_file: Path) -> Path:
    return input_file.parent / OUTPUT_FOLDER_NAME


def temp_directory_for(input_file: Path) -> Path:
    return output_directory_for(input_file) / ".tmp"


def unique_output_path(input_file: Path, suffix: str, extension: str) -> Path:
    output_dir = output_directory_for(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{input_file.stem}_{suffix}{extension}"
    if not candidate.exists():
        return candidate

    for index in range(2, 10000):
        numbered = output_dir / f"{input_file.stem}_{suffix}_{index}{extension}"
        if not numbered.exists():
            return numbered

    raise FileExistsError(f"Could not create a unique output name for {input_file.name}")


def probe_media_record(path: Path, runner) -> MediaRecord:
    try:
        info = runner.probe(path)
        duration = duration_label(info.get("duration"))
    except Exception:
        duration = "Unknown"

    kind = "Video" if path.suffix.lower() in {".avi", ".mkv", ".mov", ".mp4", ".webm"} else "Audio"
    return MediaRecord(
        path=path,
        duration_label=duration,
        kind_label=kind,
        size_bytes=path.stat().st_size,
    )

