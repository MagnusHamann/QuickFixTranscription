"""JSON storage for QuickFixTranscription projects."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from transcription.project_schema import (
    PROJECT_SCHEMA_VERSION,
    AnnotationSpan,
    CandidateSuggestion,
    EditHistoryEntry,
    ExportProfile,
    JeffersonAnnotation,
    Project,
    ProjectWordToken,
    Recording,
    ReviewFlag,
    SegmentState,
    SpeakerTurn,
    utc_now,
)


PROJECT_FILE_EXTENSION = ".qftproj"
T = TypeVar("T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_workspace_for(source_path: Path) -> Path:
    output = source_path.parent / "QuickFixTranscription"
    return output / f"{source_path.stem}_project"


def project_file_for(source_path: Path) -> Path:
    return project_workspace_for(source_path) / f"{source_path.stem}{PROJECT_FILE_EXTENSION}"


def copy_source_to_workspace(source_path: Path, workspace: Path) -> Path:
    media_dir = workspace / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    destination = media_dir / source_path.name
    if not destination.exists():
        shutil.copy2(source_path, destination)
    return destination


def save_project(project: Project, path: Path | None = None) -> Path:
    target = path or (Path(project.workspace_path) / f"{project.name}{PROJECT_FILE_EXTENSION}")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _to_jsonable(project)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def load_project(path: Path) -> Project:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("schema_version", 0))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Project schema {version} is newer than this app supports.")
    if version < 1:
        raise ValueError("Project file is missing a supported schema version.")
    return _from_dict(Project, data)


def append_history(project: Project, entry: EditHistoryEntry) -> Project:
    return Project(
        id=project.id,
        name=project.name,
        workspace_path=project.workspace_path,
        created_at=project.created_at,
        updated_at=utc_now(),
        schema_version=project.schema_version,
        recordings=project.recordings,
        export_profiles=project.export_profiles,
        edit_history=(*project.edit_history, entry),
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _coerce_value(hints.get(field.name, field.type), data[field.name])
    return cls(**kwargs)


def _coerce_value(expected_type: Any, value: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is tuple and args:
        item_type = args[0]
        return tuple(_coerce_value(item_type, item) for item in value)
    if origin is dict:
        return dict(value)
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _from_dict(expected_type, value)
    return value


PROJECT_TYPES = (
    AnnotationSpan,
    CandidateSuggestion,
    EditHistoryEntry,
    ExportProfile,
    JeffersonAnnotation,
    Project,
    ProjectWordToken,
    Recording,
    ReviewFlag,
    SegmentState,
    SpeakerTurn,
)
