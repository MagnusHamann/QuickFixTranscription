"""Versioned project objects for human-in-the-loop transcription review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class AnnotationSpan:
    id: str
    start: float | None = None
    end: float | None = None
    segment_id: str | None = None
    word_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectWordToken:
    id: str
    text: str
    corrected_text: str
    start: float | None = None
    end: float | None = None
    speaker_label: str = "SP1"
    confidence: float | None = None


@dataclass(frozen=True)
class CandidateSuggestion:
    id: str
    kind: str
    label: str
    detail: str
    span: AnnotationSpan
    confidence: float | None = None
    suggested_symbol: str = ""
    status: str = "pending"
    source: str = "local_rule_engine"


@dataclass(frozen=True)
class JeffersonAnnotation:
    id: str
    kind: str
    symbol: str
    span: AnnotationSpan
    source_candidate_id: str | None = None
    note: str = ""


@dataclass(frozen=True)
class ReviewFlag:
    id: str
    kind: str
    message: str
    span: AnnotationSpan
    resolved: bool = False


@dataclass(frozen=True)
class SpeakerTurn:
    id: str
    speaker_label: str
    start: float | None
    end: float | None
    segment_ids: tuple[str, ...] = ()
    uncertain: bool = False


@dataclass(frozen=True)
class SegmentState:
    id: str
    recording_id: str
    start: float | None
    end: float | None
    speaker_label: str
    raw_asr_text: str
    corrected_text: str
    tokens: tuple[ProjectWordToken, ...] = ()
    candidate_annotations: tuple[CandidateSuggestion, ...] = ()
    confirmed_annotations: tuple[JeffersonAnnotation, ...] = ()
    review_flags: tuple[ReviewFlag, ...] = ()
    confidence: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class Recording:
    id: str
    source_media_path: str
    source_checksum: str | None = None
    workspace_media_path: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    segments: tuple[SegmentState, ...] = ()
    speaker_turns: tuple[SpeakerTurn, ...] = ()


@dataclass(frozen=True)
class ExportProfile:
    id: str = "default"
    line_width: int = 50
    include_line_numbers: bool = True
    include_pending_suggestions: bool = False
    format_name: str = "jefferson_text"


@dataclass(frozen=True)
class EditHistoryEntry:
    id: str
    timestamp: str
    action: str
    target_id: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    workspace_path: str
    created_at: str
    updated_at: str
    schema_version: int = PROJECT_SCHEMA_VERSION
    recordings: tuple[Recording, ...] = ()
    export_profiles: tuple[ExportProfile, ...] = (ExportProfile(),)
    edit_history: tuple[EditHistoryEntry, ...] = ()

    @property
    def path(self) -> Path:
        return Path(self.workspace_path)
