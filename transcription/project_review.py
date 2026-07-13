"""Review-state updates for project candidate suggestions."""

from __future__ import annotations

import hashlib

from transcription.project_schema import (
    EditHistoryEntry,
    JeffersonAnnotation,
    Project,
    Recording,
    SegmentState,
    utc_now,
)


def set_candidate_status(project: Project, candidate_id: str, status: str) -> Project:
    if status not in {"pending", "confirmed", "rejected"}:
        raise ValueError(f"Unsupported candidate status: {status}")

    changed = False
    recordings: list[Recording] = []
    before: dict[str, object] = {}
    after: dict[str, object] = {}

    for recording in project.recordings:
        segments: list[SegmentState] = []
        for segment in recording.segments:
            candidates = []
            confirmed = list(segment.confirmed_annotations)
            for candidate in segment.candidate_annotations:
                if candidate.id != candidate_id:
                    candidates.append(candidate)
                    continue
                changed = True
                before = {"status": candidate.status}
                confirmed = [annotation for annotation in confirmed if annotation.source_candidate_id != candidate.id]
                updated = type(candidate)(
                    id=candidate.id,
                    kind=candidate.kind,
                    label=candidate.label,
                    detail=candidate.detail,
                    span=candidate.span,
                    confidence=candidate.confidence,
                    suggested_symbol=candidate.suggested_symbol,
                    status=status,
                    source=candidate.source,
                )
                after = {"status": status}
                candidates.append(updated)
                if status == "confirmed":
                    confirmed.append(
                        JeffersonAnnotation(
                            id=_id("annotation", candidate.id),
                            kind=candidate.kind,
                            symbol=candidate.suggested_symbol,
                            span=candidate.span,
                            source_candidate_id=candidate.id,
                        )
                    )
            segments.append(
                SegmentState(
                    id=segment.id,
                    recording_id=segment.recording_id,
                    start=segment.start,
                    end=segment.end,
                    speaker_label=segment.speaker_label,
                    raw_asr_text=segment.raw_asr_text,
                    corrected_text=segment.corrected_text,
                    tokens=segment.tokens,
                    candidate_annotations=tuple(candidates),
                    confirmed_annotations=tuple(confirmed),
                    review_flags=segment.review_flags,
                    confidence=segment.confidence,
                    notes=segment.notes,
                )
            )
        recordings.append(
            Recording(
                id=recording.id,
                source_media_path=recording.source_media_path,
                source_checksum=recording.source_checksum,
                workspace_media_path=recording.workspace_media_path,
                language=recording.language,
                duration_seconds=recording.duration_seconds,
                segments=tuple(segments),
                speaker_turns=recording.speaker_turns,
            )
        )

    if not changed:
        raise ValueError(f"Candidate not found: {candidate_id}")

    entry = EditHistoryEntry(
        id=_id("history", candidate_id, status, utc_now()),
        timestamp=utc_now(),
        action=f"candidate_{status}",
        target_id=candidate_id,
        before=before,
        after=after,
    )
    return Project(
        id=project.id,
        name=project.name,
        workspace_path=project.workspace_path,
        created_at=project.created_at,
        updated_at=utc_now(),
        schema_version=project.schema_version,
        recordings=tuple(recordings),
        export_profiles=project.export_profiles,
        edit_history=(*project.edit_history, entry),
    )


def _id(*parts: object) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
