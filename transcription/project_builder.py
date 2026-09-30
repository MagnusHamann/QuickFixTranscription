"""Build structured review projects from local transcript artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from transcription.event_engine import attach_suggestions_to_segments, suggestions_for_project_segments
from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.project_schema import (
    Project,
    ProjectWordToken,
    Recording,
    SegmentState,
    SpeakerTurn,
    utc_now,
)
from transcription.project_store import copy_source_to_workspace, project_workspace_for, sha256_file
from transcription.speakers import speaker_label


def build_project_from_transcript(
    source_path: Path,
    result: TranscriptResult,
    annotated_result: TranscriptResult | None = None,
    workspace: Path | None = None,
    copy_source: bool = True,
) -> Project:
    """Create a versioned project state with pending candidate suggestions."""
    workspace_path = workspace or project_workspace_for(source_path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    checksum = sha256_file(source_path) if source_path.exists() else None
    workspace_media = copy_source_to_workspace(source_path, workspace_path) if copy_source and source_path.exists() else None

    recording_id = _id("recording", source_path.name, checksum or str(source_path.resolve()))
    labels: dict[str, str] = {}
    segments = tuple(_segment_state(recording_id, index, segment, labels) for index, segment in enumerate(result.segments))
    annotated_segments = tuple(annotated_result.segments) if annotated_result else ()
    suggestions = suggestions_for_project_segments(segments, annotated_segments)
    segments = attach_suggestions_to_segments(segments, suggestions)
    speaker_turns = _speaker_turns(recording_id, segments)

    now = utc_now()
    project_id = _id("project", source_path.name, checksum or str(source_path.resolve()))
    recording = Recording(
        id=recording_id,
        source_media_path=str(source_path.resolve()),
        source_checksum=checksum,
        workspace_media_path=str(workspace_media.resolve()) if workspace_media else None,
        language=result.language,
        segments=segments,
        speaker_turns=speaker_turns,
    )
    return Project(
        id=project_id,
        name=source_path.stem,
        workspace_path=str(workspace_path.resolve()),
        created_at=now,
        updated_at=now,
        recordings=(recording,),
    )


def _segment_state(
    recording_id: str,
    index: int,
    segment: TranscriptSegment,
    labels: dict[str, str],
) -> SegmentState:
    segment_id = _id("segment", recording_id, index, segment.start, segment.end, segment.text)
    speaker = speaker_label(segment, labels)
    tokens = tuple(_token(segment_id, token_index, token, speaker) for token_index, token in enumerate(_tokens(segment)))
    confidence_values = [token.confidence for token in tokens if token.confidence is not None]
    confidence = min(confidence_values) if confidence_values else None
    return SegmentState(
        id=segment_id,
        recording_id=recording_id,
        start=segment.start,
        end=segment.end,
        speaker_label=speaker,
        raw_asr_text=segment.text,
        corrected_text=segment.text,
        tokens=tokens,
        confidence=confidence,
    )


def _tokens(segment: TranscriptSegment) -> tuple[WordToken, ...]:
    if segment.words:
        return segment.words
    words = [word for word in segment.text.split() if word.strip()]
    if not words:
        return ()
    if segment.start is None or segment.end is None or segment.end <= segment.start:
        return tuple(WordToken(text=word, speaker=segment.speaker) for word in words)
    duration = (segment.end - segment.start) / len(words)
    return tuple(
        WordToken(
            text=word,
            start=segment.start + index * duration,
            end=segment.start + (index + 1) * duration,
            speaker=segment.speaker,
        )
        for index, word in enumerate(words)
    )


def _token(segment_id: str, index: int, token: WordToken, speaker: str) -> ProjectWordToken:
    token_id = _id("word", segment_id, index, token.text, token.start, token.end)
    return ProjectWordToken(
        id=token_id,
        text=token.text,
        corrected_text=token.text,
        start=token.start,
        end=token.end,
        speaker_label=speaker,
        confidence=token.confidence,
    )


def _speaker_turns(recording_id: str, segments: tuple[SegmentState, ...]) -> tuple[SpeakerTurn, ...]:
    turns: list[SpeakerTurn] = []
    current_speaker = ""
    current_segments: list[SegmentState] = []

    def flush() -> None:
        if not current_segments:
            return
        start = current_segments[0].start
        end = current_segments[-1].end
        turns.append(
            SpeakerTurn(
                id=_id("turn", recording_id, current_speaker, start, end, len(turns)),
                speaker_label=current_speaker,
                start=start,
                end=end,
                segment_ids=tuple(segment.id for segment in current_segments),
                uncertain=len({segment.speaker_label for segment in current_segments}) > 1,
            )
        )

    for segment in segments:
        if not current_segments:
            current_speaker = segment.speaker_label
            current_segments = [segment]
            continue
        if segment.speaker_label != current_speaker:
            flush()
            current_speaker = segment.speaker_label
            current_segments = [segment]
        else:
            current_segments.append(segment)
    flush()
    return tuple(turns)


def _id(*parts: object) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
