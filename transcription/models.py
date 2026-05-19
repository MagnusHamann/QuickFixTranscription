"""Data models for QuickFixTranscription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from transcription.time_utils import validate_time_range


@dataclass(frozen=True)
class MediaRecord:
    path: Path
    duration_label: str
    kind_label: str
    size_bytes: int


@dataclass(frozen=True)
class WordToken:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None

    def shifted(self, offset_seconds: float) -> "WordToken":
        if not offset_seconds:
            return self
        start = self.start + offset_seconds if self.start is not None else None
        end = self.end + offset_seconds if self.end is not None else None
        return WordToken(text=self.text, start=start, end=end, speaker=self.speaker, confidence=self.confidence)


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    words: tuple[WordToken, ...] = field(default_factory=tuple)

    def shifted(self, offset_seconds: float) -> "TranscriptSegment":
        if not offset_seconds:
            return self
        start = self.start + offset_seconds if self.start is not None else None
        end = self.end + offset_seconds if self.end is not None else None
        words = tuple(word.shifted(offset_seconds) for word in self.words)
        return TranscriptSegment(text=self.text, start=start, end=end, speaker=self.speaker, words=words)


@dataclass(frozen=True)
class TranscriptResult:
    source_path: Path
    language: str | None
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text.strip())

    def shifted(self, offset_seconds: float) -> "TranscriptResult":
        return TranscriptResult(
            source_path=self.source_path,
            language=self.language,
            segments=[segment.shifted(offset_seconds) for segment in self.segments],
        )


@dataclass(frozen=True)
class TranscriptionOptions:
    whisper_executable: str
    model_path: str
    language_code: str = ""
    transcribe_section: bool = False
    start_time: str = ""
    finish_time: str = ""
    jeffersonian: bool = False
    jeffersonian_line_width: int = 50
    keep_temp_audio: bool = False

    def validate(self) -> tuple[int | None, int | None]:
        if not self.whisper_executable.strip():
            raise ValueError("Choose a local whisper.cpp executable.")
        if not Path(self.whisper_executable).expanduser().exists():
            raise ValueError("The selected whisper.cpp executable was not found.")

        if not self.model_path.strip():
            raise ValueError("Choose a local Whisper model file.")
        if not Path(self.model_path).expanduser().exists():
            raise ValueError("The selected Whisper model file was not found.")

        if not 20 <= self.jeffersonian_line_width <= 200:
            raise ValueError("Jeffersonian line width must be between 20 and 200 characters.")

        if self.transcribe_section:
            return validate_time_range(self.start_time, self.finish_time)

        if self.start_time.strip() or self.finish_time.strip():
            raise ValueError("Tick Transcribe section before entering start or finish times.")

        return None, None
