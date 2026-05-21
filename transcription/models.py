"""Data models for QuickFixTranscription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

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
    use_mfa_alignment: bool = False
    mfa_executable: str = ""
    mfa_acoustic_model: str = ""
    mfa_dictionary: str = ""
    use_ipa_font_regular: bool = False
    use_ipa_font_jeffersonian: bool = False
    export_mfa_phone_transcript: bool = False
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

        if self.use_mfa_alignment:
            if not self.mfa_executable.strip():
                raise ValueError("Choose a local MFA executable before enabling heavy MFA alignment.")
            mfa_path = Path(self.mfa_executable).expanduser()
            if not mfa_path.exists() and shutil.which(self.mfa_executable) is None:
                raise ValueError("The selected MFA executable was not found.")

            if not self.mfa_acoustic_model.strip():
                raise ValueError("Choose a local MFA acoustic model before enabling heavy MFA alignment.")
            if not Path(self.mfa_acoustic_model).expanduser().exists():
                raise ValueError("The selected MFA acoustic model was not found.")

            if not self.mfa_dictionary.strip():
                raise ValueError("Choose a local MFA pronunciation dictionary before enabling heavy MFA alignment.")
            if not Path(self.mfa_dictionary).expanduser().exists():
                raise ValueError("The selected MFA pronunciation dictionary was not found.")

        if self.export_mfa_phone_transcript and not self.use_mfa_alignment:
            raise ValueError("Enable heavy MFA alignment before exporting an MFA phone-tier transcript.")

        if self.transcribe_section:
            return validate_time_range(self.start_time, self.finish_time)

        if self.start_time.strip() or self.finish_time.strip():
            raise ValueError("Tick Transcribe section before entering start or finish times.")

        return None, None
