"""Data models for QuickFixTranscription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

from transcription.time_utils import validate_time_range


VERBATIM_TRANSCRIPTION = "verbatim"
BROAD_JEFFERSONIAN_TRANSCRIPTION = "broad_jeffersonian"
NARROW_JEFFERSONIAN_TRANSCRIPTION = "narrow_jeffersonian"
ASR_BACKEND_DOTE_WHISPER = "dote_whisper"
ASR_BACKEND_DANISH_WHISPER = "danish_whisper"

ASR_BACKEND_LABELS = {
    ASR_BACKEND_DOTE_WHISPER: "DOTE Whisper (all languages)",
    ASR_BACKEND_DANISH_WHISPER: "Røst v3 Danish Whisper",
}

TRANSCRIPTION_MODE_LABELS = {
    VERBATIM_TRANSCRIPTION: "Verbatim transcription",
    BROAD_JEFFERSONIAN_TRANSCRIPTION: "Broad Jeffersonian transcription",
    NARROW_JEFFERSONIAN_TRANSCRIPTION: "Narrow Jeffersonian transcription",
}

TRANSCRIPTION_MODE_OUTPUT_SUFFIXES = {
    VERBATIM_TRANSCRIPTION: "verbatim",
    BROAD_JEFFERSONIAN_TRANSCRIPTION: "broad_jeffersonian",
    NARROW_JEFFERSONIAN_TRANSCRIPTION: "narrow_jeffersonian",
}


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
    speaker_uncertain: bool = False

    def shifted(self, offset_seconds: float) -> "TranscriptSegment":
        if not offset_seconds:
            return self
        start = self.start + offset_seconds if self.start is not None else None
        end = self.end + offset_seconds if self.end is not None else None
        words = tuple(word.shifted(offset_seconds) for word in self.words)
        return TranscriptSegment(
            text=self.text,
            start=start,
            end=end,
            speaker=self.speaker,
            words=words,
            speaker_uncertain=self.speaker_uncertain,
        )


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
    asr_backend: str = ASR_BACKEND_DOTE_WHISPER
    danish_model_path: str = ""
    transcription_mode: str = VERBATIM_TRANSCRIPTION
    language_code: str = ""
    transcribe_section: bool = False
    start_time: str = ""
    finish_time: str = ""
    jeffersonian: bool = False
    jeffersonian_line_width: int = 50
    prefer_gpu: bool = True
    model_profile: str = "auto"
    use_cache: bool = True
    resume_completed: bool = True
    output_rtf: bool = True
    output_json: bool = False
    block_cloud_synced_paths: bool = False
    use_mfa_alignment: bool = False
    mfa_executable: str = ""
    mfa_acoustic_model: str = ""
    mfa_dictionary: str = ""
    pyannote_pipeline_path: str = ""
    known_speakers: int = 0
    use_ipa_font_regular: bool = False
    use_ipa_font_jeffersonian: bool = False
    export_mfa_phone_transcript: bool = False
    keep_temp_audio: bool = False

    def selected_mode(self) -> str:
        if self.transcription_mode not in TRANSCRIPTION_MODE_LABELS:
            raise ValueError(f"Choose a valid transcription type: {self.transcription_mode}")
        if self.transcription_mode == VERBATIM_TRANSCRIPTION and self.jeffersonian:
            return NARROW_JEFFERSONIAN_TRANSCRIPTION if self.use_mfa_alignment else BROAD_JEFFERSONIAN_TRANSCRIPTION
        return self.transcription_mode

    @property
    def needs_mfa_alignment(self) -> bool:
        return self.selected_mode() == NARROW_JEFFERSONIAN_TRANSCRIPTION or self.use_mfa_alignment

    @property
    def is_jeffersonian(self) -> bool:
        return self.selected_mode() in {BROAD_JEFFERSONIAN_TRANSCRIPTION, NARROW_JEFFERSONIAN_TRANSCRIPTION}

    def validate(self) -> tuple[int | None, int | None]:
        mode = self.selected_mode()
        if self.asr_backend not in ASR_BACKEND_LABELS:
            raise ValueError("Choose a valid local speech-recognition engine.")
        if self.asr_backend == ASR_BACKEND_DANISH_WHISPER:
            if self.language_code not in {"", "da"}:
                raise ValueError("Røst v3 supports Danish only. Choose Danish or Auto-detect.")
            from transcription.danish_whisper_setup import (
                danish_whisper_model_is_ready,
                danish_whisper_runtime_is_ready,
            )

            if not self.danish_model_path.strip() or not danish_whisper_model_is_ready(self.danish_model_path):
                raise ValueError("Choose or install a complete local Røst v3 model folder.")
            if not danish_whisper_runtime_is_ready():
                raise ValueError("The local Røst v3 runtime is not installed. Use Install Røst v3 in Advanced setup.")
        if not self.output_rtf and not self.output_json:
            raise ValueError("Select at least one output format: RTF or JSON.")
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
        if self.model_profile not in {"auto", "fast draft", "balanced", "better accuracy", "highest accuracy"}:
            raise ValueError("Choose a valid model profile.")

        pyannote_pipeline_path = self.pyannote_pipeline_path.strip()
        if pyannote_pipeline_path and not Path(pyannote_pipeline_path).expanduser().exists():
            raise ValueError("The selected local pyannote pipeline was not found.")

        if not 0 <= self.known_speakers <= 20:
            raise ValueError("Known speaker count must be 0 for auto or between 1 and 20.")

        if mode == NARROW_JEFFERSONIAN_TRANSCRIPTION or self.use_mfa_alignment:
            if not self.mfa_executable.strip():
                raise ValueError("Choose a local MFA executable before enabling narrow Jeffersonian transcription.")
            mfa_path = Path(self.mfa_executable).expanduser()
            if not mfa_path.exists() and shutil.which(self.mfa_executable) is None:
                raise ValueError("The selected MFA executable was not found.")

            if not self.mfa_acoustic_model.strip():
                raise ValueError("Choose a local MFA acoustic model before enabling narrow Jeffersonian transcription.")
            if not Path(self.mfa_acoustic_model).expanduser().exists():
                raise ValueError("The selected MFA acoustic model was not found.")

            if not self.mfa_dictionary.strip():
                raise ValueError("Choose a local MFA pronunciation dictionary before enabling narrow Jeffersonian transcription.")
            if not Path(self.mfa_dictionary).expanduser().exists():
                raise ValueError("The selected MFA pronunciation dictionary was not found.")

        if self.export_mfa_phone_transcript and mode != NARROW_JEFFERSONIAN_TRANSCRIPTION:
            raise ValueError("Choose narrow Jeffersonian transcription before exporting an MFA phone-tier transcript.")

        if self.transcribe_section:
            return validate_time_range(self.start_time, self.finish_time)

        if self.start_time.strip() or self.finish_time.strip():
            raise ValueError("Tick Transcribe section before entering start or finish times.")

        return None, None
