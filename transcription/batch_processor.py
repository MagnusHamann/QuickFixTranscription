"""Sequential QuickFixTranscription worker."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ffmpeg.ffmpeg_runner import FFmpegRunner
from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.engine import WhisperCppEngine
from transcription.file_utils import temp_directory_for, unique_output_path
from transcription.font_assets import IPA_FONT_FAMILY
from transcription.jeffersonian import format_simple_jeffersonian
from transcription.media import build_audio_extract_command, command_to_text
from transcription.mfa_alignment import apply_mfa_word_alignment, phone_tier_lines, run_mfa_alignment
from transcription.models import MediaRecord, TranscriptionOptions
from transcription.rtf_exporter import transcript_lines, write_rtf


class TranscriptionBatchProcessor(QObject):
    """Runs local extraction, local transcription, and RTF export."""

    log_message = Signal(str)
    progress_changed = Signal(int)
    status_changed = Signal(str)
    finished = Signal(int, int)

    def __init__(self, records: list[MediaRecord], options: TranscriptionOptions, runner: FFmpegRunner) -> None:
        super().__init__()
        self.records = records
        self.options = options
        self.runner = runner
        self._cancelled = False
        self.engine: WhisperCppEngine | None = None

    def cancel(self) -> None:
        self._cancelled = True
        self.runner.terminate()
        if self.engine:
            self.engine.terminate()

    def cancelled(self) -> bool:
        return self._cancelled

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        if self.options.keep_temp_audio:
            return
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    @Slot()
    def run(self) -> None:
        completed = 0
        failed = 0
        total = len(self.records)

        try:
            start_seconds, finish_seconds = self.options.validate()
        except Exception as exc:
            self.log_message.emit(f"Invalid transcription options: {exc}")
            self.finished.emit(0, total)
            return

        self.engine = WhisperCppEngine(self.options.whisper_executable, self.options.model_path)

        for index, record in enumerate(self.records, start=1):
            if self._cancelled:
                self.log_message.emit("Transcription cancelled.")
                break

            self.status_changed.emit(f"Transcribing {index} of {total}: {record.path.name}")
            self.log_message.emit("")
            self.log_message.emit(f"Processing: {record.path}")
            temp_dir = temp_directory_for(record.path)
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_audio = temp_dir / f"{record.path.stem}_transcription.wav"

            try:
                if not self.runner.ffmpeg_path:
                    raise RuntimeError("FFmpeg was not found.")
                command = build_audio_extract_command(
                    self.runner.ffmpeg_path,
                    record.path,
                    temp_audio,
                    start_seconds=start_seconds,
                    finish_seconds=finish_seconds,
                )
                self.log_message.emit(command_to_text(command))
                return_code = self.runner.run(command, self.log_message.emit, self.cancelled)
                if return_code == -1:
                    self.log_message.emit("Stopped by user.")
                    break
                if return_code != 0:
                    raise RuntimeError(f"FFmpeg exited with code {return_code}.")

                result = self.engine.transcribe(
                    temp_audio,
                    record.path,
                    temp_dir,
                    self.options.language_code,
                    self.log_message.emit,
                    self.cancelled,
                )

                working_result = result
                mfa_result = None
                if self.options.use_mfa_alignment:
                    mfa_result = run_mfa_alignment(
                        temp_audio,
                        working_result,
                        temp_dir,
                        self.options,
                        self.log_message.emit,
                        self.cancelled,
                    )
                    working_result = apply_mfa_word_alignment(working_result, mfa_result.words)
                    if mfa_result.textgrid_path:
                        textgrid_path = unique_output_path(record.path, "mfa_alignment", ".TextGrid")
                        shutil.copy2(mfa_result.textgrid_path, textgrid_path)
                        self.log_message.emit(f"Created: {textgrid_path}")

                raw_result = working_result.shifted(start_seconds) if start_seconds else working_result

                raw_path = unique_output_path(record.path, "transcript", ".rtf")
                raw_font = IPA_FONT_FAMILY if self.options.use_ipa_font_regular else "Calibri"
                write_rtf(raw_path, f"Transcript: {record.path.name}", transcript_lines(raw_result), font_name=raw_font)
                self.log_message.emit(f"Created: {raw_path}")

                if self.options.export_mfa_phone_transcript and mfa_result:
                    phone_path = unique_output_path(record.path, "mfa_phones", ".rtf")
                    phone_result = working_result.shifted(start_seconds) if start_seconds else working_result
                    write_rtf(
                        phone_path,
                        f"MFA phone-tier transcript: {record.path.name}",
                        phone_tier_lines(phone_result, mfa_result.phones),
                        font_name=IPA_FONT_FAMILY,
                    )
                    self.log_message.emit(f"Created: {phone_path}")

                if self.options.jeffersonian:
                    jeffersonian_result = apply_local_acoustic_annotations(temp_audio, working_result, self.log_message.emit)
                    if start_seconds:
                        jeffersonian_result = jeffersonian_result.shifted(start_seconds)
                    jeffersonian_path = unique_output_path(record.path, "jeffersonian", ".rtf")
                    write_rtf(
                        jeffersonian_path,
                        f"Simple Jeffersonian transcript: {record.path.name}",
                        format_simple_jeffersonian(
                            jeffersonian_result,
                            max_text_columns=self.options.jeffersonian_line_width,
                            language_code=self.options.language_code or jeffersonian_result.language,
                        ),
                        font_name=IPA_FONT_FAMILY if self.options.use_ipa_font_jeffersonian else "Courier New",
                        include_title=False,
                    )
                    self.log_message.emit(f"Created: {jeffersonian_path}")

                completed += 1
            except Exception as exc:
                failed += 1
                self.log_message.emit(f"Failed: {exc}")
            finally:
                self._cleanup_temp_dir(temp_dir)

            self.progress_changed.emit(int((index / total) * 100))

        if total == 0:
            self.progress_changed.emit(0)
        elif not self._cancelled:
            self.progress_changed.emit(100)

        self.finished.emit(completed, failed)
