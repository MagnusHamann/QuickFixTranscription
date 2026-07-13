"""Sequential QuickFixTranscription worker."""

from __future__ import annotations

from pathlib import Path

import shutil

from PySide6.QtCore import QObject, Signal, Slot

from ffmpeg.ffmpeg_runner import FFmpegRunner
from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.engine import WhisperCppEngine
from transcription.file_utils import temp_directory_for, unique_output_path
from transcription.font_assets import IPA_FONT_FAMILY
from transcription.jeffersonian import BROAD_JEFFERSONIAN_PROFILE, NARROW_JEFFERSONIAN_PROFILE, format_simple_jeffersonian
from transcription.media import build_audio_extract_command, build_channel_extract_command, command_to_text
from transcription.mfa_alignment import apply_mfa_word_alignment, run_mfa_alignment
from transcription.models import (
    BROAD_JEFFERSONIAN_TRANSCRIPTION,
    NARROW_JEFFERSONIAN_TRANSCRIPTION,
    TRANSCRIPTION_MODE_LABELS,
    TRANSCRIPTION_MODE_OUTPUT_SUFFIXES,
    VERBATIM_TRANSCRIPTION,
    MediaRecord,
    TranscriptResult,
    TranscriptionOptions,
)
from transcription.offline_guard import offline_processing_guard, reject_remote_source
from transcription.overlap_analysis import channels_are_distinct, has_cross_speaker_overlap, merge_channel_results
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

    def _build_broad_result(
        self,
        record: MediaRecord,
        fallback: TranscriptResult,
        temp_dir: Path,
        start_seconds: int | None,
        finish_seconds: int | None,
    ) -> TranscriptResult:
        """Return a broad transcript with local overlap evidence where available."""
        if self.engine is None or not self.runner.ffmpeg_path:
            return fallback

        try:
            probe = self.runner.probe(record.path)
            audio_channels = int(probe.get("audio_channels") or 0)
        except Exception as exc:
            self.log_message.emit(f"Broad overlap analysis skipped: could not inspect audio channels ({exc}).")
            return fallback

        if audio_channels < 2:
            self.log_message.emit("Broad overlap analysis: mono or single-channel audio, using Whisper timing only.")
            return fallback

        channel_paths = [temp_dir / f"{record.path.stem}_channel_{index + 1}.wav" for index in range(2)]
        for channel_index, channel_path in enumerate(channel_paths):
            command = build_channel_extract_command(
                self.runner.ffmpeg_path,
                record.path,
                channel_path,
                channel_index,
                start_seconds=start_seconds,
                finish_seconds=finish_seconds,
            )
            self.log_message.emit(command_to_text(command))
            return_code = self.runner.run(command, self.log_message.emit, self.cancelled)
            if return_code == -1:
                raise RuntimeError("Transcription stopped by user.")
            if return_code != 0:
                self.log_message.emit(
                    f"Broad overlap analysis skipped: could not extract channel {channel_index + 1}."
                )
                return fallback

        if not channels_are_distinct(channel_paths[0], channel_paths[1]):
            self.log_message.emit("Broad overlap analysis: channels are not distinct enough for speaker overlap detection.")
            return fallback

        self.log_message.emit("Broad overlap analysis: distinct channels found, transcribing channels locally.")
        channel_results: list[tuple[str, TranscriptResult]] = []
        for channel_index, channel_path in enumerate(channel_paths):
            result = self.engine.transcribe(
                channel_path,
                record.path,
                temp_dir,
                self.options.language_code,
                self.log_message.emit,
                self.cancelled,
            )
            channel_results.append((f"channel_{channel_index + 1}", result))

        merged = merge_channel_results(record.path, fallback, channel_results)
        if merged is None:
            self.log_message.emit("Broad overlap analysis: channel transcripts did not contain two usable speakers.")
            return fallback

        self.log_message.emit("Broad overlap analysis: using channel-separated local transcript for overlap brackets.")
        return merged

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
                with offline_processing_guard():
                    reject_remote_source(record.path)
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

                    mode = self.options.selected_mode()
                    broad_result = result
                    if mode in {BROAD_JEFFERSONIAN_TRANSCRIPTION, NARROW_JEFFERSONIAN_TRANSCRIPTION}:
                        broad_result = self._build_broad_result(
                            record,
                            result,
                            temp_dir,
                            start_seconds,
                            finish_seconds,
                        )
                    working_result = broad_result
                    if self.options.needs_mfa_alignment:
                        mfa_result = run_mfa_alignment(
                            temp_audio,
                            broad_result,
                            temp_dir,
                            self.options,
                            self.log_message.emit,
                            self.cancelled,
                        )
                        if has_cross_speaker_overlap(working_result):
                            self.log_message.emit(
                                "Narrow MFA retiming skipped for overlapping broad transcript to preserve overlap brackets."
                            )
                        else:
                            working_result = apply_mfa_word_alignment(working_result, mfa_result.words)

                    output_result = working_result.shifted(start_seconds) if start_seconds else working_result
                    output_path = unique_output_path(record.path, TRANSCRIPTION_MODE_OUTPUT_SUFFIXES[mode], ".rtf")
                    label = TRANSCRIPTION_MODE_LABELS[mode]

                    if mode == VERBATIM_TRANSCRIPTION:
                        font = IPA_FONT_FAMILY if self.options.use_ipa_font_regular else "Calibri"
                        lines = transcript_lines(output_result)
                        write_rtf(output_path, f"{label}: {record.path.name}", lines, font_name=font)
                    elif mode in {BROAD_JEFFERSONIAN_TRANSCRIPTION, NARROW_JEFFERSONIAN_TRANSCRIPTION}:
                        if mode == NARROW_JEFFERSONIAN_TRANSCRIPTION:
                            output_result = apply_local_acoustic_annotations(temp_audio, output_result, self.log_message.emit)
                        lines = format_simple_jeffersonian(
                            output_result,
                            max_text_columns=self.options.jeffersonian_line_width,
                            language_code=self.options.language_code or output_result.language,
                            profile=NARROW_JEFFERSONIAN_PROFILE if mode == NARROW_JEFFERSONIAN_TRANSCRIPTION else BROAD_JEFFERSONIAN_PROFILE,
                        )
                        write_rtf(
                            output_path,
                            f"{label}: {record.path.name}",
                            lines,
                            font_name=IPA_FONT_FAMILY if self.options.use_ipa_font_jeffersonian else "Courier New",
                            include_title=False,
                        )
                    else:
                        raise RuntimeError(f"Unsupported transcription type: {mode}")
                    self.log_message.emit(f"Created {label}: {output_path}")

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
