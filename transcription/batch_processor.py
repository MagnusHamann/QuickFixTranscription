"""Sequential QuickFixTranscription worker."""

from __future__ import annotations

from pathlib import Path

import shutil

from PySide6.QtCore import QObject, Signal, Slot

from ffmpeg.ffmpeg_runner import FFmpegRunner
from transcription.acoustic_analysis import apply_local_acoustic_annotations
from transcription.cache import cache_key, load_cached_transcript, save_cached_transcript
from transcription.dote_engine import DoteWhisperEngine
from transcription.file_utils import default_output_path, temp_directory_for
from transcription.font_assets import IPA_FONT_FAMILY
from transcription.jeffersonian import BROAD_JEFFERSONIAN_PROFILE, NARROW_JEFFERSONIAN_PROFILE, format_simple_jeffersonian
from transcription.json_exporter import has_current_json_output, write_json_transcript
from transcription.media import build_audio_extract_command, command_to_text
from transcription.mfa_alignment import apply_mfa_word_alignment, run_mfa_alignment
from transcription.models import (
    ASR_BACKEND_DANISH_WHISPER,
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
from transcription.overlap_analysis import has_cross_speaker_overlap
from transcription.rtf_exporter import has_current_output_version, transcript_lines, write_rtf


class TranscriptionBatchProcessor(QObject):
    """Runs local extraction, local transcription, and selected exports."""

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
        self.engine: DoteWhisperEngine | None = None

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

    def _selected_output_key(self, record: MediaRecord, mode: str) -> str:
        base_pipeline = "dote-1.0.2-roest-v3-v1" if self.options.asr_backend == ASR_BACKEND_DANISH_WHISPER else "dote-1.0.2"
        return cache_key(
            "selected_output",
            record.path,
            self.options,
            {
                "mode": mode,
                "jeffersonian_line_width": self.options.jeffersonian_line_width,
                "known_speakers": self.options.known_speakers,
                "base_pipeline": base_pipeline,
                "use_ipa_font_regular": self.options.use_ipa_font_regular,
                "use_ipa_font_jeffersonian": self.options.use_ipa_font_jeffersonian,
            },
        )

    def _transcribe_with_cache(
        self,
        stage: str,
        audio_path: Path,
        source_path: Path,
        work_dir: Path,
        extra: dict[str, object] | None = None,
    ) -> TranscriptResult:
        if self.engine is None:
            raise RuntimeError("Transcription engine was not initialized.")
        key = cache_key(stage, source_path, self.options, extra)
        if self.options.use_cache:
            cached = load_cached_transcript(stage, key, source_path)
            if cached is not None:
                self.log_message.emit(f"Using cached local {stage.replace('_', ' ')} result.")
                return cached

        result = self.engine.transcribe(
            audio_path,
            source_path,
            work_dir,
            self.options.language_code,
            self.log_message.emit,
            self.cancelled,
        )
        if self.options.use_cache:
            try:
                save_cached_transcript(stage, key, result)
                self.log_message.emit(f"Cached local {stage.replace('_', ' ')} result for faster reruns.")
            except Exception as exc:
                self.log_message.emit(f"Local cache save skipped: {exc}")
        return result

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

        self.engine = DoteWhisperEngine(
            self.options.whisper_executable,
            self.options.model_path,
            self.options.prefer_gpu,
            known_speakers=self.options.known_speakers,
            asr_backend=self.options.asr_backend,
            danish_model_path=self.options.danish_model_path,
        )

        for index, record in enumerate(self.records, start=1):
            if self._cancelled:
                self.log_message.emit("Transcription cancelled.")
                break

            self.status_changed.emit(f"Transcribing {index} of {total}: {record.path.name}")
            self.log_message.emit("")
            self.log_message.emit(f"Processing: {record.path}")
            mode = self.options.selected_mode()
            selected_output_key = self._selected_output_key(record, mode)
            if self.options.resume_completed and not self.options.transcribe_section:
                suffix = TRANSCRIPTION_MODE_OUTPUT_SUFFIXES[mode]
                expected_outputs: list[tuple[Path, bool]] = []
                if self.options.output_rtf:
                    rtf_output = default_output_path(record.path, suffix, ".rtf")
                    expected_outputs.append(
                        (rtf_output, rtf_output.exists() and has_current_output_version(rtf_output, selected_output_key))
                    )
                if self.options.output_json:
                    json_output = default_output_path(record.path, suffix, ".json")
                    expected_outputs.append(
                        (json_output, json_output.exists() and has_current_json_output(json_output, selected_output_key))
                    )
                if expected_outputs and all(is_current for _path, is_current in expected_outputs):
                    completed += 1
                    outputs = ", ".join(str(path) for path, _is_current in expected_outputs)
                    self.log_message.emit(f"Skipping existing selected transcript output(s): {outputs}")
                    self.progress_changed.emit(int((index / total) * 100))
                    continue
                for expected_output, is_current in expected_outputs:
                    if expected_output.exists() and not is_current:
                        self.log_message.emit(f"Refreshing outdated selected transcript: {expected_output}")

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

                    result = self._transcribe_with_cache(
                        "dote_base",
                        temp_audio,
                        record.path,
                        temp_dir,
                        {
                            "audio_role": "main",
                            "base_pipeline": (
                                "dote-1.0.2-roest-v3-v1"
                                if self.options.asr_backend == ASR_BACKEND_DANISH_WHISPER
                                else "dote-1.0.2"
                            ),
                            "known_speakers": self.options.known_speakers,
                        },
                    )
                    working_result = result
                    if self.options.needs_mfa_alignment:
                        mfa_result = run_mfa_alignment(
                            temp_audio,
                            result,
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
                    label = TRANSCRIPTION_MODE_LABELS[mode]
                    include_title = True

                    if mode == VERBATIM_TRANSCRIPTION:
                        font = IPA_FONT_FAMILY if self.options.use_ipa_font_regular else "Calibri"
                        lines = transcript_lines(output_result)
                    elif mode in {BROAD_JEFFERSONIAN_TRANSCRIPTION, NARROW_JEFFERSONIAN_TRANSCRIPTION}:
                        if mode == NARROW_JEFFERSONIAN_TRANSCRIPTION:
                            output_result = apply_local_acoustic_annotations(temp_audio, output_result, self.log_message.emit)
                        lines = format_simple_jeffersonian(
                            output_result,
                            max_text_columns=self.options.jeffersonian_line_width,
                            language_code=self.options.language_code or output_result.language,
                            profile=NARROW_JEFFERSONIAN_PROFILE if mode == NARROW_JEFFERSONIAN_TRANSCRIPTION else BROAD_JEFFERSONIAN_PROFILE,
                        )
                        font = IPA_FONT_FAMILY if self.options.use_ipa_font_jeffersonian else "Courier New"
                        include_title = False
                    else:
                        raise RuntimeError(f"Unsupported transcription type: {mode}")

                    suffix = TRANSCRIPTION_MODE_OUTPUT_SUFFIXES[mode]
                    if self.options.output_rtf:
                        output_path = default_output_path(record.path, suffix, ".rtf")
                        write_rtf(
                            output_path,
                            f"{label}: {record.path.name}",
                            lines,
                            font_name=font,
                            include_title=include_title,
                            output_key=selected_output_key,
                        )
                        self.log_message.emit(f"Created {label} RTF: {output_path}")
                    if self.options.output_json:
                        output_path = default_output_path(record.path, suffix, ".json")
                        write_json_transcript(
                            output_path,
                            output_result,
                            mode,
                            lines,
                            output_key=selected_output_key,
                        )
                        self.log_message.emit(f"Created {label} JSON: {output_path}")

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
