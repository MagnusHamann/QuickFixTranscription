"""Standalone local pyannote diarization diagnostic."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ffmpeg.ffmpeg_runner import FFmpegRunner
from transcription.dependencies import find_pyannote_pipeline
from transcription.media import build_audio_extract_command, command_to_text
from transcription.offline_guard import offline_processing_guard
from transcription.pyannote_diarization import (
    DiarizationTurn,
    _annotation_summary,
    _load_pyannote_pipeline,
    _load_waveform,
    _select_diarization_annotation,
    _select_torch_device,
    _turns_from_annotation,
)


DEFAULT_TARGET_SPEAKERS = 2
DEFAULT_MAX_SPEAKERS = 6


@dataclass(frozen=True)
class DiarizationDiagnosticReport:
    exit_code: int
    lines: tuple[str, ...]
    turns: tuple[DiarizationTurn, ...]

    @property
    def detected_speakers(self) -> tuple[str, ...]:
        return tuple(sorted({turn.speaker for turn in self.turns}))

    @property
    def speech_segments(self) -> int:
        return len(self.turns)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _timepoint_label(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, hundredths = divmod(remainder, 100)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{hundredths:02d}"
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"


def _audio_properties_from_probe(info: dict[str, object]) -> list[str]:
    lines = [
        f"  channels={info.get('audio_channels') or 'unknown'}",
        f"  sample_rate={info.get('audio_sample_rate') or 'unknown'}",
    ]
    codec = info.get("audio_codec_name")
    if codec:
        lines.append(f"  codec={codec}")
    duration = info.get("duration")
    if isinstance(duration, (int, float)):
        lines.append(f"  duration={duration:.2f} seconds")
    else:
        lines.append("  duration=unknown")
    return lines


def _canonical_wav_properties(path: Path) -> list[str]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
    duration = frame_count / sample_rate if sample_rate else 0.0
    return [
        f"  channels={channels}",
        f"  sample_rate={sample_rate}",
        "  codec=pcm_s16le",
        f"  sample_format={sample_width * 8}-bit PCM",
        f"  duration={duration:.2f} seconds",
    ]


def _model_dependency_hint(pipeline_path: Path) -> str | None:
    config_path = pipeline_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        content = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"pyannote\.audio:\s*([0-9][0-9A-Za-z.\-+]*)", content)
    if match:
        return match.group(1)
    return None


def _build_attempts(known_speakers: int) -> list[dict[str, int]]:
    attempts: list[dict[str, int]] = []
    if known_speakers > 0:
        attempts.append({"num_speakers": known_speakers})
        if known_speakers < DEFAULT_TARGET_SPEAKERS:
            attempts.append({"min_speakers": DEFAULT_TARGET_SPEAKERS, "max_speakers": DEFAULT_TARGET_SPEAKERS})
    else:
        attempts.append({"num_speakers": DEFAULT_TARGET_SPEAKERS})
        for max_speakers in range(DEFAULT_TARGET_SPEAKERS + 1, DEFAULT_MAX_SPEAKERS + 1):
            attempts.append({"min_speakers": DEFAULT_TARGET_SPEAKERS, "max_speakers": max_speakers})
    return attempts


def diagnose_diarization(
    audio_path: Path,
    pipeline_path: str | None = None,
    known_speakers: int = 0,
    prefer_gpu: bool = True,
    runner: FFmpegRunner | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> DiarizationDiagnosticReport:
    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        if log_callback is not None:
            log_callback(message)

    runner = runner or FFmpegRunner()
    resolved_audio = audio_path.expanduser()
    resolved_pipeline_path = (pipeline_path.strip() if pipeline_path else find_pyannote_pipeline() or "").strip()

    with offline_processing_guard():
        log("Diarization diagnostic")
        log("----------------------")
        log(f"Input: {resolved_audio}")

        if not resolved_audio.exists():
            log("✗ Input file was not found.")
            return DiarizationDiagnosticReport(2, tuple(lines), ())
        if not runner.ready or not runner.ffmpeg_path or not runner.ffprobe_path:
            log("✗ FFmpeg/FFprobe are not ready locally.")
            return DiarizationDiagnosticReport(2, tuple(lines), ())

        try:
            original_probe = runner.probe(resolved_audio)
        except Exception as exc:
            log(f"✗ Could not inspect the source media: {exc}")
            return DiarizationDiagnosticReport(2, tuple(lines), ())

        log("Original:")
        for entry in _audio_properties_from_probe(original_probe):
            log(entry)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            canonical_wav = temp_root / f"{resolved_audio.stem}_canonical.wav"
            command = build_audio_extract_command(runner.ffmpeg_path, resolved_audio, canonical_wav)
            log("Canonical diarization audio:")
            log(f"  ffmpeg command={command_to_text(command)}")
            return_code = runner.run(command, log, lambda: False)
            if return_code != 0:
                log(f"✗ FFmpeg exited with code {return_code}.")
                return DiarizationDiagnosticReport(2, tuple(lines), ())

            try:
                for entry in _canonical_wav_properties(canonical_wav):
                    log(entry)
            except Exception as exc:
                log(f"✗ Could not inspect the canonical WAV: {exc}")
                return DiarizationDiagnosticReport(2, tuple(lines), ())

            if not resolved_pipeline_path:
                log("✗ No local pyannote pipeline was found.")
                return DiarizationDiagnosticReport(2, tuple(lines), ())

            pipeline_path_obj = Path(resolved_pipeline_path)
            log(f"Model: {pipeline_path_obj.name}")
            log(f"Model path: {pipeline_path_obj}")
            dependency_hint = _model_dependency_hint(pipeline_path_obj)
            if dependency_hint:
                log(f"Model dependency hint: pyannote.audio {dependency_hint}")
            installed_pyannote = _package_version('pyannote.audio')
            log(f"pyannote.audio version: {installed_pyannote}")
            log(f"torch version: {_package_version('torch')}")
            log(f"torchaudio version: {_package_version('torchaudio')}")
            log(f"transformers version: {_package_version('transformers')}")
            log(f"huggingface_hub version: {_package_version('huggingface_hub')}")
            if dependency_hint and installed_pyannote != "not installed":
                expected_prefix = ".".join(dependency_hint.split(".")[:2])
                installed_prefix = ".".join(installed_pyannote.split(".")[:2])
                if expected_prefix and installed_prefix and expected_prefix != installed_prefix:
                    log(
                        f"WARNING: pyannote.audio {installed_pyannote} does not match the model's {dependency_hint} hint."
                    )

            device, device_name = _select_torch_device(prefer_gpu)
            log(f"Requested device: {device_name}")

            try:
                Pipeline = _load_pyannote_pipeline()
                pipeline = Pipeline.from_pretrained(resolved_pipeline_path)
            except Exception as exc:
                log(f"✗ Model failed to load: {exc}")
                return DiarizationDiagnosticReport(2, tuple(lines), ())

            try:
                pipeline.to(device)
            except Exception as exc:
                if device_name == "cpu":
                    log(f"âœ— Model could not be placed on CPU: {exc}")
                    return DiarizationDiagnosticReport(2, tuple(lines), ())
                try:
                    device, device_name = _select_torch_device(False)
                    pipeline.to(device)
                except Exception as cpu_exc:
                    log(f"âœ— Model could not use the requested device or CPU: {cpu_exc}")
                    return DiarizationDiagnosticReport(2, tuple(lines), ())
            log(f"Device: {device_name}")

            try:
                audio_input = _load_waveform(canonical_wav)
            except Exception as exc:
                log(f"✗ Canonical WAV could not be loaded for pyannote: {exc}")
                return DiarizationDiagnosticReport(2, tuple(lines), ())

            attempts = _build_attempts(known_speakers)
            best_turns: tuple[DiarizationTurn, ...] = ()
            best_count = 0
            raw_output_summary: str | None = None

            for attempt in attempts:
                description = ", ".join(f"{key}={value}" for key, value in attempt.items())
                log(f"Running local pyannote.audio diarization with {description}.")
                raw_output = pipeline(audio_input, **attempt)
                raw_output_summary = _annotation_summary(raw_output)
                log(f"pyannote raw output: {raw_output_summary}")
                source_name, annotation = _select_diarization_annotation(raw_output)
                if source_name != type(raw_output).__name__:
                    log(f"Using pyannote output source: {source_name}.")
                turns = _turns_from_annotation(annotation, lambda speaker_name, _speaker_number: speaker_name)
                speaker_count = len({turn.speaker for turn in turns})
                if speaker_count > best_count:
                    best_turns = turns
                    best_count = speaker_count
                if speaker_count >= DEFAULT_TARGET_SPEAKERS:
                    break
                if speaker_count == 1 and known_speakers == 0:
                    log("Local pyannote.audio diarization still found only one speaker; retrying with a wider speaker range.")

            if raw_output_summary is None:
                log("✗ Pyannote did not produce any output.")
                return DiarizationDiagnosticReport(3, tuple(lines), ())

            if best_turns:
                log(f"Speech segments: {len(best_turns)}")
                log(f"Detected speakers: {best_count}")
                for turn in best_turns:
                    log(f"{_timepoint_label(turn.start)} - {_timepoint_label(turn.end)} {turn.speaker}")
                return DiarizationDiagnosticReport(0, tuple(lines), best_turns)

            log("Speech segments: 0")
            log("Detected speakers: 0")
            log("✗ Diarization failure: pyannote returned zero speaker turns.")
            return DiarizationDiagnosticReport(3, tuple(lines), ())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose the local pyannote diarization pipeline.")
    parser.add_argument("audio", help="Path to a local audio or video file.")
    parser.add_argument("--pipeline-path", default="", help="Path to the local pyannote pipeline folder.")
    parser.add_argument("--known-speakers", type=int, default=0, help="Known speaker count hint.")
    parser.add_argument("--prefer-gpu", action="store_true", help="Prefer GPU acceleration if available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    report = diagnose_diarization(
        Path(args.audio),
        pipeline_path=args.pipeline_path,
        known_speakers=args.known_speakers,
        prefer_gpu=bool(args.prefer_gpu and not args.cpu),
    )
    for line in report.lines:
        print(line)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
