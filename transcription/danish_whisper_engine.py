"""Strictly local Røst v3 Faster-Whisper inference with word timestamps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from transcription.danish_whisper_setup import (
    danish_whisper_model_is_ready,
    danish_whisper_python_executable,
)
from transcription.models import TranscriptResult, TranscriptSegment, WordToken
from transcription.offline_guard import OFFLINE_ENVIRONMENT, offline_processing_guard


def _worker(model_path: Path, manifest_path: Path, output_path: Path, prefer_gpu: bool) -> int:
    try:
        if not danish_whisper_model_is_ready(model_path):
            raise RuntimeError(f"The local Røst v3 model folder is incomplete: {model_path}")

        from faster_whisper import WhisperModel

        audio_paths = json.loads(manifest_path.read_text(encoding="utf-8"))["audio_paths"]
        results: list[dict[str, object]] = []
        gpu_fallback = ""
        with offline_processing_guard():
            device = "cuda" if prefer_gpu else "cpu"
            compute_type = "float16" if prefer_gpu else "int8"
            try:
                model = WhisperModel(str(model_path), device=device, compute_type=compute_type, local_files_only=True)
            except Exception as exc:
                if not prefer_gpu:
                    raise
                gpu_fallback = str(exc)
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(str(model_path), device=device, compute_type=compute_type, local_files_only=True)

            for audio_path in audio_paths:
                generated, info = model.transcribe(
                    str(audio_path),
                    language="da",
                    beam_size=5,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                segments: list[dict[str, object]] = []
                for segment in generated:
                    words = [
                        {
                            "text": str(word.word).strip(),
                            "start": float(word.start),
                            "end": float(word.end),
                            "confidence": float(word.probability),
                        }
                        for word in (segment.words or [])
                        if str(word.word).strip()
                    ]
                    text = str(segment.text).strip()
                    if text:
                        segments.append(
                            {
                                "text": text,
                                "start": float(segment.start),
                                "end": float(segment.end),
                                "words": words,
                            }
                        )
                results.append({"language": str(info.language or "da"), "segments": segments})

        output_path.write_text(
            json.dumps({"device": device, "gpu_fallback": gpu_fallback, "results": results}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        output_path.write_text(json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        return 1


def _decode_result(payload: dict[str, object], source_path: Path) -> TranscriptResult:
    output_segments: list[TranscriptSegment] = []
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise RuntimeError("Røst v3 returned an invalid segment list.")
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        raw_words = raw_segment.get("words", [])
        words = tuple(
            WordToken(
                str(word.get("text", "")),
                float(word["start"]),
                float(word["end"]),
                confidence=float(word["confidence"]),
            )
            for word in raw_words
            if isinstance(word, dict) and str(word.get("text", "")).strip()
        )
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        output_segments.append(
            TranscriptSegment(
                text,
                float(raw_segment["start"]),
                float(raw_segment["end"]),
                words=words,
            )
        )
    return TranscriptResult(source_path, str(payload.get("language") or "da"), output_segments)


class DanishWhisperBatchTranscriber:
    def __init__(self, model_path: str, prefer_gpu: bool) -> None:
        self.model_path = Path(model_path).expanduser()
        self.prefer_gpu = prefer_gpu
        self.current_process: subprocess.Popen[str] | None = None

    def terminate(self) -> None:
        if self.current_process is not None and self.current_process.poll() is None:
            self.current_process.terminate()

    def transcribe(
        self,
        audio_paths: list[Path],
        work_dir: Path,
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> list[TranscriptResult]:
        manifest = work_dir / "danish_whisper_input.json"
        output = work_dir / "danish_whisper_output.json"
        error_log = work_dir / "danish_whisper_error.log"
        manifest.write_text(json.dumps({"audio_paths": [str(path) for path in audio_paths]}), encoding="utf-8")
        python = danish_whisper_python_executable()
        if not python.is_file():
            raise RuntimeError("The isolated local Røst v3 Python runtime is missing.")
        command = [
            str(python),
            "-m",
            "transcription.danish_whisper_engine",
            "--worker",
            str(self.model_path),
            str(manifest),
            str(output),
        ]
        if self.prefer_gpu:
            command.append("--gpu")
        environment = os.environ.copy()
        environment.update(OFFLINE_ENVIRONMENT)
        if not self.prefer_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        log_callback("Røst v3: transcribing diarized turns locally with networking disabled.")
        started = time.monotonic()
        with error_log.open("w", encoding="utf-8") as error_handle:
            self.current_process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=environment,
                stdout=error_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            next_update = started + 30.0
            while self.current_process.poll() is None:
                if cancelled():
                    self.terminate()
                    raise RuntimeError("Røst v3 transcription stopped by user.")
                now = time.monotonic()
                if now >= next_update:
                    log_callback(f"Røst v3 is running locally ({int(now - started)} seconds elapsed).")
                    next_update = now + 30.0
                time.sleep(0.1)
            return_code = self.current_process.wait()
            self.current_process = None
        payload = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
        if return_code != 0:
            detail = str(payload.get("error") or "").strip()
            if not detail and error_log.is_file():
                detail = error_log.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(f"Local Røst v3 transcription failed: {detail or f'worker exited with code {return_code}'}")
        raw_results = payload.get("results")
        if (
            not isinstance(raw_results, list)
            or len(raw_results) != len(audio_paths)
            or not all(isinstance(result, dict) for result in raw_results)
        ):
            raise RuntimeError("Local Røst v3 returned an invalid result list.")
        device = str(payload.get("device") or "unknown")
        gpu_fallback = str(payload.get("gpu_fallback") or "").strip()
        if gpu_fallback:
            log_callback(f"Røst v3 GPU unavailable; using CPU instead: {gpu_fallback}")
        log_callback(f"Røst v3 inference device: {device}.")
        return [_decode_result(result, path) for result, path in zip(raw_results, audio_paths)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QuickFix's offline Røst v3 worker.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("This module is an internal worker; pass --worker.")
    return _worker(args.model_path, args.manifest, args.output, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
