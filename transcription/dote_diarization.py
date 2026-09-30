"""Offline Sherpa-ONNX speaker diarization used by the integrated DOTE pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
from collections.abc import Callable, Iterable

from transcription.dote_setup import dote_model_paths
from transcription.offline_guard import OFFLINE_ENVIRONMENT
from transcription.pyannote_diarization import DiarizationResult, DiarizationTurn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTE_CLUSTERING_THRESHOLD = 0.5
DOTE_MIN_SPEECH_SECONDS = 0.3
DOTE_MIN_SILENCE_SECONDS = 0.5
DOTE_MERGE_SAME_SPEAKER_GAP_SECONDS = 1.0


def _offline_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(OFFLINE_ENVIRONMENT)
    return env


def _read_canonical_wav(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"numpy is required by the local DOTE diarization runtime: {exc}") from exc

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)

    if channels != 1 or sample_rate != 16000 or sample_width != 2:
        raise RuntimeError(
            "DOTE diarization requires the canonical FFmpeg WAV "
            f"(mono, 16000 Hz, pcm_s16le); got channels={channels}, "
            f"sample_rate={sample_rate}, sample_width={sample_width}."
        )
    if frame_count <= 0:
        raise RuntimeError("The canonical diarization WAV has zero duration.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, frame_count / sample_rate


def _field(value: object, *names: str) -> object | None:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _raw_segments(result: object) -> Iterable[object]:
    sort_method = getattr(result, "sort_by_start_time", None)
    if callable(sort_method):
        sorted_segments = sort_method()
        if sorted_segments is not None:
            return tuple(sorted_segments)
    num_segments = getattr(result, "num_segments", None)
    get_segment = getattr(result, "get", None)
    if isinstance(num_segments, int) and callable(get_segment):
        return tuple(get_segment(index) for index in range(num_segments))
    segments = getattr(result, "segments", None)
    if segments is not None:
        return tuple(segments)
    try:
        return tuple(result)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(
            f"Unsupported sherpa-onnx diarization result type: {type(result).__name__}."
        ) from exc


def _speaker_name(value: object) -> str:
    speaker = _field(value, "speaker", "speaker_id", "label")
    if speaker is None and isinstance(value, (tuple, list)) and len(value) >= 3:
        speaker = value[2]
    if speaker is None:
        raise RuntimeError("A Sherpa diarization segment did not contain a speaker identifier.")
    return str(speaker)


def _segment_bounds(value: object) -> tuple[float, float]:
    start = _field(value, "start", "start_time")
    end = _field(value, "end", "end_time")
    if (start is None or end is None) and isinstance(value, (tuple, list)) and len(value) >= 2:
        start, end = value[0], value[1]
    try:
        return float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("A Sherpa diarization segment had invalid timestamps.") from exc


def _map_and_merge_turns(raw_segments: Iterable[object]) -> tuple[DiarizationTurn, ...]:
    parsed: list[tuple[float, float, str]] = []
    for value in raw_segments:
        start, end = _segment_bounds(value)
        if end > start:
            parsed.append((max(0.0, start), end, _speaker_name(value)))
    parsed.sort(key=lambda item: (item[0], item[1], item[2]))

    speaker_map: dict[str, str] = {}
    mapped: list[DiarizationTurn] = []
    for start, end, raw_speaker in parsed:
        speaker = speaker_map.setdefault(raw_speaker, f"SP{len(speaker_map) + 1}")
        if (
            mapped
            and mapped[-1].speaker == speaker
            and start <= mapped[-1].end + DOTE_MERGE_SAME_SPEAKER_GAP_SECONDS
        ):
            previous = mapped[-1]
            mapped[-1] = DiarizationTurn(previous.start, max(previous.end, end), speaker)
        else:
            mapped.append(DiarizationTurn(start, end, speaker))
    return tuple(mapped)


def _run_worker(audio_path: Path, output_path: Path, known_speakers: int, dependency_root: Path | None) -> int:
    try:
        import sherpa_onnx

        samples, duration = _read_canonical_wav(audio_path)
        segmentation_path, embedding_path = dote_model_paths(dependency_root)
        for label, model_path in (("segmentation", segmentation_path), ("embedding", embedding_path)):
            if not model_path.is_file():
                raise RuntimeError(f"The local DOTE {label} model is missing: {model_path}")

        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_path)
            ),
            num_threads=max(1, min(4, os.cpu_count() or 1)),
            provider="cpu",
        )
        embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_path),
            num_threads=max(1, min(4, os.cpu_count() or 1)),
            provider="cpu",
        )
        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=known_speakers if known_speakers > 0 else -1,
            threshold=DOTE_CLUSTERING_THRESHOLD,
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            clustering=clustering,
            min_duration_on=DOTE_MIN_SPEECH_SECONDS,
            min_duration_off=DOTE_MIN_SILENCE_SECONDS,
        )
        diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        result = diarizer.process(samples)
        turns = _map_and_merge_turns(_raw_segments(result))
        if not turns:
            raise RuntimeError("DOTE Sherpa diarization returned no speech or speaker turns.")

        payload = {
            "sherpa_onnx_version": getattr(sherpa_onnx, "__version__", "unknown"),
            "device": "cpu",
            "duration": duration,
            "turns": [
                {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
                for turn in turns
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        output_path.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 1


class DoteDiarizer:
    """Runs native Sherpa inference out of process so failures are explicit and cancellable."""

    def __init__(self, known_speakers: int = 0, dependency_root: Path | None = None) -> None:
        self.known_speakers = known_speakers
        self.dependency_root = dependency_root
        self.current_process: subprocess.Popen[str] | None = None

    def terminate(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()

    def diarize(
        self,
        audio_path: Path,
        work_dir: Path,
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> DiarizationResult:
        output_path = work_dir / f"{audio_path.stem}_dote_diarization.json"
        error_path = work_dir / f"{audio_path.stem}_dote_diarization.log"
        command = [
            sys.executable,
            "-m",
            "transcription.dote_diarization",
            "--worker",
            str(audio_path),
            str(output_path),
            "--known-speakers",
            str(self.known_speakers),
        ]
        if self.dependency_root is not None:
            command.extend(["--dependency-root", str(self.dependency_root)])

        log_callback("DOTE base pipeline: running local Sherpa-ONNX speaker diarization first.")
        log_callback("Diarization device: CPU (Whisper may use the GPU separately).")
        start_time = time.monotonic()
        with error_path.open("w", encoding="utf-8") as error_handle:
            self.current_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=_offline_environment(),
                stdout=error_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            next_update = start_time + 30.0
            while self.current_process.poll() is None:
                if cancelled():
                    self.terminate()
                    raise RuntimeError("DOTE diarization stopped by user.")
                now = time.monotonic()
                if now >= next_update:
                    log_callback(f"DOTE diarization is running locally ({int(now - start_time)} seconds elapsed).")
                    next_update = now + 30.0
                time.sleep(0.1)
            return_code = self.current_process.wait()
            self.current_process = None

        payload: dict[str, object] = {}
        if output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        if return_code != 0:
            detail = str(payload.get("error") or "").strip()
            if not detail and error_path.is_file():
                detail = error_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(f"DOTE speaker diarization failed: {detail or f'worker exited with code {return_code}'}")

        turns = tuple(
            DiarizationTurn(float(item["start"]), float(item["end"]), str(item["speaker"]))
            for item in payload.get("turns", [])  # type: ignore[union-attr]
            if isinstance(item, dict)
        )
        speakers = sorted({turn.speaker for turn in turns})
        if not turns or not speakers:
            raise RuntimeError("DOTE speaker diarization completed but returned no speaker turns.")
        log_callback(
            f"DOTE diarization completed: {len(turns)} turn(s), {len(speakers)} speaker(s), "
            f"sherpa-onnx {payload.get('sherpa_onnx_version', 'unknown')}."
        )
        return DiarizationResult(attribution_turns=turns, overlap_turns=turns, utterance_turns=turns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QuickFix's offline DOTE diarization worker.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--known-speakers", type=int, default=0)
    parser.add_argument("--dependency-root", type=Path)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("This module is an internal worker; pass --worker.")
    return _run_worker(args.audio, args.output, args.known_speakers, args.dependency_root)


if __name__ == "__main__":
    raise SystemExit(main())
