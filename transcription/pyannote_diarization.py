"""Optional local pyannote.audio diarization layer."""

from __future__ import annotations

import argparse
import json
import os
import threading
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
import wave
import warnings
from typing import Callable

from transcription.dependencies import find_pyannote_pipeline
from transcription.models import TranscriptResult, TranscriptSegment, TranscriptionOptions, WordToken
from transcription.offline_guard import OFFLINE_ENVIRONMENT


MIN_OVERLAP_PLACEHOLDER_SECONDS = 0.2
AUTO_PYANNOTE_TARGET_SPEAKERS = 2
AUTO_PYANNOTE_MAX_SPEAKERS = 6
MERGE_NEARBY_SPEAKER_GAP_SECONDS = 0.15
MIN_STABLE_SPEAKER_RUN_SECONDS = 0.25
WORD_SPEAKER_CONTEXT_SECONDS = 0.12
UTTERANCE_SPLIT_GAP_SECONDS = 0.8
SEGMENT_DOMINANT_SPEAKER_SHARE = 0.68
MAX_COMPETING_SPEAKER_SECONDS_FOR_SEGMENT_ASSIGNMENT = 0.45
NEAREST_SPEAKER_MAX_GAP_SECONDS = 0.75
UTTERANCE_EMBEDDING_MIN_SECONDS = 0.35
UTTERANCE_EMBEDDING_MIN_SIMILARITY = 0.10
UTTERANCE_EMBEDDING_MIN_MARGIN = 0.08
UTTERANCE_NEIGHBOUR_MAX_GAP_SECONDS = 3.0
UTTERANCE_OVERLAP_UNCERTAIN_SECONDS = 0.2


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str
    uncertain: bool = False


@dataclass(frozen=True)
class DiarizationResult:
    """Pyannote evidence kept in the two forms needed downstream.

    ``attribution_turns`` is exclusive and is used to give each Whisper word
    one speaker. ``overlap_turns`` retains simultaneous speakers so the
    Jeffersonian renderer can mark overlap without corrupting word ownership.
    """

    attribution_turns: tuple[DiarizationTurn, ...]
    overlap_turns: tuple[DiarizationTurn, ...]
    utterance_turns: tuple[DiarizationTurn, ...] = ()

    def __iter__(self):
        return iter(self.attribution_turns)

    def __len__(self) -> int:
        return len(self.attribution_turns)

    def __getitem__(self, index: int) -> DiarizationTurn:
        return self.attribution_turns[index]


@dataclass(frozen=True)
class _UtteranceSpeakerCandidate:
    start: float
    end: float
    best_speaker: str | None
    best_score: float | None
    margin: float | None
    confident: bool
    overlapping: bool = False


def _load_pyannote_pipeline():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*torchcodec is not installed correctly.*",
        )
        from pyannote.audio import Pipeline  # type: ignore

    return Pipeline


def _select_torch_device(prefer_gpu: bool) -> tuple[object, str]:
    """Return the torch.device required by current pyannote releases."""
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"torch is not installed locally: {exc}") from exc

    device_name = "cpu"
    if prefer_gpu:
        try:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
        except Exception:
            device_name = "cpu"
    return torch.device(device_name), device_name


def _resolve_pipeline_path(options: TranscriptionOptions) -> str:
    return options.pyannote_pipeline_path.strip() or find_pyannote_pipeline() or ""


def _pyannote_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(OFFLINE_ENVIRONMENT)
    return env


def _worker_cache_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "QuickFixAppDependencies" / ".cache" / "matplotlib"


def _annotation_length(annotation: object) -> int | None:
    try:
        return len(annotation)  # type: ignore[arg-type]
    except Exception:
        return None


def _select_diarization_annotation(output: object) -> tuple[str, object]:
    """Return the best annotation object from whatever pyannote emitted."""
    candidates: list[tuple[str, object, int | None]] = []
    for name in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, name, None)
        if annotation is not None:
            candidates.append((name, annotation, _annotation_length(annotation)))

    if candidates:
        for name, annotation, count in candidates:
            if count and count > 0:
                return name, annotation
        # If pyannote produced empty annotations, still return the preferred
        # source so diagnostics can explain the lack of speech segments.
        return candidates[0][0], candidates[0][1]

    if hasattr(output, "itertracks"):
        return type(output).__name__, output

    raise RuntimeError(
        f"Unsupported pyannote diarization output type: {type(output).__name__}. "
        "Expected a pyannote Annotation or an object exposing speaker_diarization."
    )


def _annotation_summary(output: object) -> str:
    speaker = getattr(output, "speaker_diarization", None)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if speaker is None and exclusive is None:
        count = _annotation_length(output)
        return f"{type(output).__name__}(tracks={count if count is not None else 'unknown'})"
    speaker_count = _annotation_length(speaker)
    exclusive_count = _annotation_length(exclusive)
    return (
        f"{type(output).__name__}(speaker_diarization={speaker_count if speaker_count is not None else 'unknown'}, "
        f"exclusive_speaker_diarization={exclusive_count if exclusive_count is not None else 'unknown'})"
    )


def _wave_properties(audio_path: Path) -> tuple[str, ...]:
    with wave.open(str(audio_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
    duration = frame_count / sample_rate if sample_rate else 0.0
    return (
        f"  path={audio_path}",
        f"  channels={channels}",
        f"  sample_rate={sample_rate}",
        "  codec=pcm_s16le",
        f"  sample_format={sample_width * 8}-bit PCM",
        f"  duration={duration:.2f} seconds",
    )


def _log_canonical_audio_input(audio_path: Path, log_callback: Callable[[str], None]) -> None:
    log_callback("Canonical pyannote audio:")
    try:
        for entry in _wave_properties(audio_path):
            log_callback(entry)
    except Exception as exc:
        log_callback(f"Could not inspect canonical pyannote audio: {exc}")


def _turns_from_annotation(
    annotation: object,
    speaker_labeler: Callable[[str, int], str],
) -> tuple[DiarizationTurn, ...]:
    turns: list[DiarizationTurn] = []
    speaker_numbers: dict[str, int] = {}
    next_number = 1
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        speaker_name = str(speaker)
        if speaker_name not in speaker_numbers:
            speaker_numbers[speaker_name] = next_number
            next_number += 1
        turns.append(
            DiarizationTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker=speaker_labeler(speaker_name, speaker_numbers[speaker_name]),
            )
        )
    return tuple(sorted(turns, key=lambda item: (item.start, item.end, item.speaker)))


def _diarization_result_from_output(output: object) -> DiarizationResult:
    """Extract exclusive and overlap-preserving annotations from pyannote."""
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    regular = getattr(output, "speaker_diarization", None)

    if exclusive is None and regular is None:
        if not hasattr(output, "itertracks"):
            raise RuntimeError(
                f"Unsupported pyannote diarization output type: {type(output).__name__}. "
                "Expected a pyannote Annotation or a DiarizeOutput wrapper."
            )
        exclusive = output
        regular = output

    if exclusive is None or not (_annotation_length(exclusive) or 0):
        exclusive = regular
    if regular is None or not (_annotation_length(regular) or 0):
        regular = exclusive

    if exclusive is None or regular is None:
        return DiarizationResult((), ())

    raw_attribution = _turns_from_annotation(exclusive, lambda speaker_name, _number: speaker_name)
    raw_overlap = _turns_from_annotation(regular, lambda speaker_name, _number: speaker_name)
    labels = _speaker_label_map_from_turns(raw_attribution, raw_overlap)

    def relabel(turns: tuple[DiarizationTurn, ...]) -> tuple[DiarizationTurn, ...]:
        return tuple(
            DiarizationTurn(turn.start, turn.end, labels[turn.speaker], turn.uncertain)
            for turn in turns
        )

    return DiarizationResult(relabel(raw_attribution), relabel(raw_overlap))


def _speaker_label_map_from_turns(
    raw_attribution: tuple[DiarizationTurn, ...],
    raw_overlap: tuple[DiarizationTurn, ...],
) -> dict[str, str]:
    speaker_order = sorted(
        {turn.speaker for turn in (*raw_attribution, *raw_overlap)},
        key=lambda speaker: (
            min(
                turn.start
                for turn in (*raw_attribution, *raw_overlap)
                if turn.speaker == speaker
            ),
            speaker,
        ),
    )
    return {speaker: f"pyannote_{index}" for index, speaker in enumerate(speaker_order, start=1)}


def _normalize_turns(turns: tuple[DiarizationTurn, ...]) -> tuple[DiarizationTurn, ...]:
    normalized: list[DiarizationTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end, item.speaker)):
        if turn.end <= turn.start:
            continue
        if normalized:
            previous = normalized[-1]
            if (
                previous.speaker == turn.speaker
                and turn.start - previous.end <= MERGE_NEARBY_SPEAKER_GAP_SECONDS
            ):
                normalized[-1] = DiarizationTurn(
                    previous.start,
                    max(previous.end, turn.end),
                    previous.speaker,
                    previous.uncertain or turn.uncertain,
                )
                continue
        normalized.append(turn)
    return tuple(normalized)


def run_pyannote_diarization(
    audio_path: Path,
    options: TranscriptionOptions,
    log_callback: Callable[[str], None],
    cancelled: Callable[[], bool],
    utterance_intervals: tuple[tuple[float, float], ...] = (),
) -> DiarizationResult:
    """Run local pyannote diarization and return timed speaker turns."""
    pipeline_path = _resolve_pipeline_path(options)
    if not pipeline_path:
        raise RuntimeError("No complete local pyannote pipeline was found.")

    try:
        log_callback("Loading local pyannote.audio module.")
        Pipeline = _load_pyannote_pipeline()
        log_callback("Loaded local pyannote.audio module.")
    except ImportError as exc:
        raise RuntimeError(f"pyannote.audio is not installed in the local app environment: {exc}") from exc

    try:
        audio_input = _load_waveform(audio_path)
        waveform = audio_input.get("waveform")
        shape = tuple(int(dimension) for dimension in getattr(waveform, "shape", ()))
        shape_text = "x".join(str(dimension) for dimension in shape) if shape else "unknown"
        log_callback("Decoded local WAV into an in-memory waveform for pyannote.audio.")
        log_callback(
            "Exact pyannote input: waveform dict with "
            f"waveform_shape={shape_text} and sample_rate={audio_input.get('sample_rate')}."
        )
    except Exception as exc:
        raise RuntimeError(f"Local pyannote waveform decode failed: {exc}") from exc

    if cancelled():
        raise RuntimeError("Transcription stopped by user.")

    log_callback(f"Loading local pyannote pipeline from {pipeline_path}.")
    pipeline = Pipeline.from_pretrained(pipeline_path)
    log_callback("Loaded local pyannote pipeline.")
    device, device_name = _select_torch_device(options.prefer_gpu)
    try:
        _log_canonical_audio_input(audio_path, log_callback)
        pipeline.to(device)
        log_callback(f"Local pyannote.audio diarization will run on {device_name}.")
    except Exception as exc:
        if device_name == "cpu":
            raise RuntimeError(f"The local pyannote pipeline could not be placed on CPU: {exc}") from exc
        log_callback(f"Requested pyannote device '{device_name}' was unavailable ({exc}); retrying on CPU.")
        try:
            cpu_device, device_name = _select_torch_device(False)
            pipeline.to(cpu_device)
            log_callback("Local pyannote.audio diarization will run on cpu.")
        except Exception as cpu_exc:
            raise RuntimeError(
                "The local pyannote pipeline could not be placed on CPU or the requested accelerator: "
                f"{cpu_exc}"
            ) from cpu_exc

    attempts: list[dict[str, int]] = []
    if options.known_speakers > 0:
        attempts.append({"num_speakers": options.known_speakers})
        if options.known_speakers < AUTO_PYANNOTE_TARGET_SPEAKERS:
            attempts.append({"min_speakers": AUTO_PYANNOTE_TARGET_SPEAKERS, "max_speakers": AUTO_PYANNOTE_TARGET_SPEAKERS})
    else:
        attempts.append({"num_speakers": AUTO_PYANNOTE_TARGET_SPEAKERS})
        for max_speakers in range(AUTO_PYANNOTE_TARGET_SPEAKERS + 1, AUTO_PYANNOTE_MAX_SPEAKERS + 1):
            attempts.append({"min_speakers": AUTO_PYANNOTE_TARGET_SPEAKERS, "max_speakers": max_speakers})

    best_result = DiarizationResult((), ())
    best_count = 0
    for attempt in attempts:
        if cancelled():
            raise RuntimeError("Transcription stopped by user.")
        description = ", ".join(f"{key}={value}" for key, value in attempt.items())
        log_callback("Starting pyannote inference.")
        log_callback(f"Running local pyannote.audio diarization from {pipeline_path} with {description}.")
        result = _run_pipeline_attempt(
            pipeline,
            audio_input,
            attempt,
            log_callback,
            utterance_intervals=utterance_intervals,
        )
        speaker_count = len({turn.speaker for turn in result.attribution_turns})
        if speaker_count > best_count:
            best_result = result
            best_count = speaker_count
        if speaker_count >= AUTO_PYANNOTE_TARGET_SPEAKERS:
            return result
        if speaker_count == 1 and options.known_speakers == 0:
            log_callback(
                "Local pyannote.audio diarization still found only one speaker; retrying with a wider speaker range."
            )

    if best_count:
        return best_result
    raise RuntimeError("Diarization failure: pyannote returned zero speaker turns after all attempts.")


def run_pyannote_diarization_isolated(
    audio_path: Path,
    options: TranscriptionOptions,
    log_callback: Callable[[str], None],
    cancelled: Callable[[], bool],
    transcript: TranscriptResult | None = None,
) -> DiarizationResult:
    """Run local pyannote diarization in a child Python process for crash isolation."""
    pipeline_path = _resolve_pipeline_path(options)
    if not pipeline_path:
        raise RuntimeError("No complete local pyannote pipeline was found.")

    command = [
        sys.executable,
        "-m",
        "transcription.pyannote_diarization",
        "--worker",
        "--audio",
        str(audio_path),
        "--pipeline-path",
        pipeline_path,
        "--known-speakers",
        str(options.known_speakers),
    ]
    if options.prefer_gpu:
        command.append("--prefer-gpu")
    else:
        command.append("--cpu")

    utterance_file_path: Path | None = None
    if transcript is not None:
        intervals = [
            {"start": segment.start, "end": segment.end}
            for segment in transcript.segments
            if segment.start is not None and segment.end is not None and segment.end > segment.start
        ]
        if intervals:
            utterance_file = tempfile.NamedTemporaryFile(
                prefix="quickfix_pyannote_utterances_", suffix=".json", delete=False
            )
            utterance_file_path = Path(utterance_file.name)
            utterance_file.close()
            utterance_file_path.write_text(json.dumps(intervals), encoding="utf-8")
            command.extend(["--utterance-path", str(utterance_file_path)])

    result_file = tempfile.NamedTemporaryFile(prefix="quickfix_pyannote_", suffix=".json", delete=False)
    result_file_path = Path(result_file.name)
    result_file.close()
    command.extend(["--result-path", str(result_file_path)])

    env = _pyannote_env()
    env.setdefault("MPLCONFIGDIR", str(_worker_cache_dir()))
    log_callback("Running local pyannote.audio diarization in an isolated local Python process.")
    log_callback(f"Pyannote worker result file: {result_file_path}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    stderr_lines: list[str] = []

    stderr_pipe = getattr(process, "stderr", None)

    def _forward_stderr() -> None:
        if stderr_pipe is None:
            return
        for line in stderr_pipe:
            clean = line.rstrip()
            if not clean:
                continue
            stderr_lines.append(clean)
            log_callback(clean)

    stderr_thread = threading.Thread(target=_forward_stderr, daemon=True)
    stderr_thread.start()

    last_heartbeat = time.monotonic()
    while process.poll() is None:
        if cancelled():
            process.terminate()
            raise RuntimeError("Transcription stopped by user.")
        if time.monotonic() - last_heartbeat >= 30:
            log_callback("Local pyannote.audio diarization is still running...")
            last_heartbeat = time.monotonic()
        time.sleep(0.1)

    stderr_thread.join(timeout=2.0)
    try:
        if process.returncode != 0:
            detail = stderr_lines[-1] if stderr_lines else "No technical detail was returned."
            raise RuntimeError(
                f"Local pyannote.audio diarization worker exited with code {process.returncode}: {detail}"
            )

        try:
            stdout = result_file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"Local pyannote.audio diarization worker did not write a result file: {exc}") from exc

        try:
            raw_turns = json.loads(stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Local pyannote.audio diarization worker returned invalid JSON: {exc}") from exc
    finally:
        result_file_path.unlink(missing_ok=True)
        if utterance_file_path is not None:
            utterance_file_path.unlink(missing_ok=True)

    if isinstance(raw_turns, list):
        raw_attribution_turns = raw_turns
        raw_overlap_turns = raw_turns
        raw_utterance_turns = []
    elif isinstance(raw_turns, dict):
        raw_attribution_turns = raw_turns.get("attribution_turns", [])
        raw_overlap_turns = raw_turns.get("overlap_turns", raw_attribution_turns)
        raw_utterance_turns = raw_turns.get("utterance_turns", [])
    else:
        raise RuntimeError("Local pyannote.audio diarization worker returned an unsupported result shape.")

    def parse_turns(payload: object) -> tuple[DiarizationTurn, ...]:
        if not isinstance(payload, list):
            return ()
        turns: list[DiarizationTurn] = []
        for raw_turn in payload:
            if not isinstance(raw_turn, dict):
                continue
            try:
                turn = DiarizationTurn(
                    start=float(raw_turn.get("start", 0.0)),
                    end=float(raw_turn.get("end", 0.0)),
                    speaker=str(raw_turn.get("speaker", "pyannote_1")),
                    uncertain=bool(raw_turn.get("uncertain", False)),
                )
            except (TypeError, ValueError):
                continue
            if turn.end > turn.start:
                turns.append(turn)
        return tuple(sorted(turns, key=lambda item: (item.start, item.end, item.speaker)))

    result = DiarizationResult(
        parse_turns(raw_attribution_turns),
        parse_turns(raw_overlap_turns),
        parse_turns(raw_utterance_turns),
    )
    if result.attribution_turns:
        log_callback(
            f"Local pyannote.audio diarization produced "
            f"{len({turn.speaker for turn in result.attribution_turns})} speaker label(s), "
            f"{len(result.attribution_turns)} exclusive turn(s), and "
            f"{len(result.overlap_turns)} overlap-preserving turn(s)."
        )
        if result.utterance_turns:
            uncertain_count = sum(turn.uncertain for turn in result.utterance_turns)
            log_callback(
                f"Local acoustic refinement assigned {len(result.utterance_turns)} Whisper utterance(s); "
                f"{uncertain_count} require speaker review."
            )
        return result
    raise RuntimeError("Diarization failure: pyannote returned zero speaker turns.")


def _run_pipeline_attempt(
    pipeline,
    audio_input: dict[str, object],
    speaker_kwargs: dict[str, int],
    log_callback: Callable[[str], None],
    utterance_intervals: tuple[tuple[float, float], ...] = (),
) -> DiarizationResult:
    raw_output = pipeline(audio_input, **speaker_kwargs)
    log_callback(f"pyannote raw output: {_annotation_summary(raw_output)}")
    result = _diarization_result_from_output(raw_output)
    if utterance_intervals:
        utterance_turns = _utterance_embedding_turns(
            pipeline,
            audio_input,
            raw_output,
            result,
            utterance_intervals,
            log_callback,
        )
        result = DiarizationResult(result.attribution_turns, result.overlap_turns, utterance_turns)
    if result.attribution_turns:
        log_callback(
            f"Local pyannote.audio diarization produced "
            f"{len({turn.speaker for turn in result.attribution_turns})} speaker label(s) from "
            f"{len(result.attribution_turns)} exclusive turn(s) and "
            f"{len(result.overlap_turns)} overlap-preserving turn(s)."
        )
    else:
        log_callback("Local pyannote.audio diarization produced no speaker turns.")
    return result


def _utterance_embedding_turns(
    pipeline: object,
    audio_input: dict[str, object],
    raw_output: object,
    diarization: DiarizationResult,
    utterance_intervals: tuple[tuple[float, float], ...],
    log_callback: Callable[[str], None],
) -> tuple[DiarizationTurn, ...]:
    """Classify Whisper utterances against local pyannote speaker centroids."""
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        annotation = getattr(raw_output, "speaker_diarization", None)
        embeddings = np.asarray(getattr(raw_output, "speaker_embeddings", None))
        raw_labels = list(annotation.labels()) if annotation is not None and hasattr(annotation, "labels") else []
        if embeddings.ndim != 2 or not raw_labels or embeddings.shape[0] != len(raw_labels):
            return ()

        raw_attribution = _turns_from_annotation(
            getattr(raw_output, "exclusive_speaker_diarization", annotation),
            lambda speaker_name, _number: speaker_name,
        )
        raw_overlap = _turns_from_annotation(annotation, lambda speaker_name, _number: speaker_name)
        label_map = _speaker_label_map_from_turns(raw_attribution, raw_overlap)

        centroid_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_centroids = embeddings / np.maximum(centroid_norms, 1e-12)
        waveform = audio_input.get("waveform")
        sample_rate = int(audio_input.get("sample_rate") or 0)
        embedding_model = getattr(pipeline, "_embedding", None)
        if waveform is None or sample_rate <= 0 or embedding_model is None:
            return ()
        sample_count = int(getattr(waveform, "shape", (0, 0))[-1])
        minimum_samples = int(getattr(embedding_model, "min_num_samples", sample_rate))
        overlap_windows = _simultaneous_speaker_windows(diarization.overlap_turns)

        candidates: list[_UtteranceSpeakerCandidate] = []
        for start, end in utterance_intervals:
            overlapping = (
                _interval_overlap_duration(start, end, overlap_windows)
                >= UTTERANCE_OVERLAP_UNCERTAIN_SECONDS - 1e-9
            )
            if end - start < UTTERANCE_EMBEDDING_MIN_SECONDS:
                candidates.append(
                    _UtteranceSpeakerCandidate(start, end, None, None, None, False, overlapping)
                )
                continue
            first_sample = max(0, min(sample_count, int(start * sample_rate)))
            last_sample = max(first_sample, min(sample_count, int(end * sample_rate)))
            chunk = waveform[:, first_sample:last_sample]
            if int(chunk.shape[-1]) < minimum_samples:
                chunk = torch.nn.functional.pad(chunk, (0, minimum_samples - int(chunk.shape[-1])))

            utterance_embedding = np.asarray(embedding_model(chunk.unsqueeze(0)))[0]
            utterance_norm = float(np.linalg.norm(utterance_embedding))
            if not np.isfinite(utterance_norm) or utterance_norm <= 1e-12:
                candidates.append(
                    _UtteranceSpeakerCandidate(start, end, None, None, None, False, overlapping)
                )
                continue
            scores = normalized_centroids @ (utterance_embedding / utterance_norm)
            ranking = np.argsort(scores)[::-1]
            best_index = int(ranking[0])
            best_score = float(scores[best_index])
            second_score = float(scores[int(ranking[1])]) if len(ranking) > 1 else -1.0
            raw_label = str(raw_labels[best_index])
            mapped_label = label_map.get(raw_label)
            margin = best_score - second_score
            usable_speaker = mapped_label if np.isfinite(best_score) and best_score >= UTTERANCE_EMBEDDING_MIN_SIMILARITY else None
            confident = bool(
                usable_speaker
                and np.isfinite(margin)
                and margin >= UTTERANCE_EMBEDDING_MIN_MARGIN
            )
            candidates.append(
                _UtteranceSpeakerCandidate(
                    start=start,
                    end=end,
                    best_speaker=usable_speaker,
                    best_score=best_score if np.isfinite(best_score) else None,
                    margin=margin if np.isfinite(margin) else None,
                    confident=confident,
                    overlapping=overlapping,
                )
            )

        assignments = _resolve_utterance_speaker_candidates(candidates, diarization.attribution_turns)
        confident_count = sum(not turn.uncertain for turn in assignments)
        uncertain_count = sum(turn.uncertain for turn in assignments)
        overlap_count = sum(candidate.overlapping for candidate in candidates)

        log_callback(
            f"Local acoustic refinement accepted {confident_count} confident Whisper utterance assignment(s) "
            f"and retained {uncertain_count} borderline assignment(s) for review; "
            f"{overlap_count} utterance(s) crossed simultaneous speech."
        )
        return assignments
    except Exception as exc:
        log_callback(f"Local utterance embedding refinement was skipped: {exc}")
        return ()


def _resolve_utterance_speaker_candidates(
    candidates: list[_UtteranceSpeakerCandidate],
    attribution_turns: tuple[DiarizationTurn, ...],
) -> tuple[DiarizationTurn, ...]:
    """Resolve borderline utterances using acoustic, timeline, and neighbour evidence."""
    resolved: list[DiarizationTurn] = []
    for index, candidate in enumerate(candidates):
        if candidate.confident and candidate.best_speaker and not candidate.overlapping:
            resolved.append(DiarizationTurn(candidate.start, candidate.end, candidate.best_speaker))
            continue

        previous_speaker = _nearby_confident_candidate_speaker(candidates, index, -1)
        next_speaker = _nearby_confident_candidate_speaker(candidates, index, 1)
        timeline_speaker = _dominant_speaker(candidate.start, candidate.end, attribution_turns)
        neighbours = {speaker for speaker in (previous_speaker, next_speaker) if speaker}

        if previous_speaker and previous_speaker == next_speaker:
            speaker = previous_speaker
        elif candidate.best_speaker and candidate.best_speaker in neighbours:
            speaker = candidate.best_speaker
        elif timeline_speaker and timeline_speaker in neighbours:
            speaker = timeline_speaker
        elif candidate.best_speaker and candidate.best_speaker == timeline_speaker:
            speaker = candidate.best_speaker
        else:
            speaker = timeline_speaker or candidate.best_speaker or previous_speaker or next_speaker

        if speaker:
            resolved.append(DiarizationTurn(candidate.start, candidate.end, speaker, uncertain=True))

    return tuple(resolved)


def _nearby_confident_candidate_speaker(
    candidates: list[_UtteranceSpeakerCandidate],
    current_index: int,
    direction: int,
) -> str | None:
    index = current_index + direction
    while 0 <= index < len(candidates):
        candidate = candidates[index]
        current = candidates[current_index]
        gap = current.start - candidate.end if direction < 0 else candidate.start - current.end
        if gap > UTTERANCE_NEIGHBOUR_MAX_GAP_SECONDS:
            return None
        if candidate.confident and candidate.best_speaker and not candidate.overlapping:
            return candidate.best_speaker
        index += direction
    return None


def _load_waveform(audio_path: Path) -> dict[str, object]:
    """Load a local WAV file into the waveform form pyannote.audio accepts."""
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"torch is not installed locally: {exc}") from exc

    with wave.open(str(audio_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)

    if sample_rate <= 0:
        raise RuntimeError("The WAV file reported an invalid sample rate.")
    if sample_width != 2:
        raise RuntimeError("Local pyannote diarization expects 16-bit PCM WAV audio.")
    if channels <= 0:
        raise RuntimeError("The WAV file reported no audio channels.")

    samples = array("h")
    samples.frombytes(frames)
    waveform = torch.tensor(samples, dtype=torch.float32)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
    else:
        waveform = waveform.unsqueeze(0)
    waveform = waveform / 32768.0
    return {"waveform": waveform, "sample_rate": sample_rate}


def _simultaneous_speaker_windows(
    turns: tuple[DiarizationTurn, ...],
) -> tuple[tuple[float, float], ...]:
    """Return merged windows where at least two different speakers are active."""
    windows: list[tuple[float, float]] = []
    for index, turn in enumerate(turns):
        for other in turns[index + 1 :]:
            if turn.speaker == other.speaker:
                continue
            start = max(turn.start, other.start)
            end = min(turn.end, other.end)
            if end > start:
                windows.append((start, end))

    if not windows:
        return ()

    windows.sort()
    merged: list[tuple[float, float]] = [windows[0]]
    for start, end in windows[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1e-9:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _interval_overlap_duration(
    start: float,
    end: float,
    windows: tuple[tuple[float, float], ...],
) -> float:
    if end <= start:
        return 0.0
    return sum(max(0.0, min(end, window_end) - max(start, window_start)) for window_start, window_end in windows)


def apply_diarization_to_transcript(
    result: TranscriptResult,
    diarization: DiarizationResult | tuple[DiarizationTurn, ...],
) -> TranscriptResult:
    """Overlay diarization turns onto a transcript without changing recognized words."""
    if isinstance(diarization, DiarizationResult):
        turns = _normalize_turns(diarization.attribution_turns)
        overlap_turns = _normalize_turns(diarization.overlap_turns)
        utterance_turns = diarization.utterance_turns
    else:
        turns = _normalize_turns(diarization)
        overlap_turns = turns
        utterance_turns = ()
    if not turns:
        return result

    overlap_windows = _simultaneous_speaker_windows(overlap_turns)
    segments: list[TranscriptSegment] = []
    for segment in result.segments:
        if segment.words:
            # Whisper segments are useful utterance candidates. Retaining them
            # prevents tiny diarization boundary jitter from fragmenting a
            # coherent reply into alternating one-word speaker rows.
            segments.extend(
                _segments_from_words(segment, turns, utterance_turns, overlap_windows)
            )
            continue
        speaker = (
            _dominant_speaker(segment.start, segment.end, turns)
            or _nearest_speaker(segment.start, segment.end, turns)
            or segment.speaker
        )
        segments.append(
            TranscriptSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                speaker=speaker,
                words=segment.words,
                speaker_uncertain=segment.speaker_uncertain,
            )
        )

    segments.extend(_overlap_placeholders(overlap_turns, segments))
    segments = _mark_overlap_uncertainty(segments, overlap_turns)
    segments.sort(key=lambda item: (item.start is None, item.start or 0.0, item.end or 0.0, item.speaker or ""))
    return TranscriptResult(source_path=result.source_path, language=result.language, segments=segments)


def _segments_from_words(
    segment: TranscriptSegment,
    turns: tuple[DiarizationTurn, ...],
    utterance_turns: tuple[DiarizationTurn, ...] = (),
    overlap_windows: tuple[tuple[float, float], ...] = (),
) -> list[TranscriptSegment]:
    utterance_assignment = _matching_utterance_turn(segment, utterance_turns)
    utterance_conflict = bool(
        utterance_assignment
        and not utterance_assignment.uncertain
        and not _whole_segment_assignment_is_supported(
            segment,
            utterance_assignment.speaker,
            turns,
        )
    )
    overlap_affected = bool(
        segment.start is not None
        and segment.end is not None
        and _interval_overlap_duration(segment.start, segment.end, overlap_windows)
        >= UTTERANCE_OVERLAP_UNCERTAIN_SECONDS - 1e-9
    )
    words = _stabilize_word_speakers(
        segment,
        turns,
        utterance_turns,
        ignore_uncertain_utterance=overlap_affected,
    )
    rows: list[TranscriptSegment] = []
    current_speaker: str | None = None
    current_words: list[WordToken] = []

    def flush() -> None:
        nonlocal current_words
        if not current_words:
            return
        text = " ".join(word.text for word in current_words)
        rows.append(
            TranscriptSegment(
                text=text,
                start=current_words[0].start,
                end=current_words[-1].end,
                speaker=current_speaker or segment.speaker,
                words=tuple(current_words),
                speaker_uncertain=bool(
                    utterance_assignment
                    and (utterance_assignment.uncertain or utterance_conflict)
                ),
            )
        )
        current_words = []

    for word in words:
        speaker = word.speaker or segment.speaker
        retagged = word if word.speaker == speaker else WordToken(
            text=word.text,
            start=word.start,
            end=word.end,
            speaker=speaker,
            confidence=word.confidence,
        )
        previous_word = current_words[-1] if current_words else None
        gap = None
        if previous_word and previous_word.end is not None and word.start is not None:
            gap = word.start - previous_word.end
        if current_words and (
            speaker != current_speaker
            or (gap is not None and gap >= UTTERANCE_SPLIT_GAP_SECONDS)
        ):
            flush()
        current_speaker = speaker
        current_words.append(retagged)

    flush()
    return rows


def _matching_utterance_turn(
    segment: TranscriptSegment,
    utterance_turns: tuple[DiarizationTurn, ...],
) -> DiarizationTurn | None:
    if segment.start is None or segment.end is None:
        return None
    matches: list[tuple[float, DiarizationTurn]] = []
    for turn in utterance_turns:
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > 0:
            matches.append((overlap, turn))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], -item[1].start, item[1].speaker))[1]


def _stabilize_word_speakers(
    segment: TranscriptSegment,
    turns: tuple[DiarizationTurn, ...],
    utterance_turns: tuple[DiarizationTurn, ...] = (),
    ignore_uncertain_utterance: bool = False,
) -> tuple[WordToken, ...]:
    if not segment.words:
        return ()

    usable_utterance_turns = (
        tuple(turn for turn in utterance_turns if not turn.uncertain)
        if ignore_uncertain_utterance
        else utterance_turns
    )
    segment_speaker = _dominant_speaker(segment.start, segment.end, usable_utterance_turns)
    if segment_speaker and not _whole_segment_assignment_is_supported(segment, segment_speaker, turns):
        segment_speaker = None
    if segment_speaker is None:
        segment_speaker = _confident_segment_speaker(segment, turns)
    if segment_speaker:
        return tuple(
            WordToken(
                text=word.text,
                start=word.start,
                end=word.end,
                speaker=segment_speaker,
                confidence=word.confidence,
            )
            for word in segment.words
        )

    provisional = tuple(
        WordToken(
            text=word.text,
            start=word.start,
            end=word.end,
            speaker=_dominant_speaker(word.start, word.end, turns)
            or _dominant_speaker(word.start, word.end, turns, context_seconds=WORD_SPEAKER_CONTEXT_SECONDS)
            or _nearest_speaker(word.start, word.end, turns)
            or segment.speaker,
            confidence=word.confidence,
        )
        for word in segment.words
    )

    if len(provisional) < 3:
        return provisional

    stabilized = list(provisional)
    for _pass in range(2):
        runs = _speaker_runs(stabilized)
        changed = False
        for run_index, run in enumerate(runs):
            if run.speaker is None:
                continue
            if run.duration >= MIN_STABLE_SPEAKER_RUN_SECONDS:
                continue
            if run.word_count >= 2 and run.duration >= MIN_STABLE_SPEAKER_RUN_SECONDS / 2:
                continue

            previous_run = runs[run_index - 1] if run_index > 0 else None
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            replacement = _choose_run_replacement(run.speaker, previous_run, next_run)
            if not replacement or replacement == run.speaker:
                continue

            for word_index in range(run.start_index, run.end_index):
                original = stabilized[word_index]
                stabilized[word_index] = WordToken(
                    text=original.text,
                    start=original.start,
                    end=original.end,
                    speaker=replacement,
                    confidence=original.confidence,
                )
                changed = True

        if not changed:
            break

    return tuple(stabilized)


@dataclass(frozen=True)
class _SpeakerRun:
    speaker: str | None
    start_index: int
    end_index: int
    start_time: float | None
    end_time: float | None

    @property
    def word_count(self) -> int:
        return self.end_index - self.start_index

    @property
    def duration(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)


def _speaker_runs(words: list[WordToken]) -> list[_SpeakerRun]:
    runs: list[_SpeakerRun] = []
    if not words:
        return runs
    start_index = 0
    current_speaker = words[0].speaker
    for index, word in enumerate(words[1:], start=1):
        if word.speaker == current_speaker:
            continue
        runs.append(
            _SpeakerRun(
                speaker=current_speaker,
                start_index=start_index,
                end_index=index,
                start_time=words[start_index].start,
                end_time=words[index - 1].end,
            )
        )
        start_index = index
        current_speaker = word.speaker
    runs.append(
        _SpeakerRun(
            speaker=current_speaker,
            start_index=start_index,
            end_index=len(words),
            start_time=words[start_index].start,
            end_time=words[-1].end,
        )
    )
    return runs


def _choose_run_replacement(
    current_speaker: str | None,
    previous_run: _SpeakerRun | None,
    next_run: _SpeakerRun | None,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    if previous_run and previous_run.speaker and previous_run.speaker != current_speaker:
        candidates.append((previous_run.duration, previous_run.speaker))
    if next_run and next_run.speaker and next_run.speaker != current_speaker:
        candidates.append((next_run.duration, next_run.speaker))

    if previous_run and next_run and previous_run.speaker == next_run.speaker and previous_run.speaker != current_speaker:
        return previous_run.speaker

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_duration, best_speaker = candidates[0]
    if best_duration <= 0:
        return None
    return best_speaker


def _dominant_speaker(
    start: float | None,
    end: float | None,
    turns: tuple[DiarizationTurn, ...],
    context_seconds: float = 0.0,
) -> str | None:
    if start is None or end is None or end < start:
        return None
    if end == start:
        start = max(0.0, start - 0.01)
        end = end + 0.01
    if context_seconds > 0:
        start = max(0.0, start - context_seconds)
        end = end + context_seconds
    overlaps: dict[str, float] = {}
    for turn in turns:
        amount = min(end, turn.end) - max(start, turn.start)
        if amount > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + amount
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda item: (item[1], item[0]))[0]


def _confident_segment_speaker(
    segment: TranscriptSegment,
    turns: tuple[DiarizationTurn, ...],
) -> str | None:
    start = segment.start
    end = segment.end
    if segment.words:
        timed_starts = [word.start for word in segment.words if word.start is not None]
        timed_ends = [word.end for word in segment.words if word.end is not None]
        if timed_starts:
            start = min(timed_starts)
        if timed_ends:
            end = max(timed_ends)
    if start is None or end is None or end <= start:
        return None

    overlaps: dict[str, float] = {}
    for turn in turns:
        amount = min(end, turn.end) - max(start, turn.start)
        if amount > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + amount
    if not overlaps:
        return _nearest_speaker(start, end, turns)

    ranked = sorted(overlaps.items(), key=lambda item: (item[1], item[0]), reverse=True)
    dominant_speaker, dominant_duration = ranked[0]
    total_duration = sum(overlaps.values())
    if total_duration <= 0:
        return None
    competing_duration = total_duration - dominant_duration
    if len(ranked) == 1 or (
        dominant_duration / total_duration >= SEGMENT_DOMINANT_SPEAKER_SHARE
        and competing_duration <= MAX_COMPETING_SPEAKER_SECONDS_FOR_SEGMENT_ASSIGNMENT
    ):
        return dominant_speaker
    return None


def _whole_segment_assignment_is_supported(
    segment: TranscriptSegment,
    speaker: str,
    turns: tuple[DiarizationTurn, ...],
) -> bool:
    """Reject whole-segment labels when a sustained second speaker is present."""
    start = segment.start
    end = segment.end
    if segment.words:
        starts = [word.start for word in segment.words if word.start is not None]
        ends = [word.end for word in segment.words if word.end is not None]
        if starts:
            start = min(starts)
        if ends:
            end = max(ends)
    if start is None or end is None or end <= start:
        return True

    overlaps: dict[str, float] = {}
    for turn in turns:
        amount = min(end, turn.end) - max(start, turn.start)
        if amount > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + amount
    if not overlaps:
        return True
    if len(overlaps) == 1:
        # A clean utterance embedding may correct a coarse exclusive timeline.
        # The dangerous case is an actual internal change between two labels.
        return True

    assigned_duration = overlaps.get(speaker, 0.0)
    competing_duration = sum(overlaps.values()) - assigned_duration
    return assigned_duration > 0 and competing_duration <= MAX_COMPETING_SPEAKER_SECONDS_FOR_SEGMENT_ASSIGNMENT


def _nearest_speaker(
    start: float | None,
    end: float | None,
    turns: tuple[DiarizationTurn, ...],
) -> str | None:
    """Bridge short diarization gaps without inventing an extra speaker."""
    if start is None or end is None or not turns:
        return None
    if end < start:
        start, end = end, start

    distances: list[tuple[float, str]] = []
    for turn in turns:
        if end < turn.start:
            distance = turn.start - end
        elif start > turn.end:
            distance = start - turn.end
        else:
            distance = 0.0
        distances.append((distance, turn.speaker))

    nearest_distance = min(distance for distance, _speaker in distances)
    if nearest_distance > NEAREST_SPEAKER_MAX_GAP_SECONDS:
        return None
    nearest_speakers = {
        speaker for distance, speaker in distances if abs(distance - nearest_distance) <= 1e-9
    }
    if len(nearest_speakers) != 1:
        return None
    return next(iter(nearest_speakers))


def _overlap_placeholders(
    turns: tuple[DiarizationTurn, ...],
    existing_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    placeholders: list[TranscriptSegment] = []
    for index, turn in enumerate(turns):
        for other in turns[index + 1 :]:
            if turn.speaker == other.speaker:
                continue
            start = max(turn.start, other.start)
            end = min(turn.end, other.end)
            if end - start < MIN_OVERLAP_PLACEHOLDER_SECONDS:
                continue
            all_segments = [*existing_segments, *placeholders]
            for speaker in (turn.speaker, other.speaker):
                if _has_segment_for_speaker(all_segments, speaker, start, end):
                    continue
                placeholder = TranscriptSegment(
                    "(     )",
                    start=start,
                    end=end,
                    speaker=speaker,
                    speaker_uncertain=True,
                )
                placeholders.append(placeholder)
                all_segments.append(placeholder)
    return placeholders


def _mark_overlap_uncertainty(
    segments: list[TranscriptSegment],
    turns: tuple[DiarizationTurn, ...],
) -> list[TranscriptSegment]:
    """Flag rows whose speaker evidence includes sustained simultaneous speech."""
    windows = _simultaneous_speaker_windows(turns)
    if not windows:
        return segments

    marked: list[TranscriptSegment] = []
    for segment in segments:
        if segment.start is None or segment.end is None:
            marked.append(segment)
            continue
        overlap_duration = _interval_overlap_duration(segment.start, segment.end, windows)
        if overlap_duration < UTTERANCE_OVERLAP_UNCERTAIN_SECONDS - 1e-9:
            marked.append(segment)
            continue
        marked.append(
            TranscriptSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                words=segment.words,
                speaker_uncertain=True,
            )
        )
    return marked


def _has_segment_for_speaker(
    segments: list[TranscriptSegment],
    speaker: str,
    start: float,
    end: float,
) -> bool:
    for segment in segments:
        if segment.speaker != speaker or segment.start is None or segment.end is None:
            continue
        if min(end, segment.end) - max(start, segment.start) > 0:
            return True
    return False


def _worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local pyannote diarization in worker mode.")
    parser.add_argument("--worker", action="store_true", help="Enable worker mode.")
    parser.add_argument("--audio", required=True, help="Path to local WAV audio.")
    parser.add_argument("--pipeline-path", required=True, help="Path to the local pyannote pipeline folder.")
    parser.add_argument("--known-speakers", type=int, default=0, help="Known speaker count hint.")
    parser.add_argument("--prefer-gpu", action="store_true", help="Prefer GPU acceleration if available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--result-path", required=True, help="Path to write the JSON output file.")
    parser.add_argument("--utterance-path", help="Optional local JSON file containing Whisper utterance intervals.")
    args = parser.parse_args(argv)

    if not args.worker:
        parser.error("This module only supports worker mode when run as a script.")

    class _Options:
        pyannote_pipeline_path = str(args.pipeline_path)
        known_speakers = int(args.known_speakers)
        prefer_gpu = bool(args.prefer_gpu and not args.cpu)

    def _stderr_log(message: str) -> None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    utterance_intervals: tuple[tuple[float, float], ...] = ()
    if args.utterance_path:
        try:
            raw_intervals = json.loads(Path(args.utterance_path).read_text(encoding="utf-8"))
            utterance_intervals = tuple(
                (float(item["start"]), float(item["end"]))
                for item in raw_intervals
                if isinstance(item, dict)
                and item.get("start") is not None
                and item.get("end") is not None
                and float(item["end"]) > float(item["start"])
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read local Whisper utterance intervals: {exc}") from exc

    result = run_pyannote_diarization(
        Path(args.audio),
        _Options(),
        _stderr_log,
        lambda: False,
        utterance_intervals=utterance_intervals,
    )
    payload = {
        "attribution_turns": [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker,
                "uncertain": turn.uncertain,
            }
            for turn in result.attribution_turns
        ],
        "overlap_turns": [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker,
                "uncertain": turn.uncertain,
            }
            for turn in result.overlap_turns
        ],
        "utterance_turns": [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker,
                "uncertain": turn.uncertain,
            }
            for turn in result.utterance_turns
        ],
    }
    Path(args.result_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    return _worker_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
