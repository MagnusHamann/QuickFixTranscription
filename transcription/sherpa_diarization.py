"""Optional local sherpa-onnx diarization layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transcription.acoustic_analysis import PcmAudio
from transcription.models import TranscriptResult, TranscriptSegment, TranscriptionOptions, WordToken


MIN_OVERLAP_PLACEHOLDER_SECONDS = 0.2


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str


def run_sherpa_diarization(
    audio_path: Path,
    options: TranscriptionOptions,
    log_callback: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> tuple[DiarizationTurn, ...]:
    """Run local sherpa-onnx diarization and return timed speaker turns."""
    try:
        import numpy as np  # type: ignore
        import sherpa_onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sherpa-onnx diarization requires local Python packages: sherpa-onnx and numpy.") from exc

    segmentation_model = str(Path(options.sherpa_segmentation_model).expanduser())
    embedding_model = str(Path(options.sherpa_embedding_model).expanduser())
    num_speakers = options.sherpa_num_speakers if options.sherpa_num_speakers > 0 else -1

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("sherpa-onnx diarization config is invalid. Check local model paths.")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    audio = PcmAudio.from_wav(audio_path)
    if audio.sample_rate != diarizer.sample_rate:
        raise RuntimeError(
            f"sherpa-onnx expected {diarizer.sample_rate} Hz audio, but got {audio.sample_rate} Hz."
        )

    samples = np.asarray(audio.samples, dtype=np.float32)
    last_reported = -10

    def progress_callback(processed: int, total: int) -> int:
        nonlocal last_reported
        if cancelled():
            return 1
        if total > 0:
            percent = int((processed / total) * 100)
            if percent >= last_reported + 10:
                last_reported = percent
                log_callback(f"sherpa-onnx diarization progress: {percent}%")
        return 0

    log_callback("Running local sherpa-onnx diarization.")
    raw_result = diarizer.process(samples, callback=progress_callback).sort_by_start_time()
    turns = tuple(
        DiarizationTurn(start=float(item.start), end=float(item.end), speaker=f"sherpa_{int(item.speaker) + 1}")
        for item in raw_result
        if float(item.end) > float(item.start)
    )
    log_callback(f"sherpa-onnx diarization produced {len(turns)} speaker turn(s).")
    return turns


def apply_diarization_to_transcript(
    result: TranscriptResult,
    turns: tuple[DiarizationTurn, ...],
) -> TranscriptResult:
    """Overlay diarization turns onto a transcript without changing recognized words."""
    if not turns:
        return result

    segments: list[TranscriptSegment] = []
    for segment in result.segments:
        if segment.words:
            segments.extend(_segments_from_words(segment, turns))
        else:
            speaker = _dominant_speaker(segment.start, segment.end, turns) or segment.speaker
            segments.append(
                TranscriptSegment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker,
                    words=segment.words,
                )
            )

    segments.extend(_overlap_placeholders(turns, segments))
    segments.sort(key=lambda item: (item.start is None, item.start or 0.0, item.end or 0.0, item.speaker or ""))
    return TranscriptResult(source_path=result.source_path, language=result.language, segments=segments)


def _segments_from_words(segment: TranscriptSegment, turns: tuple[DiarizationTurn, ...]) -> list[TranscriptSegment]:
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
            )
        )
        current_words = []

    for word in segment.words:
        speaker = _dominant_speaker(word.start, word.end, turns) or segment.speaker
        retagged = WordToken(
            text=word.text,
            start=word.start,
            end=word.end,
            speaker=speaker,
            confidence=word.confidence,
        )
        if current_words and speaker != current_speaker:
            flush()
        current_speaker = speaker
        current_words.append(retagged)

    flush()
    return rows


def _dominant_speaker(start: float | None, end: float | None, turns: tuple[DiarizationTurn, ...]) -> str | None:
    if start is None or end is None or end <= start:
        return None
    overlaps: dict[str, float] = {}
    for turn in turns:
        amount = min(end, turn.end) - max(start, turn.start)
        if amount > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + amount
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda item: (item[1], item[0]))[0]


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
                placeholder = TranscriptSegment("(     )", start=start, end=end, speaker=speaker)
                placeholders.append(placeholder)
                all_segments.append(placeholder)
    return placeholders


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
