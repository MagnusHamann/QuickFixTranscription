"""Small deterministic CPU VAD fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcription.acoustic_analysis import PcmAudio, rms


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float
    energy: float


def energy_vad(path: Path, frame_seconds: float = 0.03, threshold_ratio: float = 2.5) -> tuple[SpeechRegion, ...]:
    """Return speech-ish regions using local RMS energy only.

    This is a CPU fallback, not a replacement for a stronger local VAD such as
    Silero. It is deterministic and never leaves the machine.
    """
    audio = PcmAudio.from_wav(path)
    frame_size = max(1, int(audio.sample_rate * frame_seconds))
    frames: list[tuple[float, float, float]] = []
    for start_index in range(0, len(audio.samples), frame_size):
        end_index = min(len(audio.samples), start_index + frame_size)
        start = start_index / audio.sample_rate
        end = end_index / audio.sample_rate
        frames.append((start, end, rms(audio.samples[start_index:end_index])))
    if not frames:
        return ()
    floor = sorted(frame[2] for frame in frames)[max(0, int(len(frames) * 0.2) - 1)]
    threshold = max(floor * threshold_ratio, 0.005)
    regions: list[SpeechRegion] = []
    current_start: float | None = None
    current_end = 0.0
    energy_values: list[float] = []
    for start, end, energy in frames:
        if energy >= threshold:
            if current_start is None:
                current_start = start
                energy_values = []
            current_end = end
            energy_values.append(energy)
        elif current_start is not None:
            regions.append(SpeechRegion(current_start, current_end, sum(energy_values) / len(energy_values)))
            current_start = None
            energy_values = []
    if current_start is not None:
        regions.append(SpeechRegion(current_start, current_end, sum(energy_values) / len(energy_values)))
    return tuple(regions)
