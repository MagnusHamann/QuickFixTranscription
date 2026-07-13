"""Optional local WhisperX draft ASR adapter."""

from __future__ import annotations

from pathlib import Path


class LocalWhisperXUnavailable(RuntimeError):
    pass


class LocalWhisperXEngine:
    """WhisperX wrapper for future GPU draft ASR.

    This adapter is intentionally strict: callers must pass local model paths and
    the module never attempts model downloads or authentication.
    """

    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        if not model_path.exists():
            raise LocalWhisperXUnavailable("Choose a local WhisperX model path before using the GPU backend.")
        try:
            import whisperx  # type: ignore
        except ImportError as exc:
            raise LocalWhisperXUnavailable("WhisperX is not installed in this local environment.") from exc
        self.whisperx = whisperx
        self.model_path = model_path
        self.device = device

    def transcribe(self, audio_path: Path) -> dict:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        raise NotImplementedError("WhisperX backend is scaffolded but not enabled until local model loading is configured.")
