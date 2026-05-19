"""FFmpeg discovery, probing, and process execution."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_local_windows_tool(name: str) -> str | None:
    if platform.system() != "Windows":
        return None
    candidate = PROJECT_ROOT / ".tools" / "ffmpeg" / "bin" / f"{name}.exe"
    return str(candidate) if candidate.exists() else None


def find_ffmpeg_tools() -> tuple[str | None, str | None]:
    ffmpeg = find_local_windows_tool("ffmpeg") or shutil.which("ffmpeg")
    ffprobe = find_local_windows_tool("ffprobe") or shutil.which("ffprobe")
    return ffmpeg, ffprobe


class FFmpegRunner:
    """Small wrapper around FFmpeg/FFprobe subprocess calls."""

    def __init__(self) -> None:
        self.ffmpeg_path, self.ffprobe_path = find_ffmpeg_tools()
        self.current_process: subprocess.Popen[str] | None = None

    @property
    def ready(self) -> bool:
        return bool(self.ffmpeg_path and self.ffprobe_path)

    def probe(self, path: Path) -> dict:
        if not self.ffprobe_path:
            raise RuntimeError("FFprobe was not found.")
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "FFprobe failed.")
        parsed = json.loads(completed.stdout)

        video_stream = next((stream for stream in parsed.get("streams", []) if stream.get("codec_type") == "video"), {})
        duration = parsed.get("format", {}).get("duration") or video_stream.get("duration")
        return {
            "duration": float(duration) if duration else None,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
        }

    def run(
        self,
        command: list[str],
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> int:
        self.current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert self.current_process.stdout is not None
        for line in self.current_process.stdout:
            if cancelled():
                self.terminate()
                return -1
            clean = line.rstrip()
            if clean:
                log_callback(clean)
        return_code = self.current_process.wait()
        self.current_process = None
        return return_code

    def terminate(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
