"""Local FFmpeg extraction for transcription input audio."""

from __future__ import annotations

import subprocess
from pathlib import Path

from transcription.time_utils import format_seconds_for_ffmpeg


def command_to_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def reject_remote_input(input_file: Path) -> None:
    raw = str(input_file)
    if "://" in raw:
        raise ValueError("Remote URLs are not allowed as FFmpeg inputs.")


def build_audio_extract_command(
    ffmpeg_path: str,
    input_file: Path,
    output_file: Path,
    start_seconds: int | None = None,
    finish_seconds: int | None = None,
) -> list[str]:
    """Build a local FFmpeg command for Whisper-friendly WAV extraction."""
    reject_remote_input(input_file)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_file),
    ]

    if start_seconds is not None:
        command.extend(["-ss", format_seconds_for_ffmpeg(start_seconds)])

    if finish_seconds is not None:
        if start_seconds is not None:
            duration = finish_seconds - start_seconds
            command.extend(["-t", format_seconds_for_ffmpeg(duration)])
        else:
            command.extend(["-to", format_seconds_for_ffmpeg(finish_seconds)])

    command.extend(
        [
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_file),
        ]
    )
    return command


def build_channel_extract_command(
    ffmpeg_path: str,
    input_file: Path,
    output_file: Path,
    channel_index: int,
    start_seconds: int | None = None,
    finish_seconds: int | None = None,
) -> list[str]:
    """Build a local FFmpeg command extracting one source channel as mono WAV."""
    if channel_index < 0:
        raise ValueError("Channel index must be zero or greater.")
    reject_remote_input(input_file)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_file),
    ]

    if start_seconds is not None:
        command.extend(["-ss", format_seconds_for_ffmpeg(start_seconds)])

    if finish_seconds is not None:
        if start_seconds is not None:
            duration = finish_seconds - start_seconds
            command.extend(["-t", format_seconds_for_ffmpeg(duration)])
        else:
            command.extend(["-to", format_seconds_for_ffmpeg(finish_seconds)])

    command.extend(
        [
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-vn",
            "-af",
            f"pan=mono|c0=c{channel_index}",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_file),
        ]
    )
    return command
