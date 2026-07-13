"""Runtime guards for local-only recording processing."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import socket
from typing import Iterator
from urllib.parse import urlparse


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_DISABLED": "true",
    "QUICKFIX_OFFLINE_PROCESSING": "1",
}


def reject_remote_source(value: str | Path) -> None:
    if isinstance(value, Path):
        return
    text = str(value)
    if _looks_like_windows_path(text):
        return
    parsed = urlparse(text)
    if parsed.scheme and parsed.scheme.lower() not in {"", "file"}:
        raise ValueError(f"Remote media sources are not allowed during processing: {parsed.scheme}")


def _looks_like_windows_path(text: str) -> bool:
    return len(text) >= 3 and text[1:3] in {":\\", ":/"} and text[0].isalpha()


@contextmanager
def offline_processing_guard() -> Iterator[None]:
    """Block Python network calls and mark subprocess environments offline."""
    old_env = {key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}
    os.environ.update(OFFLINE_ENVIRONMENT)
    original_create_connection = socket.create_connection
    original_socket_connect = socket.socket.connect

    def blocked_create_connection(*_args, **_kwargs):
        raise RuntimeError("Network access is disabled during QuickFixTranscription processing.")

    def blocked_socket_connect(*_args, **_kwargs):
        raise RuntimeError("Network access is disabled during QuickFixTranscription processing.")

    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    socket.socket.connect = blocked_socket_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.socket.connect = original_socket_connect  # type: ignore[assignment]
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
