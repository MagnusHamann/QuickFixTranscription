"""Local IPA-capable font asset helpers."""

from __future__ import annotations

from pathlib import Path

from ffmpeg.ffmpeg_runner import PROJECT_ROOT
from quickfix_sibling_apps import quickfix_app_roots, quickfix_dependency_root


IPA_FONT_FAMILY = "Charis"
IPA_FONT_DIR = quickfix_dependency_root(PROJECT_ROOT) / ".tools" / "fonts" / "Charis"
IPA_FONT_FILE_CANDIDATES = (
    "Charis-Regular.ttf",
    "Charis-Regular.otf",
    "CharisSIL-Regular.ttf",
    "CharisSILR.ttf",
)


def find_ipa_font_file() -> Path | None:
    for root in quickfix_app_roots(PROJECT_ROOT):
        font_dir = root / ".tools" / "fonts" / "Charis"
        found = _find_font_in_dir(font_dir)
        if found:
            return found
    return None


def _find_font_in_dir(font_dir: Path) -> Path | None:
    if not font_dir.exists():
        return None
    for name in IPA_FONT_FILE_CANDIDATES:
        found = next(font_dir.rglob(name), None)
        if found and found.exists():
            return found
    found = next(font_dir.rglob("*.ttf"), None)
    if found and found.exists():
        return found
    found = next(font_dir.rglob("*.otf"), None)
    return found if found and found.exists() else None
