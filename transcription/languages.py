"""Local language choices supported by Whisper-style engines."""

from __future__ import annotations


LANGUAGE_CHOICES: tuple[tuple[str, str], ...] = (
    ("", "Auto-detect"),
    ("en", "English"),
    ("cy", "Welsh"),
    ("ga", "Irish"),
    ("gd", "Scottish Gaelic"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("no", "Norwegian"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("sk", "Slovak"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("bg", "Bulgarian"),
    ("el", "Greek"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("ur", "Urdu"),
    ("bn", "Bengali"),
    ("pa", "Punjabi"),
    ("zh", "Mandarin Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("tl", "Tagalog"),
)


def language_name(code: str | None) -> str:
    if not code:
        return "Auto-detect"
    lookup = dict(LANGUAGE_CHOICES)
    return lookup.get(code, code)
