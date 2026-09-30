"""Jeffersonian non-word sound normalization."""

from __future__ import annotations

import re


UNCERTAIN_MARKER = "(     )"
LOW_CONFIDENCE_UNCERTAIN_THRESHOLD = 0.30

COUGH_MARKER = "((cough))"
THROAT_CLEAR_MARKER = "((clears throat))"
SNIFF_MARKER = ".snih."
SIGH_MARKER = "((sigh))"
BILABIAL_CLICK_MARKER = ".mt."
NON_BILABIAL_CLICK_MARKER = ".dt."
INBREATH_MARKER = ".hhh"
OUTBREATH_MARKER = "hhh"

KNOWN_NONWORD_MARKERS = (
    COUGH_MARKER,
    THROAT_CLEAR_MARKER,
    SNIFF_MARKER,
    SIGH_MARKER,
    BILABIAL_CLICK_MARKER,
    NON_BILABIAL_CLICK_MARKER,
    INBREATH_MARKER,
    OUTBREATH_MARKER,
    UNCERTAIN_MARKER,
)

UNCERTAIN_CANONICAL = {
    "?",
    "??",
    "???",
    "inaudible",
    "incomprehensible",
    "uncertain",
    "unclear",
    "unintelligible",
    "unknown",
}
COUGH_CANONICAL = {"cough", "coughs", "coughing"}
THROAT_CLEAR_CANONICAL = {
    "clear throat",
    "clears throat",
    "cleared throat",
    "clearing throat",
    "throat clear",
    "throat clearing",
}
SNIFF_CANONICAL = {"sniff", "sniffs", "sniffing", "sniffle", "sniffles"}
SIGH_CANONICAL = {"sigh", "sighs", "sighing"}
INBREATH_CANONICAL = {"inbreath", "in breath", "inhale", "inhales", "inhaling", "breath in", "breathing in"}
OUTBREATH_CANONICAL = {"outbreath", "out breath", "exhale", "exhales", "exhaling", "breath out", "breathing out"}
BILABIAL_CLICK_CANONICAL = {
    "bilabial click",
    "bilabial clicks",
    "lip smack",
    "lip smacks",
    "lipsmack",
    "mouth click",
    "mouth clicks",
    "smack",
}
NON_BILABIAL_CLICK_CANONICAL = {
    "click",
    "clicks",
    "dental click",
    "non bilabial click",
    "nonbilabial click",
    "tongue click",
    "tsk",
    "tut",
}

FILLED_PAUSE_CANONICAL = {
    "uh": "eh",
    "er": "eh",
    "\u0259": "eh",
    "um": "uhm",
    "umm": "uhm",
    "uhm": "uhm",
    "erm": "uhm",
}

BACKCHANNEL_CANONICAL = {
    "hm": "hm",
    "hmm": "hm",
    "mm": "mm",
    "mmm": "mm",
    "mhm": "mhm",
    "mmhm": "mhm",
    "uh huh": "uh huh",
}

PHRASE_REPLACEMENTS = (
    (re.compile(r"\b(?:bilabial clicks?|lip smacks?|mouth clicks?|lipsmack|smack)\b", re.IGNORECASE), BILABIAL_CLICK_MARKER),
    (
        re.compile(
            r"\b(?:non[- ]?bilabial clicks?|tongue clicks?|dental clicks?|tsk|tut|clicks?)\b",
            re.IGNORECASE,
        ),
        NON_BILABIAL_CLICK_MARKER,
    ),
    (
        re.compile(
            r"\b(?:clears? throat|cleared throat|clearing throat|throat clear(?:ing)?)\b",
            re.IGNORECASE,
        ),
        THROAT_CLEAR_MARKER,
    ),
    (re.compile(r"\b(?:coughs?|coughing)\b", re.IGNORECASE), COUGH_MARKER),
    (re.compile(r"\b(?:sniffs?|sniffing|sniffles?)\b", re.IGNORECASE), SNIFF_MARKER),
    (re.compile(r"\b(?:sighs?|sighing)\b", re.IGNORECASE), SIGH_MARKER),
    (re.compile(r"\b(?:inbreath|in breath|inhales?|inhaling|breath(?:ing)? in)\b", re.IGNORECASE), INBREATH_MARKER),
    (re.compile(r"\b(?:outbreath|out breath|exhales?|exhaling|breath(?:ing)? out)\b", re.IGNORECASE), OUTBREATH_MARKER),
    (
        re.compile(r"\b(?:inaudible|incomprehensible|uncertain|unclear|unintelligible|unknown|\?{1,})\b", re.IGNORECASE),
        UNCERTAIN_MARKER,
    ),
    (re.compile(r"\b(?:uh|er)\b", re.IGNORECASE), "eh"),
    (re.compile(r"\b(?:um|umm|uhm|erm)\b", re.IGNORECASE), "uhm"),
    (re.compile(r"\b(?:hmm?|mmm?|mhm|mmhm|uh huh)\b", re.IGNORECASE), lambda match: BACKCHANNEL_CANONICAL[_canonicalize(match.group(0))]),
)


def _canonicalize(text: str) -> str:
    canonical = text.strip().lower()
    canonical = re.sub(r"^[\[\(\{<\s]+|[\]\)\}>\s]+$", "", canonical)
    canonical = canonical.replace("-", " ")
    canonical = re.sub(r"[^a-z0-9\u0259?]+", " ", canonical)
    return " ".join(canonical.split())


def is_low_confidence(confidence: float | None) -> bool:
    return confidence is not None and confidence < LOW_CONFIDENCE_UNCERTAIN_THRESHOLD


def is_nonword_marker(text: str) -> bool:
    return text in KNOWN_NONWORD_MARKERS


def _low_confidence_guess(text: str) -> str:
    canonical = _canonicalize(text)
    if not canonical or canonical in UNCERTAIN_CANONICAL:
        return UNCERTAIN_MARKER
    if canonical in FILLED_PAUSE_CANONICAL:
        canonical = FILLED_PAUSE_CANONICAL[canonical]
    elif canonical in BACKCHANNEL_CANONICAL:
        canonical = BACKCHANNEL_CANONICAL[canonical]
    elif canonical in COUGH_CANONICAL:
        canonical = "cough"
    elif canonical in THROAT_CLEAR_CANONICAL:
        canonical = "clears throat"
    elif canonical in SNIFF_CANONICAL:
        canonical = "sniff"
    elif canonical in SIGH_CANONICAL:
        canonical = "sigh"
    elif canonical in INBREATH_CANONICAL:
        canonical = "inbreath"
    elif canonical in OUTBREATH_CANONICAL:
        canonical = "outbreath"
    elif canonical in BILABIAL_CLICK_CANONICAL:
        canonical = "bilabial click"
    elif canonical in NON_BILABIAL_CLICK_CANONICAL:
        canonical = "click"
    return f"({canonical})"


def normalize_nonword_token(text: str, confidence: float | None = None) -> str | None:
    """Return a Jeffersonian token for supported non-word sounds."""
    if is_low_confidence(confidence):
        return _low_confidence_guess(text)

    canonical = _canonicalize(text)
    if not canonical:
        return None
    if canonical in UNCERTAIN_CANONICAL:
        return UNCERTAIN_MARKER
    if canonical in COUGH_CANONICAL:
        return COUGH_MARKER
    if canonical in THROAT_CLEAR_CANONICAL:
        return THROAT_CLEAR_MARKER
    if canonical in SNIFF_CANONICAL:
        return SNIFF_MARKER
    if canonical in SIGH_CANONICAL:
        return SIGH_MARKER
    if canonical in INBREATH_CANONICAL:
        return INBREATH_MARKER
    if canonical in OUTBREATH_CANONICAL:
        return OUTBREATH_MARKER
    if canonical in BILABIAL_CLICK_CANONICAL:
        return BILABIAL_CLICK_MARKER
    if canonical in NON_BILABIAL_CLICK_CANONICAL:
        return NON_BILABIAL_CLICK_MARKER
    if canonical in FILLED_PAUSE_CANONICAL:
        return FILLED_PAUSE_CANONICAL[canonical]
    if canonical in BACKCHANNEL_CANONICAL:
        return BACKCHANNEL_CANONICAL[canonical]
    return None


def is_supported_nonword_source(text: str) -> bool:
    return normalize_nonword_token(text) is not None


def replace_nonword_phrases(text: str) -> str:
    normalized = text.replace("\u0259", "eh").replace("\u018f", "eh")
    for pattern, replacement in PHRASE_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    for marker in KNOWN_NONWORD_MARKERS:
        normalized = re.sub(r"\[\s*" + re.escape(marker) + r"\s*\]", marker, normalized)
    return normalized


def protect_nonword_markers(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    protected = text
    for index, marker in enumerate(KNOWN_NONWORD_MARKERS):
        placeholder = f"QFTNONWORD{index}TOKEN"
        if marker in protected:
            protected = protected.replace(marker, placeholder)
            replacements[placeholder] = marker
    return protected, replacements


def restore_nonword_markers(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for placeholder, marker in replacements.items():
        restored = restored.replace(placeholder, marker)
    return restored
