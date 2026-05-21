"""Curated local MFA acoustic/dictionary preset pairs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MfaPreset:
    id: str
    label: str
    language_code: str
    acoustic_model: str
    dictionary_model: str
    note: str = ""


MFA_PRESETS: tuple[MfaPreset, ...] = (
    MfaPreset("english_uk_mfa", "English (UK, MFA phones)", "en", "english_mfa", "english_uk_mfa"),
    MfaPreset("english_us_arpa", "English (US, ARPA phones)", "en", "english_us_arpa", "english_us_arpa"),
    MfaPreset("english_us_mfa", "English (US, MFA phones)", "en", "english_mfa", "english_us_mfa"),
    MfaPreset("english_mfa", "English (global, MFA phones)", "en", "english_mfa", "english_mfa"),
    MfaPreset("english_nonnative_mfa", "English (nonnative, MFA phones)", "en", "english_mfa", "english_nonnative_mfa"),
    MfaPreset("french_mfa", "French (MFA phones)", "fr", "french_mfa", "french_mfa"),
    MfaPreset("german_mfa", "German (MFA phones)", "de", "german_mfa", "german_mfa"),
    MfaPreset("spanish_mfa", "Spanish (MFA phones)", "es", "spanish_mfa", "spanish_mfa"),
    MfaPreset("spanish_spain_mfa", "Spanish (Spain, MFA phones)", "es", "spanish_mfa", "spanish_spain_mfa"),
    MfaPreset(
        "spanish_latin_america_mfa",
        "Spanish (Latin America, MFA phones)",
        "es",
        "spanish_mfa",
        "spanish_latin_america_mfa",
    ),
    MfaPreset("italian_cv", "Italian (CV/Epitran phones)", "it", "italian_cv", "italian_cv"),
    MfaPreset("portuguese_mfa", "Portuguese (MFA phones)", "pt", "portuguese_mfa", "portuguese_mfa"),
    MfaPreset("portuguese_portugal_mfa", "Portuguese (Portugal, MFA phones)", "pt", "portuguese_mfa", "portuguese_portugal_mfa"),
    MfaPreset("portuguese_brazil_mfa", "Portuguese (Brazil, MFA phones)", "pt", "portuguese_mfa", "portuguese_brazil_mfa"),
    MfaPreset("dutch_cv", "Dutch (CV/Epitran phones)", "nl", "dutch_cv", "dutch_cv"),
    MfaPreset("swedish_mfa", "Swedish (MFA phones)", "sv", "swedish_mfa", "swedish_mfa"),
    MfaPreset("polish_mfa", "Polish (MFA phones)", "pl", "polish_mfa", "polish_mfa"),
    MfaPreset("czech_mfa", "Czech (MFA phones)", "cs", "czech_mfa", "czech_mfa"),
    MfaPreset("hungarian_cv", "Hungarian (CV/XPF phones)", "hu", "hungarian_cv", "hungarian_cv"),
    MfaPreset("romanian_cv", "Romanian (CV/XPF phones)", "ro", "romanian_cv", "romanian_cv"),
    MfaPreset("bulgarian_mfa", "Bulgarian (MFA phones)", "bg", "bulgarian_mfa", "bulgarian_mfa"),
    MfaPreset("greek_cv", "Greek (CV/XPF phones)", "el", "greek_cv", "greek_cv"),
    MfaPreset("turkish_mfa", "Turkish (MFA phones)", "tr", "turkish_mfa", "turkish_mfa"),
    MfaPreset("ukrainian_mfa", "Ukrainian (MFA phones)", "uk", "ukrainian_mfa", "ukrainian_mfa"),
    MfaPreset("russian_mfa", "Russian (MFA phones)", "ru", "russian_mfa", "russian_mfa"),
    MfaPreset("mandarin_china_mfa", "Mandarin Chinese (China, MFA phones)", "zh", "mandarin_mfa", "mandarin_china_mfa"),
    MfaPreset("mandarin_taiwan_mfa", "Mandarin Chinese (Taiwan, MFA phones)", "zh", "mandarin_mfa", "mandarin_taiwan_mfa"),
    MfaPreset(
        "mandarin_pinyin",
        "Mandarin Chinese (Pinyin dictionary)",
        "zh",
        "mandarin_mfa",
        "mandarin_pinyin",
        "Use when the alignment transcript is already Pinyin.",
    ),
    MfaPreset("japanese_mfa", "Japanese (MFA phones)", "ja", "japanese_mfa", "japanese_mfa"),
    MfaPreset("korean_mfa", "Korean (MFA phones)", "ko", "korean_mfa", "korean_mfa"),
    MfaPreset("korean_jamo_mfa", "Korean (Jamo MFA phones)", "ko", "korean_mfa", "korean_jamo_mfa"),
    MfaPreset("vietnamese_mfa", "Vietnamese (MFA phones)", "vi", "vietnamese_mfa", "vietnamese_mfa"),
    MfaPreset("vietnamese_hanoi_mfa", "Vietnamese (Hanoi, MFA phones)", "vi", "vietnamese_mfa", "vietnamese_hanoi_mfa"),
    MfaPreset(
        "vietnamese_ho_chi_minh_city_mfa",
        "Vietnamese (Ho Chi Minh City, MFA phones)",
        "vi",
        "vietnamese_mfa",
        "vietnamese_ho_chi_minh_city_mfa",
    ),
    MfaPreset("thai_mfa", "Thai (MFA phones)", "th", "thai_mfa", "thai_mfa"),
)

DEFAULT_MFA_PRESET_ID = "english_uk_mfa"
DEFAULT_SETUP_MFA_PRESET_IDS = ("english_uk_mfa", "english_us_arpa")


def mfa_preset_by_id(preset_id: str) -> MfaPreset | None:
    for preset in MFA_PRESETS:
        if preset.id == preset_id:
            return preset
    return None


def preset_for_language_code(language_code: str | None) -> MfaPreset | None:
    if not language_code:
        return mfa_preset_by_id(DEFAULT_MFA_PRESET_ID)
    normalized = language_code.strip().lower()
    for preset in MFA_PRESETS:
        if preset.language_code == normalized:
            return preset
    return None


def language_codes_with_mfa_presets() -> set[str]:
    return {preset.language_code for preset in MFA_PRESETS}
