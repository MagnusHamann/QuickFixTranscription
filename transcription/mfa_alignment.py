"""Optional local Montreal Forced Aligner integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from transcription.models import TranscriptResult, TranscriptSegment, TranscriptionOptions, WordToken
from transcription.dependencies import MFA_ROOT_DIR
from transcription.speakers import speaker_label
from transcription.time_utils import format_timestamp


@dataclass(frozen=True)
class AlignedInterval:
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class MfaAlignmentResult:
    textgrid_path: Path | None
    words: tuple[AlignedInterval, ...]
    phones: tuple[AlignedInterval, ...]


TEXTGRID_VALUE_PATTERN = re.compile(r'^(?P<key>\w+)\s*=\s*(?P<value>.*)$')
WORD_SPLIT_PATTERN = re.compile(r"\S+")


def _unquote_textgrid_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def _parse_float(value: str) -> float | None:
    try:
        return float(_unquote_textgrid_value(value))
    except ValueError:
        return None


def parse_textgrid(path: Path) -> MfaAlignmentResult:
    """Parse word and phone intervals from a Praat TextGrid."""
    current_tier = ""
    current_interval: dict[str, str] | None = None
    words: list[AlignedInterval] = []
    phones: list[AlignedInterval] = []

    def flush_interval() -> None:
        nonlocal current_interval
        if current_interval is None:
            return
        start = _parse_float(current_interval.get("xmin", ""))
        end = _parse_float(current_interval.get("xmax", ""))
        label = _unquote_textgrid_value(current_interval.get("text", "")).strip()
        current_interval = None
        if start is None or end is None or not label:
            return
        tier = current_tier.lower()
        interval = AlignedInterval(label=label, start=start, end=end)
        if "phone" in tier:
            phones.append(interval)
        elif "word" in tier:
            words.append(interval)

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("item ["):
            flush_interval()
            current_tier = ""
            continue
        if line.startswith("intervals ["):
            flush_interval()
            current_interval = {}
            continue

        match = TEXTGRID_VALUE_PATTERN.match(line)
        if not match:
            continue
        key = match.group("key")
        value = match.group("value")
        if current_interval is not None and key in {"xmin", "xmax", "text"}:
            current_interval[key] = value
        elif key == "name":
            current_tier = _unquote_textgrid_value(value)

    flush_interval()
    return MfaAlignmentResult(textgrid_path=path, words=tuple(words), phones=tuple(phones))


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "audio"


def _write_lab_file(path: Path, result: TranscriptResult) -> None:
    text = " ".join(result.text.split())
    path.write_text(text, encoding="utf-8")


def _find_textgrid(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("*.TextGrid"))
    if candidates:
        return candidates[0]
    candidates = sorted(output_dir.rglob("*.textgrid"))
    return candidates[0] if candidates else None


def run_mfa_alignment(
    audio_path: Path,
    transcript: TranscriptResult,
    work_dir: Path,
    options: TranscriptionOptions,
    log_callback: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> MfaAlignmentResult:
    """Run local MFA alignment and parse the resulting TextGrid."""
    mfa_root = work_dir / "mfa"
    corpus_dir = mfa_root / "corpus"
    output_dir = mfa_root / "aligned"
    if mfa_root.exists():
        shutil.rmtree(mfa_root, ignore_errors=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(audio_path)
    corpus_audio = corpus_dir / f"{stem}.wav"
    lab_file = corpus_dir / f"{stem}.lab"
    shutil.copy2(audio_path, corpus_audio)
    _write_lab_file(lab_file, transcript)

    command = [
        *_mfa_command_prefix(options.mfa_executable),
        "align",
        str(corpus_dir),
        str(Path(options.mfa_dictionary).expanduser()),
        str(Path(options.mfa_acoustic_model).expanduser()),
        str(output_dir),
        "--clean",
        "--overwrite",
    ]

    log_callback("Running optional HEAVY local MFA alignment. This can take noticeably longer.")
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(MFA_ROOT_DIR)
    mfa_path_parts = [
        str(Path(options.mfa_executable).expanduser().parent.parent / "Library" / "bin"),
        str(Path(options.mfa_executable).expanduser().parent),
    ]
    env["PATH"] = os.pathsep.join(mfa_path_parts) + os.pathsep + env.get("PATH", "")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    while process.poll() is None:
        if cancelled():
            process.terminate()
            raise RuntimeError("MFA alignment stopped by user.")
        time.sleep(0.2)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"MFA alignment exited with code {process.returncode}.")

    textgrid_path = _find_textgrid(output_dir)
    if not textgrid_path:
        raise RuntimeError("MFA alignment did not create a TextGrid file.")
    parsed = parse_textgrid(textgrid_path)
    log_callback(f"MFA alignment produced {len(parsed.words)} word interval(s) and {len(parsed.phones)} phone interval(s).")
    return parsed


def _fallback_word_tokens(segment: TranscriptSegment) -> tuple[WordToken, ...]:
    if segment.words:
        return segment.words
    return tuple(WordToken(match.group(0), speaker=segment.speaker) for match in WORD_SPLIT_PATTERN.finditer(segment.text))


def apply_mfa_word_alignment(result: TranscriptResult, aligned_words: tuple[AlignedInterval, ...]) -> TranscriptResult:
    """Return a transcript whose word and segment timings come from MFA intervals."""
    if not aligned_words:
        return result

    aligned_index = 0
    aligned_segments: list[TranscriptSegment] = []
    for segment in result.segments:
        source_words = _fallback_word_tokens(segment)
        new_words: list[WordToken] = []
        for word in source_words:
            if aligned_index >= len(aligned_words):
                break
            aligned = aligned_words[aligned_index]
            aligned_index += 1
            new_words.append(
                WordToken(
                    text=word.text,
                    start=aligned.start,
                    end=aligned.end,
                    speaker=word.speaker or segment.speaker,
                    confidence=word.confidence,
                )
            )

        if new_words:
            start = new_words[0].start
            end = new_words[-1].end
            words = tuple(new_words)
        else:
            start = segment.start
            end = segment.end
            words = segment.words
        aligned_segments.append(
            TranscriptSegment(
                text=segment.text,
                start=start,
                end=end,
                speaker=segment.speaker,
                words=words,
            )
        )

    return TranscriptResult(source_path=result.source_path, language=result.language, segments=aligned_segments)


def _speaker_for_interval(interval: AlignedInterval, result: TranscriptResult, labels: dict[str, str]) -> str:
    midpoint = interval.start + ((interval.end - interval.start) / 2)
    for segment in result.segments:
        if segment.start is None or segment.end is None:
            continue
        if segment.start <= midpoint <= segment.end:
            return speaker_label(segment, labels)
    fallback = TranscriptSegment("", speaker="speaker_1")
    return speaker_label(fallback, labels)


def phone_tier_lines(result: TranscriptResult, phones: tuple[AlignedInterval, ...], max_symbols_per_line: int = 24) -> list[str]:
    """Format MFA phone intervals as an auditable phone-tier transcript."""
    if not phones:
        return ["No MFA phone-tier intervals were found."]

    labels: dict[str, str] = {}
    lines: list[str] = []
    current_speaker = ""
    current_start: float | None = None
    current_symbols: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_symbols
        if current_start is None or not current_symbols:
            return
        lines.append(f"[{format_timestamp(current_start)}] {current_speaker}: {' '.join(current_symbols)}")
        current_start = None
        current_symbols = []

    for phone in phones:
        speaker = _speaker_for_interval(phone, result, labels)
        if speaker != current_speaker or len(current_symbols) >= max_symbols_per_line:
            flush()
            current_speaker = speaker
            current_start = phone.start
        if current_start is None:
            current_start = phone.start
        current_symbols.append(phone.label)

    flush()
    return lines


def _mfa_command_prefix(mfa_executable: str) -> list[str]:
    path = Path(mfa_executable).expanduser()
    if path.name.lower() == "mfa.exe" and path.parent.name.lower() == "scripts":
        python = path.parent.parent / "python.exe"
        if python.exists():
            return [str(python), "-m", "montreal_forced_aligner.command_line.mfa"]
    return [mfa_executable]
