"""Optional local Montreal Forced Aligner integration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
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
MFA_LAUNCHER_FAILURE_PATTERNS = ("failed to create process",)


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

    command_args = [
        "align",
        str(corpus_dir),
        str(Path(options.mfa_dictionary).expanduser()),
        str(Path(options.mfa_acoustic_model).expanduser()),
        str(output_dir),
        "--clean",
        "--overwrite",
        "--single_speaker",
    ]

    log_callback("Running narrow Jeffersonian local MFA alignment. This can take noticeably longer.")
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(MFA_ROOT_DIR)
    env["PATH"] = os.pathsep.join(_mfa_path_parts(options.mfa_executable)) + os.pathsep + env.get("PATH", "")
    command = [*_mfa_command_prefix(options.mfa_executable), *command_args]
    return_code, output_lines = _run_streamed_mfa_process(command, env, log_callback, cancelled)
    fallback_prefix = _mfa_module_command_prefix(options.mfa_executable)
    used_fallback = False

    def retry_with_module(reason: str) -> bool:
        nonlocal return_code, output_lines, used_fallback
        if fallback_prefix is None or used_fallback:
            return False
        log_callback(reason)
        fallback_command = [*fallback_prefix, *command_args]
        return_code, output_lines = _run_streamed_mfa_process(fallback_command, env, log_callback, cancelled)
        used_fallback = True
        return True

    if return_code != 0 or _mfa_launcher_failed(output_lines):
        retry_with_module("Direct MFA launch failed; retrying with Python module fallback.")

    if return_code != 0:
        detail = _last_output_summary(output_lines)
        if detail:
            raise RuntimeError(f"MFA alignment exited with code {return_code}. Last MFA output: {detail}")
        raise RuntimeError(f"MFA alignment exited with code {return_code}.")

    textgrid_path = _find_textgrid(output_dir)
    if not textgrid_path and not used_fallback:
        retry_with_module("MFA did not produce a TextGrid; retrying with Python module fallback.")
        if return_code != 0:
            detail = _last_output_summary(output_lines)
            if detail:
                raise RuntimeError(f"MFA alignment exited with code {return_code}. Last MFA output: {detail}")
            raise RuntimeError(f"MFA alignment exited with code {return_code}.")
        textgrid_path = _find_textgrid(output_dir)

    if not textgrid_path:
        raise RuntimeError("MFA alignment did not create a TextGrid file.")
    parsed = parse_textgrid(textgrid_path)
    log_callback(f"MFA alignment produced {len(parsed.words)} word interval(s) and {len(parsed.phones)} phone interval(s).")
    return parsed


def _run_streamed_mfa_process(
    command: list[str],
    env: dict[str, str],
    log_callback: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> tuple[int, list[str]]:
    log_callback("Running MFA: " + subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output_lines: list[str] = []
    output_queue: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            output_queue.put(line.rstrip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def drain_output() -> None:
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                return
            if line.strip():
                output_lines.append(line)
                log_callback(line)

    while process.poll() is None:
        drain_output()
        if cancelled():
            process.terminate()
            raise RuntimeError("MFA alignment stopped by user.")
        time.sleep(0.2)

    process.wait()
    reader.join(timeout=1)
    drain_output()
    return int(process.returncode or 0), output_lines


def _fallback_word_tokens(segment: TranscriptSegment) -> tuple[WordToken, ...]:
    if segment.words:
        return segment.words
    return tuple(WordToken(match.group(0), speaker=segment.speaker) for match in WORD_SPLIT_PATTERN.finditer(segment.text))


def apply_mfa_word_alignment(result: TranscriptResult, aligned_words: tuple[AlignedInterval, ...]) -> TranscriptResult:
    """Return a transcript with MFA timing layered onto the existing words.

    MFA is a timing refinement pass. It must never remove words that Whisper
    found in the broad transcript, because narrow Jeffersonian output is built
    on top of that broad transcript.
    """
    if not aligned_words:
        return result

    aligned_index = 0
    aligned_segments: list[TranscriptSegment] = []
    for segment in result.segments:
        source_words = _fallback_word_tokens(segment)
        new_words: list[WordToken] = []
        aligned_count = 0
        for word in source_words:
            if aligned_index < len(aligned_words):
                aligned = aligned_words[aligned_index]
                aligned_index += 1
                aligned_count += 1
                new_words.append(
                    WordToken(
                        text=word.text,
                        start=aligned.start,
                        end=aligned.end,
                        speaker=word.speaker or segment.speaker,
                        confidence=word.confidence,
                    )
                )
            else:
                new_words.append(
                    WordToken(
                        text=word.text,
                        start=word.start,
                        end=word.end,
                        speaker=word.speaker or segment.speaker,
                        confidence=word.confidence,
                    )
                )

        if new_words:
            if aligned_count == len(source_words):
                start = new_words[0].start
                end = new_words[-1].end
            else:
                start = segment.start if segment.start is not None else new_words[0].start
                end = segment.end if segment.end is not None else new_words[-1].end
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
    return [mfa_executable]


def _mfa_launcher_failed(output_lines: list[str]) -> bool:
    for line in output_lines:
        lower = line.lower()
        if any(pattern in lower for pattern in MFA_LAUNCHER_FAILURE_PATTERNS):
            return True
    return False


def _mfa_module_command_prefix(mfa_executable: str) -> list[str] | None:
    path = Path(mfa_executable).expanduser()
    if path.name.lower() == "mfa.exe" and path.parent.name.lower() == "scripts":
        python = path.parent.parent / "python.exe"
        if python.exists():
            return [str(python), "-m", "montreal_forced_aligner.command_line.mfa"]
    return None


def _mfa_path_parts(mfa_executable: str) -> list[str]:
    executable = Path(mfa_executable).expanduser()
    env_dir = executable.parent.parent
    library_bin = env_dir / "Library" / "bin"
    scripts = executable.parent
    return [
        str(_windows_no_space_alias(library_bin)),
        str(scripts),
        str(env_dir / "bin"),
        str(env_dir),
    ]


def _windows_no_space_alias(target: Path) -> Path:
    """Return a no-space junction to target on Windows when MFA needs one."""
    if platform.system() != "Windows" or not target.exists() or " " not in str(target):
        return target

    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "QuickFixApps" / "mfa-paths"
    digest = hashlib.sha1(str(target.resolve()).encode("utf-8")).hexdigest()[:12]
    alias = base / f"bin-{digest}"
    if " " in str(alias):
        return target
    if alias.exists():
        return alias

    try:
        base.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return target
    if completed.returncode == 0 and alias.exists():
        return alias
    return target


def _last_output_summary(lines: list[str], max_lines: int = 6) -> str:
    useful = [line.strip() for line in lines if line.strip()]
    return " | ".join(useful[-max_lines:])
