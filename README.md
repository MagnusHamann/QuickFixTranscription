# QuickFixTranscription

QuickFixTranscription is a local desktop app for batch transcription of sensitive audio and video recordings.

The app uses local FFmpeg extraction plus a local `whisper.cpp` executable and local Whisper model file. Recording processing is local-only: media, extracted audio, transcripts, filenames, diarization data, language-detection data, and processing results must not be sent to external services.

## Features

- Drag-and-drop audio/video files or folders.
- Batch process common media files including `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`, and `.flac`.
- Optional start and finish time selection using local FFmpeg segment extraction.
- Language dropdown with `Auto-detect` and common Whisper language options.
- Local `whisper.cpp` executable and local model path selection.
- `.rtf` transcript export into `QuickFixTranscription` beside each source file.
- Optional simple Jeffersonian `.rtf` output with speaker lines, overlap brackets, and timed silences.
- Local acoustic Jeffersonian cues for loud/quiet speech, pitch shifts, speaking rate, likely prolongation, and possible cut-offs when word timing is available.
- Supported non-word sound conventions for Jeffersonian output, including `((cough))`, `((clears throat))`, `.snih.`, `((sigh))`, `.mt.`, `.dt.`, `.hhh`, `hhh`, `uhm`, `eh`, `hm`, `mm`, `mhm`, and uncertain `(     )`.
- Optional HEAVY Montreal Forced Aligner pass for slower local word/phone alignment and TextGrid export, with setup installing local UK English and US English MFA presets when possible.
- Optional IPA-capable Charis font for regular and Jeffersonian RTF export.
- Optional MFA phone-tier RTF export for IPA-style symbols when the chosen local MFA dictionary/model uses them.

## Processing Privacy Rule

QuickFixTranscription may use network access during explicit setup/update to install general dependencies, but recording processing itself must remain local.

The app must not upload or transmit:

- media files
- extracted audio
- transcript text
- filenames
- diarization data
- language-detection data
- processing results

No cloud transcription, hosted Whisper service, OpenAI API, remote diarization, or cloud fallback is allowed.

## Jeffersonian Output

The Jeffersonian output is generated locally from transcript timing and acoustic analysis of the temporary WAV file.

Automatically supported:

- speaker labels `SP1:`, `SP2:`, `SP3:`
- numbered lines
- recognized word order, repetitions, repairs, false starts, fillers, and cut-offs are preserved rather than corrected
- ordinary ASR punctuation is stripped from Jeffersonian text
- aligned overlap brackets `[ ]` in a monospace RTF
- timed silences of `(0.2)` or longer
- inline silences inside a speaker turn when word timing exists
- separate indented silence lines between speaker turns
- louder words as `WORD`
- quieter words as `°word°`
- rising/falling pitch shifts as `↑word` or `↓word`
- faster/slower stretches as `>word word<` or `<word word>`
- likely prolongation as `wo::rd`
- conservative possible cut-off marking as `word-`
- supported non-word sounds as `((cough))`, `((clears throat))`, `.snih.`, `((sigh))`, `.mt.` for bilabial clicks, `.dt.` for non-bilabial clicks, `.hhh` for inbreath, and `hhh` for outbreath
- backchannels such as `hm`, `mm`, and `mhm`
- schwa-like hesitation as `eh`
- low-confidence words as a best guess in parentheses, such as `(example)`
- explicitly unclear words/sounds with no usable guess as `(     )`

These are heuristic first-pass annotations. Human review is still needed for conversation-analysis quality transcripts, especially for emphasis, breath, smiley voice, shaky voice, laughter, analyst comments, exact intonation marks, and uncertain or misclassified non-word sounds.

## Optional Heavy MFA Alignment

The app can optionally run Montreal Forced Aligner after the local Whisper pass. This is a heavy local alignment step and can take much longer than ordinary transcription.

Setup tries to preinstall local MFA into the shared `QuickFixAppDependencies` folder and download the default UK English preset (`english_mfa` acoustic model plus `english_uk_mfa` dictionary) as well as the legacy US ARPA pair (`english_us_arpa`). The UI has an MFA preset dropdown for UK English, US English, Mandarin, French, German, Spanish, Portuguese, Japanese, Korean, Russian, Ukrainian, Swedish, Thai, Vietnamese, and the other app languages where MFA provides both an acoustic model and a dictionary. Use **Download selected MFA preset** to fetch an additional preset into the local shared dependency folder.

MFA requires local paths for:

- an `mfa` executable
- an MFA acoustic model
- an MFA pronunciation dictionary

When enabled, the app creates a local MFA corpus from the temporary WAV and raw transcript, runs `mfa align`, exports the TextGrid into the source file's `QuickFixTranscription` folder, and uses the MFA word intervals to refine transcript timings before Jeffersonian formatting.

MFA can help with:

- more precise word boundaries
- phone-level intervals in the TextGrid
- better silence placement
- better evidence for likely prolongation, cut-offs, and within-word timing review
- aligning a corrected transcript back onto audio

MFA does not by itself solve speaker diarization or true overlap detection in mixed mono audio. For in-word overlap, the app still needs reliable speaker-separated channels, local diarization, or source separation before alignment.

Phone labels are model/dictionary dependent. MFA can produce a phone-tier alignment, but that is not automatically the same as a full IPA phonetic transcription unless the chosen dictionary/model uses IPA-style phone labels.

## IPA Font And IPA-Style Output

Setup downloads the current Charis font package from SIL into `../QuickFixAppDependencies/.tools/fonts/Charis`. Charis is a Unicode font family suited to IPA display. The app can use this font for regular transcripts, Jeffersonian transcripts, and MFA phone-tier output.

Important distinction:

- an IPA font displays IPA symbols correctly
- it does not convert ordinary spelling into IPA

For actual IPA-style output, enable HEAVY MFA alignment and `Export MFA phone-tier transcript`. The app will export a separate `.rtf` from the MFA phone tier. Whether those symbols are true IPA depends on the local MFA acoustic model and pronunciation dictionary. Many MFA resources use model-specific phone labels rather than strict IPA.

For Jeffersonian output, keep in mind that overlap alignment is visually strongest in monospace fonts. Charis improves IPA display, but it is not a monospace font.

## Dependencies

The bootstrap scripts install/check:

- Python
- FFmpeg/FFprobe
- Python package dependencies from `requirements.txt`
- the default local Whisper model, `../QuickFixAppDependencies/models/ggml-base.bin`
- the local Charis IPA-capable font package
- on Windows, the official `whisper.cpp` x64 release zip into `../QuickFixAppDependencies/.tools/whisper`
- on macOS, `whisper-cpp` through Homebrew
- optional local MFA executable plus default UK English and US English MFA acoustic/dictionary presets

The default model is downloaded during setup/launch from the `ggml-org/whisper.cpp` model distribution on Hugging Face and verified by SHA1 checksum before use. The Windows `whisper.cpp` zip is downloaded from the official `ggml-org/whisper.cpp` GitHub release and verified by SHA256 checksum before use. These are local files after download; recordings are still processed locally.

On Linux, setup installs Python and FFmpeg through the system package manager. If Homebrew is available, it also installs `whisper-cpp`; otherwise, choose or install a local `whisper.cpp` executable manually.

Setup creates and reuses one sibling dependency folder next to the app folders:

```text
QuickFixApps/
  QuickFixAppDependencies/
    .tools/
    .venvs/
    models/
  QuickFixEditing/
  QuickFixTranscription/
  QuickFixPhonemeAlignment/
```

The app checks common local locations inside `QuickFixAppDependencies` first, including:

```text
QuickFixAppDependencies/.tools/whisper/
QuickFixAppDependencies/.tools/whisper/Release/
QuickFixAppDependencies/.tools/whisper/bin/
QuickFixAppDependencies/.tools/whisper/build/bin/
QuickFixAppDependencies/.tools/whisper/build/bin/Release/
QuickFixAppDependencies/models/
QuickFixAppDependencies/.tools/models/
QuickFixAppDependencies/.tools/whisper/models/
```

QuickFixTranscription can reuse shared FFmpeg, whisper.cpp, Whisper models, IPA fonts, and MFA assets when they are already present. You can also choose the executable and model manually in the app.

## GitHub Upload

This folder is intended to be uploaded to GitHub as source code only.

Do not upload:

- `../QuickFixAppDependencies/`
- app-local `.venv/`
- app-local `.tools/`
- app-local `models/*.bin`
- app-local `models/*.gguf`
- generated `QuickFixTranscription/` output folders

After someone clones/downloads the repository, they can run the app by clicking the platform launcher. The launcher runs setup if needed and downloads large runtime assets locally.

## Install And Run

### Windows

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\agent-bootstrap.ps1 -Yes
.\Run QuickFixTranscription Windows.cmd
```

### Linux/macOS

```sh
chmod +x ./agent-bootstrap.sh "./Run QuickFixTranscription Linux.sh" "./Run QuickFixTranscription macOS.command"
./agent-bootstrap.sh --yes
./"Run QuickFixTranscription Linux.sh"
```

On macOS, after `chmod +x`, you can also double-click:

```text
Run QuickFixTranscription macOS.command
```

## Manual Python Run

```sh
python3 -m venv ../QuickFixAppDependencies/.venvs/QuickFixTranscription
. ../QuickFixAppDependencies/.venvs/QuickFixTranscription/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows:

```powershell
py -3 -m venv ..\QuickFixAppDependencies\.venvs\QuickFixTranscription
..\QuickFixAppDependencies\.venvs\QuickFixTranscription\Scripts\python.exe -m pip install -r requirements.txt
..\QuickFixAppDependencies\.venvs\QuickFixTranscription\Scripts\python.exe .\main.py
```

## Output Rules

The app never overwrites original files.

All outputs are saved into a subfolder inside the input file's parent folder:

```text
QuickFixTranscription
```

If a filename already exists, the app adds a number instead of overwriting it.

## Project Structure

```text
main.py
ui/
transcription/
ffmpeg/
tests/
```

See [QUICKFIXTRANSCRIPTION_SPEC.md](QUICKFIXTRANSCRIPTION_SPEC.md) for the full design and security specification.
