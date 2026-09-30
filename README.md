# QuickFixTranscription

QuickFixTranscription is a local desktop app for batch transcription of sensitive audio and video recordings.

The app uses one integrated DOTE base pipeline: local FFmpeg extraction, local Sherpa-ONNX speaker diarization, and DOTE's bundled local `whisper.cpp`. Recording processing is local-only: media, extracted audio, transcripts, filenames, diarization data, language-detection data, and processing results must not be sent to external services.

## Features

- Drag-and-drop audio/video files or folders.
- Batch process common media files including `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`, `.aac`, and `.flac`.
- Optional start and finish time selection using local FFmpeg segment extraction.
- Language dropdown with `Auto-detect` and common Whisper language options.
- DOTE's bundled local `whisper.cpp` executable and a local model path selector.
- Optional **Prefer GPU acceleration when available** checkbox for local `whisper.cpp` builds that support GPU inference; unticking it forces CPU mode.
- Preflight check before each batch for local dependencies, disk space, GPU hints, model/profile fit, cache location, and cloud-synced source/output folders.
- Model profile selector for fast draft, balanced, better accuracy, and highest accuracy guidance.
- Local cache and local temp workspace for faster reruns; these can be disabled or cleared from advanced setup.
- Resume mode skips a source when every selected output format is already current.
- Independent RTF and structured JSON output checkboxes; select either format or both from one transcription run.
- Transcription type selector: verbatim transcription or broad Jeffersonian transcription.
- Verbatim is now the direct output of the integrated DOTE base pipeline.
- Broad Jeffersonian uses exactly the same DOTE words, timings, and speaker turns, then adds QuickFix temporal/sequential notation such as overlap, latching, and timed pauses. It does not run a separate ASR or diarization pass.
- JSON exports preserve DOTE-style segment and word timestamps, stable speaker labels, confidence values, uncertainty flags, and the rendered transcript lines.
- Optional IPA-capable Charis font for regular and Jeffersonian RTF export.

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

The app warns when sources or outputs appear to be inside OneDrive, Dropbox, iCloud, Google Drive, Box, or similar synced folders. The app still processes locally, but those folders may sync files outside the computer. Enable **Strict local folder mode** in advanced setup to block such paths before transcription starts.

## Jeffersonian Output

The selected output format or formats are generated locally. DOTE first determines speaker turns with Sherpa-ONNX, then transcribes each turn with DOTE's bundled Whisper executable using no cross-turn context. Verbatim renders that structured result directly. Broad Jeffersonian consumes the same result and adds deterministic temporal/sequential notation.

Broad Jeffersonian transcription supports:

- speaker labels `SP1:`, `SP2:`, `SP3:`
- wrapped continuation lines leave the speaker column blank; a speaker label is repeated only for a new turn or a speaker change
- recognized word order, repetitions, repairs, false starts, fillers, and cut-offs are preserved rather than corrected
- ordinary ASR punctuation is stripped from Jeffersonian text
- aligned overlap brackets `[ ]` in a monospace RTF
- simultaneous Sherpa speaker turns remain separate so broad mode can render overlap without flattening the second speaker
- latching with `=` for speaker changes with effectively no gap
- timed silences of `(0.2)` or longer
- inline silences inside a speaker turn when word timing exists
- separate indented silence lines between speaker turns

Verbatim transcription renders detected speech for which no usable words were
recovered as `(unclear)`. Explicit ASR markers such as `[inaudible]` and
`[unintelligible]` are normalized to the same readable marker.

## IPA Font

Setup downloads the current Charis font package from SIL into `../QuickFixAppDependencies/.tools/fonts/Charis`. Charis is a Unicode font family suited to IPA display. The app can use this font for regular transcripts and Jeffersonian transcripts.

Important distinction:

- an IPA font displays IPA symbols correctly
- it does not convert ordinary spelling into IPA

The app no longer exports a separate phone-tier file during normal transcription. Whether MFA phone symbols are true IPA depends on the local MFA acoustic model and pronunciation dictionary. Many MFA resources use model-specific phone labels rather than strict IPA.

For Jeffersonian output, keep in mind that overlap alignment is visually strongest in monospace fonts. Charis improves IPA display, but it is not a monospace font.

## Dependencies

The bootstrap scripts install/check:

- on Windows, PowerShell via `winget` if neither `pwsh.exe` nor `powershell.exe` is available
- Python
- FFmpeg/FFprobe
- Python package dependencies from `requirements.txt`
- the default local Whisper model, `../QuickFixAppDependencies/models/ggml-base.bin`
- the local Charis IPA-capable font package
- on Windows, the pinned DOTE Whisper 1.0.2 installer into `../QuickFixAppDependencies/.tools/dote-whisper/installers`, followed by a silent local installation
- `sherpa-onnx==1.12.38`
- the pinned local Sherpa segmentation and ERes2Net speaker-embedding models under `../QuickFixAppDependencies/.tools/dote-whisper/models/diarization`

The default Whisper model is downloaded during setup/launch from the `ggml-org/whisper.cpp` model distribution on Hugging Face and verified by SHA1 checksum before use. The DOTE installer and both Sherpa models are downloaded only during setup and verified by SHA256 checksum before use. Recording processing performs no downloads and runs with the app's offline guard enabled.

Local diarization setup downloads package/model files only into the local Python environment and shared dependency folder. It is a setup action, not part of recording processing, and it does not upload media, transcripts, filenames, diarization data, or processing results.

On Linux, setup installs Python and FFmpeg through the system package manager and will also try to add Homebrew and install `whisper-cpp` automatically when possible; otherwise, choose or install a local `whisper.cpp` executable manually.

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
QuickFixAppDependencies/.tools/dote-whisper/installers/
QuickFixAppDependencies/.tools/dote-whisper/models/diarization/
QuickFixAppDependencies/models/
```

QuickFixTranscription uses the Whisper executable bundled with the installed DOTE package and can reuse shared FFmpeg, Whisper models, and IPA fonts when they are already present. Advanced setup still allows the DOTE executable and Whisper model paths to be selected manually.

## Local Cache And Resume

The structured DOTE base transcript is cached locally under the computer's local app-data/cache area, not in the source media folder. This lets the user switch between verbatim and Broad Jeffersonian rendering without repeating ASR or diarization. The cache contains media-derived transcript artifacts, so keep it enabled only on trusted local machines and use **Clear local cache** when needed.

Temporary WAV and analysis files are created in the local app temp area and cleaned up after each file unless a development/debug option keeps temp audio.

Resume mode is enabled by default. If the exact selected output already exists, the app skips that file on the next batch run. Section-only transcriptions are not skipped automatically, because start/finish choices can produce different content with the same output suffix.

## GitHub Upload

This folder is intended to be uploaded to GitHub as source code only.

Do not upload:

- `../QuickFixAppDependencies/`
- app-local `.venv/`
- app-local `.tools/`
- app-local `models/*.bin`
- app-local `models/*.gguf`
- generated `QuickFixTranscription/` output folders

After someone clones/downloads the repository, they can run the app by clicking the platform launcher. On first launch, the launcher checks for the local app runtime, installs missing prerequisites when it can, downloads large runtime assets locally, and installs the pinned DOTE Whisper package on Windows before opening the GUI.

## Install And Run

### Windows

```bat
.\Run QuickFixTranscription Windows.cmd
```

The Windows launcher checks for `pwsh.exe` or `powershell.exe` first. If neither is available, it tries to install PowerShell with `winget`, then continues setup.

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
