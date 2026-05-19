# QuickFixTranscription

QuickFixTranscription is a local desktop app for batch transcription of sensitive audio and video recordings.

The app uses local FFmpeg extraction plus a local `whisper.cpp` executable and local Whisper model file. Recording processing is local-only: media, extracted audio, transcripts, filenames, diarization data, language-detection data, and processing results will not be sent to external services.

## Features

- Drag-and-drop audio/video files or folders.
- Batch process common media files including `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`, and `.flac`.
- Optional start and finish time selection using local FFmpeg segment extraction.
- Language dropdown with `Auto-detect` and common Whisper language options.
- Local `whisper.cpp` executable and local model path selection.
- `.rtf` transcript export into `QuickFixTranscription` beside each source file.
- Optional simple Jeffersonian `.rtf` output with speaker lines, overlap brackets, and timed silences.
- Local acoustic Jeffersonian cues for loud/quiet speech, pitch shifts, speaking rate, likely prolongation, and possible cut-offs when word timing is available.

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

- speaker labels `A:`, `B:`, `C:`
- numbered lines
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

These are heuristic first-pass annotations. Human review is still needed for conversation-analysis quality transcripts, especially for emphasis, breath, smiley voice, shaky voice, laughter, analyst comments, and exact intonation marks.

## Dependencies

The bootstrap scripts install/check:

- Python
- FFmpeg/FFprobe
- Python package dependencies from `requirements.txt`
- the default local Whisper model, `models/ggml-base.bin`
- on Windows, the official `whisper.cpp` x64 release zip into `.tools/whisper`
- on macOS, `whisper-cpp` through Homebrew

The default model is downloaded during setup/launch from the `ggml-org/whisper.cpp` model distribution on Hugging Face and verified by SHA1 checksum before use. The Windows `whisper.cpp` zip is downloaded from the official `ggml-org/whisper.cpp` GitHub release and verified by SHA256 checksum before use. These are local files after download; recordings are still processed locally.

On Linux, setup installs Python and FFmpeg through the system package manager. If Homebrew is available, it also installs `whisper-cpp`; otherwise, choose or install a local `whisper.cpp` executable manually.

The app checks common local locations such as:

```text
.tools/whisper/
.tools/whisper/Release/
.tools/whisper/bin/
.tools/whisper/build/bin/
.tools/whisper/build/bin/Release/
models/
.tools/models/
.tools/whisper/models/
```

You can also choose the executable and model manually in the app.

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
python3 -m venv .venv
. ./.venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main.py
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
models/
```

See [QUICKFIXTRANSCRIPTION_SPEC.md](QUICKFIXTRANSCRIPTION_SPEC.md) for the full design and security specification.
