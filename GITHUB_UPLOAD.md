# GitHub Upload Notes

This folder is the QuickFixTranscription source repository.

Upload the source files, scripts, docs, and tests. Do **not** upload generated runtime dependencies or user outputs.

## Do Not Upload

- `../QuickFixAppDependencies/`
- app-local `.venv/`
- app-local `.tools/`
- app-local `models/*.bin`
- app-local `models/*.gguf`
- app-local `models/*.part`
- generated `QuickFixTranscription/` output folders
- logs, caches, or transcripts

The included `.gitignore` excludes these files.

## First Run After Clone

Users should run the platform launcher.

### Windows

```powershell
.\Run QuickFixTranscription Windows.cmd
```

The launcher runs setup if needed. Setup can download:

- Python through `winget`
- FFmpeg through `winget`
- Python packages into `../QuickFixAppDependencies/.venvs/QuickFixTranscription`
- the default local Whisper model into `../QuickFixAppDependencies/models/ggml-base.bin`
- the official Windows x64 `whisper.cpp` binary into `../QuickFixAppDependencies/.tools/whisper`

### macOS

```sh
chmod +x ./agent-bootstrap.sh "./Run QuickFixTranscription macOS.command"
./"Run QuickFixTranscription macOS.command"
```

Setup can install:

- Homebrew if the user approves
- Python
- FFmpeg
- `whisper-cpp`
- Python packages into `../QuickFixAppDependencies/.venvs/QuickFixTranscription`
- the default local Whisper model into `../QuickFixAppDependencies/models/ggml-base.bin`

### Linux

```sh
chmod +x ./agent-bootstrap.sh "./Run QuickFixTranscription Linux.sh"
./"Run QuickFixTranscription Linux.sh"
```

Setup can install:

- Python
- FFmpeg
- Python packages into `../QuickFixAppDependencies/.venvs/QuickFixTranscription`
- the default local Whisper model into `../QuickFixAppDependencies/models/ggml-base.bin`
- `whisper-cpp` through Homebrew if Homebrew is available

On Linux systems without a package-managed `whisper-cpp`, users can choose a local executable in the app.

## Privacy Boundary

Setup may download general runtime dependencies. Recording processing must remain local-only.

The app must never upload or transmit media, extracted audio, transcripts, filenames, diarization data, language-detection data, or processing results.
