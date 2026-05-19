# Agent Setup Guide

This repository contains QuickFixTranscription, a local-first desktop transcription app for sensitive recordings.

## Processing Privacy Rules

- Do not upload user media.
- Do not send media, extracted audio, transcripts, filenames, diarization data, language-detection data, or processing results to remote APIs.
- Do not add cloud transcription, hosted Whisper, OpenAI API transcription, remote diarization, online language detection, or cloud fallback behavior.
- Do not overwrite original files.
- Keep generated outputs inside `QuickFixTranscription` folders beside source files.
- Do not commit `.venv/`, `.tools/`, generated media, generated transcripts, model files, logs, or caches.

Setup/update may download general dependencies, but recording processing must remain fully local.

## Fresh Setup

### Linux/macOS

```sh
chmod +x ./agent-bootstrap.sh "./Run QuickFixTranscription Linux.sh" "./Run QuickFixTranscription macOS.command"
./agent-bootstrap.sh --yes
```

Run:

```sh
./"Run QuickFixTranscription Linux.sh"
```

On macOS, use `./"Run QuickFixTranscription macOS.command"`.

### Windows

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\agent-bootstrap.ps1 -Yes
```

Run:

```powershell
.\Run QuickFixTranscription Windows.cmd
```

## Development Notes

- GUI framework: PySide6.
- FFmpeg is called through Python subprocesses.
- Main entry point: `main.py`.
- UI lives in `ui/transcription_window.py` and `ui/transcription_options_panel.py`.
- Local transcription logic lives in `transcription/`.
- Local FFmpeg discovery/probing lives in `ffmpeg/ffmpeg_runner.py`.
- Core tests live in `tests/test_transcription_core.py`.

## Expected Fresh-Clone Behaviour

After bootstrap, the app should launch with a drag-and-drop GUI. Users can drop files/folders, choose local `whisper.cpp` and model paths, select language and optional section times, then start batch transcription. Outputs must be written to `QuickFixTranscription` beside each input file.
