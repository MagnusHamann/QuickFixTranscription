# QuickFixTranscription Architecture

QuickFixTranscription is a local-only desktop transcription app for sensitive audio and video data.

The runtime transcription workflow must not use cloud inference, remote APIs, telemetry, analytics, crash uploads, or remote media sources. Setup scripts may download application dependencies, models, and tools when the user explicitly runs setup, but recording processing must use only local files and local executables.

## Runtime Boundary

During transcription, the batch processor uses `offline_processing_guard()` to set offline environment variables and block Python socket connections. Media inputs are checked so remote URL sources are rejected. FFmpeg, whisper.cpp, optional MFA, optional local pyannote diarization, acoustic analysis, cache reads/writes, and RTF export all run on local files.

The default ASR backend is local `whisper.cpp`. The UI exposes a GPU preference checkbox: when enabled, the local `whisper.cpp` binary may use GPU acceleration if it was built with GPU support; when disabled, the app passes `-ng` to force CPU mode. No cloud or hosted ASR backend is used.

Before a batch starts, the app runs a local preflight check. It reports missing dependencies, disk-space risk, GPU/model-profile hints, local cache paths, and source/output folders that appear to be cloud-synced. Strict local folder mode can block cloud-synced paths before processing.

## Processing Flow

1. Ingest local files or folders.
2. Run preflight checks. Errors stop the batch; warnings ask the user before continuing.
3. If resume mode is enabled and the exact selected output already exists, skip that source file.
4. Extract local WAV audio with FFmpeg into the local app temp area.
5. Run local whisper.cpp for draft ASR, using cached local results when the source/settings/model match.
6. Run the required local pyannote pipeline on the same canonical WAV for both verbatim and Broad Jeffersonian output. Use `exclusive_speaker_diarization` to assign each Whisper word one speaker, while retaining `speaker_diarization` separately for simultaneous-speaker overlap evidence. Both timelines share one deterministic speaker mapping.
7. Overlay speaker ownership onto the existing Whisper draft without changing recognized words. Preserve non-exclusive overlap timing and insert `(     )` review placeholders only when pyannote identifies a second overlapping speaker whose words were not separately recognized.
   - Developers can run `python -m transcription.diagnostics.diarization <audio-file>` to inspect source audio properties, canonical WAV properties, installed local versions, detected speaker turns, and explicit zero-turn failure cases without involving Whisper or the GUI.
8. Cache the broad Jeffersonian timing layer so narrow mode can build on the same base without repeating broad work.
9. If narrow Jeffersonian transcription is selected, run local MFA for slower word/phone alignment. When the broad transcript already contains cross-speaker overlap, preserve that overlap timing rather than flattening it into one linear MFA timeline.
10. If narrow Jeffersonian transcription is selected, run local acoustic and timing heuristics.
11. Export the selected RTF, structured JSON, or both from the same transcript result.

## Human-In-The-Loop Rule

Machine analysis gives a first-pass transcript. The exported Jeffersonian file can include local candidates such as overlaps, pauses, pitch cues, quiet/loud speech, elongations, cut-offs, and uncertain words, but it should still be manually checked for conversation-analysis quality.

## CPU And GPU Split

Current implementation:

- CPU/local subprocesses handle FFmpeg extraction, MFA orchestration, optional local pyannote diarization orchestration, acoustic feature heuristics, Jeffersonian formatting, and RTF/JSON export.
- `whisper.cpp` handles ASR locally. GPU acceleration is optional and depends on the selected local `whisper.cpp` binary; CPU mode is always available by unticking the GPU preference checkbox.
- Local pyannote diarization runs on the canonical mono WAV for both mono and stereo sources. The original channel layout is recorded in diagnostics but never assumed to equal speaker identity.
- pyannote output is treated as timing/speaker evidence only. It does not overwrite recognized words or finalize Jefferson notation.
- Local cache/temp storage uses the computer's local app-data area. It can contain media-derived transcript artifacts and can be cleared from the UI.

Future GPU path:

- WhisperX may be used locally for ASR and word timestamps if installed with local model files.
- A fully local WhisperX or pyannote.audio setup may be added later for more advanced GPU ASR/diarization, but telemetry must be disabled and models must be local before runtime processing.
- GPU output should remain draft text, timing, or embedding data only.
- Final Jefferson decisions should stay in deterministic CPU review and rendering code.

## Current Limits

The app can create one or both selected output formats per source file. It is not yet a full waveform editor or multi-job parallel scheduler. Exact bracket placement, fine-grained intonation, analyst comments, syllable-level marking, and many conversation-analysis judgements still require human review.
