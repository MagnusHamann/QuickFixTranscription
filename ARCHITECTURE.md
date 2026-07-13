# QuickFixTranscription Architecture

QuickFixTranscription is a local-only desktop transcription app for sensitive audio and video data.

The runtime transcription workflow must not use cloud inference, remote APIs, telemetry, analytics, crash uploads, or remote media sources. Setup scripts may download application dependencies, models, and tools when the user explicitly runs setup, but recording processing must use only local files and local executables.

## Runtime Boundary

During transcription, the batch processor uses `offline_processing_guard()` to set offline environment variables and block Python socket connections. Media inputs are checked so remote URL sources are rejected. FFmpeg, whisper.cpp, optional MFA, acoustic analysis, and RTF export all run on local files.

The default ASR backend is local `whisper.cpp`. The UI exposes a GPU preference checkbox: when enabled, the local `whisper.cpp` binary may use GPU acceleration if it was built with GPU support; when disabled, the app passes `-ng` to force CPU mode. No cloud or hosted ASR backend is used.

## Processing Flow

1. Ingest local files or folders.
2. Extract local WAV audio with FFmpeg.
3. Run local whisper.cpp for draft ASR.
4. If broad or narrow Jeffersonian transcription is selected, inspect the audio channels. When the source has distinct speaker channels, extract the first two channels locally, transcribe each channel locally, and merge them into the broad transcript so overlap can be marked before narrow processing.
5. If channel overlap is not available and optional local sherpa-onnx diarization is enabled, run sherpa-onnx locally on the temporary WAV to add speaker timing and likely overlap windows. It overlays timing onto the existing ASR draft and inserts `(     )` review placeholders for overlapped speakers whose words were not separately recognized.
6. If narrow Jeffersonian transcription is selected, run local MFA for slower word/phone alignment. When the broad transcript already contains cross-speaker overlap, preserve that overlap timing rather than flattening it into one linear MFA timeline.
7. If narrow Jeffersonian transcription is selected, run local acoustic and timing heuristics.
8. Export exactly one visible `.rtf` file per input: verbatim, broad Jeffersonian, or narrow Jeffersonian.

## Human-In-The-Loop Rule

Machine analysis gives a first-pass transcript. The exported Jeffersonian file can include local candidates such as overlaps, pauses, pitch cues, quiet/loud speech, elongations, cut-offs, and uncertain words, but it should still be manually checked for conversation-analysis quality.

## CPU And GPU Split

Current implementation:

- CPU/local subprocesses handle FFmpeg extraction, MFA orchestration, sherpa-onnx setup/orchestration, acoustic feature heuristics, Jeffersonian formatting, and RTF export.
- `whisper.cpp` handles ASR locally. GPU acceleration is optional and depends on the selected local `whisper.cpp` binary; CPU mode is always available by unticking the GPU preference checkbox.
- Broad-overlap analysis uses local channel-separated ASR when channels are measurably distinct, then optional local sherpa-onnx diarization for mono/mixed recordings when the user enables it.
- sherpa-onnx output is treated as timing/speaker evidence only. It does not overwrite recognized words or finalize Jefferson notation.

Future GPU path:

- WhisperX may be used locally for ASR and word timestamps if installed with local model files.
- A fully local WhisperX or pyannote.audio setup may be added later for more advanced GPU ASR/diarization, but telemetry must be disabled and models must be local before runtime processing.
- GPU output should remain draft text, timing, or embedding data only.
- Final Jefferson decisions should stay in deterministic CPU review and rendering code.

## Current Limits

The app now enforces one selected transcript output per source file. It is not yet a full waveform editor. Exact bracket placement, fine-grained intonation, analyst comments, syllable-level marking, and many conversation-analysis judgements still require human review.
