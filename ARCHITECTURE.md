# QuickFixTranscription Architecture

QuickFixTranscription is a local-only desktop transcription app for sensitive audio and video data.

The runtime transcription workflow must not use cloud inference, remote APIs, telemetry, analytics, crash uploads, or remote media sources. Setup scripts may download application dependencies, models, and tools when the user explicitly runs setup, but recording processing must use only local files and local executables.

## Runtime Boundary

During transcription, the batch processor uses `offline_processing_guard()` to set offline environment variables and block Python socket connections. Media inputs are checked so remote URL sources are rejected. FFmpeg, Sherpa-ONNX, whisper.cpp, optional Røst v3 Faster-Whisper, cache reads/writes, and RTF/JSON export all run on local files.

The default ASR backend is local `whisper.cpp`. Optional Røst v3 is a Danish-only local Faster-Whisper backend. The UI exposes a GPU preference checkbox: when enabled, Røst v3 and the local `whisper.cpp` binary may use GPU acceleration when their installed runtimes support it; when disabled, QuickFix hides CUDA from Røst v3 and passes `-ng` to Whisper. No cloud or hosted ASR backend is used.

Røst v3 setup is intentionally separate from startup bootstrap because the model is about 3.1 GB and needed only for Danish. It is ungated and requires no account or token. Runtime inference always loads the selected local model folder and receives offline environment flags.

Before a batch starts, the app runs a local preflight check. It reports missing dependencies, disk-space risk, GPU/model-profile hints, local cache paths, and source/output folders that appear to be cloud-synced. Strict local folder mode can block cloud-synced paths before processing.

## Processing Flow

1. Ingest local files or folders.
2. Run preflight checks. Errors stop the batch; warnings ask the user before continuing.
3. If resume mode is enabled and the exact selected output already exists, skip that source file.
4. Extract local WAV audio with FFmpeg into the local app temp area.
5. Run local Sherpa-ONNX diarization on the canonical mono 16 kHz WAV and retain its overlap-capable speaker turns with one stable SP1/SP2/SP3 mapping.
6. Slice the canonical WAV by diarized speaker turn, including a small context pad that is clipped back to the actual turn.
7. With DOTE Whisper selected, transcribe every turn locally with `whisper.cpp` and no cross-turn context.
8. With Røst v3 selected, transcribe all Danish turn files in one isolated local Faster-Whisper process so the model loads once and return native word timestamps. The generic `whisper.cpp` pass is not repeated.
9. Cache the structured base result. Cache identity includes the ASR backend, Whisper files, Røst v3 weight fingerprint when selected, language, source metadata, and speaker settings.
10. Render verbatim directly or add deterministic broad Jeffersonian timing, pause, latching, and overlap notation to the same speaker-labelled timeline.
11. Export the selected RTF, structured JSON, or both from the same transcript result.

## Human-In-The-Loop Rule

Machine analysis gives a first-pass transcript. The exported Jeffersonian file can include local candidates such as overlaps, pauses, pitch cues, quiet/loud speech, elongations, cut-offs, and uncertain words, but it should still be manually checked for conversation-analysis quality.

## CPU And GPU Split

Current implementation:

- CPU/local subprocesses handle FFmpeg extraction, Sherpa-ONNX diarization orchestration, timing reconciliation, Jeffersonian formatting, and RTF/JSON export.
- `whisper.cpp` handles general ASR locally and always supplies the timing layer. GPU acceleration depends on the selected local binary; CPU mode is available by unticking the GPU preference checkbox.
- Optional Røst v3 handles Danish text and word timing locally. It uses CUDA through CTranslate2 when available and falls back to CPU otherwise.
- Local Sherpa-ONNX diarization runs on the canonical mono WAV for both mono and stereo sources. It is treated as timing/speaker evidence only and does not finalize Jefferson notation.
- Local cache/temp storage uses the computer's local app-data area. It can contain media-derived transcript artifacts and can be cleared from the UI.

GPU output remains draft text, timing, or embedding data only. Final Jefferson decisions stay in deterministic CPU rendering and human review.

## Current Limits

Røst v3 is Danish-only and performs no diarization; Sherpa-ONNX remains responsible for speakers. Conversational overlap, very noisy audio, and far-field microphones remain difficult. The app is not yet a full waveform editor or multi-job parallel scheduler. Exact bracket placement and many conversation-analysis judgements still require human review.
