# QuickFixTranscription Specification

## Purpose

Build and maintain a local-only desktop transcription app for sensitive data.
All file processing must stay on the user's computer.

The app may use local tools, local Python packages, local binaries, and local
model files, but it must never send, upload, sync, report, fetch, check, or
transmit any information outside the computer.

If a feature cannot run fully offline without any network connection or
external data transfer, it must not be implemented in QuickFixTranscription.

## Processing Privacy Rule

QuickFixTranscription must never send, upload, sync, report, fetch, check, or
transmit user media, media-derived data, transcript text, diarization data,
language-detection data, filenames, or processing results outside the computer.

This includes:

- no cloud transcription
- no OpenAI API
- no hosted Whisper services
- no telemetry
- no analytics
- no crash reporting uploads
- no automatic update checks
- no implicit model downloads while processing recordings
- no remote content in the UI
- no online language detection
- no external diarization services
- no background network requests while processing recordings

All transcription processing must work using local tools and local model files.
The app must not use cloud fallback behavior if local processing components are
missing.

The app may use network access only during an explicit setup or update phase to
download general dependencies such as Python, FFmpeg, local Whisper binaries,
Python wheels, or model files. That setup/update phase must be clearly separate
from recording processing and must never transmit user media or media-derived
data.

## Product Goals

- Bulk-transcribe files and folders through a drag-and-drop UI.
- Use a UI pattern inspired by the existing QuickFixEditing app.
- Create a sibling output folder named `QuickFixTranscription` beside each
  source file or source folder.
- Export transcripts as `.rtf`.
- Preserve original source files untouched.
- Support language selection with an `Auto-detect` option and manual override.
- Provide a checkbox for `Upgrade to simple Jeffersonian transcription`.
- Keep a raw transcript artifact internally for traceability.
- Generate a separate Jeffersonian-formatted output when that checkbox is
  enabled.

## Local-Only Architecture

Use a modular pipeline:

- `ingest` for drag-and-drop, file validation, time-range validation, and job
  creation
- `media` or `ffmpeg` for local audio extraction, segment selection, conversion,
  metadata, and duration checks
- `engine` for local transcription
- `diarization` for local speaker separation if available
- `formatter/jeffersonian` for overlap, silence, and speaker formatting
- `mfa_alignment` for optional heavy local forced alignment and TextGrid parsing
- `font_assets` for local IPA-capable font discovery
- `exporter/rtf` for `.rtf` output
- `ui` for the desktop interface

Keep the transcription engine behind an interface so the backend can be swapped
later without changing the UI.

## Engine Policy

Prefer a local Whisper implementation such as `whisper.cpp`.

Acceptable behavior:

- local Whisper model inference
- local model loading from bundled files or user-selected paths
- local CPU or GPU execution

Prohibited behavior:

- remote STT APIs
- hosted Whisper services
- OpenAI API calls
- cloud fallback behavior

Whisper is still AI, but this app must use only local/offline inference. Product
copy should describe this as local AI, offline transcription, and no external
data sharing.

## Dependency Handling

QuickFixTranscription may download and install general dependencies during an
explicit setup or update phase. This is allowed only for app preparation, not for
recording processing.

On launch, the app may check for required components such as FFmpeg, Whisper
binaries, model files, and Python packages. If anything is missing, it may offer
an explicit setup flow.

Allowed dependency behavior:

- use dependencies bundled with the app installer
- use dependencies already present on the computer
- let the user select a local folder containing offline installers, binaries,
  wheels, or model files
- run an explicit offline repair/setup flow using only local files
- run an explicit online setup/update flow for general app dependencies
- download pinned dependencies from trusted sources during setup/update
- download a default local Whisper model during setup/update, then verify it by
  checksum before use
- install optional heavy MFA runtime assets during setup/update so the local MFA
  fields can be prefilled when the user enables HEAVY MFA alignment
- provide a clear list of missing local components

Prohibited dependency behavior:

- downloading dependencies while a recording is being processed
- uploading media, media-derived data, transcripts, or filenames during setup
- sending diagnostic data that includes media content or transcript content
- package manager calls from the processing pipeline
- silent fallback to cloud services when local components are missing

Dependency setup should use pinned versions, trusted official sources, and
checksums or signatures where practical. After setup, transcription must run
locally without requiring network access.

## FFmpeg Policy

QuickFixTranscription may use a local FFmpeg binary for media decoding, audio
extraction, format conversion, duration checks, and temporary WAV creation.

FFmpeg must run locally only. It must never upload, stream, fetch, or transmit
media. The app must not use remote URLs as FFmpeg inputs.

Temporary converted audio files must be written locally and deleted after
processing unless the user explicitly chooses to keep them.

## Segment Selection

QuickFixTranscription must support optional partial-file transcription.

Users may define a start time and finish time before running transcription. The
app should use the local FFmpeg binary to extract only that time range into a
temporary local audio file, then pass that temporary file to the local
transcription engine.

Preferred behavior:

- if no start or finish time is provided, transcribe the whole file
- if only start time is provided, transcribe from start time to the end of file
- if only finish time is provided, transcribe from the beginning to finish time
- validate that finish time is after start time
- display useful validation errors for invalid time ranges
- preserve original media timestamps internally when a segment is transcribed
- delete temporary segment files after processing unless explicitly retained by
  the user

FFmpeg must be invoked only on local file paths, never URLs or network streams.

## Language Handling

Use a real language-code library for dropdown values and display names.

Recommended local packages:

- `langcodes` for language tags and normalized names
- `pycountry` as an ISO language data fallback
- `Babel` only if locale-aware display names are useful in the UI

The dropdown should:

- put `Auto-detect` first
- include only languages supported by the chosen local Whisper model
- pass the selected language as a local engine hint
- never call online language services

## Jeffersonian Mode

When `Upgrade to simple Jeffersonian transcription` is enabled, run a local
post-processing formatter after raw transcription.

Output rules:

- write one line per speaker
- label speakers `SP1`, `SP2`, `SP3`, and so on
- preserve recognized word order, repetitions, repairs, false starts, fillers,
  and cut-offs rather than correcting or smoothing speech
- strip ordinary ASR punctuation from Jeffersonian text
- mark overlapping speech with `[]`
- mark silences in tenths of a second, such as `(0.2)`
- mark low-confidence words as a best guess in parentheses, such as `(example)`
- mark explicitly unclear words/sounds with no usable guess as `(     )`
- preserve supported non-word sound conventions, including `((cough))`,
  `((clears throat))`, `.snih.`, `((sigh))`, `.mt.`, `.dt.`, `.hhh`,
  `hhh`, `hm`, `mm`, and `mhm`
- render schwa-like hesitation as `eh`
- preserve a separate raw transcript artifact for traceability

Local acoustic analysis may add first-pass Jeffersonian cues when word timing is
available:

- louder words as `WORD`
- quieter words as `°word°`
- rising or falling pitch shifts as `↑word` or `↓word`
- faster or slower stretches as `>word word<` or `<word word>`
- likely prolongation as `wo::rd`
- conservative possible cut-off marking as `word-`

These acoustic cues are heuristics and must be treated as review aids, not
authoritative conversation-analysis annotation.

MFA can refine the timing of supported non-word tokens when they are present in
the local transcript or TextGrid, but MFA must not be presented as a fully
reliable detector of all non-word sounds. Human review is required for
misclassified clicks, breaths, coughs, throat clearing, sniffing, sighs, and
uncertain words or sounds.

The formatter should be inspired by GailBot's architecture, where speech
recognition is followed by post-processing modules for conversational features.
Do not copy any networked STT behavior from GailBot.

## Speaker And Overlap Handling

Speaker assignment and overlap detection are separate from speech recognition.
Whisper alone does not reliably provide speaker labels or true overlap
detection.

Preferred behavior:

- support channel-based speaker separation when the source has separate channels
- keep diarization as a local-only module that can be inserted later
- use word-level or segment-level timestamps where available
- mark Jeffersonian speaker labels or overlaps as approximate if no reliable
  local diarization or channel timing exists
- never use external speaker-labeling, diarization, or alignment services

## Optional Heavy MFA Alignment

Montreal Forced Aligner can be enabled as an optional heavy local alignment
step. It must be clearly labelled as slower than ordinary transcription.

Rules:

- require a local MFA executable
- require a local MFA acoustic model path
- require a local MFA pronunciation dictionary path
- attempt to install the MFA executable, the default UK English MFA pair
  (`english_mfa` acoustic model plus `english_uk_mfa` dictionary), and the
  legacy US ARPA pair (`english_us_arpa` acoustic model and dictionary) during
  explicit setup/update
- provide an MFA preset dropdown for additional local acoustic/dictionary
  pairs where MFA has both assets, including UK/US English, Mandarin, French,
  German, Spanish, Portuguese, Japanese, Korean, Russian, Ukrainian, Swedish,
  Thai, Vietnamese, and selected CV/Epitran model pairs
- allow a selected MFA preset to be downloaded into the shared local dependency
  folder without uploading recordings or transcripts
- store downloaded MFA assets in the shared `QuickFixAppDependencies` runtime
  folder when that folder is available
- do not download MFA models during recording processing
- run MFA only after local Whisper transcription has produced a transcript
- export the MFA TextGrid beside the transcript output for auditability
- use MFA word intervals to refine downstream timing when available

MFA can improve:

- word boundary precision
- silence placement
- phone-level interval review
- alignment of corrected transcripts back onto audio
- evidence for prolongations and possible cut-offs

MFA cannot, by itself, reliably detect overlapping speakers in mixed mono
audio or assign speakers. In-word overlap requires speaker-separated timing
evidence such as separate channels, reliable local diarization, or local source
separation before alignment.

## IPA Font And Phone-Tier Output

The app may download the SIL Charis font package during setup. This is a local
display/export dependency and does not process recordings.

Rules:

- keep the font files under `../QuickFixAppDependencies/.tools/fonts/Charis`
- do not commit downloaded font files to GitHub
- offer an option to use the IPA-capable font for regular RTF output
- offer an option to use the IPA-capable font for Jeffersonian RTF output
- clearly explain that a font does not convert words into IPA
- offer MFA phone-tier RTF export only when heavy MFA alignment is enabled
- label phone-tier output as dictionary/model dependent, not guaranteed IPA

For Jeffersonian transcripts, warn that the IPA font is not monospaced, so
overlap bracket alignment is strongest with the default monospace font.

## Output Rules

Write outputs only inside a sibling folder named `QuickFixTranscription`.

Do not overwrite original source files.

Avoid overwriting previous transcripts. Use safe output names such as:

- `<source_name>_transcript.rtf`
- `<source_name>_jeffersonian.rtf`
- `<source_name>_transcript_YYYYMMDD_HHMMSS.rtf` when a name already exists

RTF export must escape transcript text safely so braces, backslashes, and other
special characters cannot corrupt the file.

## Security And Privacy Rules

- no telemetry containing user media, transcripts, filenames, or media-derived
  data
- no analytics containing user media, transcripts, filenames, or media-derived
  data
- no crash reporting uploads
- no automatic update checks
- no remote content loading
- no API keys
- no dependency downloads while processing recordings
- no network processing of recordings
- do not log transcript text
- do not log audio-derived content
- do not upload or share media
- delete temporary audio files after processing unless explicitly retained by the
  user
- keep generated outputs inside `QuickFixTranscription`

Logs may include job status, local file names, local paths, durations, and error
messages, but must never include transcript content or audio payloads.

## Testing Expectations

Add or update tests for:

- local job creation
- supported file discovery
- output folder creation
- safe output naming
- `.rtf` export and RTF escaping
- segment start and finish validation
- local FFmpeg command construction for segment extraction
- Jeffersonian formatting
- overlap and silence markup
- local acoustic Jeffersonian cues
- temp file cleanup after success and failure
- missing-dependency launch behavior
- processing-mode no-network assumptions

Any dependency that attempts a network call during recording processing must be
removed or disabled.

## Implementation Approach

1. Inspect the existing repository structure.
2. Identify the app entry point and UI framework.
3. Reuse the QuickFixEditing drag-and-drop pattern where useful.
4. Add the local transcription pipeline.
5. Add local FFmpeg media extraction and segment selection.
6. Add the language dropdown.
7. Add the raw transcript representation.
8. Add the Jeffersonian formatter.
9. Add the RTF exporter.
10. Add tests.
11. Run the relevant test suite.
12. Report changed files and any remaining gaps.

## Working Style

- Make the smallest set of changes that satisfies the goal.
- Prefer readable, maintainable code.
- Keep network-prohibited behavior as an architectural constraint.
- Avoid introducing dependencies unless they are necessary for local operation.
- If a requested feature cannot be done fully offline, say so before
  implementing any fallback.
