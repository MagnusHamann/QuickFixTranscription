"""Options panel for QuickFixTranscription."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from transcription.dependencies import DependencyStatus, dependency_status, find_mfa_acoustic_model, find_mfa_dictionary
from transcription.languages import LANGUAGE_CHOICES
from transcription.mfa_presets import MFA_PRESETS, language_codes_with_mfa_presets, mfa_preset_by_id, preset_for_language_code
from transcription.models import (
    ASR_BACKEND_DOTE_WHISPER,
    ASR_BACKEND_LABELS,
    ASR_BACKEND_DANISH_WHISPER,
    BROAD_JEFFERSONIAN_TRANSCRIPTION,
    NARROW_JEFFERSONIAN_TRANSCRIPTION,
    TRANSCRIPTION_MODE_LABELS,
    VERBATIM_TRANSCRIPTION,
    TranscriptionOptions,
)


class TranscriptionOptionsPanel(QWidget):
    """Collects local transcription settings."""

    options_changed = Signal()
    mfa_preset_download_requested = Signal(str)
    danish_whisper_install_requested = Signal()
    clear_cache_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        self.whisper_path = QLineEdit()
        self.whisper_path.setPlaceholderText("Path to DOTE whisper.cpp executable")
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to local Whisper model")
        self.whisper_browse = QPushButton("Browse")
        self.model_browse = QPushButton("Browse")

        self.asr_backend = QComboBox()
        for backend, label in ASR_BACKEND_LABELS.items():
            self.asr_backend.addItem(label, backend)
        self.danish_model_path = QLineEdit()
        self.danish_model_path.setPlaceholderText("Path to local Røst v3 model folder")
        self.danish_model_browse = QPushButton("Browse")
        self.install_danish_whisper = QPushButton("Install Røst v3 Danish Whisper")
        self.install_danish_whisper.setToolTip(
            "Download the ungated Danish conversational Whisper model for offline transcription."
        )
        self.danish_hint = QLabel(
            "Optional Danish ASR trained on conversational and read Danish. Setup downloads model files only; "
            "recording processing remains offline."
        )
        self.danish_hint.setWordWrap(True)
        self.danish_hint.setStyleSheet("color: #5a6472;")

        self.language = QComboBox()
        for code, label in LANGUAGE_CHOICES:
            self.language.addItem(label, code)
        self.model_profile = QComboBox()
        for label, value in (
            ("Auto / use selected model", "auto"),
            ("Fast draft (tiny/base)", "fast draft"),
            ("Balanced (small)", "balanced"),
            ("Better accuracy (medium)", "better accuracy"),
            ("Highest accuracy (large/turbo)", "highest accuracy"),
        ):
            self.model_profile.addItem(label, value)

        self.transcribe_section = QCheckBox("Transcribe section")
        self.start_time = QLineEdit()
        self.finish_time = QLineEdit()
        self.start_time.setPlaceholderText("mm:ss or hh:mm:ss")
        self.finish_time.setPlaceholderText("mm:ss or hh:mm:ss")

        self.transcription_mode = QComboBox()
        self.transcription_mode.addItem(TRANSCRIPTION_MODE_LABELS[VERBATIM_TRANSCRIPTION], VERBATIM_TRANSCRIPTION)
        self.transcription_mode.addItem(TRANSCRIPTION_MODE_LABELS[BROAD_JEFFERSONIAN_TRANSCRIPTION], BROAD_JEFFERSONIAN_TRANSCRIPTION)
        self.jeffersonian_line_width = QSpinBox()
        self.jeffersonian_line_width.setRange(20, 200)
        self.jeffersonian_line_width.setValue(50)
        self.jeffersonian_line_width.setSingleStep(5)
        self.jeffersonian_line_width.setSuffix(" chars")
        self.jeffersonian_line_width.setToolTip("Maximum Jeffersonian transcript text characters per line. Default is 50.")
        self.prefer_gpu = QCheckBox("Prefer GPU acceleration when available")
        self.prefer_gpu.setChecked(True)
        self.prefer_gpu.setToolTip("Lets the local whisper.cpp binary use GPU acceleration when it was built with GPU support.")
        self.use_cache = QCheckBox("Use local cache for faster reruns")
        self.use_cache.setChecked(True)
        self.use_cache.setToolTip("Stores local intermediate transcript artifacts in the computer's local app cache.")
        self.resume_completed = QCheckBox("Resume batches by skipping completed outputs")
        self.resume_completed.setChecked(True)
        self.resume_completed.setToolTip("If every selected output format already exists, skip that source instead of creating duplicates.")
        self.output_rtf = QCheckBox("RTF")
        self.output_rtf.setChecked(True)
        self.output_rtf.setToolTip("Create a formatted Rich Text transcript.")
        self.output_json = QCheckBox("JSON")
        self.output_json.setChecked(False)
        self.output_json.setToolTip("Create a structured DOTE-style transcript with speakers, timestamps, words, and confidence values.")
        self.block_cloud_synced_paths = QCheckBox("Strict local folder mode")
        self.block_cloud_synced_paths.setToolTip(
            "If a source or output folder looks cloud-synced or network-backed, the app asks whether to trust it or stop transcription."
        )
        self.clear_cache = QPushButton("Clear local cache")
        self.mfa_preset = QComboBox()
        for preset in MFA_PRESETS:
            self.mfa_preset.addItem(preset.label, preset.id)
        self.mfa_preset_hint = QLabel()
        self.mfa_preset_hint.setWordWrap(True)
        self.download_mfa_preset = QPushButton("Download selected MFA preset")
        self.download_mfa_preset.setToolTip(
            "Downloads only MFA acoustic/dictionary assets. It does not upload recordings or transcripts."
        )
        self.mfa_path = QLineEdit()
        self.mfa_path.setPlaceholderText("Path to local mfa executable")
        self.mfa_model_path = QLineEdit()
        self.mfa_model_path.setPlaceholderText("Path to local MFA acoustic model")
        self.mfa_dictionary_path = QLineEdit()
        self.mfa_dictionary_path.setPlaceholderText("Path to local pronunciation dictionary")
        self.mfa_browse = QPushButton("Browse")
        self.mfa_model_browse = QPushButton("Browse")
        self.mfa_dictionary_browse = QPushButton("Browse")
        self.known_speakers = QSpinBox()
        self.known_speakers.setRange(0, 20)
        self.known_speakers.setValue(0)
        self.known_speakers.setSpecialValueText("Auto")
        self.known_speakers.setToolTip("Set a known speaker count, or leave Auto for local diarization.")
        self.diarization_auto_hint = QLabel(
            "Auto lets DOTE's local Sherpa-ONNX diarizer determine the speaker count. Set a number only when it is known."
        )
        self.diarization_auto_hint.setWordWrap(True)
        self.diarization_auto_hint.setStyleSheet("color: #5a6472;")
        self.use_ipa_font_regular = QCheckBox("Use IPA font for regular transcript")
        self.use_ipa_font_jeffersonian = QCheckBox("Use IPA font for Jeffersonian transcript")
        self.review_note = QPlainTextEdit()
        self.review_note.setPlainText(
            "Choose RTF, structured JSON, or both. Both files are generated from the same transcription run.\n\n"
            "Both transcription types use one integrated DOTE base pipeline: a local FFmpeg WAV and local Sherpa-ONNX "
            "speaker diarization. Choose DOTE Whisper for general ASR or Røst v3 for Danish text and word timing. "
            "Broad Jeffersonian transcription "
            "uses that same verbatim result and then marks temporal and sequential "
            "notation: SP1/SP2/SP3 labels, configurable line wrapping, no ASR punctuation, [overlap] brackets, "
            "latching with =, and timed pauses of 0.2s+. Voice-quality detail is left for later review.\n\n"
            "Review manually: exact overlap placement, speaker changes, uncertain timing boundaries, and any disfluencies that "
            "need careful checking."
        )
        self.review_note.setReadOnly(True)
        self.review_note.setMinimumHeight(110)
        self.review_note.setMaximumHeight(150)
        self.review_note.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.review_note.setObjectName("ReviewNote")

        self.refresh_dependencies = QPushButton("Refresh dependency check")
        self.dependencies_toggle = QToolButton()
        self.dependencies_toggle.setText("Dependencies")
        self.dependencies_toggle.setCheckable(True)
        self.dependencies_toggle.setChecked(False)
        self.dependencies_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.dependencies_toggle.setArrowType(Qt.RightArrow)
        self.dependencies_content = QWidget()
        self.dependencies_content.setVisible(False)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced local setup")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_content = QWidget()
        self.advanced_content.setVisible(False)

        self._build_layout()
        self._wire_events()
        self.apply_dependency_status(dependency_status())
        self._update_visibility()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.dependencies_box = QWidget()
        dependencies_layout = QVBoxLayout(self.dependencies_box)
        dependencies_layout.setContentsMargins(0, 0, 0, 0)
        dependencies_layout.addWidget(self.dependencies_toggle)
        status_layout = QVBoxLayout(self.dependencies_content)
        status_layout.setContentsMargins(18, 0, 0, 0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.refresh_dependencies)
        dependencies_layout.addWidget(self.dependencies_content)

        self.section_group = QGroupBox("Section")
        section_layout = QFormLayout(self.section_group)
        section_layout.addRow(self.transcribe_section)
        section_layout.addRow("Start", self.start_time)
        section_layout.addRow("Finish", self.finish_time)

        self.output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(self.output_group)
        output_layout.addWidget(QLabel("Speech recognition"))
        output_layout.addWidget(self.asr_backend)
        output_layout.addWidget(QLabel("Language"))
        output_layout.addWidget(self.language)
        output_layout.addWidget(QLabel("Model profile"))
        output_layout.addWidget(self.model_profile)
        output_layout.addWidget(QLabel("Transcription type"))
        output_layout.addWidget(self.transcription_mode)
        output_formats_row = QHBoxLayout()
        output_formats_row.addWidget(QLabel("Output formats"))
        output_formats_row.addWidget(self.output_rtf)
        output_formats_row.addWidget(self.output_json)
        output_formats_row.addStretch(1)
        output_layout.addLayout(output_formats_row)
        known_speakers_row = QHBoxLayout()
        known_speakers_row.addWidget(QLabel("Known speakers"))
        known_speakers_row.addWidget(self.known_speakers)
        known_speakers_row.addStretch(1)
        output_layout.addLayout(known_speakers_row)
        output_layout.addWidget(self.diarization_auto_hint)
        output_layout.addWidget(self.prefer_gpu)
        line_width_row = QHBoxLayout()
        line_width_row.addWidget(QLabel("Jeffersonian line width"))
        line_width_row.addWidget(self.jeffersonian_line_width)
        line_width_row.addStretch(1)
        output_layout.addLayout(line_width_row)
        output_layout.addWidget(self.review_note)

        self.advanced_box = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_box)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(self.advanced_toggle)
        advanced_content_layout = QVBoxLayout(self.advanced_content)
        advanced_content_layout.setContentsMargins(18, 0, 0, 0)

        self.engine_group = QGroupBox("DOTE local model paths")
        engine_layout = QFormLayout(self.engine_group)
        whisper_row = QHBoxLayout()
        whisper_row.addWidget(self.whisper_path)
        whisper_row.addWidget(self.whisper_browse)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path)
        model_row.addWidget(self.model_browse)
        engine_layout.addRow("DOTE whisper.cpp", whisper_row)
        engine_layout.addRow("Model", model_row)

        self.danish_group = QGroupBox("Optional Danish Røst v3 Whisper")
        danish_layout = QVBoxLayout(self.danish_group)
        danish_path_row = QHBoxLayout()
        danish_path_row.addWidget(self.danish_model_path, stretch=1)
        danish_path_row.addWidget(self.danish_model_browse)
        danish_layout.addLayout(danish_path_row)
        danish_layout.addWidget(self.install_danish_whisper)
        danish_layout.addWidget(self.danish_hint)

        self.mfa_group = QGroupBox("Narrow Jeffersonian local MFA")
        mfa_layout = QVBoxLayout(self.mfa_group)
        mfa_preset_row = QHBoxLayout()
        mfa_preset_row.addWidget(self.mfa_preset, stretch=1)
        mfa_preset_row.addWidget(self.download_mfa_preset)
        mfa_layout.addLayout(mfa_preset_row)
        mfa_layout.addWidget(self.mfa_preset_hint)
        mfa_executable_row = QHBoxLayout()
        mfa_executable_row.addWidget(self.mfa_path)
        mfa_executable_row.addWidget(self.mfa_browse)
        mfa_layout.addLayout(mfa_executable_row)
        mfa_model_row = QHBoxLayout()
        mfa_model_row.addWidget(self.mfa_model_path)
        mfa_model_row.addWidget(self.mfa_model_browse)
        mfa_layout.addLayout(mfa_model_row)
        mfa_dictionary_row = QHBoxLayout()
        mfa_dictionary_row.addWidget(self.mfa_dictionary_path)
        mfa_dictionary_row.addWidget(self.mfa_dictionary_browse)
        mfa_layout.addLayout(mfa_dictionary_row)

        self.display_group = QGroupBox("Export font")
        display_layout = QVBoxLayout(self.display_group)
        display_layout.addWidget(self.use_ipa_font_regular)
        display_layout.addWidget(self.use_ipa_font_jeffersonian)

        self.performance_group = QGroupBox("Performance and privacy")
        performance_layout = QVBoxLayout(self.performance_group)
        performance_layout.addWidget(self.use_cache)
        performance_layout.addWidget(self.resume_completed)
        performance_layout.addWidget(self.block_cloud_synced_paths)
        performance_layout.addWidget(self.clear_cache)

        advanced_content_layout.addWidget(self.engine_group)
        advanced_content_layout.addWidget(self.danish_group)
        advanced_content_layout.addWidget(self.performance_group)
        advanced_content_layout.addWidget(self.display_group)
        advanced_layout.addWidget(self.advanced_content)

        for button in (
            self.whisper_browse,
            self.model_browse,
            self.danish_model_browse,
            self.install_danish_whisper,
            self.mfa_browse,
            self.mfa_model_browse,
            self.mfa_dictionary_browse,
            self.download_mfa_preset,
            self.clear_cache,
            self.refresh_dependencies,
        ):
            button.setCursor(Qt.PointingHandCursor)
        self.dependencies_toggle.setCursor(Qt.PointingHandCursor)
        self.advanced_toggle.setCursor(Qt.PointingHandCursor)

        layout.addWidget(self.dependencies_box)
        layout.addWidget(self.output_group)
        layout.addWidget(self.section_group)
        layout.addWidget(self.advanced_box)
        layout.addStretch(1)

        self.setStyleSheet(
            """
            QPlainTextEdit#ReviewNote {
                background: #f6f8fb;
                border: 1px solid #d5dbe5;
                border-radius: 6px;
                color: #243041;
                padding: 8px;
            }
            """
        )

    def _wire_events(self) -> None:
        self.whisper_browse.clicked.connect(self.choose_whisper)
        self.model_browse.clicked.connect(self.choose_model)
        self.danish_model_browse.clicked.connect(self.choose_danish_model)
        self.install_danish_whisper.clicked.connect(lambda: self.danish_whisper_install_requested.emit())
        self.mfa_browse.clicked.connect(self.choose_mfa)
        self.mfa_model_browse.clicked.connect(self.choose_mfa_model)
        self.mfa_dictionary_browse.clicked.connect(self.choose_mfa_dictionary)
        self.download_mfa_preset.clicked.connect(self.request_mfa_preset_download)
        self.clear_cache.clicked.connect(lambda: self.clear_cache_requested.emit())
        self.refresh_dependencies.clicked.connect(self.refresh_dependency_status)
        self.dependencies_toggle.toggled.connect(self._toggle_dependencies)
        self.advanced_toggle.toggled.connect(self._toggle_advanced_setup)
        self.transcribe_section.toggled.connect(self._update_visibility)
        self.transcription_mode.currentIndexChanged.connect(self._update_visibility)
        self.asr_backend.currentIndexChanged.connect(self._asr_backend_changed)
        self.mfa_preset.currentIndexChanged.connect(self._mfa_preset_changed)
        self.language.currentIndexChanged.connect(self._language_changed)
        self.model_profile.currentIndexChanged.connect(lambda _value: self.options_changed.emit())
        self.transcription_mode.currentIndexChanged.connect(lambda _value: self.options_changed.emit())

        for widget in (
            self.whisper_path,
            self.model_path,
            self.danish_model_path,
            self.start_time,
            self.finish_time,
            self.mfa_path,
            self.mfa_model_path,
            self.mfa_dictionary_path,
        ):
            widget.textChanged.connect(self.options_changed)
        for checkbox in (
            self.transcribe_section,
            self.prefer_gpu,
            self.use_cache,
            self.resume_completed,
            self.output_rtf,
            self.output_json,
            self.block_cloud_synced_paths,
            self.use_ipa_font_regular,
            self.use_ipa_font_jeffersonian,
        ):
            checkbox.toggled.connect(self.options_changed)
        self.jeffersonian_line_width.valueChanged.connect(lambda _value: self.options_changed.emit())
        self.known_speakers.valueChanged.connect(lambda _value: self.options_changed.emit())

    def _toggle_dependencies(self, checked: bool) -> None:
        self.dependencies_content.setVisible(checked)
        self.dependencies_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _toggle_advanced_setup(self, checked: bool) -> None:
        self.advanced_content.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _update_visibility(self) -> None:
        enabled = self.transcribe_section.isChecked()
        self.start_time.setEnabled(enabled)
        self.finish_time.setEnabled(enabled)
        mode = self.current_transcription_mode()
        danish_selected = self.current_asr_backend() == ASR_BACKEND_DANISH_WHISPER
        self.language.setEnabled(not danish_selected)
        is_jeffersonian = mode != VERBATIM_TRANSCRIPTION
        mfa_enabled = mode == NARROW_JEFFERSONIAN_TRANSCRIPTION
        self.jeffersonian_line_width.setEnabled(is_jeffersonian)
        self.known_speakers.setEnabled(True)
        self.diarization_auto_hint.setEnabled(True)
        self.use_ipa_font_regular.setEnabled(mode == VERBATIM_TRANSCRIPTION)
        self.use_ipa_font_jeffersonian.setEnabled(is_jeffersonian)
        for widget in (
            self.mfa_preset,
            self.mfa_preset_hint,
            self.download_mfa_preset,
            self.mfa_path,
            self.mfa_model_path,
            self.mfa_dictionary_path,
            self.mfa_browse,
            self.mfa_model_browse,
            self.mfa_dictionary_browse,
        ):
            widget.setEnabled(mfa_enabled)
        self._update_mfa_preset_hint()

    def _asr_backend_changed(self) -> None:
        if self.current_asr_backend() == ASR_BACKEND_DANISH_WHISPER:
            danish_index = self.language.findData("da")
            if danish_index >= 0:
                self.language.setCurrentIndex(danish_index)
        self._update_visibility()
        self.options_changed.emit()

    def apply_dependency_status(self, status: DependencyStatus) -> None:
        if status.whisper_path and not self.whisper_path.text().strip():
            self.whisper_path.setText(status.whisper_path)
        if status.model_path and not self.model_path.text().strip():
            self.model_path.setText(status.model_path)
        if status.danish_whisper_model_path and not self.danish_model_path.text().strip():
            self.danish_model_path.setText(status.danish_whisper_model_path)
        if status.fully_ready:
            self.status_label.setText("Ready: all local dependencies found.")
        else:
            missing = ", ".join(status.missing_labels)
            self.status_label.setText(f"Missing local component(s): {missing}. Use setup/update or choose local paths.")
        self._update_mfa_preset_hint()

    def refresh_dependency_status(self) -> None:
        self.apply_dependency_status(dependency_status())

    def request_mfa_preset_download(self) -> None:
        self.mfa_preset_download_requested.emit(self.current_mfa_preset_id())

    def set_danish_whisper_install_running(self, running: bool) -> None:
        self.install_danish_whisper.setEnabled(not running)
        self.danish_model_browse.setEnabled(not running)
        if running:
            self.danish_hint.setText("Installing Røst v3 locally. The model is about 3.1 GB, so this may take a while.")
        else:
            self.danish_hint.setText(
                "Optional Danish ASR trained on conversational and read Danish. Setup downloads model files only; "
                "recording processing remains offline."
            )

    def set_mfa_download_running(self, running: bool) -> None:
        narrow = self.current_transcription_mode() == NARROW_JEFFERSONIAN_TRANSCRIPTION
        self.download_mfa_preset.setEnabled(not running and narrow)
        self.mfa_preset.setEnabled(not running and narrow)
        if running:
            self.mfa_preset_hint.setText("Downloading selected local MFA preset. This may take a while.")
        else:
            self._update_mfa_preset_hint()

    def current_transcription_mode(self) -> str:
        return str(self.transcription_mode.currentData() or VERBATIM_TRANSCRIPTION)

    def current_asr_backend(self) -> str:
        return str(self.asr_backend.currentData() or ASR_BACKEND_DOTE_WHISPER)

    def current_mfa_preset_id(self) -> str:
        return str(self.mfa_preset.currentData() or "")

    def _language_changed(self) -> None:
        preset = preset_for_language_code(str(self.language.currentData() or ""))
        if preset:
            index = self.mfa_preset.findData(preset.id)
            if index >= 0 and self.mfa_preset.currentIndex() != index:
                self.mfa_preset.setCurrentIndex(index)
        self._update_mfa_preset_hint()
        self.options_changed.emit()

    def _mfa_preset_changed(self) -> None:
        self._apply_selected_mfa_preset_paths(force=True)
        self._update_mfa_preset_hint()
        self.options_changed.emit()

    def _apply_selected_mfa_preset_paths(self, force: bool) -> None:
        preset = mfa_preset_by_id(self.current_mfa_preset_id())
        if not preset:
            return
        acoustic_path = find_mfa_acoustic_model(preset.acoustic_model)
        dictionary_path = find_mfa_dictionary(preset.dictionary_model)
        if acoustic_path and (force or not self.mfa_model_path.text().strip()):
            self.mfa_model_path.setText(acoustic_path)
        elif force:
            self.mfa_model_path.clear()
        if dictionary_path and (force or not self.mfa_dictionary_path.text().strip()):
            self.mfa_dictionary_path.setText(dictionary_path)
        elif force:
            self.mfa_dictionary_path.clear()

    def _update_mfa_preset_hint(self) -> None:
        preset = mfa_preset_by_id(self.current_mfa_preset_id())
        if not preset:
            self.mfa_preset_hint.setText("Choose an MFA preset for narrow Jeffersonian transcription.")
            return

        acoustic_path = find_mfa_acoustic_model(preset.acoustic_model)
        dictionary_path = find_mfa_dictionary(preset.dictionary_model)
        installed = bool(acoustic_path and dictionary_path)
        language_code = str(self.language.currentData() or "")
        mfa_languages = language_codes_with_mfa_presets()
        unsupported_note = ""
        if language_code and language_code not in mfa_languages:
            unsupported_note = (
                " The selected transcription language has no bundled MFA preset here; "
                "verbatim and broad Jeffersonian transcription still work locally."
            )
        if installed:
            text = f"Selected MFA preset is installed: {preset.acoustic_model} + {preset.dictionary_model}."
        else:
            text = f"Selected MFA preset needs download: {preset.acoustic_model} + {preset.dictionary_model}."
        if preset.note:
            text += f" {preset.note}"
        self.mfa_preset_hint.setText(text + unsupported_note)

    def choose_whisper(self) -> None:
        current = self.whisper_path.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose whisper.cpp executable", start_dir, "Executables (*.exe *);;All files (*)")
        if path:
            self.whisper_path.setText(path)

    def choose_model(self) -> None:
        current = self.model_path.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local Whisper model", start_dir, "Whisper models (*.bin *.gguf);;All files (*)")
        if path:
            self.model_path.setText(path)

    def choose_danish_model(self) -> None:
        current = self.danish_model_path.text().strip()
        start_dir = current if current else ""
        path = QFileDialog.getExistingDirectory(self, "Choose local Røst v3 model folder", start_dir)
        if path:
            self.danish_model_path.setText(path)

    def choose_mfa(self) -> None:
        current = self.mfa_path.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local MFA executable", start_dir, "Executables (*.exe *);;All files (*)")
        if path:
            self.mfa_path.setText(path)

    def choose_mfa_model(self) -> None:
        current = self.mfa_model_path.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local MFA acoustic model", start_dir, "MFA models (*.zip *.yaml *.json);;All files (*)")
        if path:
            self.mfa_model_path.setText(path)

    def choose_mfa_dictionary(self) -> None:
        current = self.mfa_dictionary_path.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local MFA pronunciation dictionary", start_dir, "Dictionaries (*.dict *.txt);;All files (*)")
        if path:
            self.mfa_dictionary_path.setText(path)

    def selected_options(self) -> TranscriptionOptions:
        return TranscriptionOptions(
            whisper_executable=self.whisper_path.text().strip(),
            model_path=self.model_path.text().strip(),
            asr_backend=self.current_asr_backend(),
            danish_model_path=self.danish_model_path.text().strip(),
            transcription_mode=self.current_transcription_mode(),
            language_code=str(self.language.currentData() or ""),
            transcribe_section=self.transcribe_section.isChecked(),
            start_time=self.start_time.text().strip(),
            finish_time=self.finish_time.text().strip(),
            jeffersonian=self.current_transcription_mode() != VERBATIM_TRANSCRIPTION,
            jeffersonian_line_width=self.jeffersonian_line_width.value(),
            prefer_gpu=self.prefer_gpu.isChecked(),
            model_profile=str(self.model_profile.currentData() or "auto"),
            use_cache=self.use_cache.isChecked(),
            resume_completed=self.resume_completed.isChecked(),
            output_rtf=self.output_rtf.isChecked(),
            output_json=self.output_json.isChecked(),
            block_cloud_synced_paths=self.block_cloud_synced_paths.isChecked(),
            use_mfa_alignment=self.current_transcription_mode() == NARROW_JEFFERSONIAN_TRANSCRIPTION,
            mfa_executable=self.mfa_path.text().strip(),
            mfa_acoustic_model=self.mfa_model_path.text().strip(),
            mfa_dictionary=self.mfa_dictionary_path.text().strip(),
            pyannote_pipeline_path="",
            known_speakers=self.known_speakers.value(),
            use_ipa_font_regular=self.use_ipa_font_regular.isChecked(),
            use_ipa_font_jeffersonian=self.use_ipa_font_jeffersonian.isChecked(),
            export_mfa_phone_transcript=False,
            keep_temp_audio=False,
        )
