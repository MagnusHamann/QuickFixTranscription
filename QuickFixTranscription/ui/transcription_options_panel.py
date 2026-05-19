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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from transcription.dependencies import DependencyStatus, dependency_status
from transcription.languages import LANGUAGE_CHOICES
from transcription.models import TranscriptionOptions


class TranscriptionOptionsPanel(QWidget):
    """Collects local transcription settings."""

    options_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        self.whisper_path = QLineEdit()
        self.whisper_path.setPlaceholderText("Path to whisper.cpp executable")
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to local Whisper model")
        self.whisper_browse = QPushButton("Browse")
        self.model_browse = QPushButton("Browse")

        self.language = QComboBox()
        for code, label in LANGUAGE_CHOICES:
            self.language.addItem(label, code)

        self.transcribe_section = QCheckBox("Transcribe section")
        self.start_time = QLineEdit()
        self.finish_time = QLineEdit()
        self.start_time.setPlaceholderText("mm:ss or hh:mm:ss")
        self.finish_time.setPlaceholderText("mm:ss or hh:mm:ss")

        self.jeffersonian = QCheckBox("Upgrade to simple Jeffersonian transcription")
        self.keep_temp_audio = QCheckBox("Keep temporary WAV files")

        self.refresh_dependencies = QPushButton("Refresh dependency check")

        self._build_layout()
        self._wire_events()
        self.apply_dependency_status(dependency_status())
        self._update_visibility()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        status_group = QGroupBox("Local setup")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.refresh_dependencies)

        engine_group = QGroupBox("Local engine")
        engine_layout = QFormLayout(engine_group)
        whisper_row = QHBoxLayout()
        whisper_row.addWidget(self.whisper_path)
        whisper_row.addWidget(self.whisper_browse)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path)
        model_row.addWidget(self.model_browse)
        engine_layout.addRow("whisper.cpp", whisper_row)
        engine_layout.addRow("Model", model_row)
        engine_layout.addRow("Language", self.language)

        section_group = QGroupBox("Section")
        section_layout = QFormLayout(section_group)
        section_layout.addRow(self.transcribe_section)
        section_layout.addRow("Start", self.start_time)
        section_layout.addRow("Finish", self.finish_time)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(self.jeffersonian)
        output_layout.addWidget(self.keep_temp_audio)

        for button in (self.whisper_browse, self.model_browse, self.refresh_dependencies):
            button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(status_group)
        layout.addWidget(engine_group)
        layout.addWidget(section_group)
        layout.addWidget(output_group)
        layout.addStretch(1)

    def _wire_events(self) -> None:
        self.whisper_browse.clicked.connect(self.choose_whisper)
        self.model_browse.clicked.connect(self.choose_model)
        self.refresh_dependencies.clicked.connect(self.refresh_dependency_status)
        self.transcribe_section.toggled.connect(self._update_visibility)

        for widget in (
            self.whisper_path,
            self.model_path,
            self.start_time,
            self.finish_time,
        ):
            widget.textChanged.connect(self.options_changed)
        for checkbox in (self.transcribe_section, self.jeffersonian, self.keep_temp_audio):
            checkbox.toggled.connect(self.options_changed)
        self.language.currentIndexChanged.connect(self.options_changed)

    def _update_visibility(self) -> None:
        enabled = self.transcribe_section.isChecked()
        self.start_time.setEnabled(enabled)
        self.finish_time.setEnabled(enabled)

    def apply_dependency_status(self, status: DependencyStatus) -> None:
        if status.whisper_path and not self.whisper_path.text().strip():
            self.whisper_path.setText(status.whisper_path)
        if status.model_path and not self.model_path.text().strip():
            self.model_path.setText(status.model_path)

        if status.ready_for_transcription:
            self.status_label.setText("Ready: local FFmpeg, whisper.cpp, and model found.")
        else:
            missing = ", ".join(status.missing_labels)
            self.status_label.setText(f"Missing local component(s): {missing}. Use setup/update or choose local paths.")

    def refresh_dependency_status(self) -> None:
        self.apply_dependency_status(dependency_status())

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

    def selected_options(self) -> TranscriptionOptions:
        return TranscriptionOptions(
            whisper_executable=self.whisper_path.text().strip(),
            model_path=self.model_path.text().strip(),
            language_code=str(self.language.currentData() or ""),
            transcribe_section=self.transcribe_section.isChecked(),
            start_time=self.start_time.text().strip(),
            finish_time=self.finish_time.text().strip(),
            jeffersonian=self.jeffersonian.isChecked(),
            keep_temp_audio=self.keep_temp_audio.isChecked(),
        )

