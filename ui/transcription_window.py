"""Main window for QuickFixTranscription."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ffmpeg.ffmpeg_runner import FFmpegRunner, find_ffmpeg_tools
from transcription.batch_processor import TranscriptionBatchProcessor
from transcription.dependencies import dependency_status
from transcription.file_utils import collect_media_files, human_size, probe_media_record
from transcription.mfa_presets import mfa_preset_by_id
from transcription.models import MediaRecord
from ui.drag_drop_widget import DragDropWidget
from ui.transcription_options_panel import TranscriptionOptionsPanel


class MfaPresetDownloadWorker(QObject):
    """Download one selected MFA preset without blocking the main UI."""

    log_message = Signal(str)
    finished = Signal(bool)

    def __init__(self, preset_id: str) -> None:
        super().__init__()
        self.preset_id = preset_id

    @Slot()
    def run(self) -> None:
        try:
            from transcription.mfa_setup import download_mfa_preset, install_mfa

            install_mfa(self.log_message.emit)
            ok = download_mfa_preset(self.preset_id, self.log_message.emit)
        except Exception as exc:
            self.log_message.emit(f"MFA preset download failed: {exc}")
            ok = False
        self.finished.emit(ok)


class SherpaSetupWorker(QObject):
    """Install optional sherpa-onnx package/models without blocking the main UI."""

    log_message = Signal(str)
    finished = Signal(bool)

    @Slot()
    def run(self) -> None:
        try:
            from transcription.sherpa_setup import setup_sherpa

            ok = setup_sherpa(self.log_message.emit)
        except Exception as exc:
            self.log_message.emit(f"sherpa-onnx setup failed: {exc}")
            ok = False
        self.finished.emit(ok)


class TranscriptionWindow(QMainWindow):
    """Top-level UI for local batch transcription."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QuickFixTranscription")
        self.records: list[MediaRecord] = []
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionBatchProcessor | None = None
        self.mfa_download_thread: QThread | None = None
        self.mfa_download_worker: MfaPresetDownloadWorker | None = None
        self.sherpa_setup_thread: QThread | None = None
        self.sherpa_setup_worker: SherpaSetupWorker | None = None
        self.runner = FFmpegRunner()

        self.drop_zone = DragDropWidget(
            "Drop audio/video files or folders here\nSupported: MP4, MOV, MKV, AVI, MP3, WAV, M4A, AAC, FLAC"
        )
        self.table = QTableWidget(0, 4)
        self.options_panel = TranscriptionOptionsPanel()
        self.start_button = QPushButton("Start Transcription")
        self.cancel_button = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.log = QPlainTextEdit()
        self.status = QStatusBar()

        self._build_ui()
        self._wire_events()
        self._check_ffmpeg()

    def _build_ui(self) -> None:
        add_files = QAction("Add files", self)
        add_folder = QAction("Add folder", self)
        clear = QAction("Clear list", self)
        self.toolbar = self.addToolBar("Files")
        self.toolbar.addAction(add_files)
        self.toolbar.addAction(add_folder)
        self.toolbar.addAction(clear)
        add_files.triggered.connect(self.choose_files)
        add_folder.triggered.connect(self.choose_folder)
        clear.triggered.connect(self.clear_files)

        self.table.setHorizontalHeaderLabels(["Filename", "Duration", "Type", "Size"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        ingest = QWidget()
        ingest_layout = QVBoxLayout(ingest)
        ingest_layout.addWidget(self.drop_zone)
        ingest_layout.addWidget(QLabel("Detected media"))
        ingest_layout.addWidget(self.table)

        options_scroll = QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setFrameShape(QScrollArea.NoFrame)
        options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        options_scroll.setWidget(self.options_panel)
        options_scroll.setMinimumWidth(360)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(ingest)
        splitter.addWidget(options_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 100)

        bottom_buttons = QHBoxLayout()
        bottom_buttons.addWidget(self.start_button)
        bottom_buttons.addWidget(self.cancel_button)
        bottom_buttons.addWidget(self.progress)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter, stretch=1)
        root_layout.addLayout(bottom_buttons)
        root_layout.addWidget(QLabel("Processing log"))
        root_layout.addWidget(self.log, stretch=0)

        self.setCentralWidget(root)
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        self.setStyleSheet(
            """
            QPushButton {
                min-height: 30px;
                padding: 6px 12px;
            }
            QPushButton#Primary {
                font-weight: 600;
            }
            QPlainTextEdit {
                background: #101827;
                color: #d7e1f0;
                font-family: Consolas, Menlo, monospace;
            }
            """
        )
        self.start_button.setObjectName("Primary")

    def _wire_events(self) -> None:
        self.drop_zone.paths_dropped.connect(self.add_paths)
        self.start_button.clicked.connect(self.start_transcription)
        self.cancel_button.clicked.connect(self.cancel_transcription)
        self.options_panel.mfa_preset_download_requested.connect(self.download_mfa_preset)
        self.options_panel.sherpa_setup_requested.connect(self.download_sherpa_setup)

    def _check_ffmpeg(self) -> None:
        ffmpeg, ffprobe = find_ffmpeg_tools()
        if not ffmpeg or not ffprobe:
            self.status.showMessage("FFmpeg not found")
            self.log.appendPlainText("FFmpeg/FFprobe not found. Run setup/update or install a local FFmpeg binary.")
            return
        self.runner.ffmpeg_path = ffmpeg
        self.runner.ffprobe_path = ffprobe
        self.status.showMessage(f"Using FFmpeg: {ffmpeg}")

    @Slot()
    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose media files",
            "",
            "Media files (*.mp4 *.mov *.mkv *.avi *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.opus *.webm *.wma)",
        )
        if paths:
            self.add_paths(paths)

    @Slot()
    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder")
        if folder:
            self.add_paths([folder])

    @Slot(list)
    def add_paths(self, raw_paths: list[str]) -> None:
        try:
            files = collect_media_files([Path(path) for path in raw_paths])
        except ValueError as exc:
            QMessageBox.warning(self, "Input problem", str(exc))
            return

        if not files:
            QMessageBox.information(self, "No media found", "No supported audio or video files were found.")
            return

        existing = {record.path.resolve() for record in self.records}
        added = 0
        for file_path in files:
            resolved = file_path.resolve()
            if resolved in existing:
                continue
            record = probe_media_record(resolved, self.runner)
            self.records.append(record)
            existing.add(resolved)
            added += 1

        self.refresh_table()
        self.status.showMessage(f"Added {added} file(s). {len(self.records)} total.")

    def refresh_table(self) -> None:
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            values = [
                record.path.name,
                record.duration_label,
                record.kind_label,
                human_size(record.size_bytes),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(record.path))
                self.table.setItem(row, column, item)

    @Slot()
    def clear_files(self) -> None:
        self.records.clear()
        self.refresh_table()
        self.status.showMessage("File list cleared")

    @Slot()
    def start_transcription(self) -> None:
        if not self.records:
            QMessageBox.information(self, "No files", "Drop or choose at least one media file first.")
            return
        if not self.runner.ready:
            QMessageBox.warning(self, "FFmpeg not found", "Install FFmpeg before transcribing files.")
            return

        options = self.options_panel.selected_options()
        try:
            options.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid options", str(exc))
            return

        self.log.clear()
        self.progress.setValue(0)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.showMessage("Transcribing...")

        self.worker_thread = QThread(self)
        self.worker = TranscriptionBatchProcessor(self.records, options, self.runner)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log)
        self.worker.progress_changed.connect(self.progress.setValue)
        self.worker.status_changed.connect(self.status.showMessage)
        self.worker.finished.connect(self.transcription_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @Slot()
    def cancel_transcription(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.append_log("Cancellation requested. Current local process will be stopped.")

    @Slot(str)
    def download_mfa_preset(self, preset_id: str) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Transcription running", "Finish or cancel transcription before downloading setup files.")
            return
        if self.sherpa_setup_thread is not None:
            QMessageBox.information(self, "sherpa-onnx setup running", "Finish the sherpa-onnx setup before downloading MFA presets.")
            return
        if self.mfa_download_thread is not None:
            QMessageBox.information(self, "MFA setup running", "An MFA preset download is already running.")
            return
        preset = mfa_preset_by_id(preset_id)
        if preset is None:
            QMessageBox.warning(self, "Unknown MFA preset", "Choose a valid MFA preset first.")
            return

        self.append_log(f"Downloading local MFA preset: {preset.label}")
        self.append_log("This setup may use the internet for model files, but it does not upload recordings or transcripts.")
        self.options_panel.set_mfa_download_running(True)
        self.status.showMessage("Downloading MFA preset...")

        self.mfa_download_thread = QThread(self)
        self.mfa_download_worker = MfaPresetDownloadWorker(preset_id)
        self.mfa_download_worker.moveToThread(self.mfa_download_thread)
        self.mfa_download_thread.started.connect(self.mfa_download_worker.run)
        self.mfa_download_worker.log_message.connect(self.append_log)
        self.mfa_download_worker.finished.connect(self.mfa_preset_download_finished)
        self.mfa_download_worker.finished.connect(self.mfa_download_thread.quit)
        self.mfa_download_worker.finished.connect(self.mfa_download_worker.deleteLater)
        self.mfa_download_thread.finished.connect(self.mfa_download_thread.deleteLater)
        self.mfa_download_thread.start()

    @Slot()
    def download_sherpa_setup(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Transcription running", "Finish or cancel transcription before downloading setup files.")
            return
        if self.mfa_download_thread is not None:
            QMessageBox.information(self, "MFA setup running", "Finish the MFA setup before downloading sherpa-onnx files.")
            return
        if self.sherpa_setup_thread is not None:
            QMessageBox.information(self, "sherpa-onnx setup running", "The sherpa-onnx setup is already running.")
            return

        self.append_log("Downloading optional local sherpa-onnx setup.")
        self.append_log("This setup may use the internet for package/model files, but it does not upload recordings or transcripts.")
        self.options_panel.set_sherpa_setup_running(True)
        self.status.showMessage("Downloading sherpa-onnx setup...")

        self.sherpa_setup_thread = QThread(self)
        self.sherpa_setup_worker = SherpaSetupWorker()
        self.sherpa_setup_worker.moveToThread(self.sherpa_setup_thread)
        self.sherpa_setup_thread.started.connect(self.sherpa_setup_worker.run)
        self.sherpa_setup_worker.log_message.connect(self.append_log)
        self.sherpa_setup_worker.finished.connect(self.sherpa_setup_finished)
        self.sherpa_setup_worker.finished.connect(self.sherpa_setup_thread.quit)
        self.sherpa_setup_worker.finished.connect(self.sherpa_setup_worker.deleteLater)
        self.sherpa_setup_thread.finished.connect(self.sherpa_setup_thread.deleteLater)
        self.sherpa_setup_thread.start()

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    @Slot(bool)
    def mfa_preset_download_finished(self, ok: bool) -> None:
        self.mfa_download_worker = None
        self.mfa_download_thread = None
        self.options_panel.set_mfa_download_running(False)
        self.options_panel.apply_dependency_status(dependency_status())
        self.status.showMessage("MFA preset download complete." if ok else "MFA preset download failed.")
        if ok:
            QMessageBox.information(self, "MFA preset ready", "The selected local MFA preset is ready.")
        else:
            QMessageBox.warning(self, "MFA preset problem", "The selected MFA preset could not be downloaded.")

    @Slot(bool)
    def sherpa_setup_finished(self, ok: bool) -> None:
        self.sherpa_setup_worker = None
        self.sherpa_setup_thread = None
        self.options_panel.set_sherpa_setup_running(False)
        self.options_panel.apply_dependency_status(dependency_status())
        self.status.showMessage("sherpa-onnx setup complete." if ok else "sherpa-onnx setup failed.")
        if ok:
            QMessageBox.information(self, "sherpa-onnx ready", "The optional local sherpa-onnx diarization setup is ready.")
        else:
            QMessageBox.warning(self, "sherpa-onnx problem", "The optional sherpa-onnx setup could not be completed.")

    @Slot(int, int)
    def transcription_finished(self, completed: int, failed: int) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None
        self.status.showMessage(f"Done. Completed: {completed}. Failed: {failed}.")
        QMessageBox.information(self, "Transcription complete", f"Completed: {completed}\nFailed: {failed}")
