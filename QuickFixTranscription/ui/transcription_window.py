"""Main window for QuickFixTranscription."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Slot
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
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ffmpeg.ffmpeg_runner import FFmpegRunner, find_ffmpeg_tools
from transcription.batch_processor import TranscriptionBatchProcessor
from transcription.file_utils import collect_media_files, human_size, probe_media_record
from transcription.models import MediaRecord
from ui.drag_drop_widget import DragDropWidget
from ui.transcription_options_panel import TranscriptionOptionsPanel


class TranscriptionWindow(QMainWindow):
    """Top-level UI for local batch transcription."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QuickFixTranscription")
        self.records: list[MediaRecord] = []
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionBatchProcessor | None = None
        self.runner = FFmpegRunner()

        self.drop_zone = DragDropWidget(
            "Drop audio/video files or folders here\nSupported: MP4, MOV, MKV, AVI, MP3, WAV, M4A, FLAC"
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

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.drop_zone)
        left_layout.addWidget(QLabel("Detected media"))
        left_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.options_panel)
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
    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    @Slot(int, int)
    def transcription_finished(self, completed: int, failed: int) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None
        self.status.showMessage(f"Done. Completed: {completed}. Failed: {failed}.")
        QMessageBox.information(self, "Transcription complete", f"Completed: {completed}\nFailed: {failed}")

