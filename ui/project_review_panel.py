"""Project review panel for local Jefferson candidate annotation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from transcription.project_exports import export_project_docx, export_project_pdf, export_project_txt
from transcription.project_renderer import (
    iter_project_suggestions,
    render_candidate_review_lines,
    render_confirmed_jefferson_lines,
)
from transcription.project_review import set_candidate_status
from transcription.project_schema import CandidateSuggestion, Project
from transcription.project_store import load_project, save_project


class ProjectReviewPanel(QWidget):
    """Small human-in-the-loop review surface for generated project files."""

    log_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.project_path: Path | None = None
        self.suggestions: tuple[CandidateSuggestion, ...] = ()

        self.path_label = QLabel("No project loaded")
        self.path_label.setWordWrap(True)
        self.local_label = QLabel("Local-only review: pending machine suggestions are never treated as final annotations.")
        self.local_label.setWordWrap(True)
        self.open_button = QPushButton("Open project")
        self.save_button = QPushButton("Save")
        self.confirm_button = QPushButton("Confirm")
        self.reject_button = QPushButton("Reject")
        self.export_txt_button = QPushButton("Export TXT")
        self.export_docx_button = QPushButton("Export DOCX")
        self.export_pdf_button = QPushButton("Export PDF")
        self.table = QTableWidget(0, 6)
        self.preview = QPlainTextEdit()

        self._build_ui()
        self._wire_events()
        self._update_actions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(self.local_label)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.save_button)
        controls.addStretch(1)
        controls.addWidget(self.confirm_button)
        controls.addWidget(self.reject_button)
        layout.addLayout(controls)

        self.table.setHorizontalHeaderLabels(["Status", "Type", "Time", "Symbol", "Source", "Detail"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("Annotation review queue"))
        layout.addWidget(self.table, stretch=2)

        export_controls = QHBoxLayout()
        export_controls.addWidget(self.export_txt_button)
        export_controls.addWidget(self.export_docx_button)
        export_controls.addWidget(self.export_pdf_button)
        export_controls.addStretch(1)
        layout.addLayout(export_controls)

        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(180)
        self.preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(QLabel("Confirmed transcript preview"))
        layout.addWidget(self.preview, stretch=2)

    def _wire_events(self) -> None:
        self.open_button.clicked.connect(self.open_project_dialog)
        self.save_button.clicked.connect(self.save_current_project)
        self.confirm_button.clicked.connect(lambda: self._set_selected_status("confirmed"))
        self.reject_button.clicked.connect(lambda: self._set_selected_status("rejected"))
        self.export_txt_button.clicked.connect(lambda: self._export("txt"))
        self.export_docx_button.clicked.connect(lambda: self._export("docx"))
        self.export_pdf_button.clicked.connect(lambda: self._export("pdf"))
        self.table.itemSelectionChanged.connect(self._update_actions)

    @Slot()
    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open QuickFixTranscription project",
            "",
            "QuickFixTranscription projects (*.qftproj);;All files (*)",
        )
        if path:
            self.load_project_path(path)

    @Slot(str)
    def load_project_path(self, raw_path: str) -> None:
        path = Path(raw_path)
        try:
            self.project = load_project(path)
        except Exception as exc:
            QMessageBox.warning(self, "Project problem", f"Could not open project:\n{exc}")
            return
        self.project_path = path
        self.path_label.setText(f"Project: {path}")
        self._refresh()
        self.log_message.emit(f"Loaded project: {path}")

    @Slot()
    def save_current_project(self) -> None:
        if self.project is None or self.project_path is None:
            return
        path = save_project(self.project, self.project_path)
        self.log_message.emit(f"Saved project: {path}")

    def _refresh(self) -> None:
        if self.project is None:
            self.suggestions = ()
            self.table.setRowCount(0)
            self.preview.clear()
            self._update_actions()
            return

        self.suggestions = iter_project_suggestions(self.project, status=None)
        self.table.setRowCount(len(self.suggestions))
        for row, suggestion in enumerate(self.suggestions):
            values = [
                suggestion.status,
                suggestion.label,
                self._time_range(suggestion),
                suggestion.suggested_symbol,
                suggestion.source,
                suggestion.detail,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 0:
                    item.setData(Qt.UserRole, suggestion.id)
                self.table.setItem(row, column, item)

        lines = [
            *render_confirmed_jefferson_lines(self.project),
            "",
            *render_candidate_review_lines(self.project),
        ]
        self.preview.setPlainText("\n".join(lines))
        self._update_actions()

    def _selected_candidate_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return str(item.data(Qt.UserRole)) if item else None

    def _set_selected_status(self, status: str) -> None:
        if self.project is None:
            return
        candidate_id = self._selected_candidate_id()
        if not candidate_id:
            return
        try:
            self.project = set_candidate_status(self.project, candidate_id, status)
        except Exception as exc:
            QMessageBox.warning(self, "Review problem", str(exc))
            return
        self._refresh()
        self.save_current_project()

    def _export(self, kind: str) -> None:
        if self.project is None or self.project_path is None:
            return
        default_path = self.project_path.with_name(f"{self.project.name}_confirmed_jefferson.{kind}")
        filters = {
            "txt": "Text files (*.txt);;All files (*)",
            "docx": "Word documents (*.docx);;All files (*)",
            "pdf": "PDF files (*.pdf);;All files (*)",
        }
        path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()}", str(default_path), filters[kind])
        if not path:
            return
        exporters = {
            "txt": export_project_txt,
            "docx": export_project_docx,
            "pdf": export_project_pdf,
        }
        try:
            created = exporters[kind](self.project, Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Export problem", str(exc))
            return
        self.log_message.emit(f"Created: {created}")

    def _update_actions(self) -> None:
        has_project = self.project is not None
        has_selection = self._selected_candidate_id() is not None
        self.save_button.setEnabled(has_project)
        self.export_txt_button.setEnabled(has_project)
        self.export_docx_button.setEnabled(has_project)
        self.export_pdf_button.setEnabled(has_project)
        self.confirm_button.setEnabled(has_project and has_selection)
        self.reject_button.setEnabled(has_project and has_selection)

    @staticmethod
    def _time_range(suggestion: CandidateSuggestion) -> str:
        start = "" if suggestion.span.start is None else f"{suggestion.span.start:.2f}s"
        end = "" if suggestion.span.end is None else f"{suggestion.span.end:.2f}s"
        if start and end:
            return f"{start}-{end}"
        return start or end
