"""Drag-and-drop input widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DragDropWidget(QWidget):
    """Large drop target that accepts files and folders."""

    paths_dropped = Signal(list)

    def __init__(self, label_text: str | None = None) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")

        self.label = QLabel(label_text or "Drop video files or folders here\nSupported: MP4, MOV, MKV, AVI")
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.setStyleSheet(
            """
            QWidget#DropZone {
                border: 2px dashed #7a8797;
                border-radius: 8px;
                background: #f6f8fb;
                color: #243041;
                font-size: 17px;
                min-height: 118px;
            }
            QWidget#DropZone[dragging="true"] {
                border-color: #2563eb;
                background: #eaf1ff;
            }
            """
        )

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.mimeData().hasUrls():
            self.setProperty("dragging", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)

        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())

        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
