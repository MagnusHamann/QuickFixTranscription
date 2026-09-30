"""Local launcher and Windows host for the separately packaged DOTE Whisper app."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import re
from typing import Iterable

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


_DOTE_INSTALLER_PATTERN = "DOTE-Whisper-win-x64-*.exe"
_DOTE_EXECUTABLE_NAMES = {"dote-whisper.exe", "dote whisper.exe"}
_SEMANTIC_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")


def _version_key(path: Path) -> tuple[int, ...]:
    version = dote_version_from_path(path)
    if version is not None:
        return version
    values = re.findall(r"\d+", path.stem)
    return tuple(int(value) for value in values)


def dote_version_from_path(path: Path) -> tuple[int, int, int] | None:
    """Read a semantic DOTE version from an installer or app-version directory."""

    for value in (path.name, path.parent.name):
        match = _SEMANTIC_VERSION_PATTERN.search(value)
        if match:
            return tuple(int(part) for part in match.groups())
    return None


def format_dote_version(version: tuple[int, int, int] | None) -> str:
    return ".".join(str(part) for part in version) if version else "unknown"


def find_dote_installer(quickfix_root: Path | None = None) -> Path | None:
    """Return the newest adjacent or automatically cached DOTE installer."""

    root = (quickfix_root or Path(__file__).resolve().parents[2]).resolve()
    installer_roots = (
        root,
        root / "QuickFixAppDependencies" / ".tools" / "dote-whisper" / "installers",
    )
    candidates = [
        path
        for installer_root in installer_roots
        for path in installer_root.glob(_DOTE_INSTALLER_PATTERN)
        if path.is_file()
    ]
    return max(candidates, key=_version_key) if candidates else None


def _candidate_install_roots(local_app_data: Path) -> Iterable[Path]:
    bases = (local_app_data, local_app_data / "Programs")
    for base in bases:
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.lower().startswith("dote-whisper"):
                yield child


def find_dote_executable(local_app_data: Path | None = None) -> Path | None:
    """Locate DOTE Whisper without confusing it with the separate DOTE app."""

    configured_base = os.environ.get("LOCALAPPDATA")
    if local_app_data is None and not configured_base:
        return None
    base = local_app_data or Path(configured_base or "")

    candidates: list[Path] = []
    try:
        roots = tuple(_candidate_install_roots(base))
    except OSError:
        return None

    for root in roots:
        try:
            children = tuple(root.iterdir())
        except OSError:
            continue
        for candidate in (root, *(child for child in children if child.is_dir() and child.name.lower().startswith("app-"))):
            try:
                for item in candidate.iterdir():
                    if item.is_file() and item.name.lower() in _DOTE_EXECUTABLE_NAMES:
                        candidates.append(item)
            except OSError:
                continue

    if not candidates:
        return None

    # Prefer the root Squirrel stub because it follows installed updates.
    candidates.sort(key=lambda path: (path.parent.name.lower().startswith("app-"), _version_key(path.parent)))
    return candidates[0]


def find_installed_dote_version(local_app_data: Path | None = None) -> tuple[int, int, int] | None:
    """Return the newest installed DOTE Whisper app-directory version."""

    configured_base = os.environ.get("LOCALAPPDATA")
    if local_app_data is None and not configured_base:
        return None
    base = local_app_data or Path(configured_base or "")
    versions: list[tuple[int, int, int]] = []
    try:
        roots = tuple(_candidate_install_roots(base))
    except OSError:
        return None
    for root in roots:
        try:
            children = tuple(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or not child.name.lower().startswith("app-"):
                continue
            version = dote_version_from_path(child)
            if version is not None:
                versions.append(version)
    return max(versions) if versions else None


def _normalised_path(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(str(path))


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((_normalised_path(path), _normalised_path(parent))) == _normalised_path(parent)
    except ValueError:
        return False


def _process_image_path(process_id: int) -> Path | None:
    if os.name != "nt":
        return None

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    process = kernel32.OpenProcess(0x1000, False, process_id)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return None
    try:
        capacity = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(capacity.value)
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(capacity)):
            return None
        return Path(buffer.value)
    finally:
        kernel32.CloseHandle(process)


def find_dote_window(executable: Path) -> int | None:
    """Find DOTE Whisper's visible top-level window in its installation tree."""

    if os.name != "nt":
        return None

    user32 = ctypes.windll.user32
    matches: list[int] = []
    install_root = executable.parent
    if install_root.name.lower().startswith("app-"):
        install_root = install_root.parent

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    @callback_type
    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.GetWindowTextLengthW(hwnd) <= 0:
            return True
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        image = _process_image_path(process_id.value)
        if image and image.name.lower() in _DOTE_EXECUTABLE_NAMES and _path_is_inside(image, install_root):
            matches.append(int(hwnd))
        return True

    user32.EnumWindows(visit, 0)
    return matches[0] if matches else None


class DoteWindowHost(QWidget):
    """Native child-window host that does not give Qt ownership of Electron."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_NativeWindow)
        self.setMinimumSize(720, 480)
        self._dote_hwnd: int | None = None
        self._original_parent: int = 0
        self._original_style: int | None = None

    @staticmethod
    def _user32():
        user32 = ctypes.windll.user32
        user32.IsWindow.argtypes = (wintypes.HWND,)
        user32.IsWindow.restype = wintypes.BOOL
        user32.GetParent.argtypes = (wintypes.HWND,)
        user32.GetParent.restype = wintypes.HWND
        user32.SetParent.argtypes = (wintypes.HWND, wintypes.HWND)
        user32.SetParent.restype = wintypes.HWND
        user32.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        return user32

    def attach(self, hwnd: int) -> bool:
        if os.name != "nt":
            return False
        user32 = self._user32()
        if not user32.IsWindow(hwnd):
            return False

        self.detach()
        self._dote_hwnd = hwnd
        self._original_parent = int(user32.GetParent(hwnd) or 0)
        self._original_style = int(user32.GetWindowLongPtrW(hwnd, -16))  # GWL_STYLE

        ws_child = 0x40000000
        removable = 0x80000000 | 0x00C00000 | 0x00040000 | 0x00020000 | 0x00010000 | 0x00080000
        child_style = (self._original_style & 0xFFFFFFFF | ws_child) & ~removable
        user32.SetWindowLongPtrW(hwnd, -16, ctypes.c_ssize_t(child_style))
        user32.SetParent(hwnd, wintypes.HWND(int(self.winId())))
        self._resize_dote()
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        return True

    def _resize_dote(self) -> None:
        if os.name != "nt" or self._dote_hwnd is None:
            return
        user32 = self._user32()
        if not user32.IsWindow(self._dote_hwnd):
            self._dote_hwnd = None
            return
        scale = self.devicePixelRatioF()
        flags = 0x0004 | 0x0010 | 0x0020  # SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        user32.SetWindowPos(
            self._dote_hwnd,
            wintypes.HWND(0),
            0,
            0,
            max(1, round(self.width() * scale)),
            max(1, round(self.height() * scale)),
            flags,
        )

    def detach(self) -> None:
        if os.name != "nt" or self._dote_hwnd is None:
            return
        user32 = self._user32()
        hwnd = self._dote_hwnd
        self._dote_hwnd = None
        if not user32.IsWindow(hwnd):
            return
        user32.SetParent(hwnd, wintypes.HWND(self._original_parent))
        if self._original_style is not None:
            user32.SetWindowLongPtrW(hwnd, -16, ctypes.c_ssize_t(self._original_style))
        flags = 0x0004 | 0x0020 | 0x0040  # SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW
        user32.SetWindowPos(hwnd, wintypes.HWND(0), 100, 100, 1200, 800, flags)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_dote()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._resize_dote()


class DoteIntegrationPanel(QWidget):
    """Install, launch, and host DOTE Whisper while keeping its workflow separate."""

    def __init__(self, quickfix_root: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.quickfix_root = (quickfix_root or Path(__file__).resolve().parents[2]).resolve()
        self.installer_path = find_dote_installer(self.quickfix_root)
        self.executable_path = find_dote_executable()
        self.installed_version = find_installed_dote_version()
        self.setup_process: QProcess | None = None
        self.window_host = DoteWindowHost()
        self.window_embedded = False
        self._poll_attempts = 0
        self._prompted_for_install = False
        self._prompted_for_update = False

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.primary_button = QPushButton()
        self.primary_button.setObjectName("Primary")
        self.separate_button = QPushButton("Open in separate window")
        self.separate_button.setVisible(False)

        self.placeholder = QWidget()
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.setAlignment(Qt.AlignCenter)
        placeholder_layout.addStretch(1)
        placeholder_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.primary_button)
        actions.addWidget(self.separate_button)
        actions.addStretch(1)
        placeholder_layout.addLayout(actions)
        placeholder_layout.addStretch(1)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.window_host)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.stack)

        self.primary_button.clicked.connect(self._primary_action)
        self.separate_button.clicked.connect(self.open_separately)
        self._refresh_state()

    def _refresh_state(self) -> None:
        self.installer_path = find_dote_installer(self.quickfix_root)
        self.executable_path = find_dote_executable()
        self.installed_version = find_installed_dote_version()
        installer_version = dote_version_from_path(self.installer_path) if self.installer_path else None
        if self.executable_path and installer_version and (
            self.installed_version is None or installer_version > self.installed_version
        ):
            self.status_label.setText(
                f"DOTE transcription {format_dote_version(installer_version)} is ready to update."
            )
            self.primary_button.setText("Update DOTE transcription")
            self.primary_button.setEnabled(True)
            return
        if self.executable_path:
            version = format_dote_version(self.installed_version)
            self.status_label.setText(f"DOTE transcription {version} is ready.")
            self.primary_button.setText("Open DOTE transcription")
            self.primary_button.setEnabled(True)
            return
        if self.installer_path:
            self.status_label.setText("DOTE transcription needs its one-time local installation.")
            self.primary_button.setText("Install DOTE transcription")
            self.primary_button.setEnabled(True)
            return
        self.status_label.setText(
            "DOTE Whisper was not found. Place DOTE-Whisper-win-x64-<version>.exe in the Quickfix folder."
        )
        self.primary_button.setText("Check again")
        self.primary_button.setEnabled(True)

    def refresh_state(self) -> None:
        """Refresh installation state after the background setup worker finishes."""

        self._refresh_state()

    def activate(self) -> None:
        """Start or offer to install DOTE when its top-level tab is selected."""

        if self.window_embedded:
            return
        self._refresh_state()
        installer_version = dote_version_from_path(self.installer_path) if self.installer_path else None
        update_available = bool(
            self.executable_path
            and installer_version
            and (self.installed_version is None or installer_version > self.installed_version)
        )
        if update_available and not self._prompted_for_update:
            self._prompted_for_update = True
            answer = QMessageBox.question(
                self,
                "Update DOTE transcription",
                f"Install DOTE Whisper {format_dote_version(installer_version)} now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.install_dote()
            else:
                self.open_dote()
            return
        if self.executable_path:
            self.open_dote()
            return
        if self.installer_path and not self._prompted_for_install:
            self._prompted_for_install = True
            answer = QMessageBox.question(
                self,
                "Install DOTE transcription",
                "Install the DOTE Whisper package found in the Quickfix folder?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.install_dote()

    def _primary_action(self) -> None:
        installer_version = dote_version_from_path(self.installer_path) if self.installer_path else None
        update_available = bool(
            self.executable_path
            and installer_version
            and (self.installed_version is None or installer_version > self.installed_version)
        )
        if update_available or (not self.executable_path and self.installer_path):
            self.install_dote()
        elif self.executable_path:
            self.open_dote()
        else:
            self.installer_path = find_dote_installer(self.quickfix_root)
            self._refresh_state()

    def install_dote(self) -> None:
        if not self.installer_path or not self.installer_path.is_file():
            self.installer_path = find_dote_installer(self.quickfix_root)
        if not self.installer_path:
            self._refresh_state()
            return
        if self.setup_process is not None:
            return

        self.status_label.setText("Installing DOTE transcription locally...")
        self.primary_button.setEnabled(False)
        self.setup_process = QProcess(self)
        self.setup_process.finished.connect(self._installation_finished)
        self.setup_process.errorOccurred.connect(self._installation_error)
        self.setup_process.start(str(self.installer_path), ["--silent"])

    def _installation_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.setup_process = None
        self._refresh_state()
        if exit_code != 0 or not self.executable_path:
            self.status_label.setText(f"DOTE installation did not complete successfully (code {exit_code}).")
            self.primary_button.setEnabled(True)
            self.primary_button.setText("Try installation again")
            return
        self.status_label.setText("DOTE transcription is installed. Opening it now...")
        self.open_dote()

    def _installation_error(self, _error: QProcess.ProcessError) -> None:
        detail = self.setup_process.errorString() if self.setup_process else "The installer could not be started."
        self.setup_process = None
        self.status_label.setText(f"DOTE installation failed: {detail}")
        self.primary_button.setEnabled(True)

    def open_dote(self) -> None:
        self.executable_path = find_dote_executable()
        if not self.executable_path:
            self._refresh_state()
            return

        existing = find_dote_window(self.executable_path)
        if existing:
            self._embed_window(existing)
            return

        launch_result = QProcess.startDetached(
            str(self.executable_path),
            [f"--user-data-dir={self._profile_directory()}"],
            str(self.executable_path.parent),
        )
        started = launch_result[0] if isinstance(launch_result, tuple) else bool(launch_result)
        if not started:
            self.status_label.setText("DOTE transcription could not be opened.")
            self.primary_button.setEnabled(True)
            return

        self.status_label.setText("Opening DOTE transcription...")
        self.primary_button.setEnabled(False)
        self._poll_attempts = 0
        QTimer.singleShot(400, self._poll_for_window)

    def _poll_for_window(self) -> None:
        if not self.executable_path:
            return
        hwnd = find_dote_window(self.executable_path)
        if hwnd:
            self._embed_window(hwnd)
            return
        self._poll_attempts += 1
        if self._poll_attempts < 75:
            QTimer.singleShot(400, self._poll_for_window)
            return
        self.status_label.setText("DOTE is running, but its window could not be placed in this tab.")
        self.primary_button.setText("Try embedding again")
        self.primary_button.setEnabled(True)
        self.separate_button.setVisible(True)

    def _embed_window(self, hwnd: int) -> None:
        if not self.window_host.attach(hwnd):
            self.status_label.setText("DOTE opened separately because its window could not be embedded.")
            self.separate_button.setVisible(True)
            return

        self.window_embedded = True
        self.stack.setCurrentWidget(self.window_host)
        self.primary_button.setEnabled(True)

    def _profile_directory(self) -> Path:
        profile = self.quickfix_root / "QuickFixAppDependencies" / ".tools" / "dote-whisper" / "profile"
        profile.mkdir(parents=True, exist_ok=True)
        return profile

    def open_separately(self) -> None:
        if not self.executable_path:
            self._refresh_state()
            return
        QProcess.startDetached(
            str(self.executable_path),
            [f"--user-data-dir={self._profile_directory()}"],
            str(self.executable_path.parent),
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.window_host.detach()
        super().closeEvent(event)
