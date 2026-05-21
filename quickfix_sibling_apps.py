"""Discovery helpers for shared QuickFix runtime dependencies."""

from __future__ import annotations

from pathlib import Path


DEPENDENCY_FOLDER_NAME = "QuickFixAppDependencies"


def quickfix_dependency_root(project_root: Path) -> Path:
    """Return the sibling folder used for downloaded/runtime dependencies."""
    return project_root.resolve().parent / DEPENDENCY_FOLDER_NAME


def quickfix_app_roots(project_root: Path) -> list[Path]:
    """Return dependency roots to search for local runtime assets.

    Runtime assets are shared outside the app source folders:

    QuickFixApps/
      QuickFixEditing/
      QuickFixTranscription/
      QuickFixPhonemeAlignment/
      QuickFixAppDependencies/

    The current app root is kept as a fallback for older local installs.
    """
    current = project_root.resolve()
    shared = quickfix_dependency_root(current)
    roots = [shared]
    if current != shared:
        roots.append(current)
    return roots
