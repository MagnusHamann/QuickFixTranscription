"""Fast local checks for the Python packages required by the app."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement


def unsatisfied_requirements(path: Path) -> tuple[str, ...]:
    """Return missing or incompatible requirements without contacting an index."""
    failures: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        requirement_text = raw_line.split("#", 1)[0].strip()
        if not requirement_text:
            continue
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as exc:
            failures.append(f"line {line_number}: invalid requirement {requirement_text!r} ({exc})")
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{requirement.name}: not installed")
            continue
        if requirement.specifier and installed not in requirement.specifier:
            failures.append(
                f"{requirement.name}: installed {installed}, required {requirement.specifier}"
            )
    return tuple(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check QuickFixTranscription Python requirements locally.")
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args(argv)
    failures = unsatisfied_requirements(args.requirements)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("QuickFixTranscription Python requirements are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
