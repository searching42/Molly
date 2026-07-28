#!/usr/bin/env python3
"""Optionally scan the current tracked tree using a private literal denylist.

The denylist is supplied through ``MOLLY_PRIVATE_DENYLIST_PATH`` and must stay
outside Git. This module never persists, hashes, or prints denylist entries.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PRIVATE_DENYLIST_ENV = "MOLLY_PRIVATE_DENYLIST_PATH"


class PrivateDenylistConfigurationError(RuntimeError):
    """Raised when the optional private denylist is configured unsafely."""


@dataclass(frozen=True)
class PrivateDenylistMatch:
    entry_number: int
    relative_path: str
    location: str
    line_number: int | None = None

    def describe(self) -> str:
        suffix = f":{self.line_number}" if self.line_number is not None else ""
        return (
            f"{self.relative_path}{suffix}: private denylist entry "
            f"#{self.entry_number} matched {self.location}"
        )


def _git_output(repository_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout


def repository_root() -> Path:
    script_checkout = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=script_checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def tracked_files(repository_root: Path) -> list[Path]:
    return [
        repository_root / item.decode("utf-8")
        for item in _git_output(repository_root, "ls-files", "-z").split(b"\0")
        if item and (repository_root / item.decode("utf-8")).is_file()
    ]


def _require_untracked_denylist(path: Path, repository_root: Path) -> None:
    try:
        relative = path.relative_to(repository_root)
    except ValueError:
        return

    tracked = {
        item.decode("utf-8")
        for item in _git_output(repository_root, "ls-files", "-z").split(b"\0")
        if item
    }
    if relative.as_posix() in tracked:
        raise PrivateDenylistConfigurationError(
            "the private denylist must not be tracked by Git"
        )

    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=repository_root,
        check=False,
    )
    if ignored.returncode != 0:
        raise PrivateDenylistConfigurationError(
            "a denylist inside the checkout must be covered by Git ignore rules"
        )


def load_private_denylist(path: Path, *, repository_root: Path) -> tuple[bytes, ...]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PrivateDenylistConfigurationError(
            "the configured private denylist is not a readable regular file"
        ) from exc
    if not resolved.is_file():
        raise PrivateDenylistConfigurationError(
            "the configured private denylist is not a readable regular file"
        )
    _require_untracked_denylist(resolved, repository_root.resolve())

    try:
        raw_entries = resolved.read_bytes()
    except OSError as exc:
        raise PrivateDenylistConfigurationError(
            "the configured private denylist is not a readable regular file"
        ) from exc

    entries: list[bytes] = []
    for raw_line in raw_entries.splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith(b"#"):
            continue
        if len(entry) < 3:
            raise PrivateDenylistConfigurationError(
                "private denylist entries must contain at least three bytes"
            )
        if entry not in entries:
            entries.append(entry)
    if not entries:
        raise PrivateDenylistConfigurationError(
            "the configured private denylist contains no entries"
        )
    return tuple(entries)


def scan_files_for_private_entries(
    *,
    repository_root: Path,
    files: Sequence[Path],
    entries: Sequence[bytes],
) -> list[PrivateDenylistMatch]:
    findings: list[PrivateDenylistMatch] = []
    root = repository_root.resolve()
    lowered_entries = tuple(entry.lower() for entry in entries)
    for path in files:
        resolved = path.resolve()
        relative = resolved.relative_to(root).as_posix()
        lowered_relative = relative.encode("utf-8").lower()
        payload = resolved.read_bytes().lower()
        for entry_number, entry in enumerate(lowered_entries, start=1):
            if entry in lowered_relative:
                findings.append(
                    PrivateDenylistMatch(
                        entry_number=entry_number,
                        relative_path=relative,
                        location="tracked path",
                    )
                )
            offset = payload.find(entry)
            if offset >= 0:
                findings.append(
                    PrivateDenylistMatch(
                        entry_number=entry_number,
                        relative_path=relative,
                        location="tracked content",
                        line_number=payload.count(b"\n", 0, offset) + 1,
                    )
                )
    return findings


def run_optional_audit(
    repository_root: Path,
    denylist_path: Path,
) -> list[PrivateDenylistMatch]:
    entries = load_private_denylist(denylist_path, repository_root=repository_root)
    return scan_files_for_private_entries(
        repository_root=repository_root,
        files=tracked_files(repository_root),
        entries=entries,
    )


def main() -> int:
    configured_path = os.environ.get(PRIVATE_DENYLIST_ENV, "").strip()
    if not configured_path:
        print(
            f"{PRIVATE_DENYLIST_ENV} is not set; optional exact-value scan was not run."
        )
        return 0

    try:
        root = repository_root()
        findings = run_optional_audit(root, Path(configured_path))
    except (PrivateDenylistConfigurationError, subprocess.CalledProcessError) as exc:
        print(f"private denylist audit configuration error: {exc}", file=sys.stderr)
        return 2

    if findings:
        for finding in findings:
            print(finding.describe(), file=sys.stderr)
        return 1
    print("private denylist audit passed for the current tracked tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
