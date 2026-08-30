"""Small POSIX-friendly append helpers shared by local Core logs."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

from .errors import PathSecurityError

try:  # pragma: no cover - the supported repository platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject symlinked components for a configured local persistence path."""

    current = path
    while True:
        if current.is_symlink():
            raise PathSecurityError(f"symlinked persistence path is not allowed: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


@contextmanager
def locked_append(path: Path) -> Iterator[int]:
    """Open a JSONL file for serialized append and yield its file descriptor."""

    path = Path(path)
    _reject_symlink_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path.parent)
    if path.exists() and path.is_symlink():
        raise PathSecurityError(f"symlinked persistence file is not allowed: {path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PathSecurityError(f"cannot safely open persistence file: {path}") from exc

    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
        os.fsync(descriptor)
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_all(descriptor: int, payload: bytes) -> None:
    """Write all bytes to an append-open descriptor."""

    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive against broken filesystems.
            raise OSError("append write made no progress")
        view = view[written:]
