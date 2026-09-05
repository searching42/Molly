"""Small re-entrant file lock shared by related state stores."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import threading
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None


class _HeldLock:
    __slots__ = ("owner", "descriptor", "depth")

    def __init__(self, owner: int, descriptor: int) -> None:
        self.owner = owner
        self.descriptor = descriptor
        self.depth = 1


class StateMutationLock:
    """Coordinate read-modify-write transactions across state stores.

    The registry makes separately constructed lock objects for the same path
    re-entrant within one thread, while the file lock covers other processes.
    This is intentionally a narrow primitive for the local server's JSON
    state roots; callers still own validation and atomic file replacement.
    """

    _condition = threading.Condition(threading.RLock())
    _held: dict[str, _HeldLock] = {}

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise ValueError("state lock root cannot be a symlink")
        self.root = configured.absolute()
        self.path = (self.root / ".molly-state.lock").absolute()

    @contextmanager
    def acquire(self, *, blocking: bool = True) -> Iterator[bool]:
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.path)
        owner = threading.get_ident()
        reentrant = False
        acquired = False
        with self._condition:
            while True:
                held = self._held.get(key)
                if held is None:
                    # Reserve only this path. Never wait on flock while
                    # holding the registry mutex used by unrelated paths.
                    self._held[key] = _HeldLock(owner, -1)
                    break
                if held.owner == owner:
                    held.depth += 1
                    reentrant = True
                    break
                if not blocking:
                    break
                self._condition.wait()
        try:
            held = self._held.get(key)
            if held is not None and held.owner == owner:
                if reentrant:
                    acquired = True
                else:
                    descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
                    held.descriptor = descriptor
                    os.fchmod(descriptor, 0o600)
                    try:
                        if fcntl is not None:
                            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                            fcntl.flock(descriptor, flags)
                        acquired = True
                    except BlockingIOError:
                        if blocking:
                            raise
            yield acquired
        finally:
            with self._condition:
                held = self._held.get(key)
                if held is None or held.owner != owner:
                    pass
                elif reentrant or held.depth > 1:
                    held.depth -= 1
                else:
                    del self._held[key]
                    try:
                        if fcntl is not None and acquired:
                            fcntl.flock(held.descriptor, fcntl.LOCK_UN)
                    finally:
                        if held.descriptor >= 0:
                            os.close(held.descriptor)
                        self._condition.notify_all()


__all__ = ["StateMutationLock"]
