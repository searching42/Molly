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
    def acquire(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.path)
        owner = threading.get_ident()
        descriptor: int | None = None
        reentrant = False
        with self._condition:
            while True:
                held = self._held.get(key)
                if held is None:
                    descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
                    try:
                        os.fchmod(descriptor, 0o600)
                        if fcntl is not None:
                            fcntl.flock(descriptor, fcntl.LOCK_EX)
                    except BaseException:
                        os.close(descriptor)
                        raise
                    self._held[key] = _HeldLock(owner, descriptor)
                    break
                if held.owner == owner:
                    held.depth += 1
                    reentrant = True
                    break
                self._condition.wait()
        try:
            yield
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
                        if fcntl is not None:
                            fcntl.flock(held.descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(held.descriptor)
                        self._condition.notify_all()


__all__ = ["StateMutationLock"]
