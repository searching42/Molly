"""Bounded errors for the host-owned CORE-07 runtime surface."""

from __future__ import annotations

from molly.core.errors import MollyCoreError


class RuntimeSurfaceError(MollyCoreError):
    """A runtime surface operation cannot be completed safely."""


class RuntimeProfileUnavailable(RuntimeSurfaceError):
    """A requested closed server-owned runtime profile is unavailable."""


class RuntimeBindingError(RuntimeSurfaceError):
    """A run is bound to a different or malformed runtime profile."""


class RuntimeStateError(RuntimeSurfaceError):
    """The configured runtime state layout is missing or unsafe."""


__all__ = [
    "RuntimeBindingError",
    "RuntimeProfileUnavailable",
    "RuntimeStateError",
    "RuntimeSurfaceError",
]
