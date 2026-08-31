"""Errors for optional observer-only exporters."""

from __future__ import annotations

from molly.core.errors import MollyCoreError


class ObservabilityError(MollyCoreError):
    """An observer projection or export cannot be completed safely."""


class ExporterUnavailableError(ObservabilityError):
    """An optional exporter dependency or server-owned configuration is absent."""


class ExporterFailedError(ObservabilityError):
    """An observer failed; Core authoritative state remains independent."""


class ObserverIntegrityError(ObservabilityError):
    """An exporter changed or invalidated authoritative facts."""


__all__ = [
    "ExporterFailedError",
    "ExporterUnavailableError",
    "ObserverIntegrityError",
    "ObservabilityError",
]
