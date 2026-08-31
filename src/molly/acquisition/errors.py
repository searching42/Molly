"""Fail-closed errors for the bounded CORE-03 acquisition subsystem."""

from __future__ import annotations

from molly.core.errors import MollyCoreError


class AcquisitionError(MollyCoreError):
    """Base class for acquisition configuration, transport, and integrity errors."""


class AcquisitionConfigurationError(AcquisitionError):
    """Server-owned acquisition configuration is invalid or incomplete."""


class AcquisitionPolicyError(AcquisitionError):
    """A URL, source, access route, or response violates the frozen policy."""


class AcquisitionTransportError(AcquisitionError):
    """A bounded network operation could not complete safely."""


class AcquisitionTimeoutError(AcquisitionTransportError):
    """The configured total operation deadline was exhausted."""


class AcquisitionCacheError(AcquisitionError):
    """A cache entry is incomplete, corrupt, or bound to a different request."""


class AcquisitionIntegrityError(AcquisitionError):
    """Acquired bytes or provenance fail an integrity check."""


class CredentialLeakError(AcquisitionIntegrityError):
    """Ephemeral access material was reflected into a durable response."""


__all__ = [
    "AcquisitionCacheError",
    "AcquisitionConfigurationError",
    "AcquisitionError",
    "AcquisitionIntegrityError",
    "AcquisitionPolicyError",
    "AcquisitionTransportError",
    "AcquisitionTimeoutError",
    "CredentialLeakError",
]
