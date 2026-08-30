"""Errors raised by the small, fail-closed Molly Core contracts."""

from __future__ import annotations


class MollyCoreError(ValueError):
    """Base class for contract and persistence failures."""


class CoreContractError(MollyCoreError):
    """A caller supplied a value outside a closed Core contract."""


class PathSecurityError(CoreContractError):
    """A configured or derived path could escape its intended root."""


class ArtifactError(MollyCoreError):
    """Base class for artifact publication and inspection failures."""


class ArtifactNotFoundError(ArtifactError):
    """The requested immutable artifact or metadata does not exist."""


class ArtifactIntegrityError(ArtifactError):
    """Stored artifact bytes or metadata fail integrity verification."""


class ArtifactConflictError(ArtifactError):
    """A no-replace publication encountered incompatible existing state."""


class LedgerError(MollyCoreError):
    """Base class for append-only ledger failures."""


class LedgerCorruptionError(LedgerError):
    """The ledger contains malformed, truncated, or tampered data."""


class LineageError(MollyCoreError):
    """A lineage relation is invalid or cannot be safely appended."""


class ValidationContractError(MollyCoreError):
    """A validation result uses an unknown scope, status, or identity."""


class ReviewError(MollyCoreError):
    """A review record is malformed or cannot be persisted safely."""


class ReviewBindingError(ReviewError):
    """A review does not bind to the exact artifact digest being inspected."""
