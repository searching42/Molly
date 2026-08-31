"""Errors for deterministic evidence and reviewed-dataset contracts."""

from molly.core.errors import CoreContractError


class EvidenceContractError(CoreContractError):
    """A bounded evidence value or schema is outside its contract."""


class EvidenceIntegrityError(EvidenceContractError):
    """A referenced evidence artifact, digest, or locator is not exact."""


__all__ = ["EvidenceContractError", "EvidenceIntegrityError"]
