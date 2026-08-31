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


class RunError(MollyCoreError):
    """Base class for bounded run-request and AgentLoop failures."""


class RunBindingError(RunError):
    """A run was resumed with a different immutable request or policy."""


class RunStateError(RunError):
    """A run's append-only facts cannot support a safe state projection."""


class BudgetError(RunError):
    """A run budget is invalid or exhausted."""


class ToolError(MollyCoreError):
    """Base class for closed tool contracts and host execution failures."""


class ToolContractError(ToolError):
    """A tool, action, schema, or execution context is outside its contract."""


class SchemaValidationError(ToolContractError):
    """Tool input or output data does not satisfy its declared JSON schema."""


class ToolAccessError(ToolContractError):
    """A host tool attempted to read outside its declared input set."""


class ToolPolicyError(ToolContractError):
    """A tool call is outside the immutable run ToolPolicy."""


class ToolExecutionError(ToolError):
    """A host-owned tool failed or returned an invalid result."""


class ApprovalError(MollyCoreError):
    """An approval is absent, stale, malformed, or bound to another call."""


class ActionError(MollyCoreError):
    """A DecisionProvider returned an unknown or malformed action."""


class ReconciliationError(MollyCoreError):
    """A factual ledger event cannot be safely projected into lineage."""


class InspectionError(MollyCoreError):
    """A read-only Core inspection cannot be produced safely."""


class InspectionIntegrityError(InspectionError):
    """Authoritative ledger, artifact, or lineage facts are contradictory."""
