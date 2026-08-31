"""Fail-closed errors for the small durable compute backend."""

from __future__ import annotations

from molly.core.errors import CoreContractError


class ComputeError(CoreContractError):
    """Base class for durable compute contract failures."""


class ComputeConflictError(ComputeError):
    """An idempotency key or job identity was reused inconsistently."""


class ComputeIntegrityError(ComputeError):
    """A durable job or output manifest failed verification."""


class ComputeExecutionError(ComputeError):
    """A server-owned compute runner failed closed."""


__all__ = ["ComputeConflictError", "ComputeError", "ComputeExecutionError", "ComputeIntegrityError"]
