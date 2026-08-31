"""Fail-closed errors for the optional BR1 scientific plugin."""

from __future__ import annotations

from molly.core.errors import CoreContractError


class Br1Error(CoreContractError):
    """Base class for bounded BR1 contract failures."""


class Br1IntegrityError(Br1Error):
    """An input or output artifact is malformed or not byte-bound."""


class Br1BindingError(Br1Error):
    """A scientific artifact is not bound to the required current run."""


class Br1RuntimeError(Br1Error):
    """A server-owned scientific runtime failed closed."""


__all__ = ["Br1BindingError", "Br1Error", "Br1IntegrityError", "Br1RuntimeError"]
