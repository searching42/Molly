"""Bounded, non-secret errors for deterministic document normalization."""

from __future__ import annotations

from molly.core.errors import CoreContractError


class DocumentContractError(CoreContractError):
    """The canonical document contract or parser boundary was violated."""


class UnsupportedDocumentError(DocumentContractError):
    """The declared source media type or document shape is not supported."""


class MalformedDocumentError(DocumentContractError):
    """The source bytes cannot be safely parsed as the declared format."""


class DocumentLimitError(DocumentContractError):
    """A bounded parser resource limit was exceeded."""


class ParserUnavailableError(DocumentContractError):
    """An optional parser or fallback backend is not installed/configured."""


class ParserQualityError(DocumentContractError):
    """A parser returned text quality below the configured acceptance floor."""


class DocumentIntegrityError(DocumentContractError):
    """A source artifact or parser output failed an integrity binding."""


__all__ = [
    "DocumentContractError",
    "DocumentIntegrityError",
    "DocumentLimitError",
    "MalformedDocumentError",
    "ParserQualityError",
    "ParserUnavailableError",
    "UnsupportedDocumentError",
]
