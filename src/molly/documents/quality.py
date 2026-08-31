"""Deterministic parser-output quality metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes, validate_identifier


class ParserQualityStatus(str, Enum):
    """Closed parser-quality vocabulary; this is not scientific confidence."""

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    INSUFFICIENT = "INSUFFICIENT"


def _quality_status(value: str | ParserQualityStatus) -> str:
    candidate = value.value if isinstance(value, ParserQualityStatus) else value
    if not isinstance(candidate, str):
        raise CoreContractError("parser quality status must be text")
    try:
        return ParserQualityStatus(candidate.strip().upper()).value
    except ValueError as exc:
        raise CoreContractError(f"unknown parser quality status: {candidate!r}") from exc


def _bounded_nonnegative_int(value: int, *, field: str, maximum: int = 10_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CoreContractError(f"{field} must be a bounded non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ParserQuality:
    """Small deterministic description of parser output quality."""

    status: str | ParserQualityStatus
    text_char_count: int = 0
    block_count: int = 0
    table_count: int = 0
    page_count: int = 0
    pages_with_text: tuple[int, ...] = ()
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _quality_status(self.status))
        for field_name in (
            "text_char_count",
            "block_count",
            "table_count",
            "page_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_nonnegative_int(getattr(self, field_name), field=field_name),
            )
        pages = tuple(
            _bounded_nonnegative_int(page, field="pages_with_text", maximum=100_000)
            for page in self.pages_with_text
        )
        if len(pages) != len(set(pages)) or any(page < 1 for page in pages):
            raise CoreContractError("pages_with_text must be unique 1-based page numbers")
        if pages and self.page_count and max(pages) > self.page_count:
            raise CoreContractError("pages_with_text exceeds page_count")
        object.__setattr__(self, "pages_with_text", pages)
        warnings = tuple(
            validate_identifier(code, field="parser warning code") for code in self.warning_codes
        )
        if len(warnings) != len(set(warnings)):
            raise CoreContractError("parser warning codes must be unique")
        object.__setattr__(self, "warning_codes", warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text_char_count": self.text_char_count,
            "block_count": self.block_count,
            "table_count": self.table_count,
            "page_count": self.page_count,
            "pages_with_text": list(self.pages_with_text),
            "warning_codes": list(self.warning_codes),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParserQuality":
        if not isinstance(value, Mapping):
            raise CoreContractError("parser_quality must be a JSON object")
        allowed = {
            "status",
            "text_char_count",
            "block_count",
            "table_count",
            "page_count",
            "pages_with_text",
            "warning_codes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise CoreContractError(f"parser_quality has unknown fields: {sorted(unknown)!r}")
        try:
            return cls(
                status=value["status"],
                text_char_count=value.get("text_char_count", 0),
                block_count=value.get("block_count", 0),
                table_count=value.get("table_count", 0),
                page_count=value.get("page_count", 0),
                pages_with_text=tuple(value.get("pages_with_text", ())),
                warning_codes=tuple(value.get("warning_codes", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("parser_quality is malformed") from exc


__all__ = ["ParserQuality", "ParserQualityStatus"]
