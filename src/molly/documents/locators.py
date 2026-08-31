"""Typed, immutable source locators for canonical document objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Mapping

from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes, validate_artifact_id


class SourceLocatorKind(str, Enum):
    """Closed locator families understood by CORE-04."""

    XML_ELEMENT = "XML_ELEMENT"
    HTML_ELEMENT = "HTML_ELEMENT"
    PDF_PAGE = "PDF_PAGE"
    PDF_REGION = "PDF_REGION"
    MINERU_ELEMENT = "MINERU_ELEMENT"


_STRUCTURAL_SEGMENT = r"[A-Za-z_][A-Za-z0-9_.:-]*\[[1-9][0-9]*\]"
_STRUCTURAL_PATH_RE = re.compile(rf"^/(?:{_STRUCTURAL_SEGMENT})(?:/(?:{_STRUCTURAL_SEGMENT}))*$")
_MAX_STRUCTURAL_PATH = 2048
_MAX_BBOX_COORDINATE = 1_000_000_000.0


def _kind_value(value: str | SourceLocatorKind) -> str:
    candidate = value.value if isinstance(value, SourceLocatorKind) else value
    if not isinstance(candidate, str):
        raise CoreContractError("source locator kind must be text")
    try:
        return SourceLocatorKind(candidate.strip().upper()).value
    except ValueError as exc:
        raise CoreContractError(f"unknown source locator kind: {candidate!r}") from exc


def _bounded_page(value: int | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100_000:
        raise CoreContractError(f"{field} must be a 1-based bounded page number")
    return value


def _bounded_index(value: int | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000_000:
        raise CoreContractError(f"{field} must be a bounded non-negative index")
    return value


def _bbox(value: tuple[float, ...] | list[float] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(value)
    if len(result) != 4:
        raise CoreContractError("bbox must contain exactly four finite coordinates")
    normalized: list[float] = []
    for coordinate in result:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise CoreContractError("bbox coordinates must be finite numbers")
        number = float(coordinate)
        if not math.isfinite(number) or abs(number) > _MAX_BBOX_COORDINATE:
            raise CoreContractError("bbox coordinates are outside the bounded finite range")
        normalized.append(0.0 if number == 0.0 else round(number, 6))
    left, top, right, bottom = normalized
    if right < left or bottom < top:
        raise CoreContractError("bbox coordinates must be ordered left/top to right/bottom")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """A bounded immutable pointer into one exact source artifact.

    ``path`` is a descriptive structural element path, never a filesystem
    path and never evaluated as XPath or another query language.
    """

    source_artifact_id: str
    kind: str | SourceLocatorKind
    path: str | None = None
    page_number: int | None = None
    element_index: int | None = None
    bbox: tuple[float, ...] | list[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_artifact_id", validate_artifact_id(self.source_artifact_id))
        object.__setattr__(self, "kind", _kind_value(self.kind))
        object.__setattr__(self, "page_number", _bounded_page(self.page_number, field="page_number"))
        object.__setattr__(self, "element_index", _bounded_index(self.element_index, field="element_index"))
        object.__setattr__(self, "bbox", _bbox(self.bbox))

        kind = SourceLocatorKind(self.kind)
        if kind in {SourceLocatorKind.XML_ELEMENT, SourceLocatorKind.HTML_ELEMENT}:
            if not isinstance(self.path, str) or len(self.path) > _MAX_STRUCTURAL_PATH:
                raise CoreContractError("XML/HTML locator requires a bounded structural path")
            if not _STRUCTURAL_PATH_RE.fullmatch(self.path):
                raise CoreContractError("locator path is not a deterministic structural element path")
            if self.page_number is not None or self.element_index is not None or self.bbox is not None:
                raise CoreContractError("XML/HTML locators cannot contain page coordinates")
        elif kind is SourceLocatorKind.PDF_PAGE:
            if self.page_number is None or self.path is not None or self.element_index is not None:
                raise CoreContractError("PDF_PAGE locator requires only a page identity")
            if self.bbox is not None:
                raise CoreContractError("PDF_PAGE locator cannot contain a region bbox")
        elif kind is SourceLocatorKind.PDF_REGION:
            if self.page_number is None or self.bbox is None or self.path is not None or self.element_index is not None:
                raise CoreContractError("PDF_REGION locator requires page_number and bbox")
        elif kind is SourceLocatorKind.MINERU_ELEMENT:
            if (
                self.page_number is None
                or self.element_index is None
                or self.path is not None
            ):
                raise CoreContractError("MINERU_ELEMENT locator requires page_number and element_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_artifact_id": self.source_artifact_id,
            "kind": self.kind,
            "path": self.path,
            "page_number": self.page_number,
            "element_index": self.element_index,
            "bbox": None if self.bbox is None else list(self.bbox),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceLocator":
        if not isinstance(value, Mapping):
            raise CoreContractError("locator must be a JSON object")
        allowed = {
            "source_artifact_id",
            "kind",
            "path",
            "page_number",
            "element_index",
            "bbox",
        }
        unknown = set(value) - allowed
        if unknown:
            raise CoreContractError(f"locator has unknown fields: {sorted(unknown)!r}")
        try:
            return cls(
                source_artifact_id=str(value["source_artifact_id"]),
                kind=value["kind"],
                path=None if value.get("path") is None else str(value["path"]),
                page_number=value.get("page_number"),
                element_index=value.get("element_index"),
                bbox=None if value.get("bbox") is None else tuple(value["bbox"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("locator is malformed") from exc


__all__ = ["SourceLocator", "SourceLocatorKind"]
