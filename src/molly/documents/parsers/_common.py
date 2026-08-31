"""Shared bounded helpers for source-format parsers.

The helpers in this module return ordinary temporary parser values.  No
parser-owned object is ever stored in a :class:`CanonicalDocument`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import re
import unicodedata
from typing import Any

from molly.core.errors import CoreContractError

from ..canonical import (
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalReference,
    CanonicalSection,
    CanonicalTable,
    deterministic_object_id,
)
from ..errors import DocumentLimitError
from ..locators import SourceLocator
from ..quality import ParserQuality, ParserQualityStatus


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_MAX_NORMALIZED_TEXT_CHARS = 1_000_000


def normalize_text(value: str, *, field: str = "source text", maximum: int = _MAX_NORMALIZED_TEXT_CHARS) -> str:
    """Apply conservative, deterministic source-text normalization."""

    if not isinstance(value, str):
        raise CoreContractError(f"{field} must be text")
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = re.sub(r"\s+", " ", value).strip()
    if "\x00" in value or len(value) > maximum:
        raise DocumentLimitError(f"{field} exceeds the normalized text limit")
    return value


def ensure_source_limit(source_bytes: bytes, config: Any) -> None:
    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise CoreContractError("document source must be bytes-like")
    if len(source_bytes) > config.max_source_bytes:
        raise DocumentLimitError("document source exceeds the configured byte limit")


def ensure_collection_limit(length: int, maximum: int, *, field: str) -> None:
    if length > maximum:
        raise DocumentLimitError(f"{field} exceeds the configured limit")


def attr_by_local_name(element: Any, name: str) -> str | None:
    """Read a non-authoritative XML/HTML attribute by its local name."""

    wanted = name.casefold()
    for key, value in getattr(element, "attrib", {}).items():
        if str(key).rsplit("}", 1)[-1].casefold() == wanted:
            return str(value)
    return None


def local_name(tag: Any) -> str:
    value = str(tag)
    return value.rsplit("}", 1)[-1].casefold()


def element_text(element: Any, *, skip_tags: Iterable[str] = ()) -> str:
    skipped = {item.casefold() for item in skip_tags}
    parts: list[str] = []

    def visit(node: Any) -> None:
        if local_name(getattr(node, "tag", "")) in skipped:
            return
        if getattr(node, "text", None):
            parts.append(str(node.text))
        for child in list(node):
            visit(child)
            if getattr(child, "tail", None):
                parts.append(str(child.tail))

    visit(element)
    return normalize_text(" ".join(parts))


def nearest_ancestor(element: Any, parent_map: Mapping[int, Any], names: set[str]) -> Any | None:
    current = parent_map.get(id(element))
    wanted = {item.casefold() for item in names}
    while current is not None:
        if local_name(getattr(current, "tag", "")) in wanted:
            return current
        current = parent_map.get(id(current))
    return None


def ancestor_chain(element: Any, parent_map: Mapping[int, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    current = element
    while current is not None:
        values.append(current)
        current = parent_map.get(id(current))
    return tuple(values)


def is_inside(element: Any, parent_map: Mapping[int, Any], names: set[str]) -> bool:
    return nearest_ancestor(element, parent_map, names) is not None


def structural_paths(root: Any, *, max_depth: int) -> dict[int, str]:
    """Return descriptive sibling-indexed paths for an in-memory DOM."""

    result: dict[int, str] = {}

    def visit(node: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            raise DocumentLimitError("document nesting depth exceeds the configured limit")
        result[id(node)] = path
        sibling_indices: dict[str, int] = {}
        for child in list(node):
            tag = getattr(child, "tag", None)
            if not isinstance(tag, str):
                continue
            name = local_name(tag)
            sibling_indices[name] = sibling_indices.get(name, 0) + 1
            child_path = f"{path}/{name}[{sibling_indices[name]}]"
            visit(child, child_path, depth + 1)

    name = local_name(getattr(root, "tag", "document")) or "document"
    visit(root, f"/{name}[1]", 1)
    return result


def parse_positive_int(value: str | None, *, field: str, default: int = 1, maximum: int = 10_000) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CoreContractError(f"{field} must be an integer") from exc
    if not 1 <= parsed <= maximum:
        raise DocumentLimitError(f"{field} is outside the bounded range")
    return parsed


def parse_nonnegative_int(value: str | None, *, field: str, default: int = 0, maximum: int = 10_000) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CoreContractError(f"{field} must be an integer") from exc
    if not 0 <= parsed <= maximum:
        raise DocumentLimitError(f"{field} is outside the bounded range")
    return parsed


def citation_identifier(text: str) -> str | None:
    match = _DOI_RE.search(text)
    return match.group(0).rstrip(".,;").lower() if match else None


def _quality(
    *,
    blocks: list[CanonicalBlock],
    tables: list[CanonicalTable],
    text_char_count: int,
    page_count: int = 0,
    pages_with_text: tuple[int, ...] = (),
    status: ParserQualityStatus = ParserQualityStatus.GOOD,
    warning_codes: tuple[str, ...] = (),
) -> ParserQuality:
    return ParserQuality(
        status=status.value,
        text_char_count=text_char_count,
        block_count=len(blocks),
        table_count=len(tables),
        page_count=page_count,
        pages_with_text=pages_with_text,
        warning_codes=warning_codes,
    )


def build_document(
    *,
    config: Any,
    source_artifact_id: str,
    source_media_type: str,
    source_content_family: str,
    parser_id: str,
    parser_version: str,
    language: str | None,
    title: str | None,
    identifiers: list[str],
    sections: list[CanonicalSection],
    blocks: list[CanonicalBlock],
    tables: list[CanonicalTable],
    figures: list[CanonicalFigure],
    references: list[CanonicalReference],
    quality: ParserQuality,
) -> CanonicalDocument:
    ensure_collection_limit(len(sections), config.max_sections, field="sections")
    ensure_collection_limit(len(blocks), config.max_blocks, field="blocks")
    ensure_collection_limit(len(tables), config.max_tables, field="tables")
    ensure_collection_limit(
        sum(len(table.cells) for table in tables), config.max_table_cells, field="table cells"
    )
    ensure_collection_limit(len(figures), config.max_figures, field="figures")
    ensure_collection_limit(len(references), config.max_references, field="references")
    return CanonicalDocument(
        schema_version=config.canonical_schema_version,
        source_artifact_id=source_artifact_id,
        source_media_type=source_media_type,
        source_content_family=source_content_family,
        parser_id=parser_id,
        parser_version=parser_version,
        parser_config_digest=config.digest,
        language=language,
        title=title,
        identifiers=tuple(dict.fromkeys(identifiers)),
        sections=tuple(sections),
        blocks=tuple(blocks),
        tables=tuple(tables),
        figures=tuple(figures),
        references=tuple(references),
        parser_quality=quality,
    )


def source_locator(source_artifact_id: str, kind: str, *, path: str | None = None, page_number: int | None = None, element_index: int | None = None, bbox: tuple[float, ...] | None = None) -> SourceLocator:
    return SourceLocator(
        source_artifact_id=source_artifact_id,
        kind=kind,
        path=path,
        page_number=page_number,
        element_index=element_index,
        bbox=bbox,
    )


def stable_id(prefix: str, source_artifact_id: str, entity_type: str, ordinal: int, locator: SourceLocator, *, extra: Mapping[str, Any] | None = None) -> str:
    return deterministic_object_id(
        prefix,
        source_artifact_id=source_artifact_id,
        entity_type=entity_type,
        ordinal=ordinal,
        locator=locator,
        extra=extra,
    )


def finite_bbox(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 4 or any(not math.isfinite(value) for value in result):
        raise CoreContractError("bbox must contain four finite coordinates")
    return result


__all__ = [
    "ancestor_chain",
    "attr_by_local_name",
    "build_document",
    "citation_identifier",
    "element_text",
    "ensure_collection_limit",
    "ensure_source_limit",
    "finite_bbox",
    "is_inside",
    "local_name",
    "nearest_ancestor",
    "normalize_text",
    "parse_nonnegative_int",
    "parse_positive_int",
    "source_locator",
    "stable_id",
    "structural_paths",
]
