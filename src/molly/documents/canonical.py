"""Source-neutral, deterministic canonical document records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    sha256_bytes,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
)

from .locators import SourceLocator
from .quality import ParserQuality, ParserQualityStatus


CANONICAL_SCHEMA_NAME = "molly.documents.canonical"
CANONICAL_SCHEMA_VERSION = "1"
DOCUMENT_MEDIA_TYPES = frozenset(
    {"application/xml", "text/xml", "text/html", "application/pdf"}
)
DOCUMENT_CONTENT_FAMILIES = frozenset({"xml", "html", "pdf"})
MAX_DOCUMENT_STRING_CHARS = 1_000_000
MAX_IDENTIFIERS = 256
MAX_SECTIONS = 10_000
MAX_BLOCKS = 50_000
MAX_TABLES = 2_000
MAX_TABLE_CELLS = 100_000
MAX_FIGURES = 10_000
MAX_REFERENCES = 20_000
MAX_CANONICAL_TEXT_BYTES = 25 * 1024 * 1024


class CanonicalBlockKind(str, Enum):
    TITLE = "TITLE"
    ABSTRACT = "ABSTRACT"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    CAPTION = "CAPTION"
    OTHER_TEXT = "OTHER_TEXT"


def _bounded_text(
    value: str,
    *,
    field: str,
    maximum: int = MAX_DOCUMENT_STRING_CHARS,
    required: bool = True,
) -> str:
    if not isinstance(value, str):
        raise CoreContractError(f"{field} must be text")
    if required and not value.strip():
        raise CoreContractError(f"{field} is required")
    if len(value) > maximum or "\x00" in value:
        raise CoreContractError(f"{field} is outside the bounded text contract")
    return value


def _optional_text(value: str | None, *, field: str, maximum: int = MAX_DOCUMENT_STRING_CHARS) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, field=field, maximum=maximum, required=False)


def _optional_identifier(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value, field=field)


def _block_kind(value: str | CanonicalBlockKind) -> str:
    candidate = value.value if isinstance(value, CanonicalBlockKind) else value
    if not isinstance(candidate, str):
        raise CoreContractError("block kind must be text")
    try:
        return CanonicalBlockKind(candidate.strip().upper()).value
    except ValueError as exc:
        raise CoreContractError(f"unknown canonical block kind: {candidate!r}") from exc


def _locator(value: SourceLocator | Mapping[str, Any], *, field: str) -> SourceLocator:
    if isinstance(value, SourceLocator):
        return value
    if isinstance(value, Mapping):
        try:
            return SourceLocator.from_dict(value)
        except Exception as exc:
            raise CoreContractError(f"{field} is malformed") from exc
    raise CoreContractError(f"{field} must be a SourceLocator")


def deterministic_object_id(
    prefix: str,
    *,
    source_artifact_id: str,
    entity_type: str,
    ordinal: int,
    locator: SourceLocator,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Derive a stable object ID from source structure, never run state."""

    validate_identifier(prefix, field="object ID prefix")
    validate_artifact_id(source_artifact_id)
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise CoreContractError("entity_type is required")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise CoreContractError("object ordinal must be a non-negative integer")
    if locator.source_artifact_id != source_artifact_id:
        raise CoreContractError("object locator source does not match object source artifact")
    body: dict[str, Any] = {
        "source_artifact_id": source_artifact_id,
        "entity_type": entity_type,
        "ordinal": ordinal,
        "locator": locator.to_dict(),
    }
    if extra is not None:
        body["extra"] = dict(extra)
    return f"{prefix}_{sha256_bytes(canonical_json_bytes(body))}"


@dataclass(frozen=True, slots=True)
class CanonicalSection:
    section_id: str
    title: str
    level: int
    parent_section_id: str | None
    locator: SourceLocator

    def __post_init__(self) -> None:
        validate_identifier(self.section_id, field="section_id")
        object.__setattr__(self, "title", _bounded_text(self.title, field="section title", required=False))
        if isinstance(self.level, bool) or not isinstance(self.level, int) or not 1 <= self.level <= 100:
            raise CoreContractError("section level must be between 1 and 100")
        object.__setattr__(
            self,
            "parent_section_id",
            _optional_identifier(self.parent_section_id, field="parent_section_id"),
        )
        object.__setattr__(self, "locator", _locator(self.locator, field="section locator"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "parent_section_id": self.parent_section_id,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalSection":
        _assert_fields(value, {"section_id", "title", "level", "parent_section_id", "locator"}, "section")
        try:
            return cls(
                section_id=str(value["section_id"]),
                title=str(value.get("title", "")),
                level=value["level"],
                parent_section_id=(
                    None if value.get("parent_section_id") is None else str(value["parent_section_id"])
                ),
                locator=SourceLocator.from_dict(value["locator"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("section is malformed") from exc


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    block_id: str
    kind: str | CanonicalBlockKind
    text: str
    section_id: str | None
    locator: SourceLocator

    def __post_init__(self) -> None:
        validate_identifier(self.block_id, field="block_id")
        object.__setattr__(self, "kind", _block_kind(self.kind))
        object.__setattr__(self, "text", _bounded_text(self.text, field="block text", required=False))
        object.__setattr__(self, "section_id", _optional_identifier(self.section_id, field="block section_id"))
        object.__setattr__(self, "locator", _locator(self.locator, field="block locator"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "text": self.text,
            "section_id": self.section_id,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalBlock":
        _assert_fields(value, {"block_id", "kind", "text", "section_id", "locator"}, "block")
        try:
            return cls(
                block_id=str(value["block_id"]),
                kind=value["kind"],
                text=str(value.get("text", "")),
                section_id=None if value.get("section_id") is None else str(value["section_id"]),
                locator=SourceLocator.from_dict(value["locator"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("block is malformed") from exc


@dataclass(frozen=True, slots=True)
class CanonicalCell:
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    is_header: bool
    text: str
    locator: SourceLocator

    def __post_init__(self) -> None:
        for name in ("row_index", "column_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100_000:
                raise CoreContractError(f"{name} must be a bounded non-negative integer")
        for name in ("row_span", "column_span"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
                raise CoreContractError(f"{name} must be a bounded positive integer")
        if not isinstance(self.is_header, bool):
            raise CoreContractError("is_header must be boolean")
        object.__setattr__(self, "text", _bounded_text(self.text, field="table cell text", required=False))
        object.__setattr__(self, "locator", _locator(self.locator, field="cell locator"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "row_span": self.row_span,
            "column_span": self.column_span,
            "is_header": self.is_header,
            "text": self.text,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalCell":
        _assert_fields(
            value,
            {"row_index", "column_index", "row_span", "column_span", "is_header", "text", "locator"},
            "cell",
        )
        try:
            return cls(
                row_index=value["row_index"],
                column_index=value["column_index"],
                row_span=value.get("row_span", 1),
                column_span=value.get("column_span", 1),
                is_header=value["is_header"],
                text=str(value.get("text", "")),
                locator=SourceLocator.from_dict(value["locator"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("cell is malformed") from exc


@dataclass(frozen=True, slots=True)
class CanonicalTable:
    table_id: str
    caption: str | None
    section_id: str | None
    locator: SourceLocator
    cells: tuple[CanonicalCell, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.table_id, field="table_id")
        object.__setattr__(self, "caption", _optional_text(self.caption, field="table caption"))
        object.__setattr__(self, "section_id", _optional_identifier(self.section_id, field="table section_id"))
        object.__setattr__(self, "locator", _locator(self.locator, field="table locator"))
        cells = tuple(
            item if isinstance(item, CanonicalCell) else CanonicalCell.from_dict(item)
            for item in self.cells
        )
        positions = [(cell.row_index, cell.column_index) for cell in cells]
        if len(cells) > MAX_TABLE_CELLS or len(positions) != len(set(positions)):
            raise CoreContractError("table cells must be bounded and have unique coordinates")
        object.__setattr__(self, "cells", cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "caption": self.caption,
            "section_id": self.section_id,
            "locator": self.locator.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalTable":
        _assert_fields(value, {"table_id", "caption", "section_id", "locator", "cells"}, "table")
        try:
            return cls(
                table_id=str(value["table_id"]),
                caption=None if value.get("caption") is None else str(value["caption"]),
                section_id=None if value.get("section_id") is None else str(value["section_id"]),
                locator=SourceLocator.from_dict(value["locator"]),
                cells=tuple(CanonicalCell.from_dict(item) for item in value.get("cells", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("table is malformed") from exc


@dataclass(frozen=True, slots=True)
class CanonicalFigure:
    figure_id: str
    label: str | None
    caption: str | None
    locator: SourceLocator

    def __post_init__(self) -> None:
        validate_identifier(self.figure_id, field="figure_id")
        object.__setattr__(self, "label", _optional_text(self.label, field="figure label", maximum=512))
        object.__setattr__(self, "caption", _optional_text(self.caption, field="figure caption"))
        object.__setattr__(self, "locator", _locator(self.locator, field="figure locator"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "label": self.label,
            "caption": self.caption,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalFigure":
        _assert_fields(value, {"figure_id", "label", "caption", "locator"}, "figure")
        try:
            return cls(
                figure_id=str(value["figure_id"]),
                label=None if value.get("label") is None else str(value["label"]),
                caption=None if value.get("caption") is None else str(value["caption"]),
                locator=SourceLocator.from_dict(value["locator"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("figure is malformed") from exc


@dataclass(frozen=True, slots=True)
class CanonicalReference:
    reference_id: str
    citation_text: str
    identifier: str | None
    locator: SourceLocator

    def __post_init__(self) -> None:
        validate_identifier(self.reference_id, field="reference_id")
        object.__setattr__(self, "citation_text", _bounded_text(self.citation_text, field="citation text"))
        object.__setattr__(self, "identifier", _optional_text(self.identifier, field="reference identifier", maximum=512))
        object.__setattr__(self, "locator", _locator(self.locator, field="reference locator"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "citation_text": self.citation_text,
            "identifier": self.identifier,
            "locator": self.locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalReference":
        _assert_fields(value, {"reference_id", "citation_text", "identifier", "locator"}, "reference")
        try:
            return cls(
                reference_id=str(value["reference_id"]),
                citation_text=str(value["citation_text"]),
                identifier=None if value.get("identifier") is None else str(value["identifier"]),
                locator=SourceLocator.from_dict(value["locator"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("reference is malformed") from exc


def _assert_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    if not isinstance(value, Mapping):
        raise CoreContractError(f"{field} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise CoreContractError(f"{field} has unknown fields: {sorted(unknown)!r}")


def _unique_ids(values: Sequence[str], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise CoreContractError(f"{field} must not contain duplicate IDs")


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """Immutable source structure independent of acquisition occurrence."""

    schema_version: str
    source_artifact_id: str
    source_media_type: str
    source_content_family: str
    parser_id: str
    parser_version: str
    parser_config_digest: str
    language: str | None = None
    title: str | None = None
    identifiers: tuple[str, ...] = ()
    sections: tuple[CanonicalSection, ...] = ()
    blocks: tuple[CanonicalBlock, ...] = ()
    tables: tuple[CanonicalTable, ...] = ()
    figures: tuple[CanonicalFigure, ...] = ()
    references: tuple[CanonicalReference, ...] = ()
    parser_quality: ParserQuality = ParserQuality(ParserQualityStatus.GOOD.value)

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_SCHEMA_VERSION:
            raise CoreContractError(f"unsupported canonical document schema version: {self.schema_version!r}")
        object.__setattr__(self, "source_artifact_id", validate_artifact_id(self.source_artifact_id))
        media_type = _bounded_text(self.source_media_type, field="source_media_type", maximum=128)
        media_type = media_type.casefold().split(";", 1)[0].strip()
        if media_type not in DOCUMENT_MEDIA_TYPES:
            raise CoreContractError(f"unsupported source media type: {media_type!r}")
        object.__setattr__(self, "source_media_type", media_type)
        family = _bounded_text(self.source_content_family, field="source_content_family", maximum=32).casefold()
        expected_family = {
            "application/xml": "xml",
            "text/xml": "xml",
            "text/html": "html",
            "application/pdf": "pdf",
        }[media_type]
        if family not in DOCUMENT_CONTENT_FAMILIES or family != expected_family:
            raise CoreContractError("source media type and content family disagree")
        object.__setattr__(self, "source_content_family", family)
        object.__setattr__(self, "parser_id", validate_identifier(self.parser_id, field="parser_id"))
        object.__setattr__(self, "parser_version", validate_identifier(self.parser_version, field="parser_version"))
        object.__setattr__(
            self,
            "parser_config_digest",
            validate_digest_reference(self.parser_config_digest, field="parser_config_digest"),
        )
        object.__setattr__(self, "language", _optional_text(self.language, field="language", maximum=64))
        object.__setattr__(self, "title", _optional_text(self.title, field="title"))
        identifiers = tuple(_bounded_text(item, field="document identifier", maximum=512) for item in self.identifiers)
        if len(identifiers) > MAX_IDENTIFIERS:
            raise CoreContractError("document identifiers exceed the bounded limit")
        _unique_ids(identifiers, field="document identifiers")
        object.__setattr__(self, "identifiers", identifiers)

        sections = tuple(
            item if isinstance(item, CanonicalSection) else CanonicalSection.from_dict(item)
            for item in self.sections
        )
        blocks = tuple(
            item if isinstance(item, CanonicalBlock) else CanonicalBlock.from_dict(item)
            for item in self.blocks
        )
        tables = tuple(
            item if isinstance(item, CanonicalTable) else CanonicalTable.from_dict(item)
            for item in self.tables
        )
        figures = tuple(
            item if isinstance(item, CanonicalFigure) else CanonicalFigure.from_dict(item)
            for item in self.figures
        )
        references = tuple(
            item if isinstance(item, CanonicalReference) else CanonicalReference.from_dict(item)
            for item in self.references
        )
        if len(sections) > MAX_SECTIONS or len(blocks) > MAX_BLOCKS or len(tables) > MAX_TABLES:
            raise CoreContractError("canonical document collection limit exceeded")
        if sum(len(table.cells) for table in tables) > MAX_TABLE_CELLS:
            raise CoreContractError("canonical document table-cell limit exceeded")
        if len(figures) > MAX_FIGURES or len(references) > MAX_REFERENCES:
            raise CoreContractError("canonical document collection limit exceeded")
        for values, field in (
            ([item.section_id for item in sections], "section"),
            ([item.block_id for item in blocks], "block"),
            ([item.table_id for item in tables], "table"),
            ([item.figure_id for item in figures], "figure"),
            ([item.reference_id for item in references], "reference"),
        ):
            _unique_ids(values, field=f"{field} IDs")
        all_ids = [
            *[item.section_id for item in sections],
            *[item.block_id for item in blocks],
            *[item.table_id for item in tables],
            *[item.figure_id for item in figures],
            *[item.reference_id for item in references],
        ]
        _unique_ids(all_ids, field="canonical object IDs")
        section_ids = {item.section_id for item in sections}
        section_by_id = {item.section_id: item for item in sections}
        for section in sections:
            if section.parent_section_id is not None and section.parent_section_id not in section_ids:
                raise CoreContractError("section references an unknown parent")
            if section.locator.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("section locator source does not match document source")
        for section in sections:
            seen: set[str] = set()
            current: CanonicalSection | None = section
            while current is not None and current.parent_section_id is not None:
                if current.section_id in seen:
                    raise CoreContractError("section hierarchy contains a cycle")
                seen.add(current.section_id)
                current = section_by_id[current.parent_section_id]
        for block in blocks:
            if block.section_id is not None and block.section_id not in section_ids:
                raise CoreContractError("block references an unknown section")
            if block.locator.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("block locator source does not match document source")
        for table in tables:
            if table.section_id is not None and table.section_id not in section_ids:
                raise CoreContractError("table references an unknown section")
            if table.locator.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("table locator source does not match document source")
            if any(cell.locator.source_artifact_id != self.source_artifact_id for cell in table.cells):
                raise CoreContractError("cell locator source does not match document source")
        for figure in figures:
            if figure.locator.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("figure locator source does not match document source")
        for reference in references:
            if reference.locator.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("reference locator source does not match document source")
        if isinstance(self.parser_quality, Mapping):
            object.__setattr__(self, "parser_quality", ParserQuality.from_dict(self.parser_quality))
        elif not isinstance(self.parser_quality, ParserQuality):
            raise CoreContractError("parser_quality must be a ParserQuality")
        if self.parser_quality.block_count != len(blocks) or self.parser_quality.table_count != len(tables):
            raise CoreContractError("parser quality counts do not match canonical objects")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "figures", figures)
        object.__setattr__(self, "references", references)
        if len(self.canonical_bytes()) > MAX_CANONICAL_TEXT_BYTES:
            raise CoreContractError("canonical document exceeds the bounded serialized size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_id": self.source_artifact_id,
            "source_media_type": self.source_media_type,
            "source_content_family": self.source_content_family,
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "parser_config_digest": self.parser_config_digest,
            "language": self.language,
            "title": self.title,
            "identifiers": list(self.identifiers),
            "sections": [item.to_dict() for item in self.sections],
            "blocks": [item.to_dict() for item in self.blocks],
            "tables": [item.to_dict() for item in self.tables],
            "figures": [item.to_dict() for item in self.figures],
            "references": [item.to_dict() for item in self.references],
            "parser_quality": self.parser_quality.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def canonical_document_sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(self.canonical_document_sha256)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalDocument":
        _assert_fields(
            value,
            {
                "schema_version",
                "source_artifact_id",
                "source_media_type",
                "source_content_family",
                "parser_id",
                "parser_version",
                "parser_config_digest",
                "language",
                "title",
                "identifiers",
                "sections",
                "blocks",
                "tables",
                "figures",
                "references",
                "parser_quality",
            },
            "canonical document",
        )
        try:
            return cls(
                schema_version=str(value["schema_version"]),
                source_artifact_id=str(value["source_artifact_id"]),
                source_media_type=str(value["source_media_type"]),
                source_content_family=str(value["source_content_family"]),
                parser_id=str(value["parser_id"]),
                parser_version=str(value["parser_version"]),
                parser_config_digest=str(value["parser_config_digest"]),
                language=None if value.get("language") is None else str(value["language"]),
                title=None if value.get("title") is None else str(value["title"]),
                identifiers=tuple(str(item) for item in value.get("identifiers", ())),
                sections=tuple(CanonicalSection.from_dict(item) for item in value.get("sections", ())),
                blocks=tuple(CanonicalBlock.from_dict(item) for item in value.get("blocks", ())),
                tables=tuple(CanonicalTable.from_dict(item) for item in value.get("tables", ())),
                figures=tuple(CanonicalFigure.from_dict(item) for item in value.get("figures", ())),
                references=tuple(CanonicalReference.from_dict(item) for item in value.get("references", ())),
                parser_quality=ParserQuality.from_dict(value["parser_quality"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("canonical document is malformed") from exc


__all__ = [
    "CANONICAL_SCHEMA_NAME",
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalBlock",
    "CanonicalBlockKind",
    "CanonicalCell",
    "CanonicalDocument",
    "CanonicalFigure",
    "CanonicalReference",
    "CanonicalSection",
    "CanonicalTable",
    "deterministic_object_id",
]
