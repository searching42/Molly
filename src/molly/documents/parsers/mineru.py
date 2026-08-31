"""Optional MinerU PDF-fallback seam with immediate output normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol, runtime_checkable

from ..canonical import (
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalCell,
    CanonicalFigure,
    CanonicalTable,
)
from ..errors import MalformedDocumentError, ParserUnavailableError
from ..quality import ParserQualityStatus
from ._common import (
    build_document,
    ensure_source_limit,
    normalize_text,
    parse_nonnegative_int,
    parse_positive_int,
    source_locator,
    stable_id,
)


@runtime_checkable
class MinerUBackend(Protocol):
    """Host-owned backend protocol; no worker/path authority is exposed."""

    def parse(self, source_bytes: bytes) -> Sequence[Mapping[str, Any]]:
        ...


def _bounded_bbox(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise MalformedDocumentError("MinerU bbox is malformed")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise MalformedDocumentError("MinerU bbox is not finite")
        values.append(round(float(item), 6))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class MinerUCell:
    row_index: int
    column_index: int
    text: str
    is_header: bool = False
    row_span: int = 1
    column_span: int = 1

    @classmethod
    def from_value(cls, value: Mapping[str, Any], config: Any) -> "MinerUCell":
        if not isinstance(value, Mapping):
            raise MalformedDocumentError("MinerU table cell is malformed")
        allowed = {"row_index", "column_index", "text", "is_header", "row_span", "column_span"}
        if set(value) - allowed:
            raise MalformedDocumentError("MinerU table cell has unsupported fields")
        is_header = value.get("is_header", False)
        if not isinstance(is_header, bool):
            raise MalformedDocumentError("MinerU table header flag is malformed")
        try:
            text = normalize_text(str(value.get("text", "")), field="MinerU table cell")
            cell = cls(
                row_index=parse_nonnegative_int(str(value.get("row_index", 0)), field="MinerU row index", maximum=100_000),
                column_index=parse_nonnegative_int(str(value.get("column_index", 0)), field="MinerU column index", maximum=100_000),
                text=text,
                is_header=is_header,
                row_span=parse_positive_int(str(value.get("row_span", 1)), field="MinerU row span", maximum=10_000),
                column_span=parse_positive_int(str(value.get("column_span", 1)), field="MinerU column span", maximum=10_000),
            )
        except (TypeError, ValueError) as exc:
            raise MalformedDocumentError("MinerU table cell is malformed") from exc
        return cell


@dataclass(frozen=True, slots=True)
class MinerUElement:
    page_number: int
    element_index: int
    element_type: str
    text: str
    bbox: tuple[float, ...] | None = None
    cells: tuple[MinerUCell, ...] = ()

    @classmethod
    def from_value(cls, value: Mapping[str, Any], config: Any) -> "MinerUElement":
        if not isinstance(value, Mapping):
            raise MalformedDocumentError("MinerU element is malformed")
        allowed = {"page_number", "element_index", "type", "kind", "text", "bbox", "cells"}
        if set(value) - allowed:
            raise MalformedDocumentError("MinerU element has unsupported fields")
        if "page_number" not in value or "element_index" not in value:
            raise MalformedDocumentError("MinerU element requires page_number and element_index")
        if isinstance(value["page_number"], bool) or not isinstance(value["page_number"], int):
            raise MalformedDocumentError("MinerU page number is malformed")
        if isinstance(value["element_index"], bool) or not isinstance(value["element_index"], int):
            raise MalformedDocumentError("MinerU element index is malformed")
        raw_type = value.get("type", value.get("kind", "text"))
        if not isinstance(raw_type, str) or not raw_type.strip():
            raise MalformedDocumentError("MinerU element type is malformed")
        element_type = raw_type.strip().casefold()
        if len(element_type) > 64 or any(char not in "abcdefghijklmnopqrstuvwxyz_-" for char in element_type):
            raise MalformedDocumentError("MinerU element type is malformed")
        try:
            page_number = parse_positive_int(
                str(value.get("page_number", "")), field="MinerU page number", maximum=config.max_page_count
            )
            element_index = parse_nonnegative_int(
                str(value.get("element_index", "")), field="MinerU element index", maximum=10_000_000
            )
            text = normalize_text(str(value.get("text", "")), field="MinerU element text")
            cells = tuple(
                MinerUCell.from_value(item, config)
                for item in value.get("cells", ())
            )
        except (TypeError, ValueError) as exc:
            raise MalformedDocumentError("MinerU element is malformed") from exc
        if len(cells) > config.max_table_cells:
            raise MalformedDocumentError("MinerU table cell count exceeds the configured limit")
        return cls(
            page_number=page_number,
            element_index=element_index,
            element_type=element_type,
            text=text,
            bbox=_bounded_bbox(value.get("bbox")),
            cells=cells,
        )


class MinerUFallbackParser:
    parser_id = "mineru"
    version = "1"

    def __init__(self, backend: MinerUBackend | None = None) -> None:
        self._backend = backend

    def parse(
        self,
        *,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
        config: Any,
    ):
        ensure_source_limit(source_bytes, config)
        if self._backend is None or not isinstance(self._backend, MinerUBackend):
            raise ParserUnavailableError("MinerU fallback is not configured")
        try:
            raw_elements = self._backend.parse(bytes(source_bytes))
        except ParserUnavailableError:
            raise
        except Exception as exc:
            raise MalformedDocumentError("MinerU fallback failed") from exc
        if not isinstance(raw_elements, Sequence) or isinstance(raw_elements, (str, bytes, bytearray)):
            raise MalformedDocumentError("MinerU fallback returned an invalid element sequence")
        if len(raw_elements) > config.max_node_count:
            raise MalformedDocumentError("MinerU element count exceeds the configured limit")
        elements = tuple(MinerUElement.from_value(value, config) for value in raw_elements)
        blocks: list[CanonicalBlock] = []
        tables: list[CanonicalTable] = []
        figures: list[CanonicalFigure] = []
        title: str | None = None
        for ordinal, element in enumerate(elements):
            locator = source_locator(
                source_artifact_id,
                "MINERU_ELEMENT",
                page_number=element.page_number,
                element_index=element.element_index,
                bbox=element.bbox,
            )
            if element.element_type in {"table"}:
                cells: list[CanonicalCell] = []
                for cell_index, cell in enumerate(element.cells):
                    cell_element_index = element.element_index * 1000 + cell_index
                    if cell_element_index > 10_000_000:
                        raise MalformedDocumentError("MinerU cell locator index exceeds the limit")
                    cell_locator = source_locator(
                        source_artifact_id,
                        "MINERU_ELEMENT",
                        page_number=element.page_number,
                        element_index=cell_element_index,
                        bbox=element.bbox,
                    )
                    cells.append(
                        CanonicalCell(
                            row_index=cell.row_index,
                            column_index=cell.column_index,
                            row_span=cell.row_span,
                            column_span=cell.column_span,
                            is_header=cell.is_header,
                            text=cell.text,
                            locator=cell_locator,
                        )
                    )
                tables.append(
                    CanonicalTable(
                        table_id=stable_id("tbl", source_artifact_id, "mineru_table", ordinal, locator),
                        caption=element.text or None,
                        section_id=None,
                        locator=locator,
                        cells=tuple(cells),
                    )
                )
                continue
            if element.element_type in {"figure", "image"}:
                figures.append(
                    CanonicalFigure(
                        figure_id=stable_id("fig", source_artifact_id, "mineru_figure", ordinal, locator),
                        label=None,
                        caption=element.text or None,
                        locator=locator,
                    )
                )
                continue
            kind = {
                "title": CanonicalBlockKind.TITLE,
                "heading": CanonicalBlockKind.HEADING,
                "paragraph": CanonicalBlockKind.PARAGRAPH,
                "list_item": CanonicalBlockKind.LIST_ITEM,
                "caption": CanonicalBlockKind.CAPTION,
                "reference": CanonicalBlockKind.OTHER_TEXT,
            }.get(element.element_type, CanonicalBlockKind.OTHER_TEXT)
            if element.text:
                if title is None and kind is CanonicalBlockKind.TITLE:
                    title = element.text
                blocks.append(
                    CanonicalBlock(
                        block_id=stable_id("blk", source_artifact_id, "mineru_block", ordinal, locator),
                        kind=kind.value,
                        text=element.text,
                        section_id=None,
                        locator=locator,
                    )
                )
        text_char_count = sum(len(block.text) for block in blocks)
        text_char_count += sum(len(cell.text) for table in tables for cell in table.cells)
        quality = config.quality(
            text_char_count=text_char_count,
            block_count=len(blocks),
            table_count=len(tables),
            page_count=max((element.page_number for element in elements), default=0),
            pages_with_text=tuple(
                sorted({element.page_number for element in elements if element.text})
            ),
            status=ParserQualityStatus.GOOD,
            warning_codes=("MINERU_FALLBACK",),
        )
        return build_document(
            config=config,
            source_artifact_id=source_artifact_id,
            source_media_type=source_media_type,
            source_content_family="pdf",
            parser_id=self.parser_id,
            parser_version=self.version,
            language=None,
            title=title,
            identifiers=[],
            sections=[],
            blocks=blocks,
            tables=tables,
            figures=figures,
            references=[],
            quality=quality,
        )


__all__ = ["MinerUBackend", "MinerUCell", "MinerUElement", "MinerUFallbackParser"]
