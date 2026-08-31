"""Deterministic, non-browser HTML parser for canonical documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

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
from ..errors import DocumentLimitError, MalformedDocumentError
from ..locators import SourceLocator
from ..quality import ParserQualityStatus
from ._common import (
    attr_by_local_name,
    build_document,
    citation_identifier,
    element_text,
    ensure_source_limit,
    is_inside,
    local_name,
    nearest_ancestor,
    normalize_text,
    parse_positive_int,
    source_locator,
    stable_id,
    structural_paths,
)


_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_SKIP_TAGS = {"script", "style", "noscript"}
_SECTION_TAGS = {"section", "article"}
_PARAGRAPH_TAGS = {"p", "paragraph"}
_REFERENCE_HINTS = {"reference", "references", "bibliography", "refs"}


@dataclass
class _HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_HtmlNode"] = field(default_factory=list)
    text: str = ""
    tail: str = ""

    def __iter__(self):
        return iter(self.children)

    def iter(self):
        yield self
        for child in self.children:
            yield from child.iter()


class _TreeParser(HTMLParser):
    def __init__(self, config: Any) -> None:
        super().__init__(convert_charrefs=True)
        self.config = config
        self.root = _HtmlNode("document")
        self.stack: list[_HtmlNode] = [self.root]
        self.node_count = 0

    def _add(self, tag: str, attrs: list[tuple[str, str | None]]) -> _HtmlNode:
        self.node_count += 1
        if self.node_count > self.config.max_node_count:
            raise DocumentLimitError("HTML node count exceeds the configured limit")
        parent = self.stack[-1]
        node = _HtmlNode(
            tag=tag.casefold(),
            attrs={str(key).casefold(): "" if value is None else str(value) for key, value in attrs},
        )
        parent.children.append(node)
        return node

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = self._add(tag, attrs)
        if node.tag not in _VOID_TAGS:
            self.stack.append(node)
            if len(self.stack) - 1 > self.config.max_nesting_depth:
                raise DocumentLimitError("HTML nesting depth exceeds the configured limit")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._add(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        wanted = tag.casefold()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == wanted:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].text += data

    def handle_entityref(self, name: str) -> None:
        self.stack[-1].text += f"&{name};"

    def handle_charref(self, name: str) -> None:
        self.stack[-1].text += f"&#{name};"

    def error(self, message: str) -> None:
        raise MalformedDocumentError("HTML parser rejected the source")


def _tree_details(source_bytes: bytes, config: Any) -> tuple[_HtmlNode, dict[int, _HtmlNode], dict[int, str]]:
    ensure_source_limit(source_bytes, config)
    try:
        text = bytes(source_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedDocumentError("HTML source is not valid UTF-8") from exc
    parser = _TreeParser(config)
    try:
        parser.feed(text)
        parser.close()
    except DocumentLimitError:
        raise
    except Exception as exc:
        raise MalformedDocumentError("HTML source is malformed") from exc
    top_level = [item for item in parser.root.children if item.tag not in _SKIP_TAGS]
    root = top_level[0] if len(top_level) == 1 else parser.root
    parent_map: dict[int, _HtmlNode] = {}
    for node in parser.root.iter():
        for child in node.children:
            parent_map[id(child)] = node
    paths = structural_paths(root, max_depth=config.max_nesting_depth)
    root_text = normalize_text(root.text + " " + " ".join(item.text for item in root.iter()), field="HTML normalized text")
    if len(root_text.encode("utf-8")) > config.max_normalized_text_bytes:
        raise DocumentLimitError("HTML normalized text exceeds the configured limit")
    return root, parent_map, paths


def _safe_text(element: _HtmlNode | None, config: Any, *, field: str, required: bool = False) -> str:
    if element is None:
        return ""
    value = element_text(element, skip_tags=_SKIP_TAGS)
    if not value and required:
        raise MalformedDocumentError(f"{field} is empty")
    if len(value) > config.max_text_chars:
        raise DocumentLimitError(f"{field} exceeds the configured text limit")
    return value


def _elements(root: _HtmlNode) -> list[_HtmlNode]:
    return [item for item in root.iter() if item.tag not in _SKIP_TAGS]


def _class_or_id_hint(element: _HtmlNode) -> str:
    return " ".join(
        value.casefold()
        for key in ("id", "class", "role")
        for value in (element.attrs.get(key, ""),)
    )


def _make_sections(
    root: _HtmlNode,
    parent_map: Mapping[int, _HtmlNode],
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> tuple[list[CanonicalSection], dict[int, str]]:
    explicit = [item for item in _elements(root) if item.tag in _SECTION_TAGS]
    heading_elements = [item for item in _elements(root) if item.tag in {f"h{index}" for index in range(1, 7)}]
    candidates = explicit or heading_elements
    sections: list[CanonicalSection] = []
    section_ids: dict[int, str] = {}
    for ordinal, element in enumerate(candidates):
        locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(element)])
        parent_id: str | None = None
        level = 1
        if explicit:
            parent = nearest_ancestor(element, parent_map, _SECTION_TAGS)
            if parent is not None:
                parent_id = section_ids.get(id(parent))
                if parent_id is None:
                    raise MalformedDocumentError("HTML section hierarchy cannot be normalized")
                level = section_ids and next(
                    section.level for section in sections if section.section_id == parent_id
                ) + 1
        else:
            heading_level = int(element.tag[1])
            level = heading_level
            for previous in reversed(sections):
                if previous.level < level:
                    parent_id = previous.section_id
                    break
        section_id = stable_id("sec", source_artifact_id, "section", ordinal, locator)
        section_ids[id(element)] = section_id
        heading = next(
            (
                item
                for item in element.children
                if item.tag == "title" or item.tag in {f"h{index}" for index in range(1, 7)}
            ),
            None,
        )
        sections.append(
            CanonicalSection(
                section_id=section_id,
                title=_safe_text(heading, config, field="section title") if heading is not None else "",
                level=level,
                parent_section_id=parent_id,
                locator=locator,
            )
        )
        if len(sections) > config.max_sections:
            raise DocumentLimitError("section count exceeds the configured limit")
    return sections, section_ids


def _nearest_section_id(element: _HtmlNode, parent_map: Mapping[int, _HtmlNode], section_ids: Mapping[int, str]) -> str | None:
    current: _HtmlNode | None = element
    while current is not None:
        if id(current) in section_ids:
            return section_ids[id(current)]
        current = parent_map.get(id(current))
    return None


def _table_rows(table: _HtmlNode, parent_map: Mapping[int, _HtmlNode]) -> list[_HtmlNode]:
    return [
        item
        for item in table.iter()
        if item is not table
        and item.tag == "tr"
        and nearest_ancestor(item, parent_map, {"table"}) is table
    ]


def _row_cells(row: _HtmlNode, parent_map: Mapping[int, _HtmlNode]) -> list[_HtmlNode]:
    return [
        item
        for item in row.iter()
        if item is not row
        and item.tag in {"th", "td"}
        and nearest_ancestor(item, parent_map, {"tr"}) is row
    ]


def _make_tables(
    root: _HtmlNode,
    parent_map: Mapping[int, _HtmlNode],
    paths: Mapping[int, str],
    source_artifact_id: str,
    section_ids: Mapping[int, str],
    config: Any,
) -> list[CanonicalTable]:
    tables: list[CanonicalTable] = []
    for ordinal, table in enumerate(item for item in _elements(root) if item.tag == "table"):
        locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(table)])
        caption_element = next((item for item in table.children if item.tag == "caption"), None)
        caption = _safe_text(caption_element, config, field="table caption") if caption_element is not None else None
        cells: list[CanonicalCell] = []
        occupied: set[tuple[int, int]] = set()
        for row_index, row in enumerate(_table_rows(table, parent_map)):
            column_index = 0
            for cell in _row_cells(row, parent_map):
                while (row_index, column_index) in occupied:
                    column_index += 1
                row_span = parse_positive_int(
                    cell.attrs.get("rowspan"), field="table row span", default=1, maximum=10_000
                )
                column_span = parse_positive_int(
                    cell.attrs.get("colspan"), field="table column span", default=1, maximum=10_000
                )
                cell_locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(cell)])
                for row_offset in range(row_span):
                    for column_offset in range(column_span):
                        position = (row_index + row_offset, column_index + column_offset)
                        if position in occupied:
                            raise MalformedDocumentError("HTML table spans overlap existing cells")
                        occupied.add(position)
                cells.append(
                    CanonicalCell(
                        row_index=row_index,
                        column_index=column_index,
                        row_span=row_span,
                        column_span=column_span,
                        is_header=cell.tag == "th",
                        text=_safe_text(cell, config, field="table cell"),
                        locator=cell_locator,
                    )
                )
                column_index += column_span
                if len(cells) > config.max_table_cells:
                    raise DocumentLimitError("table cell count exceeds the configured limit")
        tables.append(
            CanonicalTable(
                table_id=stable_id("tbl", source_artifact_id, "table", ordinal, locator),
                caption=caption,
                section_id=_nearest_section_id(table, parent_map, section_ids),
                locator=locator,
                cells=tuple(cells),
            )
        )
        if len(tables) > config.max_tables:
            raise DocumentLimitError("table count exceeds the configured limit")
    return tables


def _make_figures(
    root: _HtmlNode,
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> list[CanonicalFigure]:
    figures: list[CanonicalFigure] = []
    for ordinal, element in enumerate(item for item in _elements(root) if item.tag == "figure"):
        locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(element)])
        caption_element = next((item for item in element.children if item.tag == "figcaption"), None)
        figures.append(
            CanonicalFigure(
                figure_id=stable_id("fig", source_artifact_id, "figure", ordinal, locator),
                label=element.attrs.get("aria-label") or None,
                caption=_safe_text(caption_element, config, field="figure caption") if caption_element is not None else None,
                locator=locator,
            )
        )
        if len(figures) > config.max_figures:
            raise DocumentLimitError("figure count exceeds the configured limit")
    return figures


def _make_references(
    root: _HtmlNode,
    parent_map: Mapping[int, _HtmlNode],
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> list[CanonicalReference]:
    references: list[CanonicalReference] = []
    for ordinal, element in enumerate(item for item in _elements(root) if item.tag == "li"):
        if not is_inside(element, parent_map, {"ol", "ul", "div", "section"}):
            continue
        ancestor = parent_map.get(id(element))
        found_hint = False
        while ancestor is not None:
            if _REFERENCE_HINTS & set(_class_or_id_hint(ancestor).split()):
                found_hint = True
                break
            ancestor = parent_map.get(id(ancestor))
        if not found_hint:
            continue
        text = _safe_text(element, config, field="reference citation", required=True)
        locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(element)])
        references.append(
            CanonicalReference(
                reference_id=stable_id("ref", source_artifact_id, "reference", ordinal, locator),
                citation_text=text,
                identifier=citation_identifier(text),
                locator=locator,
            )
        )
        if len(references) > config.max_references:
            raise DocumentLimitError("reference count exceeds the configured limit")
    return references


def _make_blocks(
    root: _HtmlNode,
    parent_map: Mapping[int, _HtmlNode],
    paths: Mapping[int, str],
    source_artifact_id: str,
    section_ids: Mapping[int, str],
    config: Any,
    title_element: _HtmlNode | None,
) -> list[CanonicalBlock]:
    blocks: list[CanonicalBlock] = []
    for ordinal, element in enumerate(_elements(root)):
        tag = element.tag
        if is_inside(element, parent_map, _SKIP_TAGS):
            continue
        if title_element is not None and element is title_element:
            kind = CanonicalBlockKind.TITLE
        elif tag in {f"h{index}" for index in range(1, 7)}:
            kind = CanonicalBlockKind.HEADING
        elif tag in _PARAGRAPH_TAGS:
            if is_inside(element, parent_map, {"li", "td", "th", "caption", "figcaption"}):
                continue
            kind = CanonicalBlockKind.PARAGRAPH
        elif tag == "li":
            kind = CanonicalBlockKind.LIST_ITEM
        elif tag in {"caption", "figcaption"}:
            kind = CanonicalBlockKind.CAPTION
        else:
            continue
        text = _safe_text(element, config, field="HTML block")
        if not text:
            continue
        locator = source_locator(source_artifact_id, "HTML_ELEMENT", path=paths[id(element)])
        blocks.append(
            CanonicalBlock(
                block_id=stable_id("blk", source_artifact_id, "block", ordinal, locator),
                kind=kind.value,
                text=text,
                section_id=_nearest_section_id(element, parent_map, section_ids),
                locator=locator,
            )
        )
        if len(blocks) > config.max_blocks:
            raise DocumentLimitError("block count exceeds the configured limit")
    return blocks


class HtmlParser:
    parser_id = "html"
    version = "1"

    def parse(
        self,
        *,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
        config: Any,
    ) -> CanonicalDocument:
        root, parent_map, paths = _tree_details(source_bytes, config)
        elements = _elements(root)
        title_element = next((item for item in elements if item.tag == "title"), None)
        if title_element is None:
            title_element = next((item for item in elements if item.tag == "h1"), None)
        title = _safe_text(title_element, config, field="document title") if title_element is not None else None
        identifiers: list[str] = []
        for element in elements:
            if element.tag != "meta":
                continue
            hint = element.attrs.get("name", element.attrs.get("property", "")).casefold()
            if hint in {"doi", "citation_doi", "identifier", "citation_pdf_url"}:
                value = normalize_text(element.attrs.get("content", ""), field="document identifier", maximum=512)
                if value:
                    identifiers.append(value)
        language = attr_by_local_name(root, "lang")
        if language is not None:
            language = normalize_text(language, field="document language", maximum=64)
        sections, section_ids = _make_sections(
            root, parent_map, paths, source_artifact_id, config
        )
        tables = _make_tables(root, parent_map, paths, source_artifact_id, section_ids, config)
        figures = _make_figures(root, paths, source_artifact_id, config)
        references = _make_references(root, parent_map, paths, source_artifact_id, config)
        blocks = _make_blocks(
            root,
            parent_map,
            paths,
            source_artifact_id,
            section_ids,
            config,
            title_element,
        )
        text_char_count = sum(len(block.text) for block in blocks)
        text_char_count += sum(len(cell.text) for table in tables for cell in table.cells)
        quality = config.quality(
            text_char_count=text_char_count,
            block_count=len(blocks),
            table_count=len(tables),
        )
        return build_document(
            config=config,
            source_artifact_id=source_artifact_id,
            source_media_type=source_media_type,
            source_content_family="html",
            parser_id=self.parser_id,
            parser_version=self.version,
            language=language,
            title=title,
            identifiers=identifiers,
            sections=sections,
            blocks=blocks,
            tables=tables,
            figures=figures,
            references=references,
            quality=quality,
        )


__all__ = ["HtmlParser"]
