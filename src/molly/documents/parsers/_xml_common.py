"""Bounded XML tree handling shared by the JATS and generic XML parsers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
import xml.etree.ElementTree as ET
from typing import Any

from ..canonical import (
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalCell,
    CanonicalFigure,
    CanonicalReference,
    CanonicalSection,
    CanonicalTable,
)
from ..errors import DocumentLimitError, MalformedDocumentError, UnsupportedDocumentError
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


_XML_DANGEROUS_RE = re.compile(rb"<!\s*(?:doctype|entity)\b", re.IGNORECASE)
_SECTION_TAGS = {"sec", "section", "section-group"}
_PARAGRAPH_TAGS = {"p", "para", "paragraph"}
_HEADING_TAGS = {"title", "head", "heading", "section-title"}
_LIST_ITEM_TAGS = {"list-item", "li", "item"}
_TABLE_CONTAINER_TAGS = {"table-wrap", "table"}
_FIGURE_TAGS = {"fig", "figure"}
_REFERENCE_TAGS = {"ref" , "reference"}


def _tree_details(source_bytes: bytes, config: Any) -> tuple[Any, dict[int, Any], dict[int, str]]:
    ensure_source_limit(source_bytes, config)
    if _XML_DANGEROUS_RE.search(bytes(source_bytes)):
        raise MalformedDocumentError("XML external declaration is not permitted")
    try:
        root = ET.fromstring(bytes(source_bytes), parser=ET.XMLParser())
    except (ET.ParseError, UnicodeDecodeError, ValueError) as exc:
        raise MalformedDocumentError("XML source is malformed") from exc

    parent_map: dict[int, Any] = {}
    node_count = 0
    maximum_depth = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal node_count, maximum_depth
        node_count += 1
        maximum_depth = max(maximum_depth, depth)
        if node_count > config.max_node_count:
            raise DocumentLimitError("XML node count exceeds the configured limit")
        if depth > config.max_nesting_depth:
            raise DocumentLimitError("XML nesting depth exceeds the configured limit")
        for child in list(node):
            if not isinstance(getattr(child, "tag", None), str):
                continue
            parent_map[id(child)] = node
            visit(child, depth + 1)

    visit(root, 1)
    for element in _elements(root):
        tag = str(getattr(element, "tag", ""))
        if (
            local_name(tag) == "include"
            and "http://www.w3.org/2001/XInclude" in tag
        ):
            raise MalformedDocumentError("XML XInclude is not permitted")
    paths = structural_paths(root, max_depth=config.max_nesting_depth)
    root_text = normalize_text(" ".join(root.itertext()), field="XML normalized text")
    if len(root_text.encode("utf-8")) > config.max_normalized_text_bytes:
        raise DocumentLimitError("XML normalized text exceeds the configured limit")
    return root, parent_map, paths


def _safe_text(element: Any, config: Any, *, field: str, required: bool = False) -> str:
    value = element_text(element)
    if not value and not required:
        return ""
    if not value and required:
        raise MalformedDocumentError(f"{field} is empty")
    if len(value) > config.max_text_chars:
        raise DocumentLimitError(f"{field} exceeds the configured text limit")
    return value


def _elements(root: Any) -> list[Any]:
    return [element for element in root.iter() if isinstance(getattr(element, "tag", None), str)]


def _section_elements(root: Any, parent_map: Mapping[int, Any]) -> list[Any]:
    return [
        element
        for element in _elements(root)
        if local_name(element.tag) in _SECTION_TAGS
        and not is_inside(element, parent_map, _TABLE_CONTAINER_TAGS | _FIGURE_TAGS | _REFERENCE_TAGS)
    ]


def _nearest_section_id(element: Any, parent_map: Mapping[int, Any], section_ids: Mapping[int, str]) -> str | None:
    current = element
    while current is not None:
        value = section_ids.get(id(current))
        if value is not None:
            return value
        current = parent_map.get(id(current))
    return None


def _make_sections(
    root: Any,
    parent_map: Mapping[int, Any],
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> tuple[list[CanonicalSection], dict[int, str]]:
    sections: list[CanonicalSection] = []
    section_ids: dict[int, str] = {}
    elements = _section_elements(root, parent_map)
    for ordinal, element in enumerate(elements):
        locator = source_locator(
            source_artifact_id,
            "XML_ELEMENT",
            path=paths[id(element)],
        )
        section_id = stable_id("sec", source_artifact_id, "section", ordinal, locator)
        section_ids[id(element)] = section_id
        parent = nearest_ancestor(element, parent_map, _SECTION_TAGS)
        parent_id = section_ids.get(id(parent)) if parent is not None else None
        if parent is not None and parent_id is None:
            # Ancestors always precede descendants in ElementTree traversal;
            # this guard keeps malformed custom trees fail-closed.
            raise MalformedDocumentError("section hierarchy cannot be normalized")
        title_element = next(
            (
                child
                for child in list(element)
                if local_name(getattr(child, "tag", "")) in _HEADING_TAGS
            ),
            None,
        )
        title = _safe_text(title_element, config, field="section title") if title_element is not None else ""
        level = 1
        ancestor = parent
        while ancestor is not None:
            if local_name(getattr(ancestor, "tag", "")) in _SECTION_TAGS:
                level += 1
            ancestor = parent_map.get(id(ancestor))
        sections.append(
            CanonicalSection(
                section_id=section_id,
                title=title,
                level=level,
                parent_section_id=parent_id,
                locator=locator,
            )
        )
    return sections, section_ids


def _table_elements(root: Any, parent_map: Mapping[int, Any]) -> list[Any]:
    result: list[Any] = []
    for element in _elements(root):
        tag = local_name(element.tag)
        if tag == "table-wrap":
            result.append(element)
        elif tag == "table" and nearest_ancestor(element, parent_map, {"table-wrap"}) is None:
            result.append(element)
    return result


def _row_cells(row: Any, parent_map: Mapping[int, Any]) -> list[Any]:
    return [
        element
        for element in row.iter()
        if element is not row
        and local_name(getattr(element, "tag", "")) in {"th", "td"}
        and nearest_ancestor(element, parent_map, {"tr"}) is row
    ]


def _make_table(
    container: Any,
    parent_map: Mapping[int, Any],
    paths: Mapping[int, str],
    source_artifact_id: str,
    ordinal: int,
    section_ids: Mapping[int, str],
    config: Any,
) -> CanonicalTable:
    table_element = next(
        (element for element in container.iter() if local_name(getattr(element, "tag", "")) == "table"),
        container,
    )
    table_locator = source_locator(
        source_artifact_id,
        "XML_ELEMENT",
        path=paths[id(container)],
    )
    caption_element = next(
        (
            element
            for element in list(container)
            if local_name(getattr(element, "tag", "")) in {"caption", "title"}
        ),
        None,
    )
    caption = _safe_text(caption_element, config, field="table caption") if caption_element is not None else None
    cells: list[CanonicalCell] = []
    occupied: set[tuple[int, int]] = set()
    rows = [
        element
        for element in table_element.iter()
        if local_name(getattr(element, "tag", "")) == "tr"
        and nearest_ancestor(element, parent_map, {"table"}) is table_element
    ]
    for row_index, row in enumerate(rows):
        column_index = 0
        for cell in _row_cells(row, parent_map):
            while (row_index, column_index) in occupied:
                column_index += 1
            row_span = parse_positive_int(
                attr_by_local_name(cell, "rowspan"),
                field="table row span",
                default=1,
                maximum=10_000,
            )
            column_span = parse_positive_int(
                attr_by_local_name(cell, "colspan"),
                field="table column span",
                default=1,
                maximum=10_000,
            )
            locator = source_locator(
                source_artifact_id,
                "XML_ELEMENT",
                path=paths[id(cell)],
            )
            for row_offset in range(row_span):
                for column_offset in range(column_span):
                    position = (row_index + row_offset, column_index + column_offset)
                    if position in occupied:
                        raise MalformedDocumentError("table spans overlap existing cells")
                    occupied.add(position)
            cells.append(
                CanonicalCell(
                    row_index=row_index,
                    column_index=column_index,
                    row_span=row_span,
                    column_span=column_span,
                    is_header=local_name(cell.tag) == "th",
                    text=_safe_text(cell, config, field="table cell"),
                    locator=locator,
                )
            )
            column_index += column_span
            if len(cells) > config.max_table_cells:
                raise DocumentLimitError("table cell count exceeds the configured limit")
    table_id = stable_id("tbl", source_artifact_id, "table", ordinal, table_locator)
    return CanonicalTable(
        table_id=table_id,
        caption=caption,
        section_id=_nearest_section_id(container, parent_map, section_ids),
        locator=table_locator,
        cells=tuple(cells),
    )


def _make_figures(
    root: Any,
    parent_map: Mapping[int, Any],
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> list[CanonicalFigure]:
    figures: list[CanonicalFigure] = []
    for ordinal, element in enumerate(
        item for item in _elements(root) if local_name(item.tag) in _FIGURE_TAGS
    ):
        locator = source_locator(source_artifact_id, "XML_ELEMENT", path=paths[id(element)])
        label_element = next(
            (child for child in list(element) if local_name(getattr(child, "tag", "")) == "label"),
            None,
        )
        caption_element = next(
            (child for child in list(element) if local_name(getattr(child, "tag", "")) in {"caption", "title"}),
            None,
        )
        figures.append(
            CanonicalFigure(
                figure_id=stable_id("fig", source_artifact_id, "figure", ordinal, locator),
                label=_safe_text(label_element, config, field="figure label") if label_element is not None else None,
                caption=_safe_text(caption_element, config, field="figure caption") if caption_element is not None else None,
                locator=locator,
            )
        )
        if len(figures) > config.max_figures:
            raise DocumentLimitError("figure count exceeds the configured limit")
    return figures


def _make_references(
    root: Any,
    parent_map: Mapping[int, Any],
    paths: Mapping[int, str],
    source_artifact_id: str,
    config: Any,
) -> list[CanonicalReference]:
    references: list[CanonicalReference] = []
    for ordinal, element in enumerate(
        item
        for item in _elements(root)
        if local_name(item.tag) in _REFERENCE_TAGS
        and is_inside(item, parent_map, {"ref-list", "references", "bibliography"})
    ):
        text = _safe_text(element, config, field="reference citation", required=True)
        locator = source_locator(source_artifact_id, "XML_ELEMENT", path=paths[id(element)])
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


def _is_descendant_of(element: Any, parent_map: Mapping[int, Any], ancestor: Any) -> bool:
    current = parent_map.get(id(element))
    while current is not None:
        if current is ancestor:
            return True
        current = parent_map.get(id(current))
    return False


def _make_blocks(
    root: Any,
    parent_map: Mapping[int, Any],
    paths: Mapping[int, str],
    source_artifact_id: str,
    section_ids: Mapping[int, str],
    config: Any,
    *,
    title_element: Any | None,
    abstract_elements: tuple[Any, ...],
) -> list[CanonicalBlock]:
    blocks: list[CanonicalBlock] = []
    block_ordinal = 0

    def add(kind: CanonicalBlockKind, element: Any, text: str | None = None) -> None:
        nonlocal block_ordinal
        value = _safe_text(element, config, field="canonical block") if text is None else text
        if not value:
            return
        locator = source_locator(source_artifact_id, "XML_ELEMENT", path=paths[id(element)])
        blocks.append(
            CanonicalBlock(
                block_id=stable_id("blk", source_artifact_id, "block", block_ordinal, locator),
                kind=kind.value,
                text=value,
                section_id=_nearest_section_id(element, parent_map, section_ids),
                locator=locator,
            )
        )
        block_ordinal += 1
        if len(blocks) > config.max_blocks:
            raise DocumentLimitError("block count exceeds the configured limit")

    if title_element is not None:
        add(CanonicalBlockKind.TITLE, title_element)
    abstract_set = {id(item) for item in abstract_elements}
    for element in _elements(root):
        tag = local_name(element.tag)
        if id(element) in abstract_set:
            if tag in _PARAGRAPH_TAGS:
                add(CanonicalBlockKind.ABSTRACT, element)
            elif tag in {"abstract", "summary"}:
                add(CanonicalBlockKind.ABSTRACT, element)
            continue
        if title_element is not None and element is title_element:
            continue
        if is_inside(element, parent_map, _TABLE_CONTAINER_TAGS | _FIGURE_TAGS | _REFERENCE_TAGS):
            if tag in {"caption", "figcaption"}:
                add(CanonicalBlockKind.CAPTION, element)
            elif tag in _HEADING_TAGS and nearest_ancestor(element, parent_map, _FIGURE_TAGS | _TABLE_CONTAINER_TAGS) is not None:
                add(CanonicalBlockKind.CAPTION, element)
            continue
        if tag in _PARAGRAPH_TAGS:
            if is_inside(element, parent_map, {"abstract"} | _LIST_ITEM_TAGS):
                continue
            add(CanonicalBlockKind.PARAGRAPH, element)
        elif tag in _LIST_ITEM_TAGS:
            add(CanonicalBlockKind.LIST_ITEM, element)
        elif tag in _HEADING_TAGS:
            if nearest_ancestor(element, parent_map, _SECTION_TAGS) is not None:
                add(CanonicalBlockKind.HEADING, element)
        elif tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            add(CanonicalBlockKind.HEADING, element)
        elif tag in {"caption", "figcaption"}:
            add(CanonicalBlockKind.CAPTION, element)
    return blocks


def extract_xml_document(
    *,
    source_artifact_id: str,
    source_media_type: str,
    source_content_family: str,
    source_bytes: bytes,
    config: Any,
    parser_id: str,
    parser_version: str,
    jats: bool,
) -> Any:
    root, parent_map, paths = _tree_details(source_bytes, config)
    root_tag = local_name(root.tag)
    if jats and root_tag != "article":
        raise UnsupportedDocumentError("source is not a JATS article")
    if jats and not any(local_name(item.tag) == "article-meta" for item in _elements(root)):
        raise UnsupportedDocumentError("source does not contain JATS article metadata")

    elements = _elements(root)
    if jats:
        title_element = next((item for item in elements if local_name(item.tag) == "article-title"), None)
        abstract_roots = tuple(item for item in elements if local_name(item.tag) == "abstract")
        identifier_elements = [item for item in elements if local_name(item.tag) == "article-id"]
    else:
        title_element = next(
            (
                item
                for item in elements
                if local_name(item.tag) in {"title", "document-title", "article-title"}
                and not is_inside(
                    item,
                    parent_map,
                    _SECTION_TAGS | _TABLE_CONTAINER_TAGS | _FIGURE_TAGS | _REFERENCE_TAGS,
                )
            ),
            None,
        )
        abstract_roots = tuple(item for item in elements if local_name(item.tag) in {"abstract", "summary"})
        identifier_elements = [
            item for item in elements if local_name(item.tag) in {"article-id", "identifier", "doi"}
        ]
    title = _safe_text(title_element, config, field="document title") if title_element is not None else None
    identifiers = [
        _safe_text(item, config, field="document identifier")
        for item in identifier_elements
        if _safe_text(item, config, field="document identifier")
    ]
    language = attr_by_local_name(root, "lang")
    if language is not None:
        language = normalize_text(language, field="document language", maximum=64)
    sections, section_ids = _make_sections(
        root,
        parent_map,
        paths,
        source_artifact_id,
        config,
    )
    tables = [
        _make_table(
            container,
            parent_map,
            paths,
            source_artifact_id,
            ordinal,
            section_ids,
            config,
        )
        for ordinal, container in enumerate(_table_elements(root, parent_map))
    ]
    figures = _make_figures(root, parent_map, paths, source_artifact_id, config)
    references = _make_references(root, parent_map, paths, source_artifact_id, config)
    abstract_elements = tuple(
        item
        for abstract in abstract_roots
        for item in abstract.iter()
        if local_name(getattr(item, "tag", "")) in _PARAGRAPH_TAGS
    )
    if abstract_roots and not abstract_elements:
        abstract_elements = abstract_roots
    blocks = _make_blocks(
        root,
        parent_map,
        paths,
        source_artifact_id,
        section_ids,
        config,
        title_element=title_element,
        abstract_elements=abstract_elements,
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
        source_content_family=source_content_family,
        parser_id=parser_id,
        parser_version=parser_version,
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


__all__ = ["extract_xml_document"]
