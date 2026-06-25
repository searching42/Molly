from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent._utils import now_iso
from ai4s_agent.adapters.phase3 import _sha256_file
from ai4s_agent.document_parse_provider import DocumentParseOutputRefs
from ai4s_agent.schemas import ParsedDocument, ParsedDocumentElement, ParsedTable


class MinerUOutputBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    markdown_path: str = ""
    content_list_json_path: str = ""
    content_list_v2_json_path: str = ""
    middle_json_path: str = ""
    images_dir: str = ""
    other_paths: list[str] = Field(default_factory=list)


class NormalizedMinerUOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle: MinerUOutputBundle
    parsed_document: ParsedDocument
    warnings: list[str] = Field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _find_named_file(root: Path, suffix: str) -> Path | None:
    matches = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name.endswith(suffix)]
    return matches[0] if matches else None


def discover_mineru_output_bundle(output_dir: Path) -> MinerUOutputBundle:
    root = output_dir.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"MinerU output directory not found: {root}")
    markdown = _find_named_file(root, ".md")
    content_list = _find_named_file(root, "_content_list.json") or _find_named_file(root, "content_list.json")
    content_list_v2 = _find_named_file(root, "_content_list_v2.json") or _find_named_file(root, "content_list_v2.json")
    middle = _find_named_file(root, "_middle.json") or _find_named_file(root, "middle.json")
    images_dir = next((path for path in sorted(root.rglob("*")) if path.is_dir() and path.name.lower() == "images"), None)
    other_paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path in {markdown, content_list, content_list_v2, middle}:
            continue
        other_paths.append(str(path.relative_to(root)))
    if markdown is None and content_list is None and content_list_v2 is None:
        raise ValueError("MinerU output bundle must contain markdown or structured content_list JSON")
    return MinerUOutputBundle(
        output_dir=str(root),
        markdown_path=str(markdown) if markdown else "",
        content_list_json_path=str(content_list) if content_list else "",
        content_list_v2_json_path=str(content_list_v2) if content_list_v2 else "",
        middle_json_path=str(middle) if middle else "",
        images_dir=str(images_dir) if images_dir else "",
        other_paths=other_paths,
    )


def build_output_refs(
    *,
    output_dir: Path,
    parsed_document_json: Path,
    parsed_document_markdown: Path,
    parser_audit_json: Path,
    bundle: MinerUOutputBundle,
) -> DocumentParseOutputRefs:
    root = Path(bundle.output_dir)
    extracted_paths: list[str] = []
    for raw in [bundle.markdown_path, bundle.content_list_json_path, bundle.content_list_v2_json_path, bundle.middle_json_path]:
        if raw:
            extracted_paths.append(str(Path(raw).resolve().relative_to(root)))
    extracted_paths.extend(bundle.other_paths)
    return DocumentParseOutputRefs(
        output_dir=str(output_dir),
        parsed_document_json=str(parsed_document_json),
        parsed_document_markdown=str(parsed_document_markdown),
        parser_audit_json=str(parser_audit_json),
        content_list_json=bundle.content_list_json_path,
        content_list_v2_json=bundle.content_list_v2_json_path,
        middle_json=bundle.middle_json_path,
        extracted_paths=extracted_paths,
    )


def normalize_mineru_output_bundle(
    *,
    input_pdf: Path,
    bundle: MinerUOutputBundle,
    parser_backend: str,
) -> NormalizedMinerUOutput:
    root = Path(bundle.output_dir).resolve()
    markdown_text = Path(bundle.markdown_path).read_text(encoding="utf-8", errors="ignore") if bundle.markdown_path else ""
    content_payload = _load_content_payload(bundle)
    middle_payload = _read_json_object(Path(bundle.middle_json_path)) if bundle.middle_json_path else {}
    warnings: list[str] = []
    pages = _build_pages(content_payload, middle_payload)
    elements: list[ParsedDocumentElement] = []
    tables: list[ParsedTable] = []
    table_html_by_id: dict[str, str] = {}
    source_hash = _sha256_file(input_pdf)
    title = _extract_title_from_markdown(markdown_text, input_pdf.stem)
    if isinstance(content_payload, list) and content_payload:
        for index, raw in enumerate(content_payload, start=1):
            if not isinstance(raw, dict):
                continue
            normalized_page = _item_page(raw)
            bbox = _coerce_bbox(raw.get("bbox"))
            raw_type = str(raw.get("type") or raw.get("category") or raw.get("block_type") or "text").strip().lower()
            order = int(raw.get("order") or raw.get("sort_id") or index)
            if raw_type == "title" and title == input_pdf.stem:
                title = str(raw.get("text") or raw.get("content") or title).strip() or title
            if raw_type == "table":
                table, table_warnings = _table_from_content_item(
                    raw,
                    page=normalized_page,
                    order=order,
                    bbox=bbox,
                )
                tables.append(table)
                raw_html = str(raw.get("table_body") or raw.get("html") or raw.get("table_html") or "").strip()
                if raw_html:
                    table_html_by_id[table.table_id] = raw_html
                warnings.extend(table_warnings)
                continue
            element_text = _element_text(raw)
            element_markdown = str(raw.get("markdown") or element_text).strip()
            if not element_text and not element_markdown:
                continue
            elements.append(
                ParsedDocumentElement(
                    element_id=f"el_p{normalized_page}_{order:04d}",
                    page=normalized_page,
                    type=_element_type(raw_type),
                    text=element_text,
                    markdown=element_markdown or element_text,
                    bbox=bbox,
                    source_hash=source_hash,
                    metadata={
                        "source": "mineru_content_list",
                        "raw_type": raw_type,
                        "reading_order": order,
                        "text_level": str(raw.get("text_level") or ""),
                        "image_path": str(raw.get("image_path") or raw.get("img_path") or ""),
                        "coordinate_system": "mineru_bbox_0_1000",
                    },
                )
            )
    elif markdown_text:
        warnings.append("structured content_list JSON missing; falling back to Markdown-only normalization")
        elements.append(
            ParsedDocumentElement(
                element_id="el_p1_0001",
                page=1,
                type="markdown",
                text=markdown_text,
                markdown=markdown_text,
                bbox=None,
                source_hash=source_hash,
                metadata={
                    "source": "mineru_markdown_only",
                    "coordinate_system": "none",
                },
            )
        )
    else:
        raise ValueError("MinerU output bundle does not contain readable structured content or markdown")
    parsed_document = ParsedDocument(
        paper_id=input_pdf.stem,
        source_path=str(input_pdf),
        parser_backend=parser_backend,
        metadata={
            "title": title,
            "source_hash": source_hash,
            "mineru_output_dir": str(root),
            "mineru_markdown_path": _relative_or_empty(bundle.markdown_path, root),
            "mineru_content_list_json": _relative_or_empty(bundle.content_list_json_path, root),
            "mineru_content_list_v2_json": _relative_or_empty(bundle.content_list_v2_json_path, root),
            "mineru_middle_json": _relative_or_empty(bundle.middle_json_path, root),
            "mineru_backend": str(middle_payload.get("_backend") or ""),
            "mineru_version": str(middle_payload.get("_version_name") or ""),
            "bbox_coordinate_system": "mineru_bbox_0_1000",
            "table_html_by_id": table_html_by_id,
            "parsed_at": now_iso(),
        },
        pages=pages,
        elements=elements,
        tables=tables,
    )
    return NormalizedMinerUOutput(bundle=bundle, parsed_document=parsed_document, warnings=sorted(set(warnings)))


def _relative_or_empty(path_like: str, root: Path) -> str:
    if not path_like:
        return ""
    return str(Path(path_like).resolve().relative_to(root))


def _load_content_payload(bundle: MinerUOutputBundle) -> list[dict[str, Any]] | None:
    candidates = [bundle.content_list_json_path, bundle.content_list_v2_json_path]
    for raw_path in candidates:
        if not raw_path:
            continue
        payload = _read_json(Path(raw_path))
        if isinstance(payload, list):
            filtered = _flatten_content_items(payload)
            if payload and not filtered:
                raise ValueError(f"{Path(raw_path).name} does not contain valid structured content items")
            return filtered
        if isinstance(payload, dict):
            content_list = payload.get("content_list")
            if isinstance(content_list, list):
                filtered = _flatten_content_items(content_list)
                if content_list and not filtered:
                    raise ValueError(f"{Path(raw_path).name} does not contain valid structured content items")
                return filtered
        raise ValueError(f"{Path(raw_path).name} has an unsupported structured content format")
    return None


def _flatten_content_items(value: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page_index, item in enumerate(value):
        if isinstance(item, dict):
            items.append(_normalize_content_item(item))
        elif isinstance(item, list):
            items.extend(
                _normalize_content_item(child, page_idx=page_index, order=order)
                for order, child in enumerate(item, start=1)
                if isinstance(child, dict)
            )
    return items


def _normalize_content_item(
    item: dict[str, Any],
    *,
    page_idx: int | None = None,
    order: int | None = None,
) -> dict[str, Any]:
    normalized = dict(item)
    if page_idx is not None and "page_idx" not in normalized and "page" not in normalized:
        normalized["page_idx"] = page_idx
    if order is not None and "order" not in normalized and "sort_id" not in normalized:
        normalized["order"] = order
    content = normalized.get("content")
    if not isinstance(content, dict):
        return normalized

    for key in (
        "title_content",
        "paragraph_content",
        "text_content",
        "code_content",
        "algorithm_content",
        "math_content",
        "code_body",
        "list_items",
        "table_body",
        "table_html",
        "table_caption",
        "table_footnote",
        "img_path",
        "image_path",
        "image_caption",
        "image_footnote",
        "latex",
        "markdown",
    ):
        if key in content and key not in normalized:
            normalized[key] = content[key]
    if "text" not in normalized:
        for key in ("title_content", "paragraph_content", "text_content", "code_content", "algorithm_content", "math_content"):
            value = content.get(key)
            if value:
                normalized["text"] = value
                break
    return normalized


def _build_pages(content_payload: list[dict[str, Any]] | None, middle_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    pdf_info = middle_payload.get("pdf_info")
    if isinstance(pdf_info, list):
        for index, item in enumerate(pdf_info, start=1):
            if not isinstance(item, dict):
                continue
            page_idx = item.get("page_idx")
            page_number = int(page_idx) + 1 if isinstance(page_idx, int) else index
            seen.add(page_number)
            page_payload = {"page": page_number}
            if isinstance(item.get("width"), (int, float)):
                page_payload["width"] = float(item["width"])
            if isinstance(item.get("height"), (int, float)):
                page_payload["height"] = float(item["height"])
            page_size = item.get("page_size")
            if isinstance(page_size, list) and len(page_size) >= 2:
                if isinstance(page_size[0], (int, float)):
                    page_payload["width"] = float(page_size[0])
                if isinstance(page_size[1], (int, float)):
                    page_payload["height"] = float(page_size[1])
            pages.append(page_payload)
    if content_payload:
        for item in content_payload:
            page_number = _item_page(item)
            if page_number in seen:
                continue
            seen.add(page_number)
            pages.append({"page": page_number})
    if not pages:
        pages.append({"page": 1})
    return sorted(pages, key=lambda item: int(item.get("page") or 1))


def _item_page(item: dict[str, Any]) -> int:
    if isinstance(item.get("page_idx"), int):
        return int(item["page_idx"]) + 1
    if isinstance(item.get("page"), int):
        return int(item["page"])
    return 1


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(part) for part in value]
    except (TypeError, ValueError):
        return None


def _bbox_dict(value: list[float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x0": value[0], "y0": value[1], "x1": value[2], "y1": value[3]}


def _extract_title_from_markdown(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _element_text(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("list_items"), list):
        list_text = _span_sequence_text(raw["list_items"], delimiter="\n")
        if list_text:
            return list_text
    for key in (
        "text",
        "title_content",
        "paragraph_content",
        "text_content",
        "code_content",
        "algorithm_content",
        "math_content",
        "latex",
        "caption",
        "code_body",
        "image_caption",
        "content",
    ):
        if isinstance(raw.get(key), list):
            value = _span_text(raw[key])
            if value:
                return value
            continue
        if isinstance(raw.get(key), dict):
            value = _span_text(raw[key])
            if value:
                return value
            continue
        value = str(raw.get(key) or "").strip()
        if value:
            return value
    return ""


def _element_type(raw_type: str) -> str:
    mapping = {
        "title": "title",
        "text": "paragraph",
        "list": "list",
        "code": "code",
        "algorithm": "code",
        "equation": "equation",
        "math": "equation",
        "formula": "equation",
        "image": "image",
        "chart": "chart",
    }
    return mapping.get(raw_type, "paragraph")


def _table_from_content_item(
    raw: dict[str, Any],
    *,
    page: int,
    order: int,
    bbox: list[float] | None,
) -> tuple[ParsedTable, list[str]]:
    warnings: list[str] = []
    html = str(raw.get("table_body") or raw.get("html") or raw.get("table_html") or "").strip()
    headers: list[str] = []
    rows: list[dict[str, str]] = []
    if html:
        try:
            headers, rows, parse_warning = _table_from_html(html)
            if parse_warning:
                warnings.append(parse_warning)
        except ValueError as exc:
            warnings.append(f"table_html_parse_warning: {exc}")
    caption = _text_from_string_or_list(raw.get("table_caption") or raw.get("caption"))
    footnotes = _list_from_string_or_list(raw.get("table_footnote") or raw.get("footnote"))
    if not headers and isinstance(raw.get("headers"), list):
        headers = [str(item).strip() for item in raw["headers"] if str(item).strip()]
    if not rows and isinstance(raw.get("rows"), list):
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in raw["rows"]
            if isinstance(row, dict)
        ]
    markdown = str(raw.get("markdown") or "").strip() or _table_markdown(headers, rows)
    table = ParsedTable(
        table_id=f"table_p{page}_{order:04d}",
        caption=caption,
        headers=headers,
        rows=rows,
        footnotes=footnotes,
        page=page,
        markdown=markdown,
        source_bbox=_bbox_dict(bbox),
    )
    if html:
        table.rows = [{**row} for row in table.rows]
    return table, warnings


def _text_from_string_or_list(value: Any) -> str:
    return _span_text(value)


def _list_from_string_or_list(value: Any) -> list[str]:
    if isinstance(value, list):
        if any(isinstance(item, dict) or isinstance(item, list) for item in value):
            text = _span_text(value)
            return [text] if text else []
        return [str(item).strip() for item in value if str(item).strip()]
    clean = str(value or "").strip()
    return [clean] if clean else []


def _span_sequence_text(value: list[Any], *, delimiter: str) -> str:
    return delimiter.join(part for item in value if (part := _span_text(item)))


def _span_text(value: Any, *, _nested: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value if _nested else value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        joined = "".join(part for item in value if (part := _span_text(item, _nested=True)))
        return joined if _nested else joined.strip()
    if not isinstance(value, dict):
        return ""

    parts: list[str] = []
    for key in (
        "text",
        "value",
        "title_content",
        "paragraph_content",
        "text_content",
        "code_content",
        "algorithm_content",
        "math_content",
        "code_body",
        "latex",
        "caption",
        "content",
        "children",
    ):
        if key in value:
            part = _span_text(value[key], _nested=True)
            if part:
                parts.append(part)
    joined = "".join(parts)
    return joined if _nested else joined.strip()


def _table_markdown(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not headers:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


class _SimpleTableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None

    @property
    def rows(self) -> list[list[dict[str, Any]]]:
        return self._rows

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "tr":
            self._current_row = []
            return
        if lowered not in {"th", "td"}:
            return
        attr_map = {str(key): str(value or "") for key, value in attrs}
        self._current_cell = {
            "text": "",
            "header": lowered == "th",
            "colspan": _positive_int(attr_map.get("colspan"), default=1),
            "rowspan": _positive_int(attr_map.get("rowspan"), default=1),
        }

    def handle_data(self, data: str) -> None:
        if self._current_cell is None:
            return
        self._current_cell["text"] = (str(self._current_cell["text"]) + data).strip()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"th", "td"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(dict(self._current_cell))
            self._current_cell = None
            return
        if lowered == "tr" and self._current_row is not None:
            self._rows.append(list(self._current_row))
            self._current_row = None


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _table_from_html(html: str) -> tuple[list[str], list[dict[str, str]], str]:
    parser = _SimpleTableHtmlParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        raise ValueError("table HTML does not contain rows")
    expanded = _expand_cells(rows)
    if not expanded:
        raise ValueError("table HTML expansion produced no rows")
    header_row = expanded[0]
    headers = [cell or f"column_{index + 1}" for index, cell in enumerate(header_row)]
    body_rows = expanded[1:]
    warning = ""
    if any(len(row) != len(headers) for row in body_rows):
        warning = "table HTML required uneven row normalization"
    normalized_rows = []
    for row in body_rows:
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        normalized_rows.append({header: str(padded[index] or "") for index, header in enumerate(headers)})
    return headers, normalized_rows, warning


def _expand_cells(rows: list[list[dict[str, Any]]]) -> list[list[str]]:
    grid: list[list[str]] = []
    spanning: dict[int, tuple[int, str]] = {}
    for raw_row in rows:
        row: list[str] = []
        column = 0
        while column in spanning:
            remaining, text = spanning[column]
            row.append(text)
            if remaining <= 1:
                spanning.pop(column)
            else:
                spanning[column] = (remaining - 1, text)
            column += 1
        for cell in raw_row:
            while column in spanning:
                remaining, text = spanning[column]
                row.append(text)
                if remaining <= 1:
                    spanning.pop(column)
                else:
                    spanning[column] = (remaining - 1, text)
                column += 1
            text = str(cell.get("text") or "").strip()
            colspan = _positive_int(cell.get("colspan"), default=1)
            rowspan = _positive_int(cell.get("rowspan"), default=1)
            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    spanning[column + offset] = (rowspan - 1, text)
            column += colspan
        while column in spanning:
            remaining, text = spanning[column]
            row.append(text)
            if remaining <= 1:
                spanning.pop(column)
            else:
                spanning[column] = (remaining - 1, text)
            column += 1
        grid.append(row)
    return grid
