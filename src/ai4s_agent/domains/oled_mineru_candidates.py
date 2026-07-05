from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY


class OledMineruSourceFormat(str, Enum):
    CONTENT_LIST = "content_list"
    CONTENT_LIST_V2 = "content_list_v2"
    MINERU_LIKE = "mineru_like"
    UNKNOWN = "unknown"


class OledMineruCandidateType(str, Enum):
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    CHART = "chart"
    UNKNOWN = "unknown"


class OledMineruRelevanceSignal(str, Enum):
    OLED_KEYWORD = "oled_keyword"
    DEVICE_KEYWORD = "device_keyword"
    PROPERTY_KEYWORD = "property_keyword"
    MEASUREMENT_KEYWORD = "measurement_keyword"
    MATERIAL_ROLE_KEYWORD = "material_role_keyword"
    ENERGY_LEVEL_KEYWORD = "energy_level_keyword"
    FABRICATION_KEYWORD = "fabrication_keyword"


class OledMineruTableParseStatus(str, Enum):
    PARSED = "parsed"
    EMPTY = "empty"
    MALFORMED = "malformed"
    COMPLEX_UNSUPPORTED = "complex_unsupported"
    NOT_TABLE = "not_table"


class OledMineruCandidate(BaseModel):
    paper_id: str
    source_path: str | None = None
    source_format: OledMineruSourceFormat
    candidate_type: OledMineruCandidateType
    page_index: int | None = None
    block_index: int
    block_id: str
    section_title: str | None = None
    bbox: list[float] | None = None
    image_path: str | None = None
    raw_text: str
    caption: str | None = None
    markdown_table: str | None = None
    html_table: str | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, str]] = Field(default_factory=list)
    table_parse_status: OledMineruTableParseStatus = OledMineruTableParseStatus.NOT_TABLE
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    evidence_anchor: str
    candidate_hash: str
    relevance_signals: list[OledMineruRelevanceSignal] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("paper_id", "block_id", "raw_text", "evidence_anchor", "candidate_hash")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("matched_terms")
    @classmethod
    def validate_matched_terms(cls, value: list[str]) -> list[str]:
        return sorted({clean for item in value if (clean := _term_id(str(item)))})

    @field_validator("relevance_signals")
    @classmethod
    def validate_relevance_signals(
        cls,
        value: list[OledMineruRelevanceSignal],
    ) -> list[OledMineruRelevanceSignal]:
        seen: set[OledMineruRelevanceSignal] = set()
        ordered: list[OledMineruRelevanceSignal] = []
        for signal in value:
            if signal not in seen:
                ordered.append(signal)
                seen.add(signal)
        return ordered


class OledMineruCandidateSummary(BaseModel):
    total_candidates: int
    relevant_candidate_count: int
    paper_ids: list[str] = Field(default_factory=list)
    candidates_by_type: dict[OledMineruCandidateType, int] = Field(default_factory=dict)
    candidates_by_signal: dict[OledMineruRelevanceSignal, int] = Field(default_factory=dict)
    table_candidate_count: int = 0
    text_candidate_count: int = 0
    caption_candidate_count: int = 0
    figure_candidate_count: int = 0
    chart_candidate_count: int = 0


def detect_oled_mineru_source_format(content: Any) -> OledMineruSourceFormat:
    if isinstance(content, dict):
        if any(key in content for key in ("content_list", "pages", "mineru_output")):
            return OledMineruSourceFormat.MINERU_LIKE
        return OledMineruSourceFormat.UNKNOWN
    if not isinstance(content, list) or not content:
        return OledMineruSourceFormat.UNKNOWN
    if all(isinstance(item, dict) for item in content):
        return OledMineruSourceFormat.CONTENT_LIST
    if all(isinstance(item, list) for item in content):
        return OledMineruSourceFormat.CONTENT_LIST_V2
    if any(isinstance(item, (dict, list)) for item in content):
        return OledMineruSourceFormat.MINERU_LIKE
    return OledMineruSourceFormat.UNKNOWN


def extract_oled_mineru_candidates(
    parsed_documents: Iterable[dict[str, Any] | list[Any]],
    *,
    md_by_paper_id: Mapping[str, str] | None = None,
    include_irrelevant: bool = False,
    source_path_by_paper_id: Mapping[str, str] | None = None,
) -> list[OledMineruCandidate]:
    candidates: list[OledMineruCandidate] = []
    for index, parsed_document in enumerate(parsed_documents, start=1):
        paper_id = _paper_id_for_document(parsed_document, index)
        candidates.extend(
            extract_oled_mineru_candidates_from_document(
                parsed_document,
                paper_id=paper_id,
                source_path=(source_path_by_paper_id or {}).get(paper_id),
                md_text=(md_by_paper_id or {}).get(paper_id),
                include_irrelevant=include_irrelevant,
            )
        )
    return candidates


def extract_oled_mineru_candidates_from_document(
    parsed_document: dict[str, Any] | list[Any],
    *,
    paper_id: str | None = None,
    source_path: str | None = None,
    md_text: str | None = None,
    include_irrelevant: bool = False,
) -> list[OledMineruCandidate]:
    clean_paper_id = str(paper_id or _paper_id_for_document(parsed_document, 1)).strip() or "paper-001"
    source_format = detect_oled_mineru_source_format(parsed_document)
    blocks = _flatten_mineru_blocks(parsed_document, source_format=source_format)

    candidates: list[OledMineruCandidate] = []
    for block in blocks:
        candidate = _candidate_from_block(
            block,
            paper_id=clean_paper_id,
            source_path=source_path,
            source_format=source_format,
            md_text=md_text,
        )
        if candidate is None:
            continue
        if candidate.relevance_signals or include_irrelevant:
            candidates.append(candidate)
    return candidates


def summarize_oled_mineru_candidates(
    candidates: Iterable[OledMineruCandidate],
) -> OledMineruCandidateSummary:
    candidate_list = list(candidates)
    candidates_by_type = {candidate_type: 0 for candidate_type in OledMineruCandidateType}
    candidates_by_signal = {signal: 0 for signal in OledMineruRelevanceSignal}
    for candidate in candidate_list:
        candidates_by_type[candidate.candidate_type] += 1
        for signal in candidate.relevance_signals:
            candidates_by_signal[signal] += 1
    return OledMineruCandidateSummary(
        total_candidates=len(candidate_list),
        relevant_candidate_count=sum(1 for candidate in candidate_list if candidate.relevance_signals),
        paper_ids=sorted({candidate.paper_id for candidate in candidate_list}),
        candidates_by_type=candidates_by_type,
        candidates_by_signal=candidates_by_signal,
        table_candidate_count=candidates_by_type[OledMineruCandidateType.TABLE],
        text_candidate_count=candidates_by_type[OledMineruCandidateType.TEXT],
        caption_candidate_count=sum(1 for candidate in candidate_list if candidate.caption),
        figure_candidate_count=candidates_by_type[OledMineruCandidateType.FIGURE],
        chart_candidate_count=candidates_by_type[OledMineruCandidateType.CHART],
    )


@dataclass(frozen=True)
class _FlattenedMineruBlock:
    raw_block: dict[str, Any]
    block_index: int
    page_index: int | None
    section_title: str | None


@dataclass(frozen=True)
class _BlockFields:
    raw_text: str
    caption: str | None
    markdown_table: str | None
    html_table: str | None
    image_path: str | None


@dataclass(frozen=True)
class _ParsedTable:
    headers: list[str]
    rows: list[dict[str, str]]
    status: OledMineruTableParseStatus


def _candidate_from_block(
    block: _FlattenedMineruBlock,
    *,
    paper_id: str,
    source_path: str | None,
    source_format: OledMineruSourceFormat,
    md_text: str | None,
) -> OledMineruCandidate | None:
    candidate_type = _candidate_type(block.raw_block)
    fields = _block_fields(block.raw_block, candidate_type)
    raw_text = _candidate_raw_text(fields)
    if not raw_text:
        return None
    parsed_table = _parse_table(fields, candidate_type)
    matched_terms, relevance_signals = _relevance(raw_text)
    block_id = _block_id(paper_id, source_format, block)
    evidence_anchor = f"{block_id}:{candidate_type.value}"
    nearby_before, nearby_after = _nearby_md_context(md_text, fields.caption or raw_text)
    candidate_hash = _candidate_hash(
        {
            "paper_id": paper_id,
            "source_format": source_format.value,
            "candidate_type": candidate_type.value,
            "section_title": block.section_title,
            "bbox": _bbox(block.raw_block),
            "image_path": fields.image_path,
            "raw_text": raw_text,
            "caption": fields.caption,
            "markdown_table": fields.markdown_table,
            "html_table": fields.html_table,
            "table_headers": parsed_table.headers,
            "table_rows": parsed_table.rows,
            "table_parse_status": parsed_table.status.value,
            "relevance_signals": [signal.value for signal in relevance_signals],
            "matched_terms": matched_terms,
        }
    )
    return OledMineruCandidate(
        paper_id=paper_id,
        source_path=source_path,
        source_format=source_format,
        candidate_type=candidate_type,
        page_index=block.page_index,
        block_index=block.block_index,
        block_id=block_id,
        section_title=block.section_title,
        bbox=_bbox(block.raw_block),
        image_path=fields.image_path,
        raw_text=raw_text,
        caption=fields.caption,
        markdown_table=fields.markdown_table,
        html_table=fields.html_table,
        table_headers=parsed_table.headers,
        table_rows=parsed_table.rows,
        table_parse_status=parsed_table.status,
        nearby_text_before=nearby_before,
        nearby_text_after=nearby_after,
        evidence_anchor=evidence_anchor,
        candidate_hash=candidate_hash,
        relevance_signals=relevance_signals,
        matched_terms=matched_terms,
        metadata={
            "candidate_layer": "mineru_evidence_only",
            "md_sidecar_policy": "nearby_context_only",
            "schema_extraction_policy": "candidates_only_no_oled_layered_records",
        },
    )


def _paper_id_for_document(parsed_document: dict[str, Any] | list[Any], index: int) -> str:
    if isinstance(parsed_document, dict):
        for key in ("paper_id", "source_id", "document_id"):
            value = parsed_document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"paper-{index:03d}"


def _flatten_mineru_blocks(
    content: Any,
    *,
    source_format: OledMineruSourceFormat,
) -> list[_FlattenedMineruBlock]:
    if source_format == OledMineruSourceFormat.CONTENT_LIST:
        return _flatten_content_list(content)
    if source_format == OledMineruSourceFormat.CONTENT_LIST_V2:
        return _flatten_content_list_v2(content)
    if source_format == OledMineruSourceFormat.MINERU_LIKE:
        nested = _mineru_like_payload(content)
        nested_format = detect_oled_mineru_source_format(nested)
        return _flatten_mineru_blocks(nested, source_format=nested_format)
    return []


def _flatten_content_list(content: Any) -> list[_FlattenedMineruBlock]:
    if not isinstance(content, list):
        return []
    blocks: list[_FlattenedMineruBlock] = []
    section_title: str | None = None
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        blocks.append(
            _FlattenedMineruBlock(
                raw_block=block,
                block_index=index,
                page_index=_coerce_page_idx(block.get("page_idx")),
                section_title=section_title,
            )
        )
        if _candidate_type(block) == OledMineruCandidateType.TITLE:
            section_title = _clean_text(_block_fields(block, OledMineruCandidateType.TITLE).raw_text) or section_title
    return blocks


def _flatten_content_list_v2(content: Any) -> list[_FlattenedMineruBlock]:
    if not isinstance(content, list):
        return []
    blocks: list[_FlattenedMineruBlock] = []
    section_title: str | None = None
    block_index = 0
    for page_index, page in enumerate(content):
        if not isinstance(page, list):
            continue
        for block in page:
            if not isinstance(block, dict):
                continue
            blocks.append(
                _FlattenedMineruBlock(
                    raw_block=block,
                    block_index=block_index,
                    page_index=_coerce_page_idx(block.get("page_idx"), default=page_index),
                    section_title=section_title,
                )
            )
            if _candidate_type(block) == OledMineruCandidateType.TITLE:
                section_title = _clean_text(_block_fields(block, OledMineruCandidateType.TITLE).raw_text) or section_title
            block_index += 1
    return blocks


def _mineru_like_payload(content: Any) -> Any:
    if not isinstance(content, dict):
        return content
    for key in ("content_list", "pages", "mineru_output"):
        if key in content:
            return content[key]
    return []


def _candidate_type(block: dict[str, Any]) -> OledMineruCandidateType:
    block_type = str(block.get("type") or block.get("block_type") or "").strip().lower()
    if "table" in block_type:
        return OledMineruCandidateType.TABLE
    if "chart" in block_type:
        return OledMineruCandidateType.CHART
    if "image" in block_type or "figure" in block_type:
        return OledMineruCandidateType.FIGURE
    if block_type in {"title", "section", "heading", "header"}:
        return OledMineruCandidateType.TITLE
    if block_type in {"text", "paragraph", "para"}:
        return OledMineruCandidateType.TEXT
    if _caption(block, OledMineruCandidateType.FIGURE):
        return OledMineruCandidateType.FIGURE
    if _caption(block, OledMineruCandidateType.CHART):
        return OledMineruCandidateType.CHART
    return OledMineruCandidateType.UNKNOWN


def _block_fields(block: dict[str, Any], candidate_type: OledMineruCandidateType) -> _BlockFields:
    caption = _caption(block, candidate_type)
    html_table = _html_table(block)
    markdown_table = _markdown_table(block)
    image_path = _image_path(block)
    text_parts: list[str] = []
    if candidate_type == OledMineruCandidateType.TABLE:
        text_parts.extend([_clean_table_text(markdown_table or html_table or _flat_value(block, "table_body")), caption])
        text_parts.append(_flat_value(block, "table_footnote") or _content_value(block, "table_footnote"))
    elif candidate_type in {OledMineruCandidateType.FIGURE, OledMineruCandidateType.CHART}:
        text_parts.append(caption)
    elif candidate_type == OledMineruCandidateType.TITLE:
        text_parts.append(_flat_value(block, "text") or _content_value(block, "title_content"))
    else:
        text_parts.append(_flat_value(block, "text") or _content_value(block, "paragraph_content") or _content_value(block, "text"))
    raw_text = _join_text(text_parts)
    return _BlockFields(
        raw_text=raw_text,
        caption=caption,
        markdown_table=markdown_table,
        html_table=html_table,
        image_path=image_path,
    )


def _candidate_raw_text(fields: _BlockFields) -> str:
    return _join_text([fields.raw_text, fields.caption])


def _caption(block: dict[str, Any], candidate_type: OledMineruCandidateType) -> str | None:
    keys: tuple[str, ...]
    if candidate_type == OledMineruCandidateType.TABLE:
        keys = ("table_caption",)
    elif candidate_type == OledMineruCandidateType.CHART:
        keys = ("chart_caption", "image_caption")
    elif candidate_type == OledMineruCandidateType.FIGURE:
        keys = ("image_caption", "chart_caption")
    else:
        keys = ("table_caption", "chart_caption", "image_caption")
    for key in keys:
        value = _flat_value(block, key) or _content_value(block, key)
        if value:
            return value
    return None


def _html_table(block: dict[str, Any]) -> str | None:
    for value in (_flat_value(block, "table_body", clean=False), _content_value(block, "html", clean=False)):
        if value and "<table" in value.lower():
            return value.strip()
    return None


def _markdown_table(block: dict[str, Any]) -> str | None:
    value = _flat_value(block, "table_body", clean=False) or _content_value(block, "markdown", clean=False)
    if value and _looks_like_markdown_table(value):
        return value.strip()
    return None


def _flat_value(block: dict[str, Any], key: str, *, clean: bool = True) -> str | None:
    value = block.get(key)
    if value is None:
        return None
    text = _coerce_text(value)
    return _clean_text(text) if clean else text.strip()


def _content_value(block: dict[str, Any], key: str, *, clean: bool = True) -> str | None:
    content = block.get("content")
    if not isinstance(content, dict) or key not in content:
        return None
    text = _coerce_text(content[key])
    return _clean_text(text) if clean else text.strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _coerce_text(item).strip()))
    if isinstance(value, dict):
        text_parts: list[str] = []
        for key in ("text", "content", "paragraph_content", "title_content", "caption", "html"):
            if key in value:
                text = _coerce_text(value[key]).strip()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    return str(value)


def _clean_text(value: str | None) -> str | None:
    text = html_lib.unescape(str(value or ""))
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _clean_table_text(value: str | None) -> str | None:
    return _clean_text(value)


def _join_text(values: Iterable[str | None]) -> str:
    return "\n".join(text for value in values if (text := str(value or "").strip()))


def _parse_table(fields: _BlockFields, candidate_type: OledMineruCandidateType) -> _ParsedTable:
    if candidate_type != OledMineruCandidateType.TABLE:
        return _ParsedTable([], [], OledMineruTableParseStatus.NOT_TABLE)
    if fields.html_table:
        return _parse_html_table(fields.html_table)
    if fields.markdown_table:
        return _parse_markdown_table(fields.markdown_table)
    return _ParsedTable([], [], OledMineruTableParseStatus.EMPTY)


def _parse_markdown_table(markdown_table: str) -> _ParsedTable:
    lines = [line.strip() for line in markdown_table.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) < 2:
        return _ParsedTable([], [], OledMineruTableParseStatus.EMPTY)
    headers = _markdown_cells(table_lines[0])
    separator = _markdown_cells(table_lines[1])
    if not headers or not _is_markdown_separator(separator):
        return _ParsedTable([], [], OledMineruTableParseStatus.MALFORMED)
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = _markdown_cells(line)
        if len(cells) != len(headers):
            return _ParsedTable(headers, rows, OledMineruTableParseStatus.MALFORMED)
        rows.append(dict(zip(headers, cells, strict=True)))
    return _ParsedTable(headers, rows, OledMineruTableParseStatus.PARSED)


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_text(cell) or "" for cell in stripped.split("|")]


def _is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _parse_html_table(html_table: str) -> _ParsedTable:
    if not html_table.strip():
        return _ParsedTable([], [], OledMineruTableParseStatus.EMPTY)
    if re.search(r"\b(?:rowspan|colspan)\s*=", html_table, flags=re.IGNORECASE):
        return _ParsedTable([], [], OledMineruTableParseStatus.COMPLEX_UNSUPPORTED)
    parser = _SimpleHtmlTableParser()
    try:
        parser.feed(html_table)
        parser.close()
    except Exception:
        return _ParsedTable([], [], OledMineruTableParseStatus.MALFORMED)
    if parser.malformed:
        return _ParsedTable([], [], OledMineruTableParseStatus.MALFORMED)
    rows = [[_clean_text(cell) or "" for cell in row] for row in parser.rows if any(_clean_text(cell) for cell in row)]
    if not rows:
        return _ParsedTable([], [], OledMineruTableParseStatus.EMPTY)
    headers = rows[0]
    parsed_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        if len(row) != len(headers):
            return _ParsedTable(headers, parsed_rows, OledMineruTableParseStatus.MALFORMED)
        parsed_rows.append(dict(zip(headers, row, strict=True)))
    return _ParsedTable(headers, parsed_rows, OledMineruTableParseStatus.PARSED)


class _SimpleHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_table = False
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
        if not self._in_table:
            return
        if tag == "tr":
            if self._current_row is not None:
                self.malformed = True
            self._current_row = []
        elif tag in {"td", "th"}:
            if self._current_row is None or self._current_cell is not None:
                self.malformed = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            if self._current_cell is not None:
                self.malformed = True
                self._finish_cell()
            if self._current_row is not None:
                self.rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            if self._current_cell is not None or self._current_row is not None:
                self.malformed = True
            self._in_table = False

    def _finish_cell(self) -> None:
        if self._current_cell is None:
            self.malformed = True
            return
        if self._current_row is None:
            self.malformed = True
            self._current_cell = None
            return
        self._current_row.append(_clean_text(" ".join(self._current_cell)) or "")
        self._current_cell = None


def _relevance(text: str) -> tuple[list[str], list[OledMineruRelevanceSignal]]:
    normalized_text = _normalize_match_text(text)
    matched_terms: list[str] = []
    signals: list[OledMineruRelevanceSignal] = []
    for signal, terms in _RELEVANCE_TERMS.items():
        for term in terms:
            if _contains_term(normalized_text, term):
                matched_terms.append(_term_id(term))
                signals.append(signal)
    return sorted(set(matched_terms)), _ordered_signals(signals)


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize_match_text(term)
    if not normalized_term:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text))


def _ordered_signals(signals: Iterable[OledMineruRelevanceSignal]) -> list[OledMineruRelevanceSignal]:
    seen = set(signals)
    return [signal for signal in OledMineruRelevanceSignal if signal in seen]


def _nearby_md_context(md_text: str | None, evidence_text: str | None) -> tuple[str | None, str | None]:
    if not md_text or not evidence_text:
        return None, None
    md_clean = _clean_text(md_text) or ""
    needle = _clean_text(evidence_text) or ""
    if not md_clean or not needle:
        return None, None
    match_index = md_clean.lower().find(needle.lower())
    if match_index < 0:
        return None, None
    before = md_clean[max(0, match_index - _MD_CONTEXT_CHARS) : match_index].strip()
    after = md_clean[match_index + len(needle) : match_index + len(needle) + _MD_CONTEXT_CHARS].strip()
    return before or None, after or None


def _block_id(
    paper_id: str,
    source_format: OledMineruSourceFormat,
    block: _FlattenedMineruBlock,
) -> str:
    page = f"p{block.page_index}" if block.page_index is not None else "pna"
    return f"{paper_id}:{source_format.value}:{page}:b{block.block_index}"


def _candidate_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bbox(block: dict[str, Any]) -> list[float] | None:
    raw_bbox = block.get("bbox")
    if not isinstance(raw_bbox, list):
        return None
    bbox: list[float] = []
    for item in raw_bbox:
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            bbox.append(float(item))
        else:
            return None
    return bbox or None


def _image_path(block: dict[str, Any]) -> str | None:
    if image_path := str(block.get("img_path") or "").strip():
        return image_path
    content = block.get("content")
    if isinstance(content, dict):
        image_source = content.get("image_source")
        if isinstance(image_source, dict):
            if image_path := str(image_source.get("path") or "").strip():
                return image_path
    return None


def _coerce_page_idx(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def _looks_like_markdown_table(value: str) -> bool:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return len(lines) >= 2 and "|" in lines[0] and "|" in lines[1]


def _normalize_match_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "λ": "lambda",
        "Δ": "delta",
        "δ": "delta",
        "²": "2",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _term_id(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "λ": "lambda",
        "Δ": "delta",
        "δ": "delta",
        "²": "2",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_PROPERTY_TERMS = sorted(
    {
        term
        for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()
        for term in {definition.property_id, definition.name, *definition.aliases}
    },
    key=lambda term: len(_normalize_match_text(term)),
    reverse=True,
)
_RELEVANCE_TERMS: dict[OledMineruRelevanceSignal, tuple[str, ...]] = {
    OledMineruRelevanceSignal.OLED_KEYWORD: ("OLED", "TADF", "organic light emitting diode"),
    OledMineruRelevanceSignal.DEVICE_KEYWORD: (
        "device",
        "device stack",
        "OLED stack",
        "EL",
        "electroluminescence",
        "roll-off",
        "roll off",
    ),
    OledMineruRelevanceSignal.PROPERTY_KEYWORD: tuple(_PROPERTY_TERMS),
    OledMineruRelevanceSignal.MEASUREMENT_KEYWORD: (
        "measured",
        "measurement",
        "luminance",
        "current density",
        "cd/m2",
        "cd/m²",
        "mA/cm2",
        "mA/cm²",
        "EQE (%)",
        "100 cd",
        "curves",
    ),
    OledMineruRelevanceSignal.MATERIAL_ROLE_KEYWORD: (
        "emitter",
        "host",
        "dopant",
        "doped",
        "film",
        "emissive",
    ),
    OledMineruRelevanceSignal.ENERGY_LEVEL_KEYWORD: (
        "HOMO",
        "LUMO",
        "S1",
        "T1",
        "ΔE_ST",
        "Delta EST",
        "singlet",
        "triplet",
    ),
    OledMineruRelevanceSignal.FABRICATION_KEYWORD: (
        "fabrication",
        "fabricated",
        "thermal evaporation",
        "spin coating",
        "solution processed",
    ),
}
_MD_CONTEXT_CHARS = 160


OledMineruCandidateKind = OledMineruCandidateType
OledMineruEvidenceCandidate = OledMineruCandidate
OledMineruCandidateExtractionReport = OledMineruCandidateSummary


__all__ = [
    "OledMineruCandidate",
    "OledMineruCandidateKind",
    "OledMineruCandidateSummary",
    "OledMineruCandidateType",
    "OledMineruCandidateExtractionReport",
    "OledMineruEvidenceCandidate",
    "OledMineruRelevanceSignal",
    "OledMineruSourceFormat",
    "OledMineruTableParseStatus",
    "detect_oled_mineru_source_format",
    "extract_oled_mineru_candidates",
    "extract_oled_mineru_candidates_from_document",
    "summarize_oled_mineru_candidates",
]
