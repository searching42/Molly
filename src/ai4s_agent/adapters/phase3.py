from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import re
import shlex
import shutil
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, safe_float, write_json
from ai4s_agent.adapters.claude_scripts import WORKSPACE
from ai4s_agent.adapters.runtime import run_argv_cmd
from ai4s_agent.data_layer import check_smiles_leakage
from ai4s_agent.schemas import (
    CorpusChunk,
    CorpusMultiIndex,
    CitationLicenseReport,
    ConflictGroup,
    ConflictReport,
    DenseRetrievalIndex,
    EvidenceHit,
    ExtractionBenchmarkReport,
    ExtractedRecord,
    ExtractionConfirmationRecord,
    ExtractionConfidenceReport,
    LiteratureCorpusManifest,
    LiteratureCorpusSource,
    LiteratureAcquisitionItem,
    LiteratureAcquisitionManifest,
    LiteratureSourceProvenance,
    MergedRecord,
    ParsedDocument,
    ParsedDocumentElement,
    ParsedTable,
    UnitNormalizationReport,
)


def _resolve_path(path_like: str, *, base: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ((base or WORKSPACE) / path).resolve()
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_first_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def _table_markdown(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not headers:
        return ""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _coerce_bbox(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _coerce_source_bbox(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        try:
            return {str(k): float(v) for k, v in value.items()}
        except (TypeError, ValueError):
            return None
    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    return {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}


def _extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _parsed_document_from_mineru_output(
    *,
    input_pdf: Path,
    output_dir: Path,
    mineru_output_dir: Path,
    parser_backend: str,
) -> ParsedDocument:
    md_path = _find_first_file(mineru_output_dir, (".md",))
    json_payload = _safe_json(_find_first_file(mineru_output_dir, (".json",)) or mineru_output_dir / "missing.json")
    markdown = md_path.read_text(encoding="utf-8", errors="ignore") if md_path else ""
    source_hash = _sha256_file(input_pdf)
    paper_id = input_pdf.stem

    pages_raw = json_payload.get("pages", []) if isinstance(json_payload.get("pages"), list) else []
    pages = [page for page in pages_raw if isinstance(page, dict)]

    elements: list[ParsedDocumentElement] = []
    raw_elements = json_payload.get("elements", []) if isinstance(json_payload.get("elements"), list) else []
    for index, raw in enumerate(raw_elements, start=1):
        if not isinstance(raw, dict):
            continue
        elements.append(
            ParsedDocumentElement(
                element_id=str(raw.get("element_id") or f"el_{index:04d}"),
                page=int(raw.get("page") or 1),
                type=str(raw.get("type") or "paragraph"),
                text=str(raw.get("text") or ""),
                markdown=str(raw.get("markdown") or raw.get("text") or ""),
                bbox=_coerce_bbox(raw.get("bbox")),
                source_hash=str(raw.get("source_hash") or source_hash),
                metadata={"source": "mineru_layout_json"},
            )
        )

    if not elements and markdown:
        elements.append(
            ParsedDocumentElement(
                element_id="el_0001",
                page=1,
                type="markdown",
                text=markdown,
                markdown=markdown,
                source_hash=source_hash,
                metadata={"source": str(md_path) if md_path else "mineru_markdown"},
            )
        )

    tables: list[ParsedTable] = []
    raw_tables = json_payload.get("tables", []) if isinstance(json_payload.get("tables"), list) else []
    for index, raw in enumerate(raw_tables, start=1):
        if not isinstance(raw, dict):
            continue
        headers = [str(item) for item in raw.get("headers", [])] if isinstance(raw.get("headers"), list) else []
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in raw.get("rows", [])
            if isinstance(row, dict)
        ] if isinstance(raw.get("rows"), list) else []
        table_md = str(raw.get("markdown") or "") or _table_markdown(headers, rows)
        tables.append(
            ParsedTable(
                table_id=str(raw.get("table_id") or f"table_{index:04d}"),
                caption=str(raw.get("caption") or ""),
                headers=headers,
                rows=rows,
                footnotes=[str(item) for item in raw.get("footnotes", [])] if isinstance(raw.get("footnotes"), list) else [],
                page=int(raw.get("page") or 1),
                markdown=table_md,
                source_bbox=_coerce_source_bbox(raw.get("source_bbox") or raw.get("bbox")),
            )
        )

    return ParsedDocument(
        paper_id=paper_id,
        source_path=str(input_pdf),
        parser_backend=parser_backend,
        metadata={
            "title": _extract_title(markdown, paper_id),
            "source_hash": source_hash,
            "mineru_output_dir": str(mineru_output_dir),
            "parsed_at": now_iso(),
        },
        pages=pages,
        elements=elements,
        tables=tables,
    )


def _write_markdown(path: Path, parsed: ParsedDocument) -> Path:
    lines = [f"# Parsed Document: {parsed.paper_id}", "", f"- Parser backend: `{parsed.parser_backend}`", f"- Source: `{parsed.source_path}`", f"- Elements: `{len(parsed.elements)}`", f"- Tables: `{len(parsed.tables)}`", ""]
    for table in parsed.tables[:10]:
        lines.append(f"## {table.table_id}")
        if table.caption:
            lines.append(table.caption)
        if table.markdown:
            lines.append(table.markdown)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def _write_parser_outputs(
    *,
    adapter: str,
    run_id: str,
    input_pdf: Path,
    output_dir: Path,
    parsed_doc: ParsedDocument,
    audit_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_json = write_json(output_dir / f"{run_id}_parsed_document.json", parsed_doc.model_dump(mode="json"))
    parsed_markdown = _write_markdown(output_dir / f"{run_id}_parsed_document.md", parsed_doc)
    audit = {
        "run_id": run_id,
        "adapter": adapter,
        "input_pdf": str(input_pdf),
        "outputs": {
            "parsed_document_json": str(parsed_json),
            "parsed_document_markdown": str(parsed_markdown),
        },
        "created_at": now_iso(),
    }
    if audit_extra:
        audit.update(audit_extra)
    audit_json = write_json(output_dir / f"{run_id}_parser_audit.json", audit)
    return {
        "status": "success",
        "adapter": adapter,
        "parsed_document": parsed_doc.model_dump(mode="json"),
        "outputs": {
            "parsed_document_json": str(parsed_json),
            "parsed_document_markdown": str(parsed_markdown),
            "parser_audit_json": str(audit_json),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _read_extracted_records(path: Path) -> list[ExtractedRecord]:
    return [ExtractedRecord.model_validate(row) for row in _read_jsonl(path)]


def _load_extracted_records_from_payload(payload: dict[str, Any]) -> list[ExtractedRecord]:
    path_values: list[str] = []
    list_value = payload.get("extracted_records_jsonl_list")
    if isinstance(list_value, list):
        path_values.extend(str(item).strip() for item in list_value if str(item).strip())
    single = str(payload.get("extracted_records_jsonl") or "").strip()
    if single:
        path_values.append(single)
    records: list[ExtractedRecord] = []
    for raw_path in path_values:
        records.extend(_read_extracted_records(_resolve_path(raw_path, base=WORKSPACE)))
    return records


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_@.+%-]+", text)]


def _property_id(header: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", header.strip().lower()).strip("_")
    if normalized.endswith("_percent"):
        normalized = normalized[: -len("_percent")]
    return normalized


def _is_smiles_header(header: str) -> bool:
    normalized = _property_id(header)
    return normalized in {"smiles", "canonical_smiles", "isomeric_smiles", "structure"}


def _excluded_property_header(header: str) -> bool:
    normalized = _property_id(header)
    return normalized in {
        "",
        "id",
        "no",
        "name",
        "compound",
        "compound_id",
        "molecule",
        "smiles",
        "canonical_smiles",
        "isomeric_smiles",
        "structure",
        "note",
        "notes",
        "reference",
        "source",
        "solvent",
        "host",
        "dopant",
    }


def _parse_markdown_table(markdown: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return [], []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not any(cells):
            continue
        row = {header: cells[index] if index < len(cells) else "" for index, header in enumerate(headers)}
        rows.append(row)
    return headers, rows


def _load_parsed_documents(payload: dict[str, Any]) -> list[ParsedDocument]:
    docs: list[ParsedDocument] = []
    parsed_json_raw = str(payload.get("parsed_document_json") or "").strip()
    if parsed_json_raw:
        parsed_json = _resolve_path(parsed_json_raw, base=WORKSPACE)
        loaded = _safe_json(parsed_json)
        if loaded:
            docs.append(ParsedDocument.model_validate(loaded))

    manifest_raw = str(payload.get("corpus_manifest_json") or "").strip()
    if manifest_raw:
        manifest = _safe_json(_resolve_path(manifest_raw, base=WORKSPACE))
        for item in manifest.get("documents", []) if isinstance(manifest.get("documents"), list) else []:
            if not isinstance(item, dict):
                continue
            path_raw = str(item.get("parsed_document_json") or "").strip()
            if not path_raw:
                continue
            loaded = _safe_json(_resolve_path(path_raw, base=WORKSPACE))
            if loaded:
                docs.append(ParsedDocument.model_validate(loaded))
    return docs


def _chunks_from_document(doc: ParsedDocument) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for element in doc.elements:
        text = element.text or element.markdown
        if not text.strip():
            continue
        chunks.append(
            CorpusChunk(
                chunk_id=f"{doc.paper_id}:{element.element_id}",
                source_id=doc.paper_id,
                paper_id=doc.paper_id,
                page=element.page,
                element_id=element.element_id,
                element_type=element.type,
                text=text,
                markdown=element.markdown,
                retrieval_channels=["bm25"],
                citation_context=f"{doc.paper_id} p.{element.page} {element.element_id}",
                metadata={"source_path": doc.source_path},
            )
        )
    for table in doc.tables:
        table_text = " ".join(
            [
                table.caption,
                " ".join(table.headers),
                " ".join(" ".join(row.values()) for row in table.rows),
                table.markdown,
            ]
        ).strip()
        chunks.append(
            CorpusChunk(
                chunk_id=f"{doc.paper_id}:{table.table_id}",
                source_id=doc.paper_id,
                paper_id=doc.paper_id,
                page=table.page,
                element_id=table.table_id,
                element_type="table",
                text=table_text,
                markdown=table.markdown,
                table_id=table.table_id,
                retrieval_channels=["bm25", "table"],
                citation_context=f"{doc.paper_id} p.{table.page} {table.table_id}",
                metadata={
                    "caption": table.caption,
                    "headers": table.headers,
                    "rows": table.rows,
                    "row_count": len(table.rows),
                },
            )
        )
    return chunks


def _bm25_scores(query: str, chunks: list[CorpusChunk]) -> list[tuple[CorpusChunk, float]]:
    query_terms = _tokens(query)
    if not query_terms or not chunks:
        return []
    doc_tokens = [_tokens(chunk.text + "\n" + chunk.markdown) for chunk in chunks]
    avg_len = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    scores: list[tuple[CorpusChunk, float]] = []
    k1 = 1.5
    b = 0.75
    corpus_size = len(chunks)
    for chunk, tokens in zip(chunks, doc_tokens, strict=False):
        tf = Counter(tokens)
        length = max(len(tokens), 1)
        score = 0.0
        for term in query_terms:
            if tf[term] <= 0:
                continue
            idf = math.log(1.0 + (corpus_size - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * length / max(avg_len, 1.0))
            score += idf * (tf[term] * (k1 + 1.0)) / denom
        if "table" in chunk.retrieval_channels and any(term in {"smiles", "plqy", "lambda", "property"} for term in query_terms):
            score *= 1.2
        if score > 0:
            scores.append((chunk, round(score, 6)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _add_index_term(indices: dict[str, dict[str, list[str]]], channel: str, term: str, chunk_id: str) -> None:
    clean = term.strip().lower()
    if not clean:
        return
    postings = indices.setdefault(channel, {}).setdefault(clean, [])
    if chunk_id not in postings:
        postings.append(chunk_id)


def _build_indices_for_chunk(indices: dict[str, dict[str, list[str]]], chunk: CorpusChunk) -> None:
    for token in _tokens(chunk.text + "\n" + chunk.markdown):
        _add_index_term(indices, "text", token, chunk.chunk_id)

    if chunk.element_type == "table":
        if chunk.table_id:
            _add_index_term(indices, "table", chunk.table_id, chunk.chunk_id)
        _add_index_term(indices, "table", chunk.element_id, chunk.chunk_id)
        raw_headers = chunk.metadata.get("headers", [])
        header_text = " ".join(str(item) for item in raw_headers) if isinstance(raw_headers, list) else ""
        for token in _tokens(f"{chunk.metadata.get('caption') or ''} {header_text}"):
            _add_index_term(indices, "table", token, chunk.chunk_id)

    headers = chunk.metadata.get("headers", [])
    rows = chunk.metadata.get("rows", [])
    clean_headers = [str(item) for item in headers] if isinstance(headers, list) else []
    smiles_header = next((header for header in clean_headers if _is_smiles_header(header)), "")
    for header in clean_headers:
        property_id = _property_id(header)
        if property_id and not _excluded_property_header(header):
            _add_index_term(indices, "property", property_id, chunk.chunk_id)
            for part in property_id.split("_"):
                _add_index_term(indices, "property", part, chunk.chunk_id)
    if smiles_header and isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            smiles = str(row.get(smiles_header) or "").strip()
            if smiles:
                _add_index_term(indices, "chemical", smiles, chunk.chunk_id)


def _load_multi_index(payload: dict[str, Any]) -> CorpusMultiIndex | None:
    raw = str(payload.get("multi_index_json") or "").strip()
    if not raw:
        return None
    loaded = _safe_json(_resolve_path(raw, base=WORKSPACE))
    if not loaded:
        return None
    return CorpusMultiIndex.model_validate(loaded)


def _multi_index_channels_for_query(query: str, chunk_id: str, multi_index: CorpusMultiIndex | None) -> list[str]:
    if multi_index is None:
        return []
    query_terms = {_property_id(token) for token in _tokens(query)}
    query_terms.update(token.lower() for token in re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/]+", query))
    channels: list[str] = []
    for channel, terms in multi_index.indices.items():
        for term in query_terms:
            if chunk_id in terms.get(term, []):
                channels.append(channel)
                break
    return sorted(channels)


def _write_multi_index_summary_md(path: Path, index: CorpusMultiIndex) -> Path:
    lines = [
        f"# Corpus Multi-Index: {index.run_id}",
        "",
        f"- Chunks: {index.chunk_count}",
        f"- Source chunks: `{index.chunks_jsonl}`",
        "",
        "| Channel | Terms |",
        "| --- | ---: |",
    ]
    for channel, count in sorted(index.channel_counts.items()):
        lines.append(f"| {channel} | {count} |")
    lines.extend(
        [
            "",
            "The multi-index is a deterministic local retrieval aid. BM25 remains available and dense retrieval is still a separate future adapter.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_multi_index_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    chunks_path_raw = str(payload.get("chunks_jsonl") or "").strip()
    index_raw = str(payload.get("corpus_index_json") or "").strip()
    if not chunks_path_raw and index_raw:
        index_payload = _safe_json(_resolve_path(index_raw, base=WORKSPACE))
        chunks_path_raw = str(index_payload.get("chunks_jsonl") or "").strip()
    if not run_id or not output_dir_raw or not chunks_path_raw:
        return {
            "status": "failed",
            "adapter": "build_multi_index",
            "error": {"code": "missing_required_fields", "message": "run_id/chunks_jsonl/output_dir are required"},
        }

    try:
        chunks_path = _resolve_path(chunks_path_raw, base=WORKSPACE)
        if not chunks_path.exists():
            raise ValueError(f"chunks_jsonl not found: {chunks_path}")
        chunks = [CorpusChunk.model_validate(item) for item in _read_jsonl(chunks_path)]
        if not chunks:
            raise ValueError("chunks_jsonl did not contain any chunks")
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "build_multi_index",
            "error": {"code": "invalid_chunks", "message": str(exc)},
        }
    indices: dict[str, dict[str, list[str]]] = {"text": {}, "property": {}, "table": {}, "chemical": {}}
    for chunk in chunks:
        _build_indices_for_chunk(indices, chunk)
    channel_counts = {channel: len(terms) for channel, terms in indices.items()}
    multi_index = CorpusMultiIndex(
        run_id=run_id,
        chunk_count=len(chunks),
        chunks_jsonl=str(chunks_path),
        indices=indices,
        channel_counts=channel_counts,
        created_at=now_iso(),
        notes=[
            "Channels: text tokens, normalized property headers, table identifiers, and SMILES values from table rows.",
            "Dense retrieval remains a separate optional retrieval layer.",
        ],
    )
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    multi_index_json = write_json(output_dir / f"{run_id}_multi_index.json", multi_index.model_dump(mode="json"))
    summary_md = _write_multi_index_summary_md(output_dir / f"{run_id}_multi_index_summary.md", multi_index)
    return {
        "status": "success",
        "adapter": "build_multi_index",
        "multi_index": multi_index.model_dump(mode="json"),
        "outputs": {
            "multi_index_json": str(multi_index_json),
            "multi_index_summary_md": str(summary_md),
        },
    }


_DENSE_SYNONYMS: dict[str, list[str]] = {
    "plqy": ["photoluminescence", "quantum", "yield", "efficiency"],
    "efficiency": ["plqy", "quantum", "yield"],
    "lambda": ["emission", "wavelength"],
    "lambda_em": ["emission", "wavelength"],
    "emission": ["lambda", "wavelength"],
    "smiles": ["molecule", "structure", "chemical"],
    "molecule": ["smiles", "structure", "chemical"],
}


def _dense_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _tokens(text):
        normalized = _property_id(token)
        if not normalized:
            continue
        terms.append(normalized)
        terms.extend(_DENSE_SYNONYMS.get(normalized, []))
        if "_" in normalized:
            for part in normalized.split("_"):
                if part:
                    terms.append(part)
                    terms.extend(_DENSE_SYNONYMS.get(part, []))
    return terms


def _hash_dense_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0 for _ in range(dimension)]
    for term in _dense_terms(text):
        digest = hashlib.sha1(term.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dimension
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0:
        return vector
    return [round(item / norm, 8) for item in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(limit))


def _normalize_backend_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"sentence_transformer", "sentence_transformers", "sbert"}:
        return "sentence_transformers"
    return normalized or "deterministic_hash_embedding"


def _to_float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(item) for item in vector]


def _sentence_transformer_vectors(texts: list[str], *, model_name: str, mode: str) -> list[list[float]]:
    module = _optional_import("sentence_transformers")
    if module is None:
        raise ImportError("sentence-transformers is not installed")
    model = module.SentenceTransformer(model_name)
    if mode == "query" and hasattr(model, "encode_query"):
        encoded = model.encode_query(texts, normalize_embeddings=True)
    elif mode == "document" and hasattr(model, "encode_document"):
        encoded = model.encode_document(texts, normalize_embeddings=True)
    else:
        encoded = model.encode(texts, normalize_embeddings=True)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [_to_float_vector(vector) for vector in encoded]


def _load_dense_index(payload: dict[str, Any]) -> DenseRetrievalIndex | None:
    raw = str(payload.get("dense_index_json") or "").strip()
    if not raw:
        return None
    loaded = _safe_json(_resolve_path(raw, base=WORKSPACE))
    if not loaded:
        return None
    return DenseRetrievalIndex.model_validate(loaded)


def _dense_scores(query: str, chunks: list[CorpusChunk], dense_index: DenseRetrievalIndex | None) -> dict[str, float]:
    if dense_index is None:
        return {}
    backend = _normalize_backend_name(dense_index.embedding_backend)
    if backend == "sentence_transformers":
        try:
            query_vector = _sentence_transformer_vectors(
                [query],
                model_name=dense_index.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
                mode="query",
            )[0]
        except Exception:
            return {}
    else:
        query_vector = _hash_dense_vector(query, dense_index.dimension)
    scores: dict[str, float] = {}
    for chunk in chunks:
        vector = dense_index.vectors.get(chunk.chunk_id)
        if not vector:
            continue
        score = _cosine(query_vector, vector)
        if score > 0:
            scores[chunk.chunk_id] = round(score, 6)
    return scores


def _write_dense_index_summary_md(path: Path, index: DenseRetrievalIndex) -> Path:
    lines = [
        f"# Dense Retrieval Index: {index.run_id}",
        "",
        f"- Chunks: {index.chunk_count}",
        f"- Dimension: {index.dimension}",
        f"- Embedding backend: `{index.embedding_backend}`",
        f"- Embedding model: `{index.embedding_model or 'n/a'}`",
        f"- Source chunks: `{index.chunks_jsonl}`",
        "",
        "The production backend is `sentence_transformers` when explicitly requested and installed; deterministic hash embeddings remain the offline fallback.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_dense_index_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    chunks_path_raw = str(payload.get("chunks_jsonl") or "").strip()
    index_raw = str(payload.get("corpus_index_json") or "").strip()
    if not chunks_path_raw and index_raw:
        index_payload = _safe_json(_resolve_path(index_raw, base=WORKSPACE))
        chunks_path_raw = str(index_payload.get("chunks_jsonl") or "").strip()
    if not run_id or not output_dir_raw or not chunks_path_raw:
        return {
            "status": "failed",
            "adapter": "build_dense_index",
            "error": {"code": "missing_required_fields", "message": "run_id/chunks_jsonl/output_dir are required"},
        }
    try:
        dimension = max(8, int(payload.get("dimension") or 64))
        chunks_path = _resolve_path(chunks_path_raw, base=WORKSPACE)
        if not chunks_path.exists():
            raise ValueError(f"chunks_jsonl not found: {chunks_path}")
        chunks = [CorpusChunk.model_validate(item) for item in _read_jsonl(chunks_path)]
        if not chunks:
            raise ValueError("chunks_jsonl did not contain any chunks")
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "build_dense_index",
            "error": {"code": "invalid_dense_index_inputs", "message": str(exc)},
        }

    embedding_backend = _normalize_backend_name(str(payload.get("embedding_backend") or payload.get("backend") or "deterministic_hash_embedding"))
    embedding_model = str(payload.get("embedding_model") or payload.get("model_name") or "").strip()
    chunk_texts: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        text = "\n".join(
            [
                chunk.text,
                chunk.markdown,
                str(chunk.metadata.get("caption") or ""),
                " ".join(str(item) for item in chunk.metadata.get("headers", []) if isinstance(chunk.metadata.get("headers", []), list)),
            ]
        )
        chunk_texts.append(text)
        metadata[chunk.chunk_id] = {
            "source_id": chunk.source_id,
            "paper_id": chunk.paper_id,
            "element_type": chunk.element_type,
            "element_id": chunk.element_id,
        }
    if embedding_backend == "sentence_transformers":
        embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            encoded_vectors = _sentence_transformer_vectors(chunk_texts, model_name=embedding_model, mode="document")
        except Exception as exc:
            return {
                "status": "failed",
                "adapter": "build_dense_index",
                "error": {
                    "code": "embedding_backend_unavailable",
                    "message": str(exc),
                },
            }
        vectors = {
            chunk.chunk_id: vector
            for chunk, vector in zip(chunks, encoded_vectors, strict=False)
        }
        dimension = len(next(iter(vectors.values()), [])) or dimension
    else:
        embedding_backend = "deterministic_hash_embedding"
        vectors = {
            chunk.chunk_id: _hash_dense_vector(text, dimension)
            for chunk, text in zip(chunks, chunk_texts, strict=False)
        }

    dense_index = DenseRetrievalIndex(
        run_id=run_id,
        chunk_count=len(chunks),
        chunks_jsonl=str(chunks_path),
        dimension=dimension,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        vectors=vectors,
        metadata=metadata,
        created_at=now_iso(),
        notes=[
            "sentence-transformers backend is supported when explicitly requested and installed; deterministic hash embedding remains the offline fallback.",
            "Use BM25/multi-index channels alongside dense retrieval for SMILES, exact property names, and units.",
        ],
    )
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    dense_index_json = write_json(output_dir / f"{run_id}_dense_index.json", dense_index.model_dump(mode="json"))
    summary_md = _write_dense_index_summary_md(output_dir / f"{run_id}_dense_index_summary.md", dense_index)
    return {
        "status": "success",
        "adapter": "build_dense_index",
        "dense_index": dense_index.model_dump(mode="json"),
        "outputs": {
            "dense_index_json": str(dense_index_json),
            "dense_index_summary_md": str(summary_md),
        },
    }


def _default_remote_tmp(run_id: str) -> str:
    return f"/tmp/ai4s_agent_mineru/{run_id}"


def _build_remote_mineru_command(*, remote_pdf: str, remote_output_dir: str, mineru_bin: str, api_url: str) -> list[str]:
    argv = [mineru_bin, "-p", remote_pdf, "-o", remote_output_dir]
    if api_url:
        argv.extend(["--api-url", api_url])
    return argv


def _parser_input_paths(payload: dict[str, Any], *, adapter: str) -> tuple[str, str, str] | dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_pdf_raw = str(payload.get("input_pdf") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not input_pdf_raw or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": adapter,
            "error": {"code": "missing_required_fields", "message": "run_id/input_pdf/output_dir are required"},
        }
    return run_id, input_pdf_raw, output_dir_raw


def _prepare_pdf_parser(payload: dict[str, Any], *, adapter: str) -> tuple[str, Path, Path] | dict[str, Any]:
    resolved = _parser_input_paths(payload, adapter=adapter)
    if isinstance(resolved, dict):
        return resolved
    run_id, input_pdf_raw, output_dir_raw = resolved
    input_pdf = _resolve_path(input_pdf_raw, base=WORKSPACE)
    if not input_pdf.exists() or not input_pdf.is_file():
        return {
            "status": "failed",
            "adapter": adapter,
            "error": {"code": "pdf_not_found", "message": f"input_pdf not found: {input_pdf}"},
        }
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    return run_id, input_pdf, output_dir


def _parsed_doc_metadata(input_pdf: Path, *, backend: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "title": input_pdf.stem,
        "source_hash": _sha256_file(input_pdf),
        "parser_backend": backend,
        "parsed_at": now_iso(),
    }
    if extra:
        metadata.update({key: value for key, value in extra.items() if value not in (None, "")})
    return metadata


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _pdfplumber_tables(page: Any, *, page_number: int) -> list[ParsedTable]:
    tables: list[ParsedTable] = []
    extract_tables = getattr(page, "extract_tables", None)
    if not callable(extract_tables):
        return tables
    for index, raw_table in enumerate(extract_tables() or [], start=1):
        if not raw_table:
            continue
        header_cells = [str(cell or "").strip() for cell in raw_table[0]]
        headers = [header or f"column_{pos + 1}" for pos, header in enumerate(header_cells)]
        rows: list[dict[str, str]] = []
        for raw_row in raw_table[1:]:
            row = {
                header: str(raw_row[pos] if pos < len(raw_row) and raw_row[pos] is not None else "").strip()
                for pos, header in enumerate(headers)
            }
            if any(value for value in row.values()):
                rows.append(row)
        tables.append(
            ParsedTable(
                table_id=f"table_p{page_number}_{index:04d}",
                caption="",
                headers=headers,
                rows=rows,
                page=page_number,
                markdown=_table_markdown(headers, rows),
            )
        )
    return tables


def parse_document_pdfplumber_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_pdf_parser(payload, adapter="parse_document_pdfplumber")
    if isinstance(prepared, dict):
        return prepared
    run_id, input_pdf, output_dir = prepared
    pdfplumber = _optional_import("pdfplumber")
    if pdfplumber is None:
        return {
            "status": "failed",
            "adapter": "parse_document_pdfplumber",
            "error": {
                "code": "missing_optional_dependency",
                "message": "pdfplumber is not installed; install it only when local table-oriented PDF fallback is needed.",
            },
        }

    source_hash = _sha256_file(input_pdf)
    elements: list[ParsedDocumentElement] = []
    tables: list[ParsedTable] = []
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(str(input_pdf)) as pdf:
        for index, page in enumerate(getattr(pdf, "pages", []), start=1):
            page_number = int(getattr(page, "page_number", index) or index)
            pages.append({"page": page_number})
            extract_text = getattr(page, "extract_text", None)
            text = str(extract_text() or "").strip() if callable(extract_text) else ""
            if text:
                elements.append(
                    ParsedDocumentElement(
                        element_id=f"el_p{page_number}_text",
                        page=page_number,
                        type="paragraph",
                        text=text,
                        markdown=text,
                        source_hash=source_hash,
                        metadata={"source": "pdfplumber.extract_text"},
                    )
                )
            tables.extend(_pdfplumber_tables(page, page_number=page_number))

    parsed_doc = ParsedDocument(
        paper_id=input_pdf.stem,
        source_path=str(input_pdf),
        parser_backend="pdfplumber_local",
        metadata=_parsed_doc_metadata(input_pdf, backend="pdfplumber_local"),
        pages=pages,
        elements=elements,
        tables=tables,
    )
    return _write_parser_outputs(
        adapter="parse_document_pdfplumber",
        run_id=run_id,
        input_pdf=input_pdf,
        output_dir=output_dir,
        parsed_doc=parsed_doc,
        audit_extra={"optional_dependency": "pdfplumber"},
    )


def parse_document_pymupdf_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_pdf_parser(payload, adapter="parse_document_pymupdf")
    if isinstance(prepared, dict):
        return prepared
    run_id, input_pdf, output_dir = prepared
    pymupdf = _optional_import("pymupdf") or _optional_import("fitz")
    if pymupdf is None:
        return {
            "status": "failed",
            "adapter": "parse_document_pymupdf",
            "error": {
                "code": "missing_optional_dependency",
                "message": "PyMuPDF is not installed; install it only when fast local text/page fallback is needed.",
            },
        }

    source_hash = _sha256_file(input_pdf)
    elements: list[ParsedDocumentElement] = []
    pages: list[dict[str, Any]] = []
    doc = pymupdf.open(str(input_pdf))
    try:
        for index, page in enumerate(doc, start=1):
            page_number = int(getattr(page, "number", index - 1) or 0) + 1
            pages.append({"page": page_number})
            text = str(page.get_text("text") or "").strip()
            if not text:
                continue
            elements.append(
                ParsedDocumentElement(
                    element_id=f"el_p{page_number}_text",
                    page=page_number,
                    type="paragraph",
                    text=text,
                    markdown=text,
                    source_hash=source_hash,
                    metadata={"source": "pymupdf.get_text"},
                )
            )
    finally:
        close = getattr(doc, "close", None)
        if callable(close):
            close()

    parsed_doc = ParsedDocument(
        paper_id=input_pdf.stem,
        source_path=str(input_pdf),
        parser_backend="pymupdf_local",
        metadata=_parsed_doc_metadata(input_pdf, backend="pymupdf_local"),
        pages=pages,
        elements=elements,
        tables=[],
    )
    return _write_parser_outputs(
        adapter="parse_document_pymupdf",
        run_id=run_id,
        input_pdf=input_pdf,
        output_dir=output_dir,
        parsed_doc=parsed_doc,
        audit_extra={"optional_dependency": "pymupdf"},
    )


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter_xml(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _xml_local_name(str(element.tag)) == name]


def _xml_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip())


def _first_xml_text(root: ET.Element, name: str, *, attr_name: str = "", attr_value: str = "") -> str:
    for element in _iter_xml(root, name):
        if attr_name and str(element.attrib.get(attr_name) or "").lower() != attr_value.lower():
            continue
        text = _xml_text(element)
        if text:
            return text
    return ""


def _parsed_document_from_grobid_tei(input_pdf: Path, tei_xml: Path) -> ParsedDocument:
    root = ET.fromstring(tei_xml.read_text(encoding="utf-8", errors="ignore"))
    source_hash = _sha256_file(input_pdf)
    title = _first_xml_text(root, "title") or input_pdf.stem
    doi = _first_xml_text(root, "idno", attr_name="type", attr_value="DOI")
    license_text = _first_xml_text(root, "licence")
    elements: list[ParsedDocumentElement] = []
    for index, paragraph in enumerate(_iter_xml(root, "p"), start=1):
        text = _xml_text(paragraph)
        if not text:
            continue
        elements.append(
            ParsedDocumentElement(
                element_id=f"el_tei_p{index:04d}",
                page=1,
                type="paragraph",
                text=text,
                markdown=text,
                source_hash=source_hash,
                metadata={"source": "grobid_tei"},
            )
        )

    tables: list[ParsedTable] = []
    for table_index, table in enumerate(_iter_xml(root, "table"), start=1):
        figure = None
        caption = ""
        for candidate in _iter_xml(root, "figure"):
            if table in list(candidate.iter()):
                figure = candidate
                break
        if figure is not None:
            caption = _first_xml_text(figure, "head")
        raw_rows: list[list[str]] = []
        header_from_role: list[str] = []
        for row in _iter_xml(table, "row"):
            cells = [_xml_text(cell) for cell in _iter_xml(row, "cell")]
            if not cells:
                continue
            if str(row.attrib.get("role") or "").lower() in {"label", "header"} and not header_from_role:
                header_from_role = cells
            else:
                raw_rows.append(cells)
        headers = header_from_role or (raw_rows.pop(0) if raw_rows else [])
        if not headers:
            continue
        rows: list[dict[str, str]] = []
        for raw_row in raw_rows:
            row = {
                str(header): str(raw_row[pos] if pos < len(raw_row) else "")
                for pos, header in enumerate(headers)
            }
            if any(row.values()):
                rows.append(row)
        tables.append(
            ParsedTable(
                table_id=f"table_tei_{table_index:04d}",
                caption=caption,
                headers=[str(header) for header in headers],
                rows=rows,
                page=1,
                markdown=_table_markdown([str(header) for header in headers], rows),
            )
        )

    return ParsedDocument(
        paper_id=input_pdf.stem,
        source_path=str(input_pdf),
        parser_backend="grobid_tei",
        metadata=_parsed_doc_metadata(
            input_pdf,
            backend="grobid_tei",
            extra={"title": title, "doi": doi, "license": license_text, "grobid_tei_xml": str(tei_xml)},
        ),
        pages=[{"page": 1}],
        elements=elements,
        tables=tables,
    )


def _call_grobid_service(input_pdf: Path, *, grobid_url: str, timeout_sec: int) -> str:
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    boundary = "ai4s-agent-grobid-boundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="input"; filename="{input_pdf.name}"\r\n'.encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            input_pdf.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_document_grobid_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_pdf_parser(payload, adapter="parse_document_grobid")
    if isinstance(prepared, dict):
        return prepared
    run_id, input_pdf, output_dir = prepared
    tei_raw = str(payload.get("grobid_tei_xml") or payload.get("tei_xml") or "").strip()
    execute = _as_bool(payload.get("execute", False))
    grobid_url = str(payload.get("grobid_url") or "http://localhost:8070").strip()
    if not tei_raw and not execute:
        return {
            "status": "planned",
            "adapter": "parse_document_grobid",
            "remote": {"backend": "grobid_service", "url": grobid_url, "endpoint": "/api/processFulltextDocument"},
            "note": "Set execute=true to call a GROBID service, or pass grobid_tei_xml to normalize existing TEI output.",
        }

    if execute and not tei_raw:
        try:
            tei_text = _call_grobid_service(input_pdf, grobid_url=grobid_url, timeout_sec=int(payload.get("timeout_sec") or 120))
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            return {
                "status": "failed",
                "adapter": "parse_document_grobid",
                "error": {"code": "grobid_service_failed", "message": str(exc)},
            }
        tei_path = output_dir / f"{run_id}_grobid.tei.xml"
        tei_path.write_text(tei_text, encoding="utf-8")
    else:
        tei_path = _resolve_path(tei_raw, base=WORKSPACE)
        if not tei_path.exists() or not tei_path.is_file():
            return {
                "status": "failed",
                "adapter": "parse_document_grobid",
                "error": {"code": "grobid_tei_missing", "message": f"grobid_tei_xml not found: {tei_path}"},
            }

    try:
        parsed_doc = _parsed_document_from_grobid_tei(input_pdf, tei_path)
    except ET.ParseError as exc:
        return {
            "status": "failed",
            "adapter": "parse_document_grobid",
            "error": {"code": "invalid_grobid_tei", "message": str(exc)},
        }
    return _write_parser_outputs(
        adapter="parse_document_grobid",
        run_id=run_id,
        input_pdf=input_pdf,
        output_dir=output_dir,
        parsed_doc=parsed_doc,
        audit_extra={"grobid_url": grobid_url, "grobid_tei_xml": str(tei_path)},
    )


def _source_id(source_type: str, value: str) -> str:
    digest = hashlib.sha1(f"{source_type}:{value}".encode("utf-8")).hexdigest()[:10]
    return f"{source_type}_{digest}"


def _looks_like_doi(value: str) -> bool:
    return bool(re.match(r"^10\.\d{4,9}/\S+$", value.strip(), flags=re.IGNORECASE))


def _append_literature_source(
    candidates: list[dict[str, Any]],
    *,
    source_type: str,
    value: str,
    raw: dict[str, Any] | None = None,
) -> None:
    clean_value = str(value or "").strip()
    clean_type = str(source_type or "").strip().lower()
    if not clean_type or not clean_value:
        return
    raw_payload = raw or {}
    local_path = str(raw_payload.get("local_path") or raw_payload.get("path") or "").strip()
    if clean_type == "uploaded_pdf_folder":
        local_path = local_path or clean_value
    status = "ready_local" if local_path and _resolve_path(local_path, base=WORKSPACE).exists() else "pending_acquisition"
    candidates.append(
        {
            "source_id": _source_id(clean_type, clean_value),
            "source_type": clean_type,
            "value": clean_value,
            "title": str(raw_payload.get("title") or ""),
            "url": str(raw_payload.get("url") or (clean_value if clean_type == "url" else "")),
            "doi": str(raw_payload.get("doi") or (clean_value if clean_type == "doi" else "")),
            "local_path": local_path,
            "license": str(raw_payload.get("license") or ""),
            "status": str(raw_payload.get("status") or status),
            "metadata": {str(k): v for k, v in raw_payload.items() if k not in {"source_id", "source_type"}},
        }
    )


def _append_raw_literature_source(candidates: list[dict[str, Any]], raw: Any) -> None:
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return
        if value.startswith(("http://", "https://")):
            _append_literature_source(candidates, source_type="url", value=value, raw={"url": value})
        elif _looks_like_doi(value):
            _append_literature_source(candidates, source_type="doi", value=value, raw={"doi": value})
        else:
            _append_literature_source(candidates, source_type="search_query", value=value, raw={"query": value})
        return
    if not isinstance(raw, dict):
        return

    explicit_type = str(raw.get("source_type") or raw.get("type") or "").strip().lower()
    if explicit_type:
        value = str(
            raw.get("value")
            or raw.get("url")
            or raw.get("doi")
            or raw.get("query")
            or raw.get("record_id")
            or raw.get("database")
            or raw.get("path")
            or ""
        ).strip()
        if explicit_type == "dataset_registry" and not raw.get("value"):
            value = ":".join(part for part in [str(raw.get("registry") or "").strip(), str(raw.get("record_id") or "").strip()] if part)
        if explicit_type == "external_database" and not raw.get("value"):
            value = ":".join(part for part in [str(raw.get("database") or "").strip(), str(raw.get("query") or "").strip()] if part)
        _append_literature_source(candidates, source_type=explicit_type, value=value, raw=raw)
        return

    if raw.get("url"):
        _append_literature_source(candidates, source_type="url", value=str(raw.get("url")), raw=raw)
    elif raw.get("doi"):
        _append_literature_source(candidates, source_type="doi", value=str(raw.get("doi")), raw=raw)
    elif raw.get("search_query") or raw.get("query"):
        _append_literature_source(candidates, source_type="search_query", value=str(raw.get("search_query") or raw.get("query")), raw=raw)
    elif raw.get("registry") or raw.get("dataset_registry"):
        value = ":".join(part for part in [str(raw.get("registry") or raw.get("dataset_registry") or "").strip(), str(raw.get("record_id") or "").strip()] if part)
        _append_literature_source(candidates, source_type="dataset_registry", value=value, raw=raw)
    elif raw.get("database") or raw.get("external_database"):
        value = ":".join(part for part in [str(raw.get("database") or raw.get("external_database") or "").strip(), str(raw.get("query") or "").strip()] if part)
        _append_literature_source(candidates, source_type="external_database", value=value, raw=raw)
    elif raw.get("input_pdf_dir") or raw.get("pdf_folder"):
        _append_literature_source(candidates, source_type="uploaded_pdf_folder", value=str(raw.get("input_pdf_dir") or raw.get("pdf_folder")), raw=raw)


def _append_list_sources(candidates: list[dict[str, Any]], payload: dict[str, Any], key: str, source_type: str) -> None:
    value = payload.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                raw = dict(item)
                raw.setdefault("source_type", source_type)
                _append_raw_literature_source(candidates, raw)
            else:
                _append_literature_source(candidates, source_type=source_type, value=str(item), raw={key: str(item)})
    elif isinstance(value, str) and value.strip():
        parts = [part.strip() for part in value.split(",") if part.strip()] if source_type in {"url", "doi"} else [value.strip()]
        for part in parts:
            _append_literature_source(candidates, source_type=source_type, value=part, raw={key: part})


def _write_corpus_source_manifest_md(path: Path, manifest: LiteratureCorpusManifest) -> Path:
    lines = [
        f"# Literature Corpus Source Manifest: {manifest.run_id}",
        "",
        f"- Sources: {manifest.source_count}",
        "",
        "| Source | Type | Status | Value |",
        "| --- | --- | --- | --- |",
    ]
    for source in manifest.sources:
        lines.append(f"| {source.source_id} | {source.source_type} | {source.status} | {source.value} |")
    lines.extend(
        [
            "",
            "This manifest records source intent only. Network acquisition, PDF download, and license review remain explicit downstream steps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare_literature_corpus_sources_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "prepare_literature_corpus_sources",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    candidates: list[dict[str, Any]] = []
    for raw in payload.get("sources", []) if isinstance(payload.get("sources"), list) else []:
        _append_raw_literature_source(candidates, raw)
    _append_list_sources(candidates, payload, "search_queries", "search_query")
    _append_list_sources(candidates, payload, "urls", "url")
    _append_list_sources(candidates, payload, "dois", "doi")
    _append_list_sources(candidates, payload, "dataset_registries", "dataset_registry")
    _append_list_sources(candidates, payload, "external_databases", "external_database")
    _append_list_sources(candidates, payload, "uploaded_pdf_dirs", "uploaded_pdf_folder")
    if payload.get("input_pdf_dir") or payload.get("pdf_folder"):
        _append_literature_source(
            candidates,
            source_type="uploaded_pdf_folder",
            value=str(payload.get("input_pdf_dir") or payload.get("pdf_folder")),
            raw={"input_pdf_dir": str(payload.get("input_pdf_dir") or payload.get("pdf_folder"))},
        )

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate.get("source_type") or ""), str(candidate.get("value") or ""))
        if key[0] and key[1] and key not in deduped:
            deduped[key] = candidate
    if not deduped:
        return {
            "status": "failed",
            "adapter": "prepare_literature_corpus_sources",
            "error": {"code": "no_sources", "message": "at least one literature source is required"},
        }

    try:
        sources = [LiteratureCorpusSource.model_validate(item) for item in deduped.values()]
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "prepare_literature_corpus_sources",
            "error": {"code": "invalid_literature_sources", "message": str(exc)},
        }
    type_counts = dict(sorted(Counter(source.source_type for source in sources).items()))
    manifest = LiteratureCorpusManifest(
        run_id=run_id,
        source_count=len(sources),
        source_type_counts=type_counts,
        sources=sources,
        created_at=now_iso(),
        notes=[
            "Source manifest supports search query, URL, DOI, dataset registry, external database, and local PDF-folder entries.",
            "Non-local sources are pending acquisition; this adapter does not fetch network content.",
        ],
    )
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    manifest_json = write_json(output_dir / f"{run_id}_corpus_source_manifest.json", manifest.model_dump(mode="json"))
    manifest_md = _write_corpus_source_manifest_md(output_dir / f"{run_id}_corpus_source_manifest.md", manifest)
    return {
        "status": "success",
        "adapter": "prepare_literature_corpus_sources",
        "corpus_source_manifest": manifest.model_dump(mode="json"),
        "outputs": {
            "corpus_source_manifest_json": str(manifest_json),
            "corpus_source_manifest_md": str(manifest_md),
        },
    }


def _local_mirror_lookup(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("local_mirror") or payload.get("local_mirrors") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(key).strip() and str(value).strip()}


def _acquisition_strategy(source: LiteratureCorpusSource) -> str:
    if source.source_type == "url":
        return "download_url"
    if source.source_type == "doi":
        return "resolve_doi_then_download_pdf"
    if source.source_type == "dataset_registry":
        return "fetch_dataset_registry_record"
    if source.source_type == "external_database":
        return "query_external_database"
    if source.source_type == "search_query":
        return "manual_or_api_search_then_select_sources"
    if source.source_type == "uploaded_pdf_folder":
        return "copy_local_pdf_folder"
    return "manual_review"


def _safe_acquired_name(source: LiteratureCorpusSource, src: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.source_id).strip("._-") or "source"
    suffix = src.suffix.lower() or ".dat"
    return f"{stem}{suffix}"


def _copy_acquired_file(source: LiteratureCorpusSource, src: Path, pdf_dir: Path, dataset_dir: Path) -> LiteratureAcquisitionItem:
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        dest_dir = pdf_dir
        acquisition_type = "pdf"
    elif suffix in {".csv", ".tsv", ".json", ".jsonl", ".sdf", ".smi"}:
        dest_dir = dataset_dir
        acquisition_type = "structured_dataset"
    else:
        dest_dir = dataset_dir
        acquisition_type = "file"
    _ensure_dir(dest_dir)
    dest = dest_dir / _safe_acquired_name(source, src)
    shutil.copy2(src, dest)
    return LiteratureAcquisitionItem(
        source_id=source.source_id,
        source_type=source.source_type,
        value=source.value,
        status="acquired",
        acquisition_type=acquisition_type,
        strategy="local_mirror",
        local_path=str(src),
        output_path=str(dest),
        metadata={"title": source.title, "license": source.license},
    )


def _copy_pdf_folder(source: LiteratureCorpusSource, folder: Path, pdf_dir: Path) -> list[LiteratureAcquisitionItem]:
    items: list[LiteratureAcquisitionItem] = []
    for pdf in sorted(folder.glob("*.pdf")):
        dest = pdf_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', source.source_id).strip('._-')}_{pdf.name}"
        _ensure_dir(dest.parent)
        shutil.copy2(pdf, dest)
        items.append(
            LiteratureAcquisitionItem(
                source_id=source.source_id,
                source_type=source.source_type,
                value=source.value,
                status="acquired",
                acquisition_type="pdf",
                strategy="copy_local_pdf_folder",
                local_path=str(pdf),
                output_path=str(dest),
                metadata={"folder": str(folder)},
            )
        )
    return items


def _planned_acquisition_item(source: LiteratureCorpusSource, message: str = "") -> LiteratureAcquisitionItem:
    return LiteratureAcquisitionItem(
        source_id=source.source_id,
        source_type=source.source_type,
        value=source.value,
        status="planned",
        acquisition_type="planned",
        strategy=_acquisition_strategy(source),
        local_path=source.local_path,
        message=message or "No local mirror was provided; acquisition requires explicit network/database execution or manual source selection.",
        metadata={"title": source.title, "url": source.url, "doi": source.doi, "license": source.license},
    )


def _write_acquisition_plan_md(path: Path, manifest: LiteratureAcquisitionManifest) -> Path:
    lines = [
        f"# Literature Source Acquisition: {manifest.run_id}",
        "",
        f"- Sources: {manifest.source_count}",
        f"- Acquired: {manifest.acquired_count}",
        f"- Planned: {manifest.planned_count}",
        f"- Failed: {manifest.failed_count}",
        f"- PDF dir: `{manifest.acquired_pdf_dir}`",
        f"- Dataset dir: `{manifest.acquired_dataset_dir}`",
        "",
        "| Source | Type | Status | Strategy | Output |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in manifest.items:
        lines.append(
            f"| {item.source_id} | {item.source_type} | {item.status} | "
            f"{item.strategy} | {item.output_path or item.message} |"
        )
    lines.extend(
        [
            "",
            "Network and external database acquisition should remain explicit because source licensing and upload/download limits vary by provider.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def acquire_literature_sources_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    manifest_raw = str(payload.get("corpus_source_manifest_json") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not manifest_raw or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "acquire_literature_sources",
            "error": {"code": "missing_required_fields", "message": "run_id/corpus_source_manifest_json/output_dir are required"},
        }
    try:
        source_manifest = LiteratureCorpusManifest.model_validate(_safe_json(_resolve_path(manifest_raw, base=WORKSPACE)))
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "acquire_literature_sources",
            "error": {"code": "invalid_corpus_source_manifest", "message": str(exc)},
        }

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    pdf_dir = _ensure_dir(output_dir / "pdfs")
    dataset_dir = _ensure_dir(output_dir / "datasets")
    mirror = _local_mirror_lookup(payload)
    items: list[LiteratureAcquisitionItem] = []

    for source in source_manifest.sources:
        mirror_path_raw = mirror.get(source.source_id) or mirror.get(source.value) or mirror.get(source.doi) or mirror.get(source.url)
        local_path_raw = mirror_path_raw or source.local_path
        if local_path_raw:
            local_path = _resolve_path(local_path_raw, base=WORKSPACE)
            if local_path.exists() and local_path.is_file():
                items.append(_copy_acquired_file(source, local_path, pdf_dir, dataset_dir))
                continue
            if local_path.exists() and local_path.is_dir():
                copied = _copy_pdf_folder(source, local_path, pdf_dir)
                if copied:
                    items.extend(copied)
                    continue
            items.append(
                LiteratureAcquisitionItem(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    value=source.value,
                    status="failed",
                    acquisition_type="local_mirror",
                    strategy="local_mirror",
                    local_path=str(local_path),
                    message="local mirror path does not exist or contains no PDFs",
                )
            )
            continue
        items.append(_planned_acquisition_item(source))

    acquired_count = sum(1 for item in items if item.status == "acquired")
    planned_count = sum(1 for item in items if item.status == "planned")
    failed_count = sum(1 for item in items if item.status == "failed")
    manifest = LiteratureAcquisitionManifest(
        run_id=run_id,
        source_count=len(source_manifest.sources),
        acquired_count=acquired_count,
        planned_count=planned_count,
        failed_count=failed_count,
        acquired_pdf_dir=str(pdf_dir),
        acquired_dataset_dir=str(dataset_dir),
        items=items,
        created_at=now_iso(),
        notes=[
            "Local mirrors are copied into run-scoped acquisition directories.",
            "Pending network/database entries are planned but not fetched by default.",
        ],
    )
    manifest_json = write_json(output_dir / f"{run_id}_literature_acquisition_manifest.json", manifest.model_dump(mode="json"))
    plan_md = _write_acquisition_plan_md(output_dir / f"{run_id}_literature_acquisition_plan.md", manifest)
    status = "success" if acquired_count and not planned_count and not failed_count else ("planned" if planned_count and not acquired_count and not failed_count else "degraded")
    return {
        "status": status,
        "adapter": "acquire_literature_sources",
        "acquisition_manifest": manifest.model_dump(mode="json"),
        "outputs": {
            "acquisition_manifest_json": str(manifest_json),
            "acquisition_plan_md": str(plan_md),
            "acquired_pdf_dir": str(pdf_dir),
            "acquired_dataset_dir": str(dataset_dir),
        },
    }


def parse_document_mineru_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_pdf_raw = str(payload.get("input_pdf") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not input_pdf_raw or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "parse_document_mineru",
            "error": {"code": "missing_required_fields", "message": "run_id/input_pdf/output_dir are required"},
        }

    input_pdf = _resolve_path(input_pdf_raw, base=WORKSPACE)
    if not input_pdf.exists() or not input_pdf.is_file():
        return {
            "status": "failed",
            "adapter": "parse_document_mineru",
            "error": {"code": "pdf_not_found", "message": f"input_pdf not found: {input_pdf}"},
        }
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))

    remote_host = str(payload.get("remote_host") or "workstation2").strip()
    remote_tmp = str(payload.get("remote_tmp") or _default_remote_tmp(run_id)).rstrip("/")
    remote_pdf = f"{remote_tmp}/{input_pdf.name}"
    remote_output_dir = f"{remote_tmp}/{input_pdf.stem}_output"
    mineru_bin = str(payload.get("mineru_bin") or "mineru").strip()
    api_url = str(payload.get("mineru_api_url") or payload.get("api_url") or "").strip()
    command = _build_remote_mineru_command(
        remote_pdf=remote_pdf,
        remote_output_dir=remote_output_dir,
        mineru_bin=mineru_bin,
        api_url=api_url,
    )
    remote = {
        "backend": "remote_cli",
        "host": remote_host,
        "remote_tmp": remote_tmp,
        "remote_pdf": remote_pdf,
        "remote_output_dir": remote_output_dir,
        "mineru_bin": mineru_bin,
        "api_url": api_url,
    }

    execute = _as_bool(payload.get("execute", False))
    mineru_output_raw = str(payload.get("mineru_output_dir") or "").strip()
    if not execute and not mineru_output_raw:
        return {
            "status": "planned",
            "adapter": "parse_document_mineru",
            "command": command,
            "remote": remote,
            "note": "Set execute=true to run MinerU on workstation2, or pass mineru_output_dir to normalize existing output.",
        }

    if execute:
        timeout_sec = int(payload.get("timeout_sec") or 1800)
        local_output = _ensure_dir(output_dir / "mineru_raw")
        prep = run_argv_cmd(argv=["ssh", remote_host, f"mkdir -p {shlex.quote(remote_tmp)}"], cwd=WORKSPACE, timeout_sec=60)
        if int(prep.get("returncode", 1)) != 0:
            return {"status": "failed", "adapter": "parse_document_mineru", "error": {"code": "remote_prepare_failed", "message": str(prep)}}
        upload = run_argv_cmd(argv=["scp", str(input_pdf), f"{remote_host}:{remote_pdf}"], cwd=WORKSPACE, timeout_sec=timeout_sec)
        if int(upload.get("returncode", 1)) != 0:
            return {"status": "failed", "adapter": "parse_document_mineru", "error": {"code": "remote_upload_failed", "message": str(upload)}}
        remote_cmd = " ".join(shlex.quote(part) for part in command)
        parsed = run_argv_cmd(argv=["ssh", remote_host, remote_cmd], cwd=WORKSPACE, timeout_sec=timeout_sec)
        if int(parsed.get("returncode", 1)) != 0:
            return {"status": "failed", "adapter": "parse_document_mineru", "error": {"code": "mineru_parse_failed", "message": str(parsed)}}
        fetched = run_argv_cmd(argv=["scp", "-r", f"{remote_host}:{remote_output_dir}", str(local_output)], cwd=WORKSPACE, timeout_sec=timeout_sec)
        if int(fetched.get("returncode", 1)) != 0:
            return {"status": "failed", "adapter": "parse_document_mineru", "error": {"code": "remote_fetch_failed", "message": str(fetched)}}
        mineru_output_dir = local_output
    else:
        mineru_output_dir = _resolve_path(mineru_output_raw, base=WORKSPACE)

    if not mineru_output_dir.exists() or not mineru_output_dir.is_dir():
        return {
            "status": "failed",
            "adapter": "parse_document_mineru",
            "error": {"code": "mineru_output_missing", "message": f"mineru_output_dir not found: {mineru_output_dir}"},
        }

    parsed_doc = _parsed_document_from_mineru_output(
        input_pdf=input_pdf,
        output_dir=output_dir,
        mineru_output_dir=mineru_output_dir,
        parser_backend="mineru_remote_cli",
    )
    parsed_json = write_json(output_dir / f"{run_id}_parsed_document.json", parsed_doc.model_dump(mode="json"))
    parsed_markdown = _write_markdown(output_dir / f"{run_id}_parsed_document.md", parsed_doc)
    audit = {
        "run_id": run_id,
        "adapter": "parse_document_mineru",
        "remote": remote,
        "input_pdf": str(input_pdf),
        "mineru_output_dir": str(mineru_output_dir),
        "outputs": {
            "parsed_document_json": str(parsed_json),
            "parsed_document_markdown": str(parsed_markdown),
        },
        "created_at": now_iso(),
    }
    audit_json = write_json(output_dir / f"{run_id}_parser_audit.json", audit)
    return {
        "status": "success",
        "adapter": "parse_document_mineru",
        "parsed_document": parsed_doc.model_dump(mode="json"),
        "remote": remote,
        "outputs": {
            "parsed_document_json": str(parsed_json),
            "parsed_document_markdown": str(parsed_markdown),
            "parser_audit_json": str(audit_json),
        },
    }


def parse_pdf_folder_mineru_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_dir_raw = str(payload.get("input_pdf_dir") or payload.get("pdf_folder") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not input_dir_raw or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "parse_pdf_folder_mineru",
            "error": {"code": "missing_required_fields", "message": "run_id/input_pdf_dir/output_dir are required"},
        }
    input_dir = _resolve_path(input_dir_raw, base=WORKSPACE)
    if not input_dir.exists() or not input_dir.is_dir():
        return {
            "status": "failed",
            "adapter": "parse_pdf_folder_mineru",
            "error": {"code": "pdf_folder_not_found", "message": f"input_pdf_dir not found: {input_dir}"},
        }
    pdfs = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    mineru_root_raw = str(payload.get("mineru_output_root") or "").strip()
    mineru_root = _resolve_path(mineru_root_raw, base=WORKSPACE) if mineru_root_raw else None

    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pdf in pdfs:
        child_payload = dict(payload)
        child_payload["run_id"] = f"{run_id}_{pdf.stem}"
        child_payload["input_pdf"] = str(pdf)
        child_payload["output_dir"] = str(output_dir / pdf.stem)
        if mineru_root is not None:
            child_payload["mineru_output_dir"] = str(mineru_root / pdf.stem)
        result = parse_document_mineru_adapter(child_payload)
        if result.get("status") == "success":
            parsed = result.get("parsed_document", {})
            outputs = result.get("outputs", {}) if isinstance(result.get("outputs"), dict) else {}
            documents.append(
                {
                    "paper_id": str(parsed.get("paper_id") or pdf.stem) if isinstance(parsed, dict) else pdf.stem,
                    "source_path": str(pdf),
                    "parsed_document_json": str(outputs.get("parsed_document_json") or ""),
                    "parser_audit_json": str(outputs.get("parser_audit_json") or ""),
                }
            )
        else:
            error = result.get("error", {}) if isinstance(result.get("error"), dict) else {}
            failures.append(
                {
                    "paper_id": pdf.stem,
                    "source_path": str(pdf),
                    "code": str(error.get("code") or "parse_failed"),
                    "message": str(error.get("message") or result),
                }
            )

    manifest = {
        "run_id": run_id,
        "source_type": "uploaded_pdf_folder",
        "input_pdf_dir": str(input_dir),
        "document_count": len(documents),
        "failed_count": len(failures),
        "documents": documents,
        "failures": failures,
        "created_at": now_iso(),
    }
    manifest_json = write_json(output_dir / f"{run_id}_corpus_manifest.json", manifest)
    return {
        "status": "success" if not failures else "degraded",
        "adapter": "parse_pdf_folder_mineru",
        "corpus_manifest": manifest,
        "outputs": {"corpus_manifest_json": str(manifest_json)},
    }


def index_corpus_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "index_corpus",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }
    docs = _load_parsed_documents(payload)
    if not docs:
        return {
            "status": "failed",
            "adapter": "index_corpus",
            "error": {"code": "no_parsed_documents", "message": "parsed_document_json or corpus_manifest_json is required"},
        }

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    chunks: list[CorpusChunk] = []
    for doc in docs:
        chunks.extend(_chunks_from_document(doc))

    chunks_jsonl = _write_jsonl(output_dir / f"{run_id}_evidence_chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    index_report = {
        "run_id": run_id,
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "channels": ["bm25", "table"],
        "chunks_jsonl": str(chunks_jsonl),
        "created_at": now_iso(),
    }
    corpus_index_json = write_json(output_dir / f"{run_id}_corpus_index.json", index_report)
    index_report_json = write_json(output_dir / f"{run_id}_index_report.json", index_report)
    return {
        "status": "success",
        "adapter": "index_corpus",
        "index_report": index_report,
        "outputs": {
            "chunks_jsonl": str(chunks_jsonl),
            "corpus_index_json": str(corpus_index_json),
            "index_report_json": str(index_report_json),
        },
    }


def retrieve_evidence_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    query = str(payload.get("query") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not query or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "retrieve_evidence",
            "error": {"code": "missing_required_fields", "message": "run_id/query/output_dir are required"},
        }

    chunks_path_raw = str(payload.get("chunks_jsonl") or "").strip()
    index_raw = str(payload.get("corpus_index_json") or "").strip()
    if not chunks_path_raw and index_raw:
        index_payload = _safe_json(_resolve_path(index_raw, base=WORKSPACE))
        chunks_path_raw = str(index_payload.get("chunks_jsonl") or "").strip()
    if not chunks_path_raw:
        return {
            "status": "failed",
            "adapter": "retrieve_evidence",
            "error": {"code": "missing_index", "message": "chunks_jsonl or corpus_index_json is required"},
        }

    chunks_payload = _read_jsonl(_resolve_path(chunks_path_raw, base=WORKSPACE))
    chunks = [CorpusChunk.model_validate(item) for item in chunks_payload]
    topk = max(1, int(payload.get("topk") or 5))
    multi_index = _load_multi_index(payload)
    dense_index = _load_dense_index(payload)
    dense_score_by_chunk = _dense_scores(query, chunks, dense_index)
    bm25_scores = _bm25_scores(query, chunks)
    bm25_chunk_ids = {chunk.chunk_id for chunk, _ in bm25_scores}
    score_by_chunk: dict[str, tuple[CorpusChunk, float]] = {
        chunk.chunk_id: (chunk, score)
        for chunk, score in bm25_scores
    }
    for chunk in chunks:
        channels = _multi_index_channels_for_query(query, chunk.chunk_id, multi_index)
        if not channels:
            continue
        _, existing_score = score_by_chunk.get(chunk.chunk_id, (chunk, 0.0))
        score_by_chunk[chunk.chunk_id] = (chunk, round(existing_score + 0.25 * len(channels), 6))
    for chunk in chunks:
        dense_score = dense_score_by_chunk.get(chunk.chunk_id, 0.0)
        if dense_score <= 0:
            continue
        _, existing_score = score_by_chunk.get(chunk.chunk_id, (chunk, 0.0))
        score_by_chunk[chunk.chunk_id] = (chunk, round(existing_score + dense_score, 6))
    scored = sorted(score_by_chunk.values(), key=lambda item: item[1], reverse=True)[:topk]
    hits: list[EvidenceHit] = []
    for chunk, score in scored:
        dense_score = dense_score_by_chunk.get(chunk.chunk_id, 0.0)
        if "table" in chunk.retrieval_channels and chunk.element_type == "table":
            channel = "table"
        elif dense_score > 0 and chunk.chunk_id not in bm25_chunk_ids:
            channel = "dense"
        else:
            channel = "bm25"
        multi_index_channels = _multi_index_channels_for_query(query, chunk.chunk_id, multi_index)
        hits.append(
            EvidenceHit(
                source_id=chunk.source_id,
                page=chunk.page,
                element_id=chunk.element_id,
                element_type=chunk.element_type,
                retrieval_channel=channel,
                score=score,
                text_or_table_ref=chunk.chunk_id,
                citation_context=chunk.citation_context,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "table_id": chunk.table_id or "",
                    "text_preview": chunk.text[:300],
                    "multi_index_channels": multi_index_channels,
                    "dense_score": dense_score,
                },
            )
        )

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    evidence_hits_json = write_json(
        output_dir / f"{run_id}_evidence_hits.json",
        {"run_id": run_id, "query": query, "hits": [hit.model_dump(mode="json") for hit in hits]},
    )
    retrieval_log_jsonl = _write_jsonl(
        output_dir / f"{run_id}_retrieval_log.jsonl",
        [
            {
                "run_id": run_id,
                "query": query,
                "channel": "bm25+dense" if dense_index is not None else "bm25",
                "topk": topk,
                "hit_count": len(hits),
                "created_at": now_iso(),
            }
        ],
    )
    return {
        "status": "success",
        "adapter": "retrieve_evidence",
        "query": query,
        "hits": [hit.model_dump(mode="json") for hit in hits],
        "outputs": {
            "evidence_hits_json": str(evidence_hits_json),
            "retrieval_log_jsonl": str(retrieval_log_jsonl),
        },
    }


def _load_evidence_hits(payload: dict[str, Any]) -> list[EvidenceHit]:
    hits_payload: Any = payload.get("hits")
    evidence_hits_raw = str(payload.get("evidence_hits_json") or "").strip()
    if evidence_hits_raw:
        loaded = _safe_json(_resolve_path(evidence_hits_raw, base=WORKSPACE))
        hits_payload = loaded.get("hits", [])
    if not isinstance(hits_payload, list):
        return []
    return [EvidenceHit.model_validate(item) for item in hits_payload if isinstance(item, dict)]


def _load_chunks_by_id(payload: dict[str, Any]) -> dict[str, CorpusChunk]:
    chunks_path_raw = str(payload.get("chunks_jsonl") or "").strip()
    index_raw = str(payload.get("corpus_index_json") or "").strip()
    if not chunks_path_raw and index_raw:
        index_payload = _safe_json(_resolve_path(index_raw, base=WORKSPACE))
        chunks_path_raw = str(index_payload.get("chunks_jsonl") or "").strip()
    if not chunks_path_raw:
        return {}
    chunks_payload = _read_jsonl(_resolve_path(chunks_path_raw, base=WORKSPACE))
    chunks = [CorpusChunk.model_validate(item) for item in chunks_payload]
    return {chunk.chunk_id: chunk for chunk in chunks}


def _table_rows_from_chunk(chunk: CorpusChunk) -> tuple[list[str], list[dict[str, str]]]:
    raw_headers = chunk.metadata.get("headers", [])
    headers = [str(item) for item in raw_headers] if isinstance(raw_headers, list) else []
    raw_rows = chunk.metadata.get("rows", [])
    rows = [
        {str(key): str(value) for key, value in row.items()}
        for row in raw_rows
        if isinstance(row, dict)
    ] if isinstance(raw_rows, list) else []
    if headers and rows:
        return headers, rows
    return _parse_markdown_table(chunk.markdown)


def _write_candidate_training_csv(path: Path, records: list[ExtractedRecord], confidence_threshold: float) -> Path:
    selected = [record for record in records if record.confidence >= confidence_threshold]
    property_order: list[str] = []
    for record in selected:
        for key in record.properties:
            if key not in property_order:
                property_order.append(key)
    fieldnames = [
        "smiles",
        *property_order,
        "record_id",
        "confidence",
        "source_id",
        "paper_id",
        "page",
        "table_id",
        "citation_context",
    ]
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected:
            row: dict[str, Any] = {
                "smiles": record.smiles,
                "record_id": record.record_id,
                "confidence": record.confidence,
                "source_id": record.source_id,
                "paper_id": record.paper_id,
                "page": record.page,
                "table_id": record.table_id,
                "citation_context": record.citation_context,
            }
            for key in property_order:
                row[key] = record.properties.get(key, "")
            writer.writerow(row)
    return path


def _write_extraction_summary(path: Path, report: ExtractionConfidenceReport) -> Path:
    lines = [
        f"# Extraction Summary: {report.run_id}",
        "",
        f"- Attempted evidence hits: {report.attempted_hit_count}",
        f"- Extracted records: {report.extracted_record_count}",
        f"- Rejected records: {report.rejected_record_count}",
        f"- High confidence records: {report.high_confidence_count}",
        f"- Confidence threshold: {report.confidence_threshold}",
        "",
        "This Phase 3 MVP extracts candidate records only. Records still require human confirmation before promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source_license_requires_review(license_value: str) -> bool:
    normalized = license_value.strip().lower()
    if normalized in {"", "unknown", "n/a", "na", "none", "not specified"}:
        return True
    normalized = normalized.replace("_", "-").replace(" ", "-")
    permissive = {
        "cc-by",
        "cc-by-4.0",
        "cc-by-3.0",
        "cc0",
        "cc0-1.0",
        "public-domain",
        "mit",
        "apache-2.0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
    }
    return normalized not in permissive


def _source_citation(doc: ParsedDocument) -> str:
    citation = str(doc.metadata.get("citation") or "").strip()
    if citation:
        return citation
    title = str(doc.metadata.get("title") or doc.paper_id).strip()
    doi = str(doc.metadata.get("doi") or "").strip()
    if doi:
        return f"{title}. DOI: {doi}"
    return title


def _write_audit_summary(path: Path, report: CitationLicenseReport) -> Path:
    lines = [
        f"# Literature Audit Summary: {report.run_id}",
        "",
        f"- Sources: {report.source_count}",
        f"- Evidence hits: {report.evidence_count}",
        f"- Extracted records: {report.extracted_record_count}",
        f"- Sources requiring license review: {report.unknown_license_count}",
        "",
        "| Source | Citation | License | Evidence | Records | Review |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for source in report.sources:
        review = "yes" if source.license_requires_review else "no"
        lines.append(
            f"| {source.paper_id} | {source.citation or source.title} | {source.license} | "
            f"{source.evidence_count} | {source.extracted_record_count} | {review} |"
        )
    lines.extend(
        [
            "",
            "Candidate records remain unconfirmed until a human reviews source licensing and extraction quality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _canonical_smiles_key(smiles: str) -> str:
    return re.sub(r"\s+", "", smiles).strip()


def _property_tolerance(payload: dict[str, Any], property_id: str) -> float:
    raw_map = payload.get("property_tolerances", {})
    if isinstance(raw_map, dict) and property_id in raw_map:
        try:
            return max(0.0, float(raw_map[property_id]))
        except (TypeError, ValueError):
            pass
    try:
        return max(0.0, float(payload.get("absolute_tolerance") if payload.get("absolute_tolerance") is not None else 0.05))
    except (TypeError, ValueError):
        return 0.05


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _write_merged_candidate_csv(path: Path, records: list[MergedRecord]) -> Path:
    selected = [record for record in records if record.status == "merged" and record.properties]
    property_order: list[str] = []
    for record in selected:
        for key in record.properties:
            if key not in property_order:
                property_order.append(key)
    fieldnames = ["smiles", *property_order, "merge_id", "confidence", "source_record_ids", "source_ids", "citations"]
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected:
            row: dict[str, Any] = {
                "smiles": record.smiles,
                "merge_id": record.merge_id,
                "confidence": record.confidence,
                "source_record_ids": ";".join(record.source_record_ids),
                "source_ids": ";".join(record.source_ids),
                "citations": " | ".join(record.citations),
            }
            for key in property_order:
                row[key] = record.properties.get(key, "")
            writer.writerow(row)
    return path


def _write_conflict_report_md(path: Path, report: ConflictReport) -> Path:
    lines = [
        f"# Conflict Report: {report.run_id}",
        "",
        f"- Input records: {report.input_record_count}",
        f"- Merged records: {report.merged_record_count}",
        f"- Non-conflicting records: {report.non_conflicting_record_count}",
        f"- Conflict groups: {report.conflict_count}",
        "",
    ]
    if report.conflicts:
        lines.extend(["| Conflict | SMILES | Property | Min | Max | Tolerance | Observations |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
        for conflict in report.conflicts:
            lines.append(
                f"| {conflict.conflict_id} | {conflict.smiles} | {conflict.property_id} | "
                f"{conflict.min_value} | {conflict.max_value} | {conflict.tolerance} | {len(conflict.observations)} |"
            )
    else:
        lines.append("No conflicts detected.")
    lines.extend(
        [
            "",
            "Conflicted values are excluded from the merged candidate training dataset until human review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _csv_record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _copy_text_file(src: Path, dest: Path) -> Path:
    _ensure_dir(dest.parent)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _write_confirmation_report(path: Path, record: ExtractionConfirmationRecord, blocking_reasons: list[str] | None = None) -> Path:
    lines = [
        f"# Extraction Confirmation: {record.run_id}",
        "",
        f"- Dataset ID: {record.dataset_id}",
        f"- Status: {record.status}",
        f"- Confirmed by: {record.confirmed_by}",
        f"- Confirmed at: {record.confirmed_at}",
        f"- Record count: {record.record_count}",
        f"- Conflict count: {record.conflict_count}",
        f"- Sources requiring license review: {record.unknown_license_count}",
        f"- Confirmed dataset: `{record.confirmed_dataset_path}`",
        "",
    ]
    if record.note:
        lines.extend(["## Note", record.note, ""])
    if blocking_reasons:
        lines.extend(["## Blocking Reasons", *[f"- {reason}" for reason in blocking_reasons], ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_required_confirmation_report(path_raw: str, *, required_key: str, label: str) -> tuple[dict[str, Any], str]:
    if not path_raw:
        return {}, f"missing_{label}"
    path = _resolve_path(path_raw, base=WORKSPACE)
    if not path.exists() or not path.is_file():
        return {}, f"missing_{label}"
    report = _safe_json(path)
    if required_key not in report:
        return report, f"invalid_{label}"
    return report, ""


def _raw_header_for_property(record: ExtractedRecord, property_id: str) -> str:
    for header in record.raw_values:
        if _property_id(header) == property_id:
            return header
    return property_id


def _normalize_property_value(property_id: str, source_header: str, value: float) -> tuple[str, float, str, str, str]:
    header_norm = _property_id(source_header)
    text = f"{source_header} {property_id}".lower()
    canonical_id = property_id
    canonical_value = float(value)
    source_unit = ""
    canonical_unit = ""
    rule = "identity"

    if "%" in source_header or "percent" in text or "百分" in text or property_id.endswith("_percent"):
        canonical_id = property_id.removesuffix("_percent")
        source_unit = "%"
        canonical_unit = "fraction"
        canonical_value = float(value) / 100.0 if abs(float(value)) > 1.0 else float(value)
        rule = "percent_to_fraction"
    elif header_norm.endswith("_nm") or property_id.endswith("_nm") or "(nm" in text or " nm" in text:
        canonical_id = property_id.removesuffix("_nm")
        source_unit = "nm"
        canonical_unit = "nm"
        rule = "strip_unit_suffix"

    return canonical_id, round(canonical_value, 12), source_unit, canonical_unit, rule


def _write_normalized_candidate_csv(path: Path, records: list[ExtractedRecord]) -> Path:
    property_order: list[str] = []
    for record in records:
        for key in record.properties:
            if key not in property_order:
                property_order.append(key)
    fieldnames = [
        "smiles",
        *property_order,
        "record_id",
        "confidence",
        "source_id",
        "paper_id",
        "page",
        "table_id",
        "citation_context",
    ]
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "smiles": record.smiles,
                "record_id": record.record_id,
                "confidence": record.confidence,
                "source_id": record.source_id,
                "paper_id": record.paper_id,
                "page": record.page,
                "table_id": record.table_id,
                "citation_context": record.citation_context,
            }
            for key in property_order:
                row[key] = record.properties.get(key, "")
            writer.writerow(row)
    return path


def _write_unit_normalization_report_md(path: Path, report: UnitNormalizationReport) -> Path:
    lines = [
        f"# Unit Normalization Report: {report.run_id}",
        "",
        f"- Input records: {report.input_record_count}",
        f"- Normalized records: {report.normalized_record_count}",
        f"- Conversions: {report.conversion_count}",
        f"- Warnings: {report.warning_count}",
        "",
        "| Record | Property | Source header | Source value | Source unit | Canonical property | Canonical value | Canonical unit | Rule |",
        "| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    if report.conversions:
        for item in report.conversions:
            lines.append(
                f"| {item.get('record_id', '')} | {item.get('property_id', '')} | {item.get('source_header', '')} | "
                f"{item.get('source_value', '')} | {item.get('source_unit', '')} | {item.get('canonical_property_id', '')} | "
                f"{item.get('canonical_value', '')} | {item.get('canonical_unit', '')} | {item.get('rule', '')} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |")
    if report.warnings:
        lines.extend(["", "## Warnings"])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "Normalized records remain candidate data and still require human confirmation before training dataset promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def normalize_extracted_units_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    records_raw = str(payload.get("extracted_records_jsonl") or "").strip()
    if not run_id or not output_dir_raw or not records_raw:
        return {
            "status": "failed",
            "adapter": "normalize_extracted_units",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/extracted_records_jsonl/output_dir are required",
            },
        }
    records_path = _resolve_path(records_raw, base=WORKSPACE)
    if not records_path.exists():
        return {
            "status": "failed",
            "adapter": "normalize_extracted_units",
            "error": {"code": "missing_extracted_records", "message": f"extracted_records_jsonl not found: {records_raw}"},
        }

    try:
        records = _read_extracted_records(records_path)
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "normalize_extracted_units",
            "error": {"code": "invalid_extracted_records", "message": str(exc)},
        }

    normalized_records: list[ExtractedRecord] = []
    conversions: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for record in records:
        properties: dict[str, float] = {}
        record_converted = False
        for property_id, value in record.properties.items():
            source_header = _raw_header_for_property(record, property_id)
            canonical_id, canonical_value, source_unit, canonical_unit, rule = _normalize_property_value(property_id, source_header, value)
            if canonical_id in properties and abs(properties[canonical_id] - canonical_value) > 1e-12:
                warnings.append(
                    {
                        "record_id": record.record_id,
                        "property_id": canonical_id,
                        "reason": "canonical_property_collision",
                        "existing_value": properties[canonical_id],
                        "new_value": canonical_value,
                    }
                )
                continue
            properties[canonical_id] = canonical_value
            if canonical_id != property_id or abs(canonical_value - float(value)) > 1e-12 or source_unit or canonical_unit:
                record_converted = True
                conversions.append(
                    {
                        "record_id": record.record_id,
                        "property_id": property_id,
                        "source_header": source_header,
                        "source_unit": source_unit,
                        "canonical_property_id": canonical_id,
                        "canonical_unit": canonical_unit,
                        "source_value": float(value),
                        "canonical_value": canonical_value,
                        "rule": rule,
                    }
                )
        data = record.model_dump(mode="json")
        data["properties"] = properties
        data["confidence_factors"] = {
            **record.confidence_factors,
            "unit_normalized": record_converted,
        }
        normalized_records.append(ExtractedRecord.model_validate(data))

    report = UnitNormalizationReport(
        run_id=run_id,
        input_record_count=len(records),
        normalized_record_count=len(normalized_records),
        conversion_count=len(conversions),
        warning_count=len(warnings),
        conversions=conversions,
        warnings=warnings,
        generated_at=now_iso(),
        notes=[
            "Phase 3 MVP handles common percent-to-fraction and explicit nm suffix normalization.",
            "Original raw values remain attached to each extracted record for audit.",
        ],
    )
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    normalized_jsonl = _write_jsonl(output_dir / f"{run_id}_normalized_extracted_records.jsonl", [record.model_dump(mode="json") for record in normalized_records])
    candidate_csv = _write_normalized_candidate_csv(output_dir / f"{run_id}_normalized_candidate_training_dataset.csv", normalized_records)
    report_json = write_json(output_dir / f"{run_id}_unit_normalization_report.json", report.model_dump(mode="json"))
    report_md = _write_unit_normalization_report_md(output_dir / f"{run_id}_unit_normalization_report.md", report)
    return {
        "status": "success",
        "adapter": "normalize_extracted_units",
        "records": [record.model_dump(mode="json") for record in normalized_records],
        "report": report.model_dump(mode="json"),
        "outputs": {
            "normalized_extracted_records_jsonl": str(normalized_jsonl),
            "normalized_candidate_training_dataset_csv": str(candidate_csv),
            "unit_normalization_report_json": str(report_json),
            "unit_normalization_report_md": str(report_md),
        },
    }


def extract_records_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "extract_records",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }
    if not payload.get("evidence_hits_json") and not payload.get("hits"):
        return {
            "status": "failed",
            "adapter": "extract_records",
            "error": {"code": "missing_evidence_hits", "message": "evidence_hits_json or hits is required"},
        }
    evidence_hits_raw = str(payload.get("evidence_hits_json") or "").strip()
    if evidence_hits_raw and not _resolve_path(evidence_hits_raw, base=WORKSPACE).exists():
        return {
            "status": "failed",
            "adapter": "extract_records",
            "error": {"code": "missing_evidence_hits", "message": f"evidence_hits_json not found: {evidence_hits_raw}"},
        }

    try:
        hits = _load_evidence_hits(payload)
        chunks_by_id = _load_chunks_by_id(payload)
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "extract_records",
            "error": {"code": "invalid_extraction_inputs", "message": str(exc)},
        }
    if not chunks_by_id:
        return {
            "status": "failed",
            "adapter": "extract_records",
            "error": {"code": "missing_chunks", "message": "chunks_jsonl or corpus_index_json is required"},
        }

    property_columns_raw = payload.get("property_columns", [])
    requested_properties = {
        _property_id(str(item))
        for item in property_columns_raw
    } if isinstance(property_columns_raw, list) else set()
    confidence_threshold = float(payload.get("confidence_threshold") or 0.7)
    records: list[ExtractedRecord] = []
    rejected: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for hit in hits:
        if hit.element_type != "table":
            attempts.append(
                {
                    "evidence_ref": hit.text_or_table_ref,
                    "status": "skipped",
                    "reason": "non_table_evidence",
                    "created_at": now_iso(),
                }
            )
            continue
        chunk_id = str(hit.metadata.get("chunk_id") or hit.text_or_table_ref)
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            rejected.append(
                {
                    "evidence_ref": hit.text_or_table_ref,
                    "reason": "chunk_not_found",
                    "citation_context": hit.citation_context,
                }
            )
            continue
        headers, rows = _table_rows_from_chunk(chunk)
        if not headers or not rows:
            rejected.append(
                {
                    "evidence_ref": hit.text_or_table_ref,
                    "reason": "no_table_rows",
                    "citation_context": hit.citation_context,
                }
            )
            continue
        smiles_header = next((header for header in headers if _is_smiles_header(header)), "")
        for row_index, row in enumerate(rows):
            attempt = {
                "evidence_ref": hit.text_or_table_ref,
                "source_id": hit.source_id,
                "page": hit.page,
                "table_id": chunk.table_id or str(hit.metadata.get("table_id") or ""),
                "row_index": row_index,
                "raw_values": row,
                "created_at": now_iso(),
            }
            smiles = str(row.get(smiles_header, "")).strip() if smiles_header else ""
            properties: dict[str, float] = {}
            for header in headers:
                property_id = _property_id(header)
                if requested_properties and property_id not in requested_properties:
                    continue
                if _excluded_property_header(header):
                    continue
                value = safe_float(row.get(header))
                if value is not None:
                    properties[property_id] = value

            if not smiles:
                reason = "missing_smiles"
                rejected.append({**attempt, "reason": reason})
                attempts.append({**attempt, "status": "rejected", "reason": reason})
                continue
            if not properties:
                reason = "no_numeric_properties"
                rejected.append({**attempt, "smiles": smiles, "reason": reason})
                attempts.append({**attempt, "status": "rejected", "reason": reason})
                continue

            confidence_factors = {
                "has_smiles": True,
                "numeric_property_count": len(properties),
                "has_table_citation": bool(hit.citation_context),
                "retrieval_channel": hit.retrieval_channel,
                "source": "table_rule_extractor",
            }
            confidence = min(
                0.95,
                0.45
                + 0.20
                + min(0.20, 0.10 * len(properties))
                + (0.15 if hit.citation_context else 0.0)
                + (0.10 if hit.retrieval_channel == "table" else 0.0),
            )
            record = ExtractedRecord(
                record_id=f"rec_{len(records) + 1:06d}",
                smiles=smiles,
                properties=properties,
                source_id=hit.source_id,
                paper_id=chunk.paper_id,
                page=hit.page,
                table_id=chunk.table_id or str(hit.metadata.get("table_id") or ""),
                row_index=row_index,
                evidence_ref=hit.text_or_table_ref,
                citation_context=hit.citation_context,
                confidence=round(confidence, 3),
                confidence_factors=confidence_factors,
                raw_values={str(key): str(value) for key, value in row.items()},
            )
            records.append(record)
            attempts.append({**attempt, "status": "extracted", "record_id": record.record_id, "confidence": record.confidence})

    high_count = sum(1 for record in records if record.confidence >= confidence_threshold)
    medium_count = sum(1 for record in records if 0.5 <= record.confidence < confidence_threshold)
    low_count = sum(1 for record in records if record.confidence < 0.5)
    report = ExtractionConfidenceReport(
        run_id=run_id,
        attempted_hit_count=len(hits),
        extracted_record_count=len(records),
        rejected_record_count=len(rejected),
        high_confidence_count=high_count,
        medium_confidence_count=medium_count,
        low_confidence_count=low_count,
        confidence_threshold=confidence_threshold,
        generated_at=now_iso(),
        notes=[
            "Rule-based Phase 3.3 MVP; table rows only.",
            "Candidate training dataset includes records at or above the confidence threshold.",
        ],
    )

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    extracted_records_jsonl = _write_jsonl(output_dir / f"{run_id}_extracted_records.jsonl", [record.model_dump(mode="json") for record in records])
    rejected_records_jsonl = _write_jsonl(output_dir / f"{run_id}_rejected_records.jsonl", rejected)
    extraction_attempts_jsonl = _write_jsonl(output_dir / f"{run_id}_extraction_attempts.jsonl", attempts)
    confidence_report_json = write_json(output_dir / f"{run_id}_extraction_confidence_report.json", report.model_dump(mode="json"))
    candidate_training_dataset_csv = _write_candidate_training_csv(output_dir / f"{run_id}_candidate_training_dataset.csv", records, confidence_threshold)
    extraction_summary_md = _write_extraction_summary(output_dir / f"{run_id}_extraction_summary.md", report)

    return {
        "status": "success",
        "adapter": "extract_records",
        "records": [record.model_dump(mode="json") for record in records],
        "rejected_records": rejected,
        "confidence_report": report.model_dump(mode="json"),
        "outputs": {
            "extracted_records_jsonl": str(extracted_records_jsonl),
            "rejected_records_jsonl": str(rejected_records_jsonl),
            "extraction_attempts_jsonl": str(extraction_attempts_jsonl),
            "extraction_confidence_report_json": str(confidence_report_json),
            "candidate_training_dataset_csv": str(candidate_training_dataset_csv),
            "extraction_summary_md": str(extraction_summary_md),
        },
    }


def track_citation_provenance_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "track_citation_provenance",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    docs = _load_parsed_documents(payload)
    if not docs:
        return {
            "status": "failed",
            "adapter": "track_citation_provenance",
            "error": {"code": "no_parsed_documents", "message": "parsed_document_json or corpus_manifest_json is required"},
        }

    try:
        hits = _load_evidence_hits(payload) if (payload.get("evidence_hits_json") or payload.get("hits")) else []
        records_path_raw = str(payload.get("extracted_records_jsonl") or "").strip()
        records = _read_extracted_records(_resolve_path(records_path_raw, base=WORKSPACE)) if records_path_raw else []
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "track_citation_provenance",
            "error": {"code": "invalid_provenance_inputs", "message": str(exc)},
        }

    license_overrides_raw = payload.get("license_overrides", {})
    license_overrides = license_overrides_raw if isinstance(license_overrides_raw, dict) else {}
    default_license = str(payload.get("default_license") or "").strip()
    evidence_counts = Counter(hit.source_id for hit in hits)
    record_counts = Counter(record.source_id for record in records)

    sources: list[LiteratureSourceProvenance] = []
    for doc in docs:
        override_license = str(license_overrides.get(doc.paper_id) or license_overrides.get(doc.source_path) or "").strip()
        license_value = override_license or str(doc.metadata.get("license") or default_license or "unknown").strip()
        requires_review = _source_license_requires_review(license_value)
        title = str(doc.metadata.get("title") or doc.paper_id).strip()
        source = LiteratureSourceProvenance(
            source_id=doc.paper_id,
            paper_id=doc.paper_id,
            title=title,
            source_path=doc.source_path,
            source_hash=str(doc.metadata.get("source_hash") or ""),
            parser_backend=doc.parser_backend,
            citation=_source_citation(doc),
            doi=str(doc.metadata.get("doi") or ""),
            license=license_value,
            license_requires_review=requires_review,
            evidence_count=int(evidence_counts.get(doc.paper_id, 0)),
            extracted_record_count=int(record_counts.get(doc.paper_id, 0)),
            metadata={
                "parsed_at": doc.metadata.get("parsed_at", ""),
                "license_source": "override" if override_license else ("metadata" if doc.metadata.get("license") else "default_or_unknown"),
            },
        )
        sources.append(source)

    report = CitationLicenseReport(
        run_id=run_id,
        source_count=len(sources),
        evidence_count=len(hits),
        extracted_record_count=len(records),
        unknown_license_count=sum(1 for source in sources if source.license_requires_review),
        sources=sources,
        generated_at=now_iso(),
        notes=[
            "License values are tracked for review; this adapter does not grant reuse permission.",
            "Candidate records remain blocked from confirmed training dataset promotion until human confirmation.",
        ],
    )

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    report_json = write_json(output_dir / f"{run_id}_citation_provenance_report.json", report.model_dump(mode="json"))
    audit_summary_md = _write_audit_summary(output_dir / f"{run_id}_audit_summary.md", report)
    return {
        "status": "success",
        "adapter": "track_citation_provenance",
        "report": report.model_dump(mode="json"),
        "outputs": {
            "citation_provenance_report_json": str(report_json),
            "audit_summary_md": str(audit_summary_md),
        },
    }


def merge_extracted_records_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "merge_extracted_records",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }
    if not payload.get("extracted_records_jsonl") and not payload.get("extracted_records_jsonl_list"):
        return {
            "status": "failed",
            "adapter": "merge_extracted_records",
            "error": {"code": "missing_extracted_records", "message": "extracted_records_jsonl or extracted_records_jsonl_list is required"},
        }
    records_paths_raw: list[str] = []
    list_value = payload.get("extracted_records_jsonl_list")
    if isinstance(list_value, list):
        records_paths_raw.extend(str(item).strip() for item in list_value if str(item).strip())
    single_value = str(payload.get("extracted_records_jsonl") or "").strip()
    if single_value:
        records_paths_raw.append(single_value)
    for records_path_raw in records_paths_raw:
        if not _resolve_path(records_path_raw, base=WORKSPACE).exists():
            return {
                "status": "failed",
                "adapter": "merge_extracted_records",
                "error": {"code": "missing_extracted_records", "message": f"extracted records not found: {records_path_raw}"},
            }

    try:
        records = _load_extracted_records_from_payload(payload)
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "merge_extracted_records",
            "error": {"code": "invalid_extracted_records", "message": str(exc)},
        }

    grouped: dict[str, list[ExtractedRecord]] = {}
    display_smiles: dict[str, str] = {}
    for record in records:
        key = _canonical_smiles_key(record.smiles)
        if not key:
            continue
        grouped.setdefault(key, []).append(record)
        display_smiles.setdefault(key, record.smiles)

    conflicts: list[ConflictGroup] = []
    merged_records: list[MergedRecord] = []
    for smiles_key in sorted(grouped):
        group = grouped[smiles_key]
        property_ids = sorted({property_id for record in group for property_id in record.properties})
        merged_properties: dict[str, float] = {}
        property_status: dict[str, str] = {}
        conflict_ids: list[str] = []

        for property_id in property_ids:
            observations = [
                {
                    "record_id": record.record_id,
                    "source_id": record.source_id,
                    "paper_id": record.paper_id,
                    "citation_context": record.citation_context,
                    "confidence": record.confidence,
                    "value": record.properties[property_id],
                }
                for record in group
                if property_id in record.properties
            ]
            values = [float(item["value"]) for item in observations]
            if not values:
                continue
            min_value = min(values)
            max_value = max(values)
            tolerance = _property_tolerance(payload, property_id)
            if max_value - min_value > tolerance:
                conflict = ConflictGroup(
                    conflict_id=f"conflict_{len(conflicts) + 1:06d}",
                    smiles=display_smiles[smiles_key],
                    property_id=property_id,
                    min_value=round(min_value, 6),
                    max_value=round(max_value, 6),
                    tolerance=tolerance,
                    observations=observations,
                )
                conflicts.append(conflict)
                conflict_ids.append(conflict.conflict_id)
                property_status[property_id] = "conflict"
            else:
                merged_properties[property_id] = _mean(values)
                property_status[property_id] = "merged" if len(values) > 1 else "single_source"

        confidence = _mean([record.confidence for record in group])
        status = "conflict" if conflict_ids else "merged"
        merged_records.append(
            MergedRecord(
                merge_id=f"merge_{len(merged_records) + 1:06d}",
                smiles=display_smiles[smiles_key],
                properties=merged_properties,
                property_status=property_status,
                source_record_ids=sorted({record.record_id for record in group}),
                source_ids=sorted({record.source_id for record in group}),
                citations=sorted({record.citation_context for record in group if record.citation_context}),
                confidence=confidence,
                conflict_ids=conflict_ids,
                status=status,
            )
        )

    report = ConflictReport(
        run_id=run_id,
        input_record_count=len(records),
        merged_record_count=len(merged_records),
        conflict_count=len(conflicts),
        non_conflicting_record_count=sum(1 for record in merged_records if record.status == "merged"),
        conflicts=conflicts,
        generated_at=now_iso(),
        notes=[
            "Records are grouped by exact normalized SMILES.",
            "Conflicted values are excluded from candidate training CSV pending human review.",
        ],
    )

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    merged_records_jsonl = _write_jsonl(output_dir / f"{run_id}_merged_records.jsonl", [record.model_dump(mode="json") for record in merged_records])
    conflict_report_json = write_json(output_dir / f"{run_id}_conflict_report.json", report.model_dump(mode="json"))
    conflict_report_md = _write_conflict_report_md(output_dir / f"{run_id}_conflict_report.md", report)
    candidate_csv = _write_merged_candidate_csv(output_dir / f"{run_id}_merged_candidate_training_dataset.csv", merged_records)

    return {
        "status": "success",
        "adapter": "merge_extracted_records",
        "merged_records": [record.model_dump(mode="json") for record in merged_records],
        "conflict_report": report.model_dump(mode="json"),
        "outputs": {
            "merged_records_jsonl": str(merged_records_jsonl),
            "conflict_report_json": str(conflict_report_json),
            "conflict_report_md": str(conflict_report_md),
            "candidate_training_dataset_csv": str(candidate_csv),
        },
    }


def confirm_extracted_dataset_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    candidate_raw = str(
        payload.get("candidate_training_dataset_csv")
        or payload.get("merged_candidate_training_dataset_csv")
        or ""
    ).strip()
    actor = str(payload.get("actor") or payload.get("approved_by") or "").strip()
    confirmed = _as_bool(payload.get("confirmed", False))
    if not run_id or not output_dir_raw or not candidate_raw:
        return {
            "status": "failed",
            "adapter": "confirm_extracted_dataset",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/output_dir/candidate_training_dataset_csv are required",
            },
        }
    if not confirmed or not actor:
        return {
            "status": "failed",
            "adapter": "confirm_extracted_dataset",
            "error": {
                "code": "confirmation_required",
                "message": "confirmed=true and actor/approved_by are required before dataset promotion",
            },
        }

    candidate_csv = _resolve_path(candidate_raw, base=WORKSPACE)
    if not candidate_csv.exists() or not candidate_csv.is_file():
        return {
            "status": "failed",
            "adapter": "confirm_extracted_dataset",
            "error": {"code": "candidate_dataset_missing", "message": f"candidate dataset not found: {candidate_csv}"},
        }

    conflict_report_raw = str(payload.get("conflict_report_json") or "").strip()
    provenance_report_raw = str(payload.get("citation_provenance_report_json") or "").strip()
    conflict_report, conflict_report_error = _load_required_confirmation_report(
        conflict_report_raw,
        required_key="conflict_count",
        label="conflict_report",
    )
    provenance_report, provenance_report_error = _load_required_confirmation_report(
        provenance_report_raw,
        required_key="unknown_license_count",
        label="citation_provenance_report",
    )
    conflict_count = int(conflict_report.get("conflict_count") or 0)
    unknown_license_count = int(provenance_report.get("unknown_license_count") or 0)

    blocking_reasons: list[str] = []
    if conflict_report_error:
        blocking_reasons.append(conflict_report_error)
    if provenance_report_error:
        blocking_reasons.append(provenance_report_error)
    if conflict_count > 0 and not _as_bool(payload.get("allow_conflicts", False)):
        blocking_reasons.append("unresolved_conflicts")
    if unknown_license_count > 0 and not _as_bool(payload.get("allow_license_review", False)):
        blocking_reasons.append("license_review_required")
    if blocking_reasons:
        return {
            "status": "failed",
            "adapter": "confirm_extracted_dataset",
            "error": {
                "code": "confirmation_blocked",
                "message": "candidate dataset cannot be confirmed until blocking review items are resolved",
                "blocking_reasons": blocking_reasons,
                "conflict_count": conflict_count,
                "unknown_license_count": unknown_license_count,
            },
        }

    dataset_id = str(payload.get("dataset_id") or f"{run_id}_confirmed_training_dataset").strip()
    safe_dataset_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("._-") or "confirmed_training_dataset"
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    confirmed_csv = _copy_text_file(candidate_csv, output_dir / f"{safe_dataset_id}.csv")
    record = ExtractionConfirmationRecord(
        run_id=run_id,
        dataset_id=safe_dataset_id,
        source_dataset_path=str(candidate_csv),
        confirmed_dataset_path=str(confirmed_csv),
        confirmed_by=actor,
        confirmed_at=now_iso(),
        record_count=_csv_record_count(confirmed_csv),
        conflict_count=conflict_count,
        unknown_license_count=unknown_license_count,
        source_reports={
            "conflict_report_json": str(_resolve_path(conflict_report_raw, base=WORKSPACE)) if conflict_report_raw else "",
            "citation_provenance_report_json": str(_resolve_path(provenance_report_raw, base=WORKSPACE)) if provenance_report_raw else "",
        },
        note=str(payload.get("note") or ""),
        status="confirmed",
    )
    confirmation_json = write_json(output_dir / f"{safe_dataset_id}_confirmation_record.json", record.model_dump(mode="json"))
    confirmation_md = _write_confirmation_report(output_dir / f"{safe_dataset_id}_human_confirmation_report.md", record)

    return {
        "status": "success",
        "adapter": "confirm_extracted_dataset",
        "confirmation_record": record.model_dump(mode="json"),
        "outputs": {
            "confirmed_training_dataset_csv": str(confirmed_csv),
            "confirmation_record_json": str(confirmation_json),
            "human_confirmation_report_md": str(confirmation_md),
        },
    }


def _write_literature_workflow_summary(path: Path, report: dict[str, Any]) -> Path:
    lines = [
        f"# Literature-To-Dataset Workflow: {report.get('run_id', '')}",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Pending confirmation: {report.get('pending_confirmation', False)}",
        "",
        "## Stages",
    ]
    stage_statuses = report.get("stage_statuses", {})
    if isinstance(stage_statuses, dict):
        for stage, status in stage_statuses.items():
            lines.append(f"- {stage}: {status}")
    outputs = report.get("outputs", {})
    if isinstance(outputs, dict) and outputs:
        lines.extend(["", "## Key Outputs"])
        for key in sorted(outputs):
            lines.append(f"- {key}: `{outputs[key]}`")
    error = report.get("error")
    if isinstance(error, dict) and error:
        lines.extend(["", "## Error", f"- {error.get('code', '')}: {error.get('message', '')}"])
    lines.extend(
        [
            "",
            "Candidate datasets produced by this workflow remain unconfirmed unless the confirmation stage succeeds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _finalize_literature_workflow_report(
    *,
    output_dir: Path,
    run_id: str,
    status: str,
    stage_statuses: dict[str, str],
    outputs: dict[str, str],
    pending_confirmation: bool,
    error: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    report = {
        "run_id": run_id,
        "status": status,
        "stage_order": list(stage_statuses),
        "stage_statuses": stage_statuses,
        "pending_confirmation": pending_confirmation,
        "outputs": outputs,
        "error": error or {},
        "created_at": now_iso(),
        "notes": [
            "Evidence-grounded Phase 3 MVP workflow composed from individual adapters.",
            "Human confirmation is required before literature-derived candidates become confirmed training data.",
        ],
    }
    report_json = write_json(output_dir / f"{run_id}_literature_workflow_report.json", report)
    summary_md = _write_literature_workflow_summary(output_dir / f"{run_id}_literature_workflow_summary.md", report)
    outputs["workflow_report_json"] = str(report_json)
    outputs["workflow_summary_md"] = str(summary_md)
    report["outputs"] = outputs
    write_json(report_json, report)
    return report, report_json, summary_md


def _default_literature_query(payload: dict[str, Any]) -> str:
    query = str(payload.get("query") or "").strip()
    if query:
        return query
    properties = payload.get("property_columns", [])
    if isinstance(properties, list):
        property_terms = " ".join(str(item) for item in properties if str(item).strip())
    else:
        property_terms = ""
    return " ".join(part for part in ["SMILES", property_terms, "property table"] if part).strip()


def literature_to_dataset_workflow_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_pdf_dir = str(payload.get("input_pdf_dir") or payload.get("pdf_folder") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not input_pdf_dir or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "literature_to_dataset_workflow",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/input_pdf_dir/output_dir are required",
            },
        }

    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    stage_statuses: dict[str, str] = {}
    outputs: dict[str, str] = {}

    def record_stage(name: str, result: dict[str, Any]) -> None:
        stage_statuses[name] = str(result.get("status") or "unknown")
        result_outputs = result.get("outputs", {})
        if isinstance(result_outputs, dict):
            for key, value in result_outputs.items():
                if str(value).strip():
                    outputs[str(key)] = str(value)

    def failed_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
        error = result.get("error", {}) if isinstance(result.get("error"), dict) else {}
        report, _, _ = _finalize_literature_workflow_report(
            output_dir=output_dir,
            run_id=run_id,
            status="failed",
            stage_statuses=stage_statuses,
            outputs=outputs,
            pending_confirmation=False,
            error={
                "code": str(error.get("code") or f"{name}_failed"),
                "message": str(error.get("message") or result),
                "stage": name,
            },
        )
        return {
            "status": "failed",
            "adapter": "literature_to_dataset_workflow",
            "workflow_report": report,
            "outputs": outputs,
            "error": report["error"],
        }

    parse_payload = dict(payload)
    parse_payload.update(
        {
            "run_id": run_id,
            "input_pdf_dir": input_pdf_dir,
            "output_dir": str(output_dir / "01_parse"),
        }
    )
    parse_result = parse_pdf_folder_mineru_adapter(parse_payload)
    record_stage("parse_pdf_folder", parse_result)
    manifest = parse_result.get("corpus_manifest", {}) if isinstance(parse_result.get("corpus_manifest"), dict) else {}
    if parse_result.get("status") not in {"success", "degraded"} or int(manifest.get("document_count") or 0) <= 0:
        return failed_result("parse_pdf_folder", parse_result)

    index_result = index_corpus_adapter(
        {
            "run_id": run_id,
            "corpus_manifest_json": outputs.get("corpus_manifest_json", ""),
            "output_dir": str(output_dir / "02_index"),
        }
    )
    record_stage("index_corpus", index_result)
    if index_result.get("status") != "success":
        return failed_result("index_corpus", index_result)

    retrieve_result = retrieve_evidence_adapter(
        {
            "run_id": run_id,
            "query": _default_literature_query(payload),
            "corpus_index_json": outputs.get("corpus_index_json", ""),
            "output_dir": str(output_dir / "03_retrieve"),
            "topk": payload.get("topk", 10),
        }
    )
    record_stage("retrieve_evidence", retrieve_result)
    if retrieve_result.get("status") != "success":
        return failed_result("retrieve_evidence", retrieve_result)

    extract_result = extract_records_adapter(
        {
            "run_id": run_id,
            "evidence_hits_json": outputs.get("evidence_hits_json", ""),
            "chunks_jsonl": outputs.get("chunks_jsonl", ""),
            "output_dir": str(output_dir / "04_extract"),
            "confidence_threshold": payload.get("confidence_threshold", 0.7),
            "property_columns": payload.get("property_columns", []),
        }
    )
    record_stage("extract_records", extract_result)
    if extract_result.get("status") != "success":
        return failed_result("extract_records", extract_result)

    normalize_result = normalize_extracted_units_adapter(
        {
            "run_id": run_id,
            "extracted_records_jsonl": outputs.get("extracted_records_jsonl", ""),
            "output_dir": str(output_dir / "05_normalize_units"),
        }
    )
    record_stage("normalize_extracted_units", normalize_result)
    if normalize_result.get("status") != "success":
        return failed_result("normalize_extracted_units", normalize_result)

    provenance_result = track_citation_provenance_adapter(
        {
            "run_id": run_id,
            "corpus_manifest_json": outputs.get("corpus_manifest_json", ""),
            "evidence_hits_json": outputs.get("evidence_hits_json", ""),
            "extracted_records_jsonl": outputs.get("normalized_extracted_records_jsonl", ""),
            "output_dir": str(output_dir / "06_provenance"),
            "default_license": payload.get("default_license", ""),
            "license_overrides": payload.get("license_overrides", {}),
        }
    )
    record_stage("track_citation_provenance", provenance_result)
    if provenance_result.get("status") != "success":
        return failed_result("track_citation_provenance", provenance_result)

    merge_result = merge_extracted_records_adapter(
        {
            "run_id": run_id,
            "extracted_records_jsonl": outputs.get("normalized_extracted_records_jsonl", ""),
            "citation_provenance_report_json": outputs.get("citation_provenance_report_json", ""),
            "output_dir": str(output_dir / "07_merge"),
            "absolute_tolerance": payload.get("absolute_tolerance", 0.05),
            "property_tolerances": payload.get("property_tolerances", {}),
        }
    )
    record_stage("merge_extracted_records", merge_result)
    if merge_result.get("status") != "success":
        return failed_result("merge_extracted_records", merge_result)

    benchmark_result = evaluate_extraction_benchmark_adapter(
        {
            "run_id": run_id,
            "evidence_hits_json": outputs.get("evidence_hits_json", ""),
            "extracted_records_jsonl": outputs.get("normalized_extracted_records_jsonl", ""),
            "gold_records_jsonl": payload.get("gold_records_jsonl", ""),
            "gold_evidence_refs": payload.get("gold_evidence_refs", []),
            "conflict_report_json": outputs.get("conflict_report_json", ""),
            "extraction_confidence_report_json": outputs.get("extraction_confidence_report_json", ""),
            "citation_provenance_report_json": outputs.get("citation_provenance_report_json", ""),
            "unit_normalization_report_json": outputs.get("unit_normalization_report_json", ""),
            "candidate_training_dataset_csv": outputs.get("candidate_training_dataset_csv", ""),
            "model_metrics_before_json": payload.get("model_metrics_before_json", ""),
            "model_metrics_after_json": payload.get("model_metrics_after_json", ""),
            "output_dir": str(output_dir / "08_benchmark"),
        }
    )
    record_stage("evaluate_extraction_benchmark", benchmark_result)
    if benchmark_result.get("status") != "success":
        return failed_result("evaluate_extraction_benchmark", benchmark_result)

    should_confirm = _as_bool(payload.get("confirmed", False)) or _as_bool(payload.get("confirm_dataset", False))
    if should_confirm:
        confirm_result = confirm_extracted_dataset_adapter(
            {
                "run_id": run_id,
                "candidate_training_dataset_csv": outputs.get("candidate_training_dataset_csv", ""),
                "conflict_report_json": outputs.get("conflict_report_json", ""),
                "citation_provenance_report_json": outputs.get("citation_provenance_report_json", ""),
                "output_dir": str(output_dir / "09_confirm"),
                "confirmed": True,
                "actor": payload.get("actor") or payload.get("approved_by") or "",
                "dataset_id": payload.get("dataset_id") or "",
                "allow_conflicts": payload.get("allow_conflicts", False),
                "allow_license_review": payload.get("allow_license_review", False),
                "note": payload.get("note") or "",
            }
        )
        record_stage("confirm_extracted_dataset", confirm_result)
        if confirm_result.get("status") != "success":
            return failed_result("confirm_extracted_dataset", confirm_result)
        final_status = "success"
        pending_confirmation = False
    else:
        stage_statuses["confirm_extracted_dataset"] = "skipped"
        final_status = "waiting_confirmation"
        pending_confirmation = True

    report, _, _ = _finalize_literature_workflow_report(
        output_dir=output_dir,
        run_id=run_id,
        status=final_status,
        stage_statuses=stage_statuses,
        outputs=outputs,
        pending_confirmation=pending_confirmation,
    )
    return {
        "status": final_status,
        "adapter": "literature_to_dataset_workflow",
        "workflow_report": report,
        "outputs": outputs,
    }


def _public_dataset_csv_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("public_dataset_csvs", "public_benchmark_csvs", "benchmark_dataset_csvs"):
        value = payload.get(key)
        if isinstance(value, list):
            paths.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            paths.extend(part.strip() for part in value.split(",") if part.strip())
    single = str(payload.get("public_dataset_csv") or payload.get("public_benchmark_csv") or "").strip()
    if single:
        paths.append(single)
    return paths


def _write_benchmark_contamination_md(path: Path, report: dict[str, Any]) -> Path:
    lines = [
        f"# Benchmark Contamination Report: {report.get('run_id', '')}",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Public datasets checked: {report.get('dataset_count', 0)}",
        f"- Unique overlapping SMILES: {report.get('total_overlap_count', 0)}",
        "",
        "| Dataset | Public rows | Overlap | Examples |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in report.get("datasets", []):
        if not isinstance(item, dict):
            continue
        examples = ", ".join(str(value) for value in item.get("overlap_smiles", [])[:10])
        lines.append(
            f"| {item.get('dataset_id', '')} | {item.get('other_count', 0)} | "
            f"{item.get('overlap_count', 0)} | {examples} |"
        )
    lines.extend(
        [
            "",
            "Overlaps indicate possible public benchmark contamination or train/test leakage and should be reviewed before reporting model performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def check_public_dataset_leakage_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    train_raw = str(
        payload.get("training_dataset_csv")
        or payload.get("confirmed_training_dataset_csv")
        or payload.get("candidate_training_dataset_csv")
        or ""
    ).strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    public_paths_raw = _public_dataset_csv_paths(payload)
    if not run_id or not train_raw or not output_dir_raw or not public_paths_raw:
        return {
            "status": "failed",
            "adapter": "check_public_dataset_leakage",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/training_dataset_csv/public_dataset_csvs/output_dir are required",
            },
        }

    train_csv = _resolve_path(train_raw, base=WORKSPACE)
    if not train_csv.exists() or not train_csv.is_file():
        return {
            "status": "failed",
            "adapter": "check_public_dataset_leakage",
            "error": {"code": "training_dataset_missing", "message": f"training dataset not found: {train_csv}"},
        }

    datasets: list[dict[str, Any]] = []
    missing: list[str] = []
    total_overlap: set[str] = set()
    for raw_path in public_paths_raw:
        public_csv = _resolve_path(raw_path, base=WORKSPACE)
        if not public_csv.exists() or not public_csv.is_file():
            missing.append(str(public_csv))
            continue
        leakage = check_smiles_leakage(train_csv, public_csv)
        leakage_payload = leakage.model_dump(mode="json")
        total_overlap.update(leakage.overlap_smiles)
        datasets.append(
            {
                "dataset_id": public_csv.stem,
                "path": str(public_csv),
                **leakage_payload,
            }
        )

    if missing:
        return {
            "status": "failed",
            "adapter": "check_public_dataset_leakage",
            "error": {
                "code": "public_dataset_missing",
                "message": "one or more public benchmark datasets were not found",
                "missing": missing,
            },
        }

    report = {
        "run_id": run_id,
        "status": "overlap_detected" if total_overlap else "clear",
        "training_dataset_csv": str(train_csv),
        "dataset_count": len(datasets),
        "total_overlap_count": len(total_overlap),
        "total_overlap_smiles": sorted(total_overlap),
        "datasets": datasets,
        "generated_at": now_iso(),
        "notes": [
            "Canonical SMILES overlap is used as the Phase 3 MVP leakage signal.",
            "This does not replace property-level split analysis or benchmark-specific contamination policy.",
        ],
    }
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    report_json = write_json(output_dir / f"{run_id}_benchmark_contamination_report.json", report)
    report_md = _write_benchmark_contamination_md(output_dir / f"{run_id}_benchmark_contamination_report.md", report)
    return {
        "status": "success",
        "adapter": "check_public_dataset_leakage",
        "report": report,
        "outputs": {
            "benchmark_contamination_report_json": str(report_json),
            "benchmark_contamination_report_md": str(report_md),
        },
    }


def _record_fact_keys(records: list[ExtractedRecord]) -> set[tuple[str, str, float]]:
    facts: set[tuple[str, str, float]] = set()
    for record in records:
        smiles_key = _canonical_smiles_key(record.smiles)
        if not smiles_key:
            continue
        for property_id, value in record.properties.items():
            facts.add((smiles_key, property_id, round(float(value), 6)))
    return facts


def _gold_evidence_refs(payload: dict[str, Any]) -> list[str]:
    value = payload.get("gold_evidence_refs", [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _trainable_label_count(path: Path) -> int:
    metadata_cols = {
        "",
        "smiles",
        "canonical_smiles",
        "record_id",
        "merge_id",
        "confidence",
        "source_id",
        "paper_id",
        "page",
        "table_id",
        "citation_context",
        "source_record_ids",
        "source_ids",
        "citations",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        property_cols = [
            header
            for header in (reader.fieldnames or [])
            if _property_id(header) not in metadata_cols
        ]
        count = 0
        for row in reader:
            for header in property_cols:
                if safe_float(row.get(header)) is not None:
                    count += 1
        return count


def _metrics_by_key(path_raw: str) -> dict[str, float]:
    if not path_raw:
        return {}
    loaded = _safe_json(_resolve_path(path_raw, base=WORKSPACE))
    metrics_by_key: dict[str, float] = {}
    properties = loaded.get("properties", [])
    if isinstance(properties, list):
        for item in properties:
            if not isinstance(item, dict):
                continue
            property_id = str(item.get("property_id") or "").strip()
            metrics = item.get("metrics", {})
            if not property_id or not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                try:
                    metrics_by_key[f"{property_id}.{metric}"] = float(value)
                except (TypeError, ValueError):
                    continue
    direct_metrics = loaded.get("metrics", {})
    if isinstance(direct_metrics, dict):
        for metric, value in direct_metrics.items():
            try:
                metrics_by_key[str(metric)] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics_by_key


def _downstream_metric_delta(payload: dict[str, Any]) -> dict[str, float]:
    before = _metrics_by_key(str(payload.get("model_metrics_before_json") or payload.get("baseline_metrics_before_json") or ""))
    after = _metrics_by_key(str(payload.get("model_metrics_after_json") or payload.get("baseline_metrics_after_json") or ""))
    return {
        key: round(after[key] - before[key], 6)
        for key in sorted(before.keys() & after.keys())
    }


def _write_extraction_benchmark_md(path: Path, report: ExtractionBenchmarkReport) -> Path:
    lines = [
        f"# Extraction Benchmark Report: {report.run_id}",
        "",
        "| Metric | Value | Status |",
        "| --- | ---: | --- |",
        f"| retrieval_recall | {report.retrieval_recall} | {report.metric_statuses.get('retrieval_recall', '')} |",
        f"| extraction_precision | {report.extraction_precision} | {report.metric_statuses.get('extraction_precision', '')} |",
        f"| conflict_rate | {report.conflict_rate} | computed |",
        f"| confirmation_workload_count | {report.confirmation_workload_count} | computed |",
        f"| trainable_labels_gained | {report.trainable_labels_gained} | computed |",
        "",
        "## Downstream Model Performance Delta",
    ]
    if report.downstream_model_performance_delta:
        for key, value in report.downstream_model_performance_delta.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "Metrics with missing gold labels are reported as `null` rather than inferred.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_extraction_benchmark_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "evaluate_extraction_benchmark",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    evidence_hits_raw = str(payload.get("evidence_hits_json") or "").strip()
    if evidence_hits_raw and not _resolve_path(evidence_hits_raw, base=WORKSPACE).is_file():
        return {
            "status": "failed",
            "adapter": "evaluate_extraction_benchmark",
            "error": {"code": "missing_evidence_hits", "message": f"evidence_hits not found: {evidence_hits_raw}"},
        }
    conflict_report_raw = str(payload.get("conflict_report_json") or "").strip()
    if conflict_report_raw and not _resolve_path(conflict_report_raw, base=WORKSPACE).is_file():
        return {
            "status": "failed",
            "adapter": "evaluate_extraction_benchmark",
            "error": {"code": "missing_conflict_report", "message": f"conflict_report not found: {conflict_report_raw}"},
        }

    hits = _load_evidence_hits(payload) if (payload.get("evidence_hits_json") or payload.get("hits")) else []
    retrieved_refs = {
        ref
        for hit in hits
        for ref in [hit.text_or_table_ref, str(hit.metadata.get("chunk_id") or "")]
        if ref
    }
    gold_refs = _gold_evidence_refs(payload)
    retrieval_recall = round(len(retrieved_refs & set(gold_refs)) / len(gold_refs), 6) if gold_refs else None

    extracted_records_raw = str(payload.get("extracted_records_jsonl") or payload.get("normalized_extracted_records_jsonl") or "").strip()
    extracted_records = _read_extracted_records(_resolve_path(extracted_records_raw, base=WORKSPACE)) if extracted_records_raw else []
    gold_records_raw = str(payload.get("gold_records_jsonl") or "").strip()
    gold_records = _read_extracted_records(_resolve_path(gold_records_raw, base=WORKSPACE)) if gold_records_raw else []
    extracted_facts = _record_fact_keys(extracted_records)
    gold_facts = _record_fact_keys(gold_records)
    if gold_facts:
        extraction_precision = round(len(extracted_facts & gold_facts) / len(extracted_facts), 6) if extracted_facts else 0.0
    else:
        extraction_precision = None

    conflict_report = _safe_json(_resolve_path(str(payload.get("conflict_report_json") or ""), base=WORKSPACE)) if payload.get("conflict_report_json") else {}
    conflict_count = int(conflict_report.get("conflict_count") or 0)
    conflict_denominator = int(conflict_report.get("input_record_count") or conflict_report.get("merged_record_count") or 0)
    conflict_rate = round(conflict_count / conflict_denominator, 6) if conflict_denominator else 0.0

    confidence_report = _safe_json(_resolve_path(str(payload.get("extraction_confidence_report_json") or ""), base=WORKSPACE)) if payload.get("extraction_confidence_report_json") else {}
    citation_report = _safe_json(_resolve_path(str(payload.get("citation_provenance_report_json") or ""), base=WORKSPACE)) if payload.get("citation_provenance_report_json") else {}
    normalization_report = _safe_json(_resolve_path(str(payload.get("unit_normalization_report_json") or ""), base=WORKSPACE)) if payload.get("unit_normalization_report_json") else {}
    confirmation_workload_count = (
        conflict_count
        + int(confidence_report.get("rejected_record_count") or 0)
        + int(citation_report.get("unknown_license_count") or 0)
        + int(normalization_report.get("warning_count") or 0)
    )

    candidate_raw = str(
        payload.get("confirmed_training_dataset_csv")
        or payload.get("candidate_training_dataset_csv")
        or payload.get("normalized_candidate_training_dataset_csv")
        or ""
    ).strip()
    trainable_labels_gained = _trainable_label_count(_resolve_path(candidate_raw, base=WORKSPACE)) if candidate_raw else 0
    downstream_delta = _downstream_metric_delta(payload)
    metric_statuses = {
        "retrieval_recall": "computed" if gold_refs else "missing_gold_evidence",
        "extraction_precision": "computed" if gold_facts else "missing_gold_records",
        "downstream_model_performance_delta": "computed" if downstream_delta else "missing_before_after_metrics",
    }
    report = ExtractionBenchmarkReport(
        run_id=run_id,
        retrieval_recall=retrieval_recall,
        extraction_precision=extraction_precision,
        conflict_rate=conflict_rate,
        confirmation_workload_count=confirmation_workload_count,
        trainable_labels_gained=trainable_labels_gained,
        downstream_model_performance_delta=downstream_delta,
        metric_statuses=metric_statuses,
        counts={
            "evidence_hits": len(hits),
            "gold_evidence": len(gold_refs),
            "extracted_records": len(extracted_records),
            "gold_records": len(gold_records),
            "extracted_facts": len(extracted_facts),
            "gold_facts": len(gold_facts),
            "conflicts": conflict_count,
        },
        generated_at=now_iso(),
        notes=[
            "Gold-backed metrics are only computed when gold references or records are supplied.",
            "Confirmation workload counts rejected records, conflicts, license-review sources, and unit warnings.",
        ],
    )
    output_dir = _ensure_dir(_resolve_path(output_dir_raw, base=WORKSPACE))
    report_json = write_json(output_dir / f"{run_id}_extraction_benchmark_report.json", report.model_dump(mode="json"))
    report_md = _write_extraction_benchmark_md(output_dir / f"{run_id}_extraction_benchmark_report.md", report)
    return {
        "status": "success",
        "adapter": "evaluate_extraction_benchmark",
        "report": report.model_dump(mode="json"),
        "outputs": {
            "extraction_benchmark_report_json": str(report_json),
            "extraction_benchmark_report_md": str(report_md),
        },
    }
