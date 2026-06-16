from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ai4s_agent.adapters.phase3 as phase3_module
from ai4s_agent.adapters.phase3 import (
    check_public_dataset_leakage_adapter,
    confirm_extracted_dataset_adapter,
    evaluate_extraction_benchmark_adapter,
    extract_records_adapter,
    acquire_literature_sources_adapter,
    build_dense_index_adapter,
    build_multi_index_adapter,
    index_corpus_adapter,
    literature_to_dataset_workflow_adapter,
    merge_extracted_records_adapter,
    normalize_extracted_units_adapter,
    parse_document_grobid_adapter,
    parse_document_mineru_adapter,
    parse_document_pdfplumber_adapter,
    parse_document_pymupdf_adapter,
    parse_pdf_folder_mineru_adapter,
    prepare_literature_corpus_sources_adapter,
    retrieve_evidence_adapter,
    track_citation_provenance_adapter,
)
from ai4s_agent.schemas import EvidenceHit, ExtractedRecord, ParsedDocument, ParsedDocumentElement, ParsedTable


def _write_mineru_table_fixture(tmp_path: Path, *, paper_stem: str = "paper") -> tuple[Path, Path]:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / f"{paper_stem}.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")
    raw_root = tmp_path / "mineru_outputs"
    paper_raw = raw_root / paper_stem
    paper_raw.mkdir(parents=True)
    (paper_raw / f"{paper_stem}.md").write_text(
        "# Literature Table\n\n| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |\n",
        encoding="utf-8",
    )
    (paper_raw / "layout.json").write_text(
        json.dumps(
            {
                "pages": [{"page": 1}],
                "tables": [
                    {
                        "page": 1,
                        "caption": "OLED measurements",
                        "headers": ["SMILES", "PLQY"],
                        "rows": [{"SMILES": "CCO", "PLQY": "0.8"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pdf_dir, raw_root


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(header, "")) for header in headers) + "\n")
    return path


def test_prepare_literature_corpus_sources_adapter_normalizes_non_pdf_source_inputs(tmp_path: Path) -> None:
    result = prepare_literature_corpus_sources_adapter(
        {
            "run_id": "r-source-manifest",
            "output_dir": str(tmp_path / "sources"),
            "search_queries": ["TADF OLED PLQY"],
            "urls": ["https://example.org/paper"],
            "dois": ["10.1000/example"],
            "dataset_registries": [{"registry": "zenodo", "record_id": "12345"}],
            "external_databases": [{"database": "PubChem", "query": "CCO"}],
            "sources": [
                {"source_type": "url", "url": "https://example.org/paper"},
                {"doi": "10.1000/another"},
            ],
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "prepare_literature_corpus_sources"
    manifest = result["corpus_source_manifest"]
    assert manifest["source_count"] == 6
    assert manifest["source_type_counts"] == {
        "dataset_registry": 1,
        "doi": 2,
        "external_database": 1,
        "search_query": 1,
        "url": 1,
    }
    values_by_type = {source["source_type"]: source["value"] for source in manifest["sources"]}
    assert values_by_type["dataset_registry"] == "zenodo:12345"
    assert values_by_type["external_database"] == "PubChem:CCO"
    assert {source["status"] for source in manifest["sources"]} == {"pending_acquisition"}
    assert Path(result["outputs"]["corpus_source_manifest_json"]).exists()
    assert Path(result["outputs"]["corpus_source_manifest_md"]).exists()


def test_prepare_literature_corpus_sources_adapter_returns_structured_error_for_unknown_source_type(tmp_path: Path) -> None:
    result = prepare_literature_corpus_sources_adapter(
        {
            "run_id": "r-bad-source",
            "output_dir": str(tmp_path / "sources"),
            "sources": [{"source_type": "rss_feed", "value": "https://example.org/feed"}],
        }
    )

    assert result["status"] == "failed"
    assert result["adapter"] == "prepare_literature_corpus_sources"
    assert result["error"]["code"] == "invalid_literature_sources"


def test_acquire_literature_sources_adapter_uses_local_mirrors_and_plans_pending_sources(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf_folder"
    pdf_dir.mkdir()
    folder_pdf = pdf_dir / "folder-paper.pdf"
    folder_pdf.write_bytes(b"%PDF-1.4\n% folder\n")
    doi_pdf = tmp_path / "doi-paper.pdf"
    doi_pdf.write_bytes(b"%PDF-1.4\n% doi\n")
    registry_csv = _write_csv(tmp_path / "registry.csv", [{"smiles": "CCO", "plqy": 0.8}])
    sources = prepare_literature_corpus_sources_adapter(
        {
            "run_id": "r-acquire",
            "output_dir": str(tmp_path / "sources"),
            "input_pdf_dir": str(pdf_dir),
            "dois": ["10.1000/example"],
            "dataset_registries": [{"registry": "zenodo", "record_id": "12345"}],
            "external_databases": [{"database": "PubChem", "query": "CCO"}],
        }
    )

    result = acquire_literature_sources_adapter(
        {
            "run_id": "r-acquire",
            "corpus_source_manifest_json": sources["outputs"]["corpus_source_manifest_json"],
            "output_dir": str(tmp_path / "acquired"),
            "local_mirror": {
                "10.1000/example": str(doi_pdf),
                "zenodo:12345": str(registry_csv),
            },
        }
    )

    assert result["status"] == "degraded"
    assert result["adapter"] == "acquire_literature_sources"
    manifest = result["acquisition_manifest"]
    assert manifest["acquired_count"] == 3
    assert manifest["planned_count"] == 1
    assert manifest["failed_count"] == 0
    assert Path(result["outputs"]["acquisition_manifest_json"]).exists()
    assert Path(result["outputs"]["acquisition_plan_md"]).exists()
    assert len(list(Path(result["outputs"]["acquired_pdf_dir"]).glob("*.pdf"))) == 2
    assert len(list(Path(result["outputs"]["acquired_dataset_dir"]).glob("*.csv"))) == 1
    planned = [item for item in manifest["items"] if item["status"] == "planned"]
    assert planned[0]["source_type"] == "external_database"


def test_parse_document_mineru_adapter_normalizes_existing_output(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")
    mineru_output = tmp_path / "mineru_output"
    mineru_output.mkdir()
    (mineru_output / "paper.md").write_text(
        "# Test Paper\n\n"
        "This paragraph mentions CCO and PLQY.\n\n"
        "| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |\n",
        encoding="utf-8",
    )
    (mineru_output / "layout.json").write_text(
        json.dumps(
            {
                "pages": [{"page": 1, "width": 595.0, "height": 842.0}],
                "elements": [
                    {"page": 1, "type": "title", "text": "Test Paper", "bbox": [0, 0, 100, 20]},
                    {"page": 1, "type": "paragraph", "text": "This paragraph mentions CCO and PLQY."},
                ],
                "tables": [
                    {
                        "page": 1,
                        "caption": "Table 1",
                        "headers": ["SMILES", "PLQY"],
                        "rows": [{"SMILES": "CCO", "PLQY": "0.8"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = parse_document_mineru_adapter(
        {
            "run_id": "r-doc",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
            "mineru_output_dir": str(mineru_output),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "parse_document_mineru"
    assert result["parsed_document"]["paper_id"] == "paper"
    assert result["parsed_document"]["parser_backend"] == "mineru_remote_cli"
    assert result["parsed_document"]["tables"][0]["headers"] == ["SMILES", "PLQY"]
    assert Path(result["outputs"]["parsed_document_json"]).exists()
    assert Path(result["outputs"]["parsed_document_markdown"]).exists()
    assert Path(result["outputs"]["parser_audit_json"]).exists()


def test_parse_document_mineru_adapter_plans_workstation2_cli_without_execution(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")

    result = parse_document_mineru_adapter(
        {
            "run_id": "r-doc-plan",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
            "execute": False,
            "remote_host": "workstation2",
            "mineru_api_url": "http://127.0.0.1:8000",
        }
    )

    assert result["status"] == "planned"
    assert result["adapter"] == "parse_document_mineru"
    assert result["remote"]["host"] == "workstation2"
    assert result["remote"]["backend"] == "remote_cli"
    assert "mineru" in " ".join(result["command"])
    assert "--api-url" in result["command"]
    assert "http://127.0.0.1:8000" in result["command"]


def test_parse_document_mineru_adapter_does_not_treat_false_string_as_execute(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")

    def fail_run_argv_cmd(**_: object) -> dict[str, object]:
        raise AssertionError("remote command should not run when execute is false")

    monkeypatch.setattr(phase3_module, "run_argv_cmd", fail_run_argv_cmd)

    result = parse_document_mineru_adapter(
        {
            "run_id": "r-doc-string-false",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
            "execute": "false",
            "remote_host": "workstation2",
        }
    )

    assert result["status"] == "planned"


def test_parse_document_pdfplumber_adapter_extracts_text_and_tables_when_available(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% pdfplumber\n")

    class FakePage:
        page_number = 1

        def extract_text(self) -> str:
            return "PDFPlumber paragraph with CCO and PLQY."

        def extract_tables(self) -> list[list[list[str]]]:
            return [[["SMILES", "PLQY"], ["CCO", "0.82"]]]

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda _: FakePdf()))

    result = parse_document_pdfplumber_adapter(
        {
            "run_id": "r-pdfplumber",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "parse_document_pdfplumber"
    assert result["parsed_document"]["parser_backend"] == "pdfplumber_local"
    assert result["parsed_document"]["elements"][0]["text"] == "PDFPlumber paragraph with CCO and PLQY."
    assert result["parsed_document"]["tables"][0]["headers"] == ["SMILES", "PLQY"]
    assert result["parsed_document"]["tables"][0]["rows"] == [{"SMILES": "CCO", "PLQY": "0.82"}]
    assert Path(result["outputs"]["parsed_document_json"]).exists()
    assert Path(result["outputs"]["parser_audit_json"]).exists()


def test_parse_document_pymupdf_adapter_extracts_page_text_when_available(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% pymupdf\n")

    class FakePage:
        number = 0

        def get_text(self, mode: str = "text") -> str:
            assert mode == "text"
            return "PyMuPDF paragraph with OLED evidence."

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage()])

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=lambda _: FakeDoc()))

    result = parse_document_pymupdf_adapter(
        {
            "run_id": "r-pymupdf",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "parse_document_pymupdf"
    assert result["parsed_document"]["parser_backend"] == "pymupdf_local"
    assert result["parsed_document"]["elements"][0]["page"] == 1
    assert result["parsed_document"]["elements"][0]["text"] == "PyMuPDF paragraph with OLED evidence."


def test_parse_document_grobid_adapter_normalizes_existing_tei(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% grobid\n")
    tei = tmp_path / "paper.tei.xml"
    tei.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>GROBID Parsed OLED Paper</title></titleStmt>
      <publicationStmt><availability><licence>CC-BY-4.0</licence></availability></publicationStmt>
      <sourceDesc><biblStruct><idno type="DOI">10.1000/grobid</idno></biblStruct></sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div><p>GROBID body text with CCO and photoluminescence quantum yield.</p></div>
      <figure type="table">
        <head>OLED data</head>
        <table>
          <row role="label"><cell>SMILES</cell><cell>PLQY</cell></row>
          <row><cell>CCO</cell><cell>0.77</cell></row>
        </table>
      </figure>
    </body>
  </text>
</TEI>
""",
        encoding="utf-8",
    )

    result = parse_document_grobid_adapter(
        {
            "run_id": "r-grobid",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
            "grobid_tei_xml": str(tei),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "parse_document_grobid"
    assert result["parsed_document"]["parser_backend"] == "grobid_tei"
    assert result["parsed_document"]["metadata"]["title"] == "GROBID Parsed OLED Paper"
    assert result["parsed_document"]["metadata"]["doi"] == "10.1000/grobid"
    assert result["parsed_document"]["metadata"]["license"] == "CC-BY-4.0"
    assert result["parsed_document"]["elements"][0]["text"].startswith("GROBID body text")
    assert result["parsed_document"]["tables"][0]["headers"] == ["SMILES", "PLQY"]
    assert result["parsed_document"]["tables"][0]["rows"] == [{"SMILES": "CCO", "PLQY": "0.77"}]


def test_parse_document_mineru_adapter_executes_remote_and_normalizes_fetched_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")
    remote_raw = tmp_path / "remote_raw"
    remote_raw.mkdir()
    (remote_raw / "paper.md").write_text("# Remote Paper\n\nParsed remotely.\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int = 120) -> dict[str, object]:
        calls.append(argv)
        if argv[0] == "scp" and argv[-1].endswith(".pdf"):
            return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
        if argv[0] == "ssh":
            return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
        if argv[0] == "scp" and "paper_output" in argv[-2]:
            dest = Path(argv[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "paper.md").write_text(remote_raw.joinpath("paper.md").read_text(encoding="utf-8"), encoding="utf-8")
            return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
        return {"argv": argv, "returncode": 1, "stdout": "", "stderr": "unexpected command"}

    monkeypatch.setattr(phase3_module, "run_argv_cmd", fake_run_argv_cmd)

    result = parse_document_mineru_adapter(
        {
            "run_id": "r-doc-remote",
            "input_pdf": str(pdf),
            "output_dir": str(tmp_path / "parsed"),
            "execute": True,
            "remote_host": "workstation2",
        }
    )

    assert result["status"] == "success"
    assert result["parsed_document"]["metadata"]["title"] == "Remote Paper"
    assert any(call[0] == "scp" and call[-1].endswith(".pdf") for call in calls)
    assert any(call[0] == "ssh" and "mineru" in call[-1] for call in calls)
    assert any(call[0] == "scp" and "paper_output" in call[-2] for call in calls)


def test_parse_pdf_folder_mineru_adapter_builds_corpus_manifest(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    first = pdf_dir / "a.pdf"
    second = pdf_dir / "b.pdf"
    first.write_bytes(b"%PDF-1.4\n% a\n")
    second.write_bytes(b"%PDF-1.4\n% b\n")
    raw_root = tmp_path / "mineru_outputs"
    (raw_root / "a").mkdir(parents=True)
    (raw_root / "b").mkdir(parents=True)
    (raw_root / "a" / "a.md").write_text("# Paper A\n", encoding="utf-8")
    (raw_root / "b" / "b.md").write_text("# Paper B\n", encoding="utf-8")

    result = parse_pdf_folder_mineru_adapter(
        {
            "run_id": "r-corpus",
            "input_pdf_dir": str(pdf_dir),
            "output_dir": str(tmp_path / "parsed"),
            "mineru_output_root": str(raw_root),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "parse_pdf_folder_mineru"
    assert result["corpus_manifest"]["document_count"] == 2
    assert [doc["paper_id"] for doc in result["corpus_manifest"]["documents"]] == ["a", "b"]
    assert Path(result["outputs"]["corpus_manifest_json"]).exists()


def test_index_and_retrieve_evidence_from_parsed_document(tmp_path: Path) -> None:
    parsed = ParsedDocument(
        paper_id="paper-1",
        source_path=str(tmp_path / "paper-1.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={"title": "Photophysics Paper"},
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="el_0001",
                page=1,
                type="paragraph",
                text="The emission wavelength was measured for several molecules.",
                source_hash="sha256:abc",
            )
        ],
        tables=[
            ParsedTable(
                table_id="table_0001",
                caption="OLED molecule measurements",
                headers=["SMILES", "PLQY"],
                rows=[{"SMILES": "CCO", "PLQY": "0.8"}],
                page=1,
                markdown="| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |",
            )
        ],
    )
    parsed_json = tmp_path / "paper-1_parsed_document.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")

    index = index_corpus_adapter(
        {
            "run_id": "r-index",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "index"),
        }
    )

    assert index["status"] == "success"
    assert index["adapter"] == "index_corpus"
    assert index["index_report"]["chunk_count"] == 2
    assert Path(index["outputs"]["chunks_jsonl"]).exists()
    assert Path(index["outputs"]["corpus_index_json"]).exists()

    retrieval = retrieve_evidence_adapter(
        {
            "run_id": "r-retrieve",
            "query": "CCO PLQY",
            "corpus_index_json": index["outputs"]["corpus_index_json"],
            "output_dir": str(tmp_path / "retrieval"),
            "topk": 3,
        }
    )

    assert retrieval["status"] == "success"
    assert retrieval["adapter"] == "retrieve_evidence"
    assert retrieval["hits"][0]["element_type"] == "table"
    assert retrieval["hits"][0]["retrieval_channel"] == "table"
    assert retrieval["hits"][0]["source_id"] == "paper-1"
    assert Path(retrieval["outputs"]["evidence_hits_json"]).exists()
    assert Path(retrieval["outputs"]["retrieval_log_jsonl"]).exists()


def test_build_multi_index_adapter_enables_property_and_chemical_retrieval_boost(tmp_path: Path) -> None:
    doc = ParsedDocument(
        paper_id="paper-multi",
        source_path=str(tmp_path / "paper-multi.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={"title": "Multi Index Paper"},
        pages=[{"page": 1}],
        elements=[],
        tables=[
            ParsedTable(
                table_id="table_1",
                caption="OLED data",
                headers=["SMILES", "PLQY", "lambda_em_nm"],
                rows=[{"SMILES": "CCO", "PLQY": "0.8", "lambda_em_nm": "520"}],
                page=1,
                markdown="| SMILES | PLQY | lambda_em_nm |\n| --- | --- | --- |\n| CCO | 0.8 | 520 |",
            )
        ],
    )
    parsed_json = tmp_path / "parsed.json"
    parsed_json.write_text(json.dumps(doc.model_dump(mode="json")), encoding="utf-8")
    index = index_corpus_adapter(
        {
            "run_id": "r-multi",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "index"),
        }
    )

    multi = build_multi_index_adapter(
        {
            "run_id": "r-multi",
            "chunks_jsonl": index["outputs"]["chunks_jsonl"],
            "output_dir": str(tmp_path / "multi"),
        }
    )
    retrieval = retrieve_evidence_adapter(
        {
            "run_id": "r-multi",
            "query": "CCO PLQY",
            "corpus_index_json": index["outputs"]["corpus_index_json"],
            "multi_index_json": multi["outputs"]["multi_index_json"],
            "output_dir": str(tmp_path / "retrieve"),
            "topk": 1,
        }
    )

    assert multi["status"] == "success"
    assert multi["adapter"] == "build_multi_index"
    assert multi["multi_index"]["channel_counts"]["chemical"] == 1
    assert multi["multi_index"]["channel_counts"]["property"] >= 2
    assert Path(multi["outputs"]["multi_index_json"]).exists()
    assert Path(multi["outputs"]["multi_index_summary_md"]).exists()
    assert retrieval["status"] == "success"
    assert retrieval["hits"][0]["element_type"] == "table"
    assert retrieval["hits"][0]["metadata"]["multi_index_channels"] == ["chemical", "property", "table", "text"]


def test_build_dense_index_adapter_enables_dense_retrieval_for_synonym_query(tmp_path: Path) -> None:
    parsed = ParsedDocument(
        paper_id="paper-dense",
        source_path=str(tmp_path / "paper-dense.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={"title": "Dense Retrieval Paper"},
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="el_quantum_yield",
                page=1,
                type="paragraph",
                text="The molecule shows high photoluminescence quantum yield in thin film.",
                source_hash="sha256:dense",
            ),
            ParsedDocumentElement(
                element_id="el_voltage",
                page=1,
                type="paragraph",
                text="The device voltage and current density were measured separately.",
                source_hash="sha256:dense",
            ),
        ],
        tables=[],
    )
    parsed_json = tmp_path / "dense_parsed.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")
    index = index_corpus_adapter(
        {
            "run_id": "r-dense",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "index"),
        }
    )

    dense = build_dense_index_adapter(
        {
            "run_id": "r-dense",
            "chunks_jsonl": index["outputs"]["chunks_jsonl"],
            "output_dir": str(tmp_path / "dense"),
            "dimension": 32,
        }
    )
    retrieval = retrieve_evidence_adapter(
        {
            "run_id": "r-dense",
            "query": "PLQY efficiency",
            "corpus_index_json": index["outputs"]["corpus_index_json"],
            "dense_index_json": dense["outputs"]["dense_index_json"],
            "output_dir": str(tmp_path / "retrieve"),
            "topk": 1,
        }
    )

    assert dense["status"] == "success"
    assert dense["adapter"] == "build_dense_index"
    assert dense["dense_index"]["embedding_backend"] == "deterministic_hash_embedding"
    assert dense["dense_index"]["dimension"] == 32
    assert Path(dense["outputs"]["dense_index_json"]).exists()
    assert Path(dense["outputs"]["dense_index_summary_md"]).exists()
    assert retrieval["status"] == "success"
    assert retrieval["hits"][0]["element_id"] == "el_quantum_yield"
    assert retrieval["hits"][0]["retrieval_channel"] == "dense"
    assert retrieval["hits"][0]["metadata"]["dense_score"] > 0


def test_build_dense_index_adapter_uses_sentence_transformers_backend_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parsed = ParsedDocument(
        paper_id="paper-st",
        source_path=str(tmp_path / "paper-st.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={"title": "Sentence Transformer Paper"},
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="el_quantum_yield",
                page=1,
                type="paragraph",
                text="The molecule shows high photoluminescence quantum yield in thin film.",
                source_hash="sha256:st",
            ),
            ParsedDocumentElement(
                element_id="el_voltage",
                page=1,
                type="paragraph",
                text="The device voltage and current density were measured separately.",
                source_hash="sha256:st",
            ),
        ],
        tables=[],
    )
    parsed_json = tmp_path / "st_parsed.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")
    index = index_corpus_adapter(
        {
            "run_id": "r-st",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "index"),
        }
    )

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

        def encode_document(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
            assert normalize_embeddings is True
            return [[1.0, 0.0, 0.0] if "quantum yield" in text else [0.0, 1.0, 0.0] for text in texts]

        def encode_query(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
            assert normalize_embeddings is True
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    dense = build_dense_index_adapter(
        {
            "run_id": "r-st",
            "chunks_jsonl": index["outputs"]["chunks_jsonl"],
            "output_dir": str(tmp_path / "dense"),
            "embedding_backend": "sentence_transformers",
            "embedding_model": "fake-scientific-embedding-model",
        }
    )
    retrieval = retrieve_evidence_adapter(
        {
            "run_id": "r-st",
            "query": "PLQY efficiency",
            "corpus_index_json": index["outputs"]["corpus_index_json"],
            "dense_index_json": dense["outputs"]["dense_index_json"],
            "output_dir": str(tmp_path / "retrieve"),
            "topk": 1,
        }
    )

    assert dense["status"] == "success"
    assert dense["dense_index"]["embedding_backend"] == "sentence_transformers"
    assert dense["dense_index"]["embedding_model"] == "fake-scientific-embedding-model"
    assert dense["dense_index"]["dimension"] == 3
    assert retrieval["status"] == "success"
    assert retrieval["hits"][0]["element_id"] == "el_quantum_yield"
    assert retrieval["hits"][0]["retrieval_channel"] == "dense"
    assert retrieval["hits"][0]["metadata"]["dense_score"] == 1.0


def test_extract_records_adapter_builds_candidate_dataset_from_table_evidence(tmp_path: Path) -> None:
    parsed = ParsedDocument(
        paper_id="paper-2",
        source_path=str(tmp_path / "paper-2.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={"title": "OLED table paper"},
        pages=[{"page": 1}],
        elements=[],
        tables=[
            ParsedTable(
                table_id="table_0001",
                caption="Extracted OLED measurements",
                headers=["SMILES", "PLQY", "lambda_em", "note"],
                rows=[
                    {"SMILES": "CCO", "PLQY": "0.8", "lambda_em": "520", "note": "valid"},
                    {"SMILES": "", "PLQY": "0.3", "lambda_em": "540", "note": "missing structure"},
                ],
                page=1,
                markdown=(
                    "| SMILES | PLQY | lambda_em | note |\n"
                    "| --- | --- | --- | --- |\n"
                    "| CCO | 0.8 | 520 | valid |\n"
                    "|  | 0.3 | 540 | missing structure |"
                ),
            )
        ],
    )
    parsed_json = tmp_path / "paper-2_parsed_document.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")
    index = index_corpus_adapter(
        {
            "run_id": "r-index-extract",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "index"),
        }
    )
    retrieval = retrieve_evidence_adapter(
        {
            "run_id": "r-retrieve-extract",
            "query": "SMILES PLQY lambda_em table",
            "corpus_index_json": index["outputs"]["corpus_index_json"],
            "output_dir": str(tmp_path / "retrieval"),
            "topk": 3,
        }
    )

    result = extract_records_adapter(
        {
            "run_id": "r-extract",
            "evidence_hits_json": retrieval["outputs"]["evidence_hits_json"],
            "chunks_jsonl": index["outputs"]["chunks_jsonl"],
            "output_dir": str(tmp_path / "extraction"),
            "confidence_threshold": 0.7,
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "extract_records"
    assert result["records"][0]["smiles"] == "CCO"
    assert result["records"][0]["properties"] == {"plqy": 0.8, "lambda_em": 520.0}
    assert result["records"][0]["citation_context"] == "paper-2 p.1 table_0001"
    assert result["confidence_report"]["extracted_record_count"] == 1
    assert result["confidence_report"]["rejected_record_count"] == 1
    assert result["confidence_report"]["high_confidence_count"] == 1
    assert Path(result["outputs"]["extracted_records_jsonl"]).exists()
    assert Path(result["outputs"]["rejected_records_jsonl"]).exists()
    assert Path(result["outputs"]["extraction_confidence_report_json"]).exists()
    csv_text = Path(result["outputs"]["candidate_training_dataset_csv"]).read_text(encoding="utf-8")
    assert "smiles,plqy,lambda_em" in csv_text
    assert "CCO,0.8,520.0" in csv_text


def test_track_citation_provenance_adapter_reports_source_license_and_record_counts(tmp_path: Path) -> None:
    parsed = ParsedDocument(
        paper_id="paper-3",
        source_path=str(tmp_path / "paper-3.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={
            "title": "OLED source paper",
            "source_hash": "sha256:paper3",
            "citation": "Doe et al. OLED source paper, 2026",
            "doi": "10.1000/oled-source",
            "license": "CC-BY-4.0",
        },
        pages=[{"page": 1}],
        elements=[],
        tables=[],
    )
    parsed_json = tmp_path / "paper-3_parsed_document.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")

    hit = EvidenceHit(
        source_id="paper-3",
        page=1,
        element_id="table_0001",
        element_type="table",
        retrieval_channel="table",
        score=2.0,
        text_or_table_ref="paper-3:table_0001",
        citation_context="paper-3 p.1 table_0001",
        metadata={"chunk_id": "paper-3:table_0001"},
    )
    hits_json = tmp_path / "evidence_hits.json"
    hits_json.write_text(json.dumps({"run_id": "r-retrieve-prov", "hits": [hit.model_dump(mode="json")]}), encoding="utf-8")

    record = ExtractedRecord(
        record_id="rec_000001",
        smiles="CCO",
        properties={"plqy": 0.8},
        source_id="paper-3",
        paper_id="paper-3",
        page=1,
        table_id="table_0001",
        row_index=0,
        evidence_ref="paper-3:table_0001",
        citation_context="paper-3 p.1 table_0001",
        confidence=0.95,
        confidence_factors={"source": "table_rule_extractor"},
        raw_values={"SMILES": "CCO", "PLQY": "0.8"},
    )
    records_jsonl = tmp_path / "extracted_records.jsonl"
    records_jsonl.write_text(json.dumps(record.model_dump(mode="json")) + "\n", encoding="utf-8")

    result = track_citation_provenance_adapter(
        {
            "run_id": "r-prov",
            "parsed_document_json": str(parsed_json),
            "evidence_hits_json": str(hits_json),
            "extracted_records_jsonl": str(records_jsonl),
            "output_dir": str(tmp_path / "provenance"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "track_citation_provenance"
    assert result["report"]["source_count"] == 1
    assert result["report"]["unknown_license_count"] == 0
    assert result["report"]["sources"][0]["citation"] == "Doe et al. OLED source paper, 2026"
    assert result["report"]["sources"][0]["license"] == "CC-BY-4.0"
    assert result["report"]["sources"][0]["evidence_count"] == 1
    assert result["report"]["sources"][0]["extracted_record_count"] == 1
    assert Path(result["outputs"]["citation_provenance_report_json"]).exists()
    assert Path(result["outputs"]["audit_summary_md"]).exists()


def test_track_citation_provenance_requires_review_for_restrictive_license(tmp_path: Path) -> None:
    parsed = ParsedDocument(
        paper_id="paper-restricted",
        source_path=str(tmp_path / "paper-restricted.pdf"),
        parser_backend="mineru_remote_cli",
        metadata={
            "title": "Restricted paper",
            "source_hash": "sha256:restricted",
            "license": "All rights reserved",
        },
        pages=[{"page": 1}],
        elements=[],
        tables=[],
    )
    parsed_json = tmp_path / "paper_restricted_parsed_document.json"
    parsed_json.write_text(parsed.model_dump_json(), encoding="utf-8")

    result = track_citation_provenance_adapter(
        {
            "run_id": "r-prov-restricted",
            "parsed_document_json": str(parsed_json),
            "output_dir": str(tmp_path / "provenance"),
        }
    )

    assert result["status"] == "success"
    assert result["report"]["unknown_license_count"] == 1
    assert result["report"]["sources"][0]["license_requires_review"] is True


def test_merge_extracted_records_adapter_detects_cross_source_conflicts(tmp_path: Path) -> None:
    records = [
        ExtractedRecord(
            record_id="rec_000001",
            smiles="CCO",
            properties={"plqy": 0.8},
            source_id="paper-a",
            paper_id="paper-a",
            page=1,
            table_id="table_1",
            row_index=0,
            evidence_ref="paper-a:table_1",
            citation_context="paper-a p.1 table_1",
            confidence=0.95,
            confidence_factors={},
            raw_values={"SMILES": "CCO", "PLQY": "0.8"},
        ),
        ExtractedRecord(
            record_id="rec_000002",
            smiles="CCO",
            properties={"plqy": 0.82},
            source_id="paper-b",
            paper_id="paper-b",
            page=2,
            table_id="table_2",
            row_index=1,
            evidence_ref="paper-b:table_2",
            citation_context="paper-b p.2 table_2",
            confidence=0.9,
            confidence_factors={},
            raw_values={"SMILES": "CCO", "PLQY": "0.82"},
        ),
        ExtractedRecord(
            record_id="rec_000003",
            smiles="CCN",
            properties={"plqy": 0.2},
            source_id="paper-a",
            paper_id="paper-a",
            page=1,
            table_id="table_1",
            row_index=2,
            evidence_ref="paper-a:table_1",
            citation_context="paper-a p.1 table_1",
            confidence=0.91,
            confidence_factors={},
            raw_values={"SMILES": "CCN", "PLQY": "0.2"},
        ),
        ExtractedRecord(
            record_id="rec_000004",
            smiles="CCN",
            properties={"plqy": 0.9},
            source_id="paper-c",
            paper_id="paper-c",
            page=3,
            table_id="table_3",
            row_index=0,
            evidence_ref="paper-c:table_3",
            citation_context="paper-c p.3 table_3",
            confidence=0.93,
            confidence_factors={},
            raw_values={"SMILES": "CCN", "PLQY": "0.9"},
        ),
    ]
    records_jsonl = tmp_path / "extracted_records.jsonl"
    records_jsonl.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json")) for record in records) + "\n",
        encoding="utf-8",
    )

    result = merge_extracted_records_adapter(
        {
            "run_id": "r-merge",
            "extracted_records_jsonl": str(records_jsonl),
            "output_dir": str(tmp_path / "merge"),
            "absolute_tolerance": 0.05,
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "merge_extracted_records"
    assert result["conflict_report"]["input_record_count"] == 4
    assert result["conflict_report"]["merged_record_count"] == 2
    assert result["conflict_report"]["conflict_count"] == 1
    assert result["conflict_report"]["conflicts"][0]["smiles"] == "CCN"
    assert result["conflict_report"]["conflicts"][0]["property_id"] == "plqy"
    merged_by_smiles = {record["smiles"]: record for record in result["merged_records"]}
    assert merged_by_smiles["CCO"]["properties"]["plqy"] == 0.81
    assert merged_by_smiles["CCO"]["status"] == "merged"
    assert merged_by_smiles["CCN"]["status"] == "conflict"
    assert Path(result["outputs"]["merged_records_jsonl"]).exists()
    assert Path(result["outputs"]["conflict_report_json"]).exists()
    assert Path(result["outputs"]["conflict_report_md"]).exists()
    csv_text = Path(result["outputs"]["candidate_training_dataset_csv"]).read_text(encoding="utf-8")
    assert "CCO,0.81" in csv_text
    assert "CCN" not in csv_text


def test_confirm_extracted_dataset_adapter_requires_human_confirmation(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "merged_candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.81\n", encoding="utf-8")

    result = confirm_extracted_dataset_adapter(
        {
            "run_id": "r-confirm",
            "candidate_training_dataset_csv": str(candidate_csv),
            "output_dir": str(tmp_path / "confirmed"),
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "confirmation_required"


def test_confirm_extracted_dataset_adapter_does_not_treat_false_string_as_confirmation(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "merged_candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.81\n", encoding="utf-8")

    result = confirm_extracted_dataset_adapter(
        {
            "run_id": "r-confirm-string-false",
            "candidate_training_dataset_csv": str(candidate_csv),
            "output_dir": str(tmp_path / "confirmed"),
            "confirmed": "false",
            "actor": "user",
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "confirmation_required"


def test_confirm_extracted_dataset_adapter_blocks_missing_review_reports_when_confirmed(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "merged_candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.81\n", encoding="utf-8")

    result = confirm_extracted_dataset_adapter(
        {
            "run_id": "r-confirm-missing-reports",
            "candidate_training_dataset_csv": str(candidate_csv),
            "output_dir": str(tmp_path / "confirmed"),
            "confirmed": True,
            "actor": "user",
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "confirmation_blocked"
    assert "missing_conflict_report" in result["error"]["blocking_reasons"]
    assert "missing_citation_provenance_report" in result["error"]["blocking_reasons"]


def test_confirm_extracted_dataset_adapter_blocks_unresolved_conflicts_and_license_review(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "merged_candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.81\n", encoding="utf-8")
    conflict_report = tmp_path / "conflict_report.json"
    conflict_report.write_text(json.dumps({"conflict_count": 1}), encoding="utf-8")
    provenance_report = tmp_path / "provenance_report.json"
    provenance_report.write_text(json.dumps({"unknown_license_count": 1}), encoding="utf-8")

    result = confirm_extracted_dataset_adapter(
        {
            "run_id": "r-confirm-blocked",
            "candidate_training_dataset_csv": str(candidate_csv),
            "conflict_report_json": str(conflict_report),
            "citation_provenance_report_json": str(provenance_report),
            "output_dir": str(tmp_path / "confirmed"),
            "confirmed": True,
            "actor": "user",
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "confirmation_blocked"
    assert "unresolved_conflicts" in result["error"]["blocking_reasons"]
    assert "license_review_required" in result["error"]["blocking_reasons"]


def test_confirm_extracted_dataset_adapter_writes_confirmed_training_dataset(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "merged_candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.81\nCCN,0.7\n", encoding="utf-8")
    conflict_report = tmp_path / "conflict_report.json"
    conflict_report.write_text(json.dumps({"conflict_count": 0}), encoding="utf-8")
    provenance_report = tmp_path / "provenance_report.json"
    provenance_report.write_text(json.dumps({"unknown_license_count": 0}), encoding="utf-8")

    result = confirm_extracted_dataset_adapter(
        {
            "run_id": "r-confirm-ok",
            "dataset_id": "lit_dataset_v1",
            "candidate_training_dataset_csv": str(candidate_csv),
            "conflict_report_json": str(conflict_report),
            "citation_provenance_report_json": str(provenance_report),
            "output_dir": str(tmp_path / "confirmed"),
            "confirmed": True,
            "actor": "user",
            "note": "manual review complete",
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "confirm_extracted_dataset"
    assert result["confirmation_record"]["status"] == "confirmed"
    assert result["confirmation_record"]["record_count"] == 2
    assert result["confirmation_record"]["confirmed_by"] == "user"
    confirmed_csv = Path(result["outputs"]["confirmed_training_dataset_csv"])
    assert confirmed_csv.exists()
    assert confirmed_csv.read_text(encoding="utf-8") == candidate_csv.read_text(encoding="utf-8")
    assert Path(result["outputs"]["confirmation_record_json"]).exists()
    assert Path(result["outputs"]["human_confirmation_report_md"]).exists()


def test_literature_to_dataset_workflow_runs_to_confirmation_gate(tmp_path: Path) -> None:
    pdf_dir, raw_root = _write_mineru_table_fixture(tmp_path)

    result = literature_to_dataset_workflow_adapter(
        {
            "run_id": "r-lit-workflow",
            "input_pdf_dir": str(pdf_dir),
            "mineru_output_root": str(raw_root),
            "output_dir": str(tmp_path / "workflow"),
            "query": "SMILES PLQY table",
            "property_columns": ["PLQY"],
            "default_license": "CC-BY-4.0",
        }
    )

    assert result["status"] == "waiting_confirmation"
    assert result["adapter"] == "literature_to_dataset_workflow"
    assert result["workflow_report"]["stage_statuses"]["parse_pdf_folder"] == "success"
    assert result["workflow_report"]["stage_statuses"]["confirm_extracted_dataset"] == "skipped"
    assert result["workflow_report"]["pending_confirmation"] is True
    assert Path(result["outputs"]["candidate_training_dataset_csv"]).exists()
    assert Path(result["outputs"]["workflow_report_json"]).exists()
    assert Path(result["outputs"]["workflow_summary_md"]).exists()
    csv_text = Path(result["outputs"]["candidate_training_dataset_csv"]).read_text(encoding="utf-8")
    assert "CCO,0.8" in csv_text
    assert "confirmed_training_dataset_csv" not in result["outputs"]


def test_literature_to_dataset_workflow_confirms_when_approved(tmp_path: Path) -> None:
    pdf_dir, raw_root = _write_mineru_table_fixture(tmp_path)

    result = literature_to_dataset_workflow_adapter(
        {
            "run_id": "r-lit-confirmed",
            "input_pdf_dir": str(pdf_dir),
            "mineru_output_root": str(raw_root),
            "output_dir": str(tmp_path / "workflow"),
            "query": "SMILES PLQY table",
            "property_columns": ["PLQY"],
            "default_license": "CC-BY-4.0",
            "confirmed": True,
            "actor": "user",
            "dataset_id": "lit_confirmed_v1",
        }
    )

    assert result["status"] == "success"
    assert result["workflow_report"]["stage_statuses"]["confirm_extracted_dataset"] == "success"
    assert result["workflow_report"]["pending_confirmation"] is False
    assert Path(result["outputs"]["confirmed_training_dataset_csv"]).exists()
    assert Path(result["outputs"]["confirmation_record_json"]).exists()


def test_check_public_dataset_leakage_adapter_reports_benchmark_overlap(tmp_path: Path) -> None:
    train_csv = _write_csv(tmp_path / "confirmed_training_dataset.csv", [{"SMILES": "CCO"}, {"SMILES": "CCN"}])
    benchmark_a = _write_csv(tmp_path / "benchmark_a.csv", [{"smiles": "NCC"}, {"smiles": "CCC"}])
    benchmark_b = _write_csv(tmp_path / "benchmark_b.csv", [{"smiles": "COC"}])

    result = check_public_dataset_leakage_adapter(
        {
            "run_id": "r-public-leakage",
            "training_dataset_csv": str(train_csv),
            "public_dataset_csvs": [str(benchmark_a), str(benchmark_b)],
            "output_dir": str(tmp_path / "leakage"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "check_public_dataset_leakage"
    assert result["report"]["dataset_count"] == 2
    assert result["report"]["total_overlap_count"] == 1
    assert result["report"]["status"] == "overlap_detected"
    assert result["report"]["datasets"][0]["overlap_count"] == 1
    assert result["report"]["datasets"][0]["overlap_smiles"] == ["CCN"]
    assert Path(result["outputs"]["benchmark_contamination_report_json"]).exists()
    report_md = Path(result["outputs"]["benchmark_contamination_report_md"])
    assert report_md.exists()
    assert "CCN" in report_md.read_text(encoding="utf-8")


def test_normalize_extracted_units_adapter_writes_normalized_records_and_report(tmp_path: Path) -> None:
    records = [
        ExtractedRecord(
            record_id="rec_000001",
            smiles="CCO",
            properties={"plqy": 80.0, "lambda_em_nm": 520.0},
            source_id="paper-unit",
            paper_id="paper-unit",
            page=1,
            table_id="table_1",
            row_index=0,
            evidence_ref="paper-unit:table_1",
            citation_context="paper-unit p.1 table_1",
            confidence=0.95,
            confidence_factors={},
            raw_values={"SMILES": "CCO", "PLQY (%)": "80", "lambda_em (nm)": "520"},
        )
    ]
    records_jsonl = tmp_path / "extracted_records.jsonl"
    records_jsonl.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json")) for record in records) + "\n",
        encoding="utf-8",
    )

    result = normalize_extracted_units_adapter(
        {
            "run_id": "r-normalize",
            "extracted_records_jsonl": str(records_jsonl),
            "output_dir": str(tmp_path / "normalization"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "normalize_extracted_units"
    assert result["report"]["conversion_count"] == 2
    normalized = result["records"][0]
    assert normalized["properties"] == {"plqy": 0.8, "lambda_em": 520.0}
    assert normalized["raw_values"]["PLQY (%)"] == "80"
    assert Path(result["outputs"]["normalized_extracted_records_jsonl"]).exists()
    assert Path(result["outputs"]["normalized_candidate_training_dataset_csv"]).exists()
    report_md = Path(result["outputs"]["unit_normalization_report_md"])
    assert report_md.exists()
    assert "PLQY (%)" in report_md.read_text(encoding="utf-8")


def test_evaluate_extraction_benchmark_adapter_computes_phase3_metrics(tmp_path: Path) -> None:
    hit = EvidenceHit(
        source_id="paper-metrics",
        page=1,
        element_id="table_1",
        element_type="table",
        retrieval_channel="table",
        score=2.0,
        text_or_table_ref="paper-metrics:table_1",
        citation_context="paper-metrics p.1 table_1",
        metadata={"chunk_id": "paper-metrics:table_1"},
    )
    hits_json = tmp_path / "evidence_hits.json"
    hits_json.write_text(json.dumps({"run_id": "r-metrics", "hits": [hit.model_dump(mode="json")]}), encoding="utf-8")

    extracted = [
        ExtractedRecord(
            record_id="rec_000001",
            smiles="CCO",
            properties={"plqy": 0.8},
            source_id="paper-metrics",
            paper_id="paper-metrics",
            page=1,
            table_id="table_1",
            row_index=0,
            evidence_ref="paper-metrics:table_1",
            citation_context="paper-metrics p.1 table_1",
            confidence=0.95,
            confidence_factors={},
            raw_values={"SMILES": "CCO", "PLQY": "0.8"},
        ),
        ExtractedRecord(
            record_id="rec_000002",
            smiles="CCC",
            properties={"plqy": 0.4},
            source_id="paper-metrics",
            paper_id="paper-metrics",
            page=1,
            table_id="table_1",
            row_index=1,
            evidence_ref="paper-metrics:table_1",
            citation_context="paper-metrics p.1 table_1",
            confidence=0.9,
            confidence_factors={},
            raw_values={"SMILES": "CCC", "PLQY": "0.4"},
        ),
    ]
    extracted_jsonl = tmp_path / "extracted_records.jsonl"
    extracted_jsonl.write_text("\n".join(json.dumps(record.model_dump(mode="json")) for record in extracted) + "\n", encoding="utf-8")
    gold_jsonl = tmp_path / "gold_records.jsonl"
    gold_jsonl.write_text(json.dumps(extracted[0].model_dump(mode="json")) + "\n", encoding="utf-8")
    conflict_report = tmp_path / "conflict_report.json"
    conflict_report.write_text(json.dumps({"input_record_count": 4, "conflict_count": 1}), encoding="utf-8")
    extraction_confidence_report = tmp_path / "extraction_confidence_report.json"
    extraction_confidence_report.write_text(json.dumps({"rejected_record_count": 2}), encoding="utf-8")
    citation_report = tmp_path / "citation_provenance_report.json"
    citation_report.write_text(json.dumps({"unknown_license_count": 1}), encoding="utf-8")
    candidate_csv = _write_csv(tmp_path / "candidate_training_dataset.csv", [{"smiles": "CCO", "plqy": 0.8}, {"smiles": "CCC", "plqy": 0.4}])
    before_metrics = tmp_path / "before_metrics.json"
    before_metrics.write_text(json.dumps({"properties": [{"property_id": "plqy", "metrics": {"r2": 0.1, "mae": 0.3}}]}), encoding="utf-8")
    after_metrics = tmp_path / "after_metrics.json"
    after_metrics.write_text(json.dumps({"properties": [{"property_id": "plqy", "metrics": {"r2": 0.25, "mae": 0.2}}]}), encoding="utf-8")

    result = evaluate_extraction_benchmark_adapter(
        {
            "run_id": "r-metrics",
            "evidence_hits_json": str(hits_json),
            "gold_evidence_refs": ["paper-metrics:table_1", "paper-metrics:missing_table"],
            "extracted_records_jsonl": str(extracted_jsonl),
            "gold_records_jsonl": str(gold_jsonl),
            "conflict_report_json": str(conflict_report),
            "extraction_confidence_report_json": str(extraction_confidence_report),
            "citation_provenance_report_json": str(citation_report),
            "candidate_training_dataset_csv": str(candidate_csv),
            "model_metrics_before_json": str(before_metrics),
            "model_metrics_after_json": str(after_metrics),
            "output_dir": str(tmp_path / "metrics"),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "evaluate_extraction_benchmark"
    assert result["report"]["retrieval_recall"] == 0.5
    assert result["report"]["extraction_precision"] == 0.5
    assert result["report"]["conflict_rate"] == 0.25
    assert result["report"]["confirmation_workload_count"] == 4
    assert result["report"]["trainable_labels_gained"] == 2
    assert result["report"]["downstream_model_performance_delta"] == {"plqy.mae": -0.1, "plqy.r2": 0.15}
    assert Path(result["outputs"]["extraction_benchmark_report_json"]).exists()
    report_md = Path(result["outputs"]["extraction_benchmark_report_md"])
    assert report_md.exists()
    assert "retrieval_recall" in report_md.read_text(encoding="utf-8")


def test_evaluate_extraction_benchmark_adapter_reports_zero_precision_when_gold_exists_but_no_extractions(tmp_path: Path) -> None:
    extracted_jsonl = tmp_path / "empty_extracted_records.jsonl"
    extracted_jsonl.write_text("", encoding="utf-8")
    gold = ExtractedRecord(
        record_id="rec_gold_001",
        smiles="CCO",
        properties={"plqy": 0.8},
        source_id="paper-gold",
        paper_id="paper-gold",
        page=1,
        table_id="table_1",
        row_index=0,
        evidence_ref="paper-gold:table_1",
        citation_context="paper-gold p.1 table_1",
        confidence=1.0,
        confidence_factors={},
        raw_values={"SMILES": "CCO", "PLQY": "0.8"},
    )
    gold_jsonl = tmp_path / "gold_records.jsonl"
    gold_jsonl.write_text(json.dumps(gold.model_dump(mode="json")) + "\n", encoding="utf-8")

    result = evaluate_extraction_benchmark_adapter(
        {
            "run_id": "r-metrics-empty",
            "extracted_records_jsonl": str(extracted_jsonl),
            "gold_records_jsonl": str(gold_jsonl),
            "output_dir": str(tmp_path / "metrics"),
        }
    )

    assert result["status"] == "success"
    assert result["report"]["extraction_precision"] == 0.0
    assert result["report"]["metric_statuses"]["extraction_precision"] == "computed"
