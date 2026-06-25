from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.schemas import ParsedDocument


def test_document_parse_request_is_json_safe_and_has_conservative_defaults(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")
    request = DocumentParseRequest(
        run_id="r-doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )

    payload = request.model_dump(mode="json")

    assert json.loads(json.dumps(payload)) == payload
    assert request.parse_method == "auto"
    assert request.backend == "hybrid-engine"
    assert request.effort == "medium"
    assert request.formula_enabled is True
    assert request.table_enabled is True
    assert request.image_analysis_enabled is False
    assert request.allow_remote_upload is False


def test_document_parse_request_rejects_unknown_fields(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")

    with pytest.raises(ValidationError):
        DocumentParseRequest(
            run_id="r-doc",
            input_pdf=str(pdf),
            output_dir=str(tmp_path / "out"),
            provider="pdfplumber",
            unexpected="boom",
        )


def test_document_parse_request_rejects_missing_output_dir_field(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")

    with pytest.raises(ValidationError):
        DocumentParseRequest(
            run_id="r-doc",
            input_pdf=str(pdf),
            provider="pdfplumber",
        )


def test_document_parse_request_resolve_paths_rejects_missing_pdf(tmp_path: Path) -> None:
    request = DocumentParseRequest(
        run_id="r-doc",
        input_pdf=str(tmp_path / "missing.pdf"),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )

    with pytest.raises(ValueError, match="input_pdf does not exist"):
        request.resolve_paths()


def test_document_parse_request_resolve_paths_rejects_non_pdf_files(tmp_path: Path) -> None:
    txt = tmp_path / "paper.txt"
    txt.write_text("not a pdf", encoding="utf-8")
    request = DocumentParseRequest(
        run_id="r-doc",
        input_pdf=str(txt),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )

    with pytest.raises(ValueError, match="input_pdf must be a PDF file"):
        request.resolve_paths()


def test_document_parse_result_has_stable_schema() -> None:
    parsed = ParsedDocument(
        paper_id="paper",
        source_path="/tmp/paper.pdf",
        parser_backend="pdfplumber_local",
        metadata={},
        pages=[{"page": 1}],
        elements=[],
        tables=[],
    )
    result = DocumentParseResult(
        ok=True,
        status="success",
        provider="pdfplumber",
        parser_backend="pdfplumber_local",
        run_id="r-doc",
        input_pdf="/tmp/paper.pdf",
        parsed_document=parsed,
        outputs=DocumentParseOutputRefs(
            output_dir="/tmp/out",
            parsed_document_json="/tmp/out/r-doc_parsed_document.json",
            parsed_document_markdown="/tmp/out/r-doc_parsed_document.md",
            parser_audit_json="/tmp/out/r-doc_parser_audit.json",
        ),
        remote_task_id="",
        warnings=[],
        error=None,
        audit=DocumentParseAudit(
            source_pdf_sha256="sha256:" + ("a" * 64),
            request_provider="pdfplumber",
            task_status_history=[],
            queued_ahead_history=[],
            extracted_relative_paths=[],
            warnings=[],
        ),
    )

    payload = result.model_dump(mode="json")

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert payload["provider"] == "pdfplumber"
    assert payload["parser_backend"] == "pdfplumber_local"
    assert payload["outputs"]["parsed_document_json"].endswith("_parsed_document.json")
    assert payload["error"] is None
    assert payload["audit"]["request_provider"] == "pdfplumber"


def test_document_parse_error_details_must_be_json_safe() -> None:
    with pytest.raises(ValidationError):
        DocumentParseError(
            code="bad",
            message="bad",
            details={"path": Path("/tmp/not-json-safe")},
        )
