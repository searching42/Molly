from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.document_parse_service import DocumentParseService
from document_parse_test_helpers import write_synthetic_pdf


class _StubProvider:
    def __init__(self, provider_name: str, *, ok: bool = True, parser_backend: str = "stub") -> None:
        self.provider_name = provider_name
        self._ok = ok
        self._parser_backend = parser_backend
        self.client = SimpleNamespace(
            base_url="http://127.0.0.1:8000",
            configured=lambda: True,
        )

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        return DocumentParseResult(
            ok=self._ok,
            status="success" if self._ok else "failed",
            provider=self.provider_name,
            parser_backend=self._parser_backend,
            run_id=request.run_id,
            input_pdf=request.input_pdf,
            parsed_document=None,
            outputs=DocumentParseOutputRefs(output_dir=request.output_dir),
            remote_task_id="",
            warnings=[],
            error=None,
            audit=DocumentParseAudit(
                request_provider=request.provider,
                task_status_history=["success" if self._ok else "failed"],
                queued_ahead_history=[],
                extracted_relative_paths=[],
                warnings=[],
            ),
        )


def test_document_parse_service_selects_explicit_pdfplumber(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )
    service = DocumentParseService(
        mineru_provider=_StubProvider("mineru_api"),
        pdfplumber_provider=_StubProvider("pdfplumber", parser_backend="pdfplumber_local"),
    )

    result = service.parse(request)

    assert result.provider == "pdfplumber"
    assert result.audit.selected_provider == "pdfplumber"
    assert result.audit.selection_reason == "explicit_pdfplumber_provider"


def test_document_parse_service_selects_explicit_mineru(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )
    service = DocumentParseService(
        mineru_provider=_StubProvider("mineru_api", parser_backend="mineru_api:hybrid-engine"),
        pdfplumber_provider=_StubProvider("pdfplumber"),
    )

    result = service.parse(request)

    assert result.provider == "mineru_api"
    assert result.audit.selected_provider == "mineru_api"
    assert result.audit.selection_reason == "explicit_mineru_api_provider"


def test_document_parse_service_auto_selects_pdfplumber_when_mineru_upload_not_permitted(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="auto",
    )
    mineru = _StubProvider("mineru_api")
    mineru.client = SimpleNamespace(base_url="https://mineru.example.com", configured=lambda: True)
    service = DocumentParseService(
        mineru_provider=mineru,
        pdfplumber_provider=_StubProvider("pdfplumber"),
    )

    result = service.parse(request)

    assert result.provider == "pdfplumber"
    assert result.audit.selection_reason == "auto_selected_pdfplumber_baseline"


def test_document_parse_service_does_not_silently_fallback_after_mineru_failure(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )
    service = DocumentParseService(
        mineru_provider=_StubProvider("mineru_api", ok=False, parser_backend="mineru_api:hybrid-engine"),
        pdfplumber_provider=_StubProvider("pdfplumber"),
    )

    result = service.parse(request)

    assert result.ok is False
    assert result.provider == "mineru_api"
    assert result.audit.selected_provider == "mineru_api"
    assert result.audit.selection_reason == "explicit_mineru_api_provider"


def test_document_parse_service_rejects_explicit_mineru_without_client_configuration(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="doc",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )
    mineru = _StubProvider("mineru_api")
    mineru.client = SimpleNamespace(base_url="", configured=lambda: False)
    service = DocumentParseService(
        mineru_provider=mineru,
        pdfplumber_provider=_StubProvider("pdfplumber"),
    )

    with pytest.raises(ValueError, match="no MinerU API client is configured"):
        service.parse(request)
