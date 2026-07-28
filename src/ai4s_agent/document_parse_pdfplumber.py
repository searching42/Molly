from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso
from ai4s_agent.adapters.phase3 import _sha256_file, parse_document_pdfplumber_adapter
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseProvider,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.schemas import ParsedDocument


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class PdfPlumberDocumentParseProvider(DocumentParseProvider):
    provider_name = "pdfplumber"

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        resolved = request.resolve_paths()
        result = parse_document_pdfplumber_adapter(
            {
                "run_id": request.run_id,
                "input_pdf": str(resolved.input_pdf),
                "output_dir": str(resolved.output_dir),
            }
        )
        status = str(result.get("status") or "").strip().lower()
        outputs_payload = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        outputs = DocumentParseOutputRefs(
            output_dir=str(resolved.output_dir),
            parsed_document_json=str(outputs_payload.get("parsed_document_json") or ""),
            parsed_document_markdown=str(outputs_payload.get("parsed_document_markdown") or ""),
            parser_audit_json=str(outputs_payload.get("parser_audit_json") or ""),
        )
        warnings: list[str] = []
        if status == "success":
            parsed_document = ParsedDocument.model_validate(result.get("parsed_document") or {})
            audit_json = _read_json_object(Path(outputs.parser_audit_json)) if outputs.parser_audit_json else {}
            if isinstance(audit_json.get("warnings"), list):
                warnings = [str(item).strip() for item in audit_json["warnings"] if str(item).strip()]
            audit = DocumentParseAudit(
                source_pdf_sha256=_sha256_file(resolved.input_pdf),
                request_provider=request.provider,
                selected_provider=self.provider_name,
                selection_reason="explicit_pdfplumber_provider",
                parser_backend="pdfplumber_local",
                task_status_history=["success"],
                queued_ahead_history=[],
                extracted_relative_paths=[],
                warnings=warnings,
            )
            return DocumentParseResult(
                ok=True,
                status="success",
                provider=self.provider_name,
                parser_backend="pdfplumber_local",
                run_id=request.run_id,
                input_pdf=str(resolved.input_pdf),
                parsed_document=parsed_document,
                outputs=outputs,
                remote_task_id="",
                warnings=warnings,
                error=None,
                audit=audit,
            )
        error_payload = result.get("error") if isinstance(result.get("error"), dict) else {}
        error = DocumentParseError(
            code=str(error_payload.get("code") or "pdfplumber_parse_failed"),
            message=str(error_payload.get("message") or "pdfplumber parsing failed"),
            details={
                "provider": self.provider_name,
                "adapter": "parse_document_pdfplumber",
                "status": status or "failed",
                "failed_at": now_iso(),
            },
        )
        audit = DocumentParseAudit(
            source_pdf_sha256=_sha256_file(resolved.input_pdf) if resolved.input_pdf.exists() else "",
            request_provider=request.provider,
            selected_provider=self.provider_name,
            selection_reason="explicit_pdfplumber_provider",
            parser_backend="pdfplumber_local",
            task_status_history=[status or "failed"],
            queued_ahead_history=[],
            extracted_relative_paths=[],
            warnings=[],
        )
        return DocumentParseResult(
            ok=False,
            status=status or "failed",
            provider=self.provider_name,
            parser_backend="pdfplumber_local",
            run_id=request.run_id,
            input_pdf=str(resolved.input_pdf),
            parsed_document=None,
            outputs=outputs,
            remote_task_id="",
            warnings=[],
            error=error,
            audit=audit,
        )
