from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.adapters.phase3 import _sha256_file, _write_markdown
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseProvider,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.mineru_api_client import MinerUApiClient, MinerUApiError
from ai4s_agent.mineru_output_normalizer import (
    build_output_refs,
    discover_mineru_output_bundle,
    normalize_mineru_output_bundle,
)


class MinerUApiDocumentParseProvider(DocumentParseProvider):
    provider_name = "mineru_api"

    def __init__(self, *, client: MinerUApiClient, check_health: bool = True) -> None:
        self.client = client
        self.check_health = bool(check_health)

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        resolved = request.resolve_paths()
        resolved.output_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = (resolved.output_dir / "mineru_bundle").resolve()
        parser_backend = f"mineru_api:{request.backend}"
        task_status_history: list[str] = []
        queued_history: list[int] = []
        extracted_relative_paths: list[str] = []
        warnings: list[str] = []
        remote_task_id = ""
        mineru_version = ""
        protocol_version = ""
        try:
            self.client.validate_upload_policy(request)
            if self.check_health:
                health_payload = self.client.health()
                mineru_version = str(health_payload.get("version_name") or health_payload.get("_version_name") or "").strip()
                protocol_version = str(health_payload.get("protocol_version") or "").strip()
            outcome = self.client.parse_pdf(
                request=request,
                input_pdf=resolved.input_pdf,
                output_dir=bundle_dir,
            )
            remote_task_id = outcome.remote_task_id
            task_status_history = list(outcome.task_status_history)
            queued_history = list(outcome.queued_ahead_history)
            extracted_relative_paths = list(outcome.extracted_relative_paths)
            if outcome.mineru_version:
                mineru_version = outcome.mineru_version
            if outcome.protocol_version:
                protocol_version = outcome.protocol_version
            parser_backend = f"mineru_api:{outcome.backend or request.backend}"
            bundle = discover_mineru_output_bundle(outcome.output_dir)
            normalized = normalize_mineru_output_bundle(
                input_pdf=resolved.input_pdf,
                bundle=bundle,
                parser_backend=parser_backend,
            )
            warnings = list(normalized.warnings)
            parsed_document_json = write_json(
                resolved.output_dir / f"{request.run_id}_parsed_document.json",
                normalized.parsed_document.model_dump(mode="json"),
            )
            parsed_document_markdown = _write_markdown(
                resolved.output_dir / f"{request.run_id}_parsed_document.md",
                normalized.parsed_document,
            )
            audit = DocumentParseAudit(
                source_pdf_sha256=_sha256_file(resolved.input_pdf),
                request_provider=request.provider,
                selected_provider=self.provider_name,
                selection_reason="explicit_mineru_api_provider",
                parser_backend=parser_backend,
                task_status_history=task_status_history,
                queued_ahead_history=queued_history,
                extracted_relative_paths=extracted_relative_paths,
                warnings=warnings,
                api_base_url=self.client.base_url,
                mineru_version=mineru_version,
                protocol_version=protocol_version,
            )
            audit_path = write_json(
                resolved.output_dir / f"{request.run_id}_parser_audit.json",
                {
                    "run_id": request.run_id,
                    "provider": self.provider_name,
                    "parser_backend": parser_backend,
                    "input_pdf": str(resolved.input_pdf),
                    "remote_task_id": remote_task_id,
                    "source_pdf_sha256": audit.source_pdf_sha256,
                    "api_base_url": self.client.base_url,
                    "task_status_history": task_status_history,
                    "queued_ahead_history": queued_history,
                    "extracted_relative_paths": extracted_relative_paths,
                    "warnings": warnings,
                    "mineru_version": mineru_version,
                    "protocol_version": protocol_version,
                    "created_at": now_iso(),
                },
            )
            outputs = build_output_refs(
                output_dir=resolved.output_dir,
                parsed_document_json=parsed_document_json,
                parsed_document_markdown=parsed_document_markdown,
                parser_audit_json=audit_path,
                bundle=bundle,
            )
            return DocumentParseResult(
                ok=True,
                status="success",
                provider=self.provider_name,
                parser_backend=parser_backend,
                run_id=request.run_id,
                input_pdf=str(resolved.input_pdf),
                parsed_document=normalized.parsed_document,
                outputs=outputs,
                remote_task_id=remote_task_id,
                warnings=warnings,
                error=None,
                audit=audit,
            )
        except (MinerUApiError, OSError, ValueError) as exc:
            error = _error_from_exception(exc)
            if isinstance(exc, MinerUApiError):
                remote_task_id = str(exc.details.get("task_id") or remote_task_id)
                task_status_history = [
                    str(item).strip()
                    for item in (exc.details.get("task_status_history") or task_status_history)
                    if str(item).strip()
                ]
                queued_history = [
                    int(item)
                    for item in (exc.details.get("queued_ahead_history") or queued_history)
                ]
            bundle = discover_mineru_output_bundle_fallback(bundle_dir)
            if bundle_dir.exists() and any(bundle_dir.iterdir()):
                try:
                    bundle = discover_mineru_output_bundle(bundle_dir)
                except ValueError:
                    bundle = discover_mineru_output_bundle_fallback(bundle_dir)
            audit = DocumentParseAudit(
                source_pdf_sha256=_sha256_file(resolved.input_pdf) if resolved.input_pdf.exists() else "",
                request_provider=request.provider,
                selected_provider=self.provider_name,
                selection_reason="explicit_mineru_api_provider",
                parser_backend=parser_backend,
                task_status_history=task_status_history,
                queued_ahead_history=queued_history,
                extracted_relative_paths=extracted_relative_paths,
                warnings=warnings,
                api_base_url=self.client.base_url,
                mineru_version=mineru_version,
                protocol_version=protocol_version,
            )
            audit_path = write_json(
                resolved.output_dir / f"{request.run_id}_parser_audit.json",
                {
                    "run_id": request.run_id,
                    "provider": self.provider_name,
                    "parser_backend": parser_backend,
                    "input_pdf": str(resolved.input_pdf),
                    "remote_task_id": remote_task_id,
                    "source_pdf_sha256": audit.source_pdf_sha256,
                    "api_base_url": self.client.base_url,
                    "task_status_history": task_status_history,
                    "queued_ahead_history": queued_history,
                    "extracted_relative_paths": extracted_relative_paths,
                    "warnings": warnings,
                    "mineru_version": mineru_version,
                    "protocol_version": protocol_version,
                    "error": error.model_dump(mode="json"),
                    "created_at": now_iso(),
                },
            )
            return DocumentParseResult(
                ok=False,
                status="failed",
                provider=self.provider_name,
                parser_backend=parser_backend,
                run_id=request.run_id,
                input_pdf=str(resolved.input_pdf),
                parsed_document=None,
                outputs=build_output_refs(
                    output_dir=resolved.output_dir,
                    parsed_document_json=resolved.output_dir / f"{request.run_id}_parsed_document.json",
                    parsed_document_markdown=resolved.output_dir / f"{request.run_id}_parsed_document.md",
                    parser_audit_json=audit_path,
                    bundle=bundle,
                ),
                remote_task_id=remote_task_id,
                warnings=warnings,
                error=error,
                audit=audit,
            )


def discover_mineru_output_bundle_fallback(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    from ai4s_agent.mineru_output_normalizer import MinerUOutputBundle

    return MinerUOutputBundle(output_dir=str(output_dir))


def _error_from_exception(exc: BaseException) -> DocumentParseError:
    if isinstance(exc, MinerUApiError):
        return DocumentParseError(code=exc.code, message=exc.message, details=exc.details)
    return DocumentParseError(
        code="mineru_parse_failed",
        message=str(exc).strip() or exc.__class__.__name__,
        details={"exception_type": exc.__class__.__name__},
    )
