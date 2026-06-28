from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.custom_corpus_manifest import (
    CustomCorpusManifest,
    CustomCorpusManifestError,
    load_custom_corpus_manifest,
    safe_manifest_report_summary,
    sha256_file,
)
from ai4s_agent.document_parse_corpus_live_acceptance import (
    CorpusLiveWorkflowSummary,
    _configuration_errors,
    _endpoint_profile_summary,
    _parse_document,
    _resolve_cli_endpoint,
    _workflow_rel,
    _workflow_summary,
)
from ai4s_agent.document_parse_live_acceptance import (
    DocumentParseProviderAcceptance,
    _provider_acceptance,
    _redact_details,
    _redacted_origin,
    _rel,
    _run_id_error,
    _safe_origin,
    _service,
)
from ai4s_agent.document_parse_provider import DocumentParseRequest
from ai4s_agent.mineru_endpoint_profiles import (
    MinerUEndpointProfileConfigError,
    MinerUEndpointProfileReportSummary,
)
from ai4s_agent.mineru_preflight_binding import (
    PreflightBindingSummary,
    contains_credential_marker,
    load_and_bind_preflight_report,
)
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.workflows.corpus_to_phase1_workflow import CorpusToPhase1WorkflowResult, run_corpus_to_phase1_workflow


CustomCorpusDryRunDecision = Literal["passed", "failed"]
CustomCorpusEndpointKind = Literal["mineru_api", "mineru_router"]

_SCHEMA_VERSION = "custom_corpus_dry_run.v1"
_EXPECTED_PROTOCOL_VERSION = "2"


class CustomCorpusManifestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str = ""
    manifest_sha256: str = ""
    document_count: int = 0
    pdf_hash_coverage: dict[str, int] = Field(default_factory=dict)
    source_policy: str = ""
    redaction_policy: dict[str, bool] = Field(default_factory=dict)
    documents: list[str] = Field(default_factory=list)


class CustomCorpusParseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted: int = 0
    success: int = 0
    failed: int = 0
    parsed_document_count: int = 0
    mineru_protocol_versions: list[str] = Field(default_factory=list)
    parser_warning_count: int = 0


class CustomCorpusAuditSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_record_count: int = 0
    accepted_record_count: int = 0
    rejected_record_count: int = 0
    consistent_duplicate_count: int = 0
    conflict_count: int = 0
    unresolved_conflict_count: int = 0


class CustomCorpusConfirmationBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_confirmation_confirmed: bool = False
    phase1_status: str = "not_run"
    training_dataset_admitted: bool = False


class CustomCorpusDryRunParseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_pdf_sha256: str = ""
    mineru: DocumentParseProviderAcceptance
    pdfplumber: DocumentParseProviderAcceptance | None = None
    parsed_document_path: str = ""


class CustomCorpusDryRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    generated_at: str = ""
    decision: CustomCorpusDryRunDecision
    corpus_id: str = ""
    corpus_class: str = ""
    redacted_api_origin: str = ""
    endpoint_kind: CustomCorpusEndpointKind = "mineru_api"
    requested_backend: str = "hybrid-engine"
    requested_effort: str = "medium"
    requested_parse_method: str = "auto"
    endpoint_profile: MinerUEndpointProfileReportSummary = Field(default_factory=MinerUEndpointProfileReportSummary)
    preflight_binding: PreflightBindingSummary = Field(default_factory=PreflightBindingSummary)
    manifest_summary: CustomCorpusManifestSummary = Field(default_factory=CustomCorpusManifestSummary)
    parse_summary: CustomCorpusParseSummary = Field(default_factory=CustomCorpusParseSummary)
    parse_results: list[CustomCorpusDryRunParseEntry] = Field(default_factory=list)
    corpus_audit_summary: CustomCorpusAuditSummary = Field(default_factory=CustomCorpusAuditSummary)
    confirmation_boundary: CustomCorpusConfirmationBoundary = Field(default_factory=CustomCorpusConfirmationBoundary)
    corpus_report_json_path: str = ""
    corpus_report_md_path: str = ""
    corpus_replay_manifest_path: str = ""
    corpus_reproducibility_report_path: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)


def run_custom_corpus_dry_run(
    *,
    manifest: str | Path,
    output_dir: str | Path,
    run_id: str,
    api_url: str,
    endpoint_kind: CustomCorpusEndpointKind = "mineru_api",
    backend: str = "hybrid-engine",
    effort: str = "medium",
    parse_method: str = "auto",
    allow_remote_upload: bool = False,
    compare_pdfplumber: bool = False,
    api_token: str = "",
    http_timeout_sec: float = 60.0,
    task_timeout_sec: float = 900.0,
    poll_interval_sec: float = 1.0,
    max_poll_attempts: int = 600,
    service: Any | None = None,
    transport: httpx.BaseTransport | None = None,
    workflow_runner: Any | None = None,
    generated_at: str | None = None,
    n_bits: int = 256,
    topn: int = 10,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    endpoint_profile_summary: dict[str, Any] | MinerUEndpointProfileReportSummary | None = None,
    preflight_report_path: str | Path | None = None,
    require_preflight_match: bool = False,
    preflight_artifact_sha256: str = "",
) -> CustomCorpusDryRunReport:
    generated = generated_at or now_iso()
    clean_run_id = str(run_id or "").strip()
    root = Path(output_dir).expanduser().resolve()
    run_id_error = _run_id_error(clean_run_id)
    run_root = (root / clean_run_id).resolve() if run_id_error is None else root
    warnings = ["custom corpus dry-run only; not production dataset admission or training data admission"]
    errors: list[dict[str, Any]] = []
    manifest_path = Path(manifest).expanduser()
    manifest_model: CustomCorpusManifest | None = None
    origin = ""
    preflight_binding = PreflightBindingSummary()
    forbidden_report_values: list[str] = []

    if run_id_error is not None:
        errors.append(_error(run_id_error.code, run_id_error.message))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            api_url="",
            endpoint_kind=endpoint_kind,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, root)
        return report
    if run_root.parent != root:
        errors.append(_error("invalid_run_id", "run_id must stay directly under the dry-run output root"))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            api_url="",
            endpoint_kind=endpoint_kind,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, root)
        return report
    if run_root.exists() and any(run_root.iterdir()):
        errors.append(_error("output_directory_not_empty", "run-specific custom corpus dry-run output directory must be empty"))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            api_url="",
            endpoint_kind=endpoint_kind,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report

    run_root.mkdir(parents=True, exist_ok=True)
    try:
        origin = _redacted_origin(api_url)
    except ValueError as exc:
        errors.append(_error("invalid_api_url", str(exc)))

    try:
        manifest_model = load_custom_corpus_manifest(manifest_path)
    except CustomCorpusManifestError as exc:
        errors.append(_error("invalid_manifest", str(exc)))
    except Exception as exc:
        errors.append(_error("invalid_manifest", exc.__class__.__name__))

    config_errors = _configuration_errors(
        http_timeout_sec=http_timeout_sec,
        task_timeout_sec=task_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        max_poll_attempts=max_poll_attempts,
    )
    errors.extend(config_errors)

    manifest_summary = CustomCorpusManifestSummary()
    if manifest_model is not None:
        manifest_summary = CustomCorpusManifestSummary.model_validate(
            safe_manifest_report_summary(manifest_model, manifest_path=manifest_path)
        )
        forbidden_report_values.extend(
            str(Path(document.pdf_path).expanduser().resolve()) for document in manifest_model.documents if document.pdf_path
        )
        forbidden_report_values.extend(
            str(Path(document.pdf_path).expanduser().resolve().parent) for document in manifest_model.documents if document.pdf_path
        )
        forbidden_report_values.append(str(manifest_path.expanduser().resolve()))
        forbidden_report_values.append(str(manifest_path.expanduser().resolve().parent))

    if manifest_model is not None:
        errors.extend(_pdf_preflight_errors(manifest_model))

    if preflight_report_path and not errors:
        preflight_binding, preflight_warnings, preflight_errors = load_and_bind_preflight_report(
            preflight_report_path=preflight_report_path,
            preflight_artifact_sha256=preflight_artifact_sha256,
            require_preflight_match=bool(require_preflight_match),
            expected_origin=origin,
            endpoint_profile_summary=endpoint_profile_summary,
            expected_protocol_version=_EXPECTED_PROTOCOL_VERSION,
            failure_message="preflight report does not match this custom corpus dry-run endpoint",
        )
        warnings.extend(preflight_warnings)
        errors.extend(preflight_errors)

    if errors:
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            api_url=origin,
            endpoint_kind=endpoint_kind,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            preflight_binding=preflight_binding,
            manifest_summary=manifest_summary,
            warnings=warnings,
            errors=errors,
        )
        report = _enforce_report_redaction(report, forbidden_report_values)
        _persist_report(report, run_root)
        return report

    assert manifest_model is not None
    parsed_documents_dir = run_root / "parsed_documents"
    parsed_documents_dir.mkdir(parents=True, exist_ok=True)
    parse_service = service or _service(
        api_url=api_url,
        api_token=api_token,
        transport=transport,
        http_timeout_sec=http_timeout_sec,
        task_timeout_sec=task_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        max_poll_attempts=max_poll_attempts,
        monotonic=None,
        sleep=None,
    )

    parse_results: list[CustomCorpusDryRunParseEntry] = []
    parsed_document_paths: list[str] = []
    for document in manifest_model.documents:
        pdf_path = Path(document.pdf_path).expanduser().resolve()
        mineru_result, mineru_elapsed = _parse_document(
            service=parse_service,
            request=DocumentParseRequest(
                run_id=f"{clean_run_id}-{document.document_id}-mineru",
                input_pdf=str(pdf_path),
                output_dir=str(run_root / "mineru_bundles" / document.document_id),
                provider="mineru_api",
                parse_method=parse_method,
                backend=backend,
                effort=effort,
                allow_remote_upload=allow_remote_upload,
            ),
        )
        source_hash = document.pdf_sha256 or sha256_file(pdf_path)
        mineru = _provider_acceptance(
            result=mineru_result,
            elapsed_seconds=mineru_elapsed,
            root=run_root,
            source_pdf_sha256=source_hash,
        )
        parsed_document_path = ""
        if mineru_result.ok and mineru_result.parsed_document is not None:
            parsed_document_abs = parsed_documents_dir / f"{document.document_id}_parsed_document.json"
            write_json(parsed_document_abs, mineru_result.parsed_document.model_dump(mode="json"))
            parsed_document_path = _rel(parsed_document_abs, run_root)
            parsed_document_paths.append(str(parsed_document_abs))
        elif mineru_result.ok and mineru_result.outputs.parsed_document_json:
            parsed_document_abs = parsed_documents_dir / f"{document.document_id}_parsed_document.json"
            shutil.copy2(Path(mineru_result.outputs.parsed_document_json), parsed_document_abs)
            parsed_document_path = _rel(parsed_document_abs, run_root)
            parsed_document_paths.append(str(parsed_document_abs))

        pdfplumber = None
        if compare_pdfplumber:
            pdf_result, pdf_elapsed = _parse_document(
                service=parse_service,
                request=DocumentParseRequest(
                    run_id=f"{clean_run_id}-{document.document_id}-pdfplumber",
                    input_pdf=str(pdf_path),
                    output_dir=str(run_root / "pdfplumber_baselines" / document.document_id),
                    provider="pdfplumber",
                ),
            )
            pdfplumber = _provider_acceptance(
                result=pdf_result,
                elapsed_seconds=pdf_elapsed,
                root=run_root,
                source_pdf_sha256=source_hash,
            )

        entry = CustomCorpusDryRunParseEntry(
            document_id=document.document_id,
            source_pdf_sha256=source_hash,
            mineru=mineru,
            pdfplumber=pdfplumber,
            parsed_document_path=parsed_document_path,
        )
        parse_results.append(entry)
        errors.extend(_parse_entry_errors(entry))

    workflow_summary = CorpusLiveWorkflowSummary()
    workflow_result: CorpusToPhase1WorkflowResult | None = None
    if not errors:
        confirmation = DatasetConfirmation(
            confirmed=False,
            confirmed_by="",
            confirmation_source="custom-corpus-dry-run",
            confirmation_timestamp=generated,
        )
        try:
            runner = workflow_runner or run_corpus_to_phase1_workflow
            workflow_result = runner(
                parsed_document_paths=[Path(path) for path in parsed_document_paths],
                output_dir=run_root / "corpus_workflow",
                run_id=clean_run_id,
                confirmation=confirmation,
                generated_at=generated,
                property_ids=["plqy"],
                n_bits=n_bits,
                topn=topn,
                min_numeric_ratio=min_numeric_ratio,
                min_nonempty=min_nonempty,
            )
            workflow_summary = _workflow_summary(workflow_result)
            if workflow_summary.phase1_status != "not_run":
                errors.append(_error("phase1_ran_for_custom_corpus", "custom corpus dry-run unexpectedly reached Phase 1"))
        except Exception as exc:
            errors.append(
                _error(
                    "corpus_workflow_failed",
                    str(exc).strip() or exc.__class__.__name__,
                    {"exception_type": exc.__class__.__name__},
                )
            )

    report = CustomCorpusDryRunReport(
        run_id=clean_run_id,
        generated_at=generated,
        decision="failed" if errors else "passed",
        corpus_id=manifest_model.corpus_id,
        corpus_class=manifest_model.corpus_class,
        redacted_api_origin=origin,
        endpoint_kind=endpoint_kind,
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        endpoint_profile=_endpoint_profile_summary(endpoint_profile_summary),
        preflight_binding=preflight_binding,
        manifest_summary=manifest_summary,
        parse_summary=_parse_summary(parse_results),
        parse_results=parse_results,
        corpus_audit_summary=_corpus_audit_summary(workflow_summary),
        confirmation_boundary=CustomCorpusConfirmationBoundary(
            dataset_confirmation_confirmed=False,
            phase1_status=workflow_summary.phase1_status,
            training_dataset_admitted=False,
        ),
        corpus_report_json_path=_workflow_rel(workflow_result, "corpus_report_json", run_root),
        corpus_report_md_path=_workflow_rel(workflow_result, "corpus_report_md", run_root),
        corpus_replay_manifest_path=_workflow_rel(workflow_result, "corpus_replay_manifest_json", run_root),
        corpus_reproducibility_report_path=_workflow_rel(workflow_result, "corpus_reproducibility_report_json", run_root),
        warnings=warnings,
        errors=errors,
        outputs={
            "parsed_documents": _rel(parsed_documents_dir, run_root),
            "mineru_bundles": _rel(run_root / "mineru_bundles", run_root),
            "corpus_workflow": _rel(run_root / "corpus_workflow", run_root),
        },
    )
    if compare_pdfplumber:
        report.outputs["pdfplumber_baselines"] = _rel(run_root / "pdfplumber_baselines", run_root)
    report = _enforce_report_redaction(report, forbidden_report_values)
    _persist_report(report, run_root)
    return report


def _pdf_preflight_errors(manifest: CustomCorpusManifest) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for document in manifest.documents:
        path = Path(document.pdf_path).expanduser()
        if not path.exists() or not path.is_file():
            errors.append(_error("missing_pdf", "manifest PDF is missing", {"document_id": document.document_id}))
            continue
        if path.suffix.lower() != ".pdf":
            errors.append(_error("invalid_pdf_path", "manifest document must reference a PDF", {"document_id": document.document_id}))
            continue
        if document.pdf_sha256:
            observed = sha256_file(path)
            if observed != document.pdf_sha256:
                errors.append(
                    _error(
                        "pdf_sha256_mismatch",
                        "manifest PDF SHA-256 does not match local file",
                        {"document_id": document.document_id, "observed_sha256": observed, "expected_sha256": document.pdf_sha256},
                    )
                )
    return errors


def _parse_entry_errors(entry: CustomCorpusDryRunParseEntry) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    mineru = entry.mineru
    if not mineru.ok:
        details = _redact_details(mineru.error.details if mineru.error else {})
        errors.append(
            _error(
                str(mineru.error.code if mineru.error else "mineru_parse_failed"),
                str(mineru.error.message if mineru.error else "MinerU parsing failed"),
                {"document_id": entry.document_id, **details},
            )
        )
        return errors
    if not mineru.protocol_version:
        errors.append(_error("missing_protocol_version", "MinerU API protocol version is absent", {"document_id": entry.document_id}))
    elif str(mineru.protocol_version).strip() != _EXPECTED_PROTOCOL_VERSION:
        errors.append(
            _error(
                "unsupported_protocol_version",
                f"MinerU API protocol version must be {_EXPECTED_PROTOCOL_VERSION}",
                {"document_id": entry.document_id, "observed": mineru.protocol_version},
            )
        )
    if not entry.parsed_document_path:
        errors.append(_error("missing_parsed_document", "ParsedDocument output is absent", {"document_id": entry.document_id}))
    return errors


def _parse_summary(entries: list[CustomCorpusDryRunParseEntry]) -> CustomCorpusParseSummary:
    protocols = sorted(
        {
            str(entry.mineru.protocol_version).strip()
            for entry in entries
            if str(entry.mineru.protocol_version).strip()
        }
    )
    return CustomCorpusParseSummary(
        attempted=len(entries),
        success=sum(1 for entry in entries if entry.mineru.ok),
        failed=sum(1 for entry in entries if not entry.mineru.ok),
        parsed_document_count=sum(1 for entry in entries if entry.parsed_document_path),
        mineru_protocol_versions=protocols,
        parser_warning_count=sum(len(entry.mineru.warnings) + (len(entry.pdfplumber.warnings) if entry.pdfplumber else 0) for entry in entries),
    )


def _corpus_audit_summary(summary: CorpusLiveWorkflowSummary) -> CustomCorpusAuditSummary:
    return CustomCorpusAuditSummary(
        extracted_record_count=summary.extracted_record_count,
        accepted_record_count=summary.accepted_record_count,
        rejected_record_count=summary.rejected_record_count,
        consistent_duplicate_count=summary.consistent_duplicate_count,
        conflict_count=summary.conflict_count,
        unresolved_conflict_count=summary.unresolved_conflict_count,
    )


def _report(
    *,
    run_id: str,
    generated_at: str,
    decision: CustomCorpusDryRunDecision,
    api_url: str,
    endpoint_kind: CustomCorpusEndpointKind,
    backend: str,
    effort: str,
    parse_method: str,
    endpoint_profile_summary: dict[str, Any] | MinerUEndpointProfileReportSummary | None = None,
    preflight_binding: PreflightBindingSummary | None = None,
    manifest_summary: CustomCorpusManifestSummary | None = None,
    warnings: list[str],
    errors: list[dict[str, Any]],
) -> CustomCorpusDryRunReport:
    return CustomCorpusDryRunReport(
        run_id=run_id,
        generated_at=generated_at,
        decision=decision,
        redacted_api_origin=_safe_origin(api_url),
        endpoint_kind=endpoint_kind,
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        endpoint_profile=_endpoint_profile_summary(endpoint_profile_summary),
        preflight_binding=preflight_binding or PreflightBindingSummary(),
        manifest_summary=manifest_summary or CustomCorpusManifestSummary(),
        warnings=warnings,
        errors=errors,
    )


def _persist_report(report: CustomCorpusDryRunReport, run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = write_json(run_root / "dry_run_report.json", report.model_dump(mode="json"))
    summary_path = run_root / "dry_run_summary.md"
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    report.outputs["dry_run_report"] = _rel(report_path, run_root)
    report.outputs["dry_run_summary"] = _rel(summary_path, run_root)
    write_json(report_path, report.model_dump(mode="json"))


def _summary_markdown(report: CustomCorpusDryRunReport) -> str:
    lines = [
        f"# Custom Corpus Dry-Run: {report.run_id}",
        "",
        f"- decision: {report.decision}",
        f"- corpus_id: {report.corpus_id}",
        f"- corpus_class: {report.corpus_class}",
        f"- redacted_api_origin: {report.redacted_api_origin}",
        f"- documents: {report.manifest_summary.document_count}",
        f"- parses: {report.parse_summary.success}/{report.parse_summary.attempted}",
        f"- Phase 1 status: {report.confirmation_boundary.phase1_status}",
        f"- training_dataset_admitted: {str(report.confirmation_boundary.training_dataset_admitted).lower()}",
        "",
        "Custom corpus dry-run only. This is not production dataset admission or training data admission.",
    ]
    if report.errors:
        lines.append("")
        lines.append("## Errors")
        for error in report.errors:
            lines.append(f"- {error.get('code')}: {error.get('message')}")
    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in report.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _enforce_report_redaction(
    report: CustomCorpusDryRunReport,
    forbidden_values: list[str],
) -> CustomCorpusDryRunReport:
    forbidden = _normalized_forbidden_values(forbidden_values)
    if not _contains_forbidden_material(report, forbidden):
        return report
    minimal = _minimal_safe_failure_report(report, forbidden)
    if _contains_forbidden_material(minimal, forbidden):
        minimal = _ultra_minimal_safe_failure_report(report, forbidden)
    return minimal


def _minimal_safe_failure_report(
    report: CustomCorpusDryRunReport,
    forbidden_values: list[str],
) -> CustomCorpusDryRunReport:
    endpoint_profile = report.endpoint_profile
    if _contains_forbidden_material(endpoint_profile, forbidden_values):
        endpoint_profile = MinerUEndpointProfileReportSummary()
    preflight_binding = report.preflight_binding
    if _contains_forbidden_material(preflight_binding, forbidden_values):
        preflight_binding = PreflightBindingSummary()
    manifest_summary = _safe_minimal_manifest_summary(report.manifest_summary, forbidden_values)
    warnings = [
        warning
        for warning in report.warnings
        if not _contains_forbidden_material(str(warning), forbidden_values)
    ]
    return CustomCorpusDryRunReport(
        run_id=report.run_id,
        generated_at=report.generated_at,
        decision="failed",
        corpus_id=report.corpus_id,
        corpus_class=report.corpus_class,
        redacted_api_origin=report.redacted_api_origin,
        endpoint_kind=report.endpoint_kind,
        requested_backend=report.requested_backend,
        requested_effort=report.requested_effort,
        requested_parse_method=report.requested_parse_method,
        endpoint_profile=endpoint_profile,
        preflight_binding=preflight_binding,
        manifest_summary=manifest_summary,
        parse_summary=CustomCorpusParseSummary(
            attempted=report.parse_summary.attempted,
            success=report.parse_summary.success,
            failed=report.parse_summary.failed,
            parsed_document_count=report.parse_summary.parsed_document_count,
            parser_warning_count=report.parse_summary.parser_warning_count,
        ),
        confirmation_boundary=CustomCorpusConfirmationBoundary(
            dataset_confirmation_confirmed=False,
            phase1_status=_safe_phase1_status(report.confirmation_boundary.phase1_status),
            training_dataset_admitted=False,
        ),
        warnings=warnings,
        errors=[_error("report_redaction_failed", "dry-run report contained forbidden private path material")],
        outputs={},
    )


def _ultra_minimal_safe_failure_report(
    report: CustomCorpusDryRunReport,
    forbidden_values: list[str],
) -> CustomCorpusDryRunReport:
    run_id = report.run_id if not _contains_forbidden_material(report.run_id, forbidden_values) else "[redacted-run-id]"
    redacted_origin = (
        report.redacted_api_origin
        if not _contains_forbidden_material(report.redacted_api_origin, forbidden_values)
        else ""
    )
    return CustomCorpusDryRunReport(
        run_id=run_id,
        generated_at=report.generated_at,
        decision="failed",
        corpus_id="",
        corpus_class="",
        redacted_api_origin=redacted_origin,
        endpoint_kind=report.endpoint_kind,
        requested_backend=report.requested_backend,
        requested_effort=report.requested_effort,
        requested_parse_method=report.requested_parse_method,
        confirmation_boundary=CustomCorpusConfirmationBoundary(
            dataset_confirmation_confirmed=False,
            phase1_status="not_run",
            training_dataset_admitted=False,
        ),
        errors=[_error("report_redaction_failed", "dry-run report contained forbidden private path material")],
        outputs={},
    )


def _safe_minimal_manifest_summary(
    summary: CustomCorpusManifestSummary,
    forbidden_values: list[str],
) -> CustomCorpusManifestSummary:
    manifest_path = str(summary.manifest_path or "").strip()
    if _contains_forbidden_material(manifest_path, forbidden_values):
        manifest_path = Path(manifest_path).name if manifest_path else ""
    documents = [
        label if not _contains_forbidden_material(label, forbidden_values) else "[redacted-document]"
        for label in summary.documents
    ]
    return CustomCorpusManifestSummary(
        manifest_path=manifest_path,
        document_count=summary.document_count,
        pdf_hash_coverage=dict(summary.pdf_hash_coverage),
        documents=documents,
    )


def _safe_phase1_status(value: str) -> str:
    clean = str(value or "").strip()
    return clean if clean in {"not_run", "success", "failed", "awaiting_confirmation"} else "not_run"


def _normalized_forbidden_values(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if len(clean) <= 4:
            continue
        if clean not in normalized:
            normalized.append(clean)
    return normalized


def _contains_forbidden_material(payload: Any, forbidden_values: list[str]) -> bool:
    if isinstance(payload, BaseModel):
        return _contains_forbidden_material(payload.model_dump(mode="json"), forbidden_values)
    if isinstance(payload, dict):
        return any(_contains_forbidden_material(value, forbidden_values) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_forbidden_material(value, forbidden_values) for value in payload)
    if isinstance(payload, str):
        if contains_credential_marker(payload):
            return True
        return any(value in payload for value in forbidden_values)
    return False


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "code": str(code or "error").strip(),
        "message": str(message or "").strip(),
        "details": _redact_details(details or {}),
    }


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        resolved = _resolve_cli_endpoint(args, parser=parser)
    except MinerUEndpointProfileConfigError as exc:
        report = _report(
            run_id=str(args.run_id or "").strip(),
            generated_at=now_iso(),
            decision="failed",
            api_url="",
            endpoint_kind="mineru_api",
            backend="hybrid-engine",
            effort="medium",
            parse_method="auto",
            warnings=["custom corpus dry-run only; not production dataset admission or training data admission"],
            errors=[_error("endpoint_profile_error", str(exc))],
        )
        _persist_report(report, _cli_report_root(args.output, args.run_id))
        output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 1
    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
    report = run_custom_corpus_dry_run(
        manifest=args.manifest,
        run_id=args.run_id,
        output_dir=args.output,
        api_url=resolved["api_url"],
        endpoint_kind=resolved["endpoint_kind"],
        backend=resolved["backend"],
        effort=resolved["effort"],
        parse_method=resolved["parse_method"],
        allow_remote_upload=resolved["allow_remote_upload"],
        compare_pdfplumber=resolved["compare_pdfplumber"],
        api_token=token,
        http_timeout_sec=float(resolved["http_timeout_sec"]),
        task_timeout_sec=float(resolved["task_timeout_sec"]),
        poll_interval_sec=float(resolved["poll_interval_sec"]),
        max_poll_attempts=int(resolved["max_poll_attempts"]),
        endpoint_profile_summary=resolved["endpoint_profile_summary"],
        preflight_report_path=args.preflight_report,
        require_preflight_match=bool(args.require_preflight_match),
        preflight_artifact_sha256=args.preflight_artifact_sha256,
        transport=transport,
        n_bits=int(args.n_bits),
        topn=int(args.topn),
        min_numeric_ratio=float(args.min_numeric_ratio),
        min_nonempty=int(args.min_nonempty),
    )
    output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    output.write("\n")
    if report.outputs.get("dry_run_report"):
        err.write(f"dry-run report: {_cli_report_root(args.output, args.run_id) / report.outputs['dry_run_report']}\n")
    return 0 if report.decision == "passed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_dry_run",
        description="Dry-run a custom local PDF corpus without Phase 1 admission.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--endpoint-profile-file", default="")
    parser.add_argument("--endpoint-profile", default=None)
    parser.add_argument("--routing-policy", default=None)
    parser.add_argument("--endpoint-kind", default=None, choices=["mineru-api", "mineru-router"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--effort", default=None, choices=["medium", "high"])
    parser.add_argument("--parse-method", default=None)
    parser.add_argument("--allow-remote-upload", action="store_true", default=None)
    parser.add_argument("--compare-pdfplumber", action="store_true", default=None)
    parser.add_argument("--http-timeout-sec", type=float, default=None)
    parser.add_argument("--task-timeout-sec", type=float, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=None)
    parser.add_argument("--max-poll-attempts", type=int, default=None)
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min-numeric-ratio", type=float, default=0.6)
    parser.add_argument("--min-nonempty", type=int, default=30)
    parser.add_argument("--preflight-report", default="")
    parser.add_argument("--preflight-artifact-sha256", default="")
    parser.add_argument("--require-preflight-match", action="store_true")
    return parser


def _cli_report_root(output_dir: str | Path, run_id: str) -> Path:
    root = Path(output_dir).expanduser().resolve()
    clean_run_id = str(run_id or "").strip()
    if _run_id_error(clean_run_id) is not None:
        return root / "invalid-run-id"
    return root / clean_run_id


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
