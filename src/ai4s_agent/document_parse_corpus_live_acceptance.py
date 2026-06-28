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
from ai4s_agent.corpus_live_acceptance_fixtures import write_synthetic_live_corpus_pdfs
from ai4s_agent.document_parse_live_acceptance import (
    DocumentParseProviderAcceptance,
    _parse_with_service,
    _provider_acceptance,
    _provider_exception_result,
    _redact_details,
    _redacted_origin,
    _rel,
    _run_id_error,
    _safe_origin,
    _service,
)
from ai4s_agent.document_parse_provider import DocumentParseRequest, DocumentParseResult
from ai4s_agent.mineru_endpoint_profiles import (
    MinerUEndpointProfileConfigError,
    MinerUEndpointProfileReportSummary,
    load_mineru_endpoint_profile_config,
    resolve_mineru_endpoint_profile,
)
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.workflows.corpus_to_phase1_workflow import CorpusToPhase1WorkflowResult, run_corpus_to_phase1_workflow


CorpusLiveDecision = Literal["passed", "failed", "awaiting_confirmation"]
CorpusEndpointKind = Literal["mineru_api", "mineru_router"]

_SCHEMA_VERSION = "document_parse_corpus_live_acceptance.v1"
_SOURCE_FIXTURE = "synthetic_oled_multi_paper_corpus_v1"
_EXPECTED_PROTOCOL_VERSION = "2"


class CorpusLiveParseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    paper_id: str
    source_pdf_path: str
    source_pdf_sha256: str
    expected_record_count: int
    mineru: DocumentParseProviderAcceptance
    pdfplumber: DocumentParseProviderAcceptance | None = None
    parsed_document_path: str = ""


class CorpusLiveWorkflowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = ""
    document_count: int = 0
    extracted_record_count: int = 0
    accepted_record_count: int = 0
    rejected_record_count: int = 0
    candidate_record_count: int = 0
    training_record_count: int = 0
    consistent_duplicate_count: int = 0
    conflict_count: int = 0
    unresolved_conflict_count: int = 0
    phase1_status: str = "not_run"
    top_ranked_candidate_count: int = 0


class CorpusLiveAcceptanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    generated_at: str
    decision: CorpusLiveDecision
    endpoint_kind: CorpusEndpointKind
    redacted_api_origin: str
    requested_backend: str
    requested_effort: str
    requested_parse_method: str
    endpoint_profile: MinerUEndpointProfileReportSummary = Field(default_factory=MinerUEndpointProfileReportSummary)
    source_fixture: str = _SOURCE_FIXTURE
    source_pdf_sha256_values: dict[str, str] = Field(default_factory=dict)
    parse_results: list[CorpusLiveParseEntry] = Field(default_factory=list)
    corpus_workflow: CorpusLiveWorkflowSummary = Field(default_factory=CorpusLiveWorkflowSummary)
    corpus_report_json_path: str = ""
    corpus_report_md_path: str = ""
    corpus_replay_manifest_path: str = ""
    corpus_reproducibility_report_path: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)


def run_document_parse_corpus_live_acceptance(
    *,
    run_id: str,
    output_dir: str | Path,
    api_url: str,
    endpoint_kind: CorpusEndpointKind = "mineru_api",
    backend: str = "hybrid-engine",
    effort: str = "medium",
    parse_method: str = "auto",
    allow_remote_upload: bool = False,
    compare_pdfplumber: bool = True,
    confirm_synthetic_dataset: bool = False,
    confirmed_by: str = "",
    api_token: str = "",
    http_timeout_sec: float = 60.0,
    task_timeout_sec: float = 900.0,
    poll_interval_sec: float = 1.0,
    max_poll_attempts: int = 600,
    service: Any | None = None,
    transport: httpx.BaseTransport | None = None,
    generated_at: str | None = None,
    n_bits: int = 256,
    topn: int = 10,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    endpoint_profile_summary: dict[str, Any] | MinerUEndpointProfileReportSummary | None = None,
) -> CorpusLiveAcceptanceReport:
    generated = generated_at or now_iso()
    clean_run_id = str(run_id or "").strip()
    root = Path(output_dir).expanduser().resolve()
    run_id_error = _run_id_error(clean_run_id)
    run_root = (root / clean_run_id).resolve() if run_id_error is None else root
    warnings = ["manual opt-in synthetic corpus acceptance only; not production scientific accuracy evidence"]
    errors: list[dict[str, Any]] = []

    if run_id_error is not None:
        errors.append(_error(run_id_error.code, run_id_error.message))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=api_url,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, root)
        return report
    if run_root.parent != root:
        errors.append(_error("invalid_run_id", "run_id must stay directly under the acceptance output root"))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=api_url,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, root)
        return report
    if run_root.exists() and any(run_root.iterdir()):
        errors.append(
            _error(
                "output_directory_not_empty",
                "run-specific acceptance output directory must be empty",
                {"output_dir": str(run_root)},
            )
        )
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=api_url,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report
    try:
        origin = _redacted_origin(api_url)
    except ValueError as exc:
        errors.append(_error("invalid_api_url", str(exc)))
        run_root.mkdir(parents=True, exist_ok=True)
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url="",
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report
    if confirm_synthetic_dataset and not str(confirmed_by or "").strip():
        errors.append(_error("missing_confirmed_by", "--confirmed-by is required with --confirm-synthetic-dataset"))
        run_root.mkdir(parents=True, exist_ok=True)
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=origin,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report
    config_errors = _configuration_errors(
        http_timeout_sec=http_timeout_sec,
        task_timeout_sec=task_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        max_poll_attempts=max_poll_attempts,
    )
    if config_errors:
        errors.extend(config_errors)
        run_root.mkdir(parents=True, exist_ok=True)
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=origin,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report

    run_root.mkdir(parents=True, exist_ok=True)
    generated_pdfs_dir = run_root / "generated_pdfs"
    parsed_documents_dir = run_root / "parsed_documents"
    parsed_documents_dir.mkdir(parents=True, exist_ok=True)
    try:
        corpus_pdfs = write_synthetic_live_corpus_pdfs(generated_pdfs_dir)
    except Exception as exc:
        errors.append(_error("fixture_generation_failed", str(exc).strip() or exc.__class__.__name__))
        report = _report(
            run_id=clean_run_id,
            generated_at=generated,
            decision="failed",
            endpoint_kind=endpoint_kind,
            api_url=origin,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            endpoint_profile_summary=endpoint_profile_summary,
            warnings=warnings,
            errors=errors,
        )
        _persist_report(report, run_root)
        return report

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
    parse_results: list[CorpusLiveParseEntry] = []
    parsed_document_paths: list[str] = []

    for item in corpus_pdfs:
        mineru_result, mineru_elapsed = _parse_document(
            service=parse_service,
            request=DocumentParseRequest(
                run_id=f"{clean_run_id}-{item.document_id}-mineru",
                input_pdf=item.pdf_path,
                output_dir=str(run_root / "mineru_bundles" / item.document_id),
                provider="mineru_api",
                parse_method=parse_method,
                backend=backend,
                effort=effort,
                allow_remote_upload=allow_remote_upload,
            ),
        )
        mineru = _provider_acceptance(
            result=mineru_result,
            elapsed_seconds=mineru_elapsed,
            root=run_root,
            source_pdf_sha256=item.sha256,
        )
        parsed_document_path = ""
        if mineru_result.ok and mineru_result.parsed_document is not None:
            parsed_document_abs = parsed_documents_dir / f"{item.document_id}_parsed_document.json"
            write_json(parsed_document_abs, mineru_result.parsed_document.model_dump(mode="json"))
            parsed_document_path = _rel(parsed_document_abs, run_root)
            parsed_document_paths.append(str(parsed_document_abs))
        elif mineru_result.ok and mineru_result.outputs.parsed_document_json:
            parsed_document_abs = parsed_documents_dir / f"{item.document_id}_parsed_document.json"
            shutil.copy2(Path(mineru_result.outputs.parsed_document_json), parsed_document_abs)
            parsed_document_path = _rel(parsed_document_abs, run_root)
            parsed_document_paths.append(str(parsed_document_abs))

        pdfplumber = None
        if compare_pdfplumber:
            pdf_result, pdf_elapsed = _parse_document(
                service=parse_service,
                request=DocumentParseRequest(
                    run_id=f"{clean_run_id}-{item.document_id}-pdfplumber",
                    input_pdf=item.pdf_path,
                    output_dir=str(run_root / "pdfplumber_baselines" / item.document_id),
                    provider="pdfplumber",
                ),
            )
            pdfplumber = _provider_acceptance(
                result=pdf_result,
                elapsed_seconds=pdf_elapsed,
                root=run_root,
                source_pdf_sha256=item.sha256,
            )

        entry = CorpusLiveParseEntry(
            document_id=item.document_id,
            paper_id=item.paper_id,
            source_pdf_path=_rel(Path(item.pdf_path), run_root),
            source_pdf_sha256=item.sha256,
            expected_record_count=item.expected_record_count,
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
            confirmed=bool(confirm_synthetic_dataset),
            confirmed_by=str(confirmed_by or "").strip() if confirm_synthetic_dataset else "",
            confirmation_source="synthetic-live-corpus-acceptance",
            confirmation_timestamp=generated,
        )
        try:
            workflow_result = run_corpus_to_phase1_workflow(
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
            errors.extend(
                _workflow_evidence_errors(
                    summary=workflow_summary,
                    workflow_result=workflow_result,
                    confirmation_expected=bool(confirm_synthetic_dataset),
                )
            )
        except Exception as exc:
            errors.append(
                _error(
                    "corpus_workflow_failed",
                    str(exc).strip() or exc.__class__.__name__,
                    {"exception_type": exc.__class__.__name__},
                )
            )

    decision = _decision(
        errors=errors,
        confirmation_expected=bool(confirm_synthetic_dataset),
        workflow_summary=workflow_summary,
    )
    report = CorpusLiveAcceptanceReport(
        run_id=clean_run_id,
        generated_at=generated,
        decision=decision,
        endpoint_kind=endpoint_kind,
        redacted_api_origin=origin,
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        endpoint_profile=_endpoint_profile_summary(endpoint_profile_summary),
        source_pdf_sha256_values={item.document_id: item.sha256 for item in corpus_pdfs},
        parse_results=parse_results,
        corpus_workflow=workflow_summary,
        corpus_report_json_path=_workflow_rel(workflow_result, "corpus_report_json", run_root),
        corpus_report_md_path=_workflow_rel(workflow_result, "corpus_report_md", run_root),
        corpus_replay_manifest_path=_workflow_rel(workflow_result, "corpus_replay_manifest_json", run_root),
        corpus_reproducibility_report_path=_workflow_rel(workflow_result, "corpus_reproducibility_report_json", run_root),
        warnings=warnings,
        errors=errors,
        outputs={
            "generated_pdfs": _rel(generated_pdfs_dir, run_root),
            "parsed_documents": _rel(parsed_documents_dir, run_root),
            "mineru_bundles": _rel(run_root / "mineru_bundles", run_root),
            "corpus_workflow": _rel(run_root / "corpus_workflow", run_root),
        },
    )
    if compare_pdfplumber:
        report.outputs["pdfplumber_baselines"] = _rel(run_root / "pdfplumber_baselines", run_root)
    _persist_report(report, run_root)
    return report


def _parse_document(*, service: Any, request: DocumentParseRequest) -> tuple[DocumentParseResult, float]:
    try:
        return _parse_with_service(service, request)
    except Exception as exc:
        result = _provider_exception_result(
            request_provider=request.provider,
            run_id=request.run_id,
            input_pdf=Path(request.input_pdf),
            output_dir=Path(request.output_dir),
            exc=exc,
        )
        return result, 0.0


def _parse_entry_errors(entry: CorpusLiveParseEntry) -> list[dict[str, Any]]:
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


def _workflow_summary(workflow_result: CorpusToPhase1WorkflowResult) -> CorpusLiveWorkflowSummary:
    workflow_report = _load_json(workflow_result.corpus_workflow_report_json)
    workflow_summary = workflow_report.get("summary") if isinstance(workflow_report.get("summary"), dict) else {}
    conflict_report = _load_json(workflow_result.corpus_conflict_report_json)
    conflict_summary = conflict_report.get("summary") if isinstance(conflict_report.get("summary"), dict) else {}
    manifest = _load_json(workflow_result.dataset_manifest_json)
    return CorpusLiveWorkflowSummary(
        status=str(workflow_result.status),
        document_count=int(workflow_summary.get("document_count") or conflict_summary.get("document_count") or 0),
        extracted_record_count=int(workflow_summary.get("extracted_record_count") or conflict_summary.get("input_record_count") or 0),
        accepted_record_count=int(workflow_summary.get("accepted_record_count") or conflict_summary.get("accepted_record_count") or 0),
        rejected_record_count=int(workflow_summary.get("rejected_record_count") or manifest.get("rejected_record_count") or 0),
        candidate_record_count=int(workflow_summary.get("candidate_record_count") or manifest.get("candidate_record_count") or 0),
        training_record_count=int(workflow_summary.get("training_record_count") or manifest.get("training_record_count") or 0),
        consistent_duplicate_count=int(conflict_summary.get("consistent_duplicate_count") or 0),
        conflict_count=int(workflow_summary.get("conflict_count") or conflict_summary.get("conflict_count") or 0),
        unresolved_conflict_count=int(
            workflow_summary.get("unresolved_conflict_count") or conflict_summary.get("unresolved_conflict_count") or 0
        ),
        phase1_status=str(workflow_summary.get("phase1_status") or "not_run"),
        top_ranked_candidate_count=int(workflow_summary.get("top_ranked_candidate_count") or 0),
    )


def _workflow_evidence_errors(
    *,
    summary: CorpusLiveWorkflowSummary,
    workflow_result: CorpusToPhase1WorkflowResult,
    confirmation_expected: bool,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if summary.status not in {"success", "awaiting_confirmation"}:
        errors.append(_error("corpus_workflow_failed", "corpus workflow did not complete"))
    if summary.consistent_duplicate_count < 1:
        errors.append(_error("expected_duplicate_missing", "corpus audit did not detect the expected consistent duplicate"))
    if summary.conflict_count < 1 or summary.unresolved_conflict_count < 1:
        errors.append(_error("expected_conflict_missing", "corpus audit did not detect the expected unresolved conflict"))
    if not _path_exists(workflow_result.corpus_replay_manifest_json):
        errors.append(_error("missing_replay_manifest", "corpus replay manifest is missing"))
    if not _path_exists(workflow_result.corpus_report_json):
        errors.append(_error("missing_corpus_report", "corpus report is missing"))
    if confirmation_expected and summary.phase1_status != "success":
        errors.append(_error("phase1_not_run", "confirmed synthetic dataset did not reach Phase 1"))
    if not confirmation_expected and summary.phase1_status != "not_run":
        errors.append(_error("phase1_ran_without_confirmation", "unconfirmed dataset unexpectedly reached Phase 1"))
    return errors


def _decision(
    *,
    errors: list[dict[str, Any]],
    confirmation_expected: bool,
    workflow_summary: CorpusLiveWorkflowSummary,
) -> CorpusLiveDecision:
    if errors:
        return "failed"
    if confirmation_expected and workflow_summary.phase1_status == "success":
        return "passed"
    return "awaiting_confirmation"


def _report(
    *,
    run_id: str,
    generated_at: str,
    decision: CorpusLiveDecision,
    endpoint_kind: CorpusEndpointKind,
    api_url: str,
    backend: str,
    effort: str,
    parse_method: str,
    endpoint_profile_summary: dict[str, Any] | MinerUEndpointProfileReportSummary | None = None,
    warnings: list[str],
    errors: list[dict[str, Any]],
) -> CorpusLiveAcceptanceReport:
    return CorpusLiveAcceptanceReport(
        run_id=run_id,
        generated_at=generated_at,
        decision=decision,
        endpoint_kind=endpoint_kind,
        redacted_api_origin=_safe_origin(api_url),
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        endpoint_profile=_endpoint_profile_summary(endpoint_profile_summary),
        warnings=warnings,
        errors=errors,
    )


def _endpoint_profile_summary(
    value: dict[str, Any] | MinerUEndpointProfileReportSummary | None,
) -> MinerUEndpointProfileReportSummary:
    if value is None:
        return MinerUEndpointProfileReportSummary()
    if isinstance(value, MinerUEndpointProfileReportSummary):
        return value
    return MinerUEndpointProfileReportSummary.model_validate(value)


def _persist_report(report: CorpusLiveAcceptanceReport, run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = write_json(run_root / "acceptance_report.json", report.model_dump(mode="json"))
    summary_path = run_root / "acceptance_summary.md"
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    report.outputs["acceptance_report"] = _rel(report_path, run_root)
    report.outputs["acceptance_summary"] = _rel(summary_path, run_root)
    write_json(report_path, report.model_dump(mode="json"))


def _summary_markdown(report: CorpusLiveAcceptanceReport) -> str:
    lines = [
        f"# MinerU Live Corpus Acceptance: {report.run_id}",
        "",
        f"- decision: {report.decision}",
        f"- endpoint_kind: {report.endpoint_kind}",
        f"- redacted_api_origin: {report.redacted_api_origin}",
        f"- documents: {len(report.parse_results)}",
        f"- corpus workflow: {report.corpus_workflow.status or 'not_run'}",
        f"- Phase 1 status: {report.corpus_workflow.phase1_status}",
        "",
        "Manual opt-in synthetic corpus acceptance only; not production scientific accuracy evidence.",
    ]
    if report.errors:
        lines.append("")
        lines.append("## Errors")
        for error in report.errors:
            lines.append(f"- {error.get('code')}: {error.get('message')}")
    return "\n".join(lines) + "\n"


def _configuration_errors(
    *,
    http_timeout_sec: float,
    task_timeout_sec: float,
    poll_interval_sec: float,
    max_poll_attempts: int,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for label, value in {
        "http_timeout_sec": http_timeout_sec,
        "task_timeout_sec": task_timeout_sec,
        "poll_interval_sec": poll_interval_sec,
    }.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed <= 0:
            errors.append(_error("configuration_error", f"{label} must be positive"))
    try:
        attempts = int(max_poll_attempts)
    except (TypeError, ValueError):
        attempts = 0
    if attempts <= 0:
        errors.append(_error("configuration_error", "max_poll_attempts must be positive"))
    return errors


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "code": str(code or "error").strip(),
        "message": str(message or "").strip(),
        "details": _redact_details(details or {}),
    }


def _workflow_rel(workflow_result: CorpusToPhase1WorkflowResult | None, attr: str, root: Path) -> str:
    if workflow_result is None:
        return ""
    value = str(getattr(workflow_result, attr, "") or "").strip()
    return _rel(Path(value), root) if value else ""


def _path_exists(path_like: str | Path) -> bool:
    return bool(str(path_like or "").strip()) and Path(path_like).expanduser().resolve().exists()


def _load_json(path_like: str | Path) -> dict[str, Any]:
    if not str(path_like or "").strip():
        return {}
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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
            endpoint_kind="mineru_api",
            api_url="",
            backend="hybrid-engine",
            effort="medium",
            parse_method="auto",
            warnings=["manual opt-in synthetic corpus acceptance only; not production scientific accuracy evidence"],
            errors=[_error("endpoint_profile_error", str(exc))],
        )
        _persist_report(report, _cli_report_root(args.output, args.run_id))
        output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        output.write("\n")
        err.write(f"acceptance report: {_cli_report_root(args.output, args.run_id) / report.outputs['acceptance_report']}\n")
        return 1
    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
    report = run_document_parse_corpus_live_acceptance(
        run_id=args.run_id,
        output_dir=args.output,
        api_url=resolved["api_url"],
        endpoint_kind=resolved["endpoint_kind"],
        backend=resolved["backend"],
        effort=resolved["effort"],
        parse_method=resolved["parse_method"],
        allow_remote_upload=resolved["allow_remote_upload"],
        compare_pdfplumber=resolved["compare_pdfplumber"],
        confirm_synthetic_dataset=bool(args.confirm_synthetic_dataset),
        confirmed_by=args.confirmed_by,
        api_token=token,
        http_timeout_sec=float(resolved["http_timeout_sec"]),
        task_timeout_sec=float(resolved["task_timeout_sec"]),
        poll_interval_sec=float(resolved["poll_interval_sec"]),
        max_poll_attempts=int(resolved["max_poll_attempts"]),
        endpoint_profile_summary=resolved["endpoint_profile_summary"],
        transport=transport,
        n_bits=int(args.n_bits),
        topn=int(args.topn),
        min_numeric_ratio=float(args.min_numeric_ratio),
        min_nonempty=int(args.min_nonempty),
    )
    output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    output.write("\n")
    if report.outputs.get("acceptance_report"):
        err.write(f"acceptance report: {_cli_report_root(args.output, args.run_id) / report.outputs['acceptance_report']}\n")
    if report.decision == "passed":
        return 0
    if report.decision == "awaiting_confirmation":
        return 2
    return 1


def _resolve_cli_endpoint(args: argparse.Namespace, *, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if args.endpoint_profile_file:
        config = load_mineru_endpoint_profile_config(args.endpoint_profile_file)
        resolved = resolve_mineru_endpoint_profile(
            config,
            profile_name=args.endpoint_profile,
            policy_name=args.routing_policy,
            profile_source_path=args.endpoint_profile_file,
            cli_overrides={
                "api_url": args.api_url,
                "endpoint_kind": args.endpoint_kind,
                "backend": args.backend,
                "effort": args.effort,
                "parse_method": args.parse_method,
                "allow_remote_upload": args.allow_remote_upload,
                "compare_pdfplumber": args.compare_pdfplumber,
                "http_timeout_sec": args.http_timeout_sec,
                "task_timeout_sec": args.task_timeout_sec,
                "poll_interval_sec": args.poll_interval_sec,
                "max_poll_attempts": args.max_poll_attempts,
            },
        )
        profile = resolved.profile
        return {
            "api_url": profile.api_url,
            "endpoint_kind": profile.endpoint_kind,
            "backend": profile.backend,
            "effort": profile.effort,
            "parse_method": profile.parse_method,
            "allow_remote_upload": profile.allow_remote_upload,
            "compare_pdfplumber": profile.compare_pdfplumber,
            "http_timeout_sec": profile.http_timeout_sec,
            "task_timeout_sec": profile.task_timeout_sec,
            "poll_interval_sec": profile.poll_interval_sec,
            "max_poll_attempts": profile.max_poll_attempts,
            "endpoint_profile_summary": resolved.redacted_summary(base_dir=Path.cwd()),
        }
    if not args.api_url:
        parser.error("--api-url is required unless --endpoint-profile-file is provided")
    return {
        "api_url": args.api_url,
        "endpoint_kind": str(args.endpoint_kind or "mineru-api").replace("-", "_"),
        "backend": args.backend or "hybrid-engine",
        "effort": args.effort or "medium",
        "parse_method": args.parse_method or "auto",
        "allow_remote_upload": bool(args.allow_remote_upload),
        "compare_pdfplumber": bool(args.compare_pdfplumber),
        "http_timeout_sec": 60.0 if args.http_timeout_sec is None else float(args.http_timeout_sec),
        "task_timeout_sec": 900.0 if args.task_timeout_sec is None else float(args.task_timeout_sec),
        "poll_interval_sec": 1.0 if args.poll_interval_sec is None else float(args.poll_interval_sec),
        "max_poll_attempts": 600 if args.max_poll_attempts is None else int(args.max_poll_attempts),
        "endpoint_profile_summary": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.document_parse_corpus_live_acceptance",
        description="Manual MinerU live corpus acceptance bridge for synthetic multi-paper PDFs.",
    )
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
    parser.add_argument("--confirm-synthetic-dataset", action="store_true")
    parser.add_argument("--confirmed-by", default="")
    parser.add_argument("--http-timeout-sec", type=float, default=None)
    parser.add_argument("--task-timeout-sec", type=float, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=None)
    parser.add_argument("--max-poll-attempts", type=int, default=None)
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min-numeric-ratio", type=float, default=0.6)
    parser.add_argument("--min-nonempty", type=int, default=30)
    return parser


def _cli_report_root(output_dir: str | Path, run_id: str) -> Path:
    root = Path(output_dir).expanduser().resolve()
    clean_run_id = str(run_id or "").strip()
    if _run_id_error(clean_run_id) is not None:
        return root / "invalid-run-id"
    return root / clean_run_id


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
