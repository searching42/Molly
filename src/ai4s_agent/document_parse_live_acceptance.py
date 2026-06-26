from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.adapters.phase3 import _sha256_file
from ai4s_agent.document_parse_benchmark import DocumentParseBenchmarkReport, evaluate_document_parse_against_gold
from ai4s_agent.document_parse_mineru import MinerUApiDocumentParseProvider
from ai4s_agent.document_parse_pdfplumber import PdfPlumberDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseError, DocumentParseRequest, DocumentParseResult
from ai4s_agent.document_parse_service import DocumentParseService
from ai4s_agent.mineru_api_client import MinerUApiClient

AcceptanceDecision = Literal["passed", "needs_review", "failed"]
EndpointKind = Literal["mineru_api", "mineru_router", "compatible_endpoint"]

_SCHEMA_VERSION = "document_parse_live_acceptance.v1"
_SOURCE_FIXTURE = "synthetic_oled_table_v1"
_GOLD: dict[str, Any] = {
    "page_count": 1,
    "text_contains": ["Synthetic", "OLED", "PLQY"],
    "tables": [
        {
            "caption": "Table 1 OLED measurements",
            "headers": ["SMILES", "PLQY", "lambda_em"],
            "rows": [{"SMILES": "CCO", "PLQY": "0.65", "lambda_em": "520"}],
        }
    ],
}


class DocumentParseAcceptanceError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()


class DocumentParseAcceptanceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_text_token_recall: float = 0.80
    header_match_rate: float = 1.0
    simple_cell_exact_match_rate: float = 0.90
    page_present: float = 1.0
    table_page_present: float = 1.0


class DocumentParseProviderAcceptance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    parser_backend: str = ""
    source_pdf_sha256: str = ""
    elapsed_seconds: float = 0.0
    remote_task_id: str = ""
    task_status_history: list[str] = Field(default_factory=list)
    queued_ahead_history: list[int] = Field(default_factory=list)
    mineru_version: str = ""
    protocol_version: str = ""
    output_bundle_refs: dict[str, str] = Field(default_factory=dict)
    markdown_path: str = ""
    content_list_path: str = ""
    content_list_v2_path: str = ""
    middle_json_path: str = ""
    parsed_document_path: str = ""
    parser_audit_path: str = ""
    benchmark_result: DocumentParseBenchmarkReport | None = None
    warnings: list[str] = Field(default_factory=list)
    error: DocumentParseAcceptanceError | None = None


class DocumentParseComparisonSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_success: dict[str, bool]
    parser_backends: dict[str, str]
    elapsed_seconds: dict[str, float]
    page_counts: dict[str, int]
    table_counts: dict[str, int]
    normalized_text_token_recall: dict[str, float]
    header_match_rate: dict[str, float]
    row_count_match: dict[str, bool]
    simple_cell_exact_match_rate: dict[str, float]
    provenance_completeness: dict[str, dict[str, float]]
    warning_counts: dict[str, int]
    mineru_better_fields: list[str] = Field(default_factory=list)
    pdfplumber_better_fields: list[str] = Field(default_factory=list)
    equal_fields: list[str] = Field(default_factory=list)


class DocumentParseLiveAcceptanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    generated_at: str
    decision: AcceptanceDecision
    endpoint_kind: EndpointKind
    redacted_api_origin: str
    requested_backend: str
    requested_effort: str
    requested_parse_method: str
    source_pdf_sha256: str = ""
    source_fixture: str = _SOURCE_FIXTURE
    mineru: DocumentParseProviderAcceptance
    pdfplumber: DocumentParseProviderAcceptance | None = None
    comparison: DocumentParseComparisonSummary | None = None
    thresholds: DocumentParseAcceptanceThresholds = Field(default_factory=DocumentParseAcceptanceThresholds)
    warnings: list[str] = Field(default_factory=list)
    errors: list[DocumentParseAcceptanceError] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)


def run_document_parse_live_acceptance(
    *,
    run_id: str,
    output_dir: str | Path,
    api_url: str,
    endpoint_kind: EndpointKind = "mineru_api",
    backend: str = "hybrid-engine",
    effort: str = "medium",
    parse_method: str = "auto",
    allow_remote_upload: bool = False,
    compare_pdfplumber: bool = True,
    thresholds: DocumentParseAcceptanceThresholds | None = None,
    api_token: str = "",
    http_timeout_sec: float = 30.0,
    task_timeout_sec: float = 300.0,
    poll_interval_sec: float = 1.0,
    max_poll_attempts: int = 120,
    transport: httpx.BaseTransport | None = None,
    monotonic: Any | None = None,
    sleep: Any | None = None,
) -> DocumentParseLiveAcceptanceReport:
    threshold_model = thresholds or DocumentParseAcceptanceThresholds()
    clean_run_id = str(run_id or "").strip()
    root = Path(output_dir).expanduser().resolve()
    run_root = root / clean_run_id if clean_run_id else root / "missing-run-id"
    if run_root.exists() and any(run_root.iterdir()):
        return _failed_report(
            run_id=clean_run_id,
            run_root=run_root,
            endpoint_kind=endpoint_kind,
            api_url=api_url,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            thresholds=threshold_model,
            errors=[
                DocumentParseAcceptanceError(
                    code="output_directory_not_empty",
                    message="run-specific acceptance output directory must be empty",
                    details={"output_dir": str(run_root)},
                )
            ],
        )
    run_root.mkdir(parents=True, exist_ok=True)
    source_pdf = run_root / "synthetic_source.pdf"
    warnings: list[str] = ["synthetic fixture comparison only; not production scientific extraction evidence"]
    errors: list[DocumentParseAcceptanceError] = []
    try:
        origin = _redacted_origin(api_url)
    except ValueError as exc:
        errors.append(DocumentParseAcceptanceError(code="invalid_api_url", message=str(exc), details={}))
        report = _failed_report(
            run_id=clean_run_id,
            run_root=run_root,
            endpoint_kind=endpoint_kind,
            api_url="",
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            thresholds=threshold_model,
            errors=errors,
            warnings=warnings,
        )
        _persist_report(report, run_root)
        return report

    try:
        _write_acceptance_pdf(source_pdf)
    except Exception as exc:
        errors.append(
            DocumentParseAcceptanceError(
                code="fixture_generation_failed",
                message=str(exc).strip() or exc.__class__.__name__,
                details={"exception_type": exc.__class__.__name__},
            )
        )
        report = _failed_report(
            run_id=clean_run_id,
            run_root=run_root,
            endpoint_kind=endpoint_kind,
            api_url=origin,
            backend=backend,
            effort=effort,
            parse_method=parse_method,
            thresholds=threshold_model,
            errors=errors,
            warnings=warnings,
        )
        _persist_report(report, run_root)
        return report

    source_hash = _sha256_file(source_pdf)
    service = _service(
        api_url=api_url,
        api_token=api_token,
        transport=transport,
        http_timeout_sec=http_timeout_sec,
        task_timeout_sec=task_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        max_poll_attempts=max_poll_attempts,
        monotonic=monotonic,
        sleep=sleep,
    )
    mineru_result, mineru_elapsed = _parse_with_service(
        service,
        DocumentParseRequest(
            run_id=f"{clean_run_id}-mineru",
            input_pdf=str(source_pdf),
            output_dir=str(run_root / "mineru"),
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
        source_pdf_sha256=source_hash,
    )
    if not mineru.ok:
        errors.append(
            DocumentParseAcceptanceError(
                code=str(mineru.error.code if mineru.error else "mineru_parse_failed"),
                message=str(mineru.error.message if mineru.error else "MinerU parsing failed"),
                details={},
            )
        )
    pdfplumber = None
    if compare_pdfplumber:
        pdf_result, pdf_elapsed = _parse_with_service(
            service,
            DocumentParseRequest(
                run_id=f"{clean_run_id}-pdfplumber",
                input_pdf=str(source_pdf),
                output_dir=str(run_root / "pdfplumber"),
                provider="pdfplumber",
            ),
        )
        pdfplumber = _provider_acceptance(
            result=pdf_result,
            elapsed_seconds=pdf_elapsed,
            root=run_root,
            source_pdf_sha256=source_hash,
        )
    errors.extend(_acceptance_findings(mineru=mineru, thresholds=threshold_model))
    comparison = _comparison(mineru, pdfplumber) if pdfplumber is not None else None
    decision = _decision(mineru=mineru, errors=errors)
    report = DocumentParseLiveAcceptanceReport(
        run_id=clean_run_id,
        generated_at=now_iso(),
        decision=decision,
        endpoint_kind=endpoint_kind,
        redacted_api_origin=origin,
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        source_pdf_sha256=source_hash,
        mineru=mineru,
        pdfplumber=pdfplumber,
        comparison=comparison,
        thresholds=threshold_model,
        warnings=warnings + list(mineru.warnings),
        errors=errors,
        outputs={"source_pdf": _rel(source_pdf, run_root)},
    )
    _persist_report(report, run_root)
    return report


def _service(
    *,
    api_url: str,
    api_token: str,
    transport: httpx.BaseTransport | None,
    http_timeout_sec: float,
    task_timeout_sec: float,
    poll_interval_sec: float,
    max_poll_attempts: int,
    monotonic: Any | None,
    sleep: Any | None,
) -> DocumentParseService:
    client = MinerUApiClient(
        base_url=api_url,
        api_token=api_token,
        transport=transport,
        http_timeout_sec=http_timeout_sec,
        task_timeout_sec=task_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        max_poll_attempts=max_poll_attempts,
        monotonic=monotonic,
        sleep=sleep,
    )
    return DocumentParseService(
        mineru_provider=MinerUApiDocumentParseProvider(client=client),
        pdfplumber_provider=PdfPlumberDocumentParseProvider(),
    )


def _parse_with_service(service: DocumentParseService, request: DocumentParseRequest) -> tuple[DocumentParseResult, float]:
    start = time.monotonic()
    result = service.parse(request)
    return result, max(0.0, time.monotonic() - start)


def _provider_acceptance(
    *,
    result: DocumentParseResult,
    elapsed_seconds: float,
    root: Path,
    source_pdf_sha256: str,
) -> DocumentParseProviderAcceptance:
    benchmark = (
        evaluate_document_parse_against_gold(
            parsed_document=result.parsed_document,
            gold=_GOLD,
            provider=result.provider,
        )
        if result.parsed_document is not None
        else None
    )
    error = (
        DocumentParseAcceptanceError(
            code=result.error.code,
            message=result.error.message,
            details=_redact_details(result.error.details),
        )
        if result.error is not None
        else None
    )
    outputs = result.outputs
    return DocumentParseProviderAcceptance(
        ok=bool(result.ok),
        provider=result.provider,
        parser_backend=result.parser_backend,
        source_pdf_sha256=source_pdf_sha256 or result.audit.source_pdf_sha256,
        elapsed_seconds=elapsed_seconds,
        remote_task_id=result.remote_task_id,
        task_status_history=list(result.audit.task_status_history),
        queued_ahead_history=list(result.audit.queued_ahead_history),
        mineru_version=result.audit.mineru_version,
        protocol_version=result.audit.protocol_version,
        output_bundle_refs={path: path for path in outputs.extracted_paths if path},
        markdown_path=_bundle_artifact_ref(outputs.extracted_paths, suffix=".md") or (
            _rel(Path(outputs.parsed_document_markdown), root) if outputs.parsed_document_markdown else ""
        ),
        content_list_path=_rel(Path(outputs.content_list_json), root) if outputs.content_list_json else "",
        content_list_v2_path=_rel(Path(outputs.content_list_v2_json), root) if outputs.content_list_v2_json else "",
        middle_json_path=_rel(Path(outputs.middle_json), root) if outputs.middle_json else "",
        parsed_document_path=_rel(Path(outputs.parsed_document_json), root) if outputs.parsed_document_json else "",
        parser_audit_path=_rel(Path(outputs.parser_audit_json), root) if outputs.parser_audit_json else "",
        benchmark_result=benchmark,
        warnings=list(result.warnings),
        error=error,
    )


def _acceptance_findings(
    *,
    mineru: DocumentParseProviderAcceptance,
    thresholds: DocumentParseAcceptanceThresholds,
) -> list[DocumentParseAcceptanceError]:
    findings: list[DocumentParseAcceptanceError] = []
    if not mineru.ok:
        return findings
    if not mineru.source_pdf_sha256:
        findings.append(DocumentParseAcceptanceError(code="missing_source_hash", message="source PDF hash is absent"))
    if not mineru.parser_audit_path:
        findings.append(DocumentParseAcceptanceError(code="missing_parser_audit", message="parser audit path is absent"))
    if not mineru.parsed_document_path:
        findings.append(DocumentParseAcceptanceError(code="missing_parsed_document", message="parsed document path is absent"))
    if not mineru.content_list_path and not mineru.content_list_v2_path:
        findings.append(DocumentParseAcceptanceError(code="structured_output_missing", message="structured MinerU content list is absent"))
    benchmark = mineru.benchmark_result
    if benchmark is None:
        findings.append(DocumentParseAcceptanceError(code="missing_benchmark", message="benchmark result is absent"))
        return findings
    if benchmark.observed_table_count <= 0:
        findings.append(DocumentParseAcceptanceError(code="expected_table_absent", message="expected table is absent"))
    if benchmark.normalized_text_token_recall < thresholds.normalized_text_token_recall:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="normalized text token recall below threshold"))
    if benchmark.header_match_rate < thresholds.header_match_rate:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="header match rate below threshold"))
    if not benchmark.row_count_match:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="row count does not match"))
    if benchmark.simple_cell_exact_match_rate < thresholds.simple_cell_exact_match_rate:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="cell exact-match rate below threshold"))
    if benchmark.provenance_completeness.get("page_present", 0.0) < thresholds.page_present:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="page provenance below threshold"))
    if benchmark.provenance_completeness.get("table_page_present", 0.0) < thresholds.table_page_present:
        findings.append(DocumentParseAcceptanceError(code="threshold_miss", message="table page provenance below threshold"))
    if mineru.warnings:
        findings.append(DocumentParseAcceptanceError(code="normalizer_warning", message="MinerU parser emitted warnings"))
    return findings


def _decision(*, mineru: DocumentParseProviderAcceptance, errors: list[DocumentParseAcceptanceError]) -> AcceptanceDecision:
    if not mineru.ok or any(error.code in {"missing_source_hash", "missing_parser_audit", "missing_parsed_document", "expected_table_absent"} for error in errors):
        return "failed"
    if errors:
        return "needs_review"
    return "passed"


def _comparison(
    mineru: DocumentParseProviderAcceptance,
    pdfplumber: DocumentParseProviderAcceptance | None,
) -> DocumentParseComparisonSummary | None:
    if pdfplumber is None:
        return None
    mineru_benchmark = mineru.benchmark_result
    pdf_benchmark = pdfplumber.benchmark_result
    metric_names = [
        "normalized_text_token_recall",
        "header_match_rate",
        "simple_cell_exact_match_rate",
    ]
    mineru_better: list[str] = []
    pdf_better: list[str] = []
    equal: list[str] = []
    for metric in metric_names:
        mineru_value = float(getattr(mineru_benchmark, metric, 0.0)) if mineru_benchmark is not None else 0.0
        pdf_value = float(getattr(pdf_benchmark, metric, 0.0)) if pdf_benchmark is not None else 0.0
        if mineru_value > pdf_value:
            mineru_better.append(metric)
        elif pdf_value > mineru_value:
            pdf_better.append(metric)
        else:
            equal.append(metric)
    return DocumentParseComparisonSummary(
        provider_success={"mineru_api": bool(mineru.ok), "pdfplumber": bool(pdfplumber.ok)},
        parser_backends={"mineru_api": mineru.parser_backend, "pdfplumber": pdfplumber.parser_backend},
        elapsed_seconds={"mineru_api": mineru.elapsed_seconds, "pdfplumber": pdfplumber.elapsed_seconds},
        page_counts={
            "mineru_api": mineru_benchmark.observed_page_count if mineru_benchmark else 0,
            "pdfplumber": pdf_benchmark.observed_page_count if pdf_benchmark else 0,
        },
        table_counts={
            "mineru_api": mineru_benchmark.observed_table_count if mineru_benchmark else 0,
            "pdfplumber": pdf_benchmark.observed_table_count if pdf_benchmark else 0,
        },
        normalized_text_token_recall={
            "mineru_api": mineru_benchmark.normalized_text_token_recall if mineru_benchmark else 0.0,
            "pdfplumber": pdf_benchmark.normalized_text_token_recall if pdf_benchmark else 0.0,
        },
        header_match_rate={
            "mineru_api": mineru_benchmark.header_match_rate if mineru_benchmark else 0.0,
            "pdfplumber": pdf_benchmark.header_match_rate if pdf_benchmark else 0.0,
        },
        row_count_match={
            "mineru_api": mineru_benchmark.row_count_match if mineru_benchmark else False,
            "pdfplumber": pdf_benchmark.row_count_match if pdf_benchmark else False,
        },
        simple_cell_exact_match_rate={
            "mineru_api": mineru_benchmark.simple_cell_exact_match_rate if mineru_benchmark else 0.0,
            "pdfplumber": pdf_benchmark.simple_cell_exact_match_rate if pdf_benchmark else 0.0,
        },
        provenance_completeness={
            "mineru_api": mineru_benchmark.provenance_completeness if mineru_benchmark else {},
            "pdfplumber": pdf_benchmark.provenance_completeness if pdf_benchmark else {},
        },
        warning_counts={"mineru_api": len(mineru.warnings), "pdfplumber": len(pdfplumber.warnings)},
        mineru_better_fields=mineru_better,
        pdfplumber_better_fields=pdf_better,
        equal_fields=equal,
    )


def _persist_report(report: DocumentParseLiveAcceptanceReport, run_root: Path) -> None:
    report_path = write_json(run_root / "acceptance_report.json", report.model_dump(mode="json"))
    summary_path = run_root / "acceptance_summary.md"
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    report.outputs["acceptance_report"] = _rel(report_path, run_root)
    report.outputs["acceptance_summary"] = _rel(summary_path, run_root)
    write_json(report_path, report.model_dump(mode="json"))


def _summary_markdown(report: DocumentParseLiveAcceptanceReport) -> str:
    lines = [
        f"# Document Parse Live Acceptance: {report.run_id}",
        "",
        f"- decision: {report.decision}",
        f"- endpoint_kind: {report.endpoint_kind}",
        f"- redacted_api_origin: {report.redacted_api_origin}",
        f"- MinerU provider: {report.mineru.ok}",
        f"- pdfplumber baseline: {report.pdfplumber.ok if report.pdfplumber else 'not run'}",
        "",
        "Synthetic fixture comparison only; not production scientific extraction evidence.",
    ]
    if report.errors:
        lines.append("")
        lines.append("## Errors")
        for error in report.errors:
            lines.append(f"- {error.code}: {error.message}")
    return "\n".join(lines) + "\n"


def _failed_report(
    *,
    run_id: str,
    run_root: Path,
    endpoint_kind: EndpointKind,
    api_url: str,
    backend: str,
    effort: str,
    parse_method: str,
    thresholds: DocumentParseAcceptanceThresholds,
    errors: list[DocumentParseAcceptanceError],
    warnings: list[str] | None = None,
) -> DocumentParseLiveAcceptanceReport:
    return DocumentParseLiveAcceptanceReport(
        run_id=run_id,
        generated_at=now_iso(),
        decision="failed",
        endpoint_kind=endpoint_kind,
        redacted_api_origin=_safe_origin(api_url),
        requested_backend=backend,
        requested_effort=effort,
        requested_parse_method=parse_method,
        mineru=DocumentParseProviderAcceptance(
            ok=False,
            provider="mineru_api",
            error=errors[0] if errors else DocumentParseAcceptanceError(code="failed", message="acceptance failed"),
        ),
        thresholds=thresholds,
        warnings=warnings or [],
        errors=errors,
        outputs={},
    )


def _write_acceptance_pdf(path: Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("reportlab is required to generate the live acceptance synthetic PDF") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Synthetic OLED Paper", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Synthetic-data notice: this document is generated for parser acceptance only.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("PLQY values are summarized for OLED emitters.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Table 1 OLED measurements", styles["BodyText"]),
    ]
    table = Table([["SMILES", "PLQY", "lambda_em"], ["CCO", "0.65", "520"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    content.extend([table, Spacer(1, 12), Paragraph("Synthetic fixture values.", styles["BodyText"])])
    doc.build(content)
    return path


def _redacted_origin(api_url: str) -> str:
    parsed = urlparse(str(api_url or "").strip())
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("api_url must not include userinfo, query, or fragment")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_url must include an http or https origin")
    return f"{parsed.scheme}://{parsed.netloc}"


def _safe_origin(api_url: str) -> str:
    try:
        return _redacted_origin(api_url)
    except ValueError:
        return ""


def _rel(path: Path, root: Path) -> str:
    if not str(path):
        return ""
    try:
        return str(path.expanduser().resolve().relative_to(root.expanduser().resolve()))
    except Exception:
        return str(path)


def _bundle_artifact_ref(paths: list[str], *, suffix: str) -> str:
    lowered_suffix = suffix.lower()
    for path in paths:
        clean = str(path or "").strip()
        if clean.lower().endswith(lowered_suffix):
            return clean
    return ""


def _redact_details(details: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in details.items():
        lowered = str(key).lower()
        if "token" in lowered or "authorization" in lowered or "secret" in lowered:
            redacted[str(key)] = "[redacted]"
        else:
            redacted[str(key)] = value
    return redacted


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
    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
    report = run_document_parse_live_acceptance(
        run_id=args.run_id,
        output_dir=args.output,
        api_url=args.api_url,
        endpoint_kind=str(args.endpoint_kind).replace("-", "_"),
        backend=args.backend,
        effort=args.effort,
        parse_method=args.parse_method,
        allow_remote_upload=bool(args.allow_remote_upload),
        compare_pdfplumber=bool(args.compare_pdfplumber),
        thresholds=DocumentParseAcceptanceThresholds(
            normalized_text_token_recall=float(args.min_text_recall),
            header_match_rate=float(args.min_header_match),
            simple_cell_exact_match_rate=float(args.min_cell_match),
        ),
        api_token=token,
        http_timeout_sec=float(args.http_timeout_sec),
        task_timeout_sec=float(args.task_timeout_sec),
        poll_interval_sec=float(args.poll_interval_sec),
        max_poll_attempts=int(args.max_poll_attempts),
        transport=transport,
    )
    output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    output.write("\n")
    if report.outputs.get("acceptance_report"):
        err.write(f"acceptance report: {Path(args.output).expanduser().resolve() / args.run_id / report.outputs['acceptance_report']}\n")
    if report.decision == "passed":
        return 0
    if report.decision == "needs_review":
        return 2
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.document_parse_live_acceptance",
        description="Manual live MinerU API acceptance runner with pdfplumber baseline comparison.",
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--endpoint-kind", default="mineru-api", choices=["mineru-api", "mineru-router", "compatible-endpoint"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", default="hybrid-engine")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--parse-method", default="auto")
    parser.add_argument("--allow-remote-upload", action="store_true")
    parser.add_argument("--compare-pdfplumber", action="store_true")
    parser.add_argument("--http-timeout-sec", type=float, default=30.0)
    parser.add_argument("--task-timeout-sec", type=float, default=300.0)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-poll-attempts", type=int, default=120)
    parser.add_argument("--min-text-recall", type=float, default=0.80)
    parser.add_argument("--min-header-match", type=float, default=1.0)
    parser.add_argument("--min-cell-match", type=float, default=0.90)
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
