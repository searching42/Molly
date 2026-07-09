from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.document_parse_cli import _service_from_args as _document_parse_service_from_args
from ai4s_agent.document_parse_provider import DocumentParseRequest, DocumentParseResult
from ai4s_agent.document_parse_service import DocumentParseService
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.workflows.corpus_to_phase1_workflow import (
    CorpusToPhase1WorkflowResult,
    run_corpus_to_phase1_workflow,
)

WorkflowRunner = Callable[..., CorpusToPhase1WorkflowResult]


@dataclass(frozen=True)
class PdfToDatasetResult:
    status: str
    output_dir: str
    input_pdf: str
    copied_pdf: str
    parsed_document_json: str
    workflow_report_json: str
    corpus_workflow_report_json: str
    extraction_manifest_json: str
    conflict_report_json: str
    conflict_summary_json: str
    candidate_dataset_csv: str
    training_dataset_csv: str
    dataset_manifest_json: str
    corpus_report_json: str
    corpus_report_md: str
    corpus_replay_manifest_json: str
    corpus_reproducibility_report_json: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_pdf_to_dataset(
    *,
    pdf: str | Path,
    output_dir: str | Path,
    run_id: str,
    provider: str = "auto",
    backend: str = "hybrid-engine",
    parse_method: str = "auto",
    effort: str = "medium",
    start_page: int | None = None,
    end_page: int | None = None,
    allow_remote_upload: bool = False,
    formula_enabled: bool = True,
    table_enabled: bool = True,
    image_analysis_enabled: bool = False,
    confirmed: bool = False,
    confirmed_by: str = "",
    confirmation_source: str = "run_pdf_to_dataset",
    confirmation_timestamp: str | None = None,
    property_ids: list[str] | None = None,
    n_bits: int = 256,
    topn: int = 10,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    parse_service: DocumentParseService | Any | None = None,
    workflow_runner: WorkflowRunner | None = None,
    generated_at: str | None = None,
) -> PdfToDatasetResult:
    """Run a single-paper PDF through parsing and the existing corpus workflow."""

    generated = generated_at or now_iso()
    source_pdf = _resolve_pdf(pdf)
    output_path = Path(output_dir).expanduser().resolve()
    input_dir = output_path / "input"
    parsed_dir = output_path / "parsed_documents"
    input_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    copied_pdf = input_dir / source_pdf.name
    if source_pdf != copied_pdf:
        shutil.copy2(source_pdf, copied_pdf)

    request = DocumentParseRequest(
        run_id=run_id,
        input_pdf=str(copied_pdf),
        output_dir=str(parsed_dir),
        provider=provider,
        parse_method=parse_method,
        backend=backend,
        effort=effort,
        formula_enabled=formula_enabled,
        table_enabled=table_enabled,
        image_analysis_enabled=image_analysis_enabled,
        start_page=start_page,
        end_page=end_page,
        allow_remote_upload=allow_remote_upload,
    )
    service = parse_service or DocumentParseService()
    parse_result = service.parse(request)
    if not parse_result.ok:
        _write_parse_failure_report(
            output_path=output_path,
            run_id=run_id,
            generated_at=generated,
            source_pdf=source_pdf,
            copied_pdf=copied_pdf,
            parse_result=parse_result,
        )
        message = parse_result.error.message if parse_result.error is not None else "document parse failed"
        raise RuntimeError(message)

    parsed_document_json = _materialize_parsed_document(
        parse_result=parse_result,
        parsed_dir=parsed_dir,
        run_id=run_id,
    )
    confirmation = DatasetConfirmation(
        confirmed=confirmed,
        confirmed_by=confirmed_by if confirmed else "",
        confirmation_source=confirmation_source,
        confirmation_timestamp=confirmation_timestamp or (generated if confirmed else None),
    )
    runner = workflow_runner or run_corpus_to_phase1_workflow
    workflow_result = runner(
        parsed_document_paths=[parsed_document_json],
        output_dir=output_path,
        run_id=run_id,
        confirmation=confirmation,
        generated_at=generated,
        property_ids=property_ids,
        n_bits=n_bits,
        topn=topn,
        min_numeric_ratio=min_numeric_ratio,
        min_nonempty=min_nonempty,
    )
    extraction_manifest_json = _alias_artifact(
        source=workflow_result.corpus_extraction_manifest_json,
        target=output_path / "extraction" / "extraction_manifest.json",
    )
    conflict_report_json = _alias_artifact(
        source=workflow_result.corpus_conflict_report_json,
        target=output_path / "conflicts" / "conflict_report.json",
    )
    conflict_summary_json = str(output_path / "conflicts" / "conflict_summary.json")
    workflow_report_json = _write_workflow_report(
        output_path=output_path,
        run_id=run_id,
        generated_at=generated,
        status=workflow_result.status,
        source_pdf=source_pdf,
        copied_pdf=copied_pdf,
        parsed_document_json=parsed_document_json,
        parse_request=request,
        parse_result=parse_result,
        workflow_result=workflow_result,
        extraction_manifest_json=extraction_manifest_json,
        conflict_report_json=conflict_report_json,
        conflict_summary_json=conflict_summary_json,
        confirmation=confirmation,
    )
    return PdfToDatasetResult(
        status=workflow_result.status,
        output_dir=str(output_path),
        input_pdf=str(source_pdf),
        copied_pdf=str(copied_pdf),
        parsed_document_json=str(parsed_document_json),
        workflow_report_json=str(workflow_report_json),
        corpus_workflow_report_json=workflow_result.corpus_workflow_report_json,
        extraction_manifest_json=extraction_manifest_json,
        conflict_report_json=conflict_report_json,
        conflict_summary_json=conflict_summary_json,
        candidate_dataset_csv=workflow_result.candidate_dataset_csv,
        training_dataset_csv=workflow_result.training_dataset_csv,
        dataset_manifest_json=workflow_result.dataset_manifest_json,
        corpus_report_json=workflow_result.corpus_report_json,
        corpus_report_md=workflow_result.corpus_report_md,
        corpus_replay_manifest_json=workflow_result.corpus_replay_manifest_json,
        corpus_reproducibility_report_json=workflow_result.corpus_reproducibility_report_json,
    )


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    parse_service: DocumentParseService | Any | None = None,
    workflow_runner: WorkflowRunner | None = None,
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
        service = parse_service or _document_parse_service_from_args(args, transport=None)
        result = run_pdf_to_dataset(
            pdf=args.pdf,
            output_dir=args.output_dir,
            run_id=args.run_id,
            provider=args.provider,
            backend=args.backend,
            parse_method=args.parse_method,
            effort=args.effort,
            start_page=args.start_page,
            end_page=args.end_page,
            allow_remote_upload=bool(args.allow_remote_upload),
            formula_enabled=bool(args.formula_enabled),
            table_enabled=bool(args.table_enabled),
            image_analysis_enabled=bool(args.image_analysis_enabled),
            confirmed=bool(args.confirm_dataset),
            confirmed_by=str(args.confirmed_by or ""),
            confirmation_source=str(args.confirmation_source or "run_pdf_to_dataset"),
            confirmation_timestamp=args.confirmation_timestamp,
            property_ids=list(args.property_ids or []),
            n_bits=int(args.n_bits),
            topn=int(args.topn),
            min_numeric_ratio=float(args.min_numeric_ratio),
            min_nonempty=int(args.min_nonempty),
            parse_service=service,
            workflow_runner=workflow_runner,
        )
        output.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 0
    except Exception as exc:
        err.write(f"pdf ingestion failed: {str(exc).strip() or exc.__class__.__name__}\n")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.run_pdf_to_dataset",
        description="Parse one scientific PDF and run the existing corpus-to-dataset workflow.",
    )
    parser.add_argument("--pdf", required=True, help="Input scientific PDF path.")
    parser.add_argument("--output-dir", required=True, help="Run-scoped output directory.")
    parser.add_argument("--run-id", required=True, help="Stable run id for all artifacts.")
    parser.add_argument("--provider", default="auto", choices=["auto", "mineru-api", "pdfplumber"])
    parser.add_argument("--api-url", dest="api_url", help="MinerU API-compatible base URL.")
    parser.add_argument("--mineru-api-url", dest="api_url", help="MinerU API-compatible base URL.")
    parser.add_argument("--backend", default="hybrid-engine")
    parser.add_argument("--parse-method", default="auto")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--allow-remote-upload", action="store_true")
    parser.add_argument("--disable-formula", action="store_false", dest="formula_enabled")
    parser.add_argument("--disable-table", action="store_false", dest="table_enabled")
    parser.add_argument("--enable-image-analysis", action="store_true", dest="image_analysis_enabled")
    parser.add_argument("--confirm-dataset", action="store_true")
    parser.add_argument("--confirmed-by", default="")
    parser.add_argument("--confirmation-source", default="run_pdf_to_dataset")
    parser.add_argument("--confirmation-timestamp")
    parser.add_argument("--property-id", action="append", dest="property_ids", default=[])
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min-numeric-ratio", type=float, default=0.6)
    parser.add_argument("--min-nonempty", type=int, default=30)
    parser.set_defaults(formula_enabled=True, table_enabled=True, image_analysis_enabled=False)
    return parser


def _resolve_pdf(pdf: str | Path) -> Path:
    path = Path(pdf).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"pdf not found: {path}")
    if not path.is_file():
        raise ValueError(f"pdf must be a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"pdf must be a PDF file: {path}")
    return path


def _materialize_parsed_document(
    *,
    parse_result: DocumentParseResult,
    parsed_dir: Path,
    run_id: str,
) -> Path:
    target = parsed_dir / f"{run_id}_parsed_document.json"
    if parse_result.parsed_document is not None:
        write_json(target, parse_result.parsed_document.model_dump(mode="json"))
        return target
    source = Path(parse_result.outputs.parsed_document_json).expanduser().resolve()
    if not source.exists():
        raise RuntimeError("document parse did not produce parsed_document.json")
    if source != target:
        shutil.copy2(source, target)
    return target


def _alias_artifact(*, source: str | Path, target: Path) -> str:
    source_path = Path(source).expanduser().resolve()
    target_path = target.expanduser().resolve()
    if not source_path.exists():
        raise RuntimeError(f"workflow artifact not found: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path != target_path:
        shutil.copy2(source_path, target_path)
    return str(target_path)


def _write_workflow_report(
    *,
    output_path: Path,
    run_id: str,
    generated_at: str,
    status: str,
    source_pdf: Path,
    copied_pdf: Path,
    parsed_document_json: Path,
    parse_request: DocumentParseRequest,
    parse_result: DocumentParseResult,
    workflow_result: CorpusToPhase1WorkflowResult,
    extraction_manifest_json: str,
    conflict_report_json: str,
    conflict_summary_json: str,
    confirmation: DatasetConfirmation,
) -> Path:
    report_path = output_path / "workflow_report.json"
    payload = {
        "schema_version": "real_pdf_to_dataset_workflow.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "status": status,
        "input": {
            "source_pdf": str(source_pdf),
            "copied_pdf": str(copied_pdf),
            "source_pdf_sha256": _sha256_file(source_pdf),
            "run_scoped_input": True,
        },
        "parse": {
            "request": parse_request.model_dump(mode="json"),
            "status": parse_result.status,
            "provider": parse_result.provider,
            "selected_provider": parse_result.audit.selected_provider,
            "selection_reason": parse_result.audit.selection_reason,
            "parser_backend": parse_result.parser_backend,
            "parsed_document_json": str(parsed_document_json),
            "parser_audit_json": parse_result.outputs.parser_audit_json,
            "warnings": parse_result.warnings,
            "mineru_version": parse_result.audit.mineru_version,
            "protocol_version": parse_result.audit.protocol_version,
        },
        "workflow": {
            "corpus_workflow_report_json": workflow_result.corpus_workflow_report_json,
            "corpus_records_json": str(output_path / "extraction" / "corpus_records.json"),
            "corpus_extraction_manifest_json": workflow_result.corpus_extraction_manifest_json,
            "extraction_manifest_json": extraction_manifest_json,
            "corpus_conflict_report_json": workflow_result.corpus_conflict_report_json,
            "conflict_report_json": conflict_report_json,
            "conflict_summary_json": conflict_summary_json,
            "candidate_dataset_csv": workflow_result.candidate_dataset_csv,
            "training_dataset_csv": workflow_result.training_dataset_csv,
            "rejected_records_json": workflow_result.rejected_records_json,
            "dataset_manifest_json": workflow_result.dataset_manifest_json,
            "corpus_report_json": workflow_result.corpus_report_json,
            "corpus_report_md": workflow_result.corpus_report_md,
            "corpus_replay_manifest_json": workflow_result.corpus_replay_manifest_json,
            "corpus_lineage_manifest_json": workflow_result.corpus_lineage_manifest_json,
            "corpus_reproducibility_report_json": workflow_result.corpus_reproducibility_report_json,
        },
        "governance": {
            "confirmation": confirmation.to_dict(),
            "confirmation_gates_preserved": True,
            "no_silent_materialization": not confirmation.confirmed,
            "notes": [
                "Single-PDF ingestion reuses existing deterministic corpus extraction and validation writers.",
                "Training rows are materialized only when explicit DatasetConfirmation is supplied.",
                "Scientific accuracy still requires human review and real corpus acceptance.",
            ],
        },
    }
    return write_json(report_path, payload)


def _write_parse_failure_report(
    *,
    output_path: Path,
    run_id: str,
    generated_at: str,
    source_pdf: Path,
    copied_pdf: Path,
    parse_result: DocumentParseResult,
) -> None:
    write_json(
        output_path / "workflow_report.json",
        {
            "schema_version": "real_pdf_to_dataset_workflow.v1",
            "run_id": run_id,
            "generated_at": generated_at,
            "status": "failed",
            "failed_stage": "parse_document",
            "input": {
                "source_pdf": str(source_pdf),
                "copied_pdf": str(copied_pdf),
                "source_pdf_sha256": _sha256_file(source_pdf),
            },
            "parse": parse_result.model_dump(mode="json"),
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
