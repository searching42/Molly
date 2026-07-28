from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidateSummary,
    extract_oled_mineru_candidates_from_document,
    summarize_oled_mineru_candidates,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import map_oled_mineru_candidates_to_schema_candidates
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    compile_oled_schema_candidates_to_layered_records,
)


class OledMineruParsedBundle(BaseModel):
    paper_id: str
    content_list_path: str | None = None
    content_list_v2_path: str | None = None
    md_path: str | None = None
    source_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("paper_id is required")
        return clean


class OledMineruAcceptanceManifest(BaseModel):
    manifest_id: str
    bundles: list[OledMineruParsedBundle]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("manifest_id")
    @classmethod
    def validate_manifest_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("manifest_id is required")
        return clean


class OledMineruAcceptancePaperResult(BaseModel):
    paper_id: str
    status: Literal["completed", "skipped", "failed"]

    source_format_counts: dict[str, int] = Field(default_factory=dict)

    mineru_candidate_count: int = 0
    semantic_candidate_count: int = 0
    compiled_record_count: int = 0

    compiled_status_counts: dict[str, int] = Field(default_factory=dict)

    mineru_summary: OledMineruCandidateSummary | None = None
    semantic_error_codes: list[str] = Field(default_factory=list)
    semantic_warning_codes: list[str] = Field(default_factory=list)
    compilation_error_codes: list[str] = Field(default_factory=list)
    compilation_warning_codes: list[str] = Field(default_factory=list)

    representative_evidence_anchors: list[str] = Field(default_factory=list)
    output_record_ids: list[str] = Field(default_factory=list)

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledMineruAcceptanceReport(BaseModel):
    manifest_id: str
    status: Literal["completed", "completed_with_warnings", "failed"]

    paper_count: int
    completed_paper_count: int
    skipped_paper_count: int
    failed_paper_count: int

    total_mineru_candidate_count: int
    total_semantic_candidate_count: int
    total_compiled_record_count: int

    compiled_status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    paper_results: list[OledMineruAcceptancePaperResult] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status in {"completed", "completed_with_warnings"} and self.failed_paper_count == 0


def load_oled_mineru_acceptance_manifest(path: str | Path) -> OledMineruAcceptanceManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"missing_manifest_file:{redact_oled_mineru_acceptance_path(manifest_path)}")
    _reject_forbidden_input(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    manifest = OledMineruAcceptanceManifest.model_validate(payload)
    base_dir = manifest_path.resolve().parent
    resolved_bundles = [_resolve_bundle_paths(bundle, base_dir) for bundle in manifest.bundles]
    return manifest.model_copy(update={"bundles": resolved_bundles})


def run_oled_mineru_acceptance_harness(
    manifest: OledMineruAcceptanceManifest,
    *,
    confirm_read_only_parsed_outputs: bool = False,
    include_irrelevant_candidates: bool = False,
) -> OledMineruAcceptanceReport:
    if not confirm_read_only_parsed_outputs:
        raise ValueError("confirmation_required:read_only_parsed_outputs")

    paper_results = [
        _run_bundle(bundle, include_irrelevant_candidates=include_irrelevant_candidates)
        for bundle in manifest.bundles
    ]
    completed = sum(1 for result in paper_results if result.status == "completed")
    skipped = sum(1 for result in paper_results if result.status == "skipped")
    failed = sum(1 for result in paper_results if result.status == "failed")
    compiled_status_counts: Counter[str] = Counter()
    finding_code_counts: Counter[str] = Counter()
    for result in paper_results:
        compiled_status_counts.update(result.compiled_status_counts)
        finding_code_counts.update(result.semantic_error_codes)
        finding_code_counts.update(result.semantic_warning_codes)
        finding_code_counts.update(result.compilation_error_codes)
        finding_code_counts.update(result.compilation_warning_codes)
        finding_code_counts.update(result.reason_codes)
    status: Literal["completed", "completed_with_warnings", "failed"]
    if failed:
        status = "failed"
    elif finding_code_counts:
        status = "completed_with_warnings"
    else:
        status = "completed"
    return OledMineruAcceptanceReport(
        manifest_id=manifest.manifest_id,
        status=status,
        paper_count=len(paper_results),
        completed_paper_count=completed,
        skipped_paper_count=skipped,
        failed_paper_count=failed,
        total_mineru_candidate_count=sum(result.mineru_candidate_count for result in paper_results),
        total_semantic_candidate_count=sum(result.semantic_candidate_count for result in paper_results),
        total_compiled_record_count=sum(result.compiled_record_count for result in paper_results),
        compiled_status_counts=dict(sorted(compiled_status_counts.items())),
        finding_code_counts=dict(sorted(finding_code_counts.items())),
        paper_results=paper_results,
        metadata={
            "runner": "oled_mineru_acceptance_harness",
            "read_only_parsed_outputs_confirmed": True,
            "metadata_key_counts": _metadata_key_counts(paper_results),
            "pdfs_read": False,
            "images_read": False,
            "llm_called": False,
            "mineru_called": False,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "model_backends_run": False,
        },
    )


def write_oled_mineru_acceptance_report_json(
    report: OledMineruAcceptanceReport,
    path: str | Path,
) -> None:
    payload = report.model_dump(mode="json", exclude_none=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def redact_oled_mineru_acceptance_path(path: str | Path) -> str:
    raw_path = Path(path)
    return raw_path.name or str(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OLED MinerU parsed-output acceptance harness.")
    parser.add_argument("--manifest", required=True, help="Path to acceptance manifest JSON.")
    parser.add_argument("--output-report", help="Optional path to write redacted acceptance report JSON.")
    parser.add_argument(
        "--confirm-read-only-parsed-outputs",
        action="store_true",
        help="Confirm that only local parsed-output JSON/MD sidecars will be read.",
    )
    parser.add_argument(
        "--include-irrelevant-candidates",
        action="store_true",
        help="Include parsed candidates without OLED relevance signals.",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_oled_mineru_acceptance_manifest(args.manifest)
        report = run_oled_mineru_acceptance_harness(
            manifest,
            confirm_read_only_parsed_outputs=args.confirm_read_only_parsed_outputs,
            include_irrelevant_candidates=args.include_irrelevant_candidates,
        )
        if args.output_report:
            write_oled_mineru_acceptance_report_json(report, args.output_report)
        else:
            summary = {
                "manifest_id": report.manifest_id,
                "status": report.status,
                "paper_count": report.paper_count,
                "completed_paper_count": report.completed_paper_count,
                "failed_paper_count": report.failed_paper_count,
                "total_mineru_candidate_count": report.total_mineru_candidate_count,
                "total_semantic_candidate_count": report.total_semantic_candidate_count,
                "total_compiled_record_count": report.total_compiled_record_count,
                "finding_code_counts": report.finding_code_counts,
            }
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.status in {"completed", "completed_with_warnings"} else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _run_bundle(
    bundle: OledMineruParsedBundle,
    *,
    include_irrelevant_candidates: bool,
) -> OledMineruAcceptancePaperResult:
    try:
        parsed_documents = _load_bundle_documents(bundle)
        if not parsed_documents:
            return OledMineruAcceptancePaperResult(
                paper_id=bundle.paper_id,
                status="skipped",
                reason_codes=["no_parsed_json_inputs"],
                metadata=_bundle_metadata(bundle),
            )
        md_text = _load_md_text(bundle)
        mineru_candidates = []
        for document, source_path in parsed_documents:
            mineru_candidates.extend(
                extract_oled_mineru_candidates_from_document(
                    document,
                    paper_id=bundle.paper_id,
                    source_path=source_path,
                    md_text=md_text,
                    include_irrelevant=include_irrelevant_candidates,
                )
            )
        mineru_summary = summarize_oled_mineru_candidates(mineru_candidates)
        semantic_report = map_oled_mineru_candidates_to_schema_candidates(mineru_candidates)
        compilation_report = compile_oled_schema_candidates_to_layered_records(semantic_report.schema_candidates)
        compiled_status_counts = Counter(record.status.value for record in compilation_report.compiled_records)
        reason_codes = []
        if not mineru_candidates:
            reason_codes.append("no_mineru_candidates")
        if not semantic_report.schema_candidates:
            reason_codes.append("no_semantic_candidates")
        if not compilation_report.compiled_records:
            reason_codes.append("no_compiled_records")
        return OledMineruAcceptancePaperResult(
            paper_id=bundle.paper_id,
            status="completed",
            source_format_counts=_source_format_counts(mineru_candidates),
            mineru_candidate_count=len(mineru_candidates),
            semantic_candidate_count=len(semantic_report.schema_candidates),
            compiled_record_count=len(compilation_report.compiled_records),
            compiled_status_counts=dict(sorted(compiled_status_counts.items())),
            mineru_summary=mineru_summary,
            semantic_error_codes=semantic_report.error_codes,
            semantic_warning_codes=semantic_report.warning_codes,
            compilation_error_codes=compilation_report.error_codes,
            compilation_warning_codes=compilation_report.warning_codes,
            representative_evidence_anchors=sorted({candidate.evidence_anchor for candidate in mineru_candidates})[:10],
            output_record_ids=sorted(record.record_id for record in compilation_report.compiled_records)[:25],
            reason_codes=reason_codes,
            metadata={
                **_bundle_metadata(bundle),
                "parsed_document_count": len(parsed_documents),
                "md_sidecar_used": md_text is not None,
                "report_payload_policy": "counts_statuses_anchors_only",
            },
        )
    except json.JSONDecodeError as exc:
        return _failed_bundle_result(bundle, "parsed_json_load_failed", str(exc))
    except OSError as exc:
        return _failed_bundle_result(bundle, "parsed_file_read_failed", str(exc))
    except ValueError as exc:
        return _failed_bundle_result(bundle, _stable_reason_code(str(exc)), str(exc))


def _resolve_bundle_paths(bundle: OledMineruParsedBundle, base_dir: Path) -> OledMineruParsedBundle:
    updates: dict[str, str | None] = {}
    for field_name in ("content_list_path", "content_list_v2_path", "md_path"):
        raw_value = getattr(bundle, field_name)
        if raw_value is None:
            updates[field_name] = None
            continue
        resolved_path = _resolve_input_path(raw_value, base_dir)
        updates[field_name] = str(resolved_path)
    return bundle.model_copy(update=updates)


def _resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    candidate_path = Path(raw_path)
    resolved_path = candidate_path if candidate_path.is_absolute() else base_dir / candidate_path
    _reject_forbidden_input(resolved_path)
    if not resolved_path.exists():
        raise ValueError(f"missing_bundle_file:{redact_oled_mineru_acceptance_path(resolved_path)}")
    return resolved_path.resolve()


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _load_bundle_documents(bundle: OledMineruParsedBundle) -> list[tuple[Any, str]]:
    documents: list[tuple[Any, str]] = []
    for path in [bundle.content_list_path, bundle.content_list_v2_path]:
        if not path:
            continue
        _reject_forbidden_input(path)
        with Path(path).open("r", encoding="utf-8") as handle:
            documents.append((json.load(handle), path))
    return documents


def _load_md_text(bundle: OledMineruParsedBundle) -> str | None:
    if not bundle.md_path:
        return None
    _reject_forbidden_input(bundle.md_path)
    return Path(bundle.md_path).read_text(encoding="utf-8")


def _source_format_counts(candidates: list[Any]) -> dict[str, int]:
    counter = Counter(candidate.source_format.value for candidate in candidates)
    return dict(sorted(counter.items()))


def _metadata_key_counts(paper_results: list[OledMineruAcceptancePaperResult]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in paper_results:
        for key in result.metadata:
            counter[f"paper_result:{key}"] += 1
    return dict(sorted(counter.items()))


def _bundle_metadata(bundle: OledMineruParsedBundle) -> dict[str, Any]:
    paths = {}
    if bundle.content_list_path:
        paths["content_list_path"] = redact_oled_mineru_acceptance_path(bundle.content_list_path)
    if bundle.content_list_v2_path:
        paths["content_list_v2_path"] = redact_oled_mineru_acceptance_path(bundle.content_list_v2_path)
    if bundle.md_path:
        paths["md_path"] = redact_oled_mineru_acceptance_path(bundle.md_path)
    metadata = {
        "source_label": bundle.source_label,
        "input_paths": paths,
    }
    if bundle.metadata:
        metadata["bundle_metadata"] = bundle.metadata
    return {key: value for key, value in metadata.items() if value not in (None, {}, [])}


def _failed_bundle_result(bundle: OledMineruParsedBundle, reason_code: str, message: str) -> OledMineruAcceptancePaperResult:
    return OledMineruAcceptancePaperResult(
        paper_id=bundle.paper_id,
        status="failed",
        reason_codes=[reason_code],
        metadata={
            **_bundle_metadata(bundle),
            "error_message": message.splitlines()[0][:200],
        },
    )


def _stable_reason_code(message: str) -> str:
    prefix = message.split(":", 1)[0].strip()
    return prefix or "acceptance_bundle_failed"


_FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


__all__ = [
    "OledMineruParsedBundle",
    "OledMineruAcceptanceManifest",
    "OledMineruAcceptancePaperResult",
    "OledMineruAcceptanceReport",
    "load_oled_mineru_acceptance_manifest",
    "run_oled_mineru_acceptance_harness",
    "write_oled_mineru_acceptance_report_json",
    "redact_oled_mineru_acceptance_path",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
