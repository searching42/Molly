from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import (
    AdmissionRequest,
    admission_validation_summary,
    load_admission_request,
)
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport
from ai4s_agent.custom_corpus_manifest import CustomCorpusManifest, load_custom_corpus_manifest, sha256_file
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest
from ai4s_agent.mineru_preflight_binding import contains_credential_marker


PackageValidationStatus = Literal["passed", "failed"]
PackageAdmissionDecision = Literal["eligible", "needs_review", "ineligible"]

_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_FORBIDDEN_MARKERS = ("/Users/", "/home/", "C:\\")


class CustomCorpusAdmissionPackageError(ValueError):
    pass


class PackageValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _SCHEMA_VERSION
    validation_status: PackageValidationStatus
    admission_decision: PackageAdmissionDecision = "ineligible"
    manifest_path: str = ""
    dry_run_report_path: str = ""
    review_manifest_path: str = ""
    admission_request_path: str = ""
    manifest_sha256: str = ""
    dry_run_report_sha256: str = ""
    review_manifest_sha256: str = ""
    admission_request_sha256: str = ""
    corpus_id: str = ""
    dry_run_id: str = ""
    review_manifest_id: str = ""
    admission_request_id: str = ""
    corpus_class: str = ""
    document_count: int = 0
    dry_run_decision: str = ""
    dry_run_phase1_status: str = ""
    dry_run_dataset_confirmation_confirmed: bool = False
    dry_run_training_dataset_admitted: bool = False
    review_record_count: int = 0
    admission_record_count: int = 0
    admit_count: int = 0
    exclude_count: int = 0
    needs_review_count: int = 0
    matched_review_record_count: int = 0
    missing_review_record_count: int = 0
    binding_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_admission_package(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
) -> dict[str, Any]:
    paths = _PackagePaths(
        manifest=Path(manifest_path).expanduser(),
        dry_run_report=Path(dry_run_report_path).expanduser(),
        review_manifest=Path(review_manifest_path).expanduser(),
        admission_request=Path(admission_request_path).expanduser(),
    )
    hashes = _hashes(paths)
    binding_errors: list[str] = []
    warnings: list[str] = []

    manifest = _load_manifest(paths.manifest, binding_errors)
    dry_run = _load_dry_run_report(paths.dry_run_report, binding_errors)
    review = _load_review_manifest(paths.review_manifest, binding_errors)
    admission, raw_admission = _load_admission_request(paths.admission_request, binding_errors)

    admission_summary = _admission_summary(admission, raw_admission)
    if admission is None and raw_admission:
        binding_errors.extend(_raw_admission_safety_errors(raw_admission))

    if manifest and dry_run and review and admission:
        binding_errors.extend(_hash_binding_errors(admission, hashes))
        binding_errors.extend(_id_binding_errors(manifest, dry_run, review, admission))
        binding_errors.extend(_dry_run_boundary_errors(dry_run))
        record_counts = _review_admission_binding_errors(
            review=review,
            admission=admission,
            review_manifest_sha256=hashes["review_manifest_sha256"],
        )
        binding_errors.extend(record_counts["errors"])
        matched_review_record_count = int(record_counts["matched"])
        missing_review_record_count = int(record_counts["missing"])
    else:
        matched_review_record_count = 0
        missing_review_record_count = 0

    binding_errors = _stable_unique(binding_errors)
    summary = PackageValidationSummary(
        validation_status="failed" if binding_errors else "passed",
        admission_decision=admission_summary["decision"],
        manifest_path=paths.manifest.name,
        dry_run_report_path=paths.dry_run_report.name,
        review_manifest_path=paths.review_manifest.name,
        admission_request_path=paths.admission_request.name,
        manifest_sha256=hashes["manifest_sha256"],
        dry_run_report_sha256=hashes["dry_run_report_sha256"],
        review_manifest_sha256=hashes["review_manifest_sha256"],
        admission_request_sha256=hashes["admission_request_sha256"],
        corpus_id=_first_present(
            getattr(manifest, "corpus_id", ""),
            getattr(dry_run, "corpus_id", ""),
            getattr(review, "corpus_id", ""),
            getattr(admission, "corpus_id", ""),
        ),
        dry_run_id=_first_present(
            getattr(dry_run, "run_id", ""),
            getattr(review, "dry_run_id", ""),
            getattr(admission, "dry_run_id", ""),
        ),
        review_manifest_id=_first_present(getattr(review, "review_manifest_id", ""), getattr(admission, "review_manifest_id", "")),
        admission_request_id=getattr(admission, "admission_request_id", "") or str(raw_admission.get("admission_request_id") or ""),
        corpus_class=getattr(manifest, "corpus_class", "") or getattr(dry_run, "corpus_class", ""),
        document_count=len(manifest.documents) if manifest else 0,
        dry_run_decision=getattr(dry_run, "decision", ""),
        dry_run_phase1_status=getattr(getattr(dry_run, "confirmation_boundary", None), "phase1_status", ""),
        dry_run_dataset_confirmation_confirmed=bool(
            getattr(getattr(dry_run, "confirmation_boundary", None), "dataset_confirmation_confirmed", False)
        ),
        dry_run_training_dataset_admitted=bool(
            getattr(getattr(dry_run, "confirmation_boundary", None), "training_dataset_admitted", False)
        ),
        review_record_count=len(review.review_records) if review else 0,
        admission_record_count=admission_summary["admission_record_count"],
        admit_count=admission_summary["admit_count"],
        exclude_count=admission_summary["exclude_count"],
        needs_review_count=admission_summary["needs_review_count"],
        matched_review_record_count=matched_review_record_count,
        missing_review_record_count=missing_review_record_count,
        binding_errors=binding_errors,
        warnings=warnings,
    )
    return _safe_summary(summary).model_dump(mode="json")


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    summary = validate_admission_package(
        manifest_path=args.manifest,
        dry_run_report_path=args.dry_run_report,
        review_manifest_path=args.review_manifest,
        admission_request_path=args.admission_request,
    )
    if args.output_summary:
        write_json(Path(args.output_summary).expanduser(), summary)
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    if summary["validation_status"] == "passed":
        return 0
    err.write("admission package validation failed\n")
    return 1


class _PackagePaths(BaseModel):
    manifest: Path
    dry_run_report: Path
    review_manifest: Path
    admission_request: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_admission_package",
        description="Validate a custom corpus admission package offline.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--output-summary", default="")
    return parser


def _hashes(paths: _PackagePaths) -> dict[str, str]:
    return {
        "manifest_sha256": _safe_sha256(paths.manifest),
        "dry_run_report_sha256": _safe_sha256(paths.dry_run_report),
        "review_manifest_sha256": _safe_sha256(paths.review_manifest),
        "admission_request_sha256": _safe_sha256(paths.admission_request),
    }


def _safe_sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _load_manifest(path: Path, binding_errors: list[str]) -> CustomCorpusManifest | None:
    try:
        return load_custom_corpus_manifest(path)
    except Exception:
        binding_errors.append("invalid_manifest")
        return None


def _load_dry_run_report(path: Path, binding_errors: list[str]) -> CustomCorpusDryRunReport | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = CustomCorpusDryRunReport.model_validate(payload)
    except Exception:
        binding_errors.append("invalid_dry_run_report")
        return None
    if report.schema_version != "custom_corpus_dry_run.v1":
        binding_errors.append("invalid_dry_run_report")
        return None
    return report


def _load_review_manifest(path: Path, binding_errors: list[str]) -> ReviewManifest | None:
    try:
        return load_review_manifest(path)
    except Exception:
        binding_errors.append("invalid_review_manifest")
        return None


def _load_admission_request(path: Path, binding_errors: list[str]) -> tuple[AdmissionRequest | None, dict[str, Any]]:
    raw: dict[str, Any] = {}
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw_payload, dict):
            raw = raw_payload
    except Exception:
        binding_errors.append("invalid_admission_request")
        return None, raw
    try:
        return load_admission_request(path), raw
    except Exception:
        binding_errors.append("invalid_admission_request")
        return None, raw


def _admission_summary(admission: AdmissionRequest | None, raw_admission: dict[str, Any]) -> dict[str, Any]:
    if admission is not None:
        summary = admission_validation_summary(admission)
        return {
            "decision": summary["decision"],
            "admission_record_count": summary["admission_record_count"],
            "admit_count": summary["admit_count"],
            "exclude_count": summary["exclude_count"],
            "needs_review_count": summary["needs_review_count"],
        }
    records = raw_admission.get("admission_records")
    if not isinstance(records, list):
        records = []
    actions = [str(record.get("action", "")) for record in records if isinstance(record, dict)]
    needs_review_count = actions.count("needs_review")
    admit_count = actions.count("admit")
    if needs_review_count:
        decision: PackageAdmissionDecision = "needs_review"
    elif admit_count:
        decision = "eligible"
    else:
        decision = "ineligible"
    return {
        "decision": decision,
        "admission_record_count": len(records),
        "admit_count": admit_count,
        "exclude_count": actions.count("exclude"),
        "needs_review_count": needs_review_count,
    }


def _raw_admission_safety_errors(raw_admission: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    records = raw_admission.get("admission_records")
    if not isinstance(records, list):
        return errors
    for record in records:
        if not isinstance(record, dict):
            continue
        action = str(record.get("action", ""))
        review_decision = str(record.get("review_decision", ""))
        if action == "admit" and review_decision == "reject":
            errors.append("rejected_record_admitted")
        if action == "admit" and review_decision == "needs_review":
            errors.append("needs_review_record_admitted")
        if action == "admit" and not str(record.get("admission_reason", "")).strip():
            errors.append("admitted_record_missing_admission_reason")
    return errors


def _hash_binding_errors(admission: AdmissionRequest, hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    if admission.source_manifest_sha256 != hashes["manifest_sha256"]:
        errors.append("manifest_hash_mismatch")
    if admission.source_dry_run_report_sha256 != hashes["dry_run_report_sha256"]:
        errors.append("dry_run_report_hash_mismatch")
    if admission.source_review_manifest_sha256 != hashes["review_manifest_sha256"]:
        errors.append("review_manifest_hash_mismatch")
    return errors


def _id_binding_errors(
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
) -> list[str]:
    errors: list[str] = []
    corpus_ids = {manifest.corpus_id, dry_run.corpus_id, review.corpus_id, admission.corpus_id}
    if len(corpus_ids) != 1:
        errors.append("corpus_id_mismatch")
    dry_run_id = dry_run.run_id
    if review.dry_run_id != dry_run_id or admission.dry_run_id != dry_run_id:
        errors.append("dry_run_id_mismatch")
    if review.review_manifest_id != admission.review_manifest_id:
        errors.append("review_manifest_id_mismatch")
    return errors


def _dry_run_boundary_errors(dry_run: CustomCorpusDryRunReport) -> list[str]:
    errors: list[str] = []
    if dry_run.decision != "passed":
        errors.append("dry_run_not_passed")
    boundary = dry_run.confirmation_boundary
    if boundary.dataset_confirmation_confirmed is not False:
        errors.append("dry_run_dataset_confirmed")
    if boundary.phase1_status != "not_run":
        errors.append("dry_run_phase1_ran")
    if boundary.training_dataset_admitted is not False:
        errors.append("dry_run_training_admitted")
    return errors


def _review_admission_binding_errors(
    *,
    review: ReviewManifest,
    admission: AdmissionRequest,
    review_manifest_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    matched = 0
    missing = 0
    records_by_review_id: dict[str, list[Any]] = {}
    for record in review.review_records:
        records_by_review_id.setdefault(record.review_id, []).append(record)
    for admission_record in admission.admission_records:
        candidates = records_by_review_id.get(admission_record.review_id, [])
        if not candidates:
            errors.append("admission_review_record_missing")
            missing += 1
            continue
        if len(candidates) > 1:
            errors.append("admission_review_record_ambiguous")
            continue
        matched += 1
        review_record = candidates[0]
        if review_record.document_id != admission_record.document_id or review_record.record_id != admission_record.record_id:
            errors.append("review_record_document_mismatch")
        if review_record.field_name != admission_record.field_name:
            errors.append("review_record_field_mismatch")
        if review_record.decision != admission_record.review_decision:
            errors.append("review_decision_mismatch")
        if admission_record.review_artifact_sha256 != review_manifest_sha256:
            errors.append("review_artifact_sha256_mismatch")
        if admission_record.source_artifact_sha256 != review_record.source_artifact_sha256:
            errors.append("source_artifact_sha256_mismatch")
        if admission_record.action == "admit" and review_record.decision == "reject":
            errors.append("rejected_record_admitted")
        if admission_record.action == "admit" and review_record.decision == "needs_review":
            errors.append("needs_review_record_admitted")
        if admission_record.action == "admit":
            if not admission_record.provenance_summary:
                errors.append("admitted_record_missing_provenance_summary")
            if not admission_record.normalized_value_summary:
                errors.append("admitted_record_missing_normalized_value_summary")
            if not admission_record.admission_reason:
                errors.append("admitted_record_missing_admission_reason")
    return {"errors": errors, "matched": matched, "missing": missing}


def _safe_summary(summary: PackageValidationSummary) -> PackageValidationSummary:
    if not _contains_forbidden_material(summary.model_dump(mode="json")):
        return summary
    minimal = PackageValidationSummary(
        validation_status="failed",
        binding_errors=["package_summary_redaction_failed"],
    )
    if _contains_forbidden_material(minimal.model_dump(mode="json")):
        raise CustomCorpusAdmissionPackageError("package_summary_redaction_failed")
    return minimal


def _contains_forbidden_material(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_contains_forbidden_material(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_forbidden_material(value) for value in payload)
    if isinstance(payload, str):
        if contains_credential_marker(payload):
            return True
        return any(marker in payload for marker in _FORBIDDEN_MARKERS)
    return False


def _first_present(*values: str) -> str:
    for value in values:
        clean = str(value or "").strip()
        if clean:
            return clean
    return ""


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        unique.append(clean)
    return unique


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
