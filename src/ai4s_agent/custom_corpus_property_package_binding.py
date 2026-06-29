from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import AdmissionRequest, load_admission_request
from ai4s_agent.custom_corpus_admission_package import validate_admission_package
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport
from ai4s_agent.custom_corpus_manifest import CustomCorpusManifest, load_custom_corpus_manifest, sha256_file
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_SCHEMA_VERSION = "custom_corpus_property_package_binding.v1"
_PRECHECK_SCHEMA_VERSION = "custom_corpus_property_admission_draft_package_precheck.v1"
_FORMAL_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_ABSOLUTE_PATH_VALUE_RE = re.compile(r'"(?:/|[A-Za-z]:\\\\)')
_FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "C:\\",
    "Authorization",
    "Bearer",
    "token",
    "secret",
    "password",
    "cookie",
    "x-api-key",
    ".pdf",
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
)
_ARTIFACTS = {
    "formal_package_validation": "custom_corpus_admission_package_validation.json",
    "wrapper_summary": "property_package_binding_summary.json",
    "evidence_markdown": "redacted_property_package_binding_evidence.md",
}


class CustomCorpusPropertyPackageBindingError(ValueError):
    pass


def run_property_package_binding(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
    property_precheck_summary_path: str | Path,
    output_dir: str | Path,
    binding_run_id: str,
    confirm_formal_package_binding: bool,
    allow_precheck_needs_review: bool = False,
) -> dict[str, Any]:
    binding_run_id = _required_safe_id(binding_run_id, field_name="binding_run_id")
    manifest = load_custom_corpus_manifest(manifest_path)
    dry_run = _load_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    precheck = _load_property_precheck(property_precheck_summary_path)

    paths = {
        "manifest_path": manifest_path,
        "dry_run_report_path": dry_run_report_path,
        "review_manifest_path": review_manifest_path,
        "admission_request_path": admission_request_path,
        "property_precheck_summary_path": property_precheck_summary_path,
    }
    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_request_sha256": _safe_sha_for_path(admission_request_path),
        "property_precheck_summary_sha256": _safe_sha_for_path(property_precheck_summary_path),
    }
    run_dir = Path(output_dir).expanduser() / binding_run_id

    binding_errors: list[str] = []
    warnings: list[str] = []
    if not confirm_formal_package_binding:
        binding_errors.append("formal_package_binding_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        binding_errors.append("output_directory_not_clean")
    _append_errors(
        binding_errors,
        _local_precheck_errors(
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            precheck=precheck,
            hashes=hashes,
            allow_precheck_needs_review=allow_precheck_needs_review,
            warnings=warnings,
        ),
    )

    if binding_errors:
        summary = _wrapper_summary(
            binding_status="failed",
            binding_run_id=binding_run_id,
            paths=paths,
            hashes=hashes,
            formal_package_validation_sha256="",
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            precheck=precheck,
            formal_summary={},
            binding_errors=_stable_unique(binding_errors),
            warnings=_stable_unique(warnings),
        )
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    formal_summary = validate_admission_package(
        manifest_path=manifest_path,
        dry_run_report_path=dry_run_report_path,
        review_manifest_path=review_manifest_path,
        admission_request_path=admission_request_path,
    )
    if formal_summary.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        binding_errors.append("formal_package_validation_schema_invalid")
    if formal_summary.get("validation_status") != "passed":
        binding_errors.append("formal_package_validation_failed")

    if binding_errors:
        binding_status = "failed"
    elif precheck["precheck_status"] == "needs_review":
        binding_status = "needs_review"
    else:
        binding_status = "passed"

    summary = _wrapper_summary(
        binding_status=binding_status,
        binding_run_id=binding_run_id,
        paths=paths,
        hashes=hashes,
        formal_package_validation_sha256="",
        manifest=manifest,
        dry_run=dry_run,
        review=review,
        admission=admission,
        precheck=precheck,
        formal_summary=formal_summary,
        binding_errors=_stable_unique(binding_errors),
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "formal": formal_summary, "evidence": evidence}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    formal_path = run_dir / _ARTIFACTS["formal_package_validation"]
    wrapper_path = run_dir / _ARTIFACTS["wrapper_summary"]
    evidence_path = run_dir / _ARTIFACTS["evidence_markdown"]
    write_json(formal_path, formal_summary)
    formal_sha = _safe_sha_for_path(formal_path)
    summary["formal_package_validation_sha256"] = formal_sha
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "evidence": evidence}):
        formal_path.unlink(missing_ok=True)
        return _minimal_redaction_failure()
    write_json(wrapper_path, summary)
    evidence_path.write_text(evidence, encoding="utf-8")
    return summary


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)
    try:
        summary = run_property_package_binding(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_request_path=args.admission_request,
            property_precheck_summary_path=args.property_precheck_summary,
            output_dir=args.output_dir,
            binding_run_id=args.binding_run_id,
            confirm_formal_package_binding=args.confirm_formal_package_binding,
            allow_precheck_needs_review=args.allow_precheck_needs_review,
        )
    except Exception as exc:
        err.write(f"property package binding invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("binding_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_package_binding",
        description="Run property-aware formal custom corpus admission package binding.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--property-precheck-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--binding-run-id", required=True)
    parser.add_argument("--confirm-formal-package-binding", action="store_true")
    parser.add_argument("--allow-precheck-needs-review", action="store_true")
    return parser


def _local_precheck_errors(
    *,
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    precheck: dict[str, Any],
    hashes: dict[str, str],
    allow_precheck_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if precheck["precheck_status"] == "failed":
        errors.append("property_precheck_failed")
    if precheck["precheck_status"] == "needs_review":
        if allow_precheck_needs_review:
            warnings.append("property_precheck_needs_review_allowed")
        else:
            errors.append("property_precheck_needs_review")
    if precheck["dry_run_decision"] != "passed" or dry_run.decision != "passed":
        errors.append("dry_run_not_passed")
    if precheck["phase1_status"] != "not_run" or dry_run.confirmation_boundary.phase1_status != "not_run":
        errors.append("dry_run_phase1_ran")
    if precheck["training_admitted"] is not False or dry_run.confirmation_boundary.training_dataset_admitted is not False:
        errors.append("dry_run_training_admitted")
    if dry_run.confirmation_boundary.dataset_confirmation_confirmed is not False:
        errors.append("dry_run_dataset_confirmed")
    if precheck["draft_status"] != "written":
        errors.append("draft_not_written")
    if precheck["precheck_errors"]:
        errors.append("property_precheck_has_errors")

    if precheck["manifest_sha256"] != hashes["manifest_sha256"]:
        errors.append("manifest_sha256_mismatch")
    if precheck["dry_run_report_sha256"] != hashes["dry_run_report_sha256"]:
        errors.append("dry_run_report_sha256_mismatch")
    if precheck["review_manifest_sha256"] != hashes["review_manifest_sha256"]:
        errors.append("review_manifest_sha256_mismatch")
    if precheck["admission_draft_sha256"] != hashes["admission_request_sha256"]:
        errors.append("admission_request_sha256_mismatch")

    if len({manifest.corpus_id, dry_run.corpus_id, review.corpus_id, admission.corpus_id, precheck["corpus_id"]}) != 1:
        errors.append("corpus_id_mismatch")
    if len({dry_run.run_id, review.dry_run_id, admission.dry_run_id, precheck["dry_run_id"]}) != 1:
        errors.append("dry_run_id_mismatch")
    if len({review.review_manifest_id, admission.review_manifest_id, precheck["review_manifest_id"]}) != 1:
        errors.append("review_manifest_id_mismatch")
    if admission.admission_request_id != precheck["admission_request_id"]:
        errors.append("admission_request_id_mismatch")

    admit_ids = [record.record_id for record in admission.admission_records if record.action == "admit"]
    exclude_ids = [record.record_id for record in admission.admission_records if record.action == "exclude"]
    admission_ids = {record.record_id for record in admission.admission_records}
    if len(admission.admission_records) != precheck["admission_record_count"]:
        errors.append("admission_record_count_mismatch")
    if len(admit_ids) != precheck["admit_count"] or set(admit_ids) != set(precheck["admit_record_ids"]):
        errors.append("admit_count_mismatch")
    if len(exclude_ids) != precheck["exclude_count"] or set(exclude_ids) != set(precheck["exclude_record_ids"]):
        errors.append("exclude_count_mismatch")
    if admission_ids & set(precheck["blocked_record_ids"]):
        errors.append("blocked_record_in_admission_request")
    return _stable_unique(errors)


def _wrapper_summary(
    *,
    binding_status: str,
    binding_run_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    formal_package_validation_sha256: str,
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    precheck: dict[str, Any],
    formal_summary: dict[str, Any],
    binding_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "binding_status": binding_status,
        "binding_run_id": binding_run_id,
        "manifest_path": _basename(paths["manifest_path"], "manifest.json"),
        "manifest_sha256": hashes["manifest_sha256"],
        "dry_run_report_path": _basename(paths["dry_run_report_path"], "dry_run_report.json"),
        "dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "review_manifest_path": _basename(paths["review_manifest_path"], "review_manifest.json"),
        "review_manifest_sha256": hashes["review_manifest_sha256"],
        "admission_request_path": _basename(paths["admission_request_path"], "custom_corpus_admission.draft.json"),
        "admission_request_sha256": hashes["admission_request_sha256"],
        "property_precheck_summary_path": _basename(paths["property_precheck_summary_path"], "property_precheck_summary.json"),
        "property_precheck_summary_sha256": hashes["property_precheck_summary_sha256"],
        "formal_package_validation_path": _ARTIFACTS["formal_package_validation"] if formal_summary else "",
        "formal_package_validation_sha256": formal_package_validation_sha256,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "review_queue_id": precheck.get("review_queue_id", ""),
        "property_candidate_manifest_id": precheck.get("property_candidate_manifest_id", ""),
        "property_precheck_status": precheck["precheck_status"],
        "formal_package_validation_status": str(formal_summary.get("validation_status", "")),
        "dry_run_decision": dry_run.decision,
        "phase1_status": dry_run.confirmation_boundary.phase1_status,
        "training_admitted": dry_run.confirmation_boundary.training_dataset_admitted,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(precheck["blocked_record_ids"]),
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": precheck["blocked_record_ids"],
        "binding_errors": binding_errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _load_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyPackageBindingError("dry-run report invalid") from exc


def _load_property_precheck(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyPackageBindingError("precheck summary invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _PRECHECK_SCHEMA_VERSION:
        raise CustomCorpusPropertyPackageBindingError("precheck summary schema_version invalid")
    clean = dict(payload)
    if clean.get("precheck_status") not in {"passed", "needs_review", "failed"}:
        raise CustomCorpusPropertyPackageBindingError("precheck_status invalid")
    for field in (
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        "review_queue_id",
        "property_candidate_manifest_id",
    ):
        clean[field] = _optional_safe_id(clean.get(field), field_name=field)
        if field in {"corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id"} and not clean[field]:
            raise CustomCorpusPropertyPackageBindingError(f"{field} invalid")
    for field in (
        "manifest_sha256",
        "dry_run_report_sha256",
        "review_manifest_sha256",
        "admission_draft_sha256",
        "property_precheck_summary_sha256",
        "formal_package_validation_sha256",
    ):
        if field in clean:
            clean[field] = _optional_sha(clean.get(field), field_name=field)
    clean["dry_run_decision"] = _safe_label(clean.get("dry_run_decision"), field_name="dry_run_decision")
    clean["phase1_status"] = _safe_label(clean.get("phase1_status"), field_name="phase1_status")
    clean["draft_status"] = _safe_label(clean.get("draft_status"), field_name="draft_status")
    clean["training_admitted"] = bool(clean.get("training_admitted"))
    clean["admission_record_count"] = int(clean.get("admission_record_count", clean.get("draft_record_count", 0)))
    clean["admit_count"] = int(clean.get("admit_count", 0))
    clean["exclude_count"] = int(clean.get("exclude_count", 0))
    clean["blocked_record_count"] = int(clean.get("blocked_record_count", 0))
    clean["admit_record_ids"] = _safe_id_list(clean.get("admit_record_ids"), field_name="admit_record_ids")
    clean["exclude_record_ids"] = _safe_id_list(clean.get("exclude_record_ids"), field_name="exclude_record_ids")
    clean["blocked_record_ids"] = _safe_id_list(clean.get("blocked_record_ids"), field_name="blocked_record_ids")
    clean["precheck_errors"] = _safe_id_list(clean.get("precheck_errors"), field_name="precheck_errors")
    clean["warnings"] = _safe_id_list(clean.get("warnings"), field_name="warnings")
    return clean


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Package Binding Evidence",
            "",
            f"- Binding run id: `{summary['binding_run_id']}`",
            f"- Binding status: `{summary['binding_status']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Property precheck status: `{summary['property_precheck_status']}`",
            f"- Formal package validation status: `{summary['formal_package_validation_status']}`",
            f"- Formal package validation SHA-256: `{summary['formal_package_validation_sha256']}`",
            f"- Admission record count: `{summary['admission_record_count']}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Binding errors: `{json.dumps(summary['binding_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- formal package binding was run.",
            "- No materialization was run.",
            "- No materialization plan was created.",
            "- No candidate/training CSV was created.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "- No training data was admitted.",
            "",
        ]
    )


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = _optional_safe_id(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    return clean


def _optional_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    if not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    return clean


def _safe_label(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    return clean


def _safe_id_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    return [_required_safe_id(item, field_name=field_name) for item in value]


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyPackageBindingError(f"{field_name} invalid")
    return f"sha256:{match.group(2).lower()}"


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _basename(path: str | Path, fallback: str) -> str:
    return Path(path).name or fallback


def _append_errors(errors: list[str], additions: list[str]) -> None:
    for error in additions:
        if error not in errors:
            errors.append(error)


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "binding_status": "failed",
        "binding_errors": ["property_package_binding_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: BaseException) -> str:
    message = str(exc or "").lower()
    if any(marker.lower() in message for marker in _FORBIDDEN_MARKERS):
        return "artifact invalid"
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(message)):
        return "artifact invalid"
    if "schema" in message:
        return "schema_version invalid"
    if "invalid" in message:
        return "artifact invalid"
    return exc.__class__.__name__


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
