from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight.v1"
_DESIGN_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_design_plan.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
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
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
    "inchi=",
    "inchikey",
    "smiles",
    "c1=cc",
    "0.72",
)
_TOP_LEVEL_REQUIRED_FIELDS = (
    "schema_version",
    "design_plan_id",
    "design_plan_status",
    "corpus_id",
    "dataset_name",
    "upstream_evidence",
    "candidate_counts",
    "source_package_refs",
    "value_resolution_contract",
    "boundary_flags",
    "redaction_status",
)
_SAFE_TOP_LEVEL_IDS = ("design_plan_id", "corpus_id", "dataset_name")
_UPSTREAM_STATUS_FIELDS = (
    "quarantined_candidate_admission_boundary_status",
    "domain_validation_boundary_status",
    "controlled_writer_value_resolution_dry_run_precheck_status",
    "property_unit_compatibility_status",
    "numeric_plausibility_status",
    "provenance_consistency_status",
    "compound_alias_association_status",
    "duplicate_conflict_status",
    "redaction_status",
)
_SOURCE_PACKAGE_REFS = (
    "row_contract_id",
    "materialization_plan_id",
    "writer_execution_request_id",
    "writer_input_binding_plan_id",
    "value_source_manifest_id",
    "controlled_writer_execution_plan_id",
    "value_resolution_dry_run_id",
)
_COUNT_FIELDS = (
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
)
_FALSE_BOUNDARY_FLAGS = (
    "controlled_writer_implemented",
    "controlled_writer_executed",
    "writer_dry_run_executed",
    "values_materialized",
    "serialized_rows_created",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)


class PropertyTrainingDatasetControlledWriterDesignPlanPreflightError(ValueError):
    pass


def preflight_property_training_dataset_controlled_writer_design_plan(
    *,
    controlled_writer_design_plan_path: str | Path,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_design_plan_passed: bool = True,
    allow_design_plan_needs_review: bool = False,
    require_domain_validation_passed: bool = True,
    require_values_resolved: bool = True,
    minimum_accepted_candidate_records: int = 1,
) -> dict[str, Any]:
    plan_path = Path(controlled_writer_design_plan_path)
    plan = _load_json(plan_path)
    plan_sha = _sha256_file(plan_path)
    errors: list[str] = []
    warnings: list[str] = []

    _append_errors(errors, _top_level_errors(plan))
    _append_errors(errors, _schema_errors(plan))
    _append_errors(errors, _safe_id_errors(plan))
    _append_errors(
        errors,
        _design_plan_status_errors(
            plan,
            require_design_plan_passed=require_design_plan_passed,
            allow_design_plan_needs_review=allow_design_plan_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(
        errors,
        _upstream_status_errors(
            plan,
            allow_design_plan_needs_review=allow_design_plan_needs_review,
            require_domain_validation_passed=require_domain_validation_passed,
            warnings=warnings,
        ),
    )
    _append_errors(
        errors,
        _candidate_count_errors(
            plan,
            minimum_accepted_candidate_records=minimum_accepted_candidate_records,
            allow_design_plan_needs_review=allow_design_plan_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _source_package_ref_errors(plan))
    _append_errors(
        errors,
        _value_resolution_contract_errors(
            plan,
            require_values_resolved=require_values_resolved,
            allow_design_plan_needs_review=allow_design_plan_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _boundary_flag_errors(plan))
    _append_errors(errors, _redaction_status_errors(plan))
    if _contains_forbidden_material(plan):
        errors.append("controlled_writer_design_plan_contains_unsafe_material")

    status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        status=status,
        plan_path=plan_path,
        plan_sha=plan_sha,
        plan=plan,
        errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        summary = _minimal_redaction_failure()
        markdown = ""

    if output_summary_path is not None:
        write_json(Path(output_summary_path), summary)
    if output_markdown_path is not None and summary.get("redaction_status") != "failed":
        Path(output_markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_markdown_path).write_text(markdown, encoding="utf-8")
    return summary


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
    try:
        summary = preflight_property_training_dataset_controlled_writer_design_plan(
            controlled_writer_design_plan_path=args.controlled_writer_design_plan,
            output_summary_path=args.output_summary,
            output_markdown_path=args.output_markdown,
            require_design_plan_passed=args.require_design_plan_passed,
            allow_design_plan_needs_review=args.allow_design_plan_needs_review,
            require_domain_validation_passed=args.require_domain_validation_passed,
            require_values_resolved=args.require_values_resolved,
            minimum_accepted_candidate_records=args.minimum_accepted_candidate_records,
        )
    except Exception as exc:
        err.write(
            "property training dataset controlled writer design plan preflight invalid: "
            f"{_safe_exception_message(exc)}\n"
        )
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight"
        ),
        description="Preflight a property training dataset controlled writer design plan package.",
    )
    parser.add_argument("--controlled-writer-design-plan", required=True)
    parser.add_argument("--output-summary")
    parser.add_argument("--output-markdown")
    parser.add_argument("--require-design-plan-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-design-plan-needs-review", action="store_true")
    parser.add_argument(
        "--require-domain-validation-passed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-values-resolved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-accepted-candidate-records", type=int, default=1)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterDesignPlanPreflightError("payload must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _top_level_errors(plan: dict[str, Any]) -> list[str]:
    return [f"{field}_missing" for field in _TOP_LEVEL_REQUIRED_FIELDS if field not in plan]


def _schema_errors(plan: dict[str, Any]) -> list[str]:
    if plan.get("schema_version") != _DESIGN_PLAN_SCHEMA_VERSION:
        return ["controlled_writer_design_plan_schema_invalid"]
    return []


def _safe_id_errors(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _SAFE_TOP_LEVEL_IDS:
        if field in plan and not _is_safe_id(plan.get(field)):
            errors.append(f"{field}_unsafe")
    return errors


def _design_plan_status_errors(
    plan: dict[str, Any],
    *,
    require_design_plan_passed: bool,
    allow_design_plan_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    status = plan.get("design_plan_status")
    if status == "passed":
        return []
    if status == "needs_review":
        if allow_design_plan_needs_review or not require_design_plan_passed:
            warnings.append("design_plan_needs_review")
            return []
        return ["design_plan_needs_review"]
    if status in {"blocked", "failed"}:
        return ["design_plan_blocked"]
    return ["design_plan_status_invalid"]


def _upstream_status_errors(
    plan: dict[str, Any],
    *,
    allow_design_plan_needs_review: bool,
    require_domain_validation_passed: bool,
    warnings: list[str],
) -> list[str]:
    upstream = plan.get("upstream_evidence")
    if not isinstance(upstream, dict):
        return ["upstream_evidence_invalid"]
    errors: list[str] = []
    for field in _UPSTREAM_STATUS_FIELDS:
        if field not in upstream:
            errors.append(f"{field}_missing")
            continue
        status = upstream.get(field)
        if status == "passed":
            continue
        if status == "needs_review":
            if (
                allow_design_plan_needs_review
                or (field == "domain_validation_boundary_status" and not require_domain_validation_passed)
            ):
                warnings.append(f"{field}_needs_review")
            else:
                errors.append(f"{field}_needs_review")
            continue
        if status in {"blocked", "failed"}:
            errors.append(f"{field}_blocked")
        else:
            errors.append(f"{field}_invalid")
    return errors


def _candidate_count_errors(
    plan: dict[str, Any],
    *,
    minimum_accepted_candidate_records: int,
    allow_design_plan_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    counts = plan.get("candidate_counts")
    if not isinstance(counts, dict):
        return ["candidate_counts_invalid"]
    errors: list[str] = []
    for field in _COUNT_FIELDS:
        if field not in counts:
            errors.append(f"{field}_missing")
        elif not _is_non_negative_int(counts.get(field)):
            errors.append(f"{field}_invalid")
    if errors:
        return errors
    accepted = counts["accepted_candidate_record_count"]
    needs_review = counts["needs_review_candidate_record_count"]
    blocked = counts["blocked_candidate_record_count"]
    if accepted < minimum_accepted_candidate_records:
        errors.append("minimum_accepted_candidate_records_not_met")
    if blocked > 0:
        errors.append("blocked_candidate_records_present")
    if needs_review > 0:
        if allow_design_plan_needs_review:
            warnings.append("needs_review_candidate_records_present")
        else:
            errors.append("needs_review_candidate_records_present")
    return errors


def _source_package_ref_errors(plan: dict[str, Any]) -> list[str]:
    refs = plan.get("source_package_refs")
    if not isinstance(refs, dict):
        return ["source_package_refs_invalid"]
    errors: list[str] = []
    for field in _SOURCE_PACKAGE_REFS:
        if field not in refs:
            errors.append(f"{field}_missing")
        elif not _is_safe_id(refs.get(field)):
            errors.append(f"{field}_unsafe")
    return errors


def _value_resolution_contract_errors(
    plan: dict[str, Any],
    *,
    require_values_resolved: bool,
    allow_design_plan_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    contract = plan.get("value_resolution_contract")
    if not isinstance(contract, dict):
        return ["value_resolution_contract_invalid"]
    errors: list[str] = []
    status = contract.get("controlled_writer_value_resolution_dry_run_status")
    if status == "passed":
        pass
    elif status == "needs_review":
        if allow_design_plan_needs_review:
            warnings.append("controlled_writer_value_resolution_dry_run_status_needs_review")
        else:
            errors.append("controlled_writer_value_resolution_dry_run_status_needs_review")
    elif status in {"blocked", "failed"}:
        errors.append("controlled_writer_value_resolution_dry_run_status_blocked")
    else:
        errors.append("controlled_writer_value_resolution_dry_run_status_invalid")

    if "values_resolved" not in contract:
        errors.append("values_resolved_missing")
    elif contract.get("values_resolved") is not True:
        if require_values_resolved:
            errors.append("values_not_resolved")
        else:
            warnings.append("values_not_resolved")

    missing = contract.get("missing_required_field_count")
    if not _is_non_negative_int(missing):
        errors.append("missing_required_field_count_invalid")
    elif missing > 0:
        if require_values_resolved:
            errors.append("missing_required_fields")
        else:
            warnings.append("missing_required_fields")
    return errors


def _boundary_flag_errors(plan: dict[str, Any]) -> list[str]:
    flags = plan.get("boundary_flags")
    if not isinstance(flags, dict):
        return ["boundary_flags_invalid"]
    errors: list[str] = []
    for field in _FALSE_BOUNDARY_FLAGS:
        if field not in flags:
            errors.append(f"{field}_missing")
        elif flags.get(field) is not False:
            errors.append(field)
    if flags.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    return errors


def _redaction_status_errors(plan: dict[str, Any]) -> list[str]:
    if plan.get("redaction_status") != "passed":
        return ["redaction_status_blocked"]
    return []


def _summary(
    *,
    status: str,
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    counts = plan.get("candidate_counts") if isinstance(plan.get("candidate_counts"), dict) else {}
    flags = plan.get("boundary_flags") if isinstance(plan.get("boundary_flags"), dict) else {}
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": status,
        "controlled_writer_design_plan_path": plan_path.name,
        "controlled_writer_design_plan_sha256": plan_sha,
        "design_plan_id": _safe_summary_value(plan.get("design_plan_id")),
        "design_plan_status": _safe_summary_value(plan.get("design_plan_status")),
        "corpus_id": _safe_summary_value(plan.get("corpus_id")),
        "dataset_name": _safe_summary_value(plan.get("dataset_name")),
        "accepted_candidate_record_count": _safe_count(counts.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_count(counts.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_count(counts.get("blocked_candidate_record_count")),
        "redaction_status": "passed",
        "controlled_writer_implemented": flags.get("controlled_writer_implemented") is True,
        "controlled_writer_executed": flags.get("controlled_writer_executed") is True,
        "writer_dry_run_executed": flags.get("writer_dry_run_executed") is True,
        "values_materialized": flags.get("values_materialized") is True,
        "serialized_rows_created": flags.get("serialized_rows_created") is True,
        "training_dataset_materialized": flags.get("training_dataset_materialized") is True,
        "dataset_artifact_created": flags.get("dataset_artifact_created") is True,
        "phase1_status": _safe_summary_value(flags.get("phase1_status")),
        "dataset_confirmation_changed": flags.get("dataset_confirmation_changed") is True,
        "model_training_run": flags.get("model_training_run") is True,
        "evaluation_run": flags.get("evaluation_run") is True,
        "preflight_errors": errors,
        "preflight_warnings": warnings,
    }


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Controlled Writer Design Plan Preflight Evidence",
            "",
            "This is a controlled writer design plan preflight only.",
            "",
            f"- preflight_status: {summary.get('preflight_status', '')}",
            f"- design_plan_id: {summary.get('design_plan_id', '')}",
            f"- corpus_id: {summary.get('corpus_id', '')}",
            f"- dataset_name: {summary.get('dataset_name', '')}",
            f"- accepted_candidate_record_count: {summary.get('accepted_candidate_record_count', 0)}",
            f"- needs_review_candidate_record_count: {summary.get('needs_review_candidate_record_count', 0)}",
            f"- blocked_candidate_record_count: {summary.get('blocked_candidate_record_count', 0)}",
            f"- preflight_errors: {summary.get('preflight_errors', [])}",
            f"- preflight_warnings: {summary.get('preflight_warnings', [])}",
            "",
            "Boundary statement:",
            "- controlled writer was not implemented",
            "- controlled writer was not executed",
            "- writer dry-run was not executed",
            "- source payloads were not read",
            "- raw values were not emitted",
            "- values were not materialized",
            "- training rows were not serialized",
            "- training dataset artifacts were not created",
            "- Phase 1 did not run",
            "- DatasetConfirmation was not changed",
            "- model training and evaluation did not run",
            "",
        ]
    )


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "blocked",
        "preflight_errors": [
            "property_training_dataset_controlled_writer_design_plan_preflight_redaction_failed"
        ],
        "redaction_status": "failed",
    }


def _contains_forbidden_material(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    if _ABSOLUTE_PATH_VALUE_RE.search(text):
        return True
    return any(marker.lower() in text for marker in _FORBIDDEN_MARKERS)


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    if _contains_forbidden_material(message):
        return "redacted invalid input"
    return message


def _safe_summary_value(value: Any) -> str:
    if isinstance(value, str) and _is_safe_id(value):
        return value
    return ""


def _safe_count(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0


def _is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and bool(_SAFE_ID_RE.fullmatch(value))


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
