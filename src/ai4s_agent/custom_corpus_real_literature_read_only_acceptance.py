from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "custom_corpus_real_literature_read_only_acceptance_manifest.v1"
PARSED_SUMMARY_SCHEMA_VERSION = "custom_corpus_real_literature_parsed_output_summary.v1"
REPORT_SCHEMA_VERSION = "custom_corpus_real_literature_read_only_acceptance_report.v1"
SUMMARY_SCHEMA_VERSION = "custom_corpus_real_literature_read_only_acceptance_summary.v1"

SUMMARY_BASENAME = "real_literature_read_only_acceptance_summary.json"
REPORT_BASENAME = "real_literature_read_only_acceptance_report.json"
MARKDOWN_BASENAME = "redacted_real_literature_read_only_acceptance_evidence.md"
PARSED_SUMMARY_BASENAMES = ("parsed_output_summary.json", "acceptance_summary.json")

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ALLOWED_MANIFEST_FIELDS = {
    "schema_version",
    "acceptance_id",
    "corpus_id",
    "domain",
    "input_mode",
    "operator_confirmed_access",
    "paper_count",
    "papers",
}
ALLOWED_PAPER_FIELDS = {"paper_id", "parsed_output_basename", "optional_notes_label"}
OLED_PROPERTY_CATEGORIES = {
    "homo",
    "lumo",
    "plqy",
    "delta_est",
    "s1",
    "t1",
    "emission_peak",
    "absorption_peak",
    "eqe",
    "current_efficiency",
    "power_efficiency",
    "luminance",
    "lifetime",
    "device_voltage",
    "host",
    "dopant",
    "unknown_property",
}
FAILURE_LABELS = {
    "parsed_output_missing",
    "parsed_output_invalid_json",
    "parsed_output_schema_invalid",
    "paper_id_mismatch",
    "table_count_missing",
    "candidate_table_count_missing",
    "property_candidate_count_missing",
    "property_field_categories_missing",
    "candidate_status_counts_missing",
    "table_not_found",
    "compound_alias_unresolved",
    "property_header_unmapped",
    "unit_missing",
    "value_ambiguous",
    "molecule_structure_missing",
    "redaction_blocked",
    "dry_run_not_attempted",
    "writer_not_attempted",
}
FORBIDDEN_MARKERS = (
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    ".pdf",
    "/home/",
    "/Users/",
    "C:\\",
    "InChI=",
    "InChIKey",
    "SMILES=",
    "C1=CC",
    "0.72",
    "Authorization",
    "Bearer",
    "token=",
    "secret=",
    "password=",
    "cookie=",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
)
BOUNDARY_FLAGS = {
    "writer_execution_requested": False,
    "writer_execution_preflight_run": False,
    "explicit_confirmation_run": False,
    "controlled_writer_executed": False,
    "training_dataset_materialized": False,
    "dataset_artifact_created": False,
    "serialized_rows_created": False,
    "phase1_status": "not_run",
    "dataset_confirmation_changed": False,
    "model_training_run": False,
    "evaluation_run": False,
}


def run_custom_corpus_real_literature_read_only_acceptance(
    *,
    manifest_path: str | Path,
    parsed_output_root: str | Path,
    output_dir: str | Path,
    max_papers: int = 5,
    operator_id: str = "local-operator",
    require_operator_confirmed_access: bool = True,
    minimum_parseable_papers: int = 1,
    minimum_candidate_tables: int = 1,
    minimum_property_candidate_count: int = 1,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    parsed_root = Path(parsed_output_root)
    output_root = Path(output_dir)

    if not manifest_file.exists():
        return _blocked_summary("real_literature_acceptance_manifest_missing")
    try:
        manifest_text = manifest_file.read_text(encoding="utf-8")
    except OSError:
        return _blocked_summary("real_literature_acceptance_manifest_missing")
    if _contains_forbidden_material(manifest_text):
        return _minimal_redaction_failure()
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError:
        return _blocked_summary("real_literature_acceptance_manifest_invalid_json")
    if not isinstance(manifest, dict):
        return _blocked_summary("real_literature_acceptance_manifest_invalid_json")

    errors, warnings = _validate_manifest(
        manifest,
        operator_id=operator_id,
        require_operator_confirmed_access=require_operator_confirmed_access,
    )
    if errors:
        if "real_literature_read_only_acceptance_redaction_failed" in errors:
            return _minimal_redaction_failure()
        return _blocked_summary(*errors)

    acceptance_id = str(manifest["acceptance_id"])
    run_dir = output_root / acceptance_id
    if run_dir.exists() and any(run_dir.iterdir()):
        return _blocked_summary("real_literature_acceptance_output_dir_not_clean")

    papers = list(manifest.get("papers", []))[: max(0, max_papers)]
    counts = _empty_counts()
    for paper in papers:
        _scan_paper_summary(paper, parsed_root, counts)

    if counts["redaction_failed"]:
        return _minimal_redaction_failure()

    if int(manifest.get("paper_count", 0)) != len(manifest.get("papers", [])):
        warnings.append("paper_count_mismatch")
    if not bool(manifest.get("operator_confirmed_access")):
        warnings.append("operator_access_not_confirmed")
    if counts["parseable_paper_count"] < minimum_parseable_papers:
        warnings.append("minimum_parseable_papers_not_met")
    if counts["candidate_table_count"] < minimum_candidate_tables:
        warnings.append("minimum_candidate_tables_not_met")
    if counts["property_candidate_count"] < minimum_property_candidate_count:
        warnings.append("minimum_property_candidate_count_not_met")
    if counts["parsed_output_missing_count"] > 0 or counts["failure_category_counts"]:
        warnings.append("parsed_output_failures_present")

    status = "acceptance_passed" if not warnings else "acceptance_needs_review"
    report = _report(
        manifest=manifest,
        operator_id=operator_id,
        status=status,
        paper_count_processed=len(papers),
        counts=counts,
        warnings=warnings,
    )
    if _contains_forbidden_material(report):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / REPORT_BASENAME
    _write_json(report_path, report)
    summary = _summary(report=report, report_path=report_path, warnings=warnings)
    markdown = _markdown(report, summary)
    if _contains_forbidden_material(summary) or _contains_forbidden_material(markdown):
        _remove_if_exists(report_path)
        return _minimal_redaction_failure()

    _write_json(run_dir / SUMMARY_BASENAME, summary)
    (run_dir / MARKDOWN_BASENAME).write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real literature read-only acceptance harness.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parsed-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-papers", type=int, default=5)
    parser.add_argument("--operator-id", default="local-operator")
    parser.add_argument("--no-require-operator-confirmed-access", action="store_true")
    parser.add_argument("--minimum-parseable-papers", type=int, default=1)
    parser.add_argument("--minimum-candidate-tables", type=int, default=1)
    parser.add_argument("--minimum-property-candidate-count", type=int, default=1)
    args = parser.parse_args(argv)

    summary = run_custom_corpus_real_literature_read_only_acceptance(
        manifest_path=args.manifest,
        parsed_output_root=args.parsed_output_root,
        output_dir=args.output_dir,
        max_papers=args.max_papers,
        operator_id=args.operator_id,
        require_operator_confirmed_access=not args.no_require_operator_confirmed_access,
        minimum_parseable_papers=args.minimum_parseable_papers,
        minimum_candidate_tables=args.minimum_candidate_tables,
        minimum_property_candidate_count=args.minimum_property_candidate_count,
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary.get("acceptance_status") == "acceptance_blocked" else 0


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    operator_id: str,
    require_operator_confirmed_access: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("real_literature_acceptance_manifest_schema_invalid")
    if _contains_forbidden_material(manifest):
        errors.append("real_literature_read_only_acceptance_redaction_failed")
    if set(manifest) - ALLOWED_MANIFEST_FIELDS:
        errors.append("real_literature_acceptance_manifest_field_not_allowed")
    if not _safe_id(manifest.get("acceptance_id")):
        errors.append("real_literature_acceptance_unsafe_acceptance_id")
    if not _safe_id(manifest.get("corpus_id")):
        errors.append("real_literature_acceptance_unsafe_corpus_id")
    if not _safe_id(operator_id):
        errors.append("real_literature_acceptance_unsafe_operator_id")
    if manifest.get("input_mode") != "local_parsed_outputs":
        errors.append("real_literature_acceptance_input_mode_invalid")
    if require_operator_confirmed_access and manifest.get("operator_confirmed_access") is not True:
        errors.append("real_literature_acceptance_operator_access_not_confirmed")
    papers = manifest.get("papers")
    if not isinstance(manifest.get("paper_count"), int) or int(manifest.get("paper_count", 0)) < 0:
        errors.append("real_literature_acceptance_paper_count_invalid")
    if not isinstance(papers, list):
        errors.append("real_literature_acceptance_papers_invalid")
    else:
        for paper in papers:
            if not isinstance(paper, dict):
                errors.append("real_literature_acceptance_papers_invalid")
                continue
            if set(paper) - ALLOWED_PAPER_FIELDS:
                errors.append("real_literature_acceptance_paper_field_not_allowed")
            if not _safe_id(paper.get("paper_id")):
                errors.append("real_literature_acceptance_unsafe_paper_id")
            basename = paper.get("parsed_output_basename")
            if not _safe_basename(basename):
                errors.append("real_literature_acceptance_unsafe_parsed_output_basename")
            if _contains_forbidden_material(paper):
                errors.append("real_literature_read_only_acceptance_redaction_failed")
    return list(dict.fromkeys(errors)), warnings


def _scan_paper_summary(paper: dict[str, Any], parsed_root: Path, counts: dict[str, Any]) -> None:
    paper_id = str(paper["paper_id"])
    parsed_dir = parsed_root / str(paper["parsed_output_basename"])
    parsed_path = _first_existing(parsed_dir)
    if parsed_path is None:
        _increment_failure(counts, "parsed_output_missing")
        counts["parsed_output_missing_count"] += 1
        return
    try:
        text = parsed_path.read_text(encoding="utf-8")
    except OSError:
        _increment_failure(counts, "parsed_output_missing")
        counts["parsed_output_missing_count"] += 1
        return
    if _contains_forbidden_material(text):
        counts["redaction_failed"] = True
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        _increment_failure(counts, "parsed_output_invalid_json")
        return
    if not isinstance(payload, dict):
        _increment_failure(counts, "parsed_output_invalid_json")
        return
    if _contains_forbidden_material(payload):
        counts["redaction_failed"] = True
        return
    if payload.get("schema_version") != PARSED_SUMMARY_SCHEMA_VERSION:
        _increment_failure(counts, "parsed_output_schema_invalid")
        return
    if payload.get("paper_id") != paper_id:
        _increment_failure(counts, "paper_id_mismatch")

    table_count = _nonnegative_int(payload.get("table_count"))
    candidate_table_count = _nonnegative_int(payload.get("candidate_table_count"))
    property_candidate_count = _nonnegative_int(payload.get("property_candidate_count"))
    if table_count is None:
        _increment_failure(counts, "table_count_missing")
    if candidate_table_count is None:
        _increment_failure(counts, "candidate_table_count_missing")
        candidate_table_count = 0
    if property_candidate_count is None:
        _increment_failure(counts, "property_candidate_count_missing")
        property_candidate_count = 0

    categories = payload.get("property_field_categories")
    if not isinstance(categories, dict):
        _increment_failure(counts, "property_field_categories_missing")
        categories = {}
    statuses = payload.get("candidate_status_counts")
    if not isinstance(statuses, dict):
        _increment_failure(counts, "candidate_status_counts_missing")
        statuses = {}

    counts["parseable_paper_count"] += 1
    counts["candidate_table_count"] += int(candidate_table_count)
    counts["property_candidate_count"] += int(property_candidate_count)
    if int(candidate_table_count) > 0:
        counts["candidate_bearing_paper_count"] += 1
    for category, value in categories.items():
        safe_category = str(category) if str(category) in OLED_PROPERTY_CATEGORIES else "unknown_property"
        count = _nonnegative_int(value) or 0
        counts["property_field_category_counts"][safe_category] = (
            counts["property_field_category_counts"].get(safe_category, 0) + count
        )
    for status in ("accepted", "needs_review", "blocked"):
        counts["candidate_status_counts"][status] += _nonnegative_int(statuses.get(status)) or 0
    failures = payload.get("failure_categories", [])
    if isinstance(failures, list):
        for failure in failures:
            label = str(failure)
            _increment_failure(counts, label if label in FAILURE_LABELS else "redaction_blocked")


def _report(
    *,
    manifest: dict[str, Any],
    operator_id: str,
    status: str,
    paper_count_processed: int,
    counts: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "acceptance_id": manifest["acceptance_id"],
        "acceptance_status": status,
        "corpus_id": manifest["corpus_id"],
        "domain": manifest.get("domain", "unknown"),
        "input_mode": manifest["input_mode"],
        "operator_id": operator_id,
        "operator_confirmed_access": bool(manifest.get("operator_confirmed_access")),
        "paper_count_declared": int(manifest.get("paper_count", 0)),
        "paper_count_processed": paper_count_processed,
        "parseable_paper_count": counts["parseable_paper_count"],
        "parsed_output_missing_count": counts["parsed_output_missing_count"],
        "candidate_bearing_paper_count": counts["candidate_bearing_paper_count"],
        "candidate_table_count": counts["candidate_table_count"],
        "property_candidate_count": counts["property_candidate_count"],
        "property_field_category_counts": dict(sorted(counts["property_field_category_counts"].items())),
        "candidate_status_counts": counts["candidate_status_counts"],
        "failure_category_counts": dict(sorted(counts["failure_category_counts"].items())),
        "redaction_status": "passed",
        **BOUNDARY_FLAGS,
        "report_errors": [],
        "report_warnings": warnings,
    }


def _summary(*, report: dict[str, Any], report_path: Path, warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "acceptance_id": report["acceptance_id"],
        "acceptance_status": report["acceptance_status"],
        "corpus_id": report["corpus_id"],
        "domain": report["domain"],
        "paper_count_processed": report["paper_count_processed"],
        "parseable_paper_count": report["parseable_paper_count"],
        "candidate_bearing_paper_count": report["candidate_bearing_paper_count"],
        "candidate_table_count": report["candidate_table_count"],
        "property_candidate_count": report["property_candidate_count"],
        "redaction_status": "passed",
        "writer_execution_requested": False,
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "report_basename": report_path.name,
        "report_sha256": _sha256_file(report_path),
        "summary_errors": [],
        "summary_warnings": warnings,
    }


def _markdown(report: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Real Literature Read-Only Acceptance Evidence",
            "",
            f"- acceptance_id: {summary['acceptance_id']}",
            f"- acceptance_status: {summary['acceptance_status']}",
            f"- corpus_id: {summary['corpus_id']}",
            f"- paper_count_processed: {summary['paper_count_processed']}",
            f"- parseable_paper_count: {summary['parseable_paper_count']}",
            f"- candidate_table_count: {summary['candidate_table_count']}",
            f"- property_candidate_count: {summary['property_candidate_count']}",
            f"- failure_category_counts: {json.dumps(report['failure_category_counts'], sort_keys=True)}",
            "",
            "This harness is for local read-only acceptance only.",
            "This harness does not commit PDFs.",
            "This harness does not commit raw MinerU outputs.",
            "This harness does not emit raw text.",
            "This harness does not emit source table content.",
            "This harness does not emit raw values.",
            "This harness does not emit paper titles in outputs.",
            "This harness does not emit DOI strings in outputs.",
            "This harness does not execute the controlled writer.",
            "This harness does not create execution requests.",
            "This harness does not run execution request preflight.",
            "This harness does not explicitly confirm execution.",
            "This harness does not materialize training datasets.",
            "This harness does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This harness does not generate conformers.",
            "This harness does not generate DPA3 structures.",
            "This harness does not run model training or evaluation.",
            "",
        ]
    )


def _empty_counts() -> dict[str, Any]:
    return {
        "parseable_paper_count": 0,
        "parsed_output_missing_count": 0,
        "candidate_bearing_paper_count": 0,
        "candidate_table_count": 0,
        "property_candidate_count": 0,
        "property_field_category_counts": {},
        "candidate_status_counts": {"accepted": 0, "needs_review": 0, "blocked": 0},
        "failure_category_counts": {},
        "redaction_failed": False,
    }


def _increment_failure(counts: dict[str, Any], label: str) -> None:
    safe_label = label if label in FAILURE_LABELS else "redaction_blocked"
    counts["failure_category_counts"][safe_label] = counts["failure_category_counts"].get(safe_label, 0) + 1


def _first_existing(parsed_dir: Path) -> Path | None:
    for basename in PARSED_SUMMARY_BASENAMES:
        candidate = parsed_dir / basename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _blocked_summary(*errors: str) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "acceptance_status": "acceptance_blocked",
        "summary_errors": list(errors),
        "summary_warnings": [],
        "redaction_status": "passed",
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
    }


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "acceptance_status": "acceptance_blocked",
        "summary_errors": ["real_literature_read_only_acceptance_redaction_failed"],
        "redaction_status": "failed",
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
    }


def _contains_forbidden_material(value: Any) -> bool:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return any(marker.lower() in text.lower() for marker in FORBIDDEN_MARKERS)


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SAFE_ID_RE.fullmatch(value))


def _safe_basename(value: Any) -> bool:
    if not _safe_id(value):
        return False
    path = Path(str(value))
    return not path.is_absolute() and path.name == str(value) and ".." not in path.parts


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
