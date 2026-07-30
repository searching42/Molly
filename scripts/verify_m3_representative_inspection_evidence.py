#!/usr/bin/env python3
"""Verify committed M3 representative inspection evidence without source paths."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _m3_inspection_validation import (  # noqa: E402
    BLOCKER_REQUIREMENTS,
    CASE_FILENAME_BY_ID,
    CASE_ROSTER,
    EVIDENCE_VERSION,
    INSPECTION_VERSION,
    OWNER_REVIEW_VERSION,
    SOURCE_CLASSES,
    _pending_review_checks,
    canonical_json_bytes,
    parse_canonical_json,
    public_privacy_violations,
    runner_code_binding,
    sha256_bytes,
    source_contract_preflight,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL_MANIFEST_PATH = (
    "docs/evidence/m3-representative-inspection-validation-v1/"
    "evidence_manifest.json"
)
_REVIEW_DECISIONS = frozenset({"approved", "changes_requested", "inconclusive"})
_REVIEWER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def verify_evidence(
    root: Path,
    *,
    require_complete: bool = False,
    expected_runner_commit: str | None = None,
) -> dict[str, Any]:
    expected_top = {
        "README.md",
        "evidence_manifest.json",
        "evidence_summary.md",
        "owner_review.json",
        "review_checklist.md",
        "cases",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_top:
        raise ValueError("evidence package roster is invalid")
    case_dir = root / "cases"
    if {path.name for path in case_dir.iterdir()} != set(CASE_FILENAME_BY_ID.values()):
        raise ValueError("evidence case roster is invalid")
    package = b"".join(
        path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    )
    if public_privacy_violations(package):
        raise ValueError("evidence package privacy scan failed")
    manifest_bytes = (root / "evidence_manifest.json").read_bytes()
    manifest = parse_canonical_json(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("evidence manifest is invalid")
    if (
        manifest.get("evidence_version") != EVIDENCE_VERSION
        or manifest.get("inspection_version") != INSPECTION_VERSION
    ):
        raise ValueError("evidence contract version is invalid")
    if manifest.get("source_contracts") != {
        "trajectory": "scientific_agent_trajectory_projection.v1",
        "audit": "scientific_agent_trajectory_audit_metrics.v1",
        "attribution": "scientific_agent_failure_attribution.v1",
    }:
        raise ValueError("evidence source contracts are invalid")
    runner_commit = manifest.get("runner_commit")
    if expected_runner_commit is not None and runner_commit != expected_runner_commit:
        raise ValueError("evidence runner commit does not match the expected commit")
    try:
        expected_binding = runner_code_binding(str(runner_commit or ""))
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise ValueError("evidence runner binding is unavailable") from exc
    if manifest.get("runner_code_binding") != expected_binding:
        raise ValueError("evidence runner code binding mismatch")

    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or [item.get("case_id") for item in cases if isinstance(item, dict)]
        != list(CASE_ROSTER)
    ):
        raise ValueError("evidence manifest case order is invalid")
    records = []
    for item in cases:
        if not isinstance(item, dict) or item.get("source_class") not in SOURCE_CLASSES:
            raise ValueError("evidence case metadata is invalid")
        case_path = root / str(item.get("case_file"))
        payload = case_path.read_bytes()
        if sha256_bytes(payload) != item.get("case_file_sha256"):
            raise ValueError("evidence case digest mismatch")
        record = parse_canonical_json(payload)
        if not isinstance(record, dict) or record.get("case_id") != item.get("case_id"):
            raise ValueError("evidence case identity mismatch")
        if record.get("human_review_status") != "pending":
            raise ValueError("case evidence is immutable; review belongs in owner_review.json")
        _verify_case_record(record, item)
        records.append(record)

    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("evidence summary is invalid")
    complete = all(
        record.get("machine_validation_status") == "passed" for record in records
    )
    design_blocked = sum(
        record.get("case_status") == "design_analysis_blocked" for record in records
    )
    not_executed = sum(
        record.get("machine_validation_status") == "not_executed"
        for record in records
    )
    if summary.get("machine_evidence_complete") is not complete:
        raise ValueError("machine evidence completeness claim is inconsistent")
    if summary.get("design_analysis_blocked_count") != design_blocked:
        raise ValueError("design-analysis blocker count is inconsistent")
    if summary.get("not_executed_count") != not_executed:
        raise ValueError("not-executed count is inconsistent")
    if manifest.get("claims") != {
        "evidence_only": True,
        "runtime_case_evidence_included": True,
        "all_cases_runtime_executed": complete,
        "design_analysis_blockers_included": not complete,
        "observer_only": True,
        "scientific_validation_claimed": False,
        "benchmark_result_claimed": False,
        "full_private_source_bundles_committed": False,
        "human_review_required": True,
    }:
        raise ValueError("evidence claim boundary is invalid")
    if (
        summary.get("human_review_status") != "pending"
        or summary.get("m3_v_status") != "not_yet_claimed"
    ):
        raise ValueError("immutable manifest cannot claim owner review or M3 V")
    review_status = _verify_owner_review(
        root / "owner_review.json",
        manifest_sha256=sha256_bytes(manifest_bytes),
    )
    if require_complete and not complete:
        raise ValueError("machine evidence is incomplete")
    return {
        "case_count": len(records),
        "machine_evidence_complete": complete,
        "design_analysis_blocked_count": design_blocked,
        "human_review_status": review_status,
        "m3_v_eligible": complete and review_status == "approved",
    }


def _verify_case_record(record: dict[str, Any], item: dict[str, Any]) -> None:
    case_id = str(record["case_id"])
    if record.get("machine_validation_status") != item.get("machine_validation_status"):
        raise ValueError("evidence case status mismatch")
    if case_id in BLOCKER_REQUIREMENTS:
        if record.get("case_status") != "design_analysis_blocked":
            raise ValueError("source-contract blocker status is invalid")
        if record.get("machine_validation_status") != "not_executed":
            raise ValueError("source-contract blocker cannot claim machine validation")
        nullable = (
            "fresh_process_bytes_equal",
            "hash_seed_bytes_equal",
            "fresh_process_distinct_pids",
            "privacy_scan_passed",
            "inspection_http_status",
            "inspection_response",
        )
        if any(record.get(key) is not None for key in nullable):
            raise ValueError("non-executed comparison fields must be null")
        expected = source_contract_preflight(case_id)
        if record.get("blocker_evidence") != expected:
            raise ValueError("source-contract blocker evidence mismatch")
        return
    if record.get("case_status") != "executed" or record.get("blocker_evidence") is not None:
        raise ValueError("executed evidence case status is invalid")
    response = record.get("inspection_response")
    if response is not None and sha256_bytes(canonical_json_bytes(response)) != record.get(
        "inspection_response_sha256"
    ):
        raise ValueError("inspection response digest mismatch")


def _verify_owner_review(path: Path, *, manifest_sha256: str) -> str:
    review = parse_canonical_json(path.read_bytes())
    if not isinstance(review, dict) or review.get("review_version") != OWNER_REVIEW_VERSION:
        raise ValueError("owner review contract is invalid")
    if review.get("reviewed_evidence_manifest_sha256") != manifest_sha256:
        raise ValueError("owner review manifest binding mismatch")
    decisions = review.get("per_case_decisions")
    if (
        not isinstance(decisions, list)
        or [item.get("case_id") for item in decisions if isinstance(item, dict)]
        != list(CASE_ROSTER)
    ):
        raise ValueError("owner review case roster is invalid")
    decision = review.get("decision")
    if decision is None:
        if any(
            review.get(field) is not None
            for field in ("reviewer", "review_date", "reviewed_commit", "notes")
        ):
            raise ValueError("pending owner review metadata must be null")
        for item in decisions:
            _verify_review_item(item, pending=True)
        return "pending"
    if decision not in _REVIEW_DECISIONS:
        raise ValueError("owner review decision is invalid")
    reviewer = review.get("reviewer")
    reviewed_commit = review.get("reviewed_commit")
    if not isinstance(reviewer, str) or not _REVIEWER.fullmatch(reviewer):
        raise ValueError("owner reviewer identity is invalid")
    if not isinstance(reviewed_commit, str) or not _FULL_SHA.fullmatch(reviewed_commit):
        raise ValueError("owner reviewed commit is invalid")
    try:
        dt.date.fromisoformat(str(review.get("review_date")))
    except ValueError as exc:
        raise ValueError("owner review date is invalid") from exc
    _verify_reviewed_manifest_commit(reviewed_commit, manifest_sha256)
    for item in decisions:
        _verify_review_item(item, pending=False)
    if decision == "approved" and any(
        item.get("decision") != "approved"
        or not all(value is True for value in item["checks"].values())
        for item in decisions
    ):
        raise ValueError("owner approval requires every case check to pass")
    notes = review.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("owner review notes are invalid")
    return str(decision)


def _verify_review_item(item: Any, *, pending: bool) -> None:
    if not isinstance(item, dict):
        raise ValueError("owner review case entry is invalid")
    case_id = str(item.get("case_id") or "")
    expected_kind = (
        "blocker_diagnosis" if case_id in BLOCKER_REQUIREMENTS else "executable_case"
    )
    if item.get("review_kind") != expected_kind:
        raise ValueError("owner review kind is invalid")
    expected_checks = set(_pending_review_checks(case_id))
    checks = item.get("checks")
    if not isinstance(checks, dict) or set(checks) != expected_checks:
        raise ValueError("owner review checks are invalid")
    if pending:
        if item.get("decision") is not None or item.get("notes") is not None:
            raise ValueError("pending per-case review must be empty")
        if any(value is not None for value in checks.values()):
            raise ValueError("pending per-case checks must be null")
        return
    if item.get("decision") not in _REVIEW_DECISIONS:
        raise ValueError("per-case owner review decision is invalid")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError("completed per-case checks must be boolean")
    notes = item.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("per-case owner review notes are invalid")


def _verify_reviewed_manifest_commit(commit: str, expected_digest: str) -> None:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{_CANONICAL_MANIFEST_PATH}"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("owner reviewed commit does not contain the evidence manifest") from exc
    if sha256_bytes(completed.stdout) != expected_digest:
        raise ValueError("owner reviewed commit does not bind this evidence manifest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-runner-commit")
    args = parser.parse_args(argv)
    try:
        result = verify_evidence(
            args.evidence_dir,
            require_complete=args.require_complete,
            expected_runner_commit=args.expected_runner_commit,
        )
    except ValueError as exc:
        print(f"evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"case_count={result['case_count']}; "
        f"machine_evidence_complete={str(result['machine_evidence_complete']).lower()}; "
        f"design_analysis_blocked={result['design_analysis_blocked_count']}; "
        f"human_review={result['human_review_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
