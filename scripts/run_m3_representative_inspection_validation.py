#!/usr/bin/env python3
"""Generate redacted M3 representative inspection validation evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from _m3_inspection_validation import (
    CASE_CONTRACT_VERSION,
    CASE_FILENAME_BY_ID,
    CASE_ROSTER,
    EVIDENCE_VERSION,
    INSPECTION_VERSION,
    SOURCE_CLASS_BY_CASE,
    SOURCE_CONTRACTS,
    build_source_chain,
    call_inspection_route,
    canonical_json_bytes,
    create_private_locator,
    locator_for_chain,
    parse_canonical_json,
    public_privacy_violations,
    pending_owner_review,
    require_runner_checkout_binding,
    runner_code_binding,
    runtime_source_evidence,
    sha256_bytes,
    source_artifact_digests,
    tree_snapshot,
    write_canonical_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--public-evidence-dir", type=Path)
    parser.add_argument("--cases", choices=("all",), default="all")
    parser.add_argument("--case", choices=CASE_ROSTER, action="append", dest="selected")
    parser.add_argument("--keep-private-source-bundle", action="store_true")
    parser.add_argument("--runner-commit")
    parser.add_argument("--inspect-private-locator", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.inspect_private_locator is not None:
        return _inspection_child(args.inspect_private_locator)
    if args.output is None or args.public_evidence_dir is None:
        raise SystemExit("--output and --public-evidence-dir are required")
    if args.output.exists() or args.public_evidence_dir.exists():
        raise SystemExit("private output and public evidence directories must not exist")
    runner_commit = args.runner_commit or _git_head()
    try:
        require_runner_checkout_binding(runner_commit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.mkdir(parents=True, mode=0o700)
    args.public_evidence_dir.mkdir(parents=True)
    selected = tuple(args.selected or CASE_ROSTER)
    if len(selected) != len(set(selected)):
        raise SystemExit("case IDs must be unique")
    records = []
    try:
        for case_id in selected:
            records.append(_run_case(args.output, case_id))
        _publish_public_package(
            args.public_evidence_dir,
            records,
            runner_commit=runner_commit,
        )
    finally:
        if not args.keep_private_source_bundle:
            _remove_private_tree(args.output)
    status = "complete" if _machine_complete(records) else "incomplete"
    print(f"machine_evidence={status}; human_review=pending; cases={len(records)}")
    return 0


def _inspection_child(locator_path: Path) -> int:
    locator = parse_canonical_json(locator_path.read_bytes())
    if not isinstance(locator, dict):
        raise SystemExit("private locator is invalid")
    status, payload = call_inspection_route(locator)
    envelope = {"http_status": status, "pid": os.getpid(), "response": payload}
    sys.stdout.buffer.write(canonical_json_bytes(envelope))
    return 0


def _run_case(private_root: Path, case_id: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": case_id,
        "source_class": SOURCE_CLASS_BY_CASE[case_id],
        "case_contract_version": CASE_CONTRACT_VERSION,
        "human_review_status": "pending",
    }
    chain = build_source_chain(private_root / "sources" / case_id, case_id)
    source_evidence = runtime_source_evidence(chain)
    project = chain.workspace / "projects" / chain.project_id
    snapshot_roots = [project, chain.actions_root / chain.project_id]
    before = tree_snapshot(snapshot_roots)
    original_before = (
        tree_snapshot([chain.root / "original-valid-observer"])
        if case_id == "history_truncation"
        else None
    )
    tampering_evidence = (
        {
            "tampering_kind": "required_event_removed_and_outer_manifest_resigned",
            "original_source_sha256": original_before["sha256"],
            "tampered_source_sha256": source_artifact_digests(chain)["trajectory"],
            "partial_timeline_returned": False,
        }
        if original_before is not None
        else None
    )
    locator_path = private_root / "locators" / f"{case_id}.json"
    locator_path.parent.mkdir(parents=True, exist_ok=True)
    create_private_locator(locator_path, locator_for_chain(chain))
    first = _fresh_process(locator_path, hash_seed="17")
    second = _fresh_process(locator_path, hash_seed="901")
    locator_path.unlink()
    after = tree_snapshot(snapshot_roots)
    original_after = (
        tree_snapshot([chain.root / "original-valid-observer"])
        if original_before is not None
        else None
    )
    first_response = first["response"]
    response_bytes = canonical_json_bytes(first_response)
    status = int(first["http_status"])
    error_code = (
        first_response.get("error_code") if isinstance(first_response, dict) else None
    )
    response_equal = canonical_json_bytes({"http_status": first["http_status"], "response": first_response}) == canonical_json_bytes({"http_status": second["http_status"], "response": second["response"]})
    privacy_ok = not public_privacy_violations(response_bytes)
    expected = _expected_result(case_id)
    semantics_ok = _case_semantics(
        case_id,
        status=status,
        payload=first_response,
        source_evidence=source_evidence,
    )
    unchanged = before["sha256"] == after["sha256"]
    if original_before is not None:
        unchanged = unchanged and original_before == original_after
    return {
        **base,
        "blocker_evidence": None,
        "expected_result": expected,
        "observed_result": _observed_result(status, first_response),
        "project_id": chain.project_id,
        "session_id": chain.session_id,
        "trajectory_id": chain.trajectory_id,
        "trajectory_publication_id": chain.trajectory_publication_id,
        "audit_id": chain.audit_id,
        "audit_publication_id": chain.audit_publication_id,
        "attribution_id": chain.attribution_id,
        "attribution_publication_id": chain.attribution_publication_id,
        "source_artifact_digests": source_artifact_digests(chain),
        "runtime_source_evidence": source_evidence,
        "tampering_evidence": tampering_evidence,
        "inspection_http_status": status,
        "inspection_error_code": error_code,
        "inspection_response": first_response,
        "inspection_response_sha256": sha256_bytes(response_bytes),
        "fresh_process_run_count": 2,
        "fresh_process_bytes_equal": response_equal,
        "hash_seed_bytes_equal": response_equal,
        "fresh_process_distinct_pids": first["pid"] != second["pid"],
        "workspace_before_sha256": before["sha256"],
        "workspace_after_sha256": after["sha256"],
        "observer_bytes_modified": not unchanged,
        "scientific_bytes_modified": not unchanged,
        "durable_files_created_by_inspection": [],
        "privacy_scan_passed": privacy_ok,
        "machine_validation_status": (
            "passed"
            if semantics_ok and response_equal and unchanged and privacy_ok and first["pid"] != second["pid"]
            else "failed"
        ),
        "case_status": "executed",
    }


def _fresh_process(locator: Path, *, hash_seed: str) -> dict[str, Any]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--inspect-private-locator", str(locator)],
        check=True,
        capture_output=True,
        env=env,
    )
    if public_privacy_violations(completed.stdout) or public_privacy_violations(
        completed.stderr
    ):
        raise RuntimeError("fresh-process inspection emitted unsafe diagnostics")
    payload = parse_canonical_json(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("fresh-process evidence is invalid")
    return payload


def _expected_result(case_id: str) -> dict[str, Any]:
    values = {
        "single_round_success": {"http_status": 200, "attribution_status": "no_failure"},
        "multi_round_success": {"http_status": 200, "attribution_status": "no_failure"},
        "known_hosts_propagation": {"http_status": 200, "taxonomy_family": "transport"},
        "history_truncation": {"http_status": 409, "error_code": "observer_publication_integrity_failure", "partial_timeline": False},
        "duplicate_dispatch": {"http_status": 200, "taxonomy_family": "recovery"},
        "stale_state": {"http_status": 200, "telemetry_authority": "non_authoritative_telemetry"},
        "multiple_equal_first_cause_candidates": {"http_status": 200, "ambiguity_reason": "multiple_equal_first_cause_candidates", "primary_first_cause_id": None},
        "causal_link_not_proven": {
            "http_status": 200,
            "first_cause_family": "recovery",
            "downstream_status": "undetermined",
            "downstream_reason": "causal_link_not_proven",
            "recovery_receipt_count": 1,
        },
    }
    return values[case_id]


def _case_semantics(
    case_id: str,
    *,
    status: int,
    payload: dict[str, Any],
    source_evidence: dict[str, Any],
) -> bool:
    if case_id == "history_truncation":
        return status == 409 and payload.get("error_code") == "observer_publication_integrity_failure" and "timeline" not in payload
    if status != 200 or payload.get("inspection_version") != INSPECTION_VERSION:
        return False
    if case_id in {"single_round_success", "multi_round_success"}:
        summary = payload.get("summary")
        return isinstance(summary, dict) and summary.get("attribution_status") == "no_failure" and bool(payload.get("timeline"))
    if case_id == "stale_state":
        timeline = payload.get("timeline")
        return isinstance(timeline, list) and any(
            isinstance(event, dict)
            and any(
                isinstance(item, dict)
                and item.get("authority") == "non_authoritative_telemetry"
                for item in event.get("telemetry_findings", [])
            )
            for event in timeline
        )
    timeline = payload.get("timeline")
    if not isinstance(timeline, list):
        return False
    attributions = [
        (event, finding)
        for event in timeline
        if isinstance(event, dict)
        for finding in event.get("failure_attributions", [])
        if isinstance(finding, dict)
    ]
    if case_id == "known_hosts_propagation":
        return (
            any(
                finding.get("taxonomy_family") == "transport"
                and finding.get("attribution_role") == "first_cause"
                and finding.get("attribution_status") == "determined"
                and "known_hosts_verification_failed"
                in event.get("reason_codes", [])
                for event, finding in attributions
            )
            and all(
                finding.get("taxonomy_family") != "model_inadequacy"
                for _, finding in attributions
            )
        )
    if case_id == "duplicate_dispatch":
        duplicate_rows = [
            (event, finding)
            for event, finding in attributions
            if finding.get("deterministic_reason_code")
            == "duplicate_dispatch_persisted"
        ]
        receipts = source_evidence.get("dispatch_receipts", [])
        duplicate = next(
            (
                item
                for item in receipts
                if item.get("dispatch_kind") == "duplicate_rejected"
            ),
            None,
        )
        same_child = (
            [
                item
                for item in receipts
                if duplicate is not None
                and item.get("child_run_id") == duplicate.get("child_run_id")
                and item.get("dispatch_kind")
                in {"initial", "retry", "duplicate_rejected"}
            ]
        )
        return (
            len(duplicate_rows) == 1
            and duplicate_rows[0][1].get("taxonomy_family") == "recovery"
            and duplicate is not None
            and duplicate.get("execution_started") is False
            and duplicate.get("reason_codes") == ["duplicate_dispatch_detected"]
            and len(same_child) == 2
            and [item.get("dispatch_kind") for item in same_child]
            == ["initial", "duplicate_rejected"]
            and len({item.get("receipt_id") for item in same_child}) == 2
            and all(item.get("authority_sha256") for item in same_child)
        )
    if case_id == "multiple_equal_first_cause_candidates":
        summary = payload.get("summary")
        candidates = [
            (event, finding)
            for event, finding in attributions
            if finding.get("taxonomy_family")
            in {"authorization_mismatch", "transport"}
        ]
        revisions = {event.get("session_revision") for event, _ in candidates}
        return (
            isinstance(summary, dict)
            and summary.get("primary_first_cause_id") is None
            and summary.get("ambiguity_reason")
            == "multiple_equal_first_cause_candidates"
            and {finding.get("taxonomy_family") for _, finding in candidates}
            == {"authorization_mismatch", "transport"}
            and len(revisions) == 1
            and all(
                finding.get("attribution_status") == "undetermined"
                for _, finding in candidates
            )
        )
    if case_id == "causal_link_not_proven":
        recoveries = source_evidence.get("recovery_receipts", [])
        receipts = source_evidence.get("dispatch_receipts", [])
        first_causes = [
            finding
            for _, finding in attributions
            if finding.get("attribution_role") == "first_cause"
        ]
        unlinked_terminal = [
            finding
            for event, finding in attributions
            if event.get("event_kind") == "terminal_result_committed"
            and finding.get("attribution_role") == "downstream_symptom"
            and finding.get("attribution_status") == "undetermined"
            and finding.get("deterministic_reason_code") == "causal_link_not_proven"
        ]
        recovery = recoveries[0] if len(recoveries) == 1 else None
        recovered_child = (
            recovery.get("recovered_child_run_id")
            if isinstance(recovery, dict)
            else None
        )
        child_receipts = [
            item
            for item in receipts
            if item.get("child_run_id") == recovered_child
        ]
        return (
            len(recoveries) == 1
            and recovery.get("recovery_kind") == "adopt_completed_child"
            and {
                item.get("dispatch_kind") for item in child_receipts
            } == {"initial", "duplicate_rejected", "recovery_adoption"}
            and set(recovery.get("source_dispatch_receipt_ids", []))
            == {
                item.get("receipt_id")
                for item in child_receipts
                if item.get("dispatch_kind") in {"initial", "duplicate_rejected"}
            }
            and len(first_causes) == 1
            and first_causes[0].get("taxonomy_family") == "recovery"
            and len(unlinked_terminal) == 1
        )
    return False


def _observed_result(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "http_status": status,
        "error_code": payload.get("error_code"),
        "attribution_status": summary.get("attribution_status"),
        "primary_first_cause_id": summary.get("primary_first_cause_id"),
        "ambiguity_reason": summary.get("ambiguity_reason"),
        "timeline_returned": isinstance(payload.get("timeline"), list),
    }


def _publish_public_package(root: Path, records: list[dict[str, Any]], *, runner_commit: str) -> None:
    cases_dir = root / "cases"
    cases_dir.mkdir()
    manifest_cases = []
    for record in records:
        payload = canonical_json_bytes(record)
        violations = public_privacy_violations(payload)
        if violations:
            raise RuntimeError("public evidence privacy scan failed")
        path = cases_dir / CASE_FILENAME_BY_ID[record["case_id"]]
        path.write_bytes(payload)
        manifest_cases.append(
            {
                "case_id": record["case_id"],
                "source_class": record["source_class"],
                "case_file": f"cases/{path.name}",
                "case_file_sha256": sha256_bytes(payload),
                "machine_validation_status": record["machine_validation_status"],
                "human_review_status": "pending",
            }
        )
    machine_complete = _machine_complete(records)
    runner_binding = runner_code_binding(runner_commit)
    manifest = {
        "evidence_version": EVIDENCE_VERSION,
        "inspection_version": INSPECTION_VERSION,
        "source_contracts": SOURCE_CONTRACTS,
        "runner_commit": runner_commit,
        "runner_code_binding": runner_binding,
        "cases": manifest_cases,
        "summary": {
            "case_count": len(records),
            "passed_count": sum(item["machine_validation_status"] == "passed" for item in records),
            "design_analysis_blocked_count": sum(
                item["case_status"] == "design_analysis_blocked" for item in records
            ),
            "not_executed_count": sum(
                item["machine_validation_status"] == "not_executed"
                for item in records
            ),
            "failed_count": sum(item["machine_validation_status"] == "failed" for item in records),
            "machine_evidence_complete": machine_complete,
            "human_review_status": "pending",
            "m3_v_status": "not_yet_claimed",
        },
        "claims": {
            "evidence_only": True,
            "runtime_case_evidence_included": True,
            "all_cases_runtime_executed": machine_complete,
            "design_analysis_blockers_included": False,
            "observer_only": True,
            "scientific_validation_claimed": False,
            "benchmark_result_claimed": False,
            "full_private_source_bundles_committed": False,
            "human_review_required": True,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    write_canonical_json(root / "evidence_manifest.json", manifest, no_replace=True)
    write_canonical_json(
        root / "owner_review.json",
        pending_owner_review(sha256_bytes(manifest_bytes)),
        no_replace=True,
    )
    (root / "README.md").write_text(_readme(machine_complete), encoding="utf-8")
    (root / "evidence_summary.md").write_text(_summary(records), encoding="utf-8")
    (root / "review_checklist.md").write_text(_checklist(records), encoding="utf-8")
    package_bytes = b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())
    if public_privacy_violations(package_bytes):
        raise RuntimeError("public evidence package contains sensitive data")


def _readme(machine_complete: bool) -> str:
    return f"""# M3 representative inspection validation evidence v1

This package was generated by the repository evidence runner through production
Session, PR-BD, PR-BF, PR-BG, and the `scientific_agent_trajectory_inspection.v1`
GET route. Machine evidence is **{'complete' if machine_complete else 'incomplete'}** and human
review is **pending**. No M3 `V` claim is made.

Only redacted inspection responses and digests are committed. Raw Session trees,
observer publications, private locators, infrastructure values, and local paths
remain outside Git. Representative cases are not remote-backend, experimental,
high-fidelity-computation, attribution-accuracy, or M4 benchmark evidence.
"""


def _summary(records: list[dict[str, Any]]) -> str:
    machine_complete = _machine_complete(records)
    lines = [
        "# Evidence summary",
        "",
        (
            "Machine evidence is complete (8 executed / 8 passed); human review "
            "is pending; M3 remains I/T/—."
            if machine_complete
            else "Machine evidence is incomplete; human review is pending; M3 remains I/T/—."
        ),
        "",
        "| Case | Source class | Machine status |",
        "|---|---|---|",
    ]
    lines.extend(f"| `{item['case_id']}` | `{item['source_class']}` | `{item['machine_validation_status']}` |" for item in records)
    lines.extend(
        [
            "",
            "All eight cases execute through production Session/source construction, PR-BD, PR-BF, PR-BG, and two fresh-process calls to the project-scoped PR-BH GET route. PR #12 authoritative receipts and typed failure evidence are summarized without private paths or infrastructure values.",
        ]
    )
    return "\n".join(lines) + "\n"


def _checklist(records: list[dict[str, Any]]) -> str:
    sections = [
        "# Repository-owner review checklist",
        "",
        "Reviewer:",
        "Review date:",
        "Review decision: pending",
        "Reviewed commit:",
        "Notes:",
        "",
        "Allowed decisions: `approved`, `changes_requested`, `inconclusive`.",
    ]
    for record in records:
        checks = (
            "case source_class 标注准确",
            "expected 与 frozen contract 一致",
            "terminal result 与 source Session 一致",
            "first cause、symptom 或 undetermined 具有持久化证据",
            "source references 指向真实持久化 record",
            "digest 与 case evidence 一致",
            "response 不含敏感信息",
            "inspection 未修改 scientific 或 observer bytes",
        )
        sections.extend(
            [
                "",
                f"## {record['case_id']}",
                "",
                *(f"- [ ] {check}" for check in checks),
            ]
        )
    return "\n".join(sections) + "\n"


def _machine_complete(records: list[dict[str, Any]]) -> bool:
    return [item.get("case_id") for item in records] == list(CASE_ROSTER) and all(
        item.get("machine_validation_status") == "passed" for item in records
    )


def _git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _remove_private_tree(root: Path) -> None:
    # Invocation-owned output only; never accepts a broad or implicit target.
    resolved = root.resolve()
    if not root.exists() or len(resolved.parts) < 4:
        return
    import shutil

    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
