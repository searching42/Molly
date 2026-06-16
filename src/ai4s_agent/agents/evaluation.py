from __future__ import annotations

from typing import Any


def compute_autonomy_metrics(payload: dict[str, Any]) -> dict[str, int]:
    """Compute lightweight Phase 4 autonomy metrics from structured agent artifacts."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    plan = object_payload(payload.get("plan_proposal"))
    verification = object_payload(payload.get("verification_report"))
    revision = object_payload(payload.get("run_plan_revision"))
    generation = object_payload(payload.get("generation_proposal"))
    review_card = object_payload(payload.get("agent_review_card"))

    run_plan = object_payload(plan.get("run_plan"))
    findings = [item for item in list_payload(verification.get("findings")) if isinstance(item, dict)]
    controls = [item for item in list_payload(review_card.get("approval_controls")) if isinstance(item, dict)]

    user_confirmations_required = len(controls)
    if not controls:
        user_confirmations_required += 1 if plan else 0
        user_confirmations_required += len(string_list(plan.get("required_gates")))
        user_confirmations_required += len(list_payload(plan.get("memory_references")))
        user_confirmations_required += len([q for q in list_payload(plan.get("questions")) if isinstance(q, dict) and q.get("blocks_execution")])
        if revision.get("user_approval_required"):
            user_confirmations_required += 1
        user_confirmations_required += len(string_list(revision.get("approvals_required")))
        user_confirmations_required += len(string_list(generation.get("required_permissions")))

    return {
        "tasks_selected_by_agent": len(string_list(run_plan.get("requested_tasks"))),
        "replans_proposed": 1 if revision.get("revision_id") else 0,
        "user_confirmations_required": user_confirmations_required,
        "verifier_catches": len(findings),
        "failed_autonomous_decisions": count_failed_autonomous_decisions(plan, findings),
    }


def count_failed_autonomous_decisions(plan: dict[str, Any], findings: list[dict[str, Any]]) -> int:
    failures = 1 if str(plan.get("status") or "").lower() == "invalid" else 0
    for finding in findings:
        severity = str(finding.get("severity") or "").lower()
        decision = str(finding.get("decision") or "").lower()
        if severity == "critical" or decision == "abort":
            failures += 1
    return failures


def object_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_payload(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
