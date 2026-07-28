from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.run_plan_artifact_verifier import (
    RunPlanArtifactDecision,
    RunPlanArtifactFinding,
    RunPlanArtifactVerification,
)


RunPlanReplanAction = Literal[
    "continue",
    "request_review",
    "rerun_task",
    "adjust_targets",
    "collect_more_data",
    "block",
]

PATCH_SCHEMA_VERSION = "reviewable_run_plan_patch.v1"


class RunPlanReplanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_source: Literal["verifier"] = "verifier"
    verifier_decision: RunPlanArtifactDecision
    proposed_action: RunPlanReplanAction
    affected_tasks: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    required_user_decisions: list[str] = Field(default_factory=list)
    proposed_run_plan_patch: dict[str, Any] = Field(default_factory=dict)
    executable: bool = False
    generated_at: str = Field(default_factory=now_iso)
    source_finding_ids: list[str] = Field(default_factory=list)

    @field_validator("affected_tasks", "rationale", "required_user_decisions", "source_finding_ids")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("proposed_run_plan_patch")
    @classmethod
    def validate_patch_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _assert_json_safe(value, "proposed_run_plan_patch")
        return value

    @model_validator(mode="after")
    def validate_non_executable(self) -> RunPlanReplanProposal:
        if self.executable is not False:
            raise ValueError("executable must remain false for reviewable replan proposals")
        if self.proposed_run_plan_patch and self.proposed_run_plan_patch.get("applied") is not False:
            raise ValueError("proposed_run_plan_patch must be marked applied=false")
        return self


def propose_replan_from_verification(
    verification: RunPlanArtifactVerification | dict[str, Any],
) -> RunPlanReplanProposal:
    """Map a read-only artifact verification result to a reviewable proposal.

    The mapper is intentionally deterministic. It does not call an LLM, mutate a
    `RunPlan`, execute adapters, enqueue work, or auto-rerun tasks.
    """

    verified = (
        verification
        if isinstance(verification, RunPlanArtifactVerification)
        else RunPlanArtifactVerification.model_validate(deepcopy(verification))
    )
    findings = list(verified.findings)
    action = _proposed_action(verified.decision, findings)
    affected_tasks = _affected_tasks(action, findings)
    operations = _operations_for(action, affected_tasks, findings)
    return RunPlanReplanProposal(
        verifier_decision=verified.decision,
        proposed_action=action,
        affected_tasks=affected_tasks,
        rationale=_rationale(verified, findings),
        required_user_decisions=_required_user_decisions(action, findings, affected_tasks),
        proposed_run_plan_patch={
            "schema_version": PATCH_SCHEMA_VERSION,
            "applied": False,
            "operations": operations,
        },
        executable=False,
        source_finding_ids=[finding.finding_id for finding in findings],
    )


def _proposed_action(
    verifier_decision: RunPlanArtifactDecision,
    findings: list[RunPlanArtifactFinding],
) -> RunPlanReplanAction:
    categories = {finding.category for finding in findings}
    if verifier_decision == "blocked":
        return "block"
    if categories & _TARGET_ADJUSTMENT_CATEGORIES:
        return "adjust_targets"
    if categories & _DATA_COLLECTION_CATEGORIES:
        return "collect_more_data"
    if verifier_decision == "rerun_recommended" or categories & _RERUN_CATEGORIES:
        return "rerun_task"
    if verifier_decision == "needs_review" or categories & _REVIEW_CATEGORIES:
        return "request_review"
    return "continue"


def _affected_tasks(action: RunPlanReplanAction, findings: list[RunPlanArtifactFinding]) -> list[str]:
    if action == "continue":
        return []
    tasks: list[str] = []
    for finding in findings:
        tasks.extend(_tasks_for_finding(finding))
    if tasks:
        return _clean_string_list(tasks)
    fallback = {
        "request_review": ["review_run_plan"],
        "rerun_task": ["run_plan_execute"],
        "adjust_targets": ["plan_targets"],
        "collect_more_data": ["collect_data"],
        "block": ["run_plan_execute"],
    }
    return list(fallback[action])


def _tasks_for_finding(finding: RunPlanArtifactFinding) -> list[str]:
    category = finding.category
    if category == "waiting_user":
        waiting_task = str(finding.evidence.get("waiting_task") or "").strip()
        if waiting_task:
            return [waiting_task]
        terminal_record = finding.evidence.get("terminal_record")
        if isinstance(terminal_record, dict):
            waiting_task = str(terminal_record.get("waiting_task") or "").strip()
            if waiting_task:
                return [waiting_task]
        return ["review_required_gate"]
    return list(_CATEGORY_TASKS.get(category, ["run_plan_execute"]))


def _operations_for(
    action: RunPlanReplanAction,
    affected_tasks: list[str],
    findings: list[RunPlanArtifactFinding],
) -> list[dict[str, Any]]:
    if action == "continue":
        return []
    if not findings:
        return _with_operation_ids(
            [
                {
                    "op": action,
                    "task_id": task_id,
                    "source_finding_id": "",
                    "category": "",
                    "reason": f"Verifier proposed action `{action}`.",
                }
                for task_id in affected_tasks
            ]
        )
    operations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        for task_id in _tasks_for_finding(finding):
            key = (action, task_id, finding.finding_id)
            if key in seen:
                continue
            seen.add(key)
            operations.append(
                {
                    "op": action,
                    "task_id": task_id,
                    "source_finding_id": finding.finding_id,
                    "category": finding.category,
                    "reason": finding.message,
                }
            )
    return _with_operation_ids(operations)


def _with_operation_ids(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operation_id": f"op_{index:06d}",
            **operation,
        }
        for index, operation in enumerate(operations, start=1)
    ]


def _rationale(
    verification: RunPlanArtifactVerification,
    findings: list[RunPlanArtifactFinding],
) -> list[str]:
    if not findings:
        return [verification.summary]
    return _clean_string_list([verification.summary, *[finding.message for finding in findings]])


def _required_user_decisions(
    action: RunPlanReplanAction,
    findings: list[RunPlanArtifactFinding],
    affected_tasks: list[str],
) -> list[str]:
    if action == "continue":
        return []
    task_text = ", ".join(affected_tasks) if affected_tasks else "affected tasks"
    decisions: list[str] = []
    if action == "rerun_task":
        decisions.append(f"Approve rerun of {task_text} before any queued execution.")
    elif action == "request_review":
        gates = _required_gates(findings)
        if gates:
            decisions.append(f"Review and approve required gate(s): {', '.join(gates)}.")
        decisions.append(f"Review verifier finding(s) for {task_text} before resume or rerun.")
    elif action == "collect_more_data":
        decisions.append(
            f"Confirm additional data collection or preprocessing for {task_text} before changing the run plan."
        )
    elif action == "adjust_targets":
        decisions.append(f"Confirm target/property changes for {task_text} before any revised run plan is created.")
    elif action == "block":
        decisions.append(f"Resolve blocked verifier finding(s) for {task_text} before any resume or rerun.")
    return _clean_string_list(decisions)


def _required_gates(findings: list[RunPlanArtifactFinding]) -> list[str]:
    gates: list[str] = []
    for finding in findings:
        raw = finding.evidence.get("required_gates")
        if isinstance(raw, list):
            gates.extend(str(item).strip() for item in raw)
        terminal_record = finding.evidence.get("terminal_record")
        if isinstance(terminal_record, dict) and isinstance(terminal_record.get("required_gates"), list):
            gates.extend(str(item).strip() for item in terminal_record["required_gates"])
    return _clean_string_list(gates)


def _clean_string_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _assert_json_safe(value: Any, label: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


_RERUN_CATEGORIES: frozenset[str] = frozenset(
    {
        "poor_model_metrics",
        "queue_not_terminal",
    }
)

_REVIEW_CATEGORIES: frozenset[str] = frozenset(
    {
        "waiting_user",
        "missing_audit",
        "high_extraction_conflict_rate",
        "empty_ranking",
        "missing_weighted_score",
    }
)

_DATA_COLLECTION_CATEGORIES: frozenset[str] = frozenset(
    {
        "empty_generation",
        "missing_trainability_properties",
        "no_trainable_labels_gained",
    }
)

_TARGET_ADJUSTMENT_CATEGORIES: frozenset[str] = frozenset(
    {
        "target_mismatch",
        "target_property_mismatch",
        "objective_mismatch",
    }
)

_CATEGORY_TASKS: dict[str, list[str]] = {
    "audit_terminal_failure": ["run_plan_execute"],
    "empty_generation": ["generate_candidates"],
    "empty_ranking": ["filter_rank"],
    "high_extraction_conflict_rate": ["merge_extracted_records", "confirm_dataset"],
    "missing_artifact": ["artifact_registry"],
    "missing_audit": ["audit"],
    "missing_trainability_properties": ["inspect_dataset", "check_trainability"],
    "missing_weighted_score": ["filter_rank"],
    "no_trainable_labels_gained": ["extract_records", "confirm_dataset"],
    "poor_model_metrics": ["train_model"],
    "poor_trainability": ["check_trainability"],
    "queue_failed": ["run_plan_execute"],
    "queue_not_terminal": ["run_plan_execute"],
    "target_mismatch": ["plan_targets"],
    "target_property_mismatch": ["plan_targets"],
    "objective_mismatch": ["rank_candidates"],
}
