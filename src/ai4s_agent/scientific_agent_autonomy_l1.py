"""Deterministic, bounded runtime policy for Autonomy L1.

This module is deliberately narrower than the PR #45 action policy.  PR #45
answers whether one exact Controller action is eligible for user-free
continuation.  This module answers whether the same action may be attempted
under the server-owned, cumulative L1 safety envelope.  It never selects an
action, grants authority, or performs an effect.

All usage values are supplied by callers from immutable Controller and
Execution Agent evidence.  The resulting snapshot is a derived projection;
it is not a ledger and it is not an execution capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ai4s_agent.scientific_agent_autonomy_policy import (
    AUTONOMY_POLICY_DIGEST,
    AUTONOMY_POLICY_VERSION,
)
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    _agent_digest,
)


AUTONOMY_L1_RUNTIME_POLICY_VERSION = "scientific-agent-autonomy-l1-runtime-policy.v1"

# These are server-owned safety bounds.  They are intentionally finite and
# conservative; they are not client-configurable and are not authority grants.
AUTONOMY_L1_MAX_TRANSITIONS = 128
AUTONOMY_L1_MAX_LLM_CALLS = 64
AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS = 86_400
AUTONOMY_L1_PER_INVOCATION_MAX_STEPS = 32

AUTONOMY_L1_REASON_CODES: tuple[str, ...] = (
    "AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED",
    "AUTONOMY_L1_LLM_BUDGET_EXHAUSTED",
    "AUTONOMY_L1_DISPATCH_BUDGET_EXHAUSTED",
    "AUTONOMY_L1_WALL_CLOCK_BUDGET_EXHAUSTED",
    "AUTONOMY_L1_TASK_GRAPH_BOUNDARY",
    "AUTONOMY_L1_RESOURCE_BOUNDARY",
    "AUTONOMY_L1_POLICY_PROHIBITED",
    "AUTONOMY_L1_POLICY_HUMAN_BOUNDARY",
    "AUTONOMY_L1_PROVIDER_UNAVAILABLE",
    "AUTONOMY_L1_LLM_OUTCOME_UNKNOWN",
    "AUTONOMY_L1_INVOCATION_BOUND_EXHAUSTED",
    "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
)

AUTONOMY_L1_RUNTIME_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-l1-runtime-policy-material.v1",
    "runtime_policy_version": AUTONOMY_L1_RUNTIME_POLICY_VERSION,
    "base_action_policy_version": AUTONOMY_POLICY_VERSION,
    "base_action_policy_digest": AUTONOMY_POLICY_DIGEST,
    "limits": {
        "max_controller_transitions": AUTONOMY_L1_MAX_TRANSITIONS,
        "max_execution_agent_llm_calls": AUTONOMY_L1_MAX_LLM_CALLS,
        "max_wall_clock_seconds": AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
        "per_invocation_max_steps": AUTONOMY_L1_PER_INVOCATION_MAX_STEPS,
        "remote_dispatch_limit": "authorized_remote_task_slot_count",
    },
    "reason_codes": list(AUTONOMY_L1_REASON_CODES),
    "budget_scope": "one_controller_execution_id",
    "evidence_sources": {
        "transitions": "verified_controller_action_receipts",
        "llm_calls": "execution_agent_llm_request_started_checkpoints",
        "dispatches": "verified_controller_action_receipts.dispatch_occurred",
        "wall_clock": "controller_execution.created_at_to_server_clock",
        "task_graph": "controller_execution.ordered_task_ids_and_task_roster_digest",
        "resources": "controller_execution_authority_and_budget_digests",
    },
    "rules": {
        "autonomy_does_not_create_authority": True,
        "serialized_policy_decisions_are_non_authoritative": True,
        "recompute_policy_against_current_inspection": True,
        "unknown_llm_outcome_never_auto_retries": True,
        "no_automatic_recovery": True,
        "no_automatic_cancel": True,
        "no_automatic_gate_approval": True,
        "no_automatic_remote_approval": True,
        "read_only_surfaces_may_not_drive_execution": True,
        "events_may_drive_execution": False,
        "missing_or_ambiguous_evidence_fails_closed": True,
        "material_change_owner": "M3.5-AUT-L2",
    },
}

AUTONOMY_L1_RUNTIME_POLICY_DIGEST = _agent_digest(
    AUTONOMY_L1_RUNTIME_POLICY_MATERIAL
)


class AutonomyL1RuntimeError(ValueError):
    """Privacy-safe fail-closed L1 runtime-policy error."""


class AutonomyL1EvidenceError(AutonomyL1RuntimeError):
    """Required immutable budget evidence is unavailable or inconsistent."""


@dataclass(frozen=True)
class AutonomyL1BudgetSnapshot:
    """Derived cumulative budget usage for one exact Controller execution."""

    controller_execution_id: str
    controller_execution_digest: str
    runtime_policy_version: str
    runtime_policy_digest: str
    transitions_used: int
    transition_limit: int
    llm_calls_used: int
    llm_call_limit: int
    remote_dispatches_used: int
    remote_dispatch_limit: int
    wall_clock_elapsed_seconds: float
    wall_clock_limit_seconds: int
    task_count: int
    task_roster_digest: str
    resource_binding_digest: str

    @property
    def wall_clock_exhausted(self) -> bool:
        return self.wall_clock_elapsed_seconds >= self.wall_clock_limit_seconds

    @property
    def task_graph_budget(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "task_roster_digest": self.task_roster_digest,
        }

    @property
    def resource_budget(self) -> dict[str, Any]:
        return {"resource_binding_digest": self.resource_binding_digest}

    def usage_projection(self) -> dict[str, int | float]:
        return {
            "transitions": self.transitions_used,
            "llm_calls": self.llm_calls_used,
            "remote_dispatches": self.remote_dispatches_used,
            "wall_clock_elapsed_seconds": self.wall_clock_elapsed_seconds,
        }

    def limits_projection(self) -> dict[str, int]:
        return {
            "transitions": self.transition_limit,
            "llm_calls": self.llm_call_limit,
            "remote_dispatches": self.remote_dispatch_limit,
            "wall_clock_seconds": self.wall_clock_limit_seconds,
        }


def _parse_timestamp(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AutonomyL1EvidenceError(f"L1 {field} evidence is unavailable")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AutonomyL1EvidenceError(f"L1 {field} evidence is invalid") from exc
    if parsed.tzinfo is None:
        raise AutonomyL1EvidenceError(f"L1 {field} evidence lacks timezone")
    return parsed.astimezone(timezone.utc)


def _nonnegative_count(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutonomyL1EvidenceError(f"L1 {field} evidence is invalid")
    return value


def resource_binding_digest(execution: AgentHarnessControllerExecution) -> str:
    """Return the exact immutable resource/budget identity for an execution."""

    if not isinstance(execution, AgentHarnessControllerExecution):
        raise AutonomyL1EvidenceError("L1 resource binding requires typed execution")
    slots = [
        {
            "planned_task_index": slot.planned_task_index,
            "task_id": slot.task_id,
            "execution_route": slot.execution_route,
            "slot_id": slot.slot_id,
            "task_authority_digest": slot.task_authority_digest,
            "dispatch_intent_digest": slot.dispatch_intent_digest,
            "compiled_options_digest": slot.compiled_options_digest,
            "input_artifacts_digest": slot.input_artifacts_digest,
            "output_contract_digest": slot.output_contract_digest,
            "remote_authority_id": slot.remote_authority_id,
            "remote_authority_digest": slot.remote_authority_digest,
        }
        for slot in execution.task_slots
    ]
    return _agent_digest(
        {
            "schema_version": "agent-autonomy-l1-resource-binding.v1",
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "budget_binding_digest": execution.budget_binding_digest,
            "aggregate_budget_digest": execution.aggregate_budget_digest,
            "remote_authority_set_id": execution.remote_authority_set_id,
            "remote_authority_set_digest": execution.remote_authority_set_digest,
            "remote_authority_roster_digest": execution.remote_authority_roster_digest,
            "ordered_task_ids": list(execution.ordered_task_ids),
            "task_roster_digest": execution.task_roster_digest,
            "task_slots": slots,
        }
    )


def validate_l1_execution_inspection(
    *,
    execution: AgentHarnessControllerExecution,
    inspection: AgentHarnessControllerInspection,
) -> None:
    """Ensure the inspected action is bound to this exact execution/graph."""

    if not isinstance(execution, AgentHarnessControllerExecution) or not isinstance(
        inspection, AgentHarnessControllerInspection
    ):
        raise AutonomyL1EvidenceError("L1 requires typed execution and inspection")
    if (
        inspection.controller_execution_id != execution.controller_execution_id
        or inspection.controller_execution_digest != execution.execution_digest
    ):
        raise AutonomyL1EvidenceError("L1 Controller execution binding is stale")
    if inspection.current_task_index is None:
        if inspection.current_task_id or inspection.current_slot_id:
            raise AutonomyL1EvidenceError("L1 task graph inspection is inconsistent")
        return
    index = inspection.current_task_index
    if index < 0 or index >= len(execution.ordered_task_ids):
        raise AutonomyL1EvidenceError("L1 task graph boundary is outside the roster")
    if index >= len(execution.task_slots):
        raise AutonomyL1EvidenceError("L1 task slot roster is incomplete")
    slot = execution.task_slots[index]
    if (
        inspection.current_task_id != execution.ordered_task_ids[index]
        or inspection.current_task_id != slot.task_id
        or inspection.current_slot_id != slot.slot_id
    ):
        raise AutonomyL1EvidenceError("L1 task graph inspection is not exact")


def build_l1_budget_snapshot(
    *,
    execution: AgentHarnessControllerExecution,
    transition_count: int,
    llm_call_count: int,
    remote_dispatch_count: int,
    now: str,
) -> AutonomyL1BudgetSnapshot:
    """Rebuild cumulative L1 usage from authoritative immutable evidence."""

    if not isinstance(execution, AgentHarnessControllerExecution):
        raise AutonomyL1EvidenceError("L1 budget requires typed execution")
    transition_count = _nonnegative_count(transition_count, field="transition")
    llm_call_count = _nonnegative_count(llm_call_count, field="LLM-call")
    remote_dispatch_count = _nonnegative_count(
        remote_dispatch_count,
        field="dispatch",
    )
    started = _parse_timestamp(execution.created_at, field="execution start")
    current = _parse_timestamp(now, field="server clock")
    elapsed = (current - started).total_seconds()
    if elapsed < 0:
        raise AutonomyL1EvidenceError("L1 server clock precedes execution start")
    resource_digest = resource_binding_digest(execution)
    remote_slots = sum(
        slot.execution_route == "remote_execution_service"
        for slot in execution.task_slots
    )
    if remote_dispatch_count > remote_slots:
        raise AutonomyL1EvidenceError("L1 remote dispatch evidence exceeds authority roster")
    return AutonomyL1BudgetSnapshot(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        runtime_policy_version=AUTONOMY_L1_RUNTIME_POLICY_VERSION,
        runtime_policy_digest=AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
        transitions_used=transition_count,
        transition_limit=AUTONOMY_L1_MAX_TRANSITIONS,
        llm_calls_used=llm_call_count,
        llm_call_limit=AUTONOMY_L1_MAX_LLM_CALLS,
        remote_dispatches_used=remote_dispatch_count,
        remote_dispatch_limit=remote_slots,
        wall_clock_elapsed_seconds=elapsed,
        wall_clock_limit_seconds=AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
        task_count=len(execution.ordered_task_ids),
        task_roster_digest=execution.task_roster_digest,
        resource_binding_digest=resource_digest,
    )


def budget_stop_reason_codes(
    snapshot: AutonomyL1BudgetSnapshot,
    *,
    action: AgentHarnessControllerAction,
    needs_llm: bool,
) -> tuple[str, ...]:
    """Return deterministic reasons that block the next L1 attempt."""

    if not isinstance(snapshot, AutonomyL1BudgetSnapshot):
        raise AutonomyL1EvidenceError("L1 budget snapshot is not typed")
    if not isinstance(action, AgentHarnessControllerAction):
        raise AutonomyL1EvidenceError("L1 action is not typed")
    reasons: list[str] = []
    if snapshot.transitions_used >= snapshot.transition_limit:
        reasons.append("AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED")
    if needs_llm and snapshot.llm_calls_used >= snapshot.llm_call_limit:
        reasons.append("AUTONOMY_L1_LLM_BUDGET_EXHAUSTED")
    if (
        action == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
        and snapshot.remote_dispatches_used >= snapshot.remote_dispatch_limit
    ):
        reasons.append("AUTONOMY_L1_DISPATCH_BUDGET_EXHAUSTED")
    if snapshot.wall_clock_exhausted:
        reasons.append("AUTONOMY_L1_WALL_CLOCK_BUDGET_EXHAUSTED")
    return tuple(reasons)


def budget_projection(snapshot: AutonomyL1BudgetSnapshot) -> dict[str, Any]:
    """Return a privacy-safe, non-authoritative session projection."""

    return {
        "usage": snapshot.usage_projection(),
        "limits": snapshot.limits_projection(),
        "task_graph": snapshot.task_graph_budget,
        "resource": snapshot.resource_budget,
        "controller_execution_id": snapshot.controller_execution_id,
        "controller_execution_digest": snapshot.controller_execution_digest,
        "runtime_policy_version": snapshot.runtime_policy_version,
        "runtime_policy_digest": snapshot.runtime_policy_digest,
    }


__all__ = [
    "AUTONOMY_L1_MAX_LLM_CALLS",
    "AUTONOMY_L1_MAX_TRANSITIONS",
    "AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS",
    "AUTONOMY_L1_PER_INVOCATION_MAX_STEPS",
    "AUTONOMY_L1_REASON_CODES",
    "AUTONOMY_L1_RUNTIME_POLICY_DIGEST",
    "AUTONOMY_L1_RUNTIME_POLICY_MATERIAL",
    "AUTONOMY_L1_RUNTIME_POLICY_VERSION",
    "AutonomyL1BudgetSnapshot",
    "AutonomyL1EvidenceError",
    "AutonomyL1RuntimeError",
    "budget_projection",
    "budget_stop_reason_codes",
    "build_l1_budget_snapshot",
    "resource_binding_digest",
    "validate_l1_execution_inspection",
]
