from __future__ import annotations

from dataclasses import replace

import pytest

from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_MAX_LLM_CALLS,
    AUTONOMY_L1_MAX_TRANSITIONS,
    AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
    AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
    AUTONOMY_L1_RUNTIME_POLICY_MATERIAL,
    AUTONOMY_L1_RUNTIME_POLICY_VERSION,
    AutonomyL1EvidenceError,
    budget_stop_reason_codes,
    build_l1_budget_snapshot,
    resource_binding_digest,
    validate_l1_execution_inspection,
)
from ai4s_agent.schemas import AgentHarnessControllerAction
from tests.execution_agent_test_support import NOW, local_controller_execution


def test_l1_runtime_policy_is_versioned_finite_and_binds_pr45() -> None:
    assert AUTONOMY_L1_RUNTIME_POLICY_VERSION == (
        "scientific-agent-autonomy-l1-runtime-policy.v1"
    )
    assert AUTONOMY_L1_RUNTIME_POLICY_MATERIAL["base_action_policy_version"] == (
        "scientific-agent-autonomy-policy.v1"
    )
    assert AUTONOMY_L1_RUNTIME_POLICY_MATERIAL["rules"][
        "events_may_drive_execution"
    ] is False
    assert AUTONOMY_L1_MAX_TRANSITIONS > 0
    assert AUTONOMY_L1_MAX_LLM_CALLS > 0
    assert AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS > 0
    assert AUTONOMY_L1_RUNTIME_POLICY_DIGEST.startswith("sha256:")


def test_budget_rebuild_uses_execution_scope_and_exact_graph_resource_identity(
    tmp_path,
) -> None:
    _storage, control_store, _controller, result = local_controller_execution(tmp_path)
    execution = result.execution
    snapshot = build_l1_budget_snapshot(
        execution=execution,
        transition_count=len(
            control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
        ),
        llm_call_count=0,
        remote_dispatch_count=0,
        now=NOW,
    )

    assert snapshot.controller_execution_id == execution.controller_execution_id
    assert snapshot.controller_execution_digest == execution.execution_digest
    assert snapshot.task_count == len(execution.ordered_task_ids)
    assert snapshot.task_roster_digest == execution.task_roster_digest
    assert snapshot.resource_binding_digest == resource_binding_digest(execution)
    assert snapshot.remote_dispatch_limit == 0
    assert budget_stop_reason_codes(
        snapshot,
        action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        needs_llm=True,
    ) == ()


def test_budget_exhaustion_is_checked_before_the_next_attempt(tmp_path) -> None:
    _storage, _control_store, _controller, result = local_controller_execution(tmp_path)
    execution = result.execution
    base = build_l1_budget_snapshot(
        execution=execution,
        transition_count=0,
        llm_call_count=0,
        remote_dispatch_count=0,
        now=NOW,
    )
    exhausted = replace(
        base,
        transitions_used=base.transition_limit,
        llm_calls_used=base.llm_call_limit,
        wall_clock_elapsed_seconds=float(base.wall_clock_limit_seconds),
    )

    reasons = budget_stop_reason_codes(
        exhausted,
        action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        needs_llm=True,
    )
    assert reasons == (
        "AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED",
        "AUTONOMY_L1_LLM_BUDGET_EXHAUSTED",
        "AUTONOMY_L1_WALL_CLOCK_BUDGET_EXHAUSTED",
    )


def test_task_graph_and_execution_digest_are_exactly_bound(tmp_path) -> None:
    _storage, _control_store, _controller, result = local_controller_execution(tmp_path)
    validate_l1_execution_inspection(
        execution=result.execution,
        inspection=result.inspection,
    )
    changed = result.inspection.model_dump(mode="json")
    changed["controller_execution_digest"] = "sha256:" + "f" * 64
    changed["inspection_digest"] = ""
    # Rebuilding this object is intentionally not enough to make it current;
    # the exact execution digest still rejects the inspection.
    from ai4s_agent.schemas import AgentHarnessControllerInspection

    stale = AgentHarnessControllerInspection(**changed)
    with pytest.raises(AutonomyL1EvidenceError):
        validate_l1_execution_inspection(
            execution=result.execution,
            inspection=stale,
        )


def test_invalid_clock_or_dispatch_evidence_fails_closed(tmp_path) -> None:
    _storage, _control_store, _controller, result = local_controller_execution(tmp_path)
    with pytest.raises(AutonomyL1EvidenceError):
        build_l1_budget_snapshot(
            execution=result.execution,
            transition_count=0,
            llm_call_count=0,
            remote_dispatch_count=0,
            now="2025-01-01T00:00:00Z",
        )
    with pytest.raises(AutonomyL1EvidenceError):
        build_l1_budget_snapshot(
            execution=result.execution,
            transition_count=0,
            llm_call_count=0,
            remote_dispatch_count=1,
            now=NOW,
        )
