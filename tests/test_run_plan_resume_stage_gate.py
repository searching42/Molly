from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.run_plan_resume_stage_gate import (
    build_waiting_stage_gate_context,
    execution_snapshot_material_fingerprint,
)
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState


def _run_plan() -> RunPlan:
    return RunPlan(
        run_id="run-stage-gate",
        requested_tasks=["inspect_dataset", "train_model", "render_report"],
        tasks=[
            PlannedTask(task_id="inspect_dataset"),
            PlannedTask(task_id="train_model"),
            PlannedTask(task_id="render_report"),
        ],
    )


def _snapshot(run_plan: RunPlan, *, task_id: str = "train_model", gates: list[str] | None = None) -> dict[str, Any]:
    required_gates = gates if gates is not None else list(AtomicTaskRegistry().get(task_id).gates)
    material = {
        "schema_version": 1,
        "run_id": run_plan.run_id,
        "task_id": task_id,
        "adapter": "train_model_baseline_adapter",
        "run_plan": run_plan.model_dump(mode="json"),
        "task_options": {},
        "payload": {},
        "input_artifacts": {},
        "approved_gates": sorted(required_gates),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "snapshot_id": f"{run_plan.run_id}:{task_id}:{digest[:16]}",
        "snapshot_hash": digest,
        **material,
    }


def _stage_state(
    run_plan: RunPlan,
    *,
    status: RunStatus = RunStatus.WAITING_USER,
    stage: str = "train_model",
    snapshot: dict[str, Any] | None = None,
) -> StageState:
    now = now_iso()
    spec_gates = list(AtomicTaskRegistry().get(stage).gates)
    return StageState(
        stage=stage,
        status=status,
        started_at=now,
        ended_at=now,
        updated_at=now,
        details={
            "required_gates": spec_gates,
            "executed_tasks": ["inspect_dataset"],
            "execution_snapshot": snapshot or _snapshot(run_plan, task_id=stage),
        },
    )


def test_waiting_stage_gate_context_accepts_executor_bare_snapshot_hash() -> None:
    run_plan = _run_plan()
    stage_state = _stage_state(run_plan)

    context = build_waiting_stage_gate_context(
        run_plan=run_plan,
        stage_state=stage_state,
        application_required_gates=["gate_replan_rerun_task"],
    )

    raw_hash = stage_state.details["execution_snapshot"]["snapshot_hash"]
    assert raw_hash and not raw_hash.startswith("sha256:")
    assert context.execution_snapshot_hash == f"sha256:{raw_hash}"
    assert context.stage == "train_model"
    assert context.status == "WAITING_USER"
    assert context.application_required_gates == ["gate_replan_rerun_task"]
    assert context.execution_required_gates == ["gate_3_train_config"]
    assert context.snapshot_task_id == "train_model"
    assert context.snapshot_run_id == "run-stage-gate"
    assert context.executable is False


def test_execution_snapshot_material_fingerprint_ignores_identity_fields() -> None:
    run_plan = _run_plan()
    snapshot = _snapshot(run_plan)
    prefixed = dict(snapshot)
    prefixed["snapshot_hash"] = f"sha256:{snapshot['snapshot_hash']}"

    assert execution_snapshot_material_fingerprint(snapshot) == execution_snapshot_material_fingerprint(prefixed)


def test_waiting_stage_gate_context_rejects_snapshot_material_mismatch() -> None:
    run_plan = _run_plan()
    snapshot = _snapshot(run_plan)
    snapshot["task_options"] = {"learning_rate": 0.1}
    stage_state = _stage_state(run_plan, snapshot=snapshot)

    with pytest.raises(ValueError, match="execution_snapshot_material_mismatch"):
        build_waiting_stage_gate_context(
            run_plan=run_plan,
            stage_state=stage_state,
            application_required_gates=[],
        )


def test_waiting_stage_gate_context_rejects_non_waiting_stage() -> None:
    run_plan = _run_plan()

    with pytest.raises(ValueError, match="stage_not_waiting_user"):
        build_waiting_stage_gate_context(
            run_plan=run_plan,
            stage_state=_stage_state(run_plan, status=RunStatus.SUCCEEDED),
            application_required_gates=[],
        )


def test_waiting_stage_gate_context_rejects_required_gate_mismatch() -> None:
    run_plan = _run_plan()
    stage_state = _stage_state(run_plan)
    stage_state.details["required_gates"] = ["gate_final_threshold"]

    with pytest.raises(ValueError, match="stage_required_gates_mismatch"):
        build_waiting_stage_gate_context(
            run_plan=run_plan,
            stage_state=stage_state,
            application_required_gates=[],
        )


def test_waiting_stage_gate_context_rejects_duplicate_stage_gates() -> None:
    run_plan = _run_plan()
    stage_state = _stage_state(run_plan)
    stage_state.details["required_gates"] = ["gate_3_train_config", "gate_3_train_config"]

    with pytest.raises(ValueError, match="stage_required_gates_invalid"):
        build_waiting_stage_gate_context(
            run_plan=run_plan,
            stage_state=stage_state,
            application_required_gates=[],
        )


def test_waiting_stage_gate_context_rejects_non_string_snapshot_gates() -> None:
    run_plan = _run_plan()
    snapshot = _snapshot(run_plan)
    snapshot["approved_gates"] = ["gate_3_train_config", 7]
    stage_state = _stage_state(run_plan, snapshot=snapshot)

    with pytest.raises(ValueError, match="execution_snapshot_gates_invalid"):
        build_waiting_stage_gate_context(
            run_plan=run_plan,
            stage_state=stage_state,
            application_required_gates=[],
        )
