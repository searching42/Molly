from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.execution_confirmation import ExecutionConfirmation
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.storage import ProjectStorage


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_resume_after_gate_records_execution_confirmation_separately_from_gate_decision(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-confirm-exec",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)

    first = executor.execute(
        project_id="proj-open-004",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("proj-open-004", "r-confirm-exec")
    assert state is not None
    snapshot = state.details["execution_snapshot"]

    result = executor.resume_after_gate(
        project_id="proj-open-004",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="alice",
        note="run this exact training snapshot",
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    gate_decisions = storage.read_gate_decisions("proj-open-004", "r-confirm-exec")
    assert len(gate_decisions) == 1
    assert gate_decisions[0]["gate"] == GateName.TRAIN_CONFIG.value
    assert gate_decisions[0]["approved_snapshot_id"] == snapshot["snapshot_id"]
    assert gate_decisions[0]["approved_snapshot_hash"] == snapshot["snapshot_hash"]

    confirmations = storage.read_execution_confirmations("proj-open-004", "r-confirm-exec")
    assert len(confirmations) == 1
    confirmation = confirmations[0]
    assert confirmation["confirmation_type"] == "execute_ready_resume"
    assert confirmation["task_id"] == "train_model"
    assert confirmation["adapter"] == snapshot["adapter"]
    assert confirmation["snapshot_id"] == snapshot["snapshot_id"]
    assert confirmation["snapshot_hash"] == snapshot["snapshot_hash"]
    assert confirmation["actor"] == "alice"
    assert confirmation["note"] == "run this exact training snapshot"
    assert confirmation["approved_gates"] == [GateName.TRAIN_CONFIG.value]
    assert confirmation["confirmed_at"]


def test_execution_confirmation_requires_identity_fields() -> None:
    with pytest.raises(ValueError):
        ExecutionConfirmation(
            run_id="r",
            task_id="train_model",
            snapshot_id="",
            snapshot_hash="abc",
            actor="alice",
        )
