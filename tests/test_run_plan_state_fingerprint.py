from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.run_plan_state_fingerprint import (
    ResumeStateBinding,
    build_resume_state_binding,
    run_plan_fingerprint,
    stage_state_fingerprint,
)
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState


RUN_ID = "run-fingerprint"


def _run_plan(*task_ids: str) -> RunPlan:
    tasks = list(task_ids) or ["inspect_dataset", "train_model", "render_report"]
    return RunPlan(
        run_id=RUN_ID,
        requested_tasks=tasks,
        tasks=[PlannedTask(task_id=task_id) for task_id in tasks],
        available_artifacts=["uploaded_dataset"],
        missing_artifacts=["candidate_dataset"],
    )


def _stage_state(
    *,
    stage: str = "train_model",
    status: RunStatus = RunStatus.WAITING_USER,
    next_stage: str | None = "render_report",
    started_at: str = "2026-01-01T00:00:00Z",
    updated_at: str = "2026-01-01T00:00:01Z",
    ended_at: str | None = "2026-01-01T00:00:02Z",
    required_gates: list[str] | None = None,
    executed_tasks: list[str] | None = None,
    snapshot_hash: str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> StageState:
    return StageState(
        stage=stage,
        next_stage=next_stage,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        ended_at=ended_at,
        details={
            "required_gates": required_gates if required_gates is not None else ["gate_replan_rerun_task"],
            "executed_tasks": executed_tasks if executed_tasks is not None else ["inspect_dataset"],
            "execution_snapshot": {
                "snapshot_id": "snapshot-1",
                "snapshot_hash": snapshot_hash,
            },
        },
        history=[],
    )


def test_run_plan_fingerprint_is_canonical() -> None:
    plan = _run_plan()
    shuffled_payload = {
        "missing_artifacts": ["candidate_dataset"],
        "tasks": [
            {"output_artifacts": [], "task_id": "inspect_dataset", "depends_on": [], "required_artifacts": []},
            {"task_id": "train_model", "depends_on": [], "required_artifacts": [], "output_artifacts": []},
            {"depends_on": [], "required_artifacts": [], "output_artifacts": [], "task_id": "render_report"},
        ],
        "available_artifacts": ["uploaded_dataset"],
        "requested_tasks": ["inspect_dataset", "train_model", "render_report"],
        "run_id": RUN_ID,
    }

    fingerprint = run_plan_fingerprint(plan)

    assert fingerprint == run_plan_fingerprint(shuffled_payload)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)


def test_run_plan_fingerprint_changes_with_semantic_plan_changes() -> None:
    base = _run_plan()

    reordered = _run_plan("train_model", "inspect_dataset", "render_report")
    renamed = _run_plan("inspect_dataset", "train_model_v2", "render_report")
    dependency = RunPlan(
        run_id=RUN_ID,
        requested_tasks=["inspect_dataset", "train_model", "render_report"],
        tasks=[
            PlannedTask(task_id="inspect_dataset"),
            PlannedTask(task_id="train_model", depends_on=["inspect_dataset"]),
            PlannedTask(task_id="render_report"),
        ],
        available_artifacts=["uploaded_dataset"],
        missing_artifacts=["candidate_dataset"],
    )
    available_changed = RunPlan(
        run_id=RUN_ID,
        requested_tasks=["inspect_dataset", "train_model", "render_report"],
        tasks=base.tasks,
        available_artifacts=["clean_dataset"],
        missing_artifacts=["candidate_dataset"],
    )
    requested_changed = RunPlan(
        run_id=RUN_ID,
        requested_tasks=["train_model", "render_report"],
        tasks=base.tasks,
        available_artifacts=["uploaded_dataset"],
        missing_artifacts=["candidate_dataset"],
    )

    base_fingerprint = run_plan_fingerprint(base)

    assert run_plan_fingerprint(reordered) != base_fingerprint
    assert run_plan_fingerprint(renamed) != base_fingerprint
    assert run_plan_fingerprint(dependency) != base_fingerprint
    assert run_plan_fingerprint(available_changed) != base_fingerprint
    assert run_plan_fingerprint(requested_changed) != base_fingerprint


def test_stage_fingerprint_ignores_volatile_timestamps() -> None:
    first = _stage_state(started_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:01Z")
    second = _stage_state(started_at=now_iso(), updated_at=now_iso(), ended_at=now_iso())

    assert stage_state_fingerprint(first) == stage_state_fingerprint(second)


def test_stage_fingerprint_changes_with_semantic_state() -> None:
    base = _stage_state()
    base_fingerprint = stage_state_fingerprint(base)

    assert stage_state_fingerprint(_stage_state(stage="render_report")) != base_fingerprint
    assert stage_state_fingerprint(_stage_state(status=RunStatus.RUNNING)) != base_fingerprint
    assert stage_state_fingerprint(_stage_state(next_stage="validate_report")) != base_fingerprint
    assert stage_state_fingerprint(_stage_state(required_gates=["gate_other"])) != base_fingerprint
    assert stage_state_fingerprint(_stage_state(executed_tasks=["inspect_dataset", "clean_dataset"])) != base_fingerprint
    assert stage_state_fingerprint(
        _stage_state(snapshot_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    ) != base_fingerprint


def test_resume_state_binding_rejects_partial_snapshot_identity() -> None:
    with pytest.raises(ValidationError, match="execution_snapshot"):
        ResumeStateBinding(
            run_plan_fingerprint=run_plan_fingerprint(_run_plan()),
            stage_fingerprint=stage_state_fingerprint(_stage_state()),
            stage="train_model",
            stage_status="WAITING_USER",
            execution_snapshot_id="snapshot-1",
        )

    with pytest.raises(ValidationError, match="execution_snapshot"):
        ResumeStateBinding(
            run_plan_fingerprint=run_plan_fingerprint(_run_plan()),
            stage_fingerprint=stage_state_fingerprint(_stage_state()),
            stage="train_model",
            stage_status="WAITING_USER",
            execution_snapshot_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_resume_state_binding_accepts_executor_snapshot_hash_format() -> None:
    binding = ResumeStateBinding(
        run_plan_fingerprint=run_plan_fingerprint(_run_plan()),
        stage_fingerprint=stage_state_fingerprint(_stage_state()),
        stage="train_model",
        stage_status="WAITING_USER",
        execution_snapshot_id="snapshot-1",
        execution_snapshot_hash="a" * 64,
    )

    assert binding.execution_snapshot_hash == "sha256:" + "a" * 64


def test_stage_fingerprint_normalizes_snapshot_hash_representation() -> None:
    bare_hash_stage = _stage_state(snapshot_hash="a" * 64)
    prefixed_hash_stage = _stage_state(snapshot_hash="sha256:" + "a" * 64)

    assert stage_state_fingerprint(bare_hash_stage) == stage_state_fingerprint(prefixed_hash_stage)


def test_build_resume_state_binding_captures_current_state() -> None:
    run_plan = _run_plan()
    stage = _stage_state()

    binding = build_resume_state_binding(run_plan, stage)

    assert binding.schema_version == "resume_state_binding.v1"
    assert binding.run_plan_fingerprint == run_plan_fingerprint(run_plan)
    assert binding.stage_fingerprint == stage_state_fingerprint(stage)
    assert binding.stage == "train_model"
    assert binding.stage_status == "WAITING_USER"
    assert binding.execution_snapshot_id == "snapshot-1"
    assert binding.execution_snapshot_hash == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
