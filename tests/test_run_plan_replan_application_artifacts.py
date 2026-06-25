from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent._utils import write_json
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactVerification
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import (
    BLOCKED_ACKNOWLEDGEMENT_ARTIFACT_ID,
    REPLAN_APPLICATION_RECORD_ARTIFACT_ID,
    REPLAN_RESUME_INTENT_ARTIFACT_ID,
    RUN_PLAN_REVISION_ARTIFACT_ID,
    RunPlanApplicationArtifactBundle,
    proposal_artifact_hash,
    write_replan_application_artifacts,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_review_artifacts import write_run_plan_review_artifacts
from ai4s_agent.run_plan_state_fingerprint import build_resume_state_binding
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


def _proposal_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run_plan(*task_ids: str, project_run_id: str = "run-apply") -> RunPlan:
    tasks = list(task_ids) or ["inspect_dataset", "train_model", "render_report"]
    return RunPlan(
        run_id=project_run_id,
        requested_tasks=tasks,
        tasks=[PlannedTask(task_id=task_id) for task_id in tasks],
        available_artifacts=["uploaded_dataset"],
    )


def _execution_snapshot(run_plan: RunPlan, *, task_id: str = "train_model") -> dict[str, Any]:
    gates = sorted(AtomicTaskRegistry().get(task_id).gates)
    material = {
        "schema_version": 1,
        "run_id": run_plan.run_id,
        "task_id": task_id,
        "adapter": "train_model_baseline_adapter",
        "run_plan": run_plan.model_dump(mode="json"),
        "task_options": {},
        "payload": {},
        "input_artifacts": {},
        "approved_gates": gates,
    }
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "snapshot_id": f"{run_plan.run_id}:{task_id}:{digest[:16]}",
        "snapshot_hash": digest,
        **material,
    }


def _stage_state(run_plan: RunPlan | None = None) -> StageState:
    plan = run_plan or _run_plan()
    now = now_iso()
    return StageState(
        stage="train_model",
        status=RunStatus.WAITING_USER,
        started_at=now,
        ended_at=now,
        updated_at=now,
        details={
            "required_gates": list(AtomicTaskRegistry().get("train_model").gates),
            "executed_tasks": ["inspect_dataset"],
            "execution_snapshot": _execution_snapshot(plan),
        },
    )


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _request(
    *,
    project_id: str = "proj-apply",
    run_id: str = "run-apply",
    action: str = "rerun_task",
    operation_id: str = "op_000001",
    proposal_hash: str,
    proposal_artifact_ref: str = "review/replan_proposal.json",
) -> ReplanApplicationRequest:
    return ReplanApplicationRequest(
        project_id=project_id,
        run_id=run_id,
        proposal_artifact_ref=proposal_artifact_ref,
        proposal_hash=proposal_hash,
        selected_action=action,  # type: ignore[arg-type]
        selected_operation_ids=[operation_id],
    )


def _write_rerun_review_artifacts(tmp_path: Path, *, project_id: str = "proj-apply", run_id: str = "run-apply") -> Path:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir(project_id, run_id)
    metrics_path = write_json(
        run_dir / "metrics" / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.5, "mae": 0.42}}]},
    )
    storage.register_artifact_path(project_id, run_id, "model_metrics", metrics_path.relative_to(run_dir).as_posix())
    write_run_plan_review_artifacts(workspace_dir=tmp_path, project_id=project_id, run_id=run_id)
    return run_dir / "review" / "replan_proposal.json"


def _write_manual_proposal(
    tmp_path: Path,
    *,
    project_id: str = "proj-apply",
    run_id: str = "run-apply",
    action: str,
    task_id: str,
) -> Path:
    run_dir = ProjectStorage(tmp_path).run_dir(project_id, run_id)
    proposal = RunPlanReplanProposal(
        verifier_decision="needs_review",
        proposed_action=action,  # type: ignore[arg-type]
        affected_tasks=[task_id],
        rationale=[f"{action} requested."],
        required_user_decisions=[f"Confirm {action}."],
        proposed_run_plan_patch={
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [
                {
                    "operation_id": "op_000001",
                    "op": action,
                    "task_id": task_id,
                    "source_finding_id": "finding_1",
                    "category": action,
                    "reason": f"{action} requested.",
                }
            ],
        },
        executable=False,
        source_finding_ids=["finding_1"],
    )
    return write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))


def test_write_replan_application_artifacts_materializes_resume_intent_from_review_proposal(tmp_path: Path) -> None:
    proposal_path = _write_rerun_review_artifacts(tmp_path)
    request = _request(proposal_hash=_proposal_hash(proposal_path))

    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert isinstance(bundle, RunPlanApplicationArtifactBundle)
    assert bundle.executable is False
    assert bundle.compiled.result_type == "resume_intent"
    assert bundle.artifact_ids == [REPLAN_APPLICATION_RECORD_ARTIFACT_ID, REPLAN_RESUME_INTENT_ARTIFACT_ID]
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    application_record = json.loads((run_dir / "review" / "replan_application_record.json").read_text(encoding="utf-8"))
    resume_intent = json.loads((run_dir / "review" / "replan_resume_intent.json").read_text(encoding="utf-8"))
    assert application_record["result_type"] == "resume_intent"
    assert application_record["selected_operation_ids"] == ["op_000001"]
    assert application_record["executable"] is False
    assert application_record["resume_state_binding"] == resume_intent["resume_state_binding"]
    assert resume_intent["action"] == "rerun_task"
    assert resume_intent["rerun_tasks"] == ["train_model"]
    assert resume_intent["executable"] is False
    registry = ProjectStorage(tmp_path).read_artifact_registry("proj-apply", "run-apply")
    assert registry[REPLAN_APPLICATION_RECORD_ARTIFACT_ID] == "review/replan_application_record.json"
    assert registry[REPLAN_RESUME_INTENT_ARTIFACT_ID] == "review/replan_resume_intent.json"
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_write_replan_application_artifacts_binds_resume_intent_to_current_state(tmp_path: Path) -> None:
    proposal_path = _write_rerun_review_artifacts(tmp_path)
    request = _request(proposal_hash=_proposal_hash(proposal_path))
    run_plan = _run_plan()
    stage_state = _stage_state()

    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
        current_run_plan=run_plan,
        stage_state=stage_state,
    )

    expected_binding = build_resume_state_binding(run_plan, stage_state).model_dump(mode="json")
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    application_record = json.loads((run_dir / "review" / "replan_application_record.json").read_text(encoding="utf-8"))
    resume_intent = json.loads((run_dir / "review" / "replan_resume_intent.json").read_text(encoding="utf-8"))
    assert application_record["resume_state_binding"] == expected_binding
    assert resume_intent["resume_state_binding"] == expected_binding
    assert bundle.application_record.resume_state_binding is not None
    assert bundle.application_record.resume_state_binding.model_dump(mode="json") == expected_binding
    assert bundle.result_artifact["resume_state_binding"] == expected_binding


def test_write_replan_application_artifacts_accepts_real_executor_waiting_snapshot(tmp_path: Path) -> None:
    project_id = "proj-exec-apply"
    run_id = "run-exec-apply"
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
    execution = RunPlanExecutor(storage=storage).execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert execution["status"] == RunStatus.WAITING_USER.value
    stage_state = storage.read_stage_state(project_id, run_id)
    assert stage_state is not None
    raw_snapshot_hash = stage_state.details["execution_snapshot"]["snapshot_hash"]
    assert len(raw_snapshot_hash) == 64
    assert not raw_snapshot_hash.startswith("sha256:")
    proposal_path = _write_manual_proposal(tmp_path, project_id=project_id, run_id=run_id, action="rerun_task", task_id="train_model")
    request = _request(project_id=project_id, run_id=run_id, proposal_hash=_proposal_hash(proposal_path))

    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
        current_run_plan=run_plan,
        stage_state=stage_state,
    )

    binding = bundle.application_record.resume_state_binding
    assert binding is not None
    assert binding.execution_snapshot_hash == "sha256:" + raw_snapshot_hash
    assert bundle.result_artifact["resume_state_binding"]["execution_snapshot_hash"] == "sha256:" + raw_snapshot_hash


def test_write_replan_application_artifacts_requires_state_for_resume_intent(tmp_path: Path) -> None:
    proposal_path = _write_rerun_review_artifacts(tmp_path)
    request = _request(proposal_hash=_proposal_hash(proposal_path))

    with pytest.raises(ValueError, match="current_run_plan and stage_state are required"):
        write_replan_application_artifacts(
            workspace_dir=tmp_path,
            request=request,
            actor="review-user",
            actor_source="header:X-Actor",
            current_run_plan=_run_plan(),
        )

    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    assert not (run_dir / "review" / "replan_application_record.json").exists()
    assert REPLAN_APPLICATION_RECORD_ARTIFACT_ID not in ProjectStorage(tmp_path).read_artifact_registry(
        "proj-apply", "run-apply"
    )


def test_write_replan_application_artifacts_rejects_rerun_task_that_is_not_waiting_stage(tmp_path: Path) -> None:
    proposal_path = _write_manual_proposal(tmp_path, action="rerun_task", task_id="render_report")
    request = _request(proposal_hash=_proposal_hash(proposal_path))

    with pytest.raises(ValueError, match="rerun_task_stage_mismatch"):
        write_replan_application_artifacts(
            workspace_dir=tmp_path,
            request=request,
            actor="review-user",
            actor_source="header:X-Actor",
            current_run_plan=_run_plan(),
            stage_state=_stage_state(_run_plan()),
        )

    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    assert not (run_dir / "review" / "replan_application_record.json").exists()
    assert not (run_dir / "review" / "replan_resume_intent.json").exists()
    registry = ProjectStorage(tmp_path).read_artifact_registry("proj-apply", "run-apply")
    assert REPLAN_APPLICATION_RECORD_ARTIFACT_ID not in registry
    assert REPLAN_RESUME_INTENT_ARTIFACT_ID not in registry


def test_write_replan_application_artifacts_materializes_run_plan_revision_draft(tmp_path: Path) -> None:
    proposal_path = _write_manual_proposal(tmp_path, action="adjust_targets", task_id="plan_targets")
    request = _request(action="adjust_targets", proposal_hash=_proposal_hash(proposal_path))

    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
    )

    assert bundle.compiled.result_type == "run_plan_revision"
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    revision = json.loads((run_dir / "review" / "run_plan_revision.json").read_text(encoding="utf-8"))
    assert revision["kind"] == "run_plan_revision_draft"
    assert revision["selected_action"] == "adjust_targets"
    assert revision["executable"] is False
    assert ProjectStorage(tmp_path).read_artifact_registry("proj-apply", "run-apply")[RUN_PLAN_REVISION_ARTIFACT_ID] == (
        "review/run_plan_revision.json"
    )


def test_write_replan_application_artifacts_materializes_blocked_acknowledgement(tmp_path: Path) -> None:
    proposal_path = _write_manual_proposal(tmp_path, action="block", task_id="artifact_registry")
    request = _request(action="block", proposal_hash=_proposal_hash(proposal_path))

    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
    )

    assert bundle.compiled.result_type == "blocked_acknowledgement"
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    acknowledgement = json.loads((run_dir / "review" / "blocked_acknowledgement.json").read_text(encoding="utf-8"))
    assert acknowledgement["blocked_reason"]
    assert acknowledgement["executable"] is False
    registry = ProjectStorage(tmp_path).read_artifact_registry("proj-apply", "run-apply")
    assert registry[BLOCKED_ACKNOWLEDGEMENT_ARTIFACT_ID] == "review/blocked_acknowledgement.json"


def test_write_replan_application_artifacts_rejects_hash_mismatch_without_writing(tmp_path: Path) -> None:
    _write_rerun_review_artifacts(tmp_path)
    request = _request(proposal_hash="sha256:not-the-proposal")

    with pytest.raises(ValueError, match="proposal_hash mismatch"):
        write_replan_application_artifacts(
            workspace_dir=tmp_path,
            request=request,
            actor="review-user",
            actor_source="header:X-Actor",
        )

    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    assert not (run_dir / "review" / "replan_application_record.json").exists()
    assert REPLAN_APPLICATION_RECORD_ARTIFACT_ID not in ProjectStorage(tmp_path).read_artifact_registry(
        "proj-apply", "run-apply"
    )


def test_write_replan_application_artifacts_rejects_historical_proposal_without_operation_ids(
    tmp_path: Path,
) -> None:
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    proposal = RunPlanReplanProposal(
        verifier_decision="rerun_recommended",
        proposed_action="rerun_task",
        affected_tasks=["train_model"],
        rationale=["rerun"],
        required_user_decisions=["approve rerun"],
        proposed_run_plan_patch={
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [
                {
                    "op": "rerun_task",
                    "task_id": "train_model",
                    "source_finding_id": "finding_1",
                    "category": "poor_model_metrics",
                    "reason": "rerun",
                }
            ],
        },
        executable=False,
    )
    proposal_path = write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))
    request = _request(proposal_hash=_proposal_hash(proposal_path))

    with pytest.raises(ValueError, match="operation_id"):
        write_replan_application_artifacts(
            workspace_dir=tmp_path,
            request=request,
            actor="review-user",
            actor_source="header:X-Actor",
        )

    assert not (run_dir / "review" / "replan_application_record.json").exists()


def test_write_replan_application_artifacts_rejects_proposal_path_escape(tmp_path: Path) -> None:
    request = _request(proposal_hash="sha256:abc", proposal_artifact_ref="../replan_proposal.json")

    with pytest.raises(ValueError, match="proposal artifact path escapes run directory"):
        write_replan_application_artifacts(
            workspace_dir=tmp_path,
            request=request,
            actor="review-user",
            actor_source="header:X-Actor",
        )


def test_proposal_artifact_hash_uses_sha256_prefix(tmp_path: Path) -> None:
    proposal_path = _write_rerun_review_artifacts(tmp_path)

    assert proposal_artifact_hash(proposal_path) == _proposal_hash(proposal_path)
