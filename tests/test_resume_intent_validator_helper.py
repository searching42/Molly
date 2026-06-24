from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import (
    REPLAN_APPLICATION_RECORD_ARTIFACT_ID,
    REPLAN_RESUME_INTENT_ARTIFACT_ID,
    proposal_artifact_hash,
    write_replan_application_artifacts,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_resume_intent_validation import validate_resume_intent
from ai4s_agent.run_plan_review_artifacts import REPLAN_PROPOSAL_ARTIFACT_ID
from ai4s_agent.run_plan_state_fingerprint import build_resume_state_binding
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


PROJECT_ID = "proj-apply"
RUN_ID = "run-apply"


def _run_plan(*task_ids: str) -> RunPlan:
    tasks = list(task_ids) or ["inspect_dataset", "train_model", "render_report"]
    return RunPlan(
        run_id=RUN_ID,
        requested_tasks=tasks,
        tasks=[PlannedTask(task_id=task_id) for task_id in tasks],
    )


def _stage_state(
    status: RunStatus = RunStatus.WAITING_USER,
    *,
    stage: str = "train_model",
    snapshot_hash: str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> StageState:
    now = now_iso()
    return StageState(
        stage=stage,
        status=status,
        started_at=now,
        ended_at=now,
        updated_at=now,
        details={
            "required_gates": ["gate_replan_rerun_task"],
            "executed_tasks": ["inspect_dataset"],
            "execution_snapshot": {
                "snapshot_id": "snapshot-resume-1",
                "snapshot_hash": snapshot_hash,
            },
        },
    )


def _write_proposal(tmp_path: Path, *, task_id: str = "train_model") -> Path:
    run_dir = ProjectStorage(tmp_path).run_dir(PROJECT_ID, RUN_ID)
    proposal = RunPlanReplanProposal(
        verifier_decision="rerun_recommended",
        proposed_action="rerun_task",
        affected_tasks=[task_id],
        rationale=["Model metrics are weak enough to recommend a rerun."],
        required_user_decisions=["Confirm rerunning train_model."],
        proposed_run_plan_patch={
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [
                {
                    "operation_id": "op_000001",
                    "op": "rerun_task",
                    "task_id": task_id,
                    "source_finding_id": "poor_model_metrics_xxx",
                    "category": "poor_model_metrics",
                    "reason": "Model metrics are weak enough to recommend a rerun.",
                }
            ],
        },
        executable=False,
        source_finding_ids=["poor_model_metrics_xxx"],
    )
    proposal_path = write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))
    ProjectStorage(tmp_path).register_artifact_path(
        PROJECT_ID,
        RUN_ID,
        REPLAN_PROPOSAL_ARTIFACT_ID,
        proposal_path.relative_to(run_dir).as_posix(),
    )
    return proposal_path


def _write_resume_intent_artifacts(tmp_path: Path) -> dict[str, Any]:
    proposal_path = _write_proposal(tmp_path)
    run_plan = _run_plan()
    stage_state = _stage_state()
    request = ReplanApplicationRequest(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash=proposal_artifact_hash(proposal_path),
        selected_action="rerun_task",
        selected_operation_ids=["op_000001"],
        executable=False,
    )
    bundle = write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
        current_run_plan=run_plan,
        stage_state=stage_state,
    )
    return {
        "bundle": bundle,
        "proposal_path": proposal_path,
        "run_dir": ProjectStorage(tmp_path).run_dir(PROJECT_ID, RUN_ID),
        "run_plan": run_plan,
        "stage_state": stage_state,
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_validate_resume_intent_returns_needs_gate_approval_for_valid_artifacts(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is True
    assert result.decision == "needs_gate_approval"
    assert result.error is None
    assert result.executable is False
    assert result.source_application_id == fixture["bundle"].application_record.application_id
    assert result.proposal_hash == fixture["bundle"].application_record.proposal_hash
    assert result.required_gates == ["gate_replan_rerun_task"]
    assert result.rerun_tasks == ["train_model"]
    assert result.affected_tasks == ["train_model"]
    assert result.resume_from_task == "train_model"
    assert result.artifact_refs == {
        REPLAN_APPLICATION_RECORD_ARTIFACT_ID: "review/replan_application_record.json",
        REPLAN_RESUME_INTENT_ARTIFACT_ID: "review/replan_resume_intent.json",
        REPLAN_PROPOSAL_ARTIFACT_ID: "review/replan_proposal.json",
    }
    assert result.resume_state_binding == build_resume_state_binding(_run_plan(), _stage_state())
    assert "proposal_hash_valid" in result.validation_findings
    assert "run_plan_fingerprint_valid" in result.validation_findings
    assert "stage_fingerprint_valid" in result.validation_findings
    assert "application_intent_state_binding_match" in result.validation_findings
    assert "missing_gate_approval:gate_replan_rerun_task" in result.validation_findings


def test_validate_resume_intent_is_read_only(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)
    before = _file_snapshot(tmp_path)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is True
    assert _file_snapshot(tmp_path) == before
    assert not (tmp_path / ".ai4s_internal").exists()


def test_validate_resume_intent_detects_proposal_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    proposal = _json(fixture["proposal_path"])
    proposal["rationale"].append("mutated after application")
    _write_object(fixture["proposal_path"], proposal)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "stale_intent"
    assert result.error == {
        "type": "proposal_hash_mismatch",
        "message": "proposal artifact hash no longer matches the replan application record",
    }
    assert "proposal_hash_mismatch" in result.validation_findings


def test_validate_resume_intent_rejects_unknown_selected_operation_id(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    proposal = _json(fixture["proposal_path"])
    proposal["proposed_run_plan_patch"]["operations"][0]["operation_id"] = "op_999999"
    _write_object(fixture["proposal_path"], proposal)
    application_path = fixture["run_dir"] / "review" / "replan_application_record.json"
    application_record = _json(application_path)
    application_record["proposal_hash"] = proposal_artifact_hash(fixture["proposal_path"])
    _write_object(application_path, application_record)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "invalid_intent"
    assert result.error == {
        "type": "unknown_operation_id",
        "message": "selected_operation_ids are not present in the current proposal artifact: op_000001",
    }


def test_validate_resume_intent_rejects_source_application_id_mismatch(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    intent_path = fixture["run_dir"] / "review" / "replan_resume_intent.json"
    intent = _json(intent_path)
    intent["source_application_id"] = "replan-application-different"
    _write_object(intent_path, intent)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "invalid_intent"
    assert result.error == {
        "type": "source_application_mismatch",
        "message": "resume intent source_application_id does not match replan_application_record.application_id",
    }


def test_validate_resume_intent_marks_missing_rerun_task_as_stale(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan("inspect_dataset", "render_report"),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "stale_intent"
    assert result.error == {
        "type": "run_plan_fingerprint_mismatch",
        "message": "current RunPlan fingerprint does not match resume state binding",
    }


def test_validate_resume_intent_marks_changed_run_plan_stale(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan("train_model", "inspect_dataset", "render_report"),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "stale_intent"
    assert result.error == {
        "type": "run_plan_fingerprint_mismatch",
        "message": "current RunPlan fingerprint does not match resume state binding",
    }


def test_validate_resume_intent_marks_changed_stage_stale(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(
            snapshot_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ),
    )

    assert result.ok is False
    assert result.decision == "stale_intent"
    assert result.error == {
        "type": "stage_fingerprint_mismatch",
        "message": "current StageState fingerprint does not match resume state binding",
    }


def test_validate_resume_intent_rejects_binding_mismatch_between_artifacts(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    intent_path = fixture["run_dir"] / "review" / "replan_resume_intent.json"
    intent = _json(intent_path)
    intent["resume_state_binding"]["stage"] = "render_report"
    _write_object(intent_path, intent)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "invalid_intent"
    assert result.error == {
        "type": "resume_state_binding_mismatch",
        "message": "replan application record and resume intent state bindings do not match",
    }


def test_validate_resume_intent_rejects_historical_unbound_intent(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    application_path = fixture["run_dir"] / "review" / "replan_application_record.json"
    intent_path = fixture["run_dir"] / "review" / "replan_resume_intent.json"
    application = _json(application_path)
    intent = _json(intent_path)
    application.pop("resume_state_binding", None)
    intent.pop("resume_state_binding", None)
    _write_object(application_path, application)
    _write_object(intent_path, intent)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "invalid_intent"
    assert result.error == {
        "type": "resume_state_binding_missing",
        "message": "resume intent artifacts do not include required resume_state_binding",
    }


def test_validate_resume_intent_marks_consumed_intent_audit_as_stale(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    intent_id = fixture["bundle"].result_artifact["intent_id"]

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
        audit_records=[{"event": "resume_intent_consumed", "intent_id": intent_id}],
    )

    assert result.ok is False
    assert result.decision == "stale_intent"
    assert result.error == {
        "type": "resume_intent_already_consumed",
        "message": "resume intent has already been consumed according to audit records",
    }


def test_validate_resume_intent_rejects_unsafe_registry_ref(tmp_path: Path) -> None:
    fixture = _write_resume_intent_artifacts(tmp_path)
    registry_path = fixture["run_dir"] / "artifact_registry.json"
    registry = _json(registry_path)
    registry["artifacts"][REPLAN_RESUME_INTENT_ARTIFACT_ID] = "../outside.json"
    _write_object(registry_path, registry)

    result = validate_resume_intent(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        current_run_plan=_run_plan(),
        stage_state=_stage_state(),
    )

    assert result.ok is False
    assert result.decision == "invalid_intent"
    assert result.error == {
        "type": "unsafe_artifact_ref",
        "message": "artifact ref must stay under run directory",
    }
