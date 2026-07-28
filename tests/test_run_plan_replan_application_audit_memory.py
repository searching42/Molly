from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import (
    RunPlanApplicationArtifactBundle,
    write_replan_application_artifacts,
)
from ai4s_agent.run_plan_replan_application_audit_memory import (
    REPLAN_APPLICATION_AUDIT_REF,
    ReplanApplicationAuditRecord,
    ReplanApplicationMemorySave,
    append_replan_application_audit_record,
    build_replan_application_memory_record,
    save_replan_application_summary_to_memory,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


def _proposal_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run_plan() -> RunPlan:
    return RunPlan(
        run_id="run-apply",
        requested_tasks=["inspect_dataset", "train_model", "render_report"],
        tasks=[
            PlannedTask(task_id="inspect_dataset"),
            PlannedTask(task_id="train_model"),
            PlannedTask(task_id="render_report"),
        ],
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


def _write_application_bundle(tmp_path: Path) -> RunPlanApplicationArtifactBundle:
    project_id = "proj-apply"
    run_id = "run-apply"
    run_dir = ProjectStorage(tmp_path).run_dir(project_id, run_id)
    proposal = RunPlanReplanProposal(
        verifier_decision="rerun_recommended",
        proposed_action="rerun_task",
        affected_tasks=["train_model"],
        rationale=["Model metrics are weak enough to recommend a rerun."],
        required_user_decisions=["Approve rerun before any queued execution."],
        proposed_run_plan_patch={
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [
                {
                    "operation_id": "op_000001",
                    "op": "rerun_task",
                    "task_id": "train_model",
                    "source_finding_id": "finding_1",
                    "category": "poor_model_metrics",
                    "reason": "Model metrics are weak enough to recommend a rerun.",
                }
            ],
        },
        executable=False,
        source_finding_ids=["finding_1"],
    )
    proposal_path = write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))
    request = ReplanApplicationRequest(
        project_id=project_id,
        run_id=run_id,
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash=_proposal_hash(proposal_path),
        selected_action="rerun_task",
        selected_operation_ids=["op_000001"],
    )
    return write_replan_application_artifacts(
        workspace_dir=tmp_path,
        request=request,
        actor="review-user",
        actor_source="header:X-Actor",
        current_run_plan=_run_plan(),
        stage_state=_stage_state(_run_plan()),
    )


def test_append_replan_application_audit_record_writes_requested_and_completed_events(tmp_path: Path) -> None:
    bundle = _write_application_bundle(tmp_path)

    requested = append_replan_application_audit_record(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        event="replan_application_requested",
        actor="review-user",
        actor_source="header:X-Actor",
        bundle=bundle,
    )
    completed = append_replan_application_audit_record(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        event="replan_application_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        bundle=bundle,
    )

    assert requested.audit_ref == REPLAN_APPLICATION_AUDIT_REF
    assert completed.audit_ref == REPLAN_APPLICATION_AUDIT_REF
    assert requested.record.executable is False
    audit_path = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply") / REPLAN_APPLICATION_AUDIT_REF
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "replan_application_requested",
        "replan_application_completed",
    ]
    assert records[0]["actor"] == "review-user"
    assert records[0]["actor_source"] == "header:X-Actor"
    assert records[0]["application_id"] == bundle.application_record.application_id
    assert records[0]["result_type"] == "resume_intent"
    assert records[0]["artifact_refs"]["replan_application_record"] == "review/replan_application_record.json"
    assert records[0]["error"] is None
    serialized = json.dumps(records)
    assert "proposed_run_plan_patch" not in serialized
    assert "selected_operations" not in serialized
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_append_replan_application_audit_record_writes_failed_event_with_error(tmp_path: Path) -> None:
    bundle = _write_application_bundle(tmp_path)

    written = append_replan_application_audit_record(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        event="replan_application_failed",
        actor="review-user",
        actor_source="header:X-Actor",
        bundle=bundle,
        error={"type": "validation_error", "message": "selected operation no longer exists"},
    )

    assert isinstance(written.record, ReplanApplicationAuditRecord)
    assert written.record.event == "replan_application_failed"
    assert written.record.error == {"type": "validation_error", "message": "selected operation no longer exists"}
    assert written.record.executable is False


def test_build_replan_application_memory_record_uses_summary_and_refs_only(tmp_path: Path) -> None:
    bundle = _write_application_bundle(tmp_path)
    audit = append_replan_application_audit_record(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        event="replan_application_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        bundle=bundle,
    )

    record = build_replan_application_memory_record(
        bundle,
        audit_refs=[audit.audit_ref],
        confirmed_by="review-user",
    )

    assert record.category == "run_plan_replan_application"
    assert record.decision == "replan_application_recorded"
    assert record.confirmed_by == "review-user"
    assert record.value == {
        "kind": "replan_application_summary",
        "project_id": "proj-apply",
        "run_id": "run-apply",
        "application_id": bundle.application_record.application_id,
        "proposal_hash": bundle.application_record.proposal_hash,
        "selected_action": "rerun_task",
        "result_type": "resume_intent",
        "selected_operation_ids": ["op_000001"],
        "affected_tasks": ["train_model"],
        "required_gates": ["gate_replan_rerun_task"],
        "artifact_refs": bundle.artifacts,
        "audit_refs": [REPLAN_APPLICATION_AUDIT_REF],
        "executable": False,
    }
    assert record.source_refs == [
        "run:run-apply:artifact:replan_application_record",
        "run:run-apply:artifact:replan_resume_intent",
        "run:run-apply:audit:review/replan_application_audit.jsonl",
    ]
    serialized = json.dumps(record.model_dump(mode="json"))
    assert "proposed_run_plan_patch" not in serialized
    assert "selected_operations" not in serialized
    assert "Model metrics are weak enough" not in serialized


def test_save_replan_application_summary_to_project_memory_is_idempotent(tmp_path: Path) -> None:
    bundle = _write_application_bundle(tmp_path)
    audit = append_replan_application_audit_record(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        event="replan_application_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        bundle=bundle,
    )

    first = save_replan_application_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        bundle=bundle,
        audit_refs=[audit.audit_ref],
        confirmed_by="review-user",
    )
    second = save_replan_application_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-apply",
        run_id="run-apply",
        bundle=bundle,
        audit_refs=[audit.audit_ref],
        confirmed_by="reviewer-b",
    )

    assert isinstance(first, ReplanApplicationMemorySave)
    assert first.saved is True
    assert first.executable is False
    assert first.record.record_id == second.record.record_id
    records = ProjectMemory(tmp_path).list_project_records("proj-apply")
    assert len(records) == 1
    assert records[0].confirmed_by == "reviewer-b"
    assert records[0].value["artifact_refs"]["replan_resume_intent"] == "review/replan_resume_intent.json"


def test_save_replan_application_summary_rejects_mismatched_bundle(tmp_path: Path) -> None:
    bundle = _write_application_bundle(tmp_path)
    payload = bundle.model_dump(mode="json")
    payload["run_id"] = "other-run"

    try:
        save_replan_application_summary_to_memory(
            workspace_dir=tmp_path,
            project_id="proj-apply",
            run_id="run-apply",
            bundle=payload,
            audit_refs=[],
            confirmed_by="review-user",
        )
    except ValueError as exc:
        assert "replan application bundle project_id/run_id mismatch" in str(exc)
    else:
        raise AssertionError("mismatched application bundle should fail")
