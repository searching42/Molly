from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_resume_intent_validation import (
    DEFAULT_RESUME_INTENT_ARTIFACT_REFS,
    ResumeIntentValidationResult,
)
from ai4s_agent.run_plan_resume_intent_validation_audit_memory import (
    RESUME_INTENT_VALIDATION_AUDIT_REF,
    ResumeIntentValidationAuditRecord,
    ResumeIntentValidationMemorySave,
    append_resume_intent_validation_audit_record,
    build_resume_intent_validation_memory_record,
    save_resume_intent_validation_summary_to_memory,
)
from ai4s_agent.storage import ProjectStorage


PROJECT_ID = "proj-resume"
RUN_ID = "run-resume"


def _validation_result(*, decision: str = "needs_gate_approval", ok: bool = True) -> ResumeIntentValidationResult:
    return ResumeIntentValidationResult(
        ok=ok,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        intent_id="resume-replan-application-run-resume-abc123",
        source_application_id="replan-application-run-resume-abc123",
        proposal_hash="sha256:" + "a" * 64,
        decision=decision,  # type: ignore[arg-type]
        required_gates=["gate_replan_rerun_task"],
        approved_gates=[],
        rerun_tasks=["train_model"],
        affected_tasks=["train_model"],
        resume_from_task="train_model",
        artifact_refs=dict(DEFAULT_RESUME_INTENT_ARTIFACT_REFS),
        validation_findings=["missing_gate_approval:gate_replan_rerun_task"],
        error=None
        if ok
        else {
            "type": "proposal_hash_mismatch",
            "message": "proposal artifact hash no longer matches the replan application record",
        },
        executable=False,
    )


def _audit_records(tmp_path: Path) -> list[dict]:
    audit_path = ProjectStorage(tmp_path).run_dir(PROJECT_ID, RUN_ID) / RESUME_INTENT_VALIDATION_AUDIT_REF
    return [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]


def test_append_resume_intent_validation_audit_record_writes_compact_completed_event(tmp_path: Path) -> None:
    result = _validation_result()

    written = append_resume_intent_validation_audit_record(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        event="resume_intent_validation_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        result=result,
    )

    assert written.audit_ref == RESUME_INTENT_VALIDATION_AUDIT_REF
    assert isinstance(written.record, ResumeIntentValidationAuditRecord)
    assert written.record.executable is False
    records = _audit_records(tmp_path)
    assert records == [
        {
            "event": "resume_intent_validation_completed",
            "timestamp": records[0]["timestamp"],
            "project_id": PROJECT_ID,
            "run_id": RUN_ID,
            "actor": "review-user",
            "actor_source": "header:X-Actor",
            "validation_decision": "needs_gate_approval",
            "intent_id": "resume-replan-application-run-resume-abc123",
            "source_application_id": "replan-application-run-resume-abc123",
            "proposal_hash": "sha256:" + "a" * 64,
            "required_gates": ["gate_replan_rerun_task"],
            "approved_gates": [],
            "rerun_tasks": ["train_model"],
            "resume_from_task": "train_model",
            "artifact_refs": DEFAULT_RESUME_INTENT_ARTIFACT_REFS,
            "error": None,
            "executable": False,
        }
    ]
    serialized = json.dumps(records)
    assert "proposed_run_plan_patch" not in serialized
    assert "selected_operations" not in serialized
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_append_resume_intent_validation_audit_record_writes_failed_error(tmp_path: Path) -> None:
    result = _validation_result(decision="stale_intent", ok=False)

    written = append_resume_intent_validation_audit_record(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        event="resume_intent_validation_failed",
        actor="review-user",
        actor_source="header:X-Actor",
        result=result,
    )

    assert written.record.validation_decision == "stale_intent"
    assert written.record.error == {
        "type": "proposal_hash_mismatch",
        "message": "proposal artifact hash no longer matches the replan application record",
    }


def test_build_resume_intent_validation_memory_record_uses_summary_and_refs_only(tmp_path: Path) -> None:
    result = _validation_result()
    audit = append_resume_intent_validation_audit_record(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        event="resume_intent_validation_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        result=result,
    )

    record = build_resume_intent_validation_memory_record(
        result,
        audit_refs=[audit.audit_ref],
        confirmed_by="review-user",
    )

    assert record.category == "run_plan_resume_intent_validation"
    assert record.decision == "resume_intent_validation_recorded"
    assert record.confirmed_by == "review-user"
    assert record.value == {
        "kind": "resume_intent_validation_summary",
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
        "validation_decision": "needs_gate_approval",
        "intent_id": "resume-replan-application-run-resume-abc123",
        "source_application_id": "replan-application-run-resume-abc123",
        "proposal_hash": "sha256:" + "a" * 64,
        "required_gates": ["gate_replan_rerun_task"],
        "approved_gates": [],
        "rerun_tasks": ["train_model"],
        "resume_from_task": "train_model",
        "artifact_refs": DEFAULT_RESUME_INTENT_ARTIFACT_REFS,
        "audit_refs": [RESUME_INTENT_VALIDATION_AUDIT_REF],
        "error": None,
        "executable": False,
    }
    assert record.source_refs == [
        "run:run-resume:artifact:replan_application_record",
        "run:run-resume:artifact:replan_resume_intent",
        "run:run-resume:artifact:replan_proposal",
        "run:run-resume:audit:review/resume_intent_validation_audit.jsonl",
    ]
    serialized = json.dumps(record.model_dump(mode="json"))
    assert "validation_findings" not in serialized
    assert "proposed_run_plan_patch" not in serialized
    assert "selected_operations" not in serialized


def test_save_resume_intent_validation_summary_to_memory_is_idempotent(tmp_path: Path) -> None:
    result = _validation_result()
    audit = append_resume_intent_validation_audit_record(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        event="resume_intent_validation_completed",
        actor="review-user",
        actor_source="header:X-Actor",
        result=result,
    )

    first = save_resume_intent_validation_summary_to_memory(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        result=result,
        audit_refs=[audit.audit_ref],
        confirmed_by="review-user",
    )
    second = save_resume_intent_validation_summary_to_memory(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        result=result,
        audit_refs=[audit.audit_ref],
        confirmed_by="reviewer-b",
    )

    assert isinstance(first, ResumeIntentValidationMemorySave)
    assert first.saved is True
    assert first.executable is False
    assert first.record.record_id == second.record.record_id
    records = ProjectMemory(tmp_path).list_project_records(PROJECT_ID)
    assert len(records) == 1
    assert records[0].confirmed_by == "reviewer-b"
    assert records[0].value["artifact_refs"]["replan_resume_intent"] == "review/replan_resume_intent.json"


def test_save_resume_intent_validation_summary_rejects_mismatched_result(tmp_path: Path) -> None:
    payload = _validation_result().model_dump(mode="json")
    payload["run_id"] = "other-run"

    try:
        save_resume_intent_validation_summary_to_memory(
            workspace_dir=tmp_path,
            project_id=PROJECT_ID,
            run_id=RUN_ID,
            result=payload,
            audit_refs=[],
            confirmed_by="review-user",
        )
    except ValueError as exc:
        assert "resume intent validation result project_id/run_id mismatch" in str(exc)
    else:
        raise AssertionError("mismatched resume intent validation result should fail")
