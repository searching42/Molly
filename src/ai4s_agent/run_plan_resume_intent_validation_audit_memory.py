from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_resume_intent_validation import ResumeIntentValidationResult
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.storage import ProjectStorage


RESUME_INTENT_VALIDATION_AUDIT_REF = "review/resume_intent_validation_audit.jsonl"
ResumeIntentValidationAuditEvent = Literal[
    "resume_intent_validation_requested",
    "resume_intent_validation_completed",
    "resume_intent_validation_failed",
]


class ResumeIntentValidationAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: ResumeIntentValidationAuditEvent
    timestamp: str = Field(default_factory=now_iso)
    project_id: str
    run_id: str
    actor: str
    actor_source: str
    validation_decision: str = ""
    intent_id: str = ""
    source_application_id: str = ""
    proposal_hash: str = ""
    required_gates: list[str] = Field(default_factory=list)
    approved_gates: list[str] = Field(default_factory=list)
    rerun_tasks: list[str] = Field(default_factory=list)
    resume_from_task: str = ""
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    executable: bool = False

    @field_validator("project_id", "run_id", "actor", "actor_source")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("resume intent validation audit text fields are required")
        return clean

    @field_validator("validation_decision", "intent_id", "source_application_id", "proposal_hash", "resume_from_task")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("required_gates", "approved_gates", "rerun_tasks")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: dict[str, str]) -> dict[str, str]:
        return _clean_string_dict(value)

    @field_validator("error")
    @classmethod
    def validate_error(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _clean_error(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("resume intent validation audit records are not executable")
        return False


class ResumeIntentValidationAuditWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    audit_ref: str
    record: ResumeIntentValidationAuditRecord
    executable: bool = False

    @field_validator("project_id", "run_id", "audit_ref")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("resume intent validation audit write text fields are required")
        return clean

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("resume intent validation audit writes are not executable")
        return False


class ResumeIntentValidationMemorySave(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    saved: bool
    record: ProjectMemoryRecord
    executable: bool = False

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("project_id and run_id are required")
        return clean

    @model_validator(mode="after")
    def validate_non_executable(self) -> ResumeIntentValidationMemorySave:
        if self.executable is not False:
            raise ValueError("resume intent validation memory save is not executable")
        return self


def append_resume_intent_validation_audit_record(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    event: ResumeIntentValidationAuditEvent,
    actor: str,
    actor_source: str,
    result: ResumeIntentValidationResult | dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> ResumeIntentValidationAuditWrite:
    """Append a compact resume-intent validation audit record.

    This helper only writes audit JSONL. It does not add routes, enqueue work,
    execute adapters, call `RunPlanExecutor.resume_after_gate(...)`, write gate
    decisions, mutate a RunPlan, call LLMs, or replace `/api/run-plan/resume`.
    """

    validation_result = _result_or_none(result)
    if validation_result is not None:
        _validate_result_identity(validation_result, project_id=project_id, run_id=run_id)
    record = ResumeIntentValidationAuditRecord(
        event=event,
        project_id=project_id,
        run_id=run_id,
        actor=actor,
        actor_source=actor_source,
        validation_decision=validation_result.decision if validation_result else "",
        intent_id=validation_result.intent_id if validation_result else "",
        source_application_id=validation_result.source_application_id if validation_result else "",
        proposal_hash=validation_result.proposal_hash if validation_result else "",
        required_gates=validation_result.required_gates if validation_result else [],
        approved_gates=validation_result.approved_gates if validation_result else [],
        rerun_tasks=validation_result.rerun_tasks if validation_result else [],
        resume_from_task=validation_result.resume_from_task if validation_result else "",
        artifact_refs=validation_result.artifact_refs if validation_result else {},
        error=_clean_error(error if error is not None else validation_result.error if validation_result else None),
        executable=False,
    )
    audit_path = _audit_path(workspace_dir, project_id=project_id, run_id=run_id)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return ResumeIntentValidationAuditWrite(
        project_id=project_id,
        run_id=run_id,
        audit_ref=RESUME_INTENT_VALIDATION_AUDIT_REF,
        record=record,
        executable=False,
    )


def save_resume_intent_validation_summary_to_memory(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    result: ResumeIntentValidationResult | dict[str, Any],
    audit_refs: list[str] | None = None,
    confirmed_by: str = "",
) -> ResumeIntentValidationMemorySave:
    """Save a compact resume-intent validation summary to project memory.

    This stores only summary fields plus artifact and audit references. It does
    not store raw artifacts, validation findings, proposals, operation payloads,
    execute adapters, write gate decisions, enqueue work, call LLMs, or mutate a
    RunPlan.
    """

    validation_result = _result_or_none(result)
    if validation_result is None:
        raise ValueError("resume intent validation result is required")
    _validate_result_identity(validation_result, project_id=project_id, run_id=run_id)
    record = build_resume_intent_validation_memory_record(
        validation_result,
        audit_refs=audit_refs or [],
        confirmed_by=confirmed_by,
    )
    saved = ProjectMemory(Path(workspace_dir)).save_project_record(project_id, record)
    return ResumeIntentValidationMemorySave(
        project_id=project_id,
        run_id=run_id,
        saved=True,
        record=saved,
        executable=False,
    )


def build_resume_intent_validation_memory_record(
    result: ResumeIntentValidationResult | dict[str, Any],
    *,
    audit_refs: list[str] | None = None,
    confirmed_by: str = "",
) -> ProjectMemoryRecord:
    validation_result = (
        result if isinstance(result, ResumeIntentValidationResult) else ResumeIntentValidationResult.model_validate(result)
    )
    clean_audit_refs = _clean_string_list(audit_refs or [])
    value = {
        "kind": "resume_intent_validation_summary",
        "project_id": validation_result.project_id,
        "run_id": validation_result.run_id,
        "validation_decision": validation_result.decision,
        "intent_id": validation_result.intent_id,
        "source_application_id": validation_result.source_application_id,
        "proposal_hash": validation_result.proposal_hash,
        "required_gates": list(validation_result.required_gates),
        "approved_gates": list(validation_result.approved_gates),
        "rerun_tasks": list(validation_result.rerun_tasks),
        "resume_from_task": validation_result.resume_from_task,
        "artifact_refs": dict(validation_result.artifact_refs),
        "audit_refs": clean_audit_refs,
        "error": _clean_error(validation_result.error),
        "executable": False,
    }
    source_refs = [
        f"run:{validation_result.run_id}:artifact:{artifact_id}"
        for artifact_id in validation_result.artifact_refs
    ]
    source_refs.extend(f"run:{validation_result.run_id}:audit:{audit_ref}" for audit_ref in clean_audit_refs)
    return ProjectMemoryRecord(
        record_id=f"resume-intent-validation-{validation_result.intent_id}",
        category="run_plan_resume_intent_validation",
        summary=_summary(validation_result),
        value=value,
        source_refs=source_refs,
        decision="resume_intent_validation_recorded",
        confirmed_by=str(confirmed_by or "").strip(),
        metadata={
            "source": "resume_intent_validation",
            "content_policy": "summary_and_artifact_refs_only",
        },
    )


def _audit_path(workspace_dir: str | Path, *, project_id: str, run_id: str) -> Path:
    run_dir = ProjectStorage(Path(workspace_dir)).run_dir(project_id, run_id)
    path = (run_dir / RESUME_INTENT_VALIDATION_AUDIT_REF).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("resume intent validation audit path escapes run directory")
    return path


def _result_or_none(result: ResumeIntentValidationResult | dict[str, Any] | None) -> ResumeIntentValidationResult | None:
    if result is None:
        return None
    return result if isinstance(result, ResumeIntentValidationResult) else ResumeIntentValidationResult.model_validate(result)


def _validate_result_identity(
    result: ResumeIntentValidationResult,
    *,
    project_id: str,
    run_id: str,
) -> None:
    if result.project_id != project_id or result.run_id != run_id:
        raise ValueError("resume intent validation result project_id/run_id mismatch")


def _summary(result: ResumeIntentValidationResult) -> str:
    return (
        f"Run {result.run_id} resume intent validation: "
        f"decision={result.decision}; "
        f"rerun_tasks={len(result.rerun_tasks)}; "
        f"required_gates={len(result.required_gates)}."
    )


def _clean_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "type": str(error.get("type") or "error"),
        "message": str(error.get("message") or ""),
    }


def _clean_string_dict(value: dict[str, str]) -> dict[str, str]:
    return {
        str(key or "").strip(): str(raw or "").strip()
        for key, raw in value.items()
        if str(key or "").strip() and str(raw or "").strip()
    }


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
