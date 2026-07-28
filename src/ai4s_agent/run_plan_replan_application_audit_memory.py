from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_replan_application_artifacts import RunPlanApplicationArtifactBundle
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.storage import ProjectStorage


REPLAN_APPLICATION_AUDIT_REF = "review/replan_application_audit.jsonl"
ReplanApplicationAuditEvent = Literal[
    "replan_application_requested",
    "replan_application_completed",
    "replan_application_failed",
]


class ReplanApplicationAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: ReplanApplicationAuditEvent
    timestamp: str = Field(default_factory=now_iso)
    project_id: str
    run_id: str
    actor: str
    actor_source: str
    application_id: str = ""
    proposal_hash: str = ""
    selected_action: str = ""
    result_type: str = ""
    result_artifact_id: str = ""
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    executable: bool = False

    @field_validator("project_id", "run_id", "actor", "actor_source")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("audit record text fields are required")
        return clean

    @field_validator("application_id", "proposal_hash", "selected_action", "result_type", "result_artifact_id")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            str(key or "").strip(): str(raw or "").strip()
            for key, raw in value.items()
            if str(key or "").strip() and str(raw or "").strip()
        }

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("replan application audit records are not executable")
        return False


class ReplanApplicationAuditWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    audit_ref: str
    record: ReplanApplicationAuditRecord
    executable: bool = False

    @field_validator("project_id", "run_id", "audit_ref")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("audit write text fields are required")
        return clean

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("replan application audit writes are not executable")
        return False


class ReplanApplicationMemorySave(BaseModel):
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
    def validate_non_executable(self) -> ReplanApplicationMemorySave:
        if self.executable is not False:
            raise ValueError("replan application memory save is not executable")
        return self


def append_replan_application_audit_record(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    event: ReplanApplicationAuditEvent,
    actor: str,
    actor_source: str,
    bundle: RunPlanApplicationArtifactBundle | dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> ReplanApplicationAuditWrite:
    """Append a compact replan application audit record.

    This helper only writes audit JSONL. It does not enqueue work, execute
    adapters, mutate a RunPlan, apply patches, call LLMs, or replace
    `/api/run-plan/execute`.
    """

    application_bundle = _bundle_or_none(bundle)
    if application_bundle is not None:
        _validate_bundle_identity(application_bundle, project_id=project_id, run_id=run_id)
    record = ReplanApplicationAuditRecord(
        event=event,
        project_id=project_id,
        run_id=run_id,
        actor=actor,
        actor_source=actor_source,
        application_id=application_bundle.application_record.application_id if application_bundle else "",
        proposal_hash=application_bundle.application_record.proposal_hash if application_bundle else "",
        selected_action=application_bundle.application_record.selected_action if application_bundle else "",
        result_type=application_bundle.application_record.result_type if application_bundle else "",
        result_artifact_id=application_bundle.result_artifact_id if application_bundle else "",
        artifact_refs=application_bundle.artifacts if application_bundle else {},
        error=_clean_error(error),
        executable=False,
    )
    audit_path = _audit_path(workspace_dir, project_id=project_id, run_id=run_id)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return ReplanApplicationAuditWrite(
        project_id=project_id,
        run_id=run_id,
        audit_ref=REPLAN_APPLICATION_AUDIT_REF,
        record=record,
        executable=False,
    )


def save_replan_application_summary_to_memory(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    bundle: RunPlanApplicationArtifactBundle | dict[str, Any],
    audit_refs: list[str] | None = None,
    confirmed_by: str = "",
) -> ReplanApplicationMemorySave:
    """Save a compact replan application summary to project memory.

    This stores only summary fields plus artifact and audit references. It does
    not store raw artifact contents, full proposal payloads, selected operation
    payloads, execute proposals, apply patches, call LLMs, enqueue work, or
    mutate a RunPlan.
    """

    application_bundle = _bundle_or_none(bundle)
    if application_bundle is None:
        raise ValueError("replan application bundle is required")
    _validate_bundle_identity(application_bundle, project_id=project_id, run_id=run_id)
    record = build_replan_application_memory_record(
        application_bundle,
        audit_refs=audit_refs or [],
        confirmed_by=confirmed_by,
    )
    saved = ProjectMemory(Path(workspace_dir)).save_project_record(project_id, record)
    return ReplanApplicationMemorySave(
        project_id=project_id,
        run_id=run_id,
        saved=True,
        record=saved,
        executable=False,
    )


def build_replan_application_memory_record(
    bundle: RunPlanApplicationArtifactBundle | dict[str, Any],
    *,
    audit_refs: list[str] | None = None,
    confirmed_by: str = "",
) -> ProjectMemoryRecord:
    application_bundle = (
        bundle if isinstance(bundle, RunPlanApplicationArtifactBundle) else RunPlanApplicationArtifactBundle.model_validate(bundle)
    )
    clean_audit_refs = _clean_string_list(audit_refs or [])
    value = {
        "kind": "replan_application_summary",
        "project_id": application_bundle.project_id,
        "run_id": application_bundle.run_id,
        "application_id": application_bundle.application_record.application_id,
        "proposal_hash": application_bundle.application_record.proposal_hash,
        "selected_action": application_bundle.application_record.selected_action,
        "result_type": application_bundle.application_record.result_type,
        "selected_operation_ids": list(application_bundle.application_record.selected_operation_ids),
        "affected_tasks": _affected_tasks(application_bundle),
        "required_gates": list(application_bundle.compiled.required_gates),
        "artifact_refs": dict(application_bundle.artifacts),
        "audit_refs": clean_audit_refs,
        "executable": False,
    }
    source_refs = [
        f"run:{application_bundle.run_id}:artifact:{artifact_id}"
        for artifact_id in application_bundle.artifact_ids
    ]
    source_refs.extend(f"run:{application_bundle.run_id}:audit:{audit_ref}" for audit_ref in clean_audit_refs)
    return ProjectMemoryRecord(
        record_id=f"replan-application-{application_bundle.application_record.application_id}",
        category="run_plan_replan_application",
        summary=_summary(application_bundle),
        value=value,
        source_refs=source_refs,
        decision="replan_application_recorded",
        confirmed_by=str(confirmed_by or "").strip(),
        metadata={
            "source": "replan_application_artifacts",
            "content_policy": "summary_and_artifact_refs_only",
        },
    )


def _audit_path(workspace_dir: str | Path, *, project_id: str, run_id: str) -> Path:
    run_dir = ProjectStorage(Path(workspace_dir)).run_dir(project_id, run_id)
    path = (run_dir / REPLAN_APPLICATION_AUDIT_REF).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("replan application audit path escapes run directory")
    return path


def _bundle_or_none(
    bundle: RunPlanApplicationArtifactBundle | dict[str, Any] | None,
) -> RunPlanApplicationArtifactBundle | None:
    if bundle is None:
        return None
    return bundle if isinstance(bundle, RunPlanApplicationArtifactBundle) else RunPlanApplicationArtifactBundle.model_validate(bundle)


def _validate_bundle_identity(
    bundle: RunPlanApplicationArtifactBundle,
    *,
    project_id: str,
    run_id: str,
) -> None:
    if bundle.project_id != project_id or bundle.run_id != run_id:
        raise ValueError("replan application bundle project_id/run_id mismatch")


def _affected_tasks(bundle: RunPlanApplicationArtifactBundle) -> list[str]:
    return _clean_string_list([operation.task_id for operation in bundle.compiled.selected_operations])


def _summary(bundle: RunPlanApplicationArtifactBundle) -> str:
    return (
        f"Run {bundle.run_id} replan application: "
        f"action={bundle.application_record.selected_action}; "
        f"result={bundle.application_record.result_type}; "
        f"operations={len(bundle.application_record.selected_operation_ids)}."
    )


def _clean_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "type": str(error.get("type") or "error"),
        "message": str(error.get("message") or ""),
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
