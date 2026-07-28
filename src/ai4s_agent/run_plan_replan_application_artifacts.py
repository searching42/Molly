from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.run_plan_replan_application import (
    BlockedAcknowledgement,
    CompiledReplanApplication,
    ReplanApplicationRecord,
    ReplanApplicationRequest,
    ReplanPatchOperation,
    ResumeIntent,
    ReviewableRunPlanPatch,
    validate_and_compile_replan_application,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_resume_stage_gate import WaitingStageGateContext, build_waiting_stage_gate_context
from ai4s_agent.run_plan_state_fingerprint import ResumeStateBinding, build_resume_state_binding
from ai4s_agent.schemas import RunPlan, StageState
from ai4s_agent.storage import ProjectStorage


REPLAN_APPLICATION_RECORD_ARTIFACT_ID = "replan_application_record"
REPLAN_RESUME_INTENT_ARTIFACT_ID = "replan_resume_intent"
RUN_PLAN_REVISION_ARTIFACT_ID = "run_plan_revision"
BLOCKED_ACKNOWLEDGEMENT_ARTIFACT_ID = "blocked_acknowledgement"


class RunPlanRevisionDraftArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "run_plan_revision_draft"
    application_id: str
    project_id: str
    run_id: str
    proposal_hash: str
    selected_action: str
    selected_operation_ids: list[str] = Field(default_factory=list)
    selected_operations: list[ReplanPatchOperation] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    validation_findings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    created_by: str
    actor_source: str
    executable: bool = False

    @field_validator("application_id", "project_id", "run_id", "proposal_hash", "created_by", "actor_source")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("revision draft text fields are required")
        return clean

    @field_validator("selected_operation_ids", "required_gates", "validation_findings")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("run-plan revision draft executable must remain false")
        return False


class RunPlanApplicationArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    generated_at: str = Field(default_factory=now_iso)
    compiled: CompiledReplanApplication
    application_record: ReplanApplicationRecord
    result_artifact_id: str
    result_artifact: dict[str, Any]
    artifact_ids: list[str]
    artifacts: dict[str, str]
    executable: bool = False

    @field_validator("project_id", "run_id", "result_artifact_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("application artifact bundle text fields are required")
        return clean

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("replan application artifacts are not executable")
        return False

    @model_validator(mode="after")
    def validate_non_executable(self) -> RunPlanApplicationArtifactBundle:
        if self.compiled.executable is not False or self.application_record.executable is not False:
            raise ValueError("compiled application and record must remain non-executable")
        return self


def proposal_artifact_hash(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def write_replan_application_artifacts(
    *,
    workspace_dir: str | Path,
    request: ReplanApplicationRequest | dict[str, Any],
    actor: str,
    actor_source: str,
    current_run_plan: RunPlan | dict[str, Any] | None = None,
    stage_state: StageState | dict[str, Any] | None = None,
) -> RunPlanApplicationArtifactBundle:
    """Materialize a compiled replan application draft as review artifacts.

    This function reads the proposal artifact, verifies its hash, compiles the
    selected advisory operations, writes review artifacts, and registers their
    artifact refs. It does not enqueue work, execute adapters, call LLMs, mutate
    a RunPlan, apply patches, or replace `/api/run-plan/execute`.
    """

    application_request = (
        request if isinstance(request, ReplanApplicationRequest) else ReplanApplicationRequest.model_validate(request)
    )
    storage = ProjectStorage(Path(workspace_dir))
    run_dir = storage.run_dir(application_request.project_id, application_request.run_id)
    proposal_path = _resolve_proposal_artifact(run_dir, application_request.proposal_artifact_ref)
    observed_hash = proposal_artifact_hash(proposal_path)
    if observed_hash != application_request.proposal_hash:
        raise ValueError("proposal_hash mismatch")
    proposal = _read_proposal(proposal_path)
    patch = ReviewableRunPlanPatch.model_validate(proposal.proposed_run_plan_patch)
    compiled = validate_and_compile_replan_application(application_request, patch, proposal)
    resume_state_binding, waiting_context = _resume_state(
        compiled=compiled,
        current_run_plan=current_run_plan,
        stage_state=stage_state,
    )
    application_id = _application_id(compiled, resume_state_binding=resume_state_binding)
    result_ref = _result_relative_path(compiled.result_type)
    application_record = ReplanApplicationRecord(
        application_id=application_id,
        project_id=compiled.project_id,
        run_id=compiled.run_id,
        proposal_artifact_ref=application_request.proposal_artifact_ref,
        proposal_hash=compiled.proposal_hash,
        selected_action=compiled.selected_action,
        selected_operation_ids=compiled.selected_operation_ids,
        result_type=compiled.result_type,
        result_ref=result_ref,
        resume_state_binding=resume_state_binding,
        actor=actor,
        actor_source=actor_source,
        executable=False,
    )
    result_artifact_id, result_artifact = _result_artifact(
        compiled=compiled,
        application_record=application_record,
        actor=actor,
        actor_source=actor_source,
        resume_state_binding=resume_state_binding,
        waiting_context=waiting_context,
    )
    artifacts = {
        REPLAN_APPLICATION_RECORD_ARTIFACT_ID: "review/replan_application_record.json",
        result_artifact_id: result_ref,
    }
    write_json(run_dir / artifacts[REPLAN_APPLICATION_RECORD_ARTIFACT_ID], application_record.model_dump(mode="json"))
    write_json(run_dir / result_ref, result_artifact)
    for artifact_id, relative_path in artifacts.items():
        storage.register_artifact_path(compiled.project_id, compiled.run_id, artifact_id, relative_path)
    return RunPlanApplicationArtifactBundle(
        project_id=compiled.project_id,
        run_id=compiled.run_id,
        compiled=compiled,
        application_record=application_record,
        result_artifact_id=result_artifact_id,
        result_artifact=result_artifact,
        artifact_ids=list(artifacts),
        artifacts=artifacts,
        executable=False,
    )


def _resolve_proposal_artifact(run_dir: Path, relative_path: str) -> Path:
    path = (run_dir / relative_path).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("proposal artifact path escapes run directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"proposal artifact not found: {relative_path}")
    return path


def _read_proposal(path: Path) -> RunPlanReplanProposal:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("proposal artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("proposal artifact JSON root must be an object")
    return RunPlanReplanProposal.model_validate(payload)


def _application_id(
    compiled: CompiledReplanApplication,
    *,
    resume_state_binding: ResumeStateBinding | None,
) -> str:
    material_parts = [
        compiled.project_id,
        compiled.run_id,
        compiled.proposal_hash,
        compiled.selected_action,
        ",".join(compiled.selected_operation_ids),
    ]
    if resume_state_binding is not None:
        material_parts.extend([
            resume_state_binding.run_plan_fingerprint,
            resume_state_binding.stage_fingerprint,
        ])
    material = "|".join(material_parts)
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"replan-application-{compiled.run_id}-{digest}"


def _result_relative_path(result_type: str) -> str:
    if result_type == "resume_intent":
        return "review/replan_resume_intent.json"
    if result_type == "run_plan_revision":
        return "review/run_plan_revision.json"
    if result_type == "blocked_acknowledgement":
        return "review/blocked_acknowledgement.json"
    raise ValueError(f"unsupported result_type: {result_type}")


def _result_artifact(
    *,
    compiled: CompiledReplanApplication,
    application_record: ReplanApplicationRecord,
    actor: str,
    actor_source: str,
    resume_state_binding: ResumeStateBinding | None,
    waiting_context: WaitingStageGateContext | None,
) -> tuple[str, dict[str, Any]]:
    if compiled.result_type == "resume_intent":
        if resume_state_binding is None or waiting_context is None:
            raise ValueError("resume_state_binding and waiting stage context are required for resume_intent artifacts")
        intent = ResumeIntent(
            intent_id=f"resume-{application_record.application_id}",
            project_id=compiled.project_id,
            run_id=compiled.run_id,
            source_application_id=application_record.application_id,
            action=compiled.selected_action,
            affected_tasks=_affected_tasks(compiled),
            application_required_gates=compiled.required_gates,
            approved_gates=[],
            rerun_tasks=_rerun_tasks(compiled),
            resume_from_task=waiting_context.stage,
            required_gates=waiting_context.execution_required_gates,
            reason=_reason(compiled),
            resume_state_binding=resume_state_binding,
            created_by=actor,
            actor_source=actor_source,
            executable=False,
        )
        return REPLAN_RESUME_INTENT_ARTIFACT_ID, intent.model_dump(mode="json")
    if compiled.result_type == "run_plan_revision":
        draft = RunPlanRevisionDraftArtifact(
            application_id=application_record.application_id,
            project_id=compiled.project_id,
            run_id=compiled.run_id,
            proposal_hash=compiled.proposal_hash,
            selected_action=compiled.selected_action,
            selected_operation_ids=compiled.selected_operation_ids,
            selected_operations=compiled.selected_operations,
            required_gates=compiled.required_gates,
            validation_findings=compiled.validation_findings,
            created_by=actor,
            actor_source=actor_source,
            executable=False,
        )
        return RUN_PLAN_REVISION_ARTIFACT_ID, draft.model_dump(mode="json")
    acknowledgement = BlockedAcknowledgement(
        application_id=application_record.application_id,
        project_id=compiled.project_id,
        run_id=compiled.run_id,
        blocked_reason=_reason(compiled),
        source_finding_ids=_source_finding_ids(compiled),
        required_user_decisions=compiled.validation_findings,
        created_by=actor,
        actor_source=actor_source,
        executable=False,
    )
    return BLOCKED_ACKNOWLEDGEMENT_ARTIFACT_ID, acknowledgement.model_dump(mode="json")


def _resume_state(
    *,
    compiled: CompiledReplanApplication,
    current_run_plan: RunPlan | dict[str, Any] | None,
    stage_state: StageState | dict[str, Any] | None,
) -> tuple[ResumeStateBinding | None, WaitingStageGateContext | None]:
    if compiled.result_type != "resume_intent":
        return None, None
    if current_run_plan is None or stage_state is None:
        raise ValueError("current_run_plan and stage_state are required for resume_intent application artifacts")
    run_plan = current_run_plan if isinstance(current_run_plan, RunPlan) else RunPlan.model_validate(current_run_plan)
    if run_plan.run_id != compiled.run_id:
        raise ValueError("current_run_plan run_id does not match replan application run_id")
    stage = stage_state if isinstance(stage_state, StageState) else StageState.model_validate(stage_state)
    waiting_context = build_waiting_stage_gate_context(
        run_plan=run_plan,
        stage_state=stage,
        application_required_gates=compiled.required_gates,
    )
    _validate_resume_action_stage_contract(compiled=compiled, waiting_stage=waiting_context.stage)
    return build_resume_state_binding(run_plan, stage), waiting_context


def _validate_resume_action_stage_contract(
    *,
    compiled: CompiledReplanApplication,
    waiting_stage: str,
) -> None:
    affected_tasks = set(_affected_tasks(compiled))
    rerun_tasks = set(_rerun_tasks(compiled))
    if compiled.selected_action == "rerun_task":
        if waiting_stage not in rerun_tasks or waiting_stage not in affected_tasks:
            raise ValueError("rerun_task_stage_mismatch")
    if compiled.selected_action == "request_review":
        if rerun_tasks:
            raise ValueError("request_review_rerun_tasks")
        if affected_tasks and waiting_stage not in affected_tasks:
            raise ValueError("request_review_stage_mismatch")
    if compiled.selected_action == "continue" and rerun_tasks:
        raise ValueError("continue_rerun_tasks")


def _affected_tasks(compiled: CompiledReplanApplication) -> list[str]:
    return _clean_string_list([operation.task_id for operation in compiled.selected_operations])


def _rerun_tasks(compiled: CompiledReplanApplication) -> list[str]:
    if compiled.selected_action != "rerun_task":
        return []
    return _affected_tasks(compiled)


def _source_finding_ids(compiled: CompiledReplanApplication) -> list[str]:
    return _clean_string_list([operation.source_finding_id for operation in compiled.selected_operations])


def _reason(compiled: CompiledReplanApplication) -> str:
    reasons = _clean_string_list([operation.reason for operation in compiled.selected_operations])
    return reasons[0] if reasons else f"User confirmed {compiled.selected_action}."


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
