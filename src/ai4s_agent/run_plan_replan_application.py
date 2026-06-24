from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.run_plan_replan_proposal import PATCH_SCHEMA_VERSION, RunPlanReplanAction, RunPlanReplanProposal


ReplanApplicationResultType = Literal["resume_intent", "run_plan_revision", "blocked_acknowledgement"]


class ReplanPatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    op: RunPlanReplanAction
    task_id: str = ""
    source_finding_id: str = ""
    category: str = ""
    reason: str = ""

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _clean_operation_id(value)

    @field_validator("task_id", "source_finding_id", "category", "reason")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        return str(value or "").strip()


class ReviewableRunPlanPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reviewable_run_plan_patch.v1"] = PATCH_SCHEMA_VERSION
    applied: bool = False
    operations: list[ReplanPatchOperation] = Field(default_factory=list)

    @field_validator("applied")
    @classmethod
    def validate_applied_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("reviewable run-plan patch must remain applied=false")
        return False

    @model_validator(mode="after")
    def validate_unique_operation_ids(self) -> ReviewableRunPlanPatch:
        seen: set[str] = set()
        for operation in self.operations:
            if operation.operation_id in seen:
                raise ValueError(f"duplicate operation_id: {operation.operation_id}")
            seen.add(operation.operation_id)
        return self


class ReplanApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    proposal_artifact_ref: str
    proposal_hash: str
    selected_action: RunPlanReplanAction
    selected_operation_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    executable: bool = False

    @field_validator("project_id", "run_id", "proposal_artifact_ref", "proposal_hash")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("selected_operation_ids")
    @classmethod
    def validate_selected_operation_ids(cls, value: list[str]) -> list[str]:
        return _clean_operation_id_list(value, label="selected_operation_ids")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("replan application request executable must remain false")
        return False


class ReplanApplicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str
    project_id: str
    run_id: str
    proposal_artifact_ref: str
    proposal_hash: str
    selected_action: RunPlanReplanAction
    selected_operation_ids: list[str] = Field(default_factory=list)
    applied: bool = True
    result_type: ReplanApplicationResultType
    result_ref: str = ""
    actor: str
    actor_source: str
    created_at: str = Field(default_factory=now_iso)
    executable: bool = False

    @field_validator(
        "application_id",
        "project_id",
        "run_id",
        "proposal_artifact_ref",
        "proposal_hash",
        "actor",
        "actor_source",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("selected_operation_ids")
    @classmethod
    def validate_selected_operation_ids(cls, value: list[str]) -> list[str]:
        return _clean_operation_id_list(value, label="selected_operation_ids")

    @field_validator("applied")
    @classmethod
    def validate_applied_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("replan application record must be marked applied=true")
        return True

    @field_validator("result_ref")
    @classmethod
    def validate_result_ref(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("executable must remain false for replan application records")
        return False

    @model_validator(mode="after")
    def validate_action_result_compatibility(self) -> ReplanApplicationRecord:
        if self.selected_action == "block" and self.result_type != "blocked_acknowledgement":
            raise ValueError("block action must create a blocked acknowledgement")
        if self.selected_action != "block" and self.result_type == "blocked_acknowledgement":
            raise ValueError("blocked acknowledgement result requires block action")
        return self


class ResumeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str
    project_id: str
    run_id: str
    source_application_id: str
    action: RunPlanReplanAction
    affected_tasks: list[str] = Field(default_factory=list)
    approved_gates: list[str] = Field(default_factory=list)
    rerun_tasks: list[str] = Field(default_factory=list)
    resume_from_task: str = ""
    required_gates: list[str] = Field(default_factory=list)
    reason: str
    created_at: str = Field(default_factory=now_iso)
    created_by: str
    actor_source: str
    executable: bool = False

    @field_validator(
        "intent_id",
        "project_id",
        "run_id",
        "source_application_id",
        "reason",
        "created_by",
        "actor_source",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("affected_tasks", "approved_gates", "rerun_tasks", "required_gates")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("resume_from_task")
    @classmethod
    def validate_resume_from_task(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("resume intent executable must remain false")
        return False

    @model_validator(mode="after")
    def validate_resume_action(self) -> ResumeIntent:
        if self.action == "block":
            raise ValueError("block cannot create a resume intent")
        return self


class BlockedAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str
    project_id: str
    run_id: str
    blocked_reason: str
    source_finding_ids: list[str] = Field(default_factory=list)
    required_user_decisions: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    created_by: str
    actor_source: str
    executable: bool = False

    @field_validator(
        "application_id",
        "project_id",
        "run_id",
        "blocked_reason",
        "created_by",
        "actor_source",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("source_finding_ids", "required_user_decisions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("blocked acknowledgement executable must remain false")
        return False


class CompiledReplanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    proposal_hash: str
    proposal_action: RunPlanReplanAction
    selected_action: RunPlanReplanAction
    selected_operation_ids: list[str] = Field(default_factory=list)
    selected_operations: list[ReplanPatchOperation] = Field(default_factory=list)
    result_type: ReplanApplicationResultType
    required_gates: list[str] = Field(default_factory=list)
    validation_findings: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("project_id", "run_id", "proposal_hash")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("selected_operation_ids")
    @classmethod
    def validate_selected_operation_ids(cls, value: list[str]) -> list[str]:
        return _clean_operation_id_list(value, label="selected_operation_ids")

    @field_validator("required_gates", "validation_findings")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("executable must remain false for compiled replan applications")
        return False


def validate_selected_operation_ids_for_patch(
    request: ReplanApplicationRequest | dict[str, Any],
    patch: ReviewableRunPlanPatch | dict[str, Any],
) -> list[str]:
    application_request = (
        request if isinstance(request, ReplanApplicationRequest) else ReplanApplicationRequest.model_validate(request)
    )
    reviewable_patch = patch if isinstance(patch, ReviewableRunPlanPatch) else ReviewableRunPlanPatch.model_validate(patch)
    known_ids = {operation.operation_id for operation in reviewable_patch.operations}
    unknown = [operation_id for operation_id in application_request.selected_operation_ids if operation_id not in known_ids]
    if unknown:
        raise ValueError(f"unknown operation_id in selected_operation_ids: {', '.join(unknown)}")
    return list(application_request.selected_operation_ids)


def validate_and_compile_replan_application(
    request: ReplanApplicationRequest | dict[str, Any],
    patch: ReviewableRunPlanPatch | dict[str, Any],
    proposal: RunPlanReplanProposal | dict[str, Any],
) -> CompiledReplanApplication:
    """Validate a user-selected replan proposal subset into a non-executable draft.

    The compiler is deterministic and side-effect free. It does not write
    files, mutate a RunPlan, enqueue work, execute adapters, call LLMs, or apply
    the patch.
    """

    application_request = (
        request if isinstance(request, ReplanApplicationRequest) else ReplanApplicationRequest.model_validate(request)
    )
    reviewable_patch = patch if isinstance(patch, ReviewableRunPlanPatch) else ReviewableRunPlanPatch.model_validate(patch)
    replan_proposal = (
        proposal if isinstance(proposal, RunPlanReplanProposal) else RunPlanReplanProposal.model_validate(proposal)
    )
    proposal_patch = ReviewableRunPlanPatch.model_validate(replan_proposal.proposed_run_plan_patch)
    if proposal_patch.model_dump(mode="json") != reviewable_patch.model_dump(mode="json"):
        raise ValueError("reviewable patch must match proposal proposed_run_plan_patch")

    selected_ids = validate_selected_operation_ids_for_patch(application_request, reviewable_patch)
    action_finding = _validate_selected_action(
        selected_action=application_request.selected_action,
        proposal_action=replan_proposal.proposed_action,
    )
    selected_operations = _selected_operations(reviewable_patch, selected_ids)
    result_type = _result_type_for(application_request.selected_action)
    required_gates = _required_gates_for(application_request.selected_action)
    validation_findings = [
        "proposal patch validated",
        "selected operation ids validated",
        action_finding,
        f"compiled result_type={result_type}",
    ]
    return CompiledReplanApplication(
        project_id=application_request.project_id,
        run_id=application_request.run_id,
        proposal_hash=application_request.proposal_hash,
        proposal_action=replan_proposal.proposed_action,
        selected_action=application_request.selected_action,
        selected_operation_ids=selected_ids,
        selected_operations=selected_operations,
        result_type=result_type,
        required_gates=required_gates,
        validation_findings=validation_findings,
        executable=False,
    )


def _validate_selected_action(*, selected_action: RunPlanReplanAction, proposal_action: RunPlanReplanAction) -> str:
    if selected_action == proposal_action:
        return "selected action matches proposal action"
    if selected_action == "block":
        return "selected action downgraded to block"
    if selected_action == "request_review" and proposal_action != "block":
        return "selected action downgraded to request_review"
    raise ValueError("selected_action must match proposal action or be downgraded to request_review/block")


def _selected_operations(
    patch: ReviewableRunPlanPatch,
    selected_operation_ids: list[str],
) -> list[ReplanPatchOperation]:
    by_id = {operation.operation_id: operation for operation in patch.operations}
    return [by_id[operation_id] for operation_id in selected_operation_ids]


def _result_type_for(action: RunPlanReplanAction) -> ReplanApplicationResultType:
    if action == "block":
        return "blocked_acknowledgement"
    if action in {"adjust_targets", "collect_more_data"}:
        return "run_plan_revision"
    return "resume_intent"


def _required_gates_for(action: RunPlanReplanAction) -> list[str]:
    gates = {
        "continue": [],
        "request_review": ["gate_replan_review"],
        "rerun_task": ["gate_replan_rerun_task"],
        "adjust_targets": ["gate_replan_adjust_targets"],
        "collect_more_data": ["gate_replan_collect_more_data"],
        "block": [],
    }
    return list(gates[action])


def _required_text(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("required text fields must be non-empty")
    return clean


def _clean_operation_id(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("operation_id is required")
    if clean in {".", ".."} or "/" in clean or "\\" in clean:
        raise ValueError("operation_id must be a stable safe identifier")
    if any(char.isspace() for char in clean):
        raise ValueError("operation_id must not contain whitespace")
    return clean


def _clean_operation_id_list(values: list[str], *, label: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        operation_id = _clean_operation_id(value)
        if operation_id in seen:
            raise ValueError(f"duplicate {label}: {operation_id}")
        seen.add(operation_id)
        cleaned.append(operation_id)
    return cleaned


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
