from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ai4s_agent.run_plan_replan_application import ReplanApplicationRecord, ResumeIntent, ReviewableRunPlanPatch
from ai4s_agent.run_plan_replan_application_artifacts import proposal_artifact_hash
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_review_artifacts import REPLAN_PROPOSAL_ARTIFACT_ID
from ai4s_agent.schemas import RunPlan, RunStatus, StageState


ResumeIntentValidationDecision = Literal[
    "resume_eligible",
    "needs_gate_approval",
    "stale_intent",
    "invalid_intent",
    "blocked",
]

DEFAULT_RESUME_INTENT_ARTIFACT_REFS = {
    "replan_application_record": "review/replan_application_record.json",
    "replan_resume_intent": "review/replan_resume_intent.json",
    "replan_proposal": "review/replan_proposal.json",
}
_REQUIRED_ARTIFACT_REF_KEYS = set(DEFAULT_RESUME_INTENT_ARTIFACT_REFS)
_FAILED_DECISIONS = {"stale_intent", "invalid_intent", "blocked"}
_CONSUMED_AUDIT_EVENTS = {
    "resume_intent_consumed",
    "resume_intent_applied",
    "run_plan_resume_requested",
    "run_plan_resume_completed",
}


class ResumeIntentValidationRequest(BaseModel):
    """Schema for future resume-intent validation requests.

    This model is data-only. It does not read files, write audit records,
    enqueue work, execute adapters, call LLMs, mutate a RunPlan, write gate
    decisions, or replace `/api/run-plan/resume`.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    intent_id: str
    source_application_id: str
    proposal_hash: str
    artifact_refs: dict[str, str]
    approved_gates: list[str] = Field(default_factory=list)
    actor: str = ""
    actor_source: str = ""
    executable: bool = False

    @field_validator("project_id", "run_id", "intent_id", "source_application_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("proposal_hash")
    @classmethod
    def validate_proposal_hash(cls, value: str) -> str:
        return _proposal_hash(value)

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: dict[str, str]) -> dict[str, str]:
        return _artifact_refs(value)

    @field_validator("approved_gates")
    @classmethod
    def validate_approved_gates(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("actor", "actor_source")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("resume intent validation request is not executable")
        return False


class ResumeIntentValidationResult(BaseModel):
    """Stable non-executing result for future resume-intent validation."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    project_id: str
    run_id: str
    intent_id: str
    source_application_id: str
    proposal_hash: str
    decision: ResumeIntentValidationDecision
    required_gates: list[str] = Field(default_factory=list)
    approved_gates: list[str] = Field(default_factory=list)
    rerun_tasks: list[str] = Field(default_factory=list)
    affected_tasks: list[str] = Field(default_factory=list)
    resume_from_task: str = ""
    artifact_refs: dict[str, str]
    validation_findings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    executable: bool = False

    @field_validator("project_id", "run_id", "intent_id", "source_application_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("proposal_hash")
    @classmethod
    def validate_proposal_hash(cls, value: str) -> str:
        return _proposal_hash(value)

    @field_validator("required_gates", "approved_gates", "rerun_tasks", "affected_tasks", "validation_findings")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("resume_from_task")
    @classmethod
    def validate_resume_from_task(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: dict[str, str]) -> dict[str, str]:
        return _artifact_refs(value)

    @field_validator("error")
    @classmethod
    def validate_error_shape(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _error(value)

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("resume intent validation result is not executable")
        return False

    @model_validator(mode="after")
    def validate_decision_error_contract(self) -> ResumeIntentValidationResult:
        if self.decision in _FAILED_DECISIONS and self.error is None:
            raise ValueError(f"{self.decision} results require an error")
        if self.decision == "resume_eligible":
            if self.ok is not True:
                raise ValueError("resume_eligible results must have ok=true")
            if self.error is not None:
                raise ValueError("resume_eligible results must not include an error")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def validate_resume_intent(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    current_run_plan: RunPlan | dict[str, Any],
    stage_state: StageState | dict[str, Any] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
    approved_gates: list[str] | None = None,
) -> ResumeIntentValidationResult:
    """Validate a materialized resume intent against current read-only state.

    The validator reads review artifacts and current state, then returns a
    deterministic validation result. It does not write audit records, enqueue
    work, execute adapters, call LLMs, mutate a RunPlan, write gate decisions,
    or replace `/api/run-plan/resume`.
    """

    project = str(project_id or "").strip() or "unknown-project"
    run = str(run_id or "").strip() or "unknown-run"
    artifact_refs = dict(DEFAULT_RESUME_INTENT_ARTIFACT_REFS)
    try:
        run_plan = (
            current_run_plan if isinstance(current_run_plan, RunPlan) else RunPlan.model_validate(current_run_plan)
        )
        if isinstance(stage_state, StageState):
            stage = stage_state
        elif stage_state is not None:
            stage = StageState.model_validate(stage_state)
        else:
            stage = None
        run_dir = _run_dir(Path(workspace_dir), project, run)
        registry = _read_artifact_registry(run_dir)
        artifact_refs = _artifact_refs_from_registry(registry)
        application_record = _read_model(
            run_dir,
            artifact_refs["replan_application_record"],
            ReplanApplicationRecord,
            label="replan_application_record",
        )
        resume_intent = _read_model(
            run_dir,
            artifact_refs["replan_resume_intent"],
            ResumeIntent,
            label="replan_resume_intent",
        )
        proposal = _read_model(
            run_dir,
            artifact_refs["replan_proposal"],
            RunPlanReplanProposal,
            label="replan_proposal",
        )
        patch = ReviewableRunPlanPatch.model_validate(proposal.proposed_run_plan_patch)
        context = _ValidationContext(
            project_id=project,
            run_id=run,
            artifact_refs=artifact_refs,
            application_record=application_record,
            resume_intent=resume_intent,
        )
        failure = _validate_context(
            context=context,
            run_plan=run_plan,
            proposal=proposal,
            patch=patch,
            proposal_path=_resolve_artifact_path(run_dir, artifact_refs["replan_proposal"]),
            stage_state=stage,
            audit_records=audit_records or [],
        )
        if failure is not None:
            return failure
        return _eligible_result(
            context=context,
            run_plan=run_plan,
            stage_state=stage,
            approved_gates=approved_gates,
        )
    except _ValidationFailure as exc:
        return _failure_result(
            project_id=project,
            run_id=run,
            artifact_refs=artifact_refs,
            decision=exc.decision,
            error_type=exc.error_type,
            message=exc.message,
            findings=exc.findings,
        )
    except (FileNotFoundError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        return _failure_result(
            project_id=project,
            run_id=run,
            artifact_refs=artifact_refs,
            decision="invalid_intent",
            error_type="invalid_resume_intent_artifact",
            message=str(exc),
            findings=["invalid_resume_intent_artifact"],
        )


def _required_text(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("resume intent validation text fields are required")
    return clean


def _proposal_hash(value: str) -> str:
    clean = str(value or "").strip()
    if not clean.startswith("sha256:"):
        raise ValueError("proposal_hash must start with sha256:")
    digest = clean.removeprefix("sha256:")
    if not digest:
        raise ValueError("proposal_hash must include a digest")
    return clean


def _artifact_refs(value: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        clean_value = str(raw or "").strip()
        if not clean_key or not clean_value:
            continue
        _validate_relative_artifact_ref(clean_value)
        result[clean_key] = clean_value
    missing = sorted(_REQUIRED_ARTIFACT_REF_KEYS - set(result))
    if missing:
        raise ValueError(f"artifact_refs missing required keys: {', '.join(missing)}")
    return result


def _validate_relative_artifact_ref(value: str) -> None:
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError("artifact ref must stay under run directory")
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact ref must stay under run directory")


def _error(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    error_type = str(value.get("type") or "").strip()
    message = str(value.get("message") or "").strip()
    if not error_type or not message:
        raise ValueError("error must include non-empty type and message")
    clean = dict(value)
    clean["type"] = error_type
    clean["message"] = message
    return clean


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


@dataclass(frozen=True)
class _ValidationContext:
    project_id: str
    run_id: str
    artifact_refs: dict[str, str]
    application_record: ReplanApplicationRecord
    resume_intent: ResumeIntent


class _ValidationFailure(Exception):
    def __init__(
        self,
        *,
        decision: ResumeIntentValidationDecision,
        error_type: str,
        message: str,
        findings: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = decision
        self.error_type = error_type
        self.message = message
        self.findings = findings or [error_type]


def _run_dir(workspace_dir: Path, project_id: str, run_id: str) -> Path:
    workspace = workspace_dir.resolve()
    projects_root = (workspace / "projects").resolve()
    project_path = (projects_root / project_id).resolve()
    if not project_path.is_relative_to(projects_root):
        raise _ValidationFailure(
            decision="invalid_intent",
            error_type="unsafe_project_id",
            message="project_id must stay under workspace projects directory",
        )
    runs_root = (project_path / "runs").resolve()
    run_path = (runs_root / run_id).resolve()
    if not run_path.is_relative_to(runs_root):
        raise _ValidationFailure(
            decision="invalid_intent",
            error_type="unsafe_run_id",
            message="run_id must stay under project runs directory",
        )
    return run_path


def _read_artifact_registry(run_dir: Path) -> dict[str, str]:
    path = run_dir / "artifact_registry.json"
    if not path.exists():
        return {}
    payload = _read_json_object(path, label="artifact_registry")
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    return {str(key): str(value) for key, value in artifacts.items()}


def _artifact_refs_from_registry(registry: dict[str, str]) -> dict[str, str]:
    refs = dict(DEFAULT_RESUME_INTENT_ARTIFACT_REFS)
    for key in _REQUIRED_ARTIFACT_REF_KEYS:
        raw = str(registry.get(key) or "").strip()
        if raw:
            refs[key] = raw
    try:
        return _artifact_refs(refs)
    except ValueError as exc:
        raise _ValidationFailure(
            decision="invalid_intent",
            error_type="unsafe_artifact_ref",
            message=str(exc),
        ) from exc


def _read_model(run_dir: Path, relative_path: str, model: type[BaseModel], *, label: str) -> Any:
    path = _resolve_artifact_path(run_dir, relative_path)
    return model.model_validate(_read_json_object(path, label=label))


def _resolve_artifact_path(run_dir: Path, relative_path: str) -> Path:
    path = (run_dir / relative_path).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("artifact path escapes run directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"resume intent validation artifact not found: {relative_path}")
    return path


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact JSON root must be an object")
    return payload


def _validate_context(
    *,
    context: _ValidationContext,
    run_plan: RunPlan,
    proposal: RunPlanReplanProposal,
    patch: ReviewableRunPlanPatch,
    proposal_path: Path,
    stage_state: StageState | None,
    audit_records: list[dict[str, Any]],
) -> ResumeIntentValidationResult | None:
    application = context.application_record
    intent = context.resume_intent
    failure = _validate_identity(context=context, run_plan=run_plan)
    if failure is not None:
        return failure
    if application.result_type != "resume_intent":
        return _invalid_from_context(
            context,
            "application_result_type",
            "replan application record does not point to a resume intent",
        )
    if application.result_ref != context.artifact_refs["replan_resume_intent"]:
        return _invalid_from_context(
            context,
            "result_ref_mismatch",
            "replan application result_ref does not match artifact registry",
        )
    if application.proposal_artifact_ref != context.artifact_refs["replan_proposal"]:
        return _invalid_from_context(
            context,
            "proposal_ref_mismatch",
            "replan application proposal_artifact_ref does not match artifact registry",
        )
    observed_hash = proposal_artifact_hash(proposal_path)
    if observed_hash != application.proposal_hash:
        raise _ValidationFailure(
            decision="stale_intent",
            error_type="proposal_hash_mismatch",
            message="proposal artifact hash no longer matches the replan application record",
            findings=["proposal_hash_mismatch"],
        )
    if intent.source_application_id != application.application_id:
        return _invalid_from_context(
            context,
            "source_application_mismatch",
            "resume intent source_application_id does not match replan_application_record.application_id",
        )
    if intent.action != application.selected_action:
        return _invalid_from_context(
            context,
            "resume_action_mismatch",
            "resume intent action does not match selected replan application action",
        )
    if proposal.proposed_action != application.selected_action and application.selected_action not in {
        "request_review",
        "block",
    }:
        return _invalid_from_context(
            context,
            "proposal_action_mismatch",
            "replan application action is not compatible with proposal action",
        )
    known_operation_ids = {operation.operation_id for operation in patch.operations}
    unknown_operation_ids = [
        operation_id
        for operation_id in application.selected_operation_ids
        if operation_id not in known_operation_ids
    ]
    if unknown_operation_ids:
        return _invalid_from_context(
            context,
            "unknown_operation_id",
            "selected_operation_ids are not present in the current proposal artifact: "
            + ", ".join(unknown_operation_ids),
        )
    failure = _validate_current_run_plan_tasks(context=context, run_plan=run_plan)
    if failure is not None:
        return failure
    failure = _validate_audit_consumption(context=context, audit_records=audit_records)
    if failure is not None:
        return failure
    _stage_findings(stage_state)
    return None


def _validate_identity(*, context: _ValidationContext, run_plan: RunPlan) -> ResumeIntentValidationResult | None:
    application = context.application_record
    intent = context.resume_intent
    if application.project_id != context.project_id or intent.project_id != context.project_id:
        return _invalid_from_context(context, "project_id_mismatch", "resume artifacts do not match requested project_id")
    if application.run_id != context.run_id or intent.run_id != context.run_id or run_plan.run_id != context.run_id:
        return _invalid_from_context(
            context,
            "run_id_mismatch",
            "resume artifacts or current RunPlan do not match requested run_id",
        )
    return None


def _validate_current_run_plan_tasks(
    *,
    context: _ValidationContext,
    run_plan: RunPlan,
) -> ResumeIntentValidationResult | None:
    intent = context.resume_intent
    task_ids = {task.task_id for task in run_plan.tasks}
    referenced_tasks = _clean_string_list([*intent.rerun_tasks, *intent.affected_tasks])
    if intent.resume_from_task:
        referenced_tasks = _clean_string_list([*referenced_tasks, intent.resume_from_task])
    missing = [task_id for task_id in referenced_tasks if task_id not in task_ids]
    if missing:
        return _failure_from_context(
            context,
            decision="stale_intent",
            error_type="missing_run_plan_task",
            message="resume intent references tasks not present in the current RunPlan: " + ", ".join(missing),
            findings=["missing_run_plan_task:" + ",".join(missing)],
        )
    return None


def _validate_audit_consumption(
    *,
    context: _ValidationContext,
    audit_records: list[dict[str, Any]],
) -> ResumeIntentValidationResult | None:
    intent_id = context.resume_intent.intent_id
    for record in audit_records:
        if str(record.get("intent_id") or "") != intent_id:
            continue
        if str(record.get("event") or "") in _CONSUMED_AUDIT_EVENTS:
            return _failure_from_context(
                context,
                decision="stale_intent",
                error_type="resume_intent_already_consumed",
                message="resume intent has already been consumed according to audit records",
                findings=["resume_intent_already_consumed"],
            )
    return None


def _eligible_result(
    *,
    context: _ValidationContext,
    run_plan: RunPlan,
    stage_state: StageState | None,
    approved_gates: list[str] | None,
) -> ResumeIntentValidationResult:
    intent = context.resume_intent
    clean_approved_gates = _approved_gates(intent, approved_gates)
    missing_gates = [gate for gate in intent.required_gates if gate not in set(clean_approved_gates)]
    decision: ResumeIntentValidationDecision = "needs_gate_approval" if missing_gates else "resume_eligible"
    findings = [
        "artifact_refs_valid",
        "proposal_hash_valid",
        "source_application_valid",
        "selected_operation_ids_valid",
        "rerun_tasks_present_in_current_run_plan",
        *_stage_findings(stage_state),
    ]
    findings.extend(f"missing_gate_approval:{gate}" for gate in missing_gates)
    return ResumeIntentValidationResult(
        ok=True,
        project_id=context.project_id,
        run_id=context.run_id,
        intent_id=intent.intent_id,
        source_application_id=intent.source_application_id,
        proposal_hash=context.application_record.proposal_hash,
        decision=decision,
        required_gates=intent.required_gates,
        approved_gates=clean_approved_gates,
        rerun_tasks=intent.rerun_tasks,
        affected_tasks=intent.affected_tasks,
        resume_from_task=intent.resume_from_task or _first_task(intent.rerun_tasks, run_plan),
        artifact_refs=context.artifact_refs,
        validation_findings=findings,
        executable=False,
    )


def _stage_findings(stage_state: StageState | None) -> list[str]:
    if stage_state is None:
        return []
    findings = [f"stage_state_status:{stage_state.status.value}", f"stage_state_stage:{stage_state.stage}"]
    if stage_state.status == RunStatus.WAITING_USER:
        findings.append("stage_state_waiting_user")
    return findings


def _approved_gates(intent: ResumeIntent, approved_gates: list[str] | None) -> list[str]:
    if approved_gates is None:
        return list(intent.approved_gates)
    return _clean_string_list(approved_gates)


def _first_task(tasks: list[str], run_plan: RunPlan) -> str:
    if tasks:
        return tasks[0]
    return run_plan.tasks[0].task_id if run_plan.tasks else ""


def _failure_from_context(
    context: _ValidationContext,
    *,
    decision: ResumeIntentValidationDecision,
    error_type: str,
    message: str,
    findings: list[str],
) -> ResumeIntentValidationResult:
    return _failure_result(
        project_id=context.project_id,
        run_id=context.run_id,
        intent_id=context.resume_intent.intent_id,
        source_application_id=context.resume_intent.source_application_id,
        proposal_hash=context.application_record.proposal_hash,
        artifact_refs=context.artifact_refs,
        decision=decision,
        error_type=error_type,
        message=message,
        findings=findings,
        required_gates=context.resume_intent.required_gates,
        approved_gates=context.resume_intent.approved_gates,
        rerun_tasks=context.resume_intent.rerun_tasks,
        affected_tasks=context.resume_intent.affected_tasks,
        resume_from_task=context.resume_intent.resume_from_task,
    )


def _failure_result(
    *,
    project_id: str,
    run_id: str,
    decision: ResumeIntentValidationDecision,
    error_type: str,
    message: str,
    artifact_refs: dict[str, str],
    findings: list[str],
    intent_id: str = "unknown-intent",
    source_application_id: str = "unknown-application",
    proposal_hash: str = "sha256:unknown",
    required_gates: list[str] | None = None,
    approved_gates: list[str] | None = None,
    rerun_tasks: list[str] | None = None,
    affected_tasks: list[str] | None = None,
    resume_from_task: str = "",
) -> ResumeIntentValidationResult:
    try:
        safe_refs = _artifact_refs(artifact_refs)
    except ValueError:
        safe_refs = dict(DEFAULT_RESUME_INTENT_ARTIFACT_REFS)
    return ResumeIntentValidationResult(
        ok=False,
        project_id=project_id,
        run_id=run_id,
        intent_id=intent_id,
        source_application_id=source_application_id,
        proposal_hash=proposal_hash,
        decision=decision,
        required_gates=required_gates or [],
        approved_gates=approved_gates or [],
        rerun_tasks=rerun_tasks or [],
        affected_tasks=affected_tasks or [],
        resume_from_task=resume_from_task,
        artifact_refs=safe_refs,
        validation_findings=findings,
        error={"type": error_type, "message": message},
        executable=False,
    )


def _invalid_from_context(context: _ValidationContext, error_type: str, message: str) -> ResumeIntentValidationResult:
    return _failure_from_context(
        context,
        decision="invalid_intent",
        error_type=error_type,
        message=message,
        findings=[error_type],
    )
