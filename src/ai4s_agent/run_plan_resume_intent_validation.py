from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
