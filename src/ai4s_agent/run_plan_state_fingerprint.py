from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ai4s_agent.schemas import RunPlan, StageState


_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
RESUME_STATE_BINDING_SCHEMA_VERSION = "resume_state_binding.v1"


class ResumeStateBinding(BaseModel):
    """Compact state binding for user-confirmed resume intents.

    This is an integrity/staleness check for review artifacts. It is not a
    signature, permission grant, or execution authorization.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["resume_state_binding.v1"] = RESUME_STATE_BINDING_SCHEMA_VERSION
    run_plan_fingerprint: str
    stage_fingerprint: str
    stage: str
    stage_status: str
    execution_snapshot_id: str = ""
    execution_snapshot_hash: str = ""

    @field_validator("run_plan_fingerprint", "stage_fingerprint", "execution_snapshot_hash")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        if not _FINGERPRINT_PATTERN.fullmatch(clean):
            raise ValueError("fingerprint must use sha256:<64 lowercase hex>")
        return clean

    @field_validator("stage", "stage_status", "execution_snapshot_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def validate_required_fields_and_snapshot_pair(self) -> ResumeStateBinding:
        if not self.run_plan_fingerprint:
            raise ValueError("run_plan_fingerprint is required")
        if not self.stage_fingerprint:
            raise ValueError("stage_fingerprint is required")
        if not self.stage:
            raise ValueError("stage is required")
        if not self.stage_status:
            raise ValueError("stage_status is required")
        has_snapshot_id = bool(self.execution_snapshot_id)
        has_snapshot_hash = bool(self.execution_snapshot_hash)
        if has_snapshot_id != has_snapshot_hash:
            raise ValueError("execution_snapshot_id and execution_snapshot_hash must be provided together")
        return self


def canonical_json_fingerprint(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-safe canonical payloads."""

    if isinstance(payload, BaseModel):
        normalized = payload.model_dump(mode="json")
    else:
        normalized = payload
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run_plan_fingerprint(run_plan: RunPlan | dict[str, Any]) -> str:
    """Fingerprint the complete schema-normalized RunPlan payload."""

    normalized = run_plan if isinstance(run_plan, RunPlan) else RunPlan.model_validate(run_plan)
    return canonical_json_fingerprint(normalized.model_dump(mode="json"))


def stage_state_fingerprint(stage_state: StageState | dict[str, Any]) -> str:
    """Fingerprint stable StageState semantics while ignoring volatile times."""

    normalized = stage_state if isinstance(stage_state, StageState) else StageState.model_validate(stage_state)
    snapshot_id, snapshot_hash = _execution_snapshot_identity(normalized)
    payload = {
        "stage": normalized.stage,
        "status": normalized.status.value,
        "next_stage": normalized.next_stage or "",
        "required_gates": _sorted_unique_strings(normalized.details.get("required_gates")),
        "executed_tasks": _ordered_strings(normalized.details.get("executed_tasks")),
        "execution_snapshot": {
            "snapshot_id": snapshot_id,
            "snapshot_hash": snapshot_hash,
        },
    }
    return canonical_json_fingerprint(payload)


def build_resume_state_binding(
    run_plan: RunPlan | dict[str, Any],
    stage_state: StageState | dict[str, Any],
) -> ResumeStateBinding:
    """Build the compact state binding recorded in resume application artifacts."""

    normalized_run_plan = run_plan if isinstance(run_plan, RunPlan) else RunPlan.model_validate(run_plan)
    normalized_stage = stage_state if isinstance(stage_state, StageState) else StageState.model_validate(stage_state)
    snapshot_id, snapshot_hash = _execution_snapshot_identity(normalized_stage)
    return ResumeStateBinding(
        run_plan_fingerprint=run_plan_fingerprint(normalized_run_plan),
        stage_fingerprint=stage_state_fingerprint(normalized_stage),
        stage=normalized_stage.stage,
        stage_status=normalized_stage.status.value,
        execution_snapshot_id=snapshot_id,
        execution_snapshot_hash=snapshot_hash,
    )


def _execution_snapshot_identity(stage_state: StageState) -> tuple[str, str]:
    raw = stage_state.details.get("execution_snapshot")
    if raw is None:
        return "", ""
    if not isinstance(raw, dict):
        raise ValueError("execution_snapshot must be an object")
    snapshot_id = str(raw.get("snapshot_id") or "").strip()
    snapshot_hash = str(raw.get("snapshot_hash") or "").strip()
    if bool(snapshot_id) != bool(snapshot_hash):
        raise ValueError("execution_snapshot snapshot_id and snapshot_hash must be provided together")
    if snapshot_hash:
        ResumeStateBinding(
            run_plan_fingerprint="sha256:" + "0" * 64,
            stage_fingerprint="sha256:" + "0" * 64,
            stage=stage_state.stage,
            stage_status=stage_state.status.value,
            execution_snapshot_id=snapshot_id,
            execution_snapshot_hash=snapshot_hash,
        )
    return snapshot_id, snapshot_hash


def _sorted_unique_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _ordered_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
