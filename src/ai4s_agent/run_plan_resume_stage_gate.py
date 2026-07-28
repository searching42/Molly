from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.run_plan_state_fingerprint import normalize_execution_snapshot_hash, run_plan_fingerprint
from ai4s_agent.schemas import RunPlan, RunStatus, StageState


class WaitingStageGateContext(BaseModel):
    """Strict read-only context for a queued resume intent waiting at an executor gate."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: Literal["WAITING_USER"] = "WAITING_USER"
    application_required_gates: list[str] = Field(default_factory=list)
    execution_required_gates: list[str] = Field(default_factory=list)
    execution_snapshot_id: str
    execution_snapshot_hash: str
    snapshot_task_id: str
    snapshot_run_id: str
    executable: bool = False

    @field_validator("stage", "execution_snapshot_id", "snapshot_task_id", "snapshot_run_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("waiting stage context text fields are required")
        return clean

    @field_validator("application_required_gates", "execution_required_gates")
    @classmethod
    def validate_gate_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("execution_snapshot_hash")
    @classmethod
    def validate_execution_snapshot_hash(cls, value: str) -> str:
        clean = normalize_execution_snapshot_hash(value)
        if not clean:
            raise ValueError("execution_snapshot_hash is required")
        return clean

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("waiting stage gate context is not executable")
        return False


def build_waiting_stage_gate_context(
    *,
    run_plan: RunPlan | dict[str, Any],
    stage_state: StageState | dict[str, Any] | None,
    application_required_gates: list[str],
    registry: AtomicTaskRegistry | None = None,
) -> WaitingStageGateContext:
    """Build and validate strict WAITING_USER stage/gate context.

    This function is side-effect free. It only validates current run-plan,
    stage-state, registry and execution-snapshot compatibility.
    """

    if stage_state is None:
        raise ValueError("stage_state_missing")
    normalized_run_plan = run_plan if isinstance(run_plan, RunPlan) else RunPlan.model_validate(run_plan)
    normalized_stage = stage_state if isinstance(stage_state, StageState) else StageState.model_validate(stage_state)
    if normalized_stage.status != RunStatus.WAITING_USER:
        raise ValueError("stage_not_waiting_user")
    stage = str(normalized_stage.stage or "").strip()
    if not stage:
        raise ValueError("waiting_stage_mismatch")
    run_plan_task_ids = {task.task_id for task in normalized_run_plan.tasks}
    if stage not in run_plan_task_ids:
        raise ValueError("waiting_stage_not_in_run_plan")
    task_registry = registry or AtomicTaskRegistry()
    try:
        spec = task_registry.get(stage)
    except ValueError as exc:
        raise ValueError("unknown_waiting_task") from exc
    try:
        execution_required_gates = _strict_string_list(list(spec.gates), label="registry_required_gates")
        stage_required_gates = _strict_string_list(
            _required_list(normalized_stage.details.get("required_gates")),
            label="stage_required_gates",
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if sorted(stage_required_gates) != sorted(execution_required_gates):
        raise ValueError("stage_required_gates_mismatch")
    snapshot = _execution_snapshot(normalized_stage)
    snapshot_id = _required_text(snapshot.get("snapshot_id"), "execution_snapshot_id")
    snapshot_hash = normalize_execution_snapshot_hash(_required_text(snapshot.get("snapshot_hash"), "execution_snapshot_hash"))
    snapshot_run_id = _required_text(snapshot.get("run_id"), "execution_snapshot.run_id")
    snapshot_task_id = _required_text(snapshot.get("task_id"), "execution_snapshot.task_id")
    if snapshot_run_id != normalized_run_plan.run_id:
        raise ValueError("execution_snapshot_run_mismatch")
    if snapshot_task_id != stage:
        raise ValueError("execution_snapshot_task_mismatch")
    if not isinstance(snapshot.get("run_plan"), dict):
        raise ValueError("execution_snapshot_run_plan_missing")
    snapshot_run_plan = RunPlan.model_validate(snapshot["run_plan"])
    if run_plan_fingerprint(snapshot_run_plan) != run_plan_fingerprint(normalized_run_plan):
        raise ValueError("execution_snapshot_run_plan_mismatch")
    try:
        snapshot_gates = _strict_string_list(_required_list(snapshot.get("approved_gates")), label="execution_snapshot_gates")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if sorted(snapshot_gates) != sorted(execution_required_gates):
        raise ValueError("execution_snapshot_gates_mismatch")
    material_hash = execution_snapshot_material_fingerprint(snapshot)
    if material_hash != snapshot_hash:
        raise ValueError("execution_snapshot_material_mismatch")
    return WaitingStageGateContext(
        stage=stage,
        status="WAITING_USER",
        application_required_gates=application_required_gates,
        execution_required_gates=execution_required_gates,
        execution_snapshot_id=snapshot_id,
        execution_snapshot_hash=snapshot_hash,
        snapshot_task_id=snapshot_task_id,
        snapshot_run_id=snapshot_run_id,
        executable=False,
    )


def execution_snapshot_material_fingerprint(snapshot: dict[str, Any]) -> str:
    """Fingerprint executor snapshot material while ignoring identity fields."""

    if not isinstance(snapshot, dict):
        raise ValueError("execution_snapshot must be an object")
    if snapshot.get("schema_version") == 2:
        material_keys = (
            "schema_version",
            "run_id",
            "task_id",
            "adapter",
            "run_plan",
            "task_options",
            "execution_payload",
            "input_artifacts",
            "resource_manifest",
            "approved_gates",
        )
    else:
        material_keys = (
            "schema_version",
            "run_id",
            "task_id",
            "adapter",
            "run_plan",
            "task_options",
            "payload",
            "input_artifacts",
            "approved_gates",
        )
    material = {key: snapshot[key] for key in material_keys if key in snapshot}
    encoded = json.dumps(
        _json_safe(material),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _execution_snapshot(stage_state: StageState) -> dict[str, Any]:
    raw = stage_state.details.get("execution_snapshot")
    if not isinstance(raw, dict):
        raise ValueError("execution_snapshot_missing")
    required = {
        "snapshot_id",
        "snapshot_hash",
        "run_id",
        "task_id",
        "run_plan",
        "approved_gates",
    }
    missing = sorted(key for key in required if key not in raw)
    if missing:
        raise ValueError("execution_snapshot_missing_fields:" + ",".join(missing))
    return raw


def _required_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("required_gates must be a list")
    return value


def _required_text(value: Any, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    return clean


def _clean_unique_strings(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _strict_string_list(values: list[Any], *, label: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError(f"{label}_invalid")
        item = raw.strip()
        if not item or item in seen:
            raise ValueError(f"{label}_invalid")
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
