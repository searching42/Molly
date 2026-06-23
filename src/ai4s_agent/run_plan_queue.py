from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.schemas import RunPlan

RUN_PLAN_EXECUTE_TASK_ID = "run_plan_execute"
RUN_PLAN_EXECUTE_KIND = "run_plan_execute"


class RunPlanExecuteQueueTask(BaseModel):
    """Serializable worker-queue task envelope for future RunPlan execution.

    This schema is intentionally only a queued task contract. It does not call
    RunPlanExecutor, define local argv, or expose API routing behavior.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: Literal["run_plan_execute"] = RUN_PLAN_EXECUTE_TASK_ID
    kind: Literal["run_plan_execute"] = RUN_PLAN_EXECUTE_KIND
    project_id: str
    run_id: str
    run_plan: RunPlan
    input_artifacts: dict[str, Any] = Field(default_factory=dict)
    task_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_non_empty_identifier(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must be non-empty")
        return clean

    @field_validator("input_artifacts", "task_options")
    @classmethod
    def validate_json_safe_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_safe(value)
        return dict(value)

    @model_validator(mode="after")
    def validate_run_id_matches_plan(self) -> "RunPlanExecuteQueueTask":
        if self.run_id != self.run_plan.run_id:
            raise ValueError("run_id must match run_plan.run_id")
        return self

    def to_task(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_run_plan_execute_task(
    *,
    project_id: str,
    run_id: str,
    run_plan: RunPlan | dict[str, Any],
    input_artifacts: dict[str, Any] | None = None,
    task_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = RunPlanExecuteQueueTask(
        task_id=RUN_PLAN_EXECUTE_TASK_ID,
        kind=RUN_PLAN_EXECUTE_KIND,
        project_id=project_id,
        run_id=run_id,
        run_plan=run_plan,
        input_artifacts={} if input_artifacts is None else input_artifacts,
        task_options={} if task_options is None else task_options,
    )
    return task.to_task()


def validate_run_plan_execute_task(value: object) -> RunPlanExecuteQueueTask:
    if not isinstance(value, dict):
        raise ValueError("run-plan queue task must be an object")
    return RunPlanExecuteQueueTask.model_validate(value)


def _validate_json_safe(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_safe(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains non-JSON value of type {type(value).__name__}")
