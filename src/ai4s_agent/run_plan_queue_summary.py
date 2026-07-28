from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunPlanQueueExecutionSummary(BaseModel):
    """Stable JSON summary for internal queued run-plan execution."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    terminal: bool
    queued_job_id: str
    final_job: dict[str, Any] | None
    final_lease: dict[str, Any] | None
    loop_results: list[str] = Field(default_factory=list)
    waiting_user: bool = False
    waiting_task: str = ""
    required_gates: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    @field_validator("queued_job_id")
    @classmethod
    def clean_queued_job_id(cls, value: str) -> str:
        return str(value or "")

    @field_validator("loop_results")
    @classmethod
    def clean_loop_results(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value]

    @field_validator("waiting_task")
    @classmethod
    def clean_waiting_task(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("required_gates")
    @classmethod
    def clean_required_gates(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("error")
    @classmethod
    def validate_error_shape(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
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

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_run_plan_queue_execution_summary(
    *,
    ok: bool,
    terminal: bool,
    queued_job_id: str = "",
    final_job: dict[str, Any] | None = None,
    final_lease: dict[str, Any] | None = None,
    loop_results: list[str] | None = None,
    waiting_user: bool = False,
    waiting_task: str = "",
    required_gates: list[str] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return RunPlanQueueExecutionSummary(
        ok=ok,
        terminal=terminal,
        queued_job_id=queued_job_id,
        final_job=final_job,
        final_lease=final_lease,
        loop_results=list(loop_results or []),
        waiting_user=waiting_user,
        waiting_task=waiting_task,
        required_gates=list(required_gates or []),
        error=error,
    ).to_json_dict()
