from __future__ import annotations

from typing import Any

from ai4s_agent.run_plan_queue import (
    RUN_PLAN_EXECUTE_KIND,
    RUN_PLAN_EXECUTE_TASK_ID,
    RunPlanExecuteQueueTask,
    validate_run_plan_execute_task,
)


LOCAL_DEMO_TASK_ID = "execute_oled_local_demo"
ALLOWLISTED_RUN_PLAN_TASKS = (LOCAL_DEMO_TASK_ID,)
NOT_ALLOWLISTED_MESSAGE = "generic_run_plan_not_allowlisted_for_oled_local_demo"


def validate_oled_local_demo_run_plan_execute_task(task_payload: object) -> RunPlanExecuteQueueTask:
    """Validate a generic RunPlan queue task is exactly the OLED local demo task."""
    if not isinstance(task_payload, dict):
        raise ValueError("task payload must be an object")
    task_id = str(task_payload.get("task_id") or "").strip()
    if task_id != RUN_PLAN_EXECUTE_TASK_ID:
        raise ValueError("task_id must be run_plan_execute")
    task_kind = str(task_payload.get("kind") or "").strip()
    if task_kind != RUN_PLAN_EXECUTE_KIND:
        raise ValueError("task kind must be run_plan_execute")
    try:
        task = validate_run_plan_execute_task(task_payload)
    except ValueError as exc:
        raise ValueError("malformed_run_plan_execute_task") from exc

    run_plan_tasks = [planned.task_id for planned in task.run_plan.tasks]
    if run_plan_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError(NOT_ALLOWLISTED_MESSAGE)
    if task.run_plan.requested_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError(NOT_ALLOWLISTED_MESSAGE)
    return task


def validate_oled_local_demo_run_plan_execute_job(job: object) -> RunPlanExecuteQueueTask:
    """Validate a queue job contains an allowlisted generic OLED RunPlan task."""
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    task_payload = job.get("task")
    if not isinstance(task_payload, dict):
        raise ValueError("job task must be an object")
    return validate_oled_local_demo_run_plan_execute_task(task_payload)


def is_oled_local_demo_run_plan_execute_job(job: object) -> bool:
    """Return whether a queue job is the allowlisted generic OLED local demo task."""
    try:
        validate_oled_local_demo_run_plan_execute_job(job)
    except ValueError:
        return False
    return True


def oled_local_demo_task_options(parsed_task: RunPlanExecuteQueueTask) -> dict[str, Any]:
    """Return copied task options for the allowlisted OLED local demo task."""
    options = parsed_task.task_options.get(LOCAL_DEMO_TASK_ID)
    return dict(options) if isinstance(options, dict) else {}
