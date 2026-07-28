from __future__ import annotations

from typing import Any

from ai4s_agent.schemas import RunPlan


LOW_RISK_QUEUED_CANARY_TASKS = frozenset(
    {
        "inspect_dataset",
        "clean_dataset",
        "check_trainability",
        "run_baseline",
        "render_report",
    }
)


def queued_canary_allowed_for_run_plan(run_plan: RunPlan) -> tuple[bool, dict[str, Any]]:
    task_ids = [str(task.task_id or "").strip() for task in run_plan.tasks]
    task_ids = [task_id for task_id in task_ids if task_id]
    if not task_ids:
        return False, {
            "allowed": False,
            "reason": "empty_task_chain",
            "task_ids": [],
            "disallowed_tasks": [],
        }
    disallowed = [task_id for task_id in task_ids if task_id not in LOW_RISK_QUEUED_CANARY_TASKS]
    if disallowed:
        return False, {
            "allowed": False,
            "reason": "contains_non_allowlisted_tasks",
            "task_ids": task_ids,
            "disallowed_tasks": disallowed,
        }
    return True, {
        "allowed": True,
        "reason": "allowlisted_low_risk_tasks",
        "task_ids": task_ids,
        "disallowed_tasks": [],
    }
