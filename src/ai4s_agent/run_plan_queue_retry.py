from __future__ import annotations

from ai4s_agent.run_plan_execute_canary_policy import queued_canary_allowed_for_run_plan
from ai4s_agent.run_plan_queue import validate_run_plan_execute_task
from ai4s_agent.worker_queue import WorkerQueue


def enqueue_queued_canary_retry(
    queue: WorkerQueue,
    *,
    source_job_id: str,
    retry_request_id: str,
    actor: str,
    reason: str,
    now: str = "",
) -> dict:
    source = queue.status(source_job_id)
    if source is None:
        raise KeyError(f"source job not found: {source_job_id}")
    try:
        task = validate_run_plan_execute_task(source.get("task"))
    except Exception as exc:
        raise ValueError(f"source job task must be a valid run-plan execute envelope: {exc}") from exc
    if task.project_id != str(source.get("project_id") or "") or task.run_id != str(source.get("run_id") or ""):
        raise ValueError("queue job project_id/run_id must match task envelope")
    allowed, policy = queued_canary_allowed_for_run_plan(task.run_plan)
    if not allowed:
        reason_code = str(policy.get("reason") or "queued_canary_not_allowlisted")
        raise ValueError(f"queued canary explicit retry requires an allowlisted run-plan chain: {reason_code}")
    return queue.enqueue_retry_of_failed_job(
        source_job_id,
        retry_request_id=retry_request_id,
        requested_by=actor,
        reason=reason,
        now=now,
    )
