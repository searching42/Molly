from __future__ import annotations

from typing import Any

from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.run_plan_queue_summary import build_run_plan_queue_execution_summary
from ai4s_agent.run_plan_task_runner import ExecutorFactory, RunPlanExecutorTaskRunner
from ai4s_agent.schemas import RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller


def run_run_plan_via_local_queue(
    *,
    queue: WorkerQueue,
    storage: ProjectStorage,
    project_id: str,
    run_plan: RunPlan | dict[str, Any],
    input_artifacts: dict[str, Any] | None = None,
    task_options: dict[str, Any] | None = None,
    max_iterations: int = 10,
    executor_factory: ExecutorFactory | None = None,
) -> dict[str, Any]:
    """Internal opt-in helper for bounded local queued run-plan execution.

    This helper composes the existing queue, poller, local loop, and one-shot
    run-plan task runner. It does not expose API routes or change the default
    synchronous `/api/run-plan/execute` path.
    """

    existing = _active_or_queued_jobs(queue)
    if existing:
        raise ValueError("run-plan local queue helper requires an empty/dedicated queue")

    queued_job = enqueue_run_plan_execute_job(
        queue,
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=task_options,
    )
    runner = RunPlanExecutorTaskRunner(storage=storage, executor_factory=executor_factory)
    poller = WorkerQueuePoller(queue, worker_id="local-run-plan-worker", runner=runner)
    loop_result = LocalWorkerLoop(poller).run_until_idle(max_iterations=max_iterations)
    final_job = queue.status(str(queued_job.get("job_id") or ""))
    final_lease = None
    if final_job is not None:
        lease_id = str(final_job.get("lease_id") or "")
        if lease_id:
            final_lease = queue.lease_status(lease_id)
    loop_actions = [item.action for item in loop_result.results]
    terminal = bool(loop_actions) and loop_actions[-1] == "idle"
    return build_run_plan_queue_execution_summary(
        ok=final_job is not None and str(final_job.get("status") or "") == "succeeded",
        terminal=terminal,
        queued_job_id=str(queued_job.get("job_id") or ""),
        final_job=final_job,
        final_lease=final_lease,
        loop_results=loop_actions,
    )


def _active_or_queued_jobs(queue: WorkerQueue) -> list[dict[str, Any]]:
    return [
        job for job in queue.list_jobs()
        if str(job.get("status") or "") in {"queued", "running"}
    ]
