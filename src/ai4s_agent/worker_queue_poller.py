from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai4s_agent.worker_queue import WorkerQueue
from ai4s_agent.worker_task_runner import TaskRunResult, WorkerTaskRunner

PollAction = Literal["idle", "recovered", "acquired", "heartbeat", "cancel_requested", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class WorkerQueuePollResult:
    worker_id: str
    action: PollAction
    recovered_job_ids: list[str] = field(default_factory=list)
    acquired_job: dict | None = None
    heartbeat_job: dict | None = None
    active_lease: dict | None = None
    cancellation_requested: bool = False
    runner_result: TaskRunResult | None = None


class WorkerQueuePoller:
    """Control-plane polling skeleton for worker queue leases.

    The poller intentionally does not execute queue tasks by itself.  It can
    optionally bind to a `WorkerTaskRunner` protocol implementation, while this
    module remains decoupled from subprocesses, RunPlanExecutor, and remote
    workers.
    """

    def __init__(
        self,
        queue: WorkerQueue,
        *,
        worker_id: str,
        poll_interval_sec: float = 0.0,
        runner: WorkerTaskRunner | None = None,
        target_job_id: str | None = None,
        target_project_id: str | None = None,
        target_run_id: str | None = None,
    ) -> None:
        self.queue = queue
        self.worker_id = str(worker_id or "").strip()
        if not self.worker_id:
            raise ValueError("worker_id required")
        if poll_interval_sec < 0:
            raise ValueError("poll_interval_sec must be non-negative")
        self.poll_interval_sec = poll_interval_sec
        self.runner = runner
        self.target_job_id = _clean_optional_target_selector(target_job_id, "target_job_id")
        self.target_project_id = _clean_optional_target_selector(target_project_id, "target_project_id")
        self.target_run_id = _clean_optional_target_selector(target_run_id, "target_run_id")

    def poll_once(self, *, now: str = "") -> WorkerQueuePollResult:
        recovered = self.queue.recover_stale_leases(now=now)
        active = self._active_lease()
        if active is not None:
            job = self.queue.status(str(active.get("job_id") or ""))
            if job is not None and bool(job.get("cancellation_requested")):
                if self.runner is not None:
                    try:
                        result = self.runner.cancel(job)
                    except Exception as exc:
                        result = TaskRunResult(state="failed", message=f"cancel failed: {_exception_message(exc)}")
                    return self._finish_runner_result(active, result, recovered_job_ids=recovered, now=now)
                return WorkerQueuePollResult(
                    worker_id=self.worker_id,
                    action="cancel_requested",
                    recovered_job_ids=recovered,
                    active_lease=active,
                    cancellation_requested=True,
                )
            if self.runner is not None and job is not None:
                try:
                    result = self.runner.poll(job)
                except Exception as exc:
                    result = TaskRunResult(state="failed", message=_exception_message(exc))
                if result.state != "running":
                    return self._finish_runner_result(active, result, recovered_job_ids=recovered, now=now)
            heartbeat_job = self.queue.heartbeat(str(active.get("lease_id") or ""), now=now)
            refreshed = self.queue.lease_status(str(active.get("lease_id") or ""))
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="heartbeat",
                recovered_job_ids=recovered,
                heartbeat_job=heartbeat_job,
                active_lease=refreshed,
                runner_result=result if self.runner is not None and job is not None else None,
            )

        acquired = self.queue.acquire(
            self.worker_id,
            now=now,
            target_job_id=self.target_job_id,
            target_project_id=self.target_project_id,
            target_run_id=self.target_run_id,
        )
        if acquired is not None:
            if self.runner is not None:
                try:
                    runner_result = self.runner.start(acquired)
                except Exception as exc:
                    runner_result = TaskRunResult(state="failed", message=_exception_message(exc))
            else:
                runner_result = None
            if runner_result is not None and runner_result.state != "running":
                active_lease = self.queue.lease_status(str(acquired.get("lease_id") or ""))
                if active_lease is None:
                    raise KeyError(f"active lease not found for acquired job: {acquired.get('job_id')}")
                return self._finish_runner_result(active_lease, runner_result, recovered_job_ids=recovered, now=now)
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="acquired",
                recovered_job_ids=recovered,
                acquired_job=acquired,
                active_lease=self.queue.lease_status(str(acquired.get("lease_id") or "")),
                runner_result=runner_result,
            )

        return WorkerQueuePollResult(
            worker_id=self.worker_id,
            action="recovered" if recovered else "idle",
            recovered_job_ids=recovered,
        )

    def poll(self, *, max_iterations: int = 1, now: str = "") -> list[WorkerQueuePollResult]:
        if max_iterations <= 0:
            return []
        results: list[WorkerQueuePollResult] = []
        for index in range(max_iterations):
            results.append(self.poll_once(now=now))
            if index + 1 < max_iterations and self.poll_interval_sec:
                time.sleep(self.poll_interval_sec)
        return results

    def cancel(self, job_id: str, *, now: str = "") -> dict:
        return self.queue.cancel(job_id, now=now)

    def _active_lease(self) -> dict | None:
        active_leases = [
            lease for lease in self.queue.list_leases()
            if str(lease.get("worker_id") or "") == self.worker_id and str(lease.get("status") or "") == "active"
        ]
        if not active_leases:
            return None
        return sorted(active_leases, key=lambda lease: (str(lease.get("acquired_at") or ""), str(lease.get("lease_id") or "")))[0]

    def _finish_runner_result(
        self,
        lease: dict,
        result: TaskRunResult,
        *,
        recovered_job_ids: list[str],
        now: str = "",
    ) -> WorkerQueuePollResult:
        lease_id = str(lease.get("lease_id") or "")
        if result.state == "succeeded":
            output = result.output if isinstance(result.output, dict) else None
            job = self.queue.complete(lease_id, now=now, result=output)
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="completed",
                recovered_job_ids=recovered_job_ids,
                heartbeat_job=job,
                active_lease=self.queue.lease_status(lease_id),
                runner_result=result,
            )
        if result.state == "failed":
            job = self.queue.fail(lease_id, reason=result.message, now=now)
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="failed",
                recovered_job_ids=recovered_job_ids,
                heartbeat_job=job,
                active_lease=self.queue.lease_status(lease_id),
                runner_result=result,
            )
        if result.state == "cancelled":
            job = self.queue.fail(lease_id, reason="cancelled", now=now)
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="cancelled",
                recovered_job_ids=recovered_job_ids,
                heartbeat_job=job,
                active_lease=self.queue.lease_status(lease_id),
                cancellation_requested=True,
                runner_result=result,
            )
        return WorkerQueuePollResult(
            worker_id=self.worker_id,
            action="heartbeat",
            recovered_job_ids=recovered_job_ids,
            active_lease=self.queue.lease_status(lease_id),
            runner_result=result,
        )


def _exception_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _clean_optional_target_selector(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    clean = str(value or "").strip()
    if not clean:
        return None
    if clean in {".", ".."} or "/" in clean or "\\" in clean or Path(clean).name != clean:
        raise ValueError(f"{label} must be a single safe selector")
    return clean
