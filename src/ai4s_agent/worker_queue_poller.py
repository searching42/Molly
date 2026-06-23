from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from ai4s_agent.worker_queue import WorkerQueue

PollAction = Literal["idle", "recovered", "acquired", "heartbeat", "cancel_requested"]


@dataclass(frozen=True)
class WorkerQueuePollResult:
    worker_id: str
    action: PollAction
    recovered_job_ids: list[str] = field(default_factory=list)
    acquired_job: dict | None = None
    heartbeat_job: dict | None = None
    active_lease: dict | None = None
    cancellation_requested: bool = False


class WorkerQueuePoller:
    """Control-plane polling skeleton for worker queue leases.

    The poller intentionally does not execute queue tasks.  It only coordinates
    recover -> acquire -> heartbeat -> cancellation visibility around
    `WorkerQueue` so a later PR can attach a real worker runner behind the same
    state transitions.
    """

    def __init__(
        self,
        queue: WorkerQueue,
        *,
        worker_id: str,
        poll_interval_sec: float = 0.0,
    ) -> None:
        self.queue = queue
        self.worker_id = str(worker_id or "").strip()
        if not self.worker_id:
            raise ValueError("worker_id required")
        if poll_interval_sec < 0:
            raise ValueError("poll_interval_sec must be non-negative")
        self.poll_interval_sec = poll_interval_sec

    def poll_once(self, *, now: str = "") -> WorkerQueuePollResult:
        recovered = self.queue.recover_stale_leases(now=now)
        active = self._active_lease()
        if active is not None:
            job = self.queue.status(str(active.get("job_id") or ""))
            if job is not None and bool(job.get("cancellation_requested")):
                return WorkerQueuePollResult(
                    worker_id=self.worker_id,
                    action="cancel_requested",
                    recovered_job_ids=recovered,
                    active_lease=active,
                    cancellation_requested=True,
                )
            heartbeat_job = self.queue.heartbeat(str(active.get("lease_id") or ""), now=now)
            refreshed = self.queue.lease_status(str(active.get("lease_id") or ""))
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="heartbeat",
                recovered_job_ids=recovered,
                heartbeat_job=heartbeat_job,
                active_lease=refreshed,
            )

        acquired = self.queue.acquire(self.worker_id, now=now)
        if acquired is not None:
            return WorkerQueuePollResult(
                worker_id=self.worker_id,
                action="acquired",
                recovered_job_ids=recovered,
                acquired_job=acquired,
                active_lease=self.queue.lease_status(str(acquired.get("lease_id") or "")),
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
