from __future__ import annotations

from datetime import datetime, timezone

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import FakeWorkerTaskRunner, TaskRunResult


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class RaisingWorkerTaskRunner:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        poll_error: Exception | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        self.start_error = start_error
        self.poll_error = poll_error
        self.cancel_error = cancel_error

    def start(self, job: dict) -> TaskRunResult:
        if self.start_error is not None:
            raise self.start_error
        return TaskRunResult(state="running", message="started")

    def poll(self, job: dict) -> TaskRunResult:
        if self.poll_error is not None:
            raise self.poll_error
        return TaskRunResult(state="running", message="running")

    def cancel(self, job: dict) -> TaskRunResult:
        if self.cancel_error is not None:
            raise self.cancel_error
        return TaskRunResult(state="cancelled", message="cancelled")


def test_worker_queue_poller_acquires_queued_job_without_executing_task(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model", "command": ["do-not-run"]})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))

    assert result.action == "acquired"
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == queued["job_id"]
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "running"
    assert status["task"]["command"] == ["do-not-run"]


def test_worker_queue_poller_acquires_targeted_job_even_if_earlier_jobs_exist(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-a", {"task_id": "first"})
    target = queue.enqueue("proj-a", "run-b", {"task_id": "target"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a", target_run_id="run-b")

    result = poller.poll_once(now="2026-01-01T00:00:00Z")

    assert result.action == "acquired"
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == target["job_id"]
    assert queue.status(target["job_id"])["status"] == "running"  # type: ignore[index]


def test_worker_queue_poller_idle_when_targeted_job_missing(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-a", {"task_id": "first"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a", target_run_id="run-missing")

    result = poller.poll_once(now="2026-01-01T00:00:00Z")

    assert result.action == "idle"
    status = queue.status(queue.list_jobs()[0]["job_id"])
    assert status is not None
    assert status["status"] == "queued"


def test_worker_queue_poller_heartbeats_existing_active_lease(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    first = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    )
    assert first.acquired_job is not None

    second = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc))
    )

    assert second.action == "heartbeat"
    assert second.heartbeat_job is not None
    assert second.heartbeat_job["job_id"] == queued["job_id"]
    assert second.heartbeat_job["heartbeat_at"] == "2026-01-01T00:00:05Z"
    assert second.acquired_job is None


def test_worker_queue_poller_surfaces_running_cancellation_without_completing_job(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")
    acquired = poller.poll_once(now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert acquired.acquired_job is not None

    cancelled = poller.cancel(queued["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)))

    assert cancelled["status"] == "running"
    assert result.action == "cancel_requested"
    assert result.cancellation_requested is True
    assert result.active_lease is not None
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "running"
    assert status["cancellation_requested"] is True


def test_worker_queue_poller_does_not_heartbeat_after_running_cancellation(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")
    acquired = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert acquired.acquired_job is not None
    original_heartbeat = acquired.acquired_job["heartbeat_at"]

    poller.cancel(queued["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)))

    assert result.action == "cancel_requested"
    assert result.heartbeat_job is None
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["heartbeat_at"] == original_heartbeat
    lease = queue.lease_status(acquired.acquired_job["lease_id"])
    assert lease is not None
    assert lease["heartbeat_at"] == original_heartbeat


def test_worker_queue_poller_recovers_stale_lease_before_acquire(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path), lease_ttl_sec=10)
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    first = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    )
    assert first.acquired_job is not None

    result = WorkerQueuePoller(queue, worker_id="worker-b").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc))
    )

    assert result.recovered_job_ids == [queued["job_id"]]
    assert result.action == "acquired"
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == queued["job_id"]
    assert result.acquired_job["worker_id"] == "worker-b"


def test_worker_queue_recovery_does_not_overwrite_terminal_jobs(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path), lease_ttl_sec=10)
    succeeded = queue.enqueue("proj-a", "run-succeeded", {"task_id": "train_model"})
    failed = queue.enqueue("proj-a", "run-failed", {"task_id": "train_model"})
    acquired_succeeded = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    )
    acquired_failed = WorkerQueuePoller(queue, worker_id="worker-b").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    )
    assert acquired_succeeded.acquired_job is not None
    assert acquired_failed.acquired_job is not None
    queue.complete(acquired_succeeded.acquired_job["lease_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    queue.fail(acquired_failed.acquired_job["lease_id"], reason="adapter failed", now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))

    recovered = queue.recover_stale_leases(now=_iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc)))

    assert recovered == []
    succeeded_status = queue.status(succeeded["job_id"])
    failed_status = queue.status(failed["job_id"])
    assert succeeded_status is not None
    assert failed_status is not None
    assert succeeded_status["status"] == "succeeded"
    assert failed_status["status"] == "failed"
    assert succeeded_status["lease_id"] == acquired_succeeded.acquired_job["lease_id"]
    assert failed_status["lease_id"] == acquired_failed.acquired_job["lease_id"]


def test_worker_queue_poller_ignores_completed_and_failed_leases_when_acquiring(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-completed", {"task_id": "train_model"})
    completed = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    )
    assert completed.acquired_job is not None
    queue.complete(completed.acquired_job["lease_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    queue.enqueue("proj-a", "run-failed", {"task_id": "train_model"})
    failed = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    )
    assert failed.acquired_job is not None
    queue.fail(failed.acquired_job["lease_id"], reason="worker failed", now=_iso(datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)))
    next_job = queue.enqueue("proj-a", "run-next", {"task_id": "train_model"})

    result = WorkerQueuePoller(queue, worker_id="worker-a").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc))
    )

    assert result.action == "acquired"
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == next_job["job_id"]
    completed_lease = queue.lease_status(completed.acquired_job["lease_id"])
    failed_lease = queue.lease_status(failed.acquired_job["lease_id"])
    assert completed_lease is not None
    assert failed_lease is not None
    assert completed_lease["status"] == "completed"
    assert failed_lease["status"] == "failed"


def test_multiple_workers_acquire_jobs_in_deterministic_order(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    first = queue.enqueue("proj-a", "run-a", {"task_id": "first"}, created_at=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    third = queue.enqueue("proj-a", "run-c", {"task_id": "third"}, created_at=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    second = queue.enqueue("proj-a", "run-b", {"task_id": "second"}, created_at=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))

    acquired = [
        WorkerQueuePoller(queue, worker_id="worker-a").poll_once(now=_iso(datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc))).acquired_job,
        WorkerQueuePoller(queue, worker_id="worker-b").poll_once(now=_iso(datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc))).acquired_job,
        WorkerQueuePoller(queue, worker_id="worker-c").poll_once(now=_iso(datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc))).acquired_job,
    ]

    assert [job["job_id"] for job in acquired if job is not None] == [
        first["job_id"],
        second["job_id"],
        third["job_id"],
    ]


def test_cancelled_queued_job_is_never_acquired_by_poller(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    cancelled = queue.enqueue("proj-a", "run-cancelled", {"task_id": "train_model"})
    available = queue.enqueue("proj-a", "run-available", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")

    cancelled_status = poller.cancel(cancelled["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    second = WorkerQueuePoller(queue, worker_id="worker-b").poll_once(
        now=_iso(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    )

    assert cancelled_status["status"] == "cancelled"
    assert first.action == "acquired"
    assert first.acquired_job is not None
    assert first.acquired_job["job_id"] == available["job_id"]
    assert second.action == "idle"
    assert queue.status(cancelled["job_id"])["status"] == "cancelled"  # type: ignore[index]


def test_worker_queue_poller_loop_runs_bounded_iterations(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")

    results = poller.poll(max_iterations=2, now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))

    assert [result.action for result in results] == ["acquired", "heartbeat"]
    assert poller.poll(max_iterations=0) == []


def test_poller_with_runner_starts_acquired_job(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=FakeWorkerTaskRunner())

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))

    assert result.action == "acquired"
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == queued["job_id"]
    assert result.runner_result == TaskRunResult(
        state="running",
        message="task started",
        output={"task_id": "train_model"},
    )
    assert queue.status(queued["job_id"])["status"] == "running"  # type: ignore[index]


def test_poller_with_runner_heartbeats_running_poll(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=FakeWorkerTaskRunner())
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.acquired_job is not None

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)))

    assert result.action == "heartbeat"
    assert result.heartbeat_job is not None
    assert result.heartbeat_job["job_id"] == queued["job_id"]
    assert result.heartbeat_job["heartbeat_at"] == "2026-01-01T00:00:05Z"
    assert result.runner_result is not None
    assert result.runner_result.state == "running"


def test_poller_with_runner_completes_succeeded_poll(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    runner = FakeWorkerTaskRunner(
        poll_results={queued["job_id"]: TaskRunResult(state="succeeded", message="done", output={"artifact_id": "report"})}
    )
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.acquired_job is not None

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)))

    assert result.action == "completed"
    assert result.runner_result == TaskRunResult(state="succeeded", message="done", output={"artifact_id": "report"})
    assert queue.status(queued["job_id"])["status"] == "succeeded"  # type: ignore[index]
    assert queue.lease_status(first.acquired_job["lease_id"])["status"] == "completed"  # type: ignore[index]


def test_poller_with_runner_terminal_state_uses_poll_now(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    runner = FakeWorkerTaskRunner(
        poll_results={queued["job_id"]: TaskRunResult(state="succeeded", message="done")}
    )
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.acquired_job is not None
    terminal_now = "2026-01-01T00:00:05Z"

    result = poller.poll_once(now=terminal_now)

    assert result.action == "completed"
    status = queue.status(queued["job_id"])
    lease = queue.lease_status(first.acquired_job["lease_id"])
    assert status is not None
    assert lease is not None
    assert status["updated_at"] == terminal_now
    assert lease["completed_at"] == terminal_now


def test_poller_with_runner_fails_failed_poll(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    runner = FakeWorkerTaskRunner(
        poll_results={queued["job_id"]: TaskRunResult(state="failed", message="adapter failed")}
    )
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.acquired_job is not None

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)))

    assert result.action == "failed"
    assert result.runner_result == TaskRunResult(state="failed", message="adapter failed")
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"]["reason"] == "adapter failed"
    assert queue.lease_status(first.acquired_job["lease_id"])["status"] == "failed"  # type: ignore[index]


def test_poller_with_runner_cancels_without_heartbeat(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=FakeWorkerTaskRunner())
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.acquired_job is not None
    original_heartbeat = first.acquired_job["heartbeat_at"]
    poller.cancel(queued["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)))

    assert result.action == "cancelled"
    assert result.runner_result == TaskRunResult(state="cancelled", message="task cancelled", output={"task_id": "train_model"})
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["heartbeat_at"] == original_heartbeat
    assert status["error"]["reason"] == "cancelled"
    lease = queue.lease_status(first.acquired_job["lease_id"])
    assert lease is not None
    assert lease["heartbeat_at"] == original_heartbeat
    assert lease["status"] == "failed"


def test_poller_with_runner_fails_job_when_start_raises(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(
        queue,
        worker_id="worker-a",
        runner=RaisingWorkerTaskRunner(start_error=ValueError("invalid task cwd")),
    )

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))

    assert result.action == "failed"
    assert result.runner_result == TaskRunResult(state="failed", message="invalid task cwd")
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == {"reason": "invalid task cwd"}
    assert result.active_lease is not None
    assert result.active_lease["status"] == "failed"


def test_poller_with_runner_fails_job_when_poll_raises(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(
        queue,
        worker_id="worker-a",
        runner=RaisingWorkerTaskRunner(poll_error=RuntimeError("poll exploded")),
    )
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.action == "acquired"

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)))

    assert result.action == "failed"
    assert result.runner_result == TaskRunResult(state="failed", message="poll exploded")
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == {"reason": "poll exploded"}
    assert result.active_lease is not None
    assert result.active_lease["status"] == "failed"


def test_poller_with_runner_fails_job_when_cancel_raises(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(
        queue,
        worker_id="worker-a",
        runner=RaisingWorkerTaskRunner(cancel_error=RuntimeError("process stuck")),
    )
    first = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert first.action == "acquired"
    poller.cancel(queued["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)))

    assert result.action == "failed"
    assert result.runner_result == TaskRunResult(state="failed", message="cancel failed: process stuck")
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == {"reason": "cancel failed: process stuck"}
    assert result.active_lease is not None
    assert result.active_lease["status"] == "failed"


def test_poller_without_runner_preserves_control_plane_only_behavior(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")
    acquired = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)))
    assert acquired.action == "acquired"
    assert acquired.runner_result is None
    poller.cancel(queued["job_id"], now=_iso(datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))

    result = poller.poll_once(now=_iso(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)))

    assert result.action == "cancel_requested"
    assert result.runner_result is None
    assert queue.status(queued["job_id"])["status"] == "running"  # type: ignore[index]
