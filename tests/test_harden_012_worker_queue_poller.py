from __future__ import annotations

from datetime import datetime, timezone

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


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


def test_worker_queue_poller_loop_runs_bounded_iterations(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-a", {"task_id": "train_model"})
    poller = WorkerQueuePoller(queue, worker_id="worker-a")

    results = poller.poll(max_iterations=2, now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))

    assert [result.action for result in results] == ["acquired", "heartbeat"]
    assert poller.poll(max_iterations=0) == []
