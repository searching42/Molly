from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def test_worker_queue_acquires_queued_jobs_in_deterministic_order(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    queue = WorkerQueue(store)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second = queue.enqueue("proj-a", "run-b", {"task_id": "train"}, created_at=_iso(base + timedelta(seconds=1)))
    first = queue.enqueue("proj-a", "run-a", {"task_id": "clean"}, created_at=_iso(base))
    tie = queue.enqueue("proj-a", "run-c", {"task_id": "score"}, created_at=_iso(base))

    acquired_first = queue.acquire("worker-a")
    acquired_tie = queue.acquire("worker-a")
    acquired_second = queue.acquire("worker-a")

    assert acquired_first is not None
    assert acquired_first["job_id"] == first["job_id"]
    assert acquired_tie is not None
    assert acquired_tie["job_id"] == tie["job_id"]
    assert acquired_second is not None
    assert acquired_second["job_id"] == second["job_id"]


def test_worker_queue_acquire_records_lease_worker_and_heartbeat(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    job = queue.enqueue("proj-a", "run-a", {"task_id": "train"})

    acquired = queue.acquire("worker-a")

    assert acquired is not None
    assert acquired["job_id"] == job["job_id"]
    assert acquired["status"] == "running"
    assert acquired["lease_id"]
    assert acquired["worker_id"] == "worker-a"
    assert acquired["heartbeat_at"]
    lease = queue.lease_status(acquired["lease_id"])
    assert lease is not None
    assert lease["job_id"] == job["job_id"]
    assert lease["worker_id"] == "worker-a"
    assert lease["heartbeat_at"] == acquired["heartbeat_at"]


def test_worker_queue_persists_queue_leases_and_lock_files(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    queue = WorkerQueue(store)
    queue.enqueue("proj-a", "run-a", {"task_id": "train"})
    acquired = queue.acquire("worker-a")

    assert acquired is not None
    assert store.queue_path.exists()
    assert store.leases_path.exists()
    assert store.lock_path.exists()


def test_worker_queue_cancelled_queued_job_is_not_acquired(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    cancelled = queue.enqueue("proj-a", "run-cancel", {"task_id": "train"})
    available = queue.enqueue("proj-a", "run-next", {"task_id": "score"})

    cancelled_state = queue.cancel(cancelled["job_id"])
    acquired = queue.acquire("worker-a")

    assert cancelled_state["status"] == "cancelled"
    assert cancelled_state["cancellation_requested"] is True
    assert acquired is not None
    assert acquired["job_id"] == available["job_id"]


def test_worker_queue_cancelled_running_job_exposes_cancellation_requested(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    job = queue.enqueue("proj-a", "run-cancel-running", {"task_id": "train"})
    acquired = queue.acquire("worker-a")
    assert acquired is not None

    cancelled = queue.cancel(job["job_id"])
    status = queue.status(job["job_id"])

    assert cancelled["status"] == "running"
    assert cancelled["cancellation_requested"] is True
    assert status is not None
    assert status["cancellation_requested"] is True
    assert status["lease_id"] == acquired["lease_id"]


def test_worker_queue_heartbeat_complete_and_fail_update_terminal_state(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queue.enqueue("proj-a", "run-complete", {"task_id": "train"})
    acquired = queue.acquire("worker-a")
    assert acquired is not None

    heartbeat = queue.heartbeat(acquired["lease_id"])
    completed = queue.complete(acquired["lease_id"])

    assert heartbeat["status"] == "running"
    assert completed["status"] == "succeeded"
    assert queue.lease_status(acquired["lease_id"])["status"] == "completed"  # type: ignore[index]

    queue.enqueue("proj-a", "run-fail", {"task_id": "predict"})
    failed_acquired = queue.acquire("worker-b")
    assert failed_acquired is not None
    failed = queue.fail(failed_acquired["lease_id"], reason="adapter failed")
    assert failed["status"] == "failed"
    assert failed["error"]["reason"] == "adapter failed"
    assert queue.lease_status(failed_acquired["lease_id"])["status"] == "failed"  # type: ignore[index]


def test_worker_queue_recovers_stale_lease_and_requeues_job(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    queue = WorkerQueue(store, lease_ttl_sec=10)
    job = queue.enqueue("proj-a", "run-stale", {"task_id": "train"})
    acquired = queue.acquire("worker-a", now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert acquired is not None

    recovered = queue.recover_stale_leases(now=_iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc)))
    reacquired = queue.acquire("worker-b", now=_iso(datetime(2026, 1, 1, 0, 0, 12, tzinfo=timezone.utc)))

    assert recovered == [job["job_id"]]
    stale_lease = queue.lease_status(acquired["lease_id"])
    assert stale_lease is not None
    assert stale_lease["status"] == "stale"
    assert reacquired is not None
    assert reacquired["job_id"] == job["job_id"]
    assert reacquired["lease_id"] != acquired["lease_id"]
    assert reacquired["worker_id"] == "worker-b"


def test_worker_queue_rejects_malformed_queue_json(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    store.queue_path.write_text("{bad json", encoding="utf-8")
    queue = WorkerQueue(store)

    with pytest.raises(ValueError, match="not valid JSON"):
        queue.enqueue("proj-a", "run-a", {"task_id": "train"})

    assert store.queue_path.read_text(encoding="utf-8") == "{bad json"


def test_worker_queue_rejects_non_object_queue_json(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    store.queue_path.write_text("[]", encoding="utf-8")
    queue = WorkerQueue(store)

    with pytest.raises(ValueError, match="JSON root must be an object"):
        queue.list_jobs()

    assert store.queue_path.read_text(encoding="utf-8") == "[]"


def test_worker_queue_rejects_malformed_lease_json(tmp_path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    queue = WorkerQueue(store)
    queue.enqueue("proj-a", "run-a", {"task_id": "train"})
    store.leases_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        queue.acquire("worker-a")

    assert store.leases_path.read_text(encoding="utf-8") == "{bad json"


def test_worker_queue_rejects_backslash_path_segments(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))

    with pytest.raises(ValueError, match="project_id must be a single safe path segment"):
        queue.enqueue("proj\\a", "run-a", {"task_id": "train"})

    with pytest.raises(ValueError, match="worker_id must be a single safe path segment"):
        queue.acquire("worker\\a")
