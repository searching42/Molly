from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai4s_agent._utils import write_json
from ai4s_agent.run_plan_queue_lifecycle import (
    cleanup_terminal_run_plan_queue,
    internal_run_plan_queue_dir,
    read_run_plan_queue_status,
    recover_stale_run_plan_queue,
)
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _queue(tmp_path: Path, *, project_id: str = "proj-a", run_id: str = "run-a") -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(internal_run_plan_queue_dir(tmp_path, project_id, run_id)))


def test_internal_run_plan_queue_dir_stays_under_workspace(tmp_path: Path) -> None:
    queue_dir = internal_run_plan_queue_dir(tmp_path, "proj-a", "run-a")

    assert queue_dir == (tmp_path / ".ai4s_internal" / "run_plan_queues" / "proj-a" / "run-a").resolve()

    with pytest.raises(ValueError, match="project_id must be a safe path component"):
        internal_run_plan_queue_dir(tmp_path, "../outside", "run-a")

    with pytest.raises(ValueError, match="run_id must be a safe path component"):
        internal_run_plan_queue_dir(tmp_path, "proj-a", "a/b")


def test_read_run_plan_queue_status_empty_queue(tmp_path: Path) -> None:
    status = read_run_plan_queue_status(_queue(tmp_path))

    assert status["jobs"] == []
    assert status["leases"] == []
    assert status["counts"] == {
        "jobs_total": 0,
        "leases_total": 0,
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
        "active_leases": 0,
        "terminal_jobs": 0,
        "terminal_leases": 0,
    }
    assert status["has_active_jobs"] is False
    assert status["has_terminal_jobs"] is False


def test_read_run_plan_queue_status_after_successful_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})
    acquired = queue.acquire("worker-a")
    assert acquired is not None
    queue.complete(acquired["lease_id"])

    status = read_run_plan_queue_status(queue)

    assert [job["status"] for job in status["jobs"]] == ["succeeded"]
    assert [lease["status"] for lease in status["leases"]] == ["completed"]
    assert status["counts"]["succeeded"] == 1
    assert status["counts"]["terminal_jobs"] == 1
    assert status["counts"]["terminal_leases"] == 1
    assert status["has_active_jobs"] is False
    assert status["has_terminal_jobs"] is True


def test_recover_stale_run_plan_queue_requeues_expired_running_job(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(internal_run_plan_queue_dir(tmp_path, "proj-a", "run-a")), lease_ttl_sec=10)
    job = queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})
    acquired = queue.acquire("worker-a", now=_iso(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert acquired is not None

    result = recover_stale_run_plan_queue(queue, now=_iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc)))

    assert result == {"recovered_job_ids": [job["job_id"]], "recovered_count": 1}
    status = read_run_plan_queue_status(queue)
    assert status["jobs"][0]["status"] == "queued"
    assert status["leases"][0]["status"] == "stale"


def test_cleanup_terminal_run_plan_queue_keeps_active_jobs(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("proj-a", "run-a", {"task_id": "terminal-success"})
    success = queue.acquire("worker-a")
    assert success is not None
    queue.complete(success["lease_id"])
    queue.enqueue("proj-a", "run-a", {"task_id": "active-queued"})

    result = cleanup_terminal_run_plan_queue(queue)

    assert result["removed_jobs"] == 1
    assert result["removed_leases"] == 1
    assert result["deleted_files"] is False
    status = read_run_plan_queue_status(queue)
    assert [job["status"] for job in status["jobs"]] == ["queued"]
    assert status["leases"] == []


def test_cleanup_terminal_run_plan_queue_deletes_files_when_all_records_terminal(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("proj-a", "run-a", {"task_id": "terminal-success"})
    success = queue.acquire("worker-a")
    assert success is not None
    queue.complete(success["lease_id"])
    queue.enqueue("proj-a", "run-a", {"task_id": "terminal-failed"})
    failed = queue.acquire("worker-b")
    assert failed is not None
    queue.fail(failed["lease_id"], reason="adapter failed")
    store = queue.store

    result = cleanup_terminal_run_plan_queue(queue)

    assert result["removed_jobs"] == 2
    assert result["removed_leases"] == 2
    assert result["deleted_files"] is True
    assert not store.queue_path.exists()
    assert not store.leases_path.exists()


def test_cleanup_terminal_run_plan_queue_rejects_escaped_store_path(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "outside"))

    with pytest.raises(ValueError, match="queue root must stay under internal run-plan queue root"):
        cleanup_terminal_run_plan_queue(queue, workspace=tmp_path)


def test_cleanup_terminal_run_plan_queue_preserves_malformed_state(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.store.queue_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        cleanup_terminal_run_plan_queue(queue)

    assert queue.store.queue_path.read_text(encoding="utf-8") == "{bad json"

