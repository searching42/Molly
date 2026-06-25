from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import pytest

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _queue(tmp_path: Path, *, lease_ttl_sec: int = 300) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path), lease_ttl_sec=lease_ttl_sec)


def _failed_job(
    queue: WorkerQueue,
    *,
    project_id: str = "proj-a",
    run_id: str = "run-a",
    task: dict | None = None,
    now: str = "2026-01-01T00:00:00Z",
    reason: str = "adapter failed",
) -> dict:
    job = queue.enqueue(project_id, run_id, task or {"task_id": "dummy", "nested": {"value": 1}})
    acquired = queue.acquire("worker-a", now=now)
    assert acquired is not None
    return queue.fail(acquired["lease_id"], reason=reason, now=now)


def _write_job_field(store: JsonWorkerQueueStore, job_id: str, **changes: object) -> None:
    payload = store.read_queue()
    for job in payload["jobs"]:
        if str(job.get("job_id") or "") == job_id:
            job.update(changes)
            break
    else:
        raise AssertionError(f"job not found: {job_id}")
    store.write_queue(payload)


def test_enqueue_retry_of_failed_job_creates_one_queued_child_and_preserves_source(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue, task={"task_id": "dummy", "nested": {"items": [1, 2, 3]}})
    before = queue.status(source["job_id"])

    child = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="rerun failed queued job",
        now="2026-01-01T00:10:00Z",
    )

    after = queue.status(source["job_id"])
    persisted_child = queue.status(child["job_id"])

    assert before is not None
    assert after == before
    assert child["job_id"] != source["job_id"]
    assert child["project_id"] == source["project_id"]
    assert child["run_id"] == source["run_id"]
    assert child["task"] == source["task"]
    assert child["status"] == "queued"
    assert child["attempts"] == 0
    assert child["lease_id"] == ""
    assert child["worker_id"] == ""
    assert child["heartbeat_at"] == ""
    assert child["cancellation_requested"] is False
    assert child["error"] == {}
    assert child["created_at"] == "2026-01-01T00:10:00Z"
    assert child["updated_at"] == "2026-01-01T00:10:00Z"
    assert child["retry_of_job_id"] == source["job_id"]
    assert child["retry_root_job_id"] == source["job_id"]
    assert child["retry_index"] == 1
    assert child["retry_request_id"] == "retry-001"
    assert child["retry_reason"] == "rerun failed queued job"
    assert child["retry_requested_by"] == "tester"
    assert child["original_project_id"] == source["project_id"]
    assert child["original_run_id"] == source["run_id"]
    assert "result" not in child
    assert "stale_recovered_at" not in child
    assert persisted_child == child


def test_mutating_returned_retry_child_does_not_mutate_persisted_jobs(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue, task={"task_id": "dummy", "nested": {"value": 1}})
    child = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="rerun failed queued job",
    )

    child["task"]["nested"]["value"] = 99
    child["error"]["reason"] = "mutated"

    persisted_source = queue.status(source["job_id"])
    persisted_child = queue.status(child["job_id"])

    assert persisted_source is not None
    assert persisted_child is not None
    assert persisted_source["task"]["nested"]["value"] == 1
    assert persisted_child["task"]["nested"]["value"] == 1
    assert persisted_child["error"] == {}


def test_retry_child_does_not_copy_source_error_or_result(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue, task={"task_id": "dummy"})
    store = queue.store
    _write_job_field(
        store,
        source["job_id"],
        result={"status": "FAILED", "message": "source result should not copy"},
        stale_recovered_at="2026-01-01T00:00:01Z",
    )

    child = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="rerun failed queued job",
    )

    assert child["error"] == {}
    assert "result" not in child
    assert "stale_recovered_at" not in child


@pytest.mark.parametrize(
    ("status", "setup"),
    [
        ("succeeded", lambda queue: queue.complete(queue.acquire("worker-a")["lease_id"])),  # type: ignore[index]
        ("cancelled", lambda queue: queue.cancel(queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})["job_id"])),
        ("queued", lambda queue: queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})),
        ("running", lambda queue: queue.acquire("worker-a") or {}),
    ],
)
def test_retry_rejects_non_failed_source_status(tmp_path: Path, status: str, setup) -> None:
    queue = _queue(tmp_path)
    if status in {"succeeded", "running"}:
        queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})
    source = setup(queue)
    job_id = source["job_id"]

    with pytest.raises(ValueError, match="source job status must be failed"):
        queue.enqueue_retry_of_failed_job(
            job_id,
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_retry_rejects_failed_source_with_cancellation_requested(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    _write_job_field(queue.store, source["job_id"], cancellation_requested=True)

    with pytest.raises(ValueError, match="source failed job cannot have cancellation_requested"):
        queue.enqueue_retry_of_failed_job(
            source["job_id"],
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_retry_rejects_failed_source_with_active_lease(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    job = queue.enqueue("proj-a", "run-a", {"task_id": "dummy"})
    acquired = queue.acquire("worker-a", now="2026-01-01T00:00:00Z")
    assert acquired is not None
    _write_job_field(queue.store, job["job_id"], status="failed")

    with pytest.raises(ValueError, match="source failed job cannot have an active lease"):
        queue.enqueue_retry_of_failed_job(
            job["job_id"],
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_retry_rejects_source_with_missing_or_non_object_task(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    _write_job_field(queue.store, source["job_id"], task="not-an-object")

    with pytest.raises(ValueError, match="source job task must be an object"):
        queue.enqueue_retry_of_failed_job(
            source["job_id"],
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_retry_rejects_retry_child_even_if_child_has_failed(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    child = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry",
    )
    acquired = queue.acquire("worker-a", target_job_id=child["job_id"], now="2026-01-01T00:10:00Z")
    assert acquired is not None
    queue.fail(acquired["lease_id"], reason="retry failed", now="2026-01-01T00:10:01Z")

    with pytest.raises(ValueError, match="retry child jobs are not eligible for explicit retry"):
        queue.enqueue_retry_of_failed_job(
            child["job_id"],
            retry_request_id="retry-002",
            requested_by="tester",
            reason="retry child",
        )


def test_retry_request_id_is_idempotent_for_same_source(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)

    first = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry",
        now="2026-01-01T00:10:00Z",
    )
    second = queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry",
        now="2026-01-01T00:20:00Z",
    )

    jobs = queue.list_jobs()
    assert first["job_id"] == second["job_id"]
    assert len(jobs) == 2
    assert len([job for job in jobs if job.get("retry_of_job_id") == source["job_id"]]) == 1


def test_retry_request_id_cannot_be_reused_for_different_source_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    first_source = _failed_job(queue, run_id="run-a")
    second_source = _failed_job(queue, run_id="run-b")
    queue.enqueue_retry_of_failed_job(
        first_source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry",
    )

    with pytest.raises(ValueError, match="retry_request_id already belongs to a different source job"):
        queue.enqueue_retry_of_failed_job(
            second_source["job_id"],
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_second_retry_request_is_rejected_after_one_shot_child_exists(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    queue.enqueue_retry_of_failed_job(
        source["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry",
    )

    with pytest.raises(ValueError, match="explicit retry child already exists for source job"):
        queue.enqueue_retry_of_failed_job(
            source["job_id"],
            retry_request_id="retry-002",
            requested_by="tester",
            reason="retry again",
        )


def test_stale_recovery_and_explicit_retry_remain_separate(tmp_path: Path) -> None:
    queue = _queue(tmp_path, lease_ttl_sec=10)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    job = queue.enqueue("proj-a", "run-a", {"task_id": "dummy"}, created_at=_iso(base))
    acquired = queue.acquire("worker-a", now=_iso(base))
    assert acquired is not None
    queue.recover_stale_leases(now=_iso(base + timedelta(seconds=11)))
    reacquired = queue.acquire("worker-b", now=_iso(base + timedelta(seconds=12)))
    assert reacquired is not None
    failed = queue.fail(reacquired["lease_id"], reason="failed after recovery", now=_iso(base + timedelta(seconds=13)))

    child = queue.enqueue_retry_of_failed_job(
        failed["job_id"],
        retry_request_id="retry-001",
        requested_by="tester",
        reason="retry after stale recovery",
        now=_iso(base + timedelta(seconds=14)),
    )

    source = queue.status(job["job_id"])
    assert source is not None
    assert source["attempts"] == 2
    assert source["stale_recovered_at"] == _iso(base + timedelta(seconds=11))
    assert child["retry_index"] == 1
    assert child["attempts"] == 0


def test_retry_rejects_empty_required_fields(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)

    with pytest.raises(ValueError, match="retry_request_id required"):
        queue.enqueue_retry_of_failed_job(source["job_id"], retry_request_id="", requested_by="tester", reason="retry")
    with pytest.raises(ValueError, match="requested_by required"):
        queue.enqueue_retry_of_failed_job(source["job_id"], retry_request_id="retry-001", requested_by="", reason="retry")
    with pytest.raises(ValueError, match="reason required"):
        queue.enqueue_retry_of_failed_job(source["job_id"], retry_request_id="retry-001", requested_by="tester", reason="")


def test_retry_rejects_missing_source_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    with pytest.raises(KeyError, match="source job not found"):
        queue.enqueue_retry_of_failed_job(
            "job-missing",
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )


def test_retry_fails_closed_on_malformed_queue_json_without_overwrite(tmp_path: Path) -> None:
    store = JsonWorkerQueueStore(tmp_path)
    raw_payload = "{bad json"
    store.queue_path.write_text(raw_payload, encoding="utf-8")
    queue = WorkerQueue(store)

    with pytest.raises(ValueError, match="not valid JSON"):
        queue.enqueue_retry_of_failed_job(
            "job-missing",
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )

    assert store.queue_path.read_text(encoding="utf-8") == raw_payload


def test_concurrent_identical_retry_requests_create_exactly_one_child(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    barrier = threading.Barrier(2)

    def _request() -> dict:
        barrier.wait()
        return queue.enqueue_retry_of_failed_job(
            source["job_id"],
            retry_request_id="retry-001",
            requested_by="tester",
            reason="retry",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_request)
        second = executor.submit(_request)
        one = first.result()
        two = second.result()

    children = [job for job in queue.list_jobs() if job.get("retry_of_job_id") == source["job_id"]]
    assert one["job_id"] == two["job_id"]
    assert len(children) == 1


def test_concurrent_different_retry_requests_leave_exactly_one_child(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    source = _failed_job(queue)
    barrier = threading.Barrier(2)

    def _request(retry_request_id: str) -> tuple[str, str]:
        barrier.wait()
        try:
            child = queue.enqueue_retry_of_failed_job(
                source["job_id"],
                retry_request_id=retry_request_id,
                requested_by="tester",
                reason="retry",
            )
            return ("ok", child["job_id"])
        except Exception as exc:
            return ("error", str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_request, "retry-001")
        second = executor.submit(_request, "retry-002")
        results = [first.result(), second.result()]

    children = [job for job in queue.list_jobs() if job.get("retry_of_job_id") == source["job_id"]]
    assert len(children) == 1
    assert sorted(kind for kind, _ in results) == ["error", "ok"]
