from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ai4s_agent.run_plan_queue import build_run_plan_execute_task
from ai4s_agent.run_plan_queue_retry import enqueue_queued_canary_retry
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import FakeWorkerTaskRunner, TaskRunResult


def _queue(tmp_path: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))


def _run_plan(run_id: str, task_ids: list[str], *, requested_tasks: list[str] | None = None) -> RunPlan:
    return RunPlan(
        run_id=run_id,
        requested_tasks=requested_tasks or [task_ids[-1]],
        tasks=[PlannedTask(task_id=task_id) for task_id in task_ids],
        available_artifacts=[],
        missing_artifacts=[],
    )


def _failed_run_plan_job(
    queue: WorkerQueue,
    *,
    project_id: str = "proj-a",
    run_id: str = "run-a",
    run_plan: RunPlan,
) -> dict:
    task = build_run_plan_execute_task(project_id=project_id, run_id=run_id, run_plan=run_plan)
    job = queue.enqueue(project_id, run_id, task)
    acquired = queue.acquire("worker-a", now="2026-01-01T00:00:00Z")
    assert acquired is not None
    return queue.fail(acquired["lease_id"], reason="adapter failed", now="2026-01-01T00:00:01Z")


def test_allowlisted_failed_run_plan_job_can_create_retry_child(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan("run-a", ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"])
    source = _failed_run_plan_job(queue, run_plan=run_plan)

    child = enqueue_queued_canary_retry(
        queue,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        actor="tester",
        reason="retry allowlisted canary job",
        now="2026-01-01T00:10:00Z",
    )

    assert child["retry_of_job_id"] == source["job_id"]
    assert child["task"] == source["task"]
    assert child["status"] == "queued"


@pytest.mark.parametrize(
    ("task_ids", "match"),
    [
        (["train_model"], "allowlisted"),
        (["generate_candidates"], "allowlisted"),
        (["literature_to_dataset_workflow"], "allowlisted"),
    ],
)
def test_non_allowlisted_run_plan_tasks_are_rejected(tmp_path: Path, task_ids: list[str], match: str) -> None:
    queue = _queue(tmp_path)
    source = _failed_run_plan_job(queue, run_plan=_run_plan("run-a", task_ids))

    with pytest.raises(ValueError, match=match):
        enqueue_queued_canary_retry(
            queue,
            source_job_id=source["job_id"],
            retry_request_id="retry-001",
            actor="tester",
            reason="retry disallowed chain",
        )


def test_generic_or_malformed_queue_tasks_are_rejected(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    generic = queue.enqueue("proj-a", "run-a", {"task_id": "generic"})
    acquired = queue.acquire("worker-a", now="2026-01-01T00:00:00Z")
    assert acquired is not None
    failed = queue.fail(acquired["lease_id"], reason="generic failure", now="2026-01-01T00:00:01Z")

    with pytest.raises(ValueError, match="valid run-plan execute envelope"):
        enqueue_queued_canary_retry(
            queue,
            source_job_id=failed["job_id"],
            retry_request_id="retry-001",
            actor="tester",
            reason="retry generic job",
        )


def test_source_task_project_and_run_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan("run-inner", ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"])
    task = build_run_plan_execute_task(project_id="proj-inner", run_id="run-inner", run_plan=run_plan)
    job = queue.enqueue("proj-outer", "run-outer", task)
    acquired = queue.acquire("worker-a", now="2026-01-01T00:00:00Z")
    assert acquired is not None
    failed = queue.fail(acquired["lease_id"], reason="adapter failed", now="2026-01-01T00:00:01Z")

    with pytest.raises(ValueError, match="queue job project_id/run_id must match task envelope"):
        enqueue_queued_canary_retry(
            queue,
            source_job_id=failed["job_id"],
            retry_request_id="retry-001",
            actor="tester",
            reason="retry mismatched job",
        )


def test_waiting_user_terminal_compatibility_state_is_not_retryable(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan("run-a", ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"])
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=run_plan)
    job = queue.enqueue("proj-a", "run-a", task)
    acquired = queue.acquire("worker-a", now="2026-01-01T00:00:00Z")
    assert acquired is not None
    queue.complete(
        acquired["lease_id"],
        now="2026-01-01T00:00:01Z",
        result={"ok": True, "status": "WAITING_USER", "waiting_task": "train_model"},
    )

    with pytest.raises(ValueError, match="source job status must be failed"):
        enqueue_queued_canary_retry(
            queue,
            source_job_id=job["job_id"],
            retry_request_id="retry-001",
            actor="tester",
            reason="retry waiting user job",
        )


def test_retry_helper_signature_accepts_no_payload_overrides() -> None:
    signature = inspect.signature(enqueue_queued_canary_retry)

    for forbidden in ["run_plan", "input_artifacts", "task_options", "task", "payload"]:
        assert forbidden not in signature.parameters


def test_retry_child_can_be_targeted_and_completed_without_mutating_original_failed_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan("run-a", ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"])
    source = _failed_run_plan_job(queue, run_plan=run_plan)
    before = queue.status(source["job_id"])
    child = enqueue_queued_canary_retry(
        queue,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        actor="tester",
        reason="retry allowlisted canary job",
    )
    runner = FakeWorkerTaskRunner(
        poll_results={
            child["job_id"]: TaskRunResult(
                state="succeeded",
                message="done",
                output={"task_id": "run_plan_execute"},
            )
        }
    )
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner, target_job_id=child["job_id"])

    first = poller.poll_once(now="2026-01-01T00:10:00Z")
    second = poller.poll_once(now="2026-01-01T00:10:01Z")

    assert first.action == "acquired"
    assert first.acquired_job is not None
    assert first.acquired_job["job_id"] == child["job_id"]
    assert second.action == "completed"
    assert queue.status(child["job_id"])["status"] == "succeeded"  # type: ignore[index]
    assert queue.status(source["job_id"]) == before


def test_worker_queue_poller_never_creates_retry_child_without_explicit_helper(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan("run-a", ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"])
    source = _failed_run_plan_job(queue, run_plan=run_plan)
    poller = WorkerQueuePoller(queue, worker_id="worker-a", target_job_id=source["job_id"])

    result = poller.poll_once(now="2026-01-01T00:10:00Z")

    assert result.action == "idle"
    assert len(queue.list_jobs()) == 1
    assert queue.status(source["job_id"])["status"] == "failed"  # type: ignore[index]
