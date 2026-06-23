from __future__ import annotations

import pytest

from ai4s_agent.worker_task_runner import FakeWorkerTaskRunner, TaskRunResult, WorkerTaskRunner


def test_fake_worker_task_runner_start_returns_running() -> None:
    runner: WorkerTaskRunner = FakeWorkerTaskRunner()
    job = {"job_id": "job-a", "task": {"task_id": "train_model"}}

    result = runner.start(job)

    assert result == TaskRunResult(state="running", message="task started", output={"task_id": "train_model"})


def test_fake_worker_task_runner_poll_can_return_succeeded_or_failed() -> None:
    succeeded = FakeWorkerTaskRunner(poll_results={"job-success": TaskRunResult(state="succeeded", message="done")})
    failed = FakeWorkerTaskRunner(poll_results={"job-fail": TaskRunResult(state="failed", message="adapter failed")})

    assert succeeded.poll({"job_id": "job-success", "task": {"task_id": "score"}}).state == "succeeded"
    failure = failed.poll({"job_id": "job-fail", "task": {"task_id": "score"}})
    assert failure.state == "failed"
    assert failure.message == "adapter failed"


def test_fake_worker_task_runner_cancel_returns_cancelled() -> None:
    runner = FakeWorkerTaskRunner()
    job = {"job_id": "job-cancel", "task": {"task_id": "train_model"}}

    result = runner.cancel(job)

    assert result.state == "cancelled"
    assert result.message == "task cancelled"
    assert result.output == {"task_id": "train_model"}


def test_fake_worker_task_runner_rejects_bad_job_task_schema() -> None:
    runner = FakeWorkerTaskRunner()

    with pytest.raises(ValueError, match="job task must be an object"):
        runner.start({"job_id": "job-a", "task": []})

    with pytest.raises(ValueError, match="job task_id required"):
        runner.poll({"job_id": "job-a", "task": {}})

    with pytest.raises(ValueError, match="job_id required"):
        runner.cancel({"task": {"task_id": "train_model"}})
