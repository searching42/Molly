from __future__ import annotations

import sys
import time
from pathlib import Path

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller
from ai4s_agent.worker_supervisor import WorkerSupervisor
from ai4s_agent.worker_task_runner import WorkerSupervisorTaskRunner


def _queue(tmp_path: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))


def _poller(queue: WorkerQueue, tmp_path: Path, *, allowed_cwd_root: Path | None = None) -> WorkerQueuePoller:
    runner = WorkerSupervisorTaskRunner(
        supervisor=WorkerSupervisor(projects_root=tmp_path / "supervisor"),
        stop_timeout_sec=1,
        allowed_cwd_root=allowed_cwd_root,
    )
    return WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)


def _dummy_task(command: list[str], *, cwd: Path) -> dict[str, object]:
    return {
        "task_id": "dummy_command",
        "command": command,
        "cwd": str(cwd),
    }


def _exit_command(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _sleep_command() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def _poll_until_terminal(
    poller: WorkerQueuePoller,
    *,
    timeout_sec: float = 3.0,
) -> WorkerQueuePollResult:
    deadline = time.time() + timeout_sec
    result = poller.poll_once()
    while result.action not in {"completed", "failed", "cancelled"} and time.time() < deadline:
        time.sleep(0.05)
        result = poller.poll_once()
    return result


def test_worker_queue_supervisor_runner_marks_exit_zero_succeeded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = _queue(tmp_path)
    poller = _poller(queue, tmp_path)
    job = queue.enqueue("proj-a", "run-success", _dummy_task(_exit_command(0), cwd=workspace))

    started = poller.poll_once()
    terminal = _poll_until_terminal(poller)

    current_job = queue.status(job["job_id"])
    assert started.action == "acquired"
    assert terminal.action == "completed"
    assert current_job is not None
    assert current_job["status"] == "succeeded"
    lease = queue.lease_status(str(current_job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"


def test_worker_queue_supervisor_runner_marks_exit_one_failed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = _queue(tmp_path)
    poller = _poller(queue, tmp_path)
    job = queue.enqueue("proj-a", "run-fail", _dummy_task(_exit_command(1), cwd=workspace))

    started = poller.poll_once()
    terminal = _poll_until_terminal(poller)

    current_job = queue.status(job["job_id"])
    assert started.action == "acquired"
    assert terminal.action == "failed"
    assert current_job is not None
    assert current_job["status"] == "failed"
    lease = queue.lease_status(str(current_job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"


def test_worker_queue_supervisor_runner_cancels_running_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = _queue(tmp_path)
    poller = _poller(queue, tmp_path)
    job = queue.enqueue("proj-a", "run-cancel", _dummy_task(_sleep_command(), cwd=workspace))

    started = poller.poll_once()
    queue.cancel(job["job_id"])
    cancelled = poller.poll_once()

    current_job = queue.status(job["job_id"])
    assert started.action == "acquired"
    assert cancelled.action == "cancelled"
    assert current_job is not None
    assert current_job["status"] == "failed"
    assert current_job["error"] == {"reason": "cancelled"}
    lease = queue.lease_status(str(current_job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"


def test_worker_queue_supervisor_runner_rejects_cwd_outside_allowed_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    queue = _queue(tmp_path)
    poller = _poller(queue, tmp_path, allowed_cwd_root=workspace)
    job = queue.enqueue("proj-a", "run-cwd-outside", _dummy_task(_exit_command(0), cwd=outside))

    result = poller.poll_once()

    current_job = queue.status(job["job_id"])
    assert result.action == "failed"
    assert result.runner_result is not None
    assert result.runner_result.message == "job task cwd must stay under allowed_cwd_root"
    assert current_job is not None
    assert current_job["status"] == "failed"
    assert current_job["error"] == {"reason": "job task cwd must stay under allowed_cwd_root"}
    lease = queue.lease_status(str(current_job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"
