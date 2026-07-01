from __future__ import annotations

import sys
from pathlib import Path

from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller
from ai4s_agent.worker_supervisor import WorkerSupervisor
from ai4s_agent.worker_task_runner import WorkerSupervisorTaskRunner


class RecordingPoller:
    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)
        self.now_values: list[str] = []

    def poll_once(self, *, now: str = "") -> WorkerQueuePollResult:
        self.now_values.append(now)
        action = self.actions.pop(0) if self.actions else "idle"
        return WorkerQueuePollResult(worker_id="worker-a", action=action)  # type: ignore[arg-type]


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


def test_local_worker_loop_run_once_delegates_to_poller() -> None:
    poller = RecordingPoller(["idle"])
    loop = LocalWorkerLoop(poller)  # type: ignore[arg-type]

    result = loop.run_once(now="2026-01-01T00:00:00Z")

    assert result.action == "idle"
    assert poller.now_values == ["2026-01-01T00:00:00Z"]


def test_local_worker_loop_run_until_idle_processes_dummy_success_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = _queue(tmp_path)
    job = queue.enqueue("proj-a", "run-success", _dummy_task(_exit_command(0), cwd=workspace))
    loop = LocalWorkerLoop(_poller(queue, tmp_path))

    result = loop.run_until_idle(max_iterations=20)

    assert result.iterations == len(result.results)
    assert result.results[-1].action == "idle"
    assert "completed" in [item.action for item in result.results]
    status = queue.status(job["job_id"])
    assert status is not None
    assert status["status"] == "succeeded"
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"


def test_local_worker_loop_run_until_idle_stops_at_idle() -> None:
    poller = RecordingPoller(["heartbeat", "idle", "heartbeat"])
    loop = LocalWorkerLoop(poller)  # type: ignore[arg-type]

    result = loop.run_until_idle(max_iterations=5, now="2026-01-01T00:00:00Z")

    assert result.iterations == 2
    assert [item.action for item in result.results] == ["heartbeat", "idle"]
    assert poller.now_values == ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]


def test_local_worker_loop_run_until_idle_respects_max_iterations() -> None:
    poller = RecordingPoller(["heartbeat", "heartbeat", "idle"])
    loop = LocalWorkerLoop(poller)  # type: ignore[arg-type]

    result = loop.run_until_idle(max_iterations=2)
    empty = loop.run_until_idle(max_iterations=0)

    assert result.iterations == 2
    assert [item.action for item in result.results] == ["heartbeat", "heartbeat"]
    assert empty.iterations == 0
    assert empty.results == []


def test_local_worker_loop_runner_exception_reaches_failed_terminal_then_idle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    queue = _queue(tmp_path)
    job = queue.enqueue("proj-a", "run-invalid-cwd", _dummy_task(_exit_command(0), cwd=outside))
    loop = LocalWorkerLoop(_poller(queue, tmp_path, allowed_cwd_root=workspace))

    result = loop.run_until_idle(max_iterations=3)

    assert [item.action for item in result.results] == ["failed", "idle"]
    status = queue.status(job["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == {"reason": "job task cwd must stay under allowed_cwd_root"}
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"
