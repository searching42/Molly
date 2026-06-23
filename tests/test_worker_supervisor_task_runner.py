from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from ai4s_agent.worker_supervisor import WorkerSupervisor
from ai4s_agent.worker_task_runner import TaskRunResult, WorkerSupervisorTaskRunner


def _job(tmp_path: Path, run_id: str, command: list[str] | str) -> dict:
    return {
        "job_id": f"job-{run_id}",
        "project_id": "proj-a",
        "run_id": run_id,
        "task": {
            "task_id": "dummy_command",
            "command": command,
            "cwd": str(tmp_path),
        },
    }


def _exit_command(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _sleep_command(seconds: int) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _wait_for_terminal(runner: WorkerSupervisorTaskRunner, job: dict, *, timeout_sec: float = 3.0) -> TaskRunResult:
    deadline = time.time() + timeout_sec
    result = runner.poll(job)
    while result.state == "running" and time.time() < deadline:
        time.sleep(0.05)
        result = runner.poll(job)
    return result


def test_worker_supervisor_task_runner_start_returns_running(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)
    runner = WorkerSupervisorTaskRunner(supervisor=supervisor)
    job = _job(tmp_path, "run-start", _sleep_command(5))

    result = runner.start(job)

    assert result.state == "running"
    assert result.message == "worker started"
    assert result.output is not None
    assert result.output["run_id"] == "run-start"
    assert result.output["pid"] > 0

    runner.cancel(job)


def test_worker_supervisor_task_runner_poll_maps_exit_zero_to_succeeded(tmp_path: Path) -> None:
    runner = WorkerSupervisorTaskRunner(supervisor=WorkerSupervisor(projects_root=tmp_path))
    job = _job(tmp_path, "run-success", _exit_command(0))

    runner.start(job)
    result = _wait_for_terminal(runner, job)

    assert result.state == "succeeded"
    assert result.message == "worker stopped"
    assert result.output is not None
    assert result.output["exit_code"] == 0


def test_worker_supervisor_task_runner_poll_maps_exit_one_to_failed(tmp_path: Path) -> None:
    runner = WorkerSupervisorTaskRunner(supervisor=WorkerSupervisor(projects_root=tmp_path))
    job = _job(tmp_path, "run-fail", _exit_command(1))

    runner.start(job)
    result = _wait_for_terminal(runner, job)

    assert result.state == "failed"
    assert result.message == "worker failed"
    assert result.output is not None
    assert result.output["exit_code"] == 1


def test_worker_supervisor_task_runner_cancel_uses_sigterm(tmp_path: Path) -> None:
    runner = WorkerSupervisorTaskRunner(supervisor=WorkerSupervisor(projects_root=tmp_path), stop_timeout_sec=2)
    job = _job(
        tmp_path,
        "run-sigterm",
        [
            sys.executable,
            "-c",
            "import signal, sys, time; "
            "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0)); "
            "time.sleep(60)",
        ],
    )
    runner.start(job)
    time.sleep(0.2)

    result = runner.cancel(job)

    assert result.state == "cancelled"
    assert result.output is not None
    assert result.output["status"] == "stopped"
    assert result.output["exit_code"] == 0


def test_worker_supervisor_task_runner_cancel_force_kills_after_timeout(tmp_path: Path) -> None:
    runner = WorkerSupervisorTaskRunner(supervisor=WorkerSupervisor(projects_root=tmp_path), stop_timeout_sec=0)
    job = _job(
        tmp_path,
        "run-sigkill",
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ],
    )
    runner.start(job)
    time.sleep(0.2)

    result = runner.cancel(job)

    assert result.state == "cancelled"
    assert result.output is not None
    assert result.output["status"] == "failed"
    assert result.output["exit_code"] < 0 or result.output["exit_code"] == 255


def test_worker_supervisor_task_runner_rejects_shell_string_command(tmp_path: Path) -> None:
    runner = WorkerSupervisorTaskRunner(supervisor=WorkerSupervisor(projects_root=tmp_path))
    job = _job(tmp_path, "run-shell-string", "echo unsafe")

    with pytest.raises(ValueError, match="job task command must be a list"):
        runner.start(job)
