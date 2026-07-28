from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from ai4s_agent.worker_supervisor import WorkerSupervisor, _process_alive, _safe_component


def _sleep_command(seconds: int) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _exit_command(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def test_supervisor_lifecycle_pending_to_running_to_stopped(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    heartbeat = supervisor.start(
        project_id="proj-ws",
        run_id="run-lifecycle",
        command=_sleep_command(2),
        cwd=tmp_path,
    )
    assert heartbeat.status == "running"
    assert heartbeat.pid > 0
    assert heartbeat.command == _sleep_command(2)

    alive = supervisor.status("proj-ws", "run-lifecycle")
    assert alive.status == "running"
    assert alive.pid == heartbeat.pid
    assert alive.command == _sleep_command(2)

    time.sleep(2.5)

    done = supervisor.status("proj-ws", "run-lifecycle")
    assert done.status == "stopped"
    assert done.exit_code == 0

    # heartbeat persisted to disk
    hb_path = tmp_path / "proj-ws" / "runs" / "run-lifecycle" / "worker_heartbeat.json"
    assert hb_path.exists()
    import json

    persisted = json.loads(hb_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "stopped"
    assert persisted["exit_code"] == 0


def test_supervisor_lifecycle_failed_process(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    heartbeat = supervisor.start(
        project_id="proj-ws",
        run_id="run-fail",
        command=_exit_command(1),
        cwd=tmp_path,
    )
    assert heartbeat.status == "running"
    assert heartbeat.pid > 0

    time.sleep(1)

    done = supervisor.status("proj-ws", "run-fail")
    assert done.status == "failed"
    assert done.exit_code == 1


def test_supervisor_stop_sends_sigterm_then_kill(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    # Worker exits cleanly on SIGTERM
    heartbeat = supervisor.start(
        project_id="proj-ws",
        run_id="run-stop",
        command=[
            sys.executable, "-c",
            "import signal, sys; "
            "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0)); "
            "import time; time.sleep(60)",
        ],
        cwd=tmp_path,
    )
    assert heartbeat.status == "running"

    # Give the subprocess a moment to install its SIGTERM handler
    time.sleep(0.2)

    stopped = supervisor.stop("proj-ws", "run-stop", timeout_sec=5)
    assert stopped.status == "stopped"
    assert stopped.exit_code == 0


def test_supervisor_stop_force_kill_after_timeout(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    heartbeat = supervisor.start(
        project_id="proj-ws",
        run_id="run-force-kill",
        command=[
            sys.executable, "-c",
            "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "import time; time.sleep(60)",
        ],
        cwd=tmp_path,
    )
    assert heartbeat.status == "running"

    stopped = supervisor.stop("proj-ws", "run-force-kill", timeout_sec=1)
    assert stopped.status == "failed"
    # SIGKILL (signal 9) on Unix; negative exit indicates signal death
    assert stopped.exit_code is not None
    assert stopped.exit_code < 0 or stopped.exit_code == 255


def test_supervisor_rejects_duplicate_start_for_running_worker(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    supervisor.start(
        project_id="proj-ws",
        run_id="run-dup",
        command=_sleep_command(30),
        cwd=tmp_path,
    )

    with pytest.raises(ValueError, match="worker already running"):
        supervisor.start(
            project_id="proj-ws",
            run_id="run-dup",
            command=_sleep_command(30),
            cwd=tmp_path,
        )

    supervisor.stop("proj-ws", "run-dup", timeout_sec=1)


def test_supervisor_status_of_unknown_worker_returns_pending(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    status = supervisor.status("proj-ws", "run-unknown")
    assert status.status == "pending"
    assert status.pid == -1
    # status is read-only and must not create directories
    unknown_run_dir = tmp_path / "proj-ws" / "runs" / "run-unknown"
    assert not unknown_run_dir.exists()


def test_supervisor_rejects_empty_project_or_run_id(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    with pytest.raises(ValueError, match="project_id must not be empty"):
        supervisor.start(project_id="", run_id="r", command=["echo"])

    with pytest.raises(ValueError, match="run_id must not be empty"):
        supervisor.start(project_id="p", run_id="   ", command=["echo"])

    with pytest.raises(ValueError, match="project_id must not be empty"):
        supervisor.status("", "r")

    with pytest.raises(ValueError, match="run_id must not be empty"):
        supervisor.status("p", "")


def test_supervisor_rejects_path_traversal_project_or_run_id(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    for bad_id in ("..", ".", "a/b", "a\\b", "../escape", "x/../y"):
        with pytest.raises(ValueError):
            supervisor.start(project_id=bad_id, run_id="r", command=["echo"])
        with pytest.raises(ValueError):
            supervisor.start(project_id="p", run_id=bad_id, command=["echo"])
        # status and stop should also reject
        with pytest.raises(ValueError):
            supervisor.status(bad_id, "r")
        with pytest.raises(ValueError):
            supervisor.status("p", bad_id)


def test_safe_component_rejects_reserved_and_traversal_names() -> None:
    for bad in ("..", ".", "a/b", "a\\b", "../evil"):
        with pytest.raises(ValueError):
            _safe_component(bad, "test")
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            _safe_component(bad, "test")
    # Valid names pass
    assert _safe_component("proj-a", "project_id") == "proj-a"
    assert _safe_component("run_001", "run_id") == "run_001"


def test_supervisor_list_workers(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    # Create a heartbeat file for a worker that already completed
    hb_dir = tmp_path / "proj-ws" / "runs" / "run-existing"
    hb_dir.mkdir(parents=True)
    import json

    hb_dir.joinpath("worker_heartbeat.json").write_text(
        json.dumps({"run_id": "run-existing", "pid": 99999, "status": "stopped", "started_at": "", "command": ["echo", "done"], "cwd": str(tmp_path), "exit_code": 0, "updated_at": ""}),
        encoding="utf-8",
    )

    all_workers = supervisor.list_workers("proj-ws")
    assert len(all_workers) == 1
    assert all_workers[0].run_id == "run-existing"
    assert all_workers[0].status == "stopped"


def test_supervisor_status_detects_stale_pid_as_failed(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    # Write a heartbeat with a PID that definitely does not exist
    hb_dir = tmp_path / "proj-ws" / "runs" / "run-stale"
    hb_dir.mkdir(parents=True)
    import json

    hb_dir.joinpath("worker_heartbeat.json").write_text(
        json.dumps({"run_id": "run-stale", "pid": 99999, "status": "running", "started_at": "", "command": ["echo"], "cwd": str(tmp_path), "exit_code": None, "updated_at": ""}),
        encoding="utf-8",
    )

    status = supervisor.status("proj-ws", "run-stale")
    assert status.status == "failed"
    assert status.exit_code == -1


def test_supervisor_allows_same_run_id_in_different_projects(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(projects_root=tmp_path)

    first = supervisor.start(
        project_id="proj-a",
        run_id="run-shared",
        command=_sleep_command(30),
        cwd=tmp_path,
    )
    second = supervisor.start(
        project_id="proj-b",
        run_id="run-shared",
        command=_sleep_command(30),
        cwd=tmp_path,
    )

    assert first.status == "running"
    assert second.status == "running"
    assert first.pid != second.pid

    supervisor.stop("proj-a", "run-shared", timeout_sec=1)
    supervisor.stop("proj-b", "run-shared", timeout_sec=1)

    # After stopping proj-a, proj-b worker should still be present on disk
    hb_b = tmp_path / "proj-b" / "runs" / "run-shared" / "worker_heartbeat.json"
    assert hb_b.exists()


def test_process_alive_helper() -> None:
    assert _process_alive(-1) is False
    assert _process_alive(99999) is False  # unlikely to exist
