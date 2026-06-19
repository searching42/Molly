from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import RunStatus


def test_worker_lease_survives_manager_restart_and_heartbeats(tmp_path: Path) -> None:
    first = JobManager(runs_dir=tmp_path)
    first.start_job("lease-run")
    leased = first.acquire_worker_lease(
        "lease-run",
        worker_id="worker-a",
        lease_ttl_sec=60,
        external_task_id="slurm-123",
    )

    lease_id = leased["worker_lease"]["lease_id"]
    restored = JobManager(runs_dir=tmp_path).read_job_state("lease-run")
    assert restored is not None
    assert restored["durable_worker_control"] is True
    assert restored["worker_lease"]["worker_id"] == "worker-a"
    assert restored["worker_lease"]["external_task_id"] == "slurm-123"
    assert restored["external_task_id"] == "slurm-123"

    heartbeat = JobManager(runs_dir=tmp_path).record_worker_heartbeat(
        "lease-run",
        worker_id="worker-a",
        lease_id=lease_id,
        lease_ttl_sec=120,
    )
    assert heartbeat["worker_lease"]["lease_id"] == lease_id
    assert heartbeat["worker_lease"]["ttl_sec"] == 120
    assert heartbeat["worker_lease"]["heartbeat_at"] == heartbeat["heartbeat_at"]


def test_worker_lease_rejects_competing_worker_until_stale(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    manager.start_job("compete-run")
    manager.acquire_worker_lease("compete-run", worker_id="worker-a", lease_ttl_sec=300)

    with pytest.raises(ValueError, match="already held"):
        manager.acquire_worker_lease("compete-run", worker_id="worker-b", lease_ttl_sec=300)

    state = manager.read_job_state("compete-run")
    assert state is not None
    state["worker_lease"]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat().replace("+00:00", "Z")
    manager._write_job_state("compete-run", state)

    stale = manager.list_stale_worker_leases()
    assert stale[0]["run_id"] == "compete-run"
    takeover = manager.acquire_worker_lease("compete-run", worker_id="worker-b", lease_ttl_sec=300)
    assert takeover["worker_lease"]["worker_id"] == "worker-b"


def test_cancel_request_is_observable_without_immediate_terminal_transition(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    manager.start_job("cancel-run")
    leased = manager.acquire_worker_lease("cancel-run", worker_id="worker-a")

    cancelled = JobManager(runs_dir=tmp_path).request_cancel(
        "cancel-run",
        actor="alice",
        reason="stop expensive remote task",
    )
    assert cancelled["status"] == RunStatus.RUNNING.value
    assert cancelled["cancel_requested"] is True
    assert cancelled["cancellation"]["requested_by"] == "alice"

    should_stop = JobManager(runs_dir=tmp_path).worker_should_stop(
        "cancel-run",
        worker_id="worker-a",
        lease_id=leased["worker_lease"]["lease_id"],
    )
    assert should_stop == {"run_id": "cancel-run", "should_stop": True, "reason": "cancel_requested"}


def test_worker_should_stop_when_lease_lost_or_job_terminal(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    manager.start_job("lost-run")
    manager.acquire_worker_lease("lost-run", worker_id="worker-a")

    lost = manager.worker_should_stop("lost-run", worker_id="worker-b")
    assert lost["should_stop"] is True
    assert lost["reason"] == "lease_lost"

    manager.release_worker_lease("lost-run", worker_id="worker-a")
    released = manager.read_job_state("lost-run")
    assert released is not None
    assert released["worker_lease"] == {}
    manager.complete_job("lost-run", status=RunStatus.SUCCEEDED)

    terminal = JobManager(runs_dir=tmp_path).worker_should_stop("lost-run", worker_id="worker-a")
    assert terminal["should_stop"] is True
    assert terminal["reason"] == "job_not_active"
