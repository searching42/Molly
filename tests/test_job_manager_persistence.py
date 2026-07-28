from pathlib import Path

import pytest

from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import RunStatus


def test_active_job_state_survives_manager_restart(tmp_path: Path) -> None:
    first = JobManager(runs_dir=tmp_path)
    started = first.start_job("persisted-run", details={"stage": "train_model"})

    second = JobManager(runs_dir=tmp_path)
    restored = second.get_job("persisted-run")

    assert restored is not None
    assert restored["run_id"] == "persisted-run"
    assert restored["status"] == RunStatus.RUNNING.value
    assert restored["attempt"] == 1
    assert restored["details"] == {"stage": "train_model"}
    assert restored["durable_state"] is True
    assert (tmp_path / "persisted-run" / "job_state.json").exists()
    assert restored["started_at"] == started["started_at"]


def test_pause_resume_and_complete_work_after_restart(tmp_path: Path) -> None:
    JobManager(runs_dir=tmp_path).start_job("restart-control")

    paused = JobManager(runs_dir=tmp_path).pause_job("restart-control")
    assert paused["status"] == RunStatus.PAUSED_BY_USER.value

    resumed = JobManager(runs_dir=tmp_path).resume_job("restart-control")
    assert resumed["status"] == RunStatus.RUNNING.value

    completed = JobManager(runs_dir=tmp_path).complete_job(
        "restart-control",
        status=RunStatus.SUCCEEDED,
    )
    assert completed["status"] == RunStatus.SUCCEEDED.value

    final_manager = JobManager(runs_dir=tmp_path)
    assert final_manager.get_job("restart-control") is None
    final_state = final_manager.read_job_state("restart-control")
    assert final_state is not None
    assert final_state["status"] == RunStatus.SUCCEEDED.value
    assert [item["event"] for item in final_state["history"]] == [
        "started",
        "paused",
        "resumed",
        "completed",
    ]


def test_duplicate_active_job_is_rejected_after_restart(tmp_path: Path) -> None:
    JobManager(runs_dir=tmp_path).start_job("duplicate-run")

    with pytest.raises(ValueError, match="already active"):
        JobManager(runs_dir=tmp_path).start_job("duplicate-run")


def test_terminal_job_can_restart_as_new_attempt_without_losing_history(tmp_path: Path) -> None:
    first = JobManager(runs_dir=tmp_path)
    first.start_job("retry-run")
    first.stop_job("retry-run")

    restarted = JobManager(runs_dir=tmp_path).start_job(
        "retry-run",
        details={"retry_stage": "clean_dataset"},
    )

    assert restarted["status"] == RunStatus.RUNNING.value
    assert restarted["attempt"] == 2
    assert restarted["details"] == {"retry_stage": "clean_dataset"}
    assert [item["event"] for item in restarted["history"]] == [
        "started",
        "cancelled",
        "restarted",
    ]


def test_list_jobs_restores_active_jobs_only(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    manager.start_job("active-a")
    manager.start_job("active-b")
    manager.start_job("finished")
    manager.complete_job("finished", status=RunStatus.SUCCEEDED)

    restored = JobManager(runs_dir=tmp_path).list_jobs()

    assert [job["run_id"] for job in restored] == ["active-a", "active-b"]


@pytest.mark.parametrize("run_id", ["", "nested/run", r"nested\run", "../escape"])
def test_job_state_rejects_non_segment_run_ids(tmp_path: Path, run_id: str) -> None:
    manager = JobManager(runs_dir=tmp_path)

    with pytest.raises(ValueError, match="single path segment"):
        manager.start_job(run_id)
