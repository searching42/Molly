from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import BackgroundJobBudget, RunStatus


def test_project_scoped_jobs_allow_same_run_id_in_different_projects(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)

    job_a = manager.start_project_job("project_a", "shared-run", details={"stage": "train"})
    job_b = manager.start_project_job("project_b", "shared-run", details={"stage": "parse"})

    assert job_a["job_key"] == {"project_id": "project_a", "run_id": "shared-run"}
    assert job_b["job_key"] == {"project_id": "project_b", "run_id": "shared-run"}
    assert job_a["details"] == {"stage": "train"}
    assert job_b["details"] == {"stage": "parse"}
    assert (tmp_path / "projects" / "project_a" / "runs" / "shared-run" / "job_state.json").exists()
    assert (tmp_path / "projects" / "project_b" / "runs" / "shared-run" / "job_state.json").exists()
    assert not (tmp_path / "shared-run" / "job_state.json").exists()


def test_project_scoped_job_state_survives_restart_and_lists_by_project(tmp_path: Path) -> None:
    JobManager(runs_dir=tmp_path).start_project_job("project_a", "run-1")
    JobManager(runs_dir=tmp_path).start_project_job("project_a", "run-2")
    JobManager(runs_dir=tmp_path).start_project_job("project_b", "run-1")
    JobManager(runs_dir=tmp_path).complete_project_job("project_a", "run-2", status=RunStatus.SUCCEEDED)

    restored = JobManager(runs_dir=tmp_path)
    active_a = restored.list_project_jobs("project_a")
    active_all = restored.list_project_jobs()

    assert [job["run_id"] for job in active_a] == ["run-1"]
    assert {(job["project_id"], job["run_id"]) for job in active_all} == {
        ("project_a", "run-1"),
        ("project_b", "run-1"),
    }
    terminal = restored.read_project_job_state("project_a", "run-2")
    assert terminal is not None
    assert terminal["status"] == RunStatus.SUCCEEDED.value


def test_project_scoped_jobs_reject_duplicate_active_key_only_within_project(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    manager.start_project_job("project_a", "same-run")
    manager.start_project_job("project_b", "same-run")

    with pytest.raises(ValueError, match="already active"):
        manager.start_project_job("project_a", "same-run")


def test_project_scoped_background_jobs_do_not_collide_on_run_id(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)
    budget = BackgroundJobBudget(max_steps=3)

    bg_a = manager.start_project_background_job("project_a", "shared-bg", task_id="retrieve_evidence", budget=budget)
    bg_b = manager.start_project_background_job("project_b", "shared-bg", task_id="extract_records", budget=budget)

    assert bg_a["project_id"] == "project_a"
    assert bg_a["run_id"] == "shared-bg"
    assert bg_a["details"]["job_key"] == {"project_id": "project_a", "run_id": "shared-bg"}
    assert bg_b["project_id"] == "project_b"
    assert bg_b["task_id"] == "extract_records"
    assert (tmp_path / "projects" / "project_a" / "runs" / "shared-bg" / "background_job_state.json").exists()
    assert (tmp_path / "projects" / "project_b" / "runs" / "shared-bg" / "background_job_state.json").exists()
    assert not (tmp_path / "shared-bg" / "background_job_state.json").exists()


def test_project_scoped_job_ids_reject_path_segments(tmp_path: Path) -> None:
    manager = JobManager(runs_dir=tmp_path)

    with pytest.raises(ValueError, match="project_id"):
        manager.start_project_job("nested/project", "run")
    with pytest.raises(ValueError, match="run_id"):
        manager.start_project_job("project", "nested/run")
    with pytest.raises(ValueError, match="project_id"):
        manager.start_project_job("..", "run")
    with pytest.raises(ValueError, match="run_id"):
        manager.start_project_job("project", "..")
    with pytest.raises(ValueError, match="project_id"):
        manager.start_project_background_job("..", "run", task_id="retrieve_evidence", budget=BackgroundJobBudget(max_steps=1))

    assert not (tmp_path / "job_state.json").exists()
    assert not (tmp_path.parent / "runs" / "job_state.json").exists()
