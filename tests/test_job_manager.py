from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import BackgroundJobBudget, RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


def test_job_lifecycle_start_pause_resume_stop(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    job = jm.start_job("r1", details={"gate": "gate_1"})
    assert job["status"] == RunStatus.RUNNING.value
    assert job["run_id"] == "r1"

    paused = jm.pause_job("r1")
    assert paused["status"] == RunStatus.PAUSED_BY_USER.value

    resumed = jm.resume_job("r1")
    assert resumed["status"] == RunStatus.RUNNING.value

    stopped = jm.stop_job("r1")
    assert stopped["status"] == RunStatus.CANCELLED.value
    with pytest.raises(KeyError):
        jm.pause_job("r1")


def test_job_complete_writes_log_to_disk(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")
    jm.add_log("r1", "WARN", "test", "something happened")
    jm.complete_job("r1", status=RunStatus.SUCCEEDED)

    log_path = tmp_path / "r1" / "job_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


def test_job_logs_added_after_stop_are_persisted(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")
    jm.stop_job("r1")
    jm.add_log("r1", "INFO", "gate", "approved after stop")

    logs = jm.get_logs("r1", limit=10)
    assert any(entry["message"] == "approved after stop" for entry in logs)
    log_path = tmp_path / "r1" / "job_log.jsonl"
    assert log_path.exists()
    assert "approved after stop" in log_path.read_text(encoding="utf-8")


def test_get_logs_merges_memory_and_disk(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")
    jm.add_log("r1", "INFO", "test", "hello")
    jm.save_job_log("r1")
    jm.add_log("r1", "INFO", "test", "world")

    logs = jm.get_logs("r1", limit=100)
    messages = [entry["message"] for entry in logs]
    assert "hello" in messages
    assert "world" in messages


def test_job_logs_escape_json_and_reject_run_path_traversal(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")
    jm.add_log("r1", "INFO", "test", 'quote " and newline\nnext')
    log_path = jm.save_job_log("r1")

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["message"] == 'quote " and newline\nnext'
    assert jm.get_logs("r1", limit=10)[-1]["message"] == 'quote " and newline\nnext'

    with pytest.raises(ValueError):
        jm.save_job_log("../escape")
    assert not (tmp_path.parent / "escape" / "job_log.jsonl").exists()


def test_list_jobs_returns_active_only(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")
    jm.start_job("r2")
    assert len(jm.list_jobs()) == 2
    jm.complete_job("r1", status=RunStatus.SUCCEEDED)
    assert len(jm.list_jobs()) == 1
    assert jm.list_jobs()[0]["run_id"] == "r2"


def test_start_job_rejects_duplicate_active_run_id(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_job("r1")

    with pytest.raises(ValueError, match="already active"):
        jm.start_job("r1")

    assert len(jm.list_jobs()) == 1


def test_complete_job_requires_existing_run(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    with pytest.raises(KeyError):
        jm.complete_job("ghost", status=RunStatus.SUCCEEDED)


def test_background_job_requires_explicit_budget_and_persists_state(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)

    with pytest.raises(ValueError, match="budget"):
        jm.start_background_job(
            "bg1",
            project_id="proj-bg",
            task_id="retrieve_evidence",
            budget=None,
        )

    job = jm.start_background_job(
        "bg1",
        project_id="proj-bg",
        task_id="retrieve_evidence",
        budget=BackgroundJobBudget(max_runtime_sec=3600, max_steps=10),
    )

    assert job["status"] == RunStatus.RUNNING.value
    assert job["budget"]["max_runtime_sec"] == 3600
    assert job["budget"]["max_steps"] == 10
    assert job["executable"] is False
    assert (tmp_path / "bg1" / "background_job_state.json").exists()

    restored = JobManager(runs_dir=tmp_path).get_background_job("bg1")
    assert restored is not None
    assert restored["run_id"] == "bg1"
    assert restored["project_id"] == "proj-bg"
    assert restored["task_id"] == "retrieve_evidence"


def test_background_job_checkpoint_and_resume_plan_are_persisted(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_background_job(
        "bg2",
        project_id="proj-bg",
        task_id="extract_records",
        budget=BackgroundJobBudget(max_runtime_sec=1800, max_steps=5),
    )

    checkpoint = jm.record_background_checkpoint(
        "bg2",
        stage="extract_records",
        cursor={"offset": 25, "source_id": "doi_abc"},
        completed_units=25,
        artifact_refs=["extracted_records_partial.json"],
    )
    resume = JobManager(runs_dir=tmp_path).background_resume_plan("bg2")

    assert checkpoint["stage"] == "extract_records"
    assert resume["resumable"] is True
    assert resume["executable"] is False
    assert resume["requires_confirmation"] is True
    assert resume["latest_checkpoint"]["cursor"] == {"offset": 25, "source_id": "doi_abc"}
    assert resume["resume_from_checkpoint_id"] == checkpoint["checkpoint_id"]


def test_background_job_resume_plan_blocks_after_budget_exhaustion(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_background_job(
        "bg3",
        project_id="proj-bg",
        task_id="retrieve_evidence",
        budget=BackgroundJobBudget(max_steps=1, max_records=10),
    )

    jm.record_background_checkpoint(
        "bg3",
        stage="retrieve_evidence",
        cursor={"query_index": 1},
        completed_units=10,
    )
    resume = jm.background_resume_plan("bg3")

    assert resume["budget_exhausted"] is True
    assert resume["resumable"] is False
    assert resume["consumed"]["steps"] == 1
    assert resume["consumed"]["records"] == 10


def test_background_job_runtime_and_cost_budgets_are_enforced(tmp_path: Path) -> None:
    jm = JobManager(runs_dir=tmp_path)
    jm.start_background_job(
        "bg-runtime",
        project_id="proj-bg",
        task_id="retrieve_evidence",
        budget=BackgroundJobBudget(max_runtime_sec=10, max_cost_usd=0.25),
    )

    checkpoint = jm.record_background_checkpoint(
        "bg-runtime",
        stage="retrieve_evidence",
        runtime_sec=10,
        cost_usd=0.1,
    )
    resume = jm.background_resume_plan("bg-runtime")

    assert checkpoint["runtime_sec"] == 10
    assert checkpoint["cost_usd"] == 0.1
    assert resume["budget_exhausted"] is True
    assert resume["resumable"] is False
    assert resume["consumed"]["runtime_sec"] == 10
    assert resume["consumed"]["cost_usd"] == 0.1


def test_background_job_budget_rejects_bool_and_empty_limits() -> None:
    with pytest.raises(ValueError):
        BackgroundJobBudget()
    with pytest.raises(ValueError):
        BackgroundJobBudget(max_runtime_sec=True)


def test_api_project_and_upload_endpoints(tmp_path: Path) -> None:
    from io import BytesIO

    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)

    with app.test_client() as client:
        resp = client.post("/api/projects", json={"project_id": "proj-a", "name": "Test Project"})
        assert resp.status_code == 200
        assert resp.json["project_id"] == "proj-a"
        assert (tmp_path / "projects" / "proj-a" / "project.json").exists()

        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert any(p["project_id"] == "proj-a" for p in resp.json["projects"])

        data = {
            "file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "test.csv"),
            "project_approved": "true",
        }
        resp = client.post("/api/projects/proj-a/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.json["filename"] == "test.csv"
        assert (tmp_path / "projects" / "proj-a" / "uploads" / "test.csv").exists()


def test_api_upload_requires_project_approval(tmp_path: Path) -> None:
    from io import BytesIO

    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)

    with app.test_client() as client:
        client.post("/api/projects", json={"project_id": "proj-a"})
        data = {"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "test.csv")}
        resp = client.post("/api/projects/proj-a/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 403
        assert resp.json["permission"]["level"] == "project-approved"


def test_api_project_upload_rejects_project_path_traversal(tmp_path: Path) -> None:
    from io import BytesIO

    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)

    with app.test_client() as client:
        resp = client.post("/api/projects", json={"project_id": "../escape"})
        assert resp.status_code == 400

        data = {
            "file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "test.csv"),
            "project_approved": "true",
        }
        resp = client.post("/api/projects/%2E%2E/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
    assert not (tmp_path.parent / "escape").exists()


def test_api_job_control_endpoints(tmp_path: Path) -> None:
    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)

    with app.test_client() as client:
        client.post("/api/plan", json={"run_id": "r1", "prompt": "test"})

        resp = client.post("/api/runs/r1/pause")
        assert resp.status_code == 200
        assert resp.json["job"]["status"] == "PAUSED_BY_USER"

        resp = client.post("/api/runs/r1/resume")
        assert resp.status_code == 200
        assert resp.json["job"]["status"] == "RUNNING"

        resp = client.get("/api/runs/r1/logs?limit=20")
        assert resp.status_code == 200
        assert len(resp.json["logs"]) >= 3

        resp = client.post("/api/runs/r1/retry", json={"stage": "clean_dataset"})
        assert resp.status_code == 409

        client.post("/api/runs/r1/stop")
        storage = ProjectStorage(workspace_dir=tmp_path)
        storage.write_stage_state(
            "proj-a",
            "r1",
            StageState(
                stage="clean_dataset",
                status=RunStatus.FAILED,
                started_at=now_iso(),
                updated_at=now_iso(),
                error={"category": "DATA", "reason": "bad csv", "retryable": True},
            ),
        )
        resp = client.post(
            "/api/runs/r1/retry",
            json={"project_id": "proj-a", "stage": "clean_dataset"},
        )
        assert resp.status_code == 200
        assert resp.json["job"]["status"] == "RUNNING"
        assert resp.json["retry_stage"] == "clean_dataset"
        stage = storage.read_stage_state("proj-a", "r1")
        assert stage is not None
        assert stage.status == RunStatus.PENDING

        resp = client.post("/api/runs/r1/stop")
        assert resp.status_code == 200

        resp = client.post("/api/runs/r1/pause")
        assert resp.status_code == 404


def test_api_upload_rejects_non_csv(tmp_path: Path) -> None:
    from io import BytesIO

    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)

    with app.test_client() as client:
        client.post("/api/projects", json={"project_id": "proj-a"})
        data = {"file": (BytesIO(b"not allowed"), "malware.exe"), "project_approved": "true"}
        resp = client.post("/api/projects/proj-a/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


def test_api_retry_rejects_non_retryable_or_non_latest_failed_stage(tmp_path: Path) -> None:
    from ai4s_agent.app import create_app

    app = create_app(base_runs_dir=tmp_path)
    storage = ProjectStorage(workspace_dir=tmp_path)

    with app.test_client() as client:
        client.post("/api/plan", json={"run_id": "r1", "prompt": "test"})
        client.post("/api/runs/r1/stop")

        storage.write_stage_state(
            "proj-a",
            "r1",
            StageState(
                stage="train_model",
                status=RunStatus.FAILED,
                started_at=now_iso(),
                updated_at=now_iso(),
                error={"category": "MODEL", "reason": "bad config", "retryable": False},
            ),
        )
        resp = client.post(
            "/api/runs/r1/retry",
            json={"project_id": "proj-a", "stage": "train_model"},
        )
        assert resp.status_code == 409
        assert "not retryable" in resp.json["error"]

        storage.write_stage_state(
            "proj-a",
            "r1",
            StageState(
                stage="train_model",
                status=RunStatus.FAILED,
                started_at=now_iso(),
                updated_at=now_iso(),
                error={"category": "MODEL", "reason": "transient", "retryable": True},
            ),
        )
        resp = client.post(
            "/api/runs/r1/retry",
            json={"project_id": "proj-a", "stage": "clean_dataset"},
        )
        assert resp.status_code == 400
        assert "latest failed stage" in resp.json["error"]
