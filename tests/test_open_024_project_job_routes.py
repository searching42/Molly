from __future__ import annotations

from ai4s_agent.app import create_app


def test_plan_route_uses_project_scoped_job_key_when_project_id_is_present(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/plan",
        json={"project_id": "project_a", "run_id": "route24-plan", "prompt": "Train a PLQY model."},
    )

    assert response.status_code == 200
    assert response.json["job_key"] == {"project_id": "project_a", "run_id": "route24-plan"}
    assert response.json["job"]["project_scoped"] is True
    assert (tmp_path / "runs" / "projects" / "project_a" / "runs" / "route24-plan" / "job_state.json").exists()
    assert not (tmp_path / "runs" / "route24-plan" / "job_state.json").exists()

    listed = client.get("/api/jobs?project_id=project_a")
    assert listed.status_code == 200
    assert listed.json["jobs"][0]["job_key"] == {"project_id": "project_a", "run_id": "route24-plan"}


def test_project_scoped_log_and_job_control_routes(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    assert client.post(
        "/api/plan",
        json={"project_id": "project_a", "run_id": "route24-control", "prompt": "Train a model."},
    ).status_code == 200

    logs = client.get("/api/projects/project_a/runs/route24-control/logs")
    assert logs.status_code == 200
    assert logs.json["job_key"] == {"project_id": "project_a", "run_id": "route24-control"}
    assert any(entry["source"] == "job_started" for entry in logs.json["logs"])

    paused = client.post("/api/projects/project_a/runs/route24-control/pause")
    assert paused.status_code == 200
    assert paused.json["job"]["status"] == "paused_by_user"
    resumed = client.post("/api/runs/route24-control/resume", json={"project_id": "project_a"})
    assert resumed.status_code == 200
    assert resumed.json["job"]["status"] == "running"
    stopped = client.post("/api/runs/route24-control/stop?project_id=project_a")
    assert stopped.status_code == 200
    assert stopped.json["job"]["status"] == "cancelled"


def test_background_job_routes_use_project_scoped_keys_when_project_id_is_present(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    first = client.post(
        "/api/background-jobs",
        json={"project_id": "project_a", "run_id": "shared-bg", "task_id": "retrieve_evidence", "budget": {"max_steps": 3}},
    )
    second = client.post(
        "/api/background-jobs",
        json={"project_id": "project_b", "run_id": "shared-bg", "task_id": "extract_records", "budget": {"max_steps": 3}},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json["job"]["details"]["job_key"] == {"project_id": "project_a", "run_id": "shared-bg"}
    assert second.json["job"]["details"]["job_key"] == {"project_id": "project_b", "run_id": "shared-bg"}
    assert not (tmp_path / "runs" / "shared-bg" / "background_job_state.json").exists()

    checkpoint = client.post(
        "/api/background-jobs/shared-bg/checkpoints",
        json={"project_id": "project_a", "stage": "retrieval", "completed_units": 10, "runtime_sec": 2},
    )
    assert checkpoint.status_code == 200
    assert checkpoint.json["checkpoint"]["checkpoint_id"] == "ckpt-project_a-shared-bg-001"

    resume_plan = client.get("/api/background-jobs/shared-bg/resume-plan?project_id=project_a")
    assert resume_plan.status_code == 200
    assert resume_plan.json["resume_plan"]["job_key"] == {"project_id": "project_a", "run_id": "shared-bg"}
    assert resume_plan.json["resume_plan"]["latest_checkpoint"]["completed_units"] == 10

    fetched = client.get("/api/background-jobs/shared-bg?project_id=project_b")
    assert fetched.status_code == 200
    assert fetched.json["job"]["task_id"] == "extract_records"
    assert fetched.json["job"]["checkpoints"] == []
