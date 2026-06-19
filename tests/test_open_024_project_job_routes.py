from __future__ import annotations

from ai4s_agent._utils import now_iso
from ai4s_agent.app import create_app
from ai4s_agent.schemas import RunStatus, StageHistoryItem, StageState
from ai4s_agent.storage import ProjectStorage


def test_plan_route_uses_project_scoped_job_key_when_project_id_is_present(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/plan",
        json={"project_id": "project_a", "run_id": "route24-plan", "prompt": "Train a PLQY model."},
    )

    assert response.status_code == 200
    assert response.json["plan_scope"] == "project"
    assert response.json["job_key"] == {"project_id": "project_a", "run_id": "route24-plan"}
    assert response.json["job"]["project_scoped"] is True
    assert (tmp_path / "projects" / "project_a" / "runs" / "route24-plan" / "plan.json").exists()
    assert (tmp_path / "runs" / "projects" / "project_a" / "runs" / "route24-plan" / "job_state.json").exists()
    assert not (tmp_path / "runs" / "route24-plan" / "plan.json").exists()

    listed = client.get("/api/jobs?project_id=project_a")
    assert listed.status_code == 200
    assert listed.json["jobs"][0]["job_key"] == {"project_id": "project_a", "run_id": "route24-plan"}


def test_project_plan_route_rejects_invalid_project_key_before_any_plan_write(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/plan",
        json={"project_id": "..", "run_id": "route24-invalid-project", "prompt": "Train a model."},
    )

    assert response.status_code == 400
    assert "project_id" in response.json["error"]
    assert not (tmp_path / "runs" / "route24-invalid-project" / "plan.json").exists()
    assert not (tmp_path / "projects" / ".." / "runs" / "route24-invalid-project" / "plan.json").exists()
    assert not (tmp_path / "runs" / "projects" / ".." / "runs" / "route24-invalid-project" / "job_state.json").exists()


def test_project_plan_route_allows_same_run_id_in_different_projects(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    first = client.post(
        "/api/plan",
        json={"project_id": "project_a", "run_id": "shared-plan", "prompt": "Train first model."},
    )
    second = client.post(
        "/api/plan",
        json={"project_id": "project_b", "run_id": "shared-plan", "prompt": "Train second model."},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json["job_key"] == {"project_id": "project_a", "run_id": "shared-plan"}
    assert second.json["job_key"] == {"project_id": "project_b", "run_id": "shared-plan"}
    assert not (tmp_path / "runs" / "shared-plan" / "plan.json").exists()
    assert (tmp_path / "projects" / "project_a" / "runs" / "shared-plan" / "plan.json").exists()
    assert (tmp_path / "projects" / "project_b" / "runs" / "shared-plan" / "plan.json").exists()
    assert (tmp_path / "runs" / "projects" / "project_a" / "runs" / "shared-plan" / "job_state.json").exists()
    assert (tmp_path / "runs" / "projects" / "project_b" / "runs" / "shared-plan" / "job_state.json").exists()

    status_a = client.get("/api/projects/project_a/runs/shared-plan/status")
    status_b = client.get("/api/projects/project_b/runs/shared-plan/status")
    assert status_a.status_code == 200
    assert status_b.status_code == 200
    assert status_a.json["status"]["plan_exists"] is True
    assert status_b.json["status"]["plan_exists"] is True
    assert status_a.json["status"]["plan_scope"] == "project"
    assert status_b.json["status"]["plan_scope"] == "project"


def test_project_gate_approval_does_not_write_or_read_legacy_namespace(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    assert client.post("/api/plan", json={"project_id": "project_a", "run_id": "shared-gate", "prompt": "Train A."}).status_code == 200
    assert client.post("/api/plan", json={"project_id": "project_b", "run_id": "shared-gate", "prompt": "Train B."}).status_code == 200

    approved_a = client.post(
        "/api/gates/approve",
        json={"project_id": "project_a", "run_id": "shared-gate", "gate": "gate_1_task_parse", "actor": "alice"},
    )
    assert approved_a.status_code == 200
    assert approved_a.json["plan_scope"] == "project"
    assert approved_a.json["next_gate"] == "gate_2_data_mining"

    status_a = client.get("/api/projects/project_a/runs/shared-gate/status")
    status_b = client.get("/api/projects/project_b/runs/shared-gate/status")
    assert len(status_a.json["status"]["gate_decisions"]) == 1
    assert status_a.json["status"]["gate_decisions"][0]["actor"] == "alice"
    assert status_b.json["status"]["gate_decisions"] == []
    assert not (tmp_path / "runs" / "shared-gate" / "gate_decisions.json").exists()

    approved_b = client.post(
        "/api/gates/approve",
        json={"project_id": "project_b", "run_id": "shared-gate", "gate": "gate_1_task_parse", "actor": "bob"},
    )
    assert approved_b.status_code == 200
    status_b_after = client.get("/api/projects/project_b/runs/shared-gate/status")
    assert status_b_after.json["status"]["gate_decisions"][0]["actor"] == "bob"


def test_project_retry_reads_project_plan_status_not_legacy_orchestrator(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    assert client.post(
        "/api/plan",
        json={"project_id": "project_a", "run_id": "project-retry", "prompt": "Train model."},
    ).status_code == 200
    assert not (tmp_path / "runs" / "project-retry" / "plan.json").exists()
    assert client.post("/api/projects/project_a/runs/project-retry/stop").status_code == 200

    now = now_iso()
    ProjectStorage(workspace_dir=tmp_path).write_stage_state(
        "project_a",
        "project-retry",
        StageState(
            stage="train_model",
            status=RunStatus.FAILED,
            started_at=now,
            updated_at=now,
            error={"retryable": True},
            details={},
            history=[StageHistoryItem(stage="train_model", status=RunStatus.FAILED, updated_at=now, note="failed")],
        ),
    )

    retry = client.post("/api/runs/project-retry/retry", json={"project_id": "project_a"})
    assert retry.status_code == 200
    assert retry.json["job_key"] == {"project_id": "project_a", "run_id": "project-retry"}
    assert retry.json["retry_stage"] == "train_model"
    state = ProjectStorage(workspace_dir=tmp_path).read_stage_state("project_a", "project-retry")
    assert state is not None
    assert state.status == RunStatus.PENDING


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
    assert paused.json["job"]["status"] == "PAUSED_BY_USER"
    resumed = client.post("/api/runs/route24-control/resume", json={"project_id": "project_a"})
    assert resumed.status_code == 200
    assert resumed.json["job"]["status"] == "RUNNING"
    stopped = client.post("/api/runs/route24-control/stop?project_id=project_a")
    assert stopped.status_code == 200
    assert stopped.json["job"]["status"] == "CANCELLED"


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


def test_legacy_background_checkpoint_falls_back_to_single_project_match(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    created = client.post(
        "/api/background-jobs",
        json={"project_id": "project_a", "run_id": "single-bg", "task_id": "retrieve_evidence", "budget": {"max_steps": 3}},
    )
    assert created.status_code == 200
    checkpoint = client.post("/api/background-jobs/single-bg/checkpoints", json={"stage": "retrieval", "completed_units": 1})
    assert checkpoint.status_code == 200
    resume_plan = client.get("/api/background-jobs/single-bg/resume-plan")
    assert resume_plan.status_code == 200
    assert resume_plan.json["resume_plan"]["job_key"] == {"project_id": "project_a", "run_id": "single-bg"}
