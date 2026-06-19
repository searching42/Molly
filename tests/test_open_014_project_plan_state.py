from __future__ import annotations

from ai4s_agent._utils import now_iso
from ai4s_agent.app import create_app
from ai4s_agent.schemas import RunStatus, StageHistoryItem, StageState
from ai4s_agent.storage import ProjectStorage


def test_project_plan_state_allows_same_run_id_across_projects(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    first = client.post("/api/plan", json={"project_id": "project_a", "run_id": "shared-plan", "prompt": "Train A."})
    second = client.post("/api/plan", json={"project_id": "project_b", "run_id": "shared-plan", "prompt": "Train B."})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json["plan_scope"] == "project"
    assert second.json["plan_scope"] == "project"
    assert first.json["job_key"] == {"project_id": "project_a", "run_id": "shared-plan"}
    assert second.json["job_key"] == {"project_id": "project_b", "run_id": "shared-plan"}
    assert (tmp_path / "projects" / "project_a" / "runs" / "shared-plan" / "plan.json").exists()
    assert (tmp_path / "projects" / "project_b" / "runs" / "shared-plan" / "plan.json").exists()
    assert not (tmp_path / "runs" / "shared-plan" / "plan.json").exists()

    status_a = client.get("/api/projects/project_a/runs/shared-plan/status")
    status_b = client.get("/api/projects/project_b/runs/shared-plan/status")
    assert status_a.status_code == 200
    assert status_b.status_code == 200
    assert status_a.json["status"]["plan_exists"] is True
    assert status_b.json["status"]["plan_exists"] is True
    assert status_a.json["status"]["plan_scope"] == "project"


def test_project_gate_approval_stays_in_project_namespace(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    assert client.post("/api/plan", json={"project_id": "project_a", "run_id": "shared-gate", "prompt": "Train A."}).status_code == 200
    assert client.post("/api/plan", json={"project_id": "project_b", "run_id": "shared-gate", "prompt": "Train B."}).status_code == 200

    approved = client.post(
        "/api/gates/approve",
        json={"project_id": "project_a", "run_id": "shared-gate", "gate": "gate_1_task_parse", "actor": "alice"},
    )
    assert approved.status_code == 200
    assert approved.json["plan_scope"] == "project"
    assert approved.json["next_gate"] == "gate_2_data_mining"

    status_a = client.get("/api/projects/project_a/runs/shared-gate/status")
    status_b = client.get("/api/projects/project_b/runs/shared-gate/status")
    assert len(status_a.json["status"]["gate_decisions"]) == 1
    assert status_a.json["status"]["gate_decisions"][0]["actor"] == "alice"
    assert status_b.json["status"]["gate_decisions"] == []
    assert not (tmp_path / "runs" / "shared-gate" / "gate_decisions.json").exists()


def test_project_retry_uses_project_state_not_legacy_plan(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    assert client.post("/api/plan", json={"project_id": "project_a", "run_id": "project-retry", "prompt": "Train."}).status_code == 200
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
