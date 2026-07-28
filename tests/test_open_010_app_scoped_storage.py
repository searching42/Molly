from __future__ import annotations

from ai4s_agent.app import create_app


def _payload(project_id: str, run_id: str) -> dict:
    return {
        "project_id": project_id,
        "run_id": run_id,
        "goal": "Train a PLQY model.",
    }


def test_modeling_plan_route_uses_current_app_storage(tmp_path) -> None:
    workspace_a = tmp_path / "workspace_a"
    workspace_b = tmp_path / "workspace_b"
    app_a = create_app(base_runs_dir=workspace_a / "runs", workspace_dir=workspace_a)
    client_a = app_a.test_client()
    app_b = create_app(base_runs_dir=workspace_b / "runs", workspace_dir=workspace_b)
    client_b = app_b.test_client()

    resp_b = client_b.post("/api/agent/modeling-plan", json=_payload("project_b", "run_b"))
    assert resp_b.status_code == 200
    resp_a = client_a.post("/api/agent/modeling-plan", json=_payload("project_a", "run_a"))
    assert resp_a.status_code == 200

    assert (workspace_a / "projects" / "project_a" / "runs" / "run_a" / "modeling_plan_proposal.json").exists()
    assert not (workspace_b / "projects" / "project_a" / "runs" / "run_a" / "modeling_plan_proposal.json").exists()
    assert (workspace_b / "projects" / "project_b" / "runs" / "run_b" / "modeling_plan_proposal.json").exists()
