from __future__ import annotations

from pathlib import Path

from ai4s_agent.app import create_app
from ai4s_agent._utils import write_json
from ai4s_agent.storage import ProjectStorage


def test_conversation_next_turn_infers_available_inputs_from_project_property_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_id = "proj-chat-context"
    run_id = "run-chat-context"
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    catalog_path = write_json(
        run_dir / "property_catalog.json",
        {
            "properties": [
                {"property_id": "delayed_fluorescence_lifetime", "source_column": "tau_df_us"},
                {"property_id": "plqy", "source_column": "PLQY (%)"},
            ]
        },
    )
    storage.register_artifact_path(project_id, run_id, "property_catalog", catalog_path.relative_to(run_dir).as_posix())

    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()
    resp = client.post(
        "/api/agent/conversation/next-turn",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Train a model for delayed fluorescence lifetime using my project data.",
                }
            ],
        },
    )

    assert resp.status_code == 200
    decision = resp.json["decision"]
    payload = decision["modeling_plan_payload"]
    assert payload["property_id"] == "delayed_fluorescence_lifetime"
    assert "available_inputs" in payload
    assert "property_catalog" in payload["available_inputs"]
    assert "delayed_fluorescence_lifetime" in payload["available_inputs"]
    assert "tau_df_us" in payload["available_inputs"]
    assert decision["status"] == "ready_for_modeling_plan"


def test_conversation_modeling_payload_prefers_explicit_available_inputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_id = "proj-chat-context-explicit"
    run_id = "run-chat-context-explicit"
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    catalog_path = write_json(
        run_dir / "property_catalog.json",
        {"properties": [{"property_id": "plqy", "source_column": "PLQY (%)"}]},
    )
    storage.register_artifact_path(project_id, run_id, "property_catalog", catalog_path.relative_to(run_dir).as_posix())

    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()
    resp = client.post(
        "/api/agent/conversation/modeling-payload",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "available_inputs": ["homo_lumo_gap"],
            "messages": [
                {"role": "user", "content": "Train a model for homo lumo gap."}
            ],
        },
    )

    assert resp.status_code == 200
    payload = resp.json["modeling_plan_payload"]
    assert payload["property_id"] == "homo_lumo_gap"
    assert payload["available_inputs"] == ["homo_lumo_gap"]
