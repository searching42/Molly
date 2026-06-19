from __future__ import annotations

from pathlib import Path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.schemas import ArtifactRef, GateName, RunStatus, StageHistoryItem, StageState
from ai4s_agent.storage import ProjectStorage


def test_conversation_run_plan_preview_builds_gated_run_plan_from_chat_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_id = "proj-chat-runplan"
    run_id = "run-chat-runplan"
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    catalog = write_json(
        run_dir / "property_catalog.json",
        {"properties": [{"property_id": "plqy", "source_column": "PLQY (%)"}]},
    )
    cleaned = run_dir / "cleaned.csv"
    cleaned.write_text("SMILES,plqy,split_group\nCCO,0.7,train\nCCN,0.6,valid\nCCC,0.5,test\n", encoding="utf-8")
    trainability = write_json(run_dir / "trainability.json", {"property_id": "plqy"})
    storage.register_artifact_path(project_id, run_id, "property_catalog", catalog.relative_to(run_dir).as_posix())
    storage.register_artifact_path(project_id, run_id, "cleaned_train_dataset", cleaned.relative_to(run_dir).as_posix())
    storage.register_artifact_path(project_id, run_id, "trainability_report", trainability.relative_to(run_dir).as_posix())

    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()
    resp = client.post(
        "/api/agent/conversation/run-plan-preview",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "messages": [{"role": "user", "content": "Train a model for PLQY using the project data."}],
        },
    )

    assert resp.status_code == 200
    body = resp.json
    assert body["ok"] is True
    assert body["modeling_plan_payload"]["property_id"] == "plqy"
    assert body["run_plan"]["requested_tasks"] == ["train_model"]
    assert [task["task_id"] for task in body["run_plan"]["tasks"]][-1] == "train_model"
    assert body["preview"]["status"] == "ready_for_controlled_execution"
    assert GateName.TRAIN_CONFIG.value in body["preview"]["required_gates"]
    assert body["execution_control"]["direct_execution"] is False
    assert body["execution_control"]["execute_endpoint"] == "/api/run-plan/execute"


def test_conversation_run_plan_preview_builds_controlled_literature_chain(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()
    resp = client.post(
        "/api/agent/conversation/run-plan-preview",
        json={
            "project_id": "proj-literature-preview",
            "run_id": "run-literature-preview",
            "modeling_plan_payload": {"run_id": "run-literature-preview", "goal": "Parse uploaded papers", "property_id": "plqy"},
            "requested_tasks": ["parse_document"],
            "available_artifacts": [],
        },
    )

    assert resp.status_code == 200
    body = resp.json
    preview = body["preview"]
    task_ids = [task["task_id"] for task in body["run_plan"]["tasks"]]
    assert preview["status"] == "ready_for_controlled_execution"
    assert task_ids[-1] == "parse_document"
    assert "prepare_literature_corpus_sources" in task_ids
    assert "acquire_literature_sources" in task_ids
    assert GateName.DATA_MINING.value in preview["required_gates"]
    assert "confirm_required_gates_before_resume" in preview["next_actions"]


def test_conversation_execution_feedback_surfaces_waiting_snapshot_and_audit_records(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage = ProjectStorage(workspace)
    project_id = "proj-feedback"
    run_id = "run-feedback"
    storage.run_dir(project_id, run_id)
    storage.register_artifact_path(project_id, run_id, "cleaned_train_dataset", "cleaned.csv")
    now = now_iso()
    state = StageState(
        run_id=run_id,
        stage="train_model",
        status=RunStatus.WAITING_USER,
        started_at=now,
        updated_at=now,
        history=[StageHistoryItem(stage="train_model", status=RunStatus.WAITING_USER, updated_at=now)],
        details={
            "required_gates": [GateName.TRAIN_CONFIG.value],
            "execution_snapshot": {
                "snapshot_id": "run-feedback:train_model:abc123",
                "snapshot_hash": "abc123",
                "task_id": "train_model",
                "adapter": "train_model_baseline_adapter",
            },
        },
    )
    storage.write_stage_state(project_id, run_id, state)
    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()

    resp = client.post(
        "/api/agent/conversation/execution-feedback",
        json={"project_id": project_id, "run_id": run_id},
    )

    assert resp.status_code == 200
    feedback = resp.json["feedback"]
    assert feedback["status"] == RunStatus.WAITING_USER.value
    assert feedback["execution_snapshot"]["snapshot_id"] == "run-feedback:train_model:abc123"
    assert feedback["execution_snapshot"]["required_gates"] == [GateName.TRAIN_CONFIG.value]
    assert "resume_run_plan" in feedback["next_actions"]
    assert feedback["artifact_registry"]["cleaned_train_dataset"] == "cleaned.csv"


def test_conversation_execution_feedback_surfaces_succeeded_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage = ProjectStorage(workspace)
    project_id = "proj-feedback-done"
    run_id = "run-feedback-done"
    storage.run_dir(project_id, run_id)
    storage.register_artifact_path(project_id, run_id, "report_markdown", "05_report/report.md")
    now = now_iso()
    state = StageState(
        run_id=run_id,
        stage="render_report",
        status=RunStatus.SUCCEEDED,
        started_at=now,
        updated_at=now,
        history=[StageHistoryItem(stage="render_report", status=RunStatus.SUCCEEDED, updated_at=now)],
        artifacts=[ArtifactRef(artifact_id="render_report_result", relative_path="render_report/adapter_result.json")],
        details={"executed_tasks": ["render_report"]},
    )
    storage.write_stage_state(project_id, run_id, state)
    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()

    resp = client.post(
        "/api/agent/conversation/execution-feedback",
        json={"project_id": project_id, "run_id": run_id},
    )

    assert resp.status_code == 200
    feedback = resp.json["feedback"]
    assert feedback["status"] == RunStatus.SUCCEEDED.value
    assert feedback["artifact_registry"]["report_markdown"] == "05_report/report.md"
    assert "review_artifacts" in feedback["next_actions"]
    assert "continue_chat_with_results" in feedback["next_actions"]
