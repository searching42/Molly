from __future__ import annotations

from io import BytesIO

from ai4s_agent.app import create_app
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.storage import ProjectStorage


def _training_csv_bytes() -> bytes:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def test_localhost_project_workflow_closes_permission_execution_artifact_and_report_loop(tmp_path) -> None:
    project_id = "proj-harden-004"
    run_id = "run-harden-004"
    actor = "alice"
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_CLIENT_PERMISSION_FLAGS"] = False
    client = app.test_client()

    created = client.post("/api/projects", json={"project_id": project_id, "name": "HARDEN-004 OLED"})
    assert created.status_code == 200

    denied_upload = client.post(
        f"/api/projects/{project_id}/upload",
        data={"file": (BytesIO(_training_csv_bytes()), "training.csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )
    assert denied_upload.status_code == 403
    assert denied_upload.json["permission"]["reason"] == "SERVER_GRANT_REQUIRED"

    grant = client.post(
        f"/api/projects/{project_id}/permissions/grants",
        json={
            "action": "upload_dataset",
            "actor": actor,
            "confirmed": True,
            "reason": "HARDEN-004 e2e upload approval",
        },
    )
    assert grant.status_code == 200

    uploaded = client.post(
        f"/api/projects/{project_id}/upload",
        data={"file": (BytesIO(_training_csv_bytes()), "training.csv")},
        headers={"X-Actor": actor},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["reason"] == "SERVER_GRANT"
    uploaded_path = uploaded.json["path"]

    audit = client.get(f"/api/projects/{project_id}/permissions/audit")
    assert audit.status_code == 200
    audit_reasons = [item["reason"] for item in audit.json["audit"]]
    assert "SERVER_GRANT_REQUIRED" in audit_reasons
    assert "SERVER_GRANT_CREATED" in audit_reasons
    assert "SERVER_GRANT" in audit_reasons

    plan = client.post(
        "/api/plan",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "prompt": "Train an OLED PLQY model and report traceable artifacts.",
        },
    )
    assert plan.status_code == 200
    assert plan.json["plan_scope"] == "project"
    assert plan.json["job_key"] == {"project_id": project_id, "run_id": run_id}

    approved_task_parse = client.post(
        "/api/gates/approve",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "gate": GateName.TASK_PARSE.value,
            "actor": actor,
        },
    )
    assert approved_task_parse.status_code == 200
    assert approved_task_parse.json["plan_scope"] == "project"

    baseline_preview = client.post(
        "/api/run-plan/expand",
        json={"run_id": run_id, "requested_tasks": ["run_baseline"], "available_artifacts": []},
    )
    assert baseline_preview.status_code == 200
    baseline_run_plan = baseline_preview.json["run_plan"]
    assert [task["task_id"] for task in baseline_run_plan["tasks"]][-1] == "run_baseline"

    baseline_execution = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": project_id,
            "run_plan": baseline_run_plan,
            "input_artifacts": {"uploaded_dataset": uploaded_path},
        },
    )
    assert baseline_execution.status_code == 200
    assert baseline_execution.json["execution"]["status"] == RunStatus.SUCCEEDED.value

    storage = ProjectStorage(tmp_path)
    registry_after_baseline = storage.read_artifact_registry(project_id, run_id)
    assert "dataset_profile" in registry_after_baseline
    assert "cleaned_train_dataset" in registry_after_baseline
    assert "trainability_report" in registry_after_baseline
    assert "baseline_report" in registry_after_baseline

    project_logs = client.get(f"/api/projects/{project_id}/runs/{run_id}/logs?limit=50")
    assert project_logs.status_code == 200
    project_log_messages = [entry["message"] for entry in project_logs.json["logs"]]
    assert any("RunPlan execution started" in message for message in project_log_messages)
    assert any("RunPlan execution completed" in message for message in project_log_messages)

    train_preview = client.post(
        "/api/run-plan/expand",
        json={"run_id": run_id, "requested_tasks": ["train_model"], "available_artifacts": []},
    )
    assert train_preview.status_code == 200
    train_run_plan = train_preview.json["run_plan"]
    assert [task["task_id"] for task in train_run_plan["tasks"]][-1] == "train_model"

    train_start = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": project_id,
            "run_plan": train_run_plan,
            "input_artifacts": {"uploaded_dataset": uploaded_path},
        },
    )
    assert train_start.status_code == 200
    assert train_start.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert train_start.json["execution"]["waiting_task"] == "train_model"
    assert train_start.json["execution"]["required_gates"] == [GateName.TRAIN_CONFIG.value]

    decision = client.post(
        "/api/agent/decision-card",
        json={"project_id": project_id, "run_id": run_id, "question": "为什么现在需要批准训练？"},
    )
    assert decision.status_code == 200
    assert decision.json["card"]["decision_required"] is True
    assert decision.json["card"]["primary_action"] == "approve_gate"

    feedback_waiting = client.post(
        "/api/agent/conversation/execution-feedback",
        json={"project_id": project_id, "run_id": run_id},
    )
    assert feedback_waiting.status_code == 200
    assert feedback_waiting.json["feedback"]["status"] == RunStatus.WAITING_USER.value
    assert feedback_waiting.json["feedback"]["execution_snapshot"]["task_id"] == "train_model"

    train_resume = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": project_id,
            "run_plan": train_run_plan,
            "approved_gates": [GateName.TRAIN_CONFIG.value],
            "actor": actor,
            "note": "Approve baseline training for HARDEN-004 e2e.",
        },
    )
    assert train_resume.status_code == 200
    assert train_resume.json["execution"]["status"] == RunStatus.SUCCEEDED.value

    registry_after_training = storage.read_artifact_registry(project_id, run_id)
    assert "trained_model" in registry_after_training
    assert "model_diagnostics_report" in registry_after_training
    assert "model_package_review" in registry_after_training

    status = client.get(f"/api/runs/{run_id}?project_id={project_id}")
    assert status.status_code == 200
    assert status.json["state_source"] == "project"
    assert status.json["stage"]["stage"] == "train_model"
    assert status.json["stage"]["status"] == RunStatus.SUCCEEDED.value
    assert "trained_model" in status.json["artifacts"]

    logs_after_resume = client.get(f"/api/projects/{project_id}/runs/{run_id}/logs?limit=80")
    assert logs_after_resume.status_code == 200
    resumed_messages = [entry["message"] for entry in logs_after_resume.json["logs"]]
    assert any("RunPlan resume requested" in message for message in resumed_messages)
    assert any("RunPlan execution completed" in message for message in resumed_messages)

    verification = client.post(f"/api/projects/{project_id}/runs/{run_id}/verify")
    assert verification.status_code == 200
    assert verification.json["outputs"]["verification_report_md"].endswith("verification_report.md")

    report = client.post(
        "/api/agent/report-summary",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Train an OLED PLQY model and report traceable artifacts.",
            "verification_report": verification.json["report"],
        },
    )
    assert report.status_code == 200
    assert "report_synthesis_proposal_md" in report.json["outputs"]

    report_preview = client.get(
        f"/api/projects/{project_id}/runs/{run_id}/report-preview?artifact_id=report_synthesis_proposal_md"
    )
    assert report_preview.status_code == 200
    assert report_preview.json["preview"]["artifact_id"] == "report_synthesis_proposal_md"

    feedback_done = client.post(
        "/api/agent/conversation/execution-feedback",
        json={"project_id": project_id, "run_id": run_id},
    )
    assert feedback_done.status_code == 200
    assert feedback_done.json["feedback"]["status"] == RunStatus.SUCCEEDED.value
    assert "report_synthesis_proposal_md" in feedback_done.json["feedback"]["artifact_registry"]
