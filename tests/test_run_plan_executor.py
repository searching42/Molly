from __future__ import annotations

import json
from pathlib import Path

import ai4s_agent.adapters as adapter_exports
import ai4s_agent.executor as executor_module
import pytest
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.storage import ProjectStorage


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_run_plan_executor_runs_low_risk_baseline_chain_and_registers_artifacts(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-exec", requested_tasks=["run_baseline"], available_artifacts=[])

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"]
    registry = storage.read_artifact_registry("proj-exec", "r-exec")
    assert "dataset_profile" in registry
    assert "cleaned_train_dataset" in registry
    assert "trainability_report" in registry
    assert "baseline_report" in registry
    for relative_path in registry.values():
        assert (storage.run_dir("proj-exec", "r-exec") / relative_path).exists()
    state = storage.read_stage_state("proj-exec", "r-exec")
    assert state is not None
    assert state.stage == "run_baseline"
    assert state.status == RunStatus.SUCCEEDED
    assert state.ended_at is not None


def test_run_plan_executor_pauses_before_high_risk_task_without_gate(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-exec-gate",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )

    assert result["status"] == RunStatus.WAITING_USER.value
    assert result["waiting_task"] == "train_model"
    assert result["required_gates"] == ["gate_3_train_config"]
    assert "train_model" not in result["executed_tasks"]
    state = storage.read_stage_state("proj-exec", "r-exec-gate")
    assert state is not None
    assert state.stage == "train_model"
    assert state.status == RunStatus.WAITING_USER
    registry = storage.read_artifact_registry("proj-exec", "r-exec-gate")
    assert "cleaned_train_dataset" in registry
    assert "model_metadata" not in registry


def test_run_plan_executor_resume_requires_current_gate_approval(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-missing-gate",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value

    try:
        executor.resume_after_gate(
            project_id="proj-exec",
            run_plan=run_plan,
            approved_gates=[],
            actor="user",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "gate approval required: gate_3_train_config" in str(exc)

    state = storage.read_stage_state("proj-exec", "r-resume-missing-gate")
    assert state is not None
    assert state.stage == "train_model"
    assert state.status == RunStatus.WAITING_USER


def test_run_plan_executor_resume_runs_waiting_task_after_gate_approval(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-train",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
        note="approve baseline training",
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == ["inspect_dataset", "clean_dataset", "check_trainability", "train_model"]
    registry = storage.read_artifact_registry("proj-exec", "r-resume-train")
    assert "trained_model" in registry
    assert "model_metadata" in registry
    assert "model_manifest" in registry
    assert "domain_model_manifest" in registry
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["trained_model"]).is_dir()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_metadata"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_manifest"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["domain_model_manifest"]).is_file()
    run_dir = storage.run_dir("proj-exec", "r-resume-train")
    model_dir = run_dir / registry["trained_model"]
    model_metadata = json.loads((run_dir / registry["model_metadata"]).read_text(encoding="utf-8"))
    _, version_dir = storage.register_model_asset(
        "proj-exec",
        "r-resume-train",
        model_dir,
        property_id="plqy",
        backend=model_metadata["backend"],
        content_hash="sha256:test-model",
        approved_by="user",
    )
    draft = storage.build_promoted_model_asset_draft("proj-exec", version_dir)
    assert draft["model_id"] == "plqy_baseline_v001"
    assert draft["property_id"] == "plqy"
    assert draft["backend"] == model_metadata["backend"]
    assert draft["feature_requirements"] == ["canonical_smiles"]
    assert draft["input_columns"] == {"canonical_smiles": "SMILES"}
    assert draft["metrics"] == model_metadata["metrics"]
    decisions = storage.read_gate_decisions("proj-exec", "r-resume-train")
    assert decisions[0]["gate"] == GateName.TRAIN_CONFIG.value
    assert decisions[0]["actor"] == "user"


def test_run_plan_executor_unimol_training_option_plans_without_registering_fake_model(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-unimol-plan",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
        task_options={
            "train_model": {
                "adapter": "train_model_unimol_legacy_adapter",
                "execute": False,
                "remote_host": "workstation2",
                "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
                "remote_tmp_base": "/tmp/ai4s-agent",
            }
        },
    )

    assert result["status"] == RunStatus.WAITING_USER.value
    assert result["planned_task"] == "train_model"
    assert result["adapter"] == "train_model_unimol_legacy"
    registry = storage.read_artifact_registry("proj-exec", "r-unimol-plan")
    assert "trained_model" not in registry
    assert "model_metadata" not in registry
    state = storage.read_stage_state("proj-exec", "r-unimol-plan")
    assert state is not None
    assert state.stage == "train_model"
    assert state.status == RunStatus.WAITING_USER
    assert state.details["planned_adapter"] == "train_model_unimol_legacy"


def test_run_plan_executor_rejects_adapter_override_for_unallowlisted_task(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-bad-override", requested_tasks=["clean_dataset"], available_artifacts=[])

    with pytest.raises(ValueError, match="adapter override not allowed"):
        RunPlanExecutor(storage=storage).execute(
            project_id="proj-exec",
            run_plan=run_plan,
            input_artifacts={"uploaded_dataset": str(dataset)},
            task_options={"clean_dataset": {"adapter": "legacy_full_flow_adapter"}},
        )


def test_run_plan_executor_allows_domain_model_prediction_override() -> None:
    adapter_name = RunPlanExecutor._adapter_name_for(
        "predict_candidates",
        "predict_candidates_baseline_adapter",
        {"adapter": "predict_candidates_domain_model_adapter"},
    )

    assert adapter_name == "predict_candidates_domain_model_adapter"


def test_run_plan_executor_enables_strict_rdkit_cleaning_by_default(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    executor = RunPlanExecutor(storage=storage)

    payload = executor._payload_for(
        "clean_dataset",
        run_id="r-clean-defaults",
        run_dir=tmp_path / "runs" / "r-clean-defaults",
        artifact_paths={"uploaded_dataset": str(dataset)},
    )

    assert payload["strict_smiles_cleaning"] is True
    assert payload["non_strict_rdkit"] is False


def test_run_plan_executor_uses_shared_strict_smiles_cleaning_helper() -> None:
    assert not hasattr(RunPlanExecutor, "_strict_smiles_cleaning_enabled")
    assert "strict_smiles_cleaning_enabled" in executor_module.__dict__


def test_run_plan_executor_resume_pauses_at_next_gate_then_completes_stub_screening(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-full",
        requested_tasks=["render_report"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == "train_model"

    after_train = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
    )

    assert after_train["status"] == RunStatus.WAITING_USER.value
    assert after_train["waiting_task"] == "generate_candidates"
    assert after_train["required_gates"] == [GateName.FINAL_THRESHOLD.value]

    final = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="user",
    )

    assert final["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-exec", "r-resume-full")
    for artifact_id in [
        "trained_model",
        "candidate_dataset",
        "candidate_predictions",
        "ranked_candidates",
        "report_markdown",
        "report_html",
    ]:
        assert artifact_id in registry
        assert (storage.run_dir("proj-exec", "r-resume-full") / registry[artifact_id]).exists()


def test_run_plan_executor_reinvent4_generation_option_plans_without_registering_fake_candidates(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-reinvent4-plan",
        requested_tasks=["render_report"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    after_train = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
    )
    assert after_train["waiting_task"] == "generate_candidates"

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="user",
        task_options={
            "generate_candidates": {
                "backend": "reinvent4",
                "execute": False,
                "remote_host": "workstation2",
                "remote_python": "/home/lbh/miniconda3/envs/REINVENT4/bin/python",
            }
        },
    )

    assert result["status"] == RunStatus.WAITING_USER.value
    assert result["planned_task"] == "generate_candidates"
    assert result["adapter"] == "generate_candidates_reinvent4"
    registry = storage.read_artifact_registry("proj-exec", "r-reinvent4-plan")
    assert "candidate_dataset" not in registry
    assert "candidate_predictions" not in registry
    state = storage.read_stage_state("proj-exec", "r-reinvent4-plan")
    assert state is not None
    assert state.stage == "generate_candidates"
    assert state.status == RunStatus.WAITING_USER
    assert state.details["planned_adapter"] == "generate_candidates_reinvent4"


def test_run_plan_executor_unimol_prediction_option_plans_without_registering_fake_predictions(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-unimol-predict-plan",
        requested_tasks=["render_report"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    after_train = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
    )
    assert after_train["waiting_task"] == "generate_candidates"

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="user",
        task_options={
            "predict_candidates": {
                "adapter": "predict_candidates_unimol_legacy_adapter",
                "execute": False,
                "remote_host": "workstation2",
                "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
                "remote_tmp_base": "/tmp/ai4s-agent",
            }
        },
    )

    assert result["status"] == RunStatus.WAITING_USER.value
    assert result["planned_task"] == "predict_candidates"
    assert result["adapter"] == "predict_candidates_unimol_legacy"
    registry = storage.read_artifact_registry("proj-exec", "r-unimol-predict-plan")
    assert "candidate_dataset" in registry
    assert "candidate_predictions" not in registry
    state = storage.read_stage_state("proj-exec", "r-unimol-predict-plan")
    assert state is not None
    assert state.stage == "predict_candidates"
    assert state.status == RunStatus.WAITING_USER
    assert state.details["planned_adapter"] == "predict_candidates_unimol_legacy"


def test_run_plan_executor_marks_stage_failed_when_adapter_raises(tmp_path: Path, monkeypatch) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-adapter-error", requested_tasks=["inspect_dataset"], available_artifacts=[])

    def raising_adapter(payload: dict) -> dict:
        raise RuntimeError("adapter exploded")

    monkeypatch.setattr(adapter_exports, "inspect_dataset_service", raising_adapter)

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )

    assert result["status"] == RunStatus.FAILED.value
    assert result["failed_task"] == "inspect_dataset"
    assert result["error"]["code"] == "adapter_exception"
    state = storage.read_stage_state("proj-exec", "r-adapter-error")
    assert state is not None
    assert state.status == RunStatus.FAILED
    assert state.error == {"code": "adapter_exception", "message": "adapter exploded"}


def test_run_plan_execute_endpoint_runs_executor_and_returns_stage_state(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-exec-api", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    assert resp.status_code == 200
    assert resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    storage = ProjectStorage(tmp_path)
    state = storage.read_stage_state("proj-exec-api", "r-exec-api")
    assert state is not None
    assert state.status == RunStatus.SUCCEEDED
    assert "baseline_report" in storage.read_artifact_registry("proj-exec-api", "r-exec-api")


def test_run_plan_execute_endpoint_writes_monitoring_log_tail(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-exec-logs", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-logs",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )
    assert resp.status_code == 200

    logs = client.get("/api/runs/r-exec-logs/logs?limit=20")
    assert logs.status_code == 200
    messages = [entry["message"] for entry in logs.json["logs"]]
    assert any("RunPlan execution started" in message for message in messages)
    assert any("waiting for user" in message for message in messages)


def test_run_plan_execute_endpoint_rejects_invalid_payloads(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-invalid-api", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    cases = [
        ({}, "project_id required"),
        ({"project_id": "proj-invalid"}, "run_plan object required"),
        (
            {
                "project_id": "proj-invalid",
                "run_plan": run_plan.model_dump(mode="json"),
                "input_artifacts": [],
            },
            "input_artifacts must be an object",
        ),
    ]

    for payload, message in cases:
        resp = client.post("/api/run-plan/execute", json=payload)
        assert resp.status_code == 400
        assert resp.json["ok"] is False
        assert message in resp.json["error"]


def test_run_plan_execute_endpoint_rejects_missing_uploaded_dataset_without_stage_state(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-missing-upload", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-missing-upload",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {},
        },
    )

    assert resp.status_code == 400
    assert resp.json["ok"] is False
    assert "missing artifact path: uploaded_dataset" in resp.json["error"]
    storage = ProjectStorage(tmp_path)
    assert storage.read_stage_state("proj-missing-upload", "r-missing-upload") is None


def test_run_plan_resume_endpoint_approves_gate_and_continues_execution(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-resume-api", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    started = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )
    assert started.status_code == 200
    assert started.json["execution"]["status"] == RunStatus.WAITING_USER.value

    resp = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "approved_gates": [GateName.TRAIN_CONFIG.value],
            "actor": "user",
        },
    )

    assert resp.status_code == 200
    assert resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    storage = ProjectStorage(tmp_path)
    assert "trained_model" in storage.read_artifact_registry("proj-resume-api", "r-resume-api")


def test_run_plan_resume_endpoint_accepts_task_options_for_unimol_plan(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-resume-api-unimol", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    resp = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "approved_gates": [GateName.TRAIN_CONFIG.value],
            "actor": "user",
            "task_options": {
                "train_model": {
                    "adapter": "train_model_unimol_legacy_adapter",
                    "execute": False,
                    "remote_host": "workstation2",
                    "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
                    "remote_tmp_base": "/tmp/ai4s-agent",
                }
            },
        },
    )

    assert resp.status_code == 200
    assert resp.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert resp.json["execution"]["planned_task"] == "train_model"
    storage = ProjectStorage(tmp_path)
    assert "trained_model" not in storage.read_artifact_registry("proj-resume-api", "r-resume-api-unimol")


def test_run_plan_resume_endpoint_rejects_missing_gate_approval(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-resume-api-missing", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    resp = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "approved_gates": [],
            "actor": "user",
        },
    )

    assert resp.status_code == 400
    assert "gate approval required: gate_3_train_config" in resp.json["error"]
