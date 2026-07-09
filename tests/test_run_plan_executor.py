from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ai4s_agent.adapters as adapter_exports
import ai4s_agent.executor as executor_module
import pytest
from ai4s_agent.agents.prediction import PredictionPreparationAgent
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.app import create_app
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.run_plan_execute_canary_policy import queued_canary_allowed_for_run_plan
from ai4s_agent.schemas import AtomicTaskSpec, GateName, PlannedTask, RiskLevel, RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any], calls: list[dict[str, Any]]) -> None:
        self.execution = dict(execution)
        self.calls = calls

    def execute(
        self,
        *,
        project_id: str,
        run_plan,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "project_id": project_id,
                "run_plan": run_plan,
                "input_artifacts": input_artifacts,
                "task_options": task_options,
            }
        )
        return dict(self.execution)


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _default_queue_dir(workspace: Path, project_id: str, run_id: str) -> Path:
    return workspace / ".ai4s_internal" / "run_plan_queues" / project_id / run_id


def _project_run_dir(workspace: Path, project_id: str, run_id: str) -> Path:
    return workspace / "projects" / project_id / "runs" / run_id


def _fake_executor_factory(execution: dict[str, Any], calls: list[dict[str, Any]]):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(execution, calls)

    return factory


def _log_messages(client, run_id: str) -> list[str]:
    logs = client.get(f"/api/runs/{run_id}/logs?limit=50")
    assert logs.status_code == 200
    return [entry["message"] for entry in logs.json["logs"]]


def _waiting_task(execution: dict[str, Any]) -> str:
    return str(execution.get("waiting_task") or execution.get("planned_task") or "")


def test_queued_canary_allowlist_policy_reports_disallowed_tasks() -> None:
    allowed_plan = RunPlan(
        run_id="r-policy-allowed",
        requested_tasks=["run_baseline"],
        tasks=[
            PlannedTask(task_id="inspect_dataset"),
            PlannedTask(task_id="clean_dataset"),
            PlannedTask(task_id="check_trainability"),
            PlannedTask(task_id="run_baseline"),
            PlannedTask(task_id="render_report"),
        ],
        available_artifacts=[],
        missing_artifacts=[],
    )
    train_plan = RunPlan(
        run_id="r-policy-train",
        requested_tasks=["train_model"],
        tasks=[PlannedTask(task_id="train_model")],
        available_artifacts=[],
        missing_artifacts=[],
    )
    unknown_plan = RunPlan(
        run_id="r-policy-unknown",
        requested_tasks=["mystery_task"],
        tasks=[PlannedTask(task_id="mystery_task")],
        available_artifacts=[],
        missing_artifacts=[],
    )
    empty_plan = RunPlan(run_id="r-policy-empty", requested_tasks=[], tasks=[], available_artifacts=[], missing_artifacts=[])

    allowed, allowed_policy = queued_canary_allowed_for_run_plan(allowed_plan)
    train_allowed, train_policy = queued_canary_allowed_for_run_plan(train_plan)
    unknown_allowed, unknown_policy = queued_canary_allowed_for_run_plan(unknown_plan)
    empty_allowed, empty_policy = queued_canary_allowed_for_run_plan(empty_plan)

    assert allowed is True
    assert allowed_policy == {
        "allowed": True,
        "reason": "allowlisted_low_risk_tasks",
        "task_ids": ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline", "render_report"],
        "disallowed_tasks": [],
    }
    assert train_allowed is False
    assert train_policy["reason"] == "contains_non_allowlisted_tasks"
    assert train_policy["disallowed_tasks"] == ["train_model"]
    assert unknown_allowed is False
    assert unknown_policy["disallowed_tasks"] == ["mystery_task"]
    assert empty_allowed is False
    assert empty_policy["reason"] == "empty_task_chain"


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


def test_run_plan_executor_inspects_confirmed_training_dataset_artifact(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "confirmed_training_dataset.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-confirmed-inspect",
        requested_tasks=["inspect_dataset"],
        available_artifacts=["confirmed_training_dataset"],
    )

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"confirmed_training_dataset": str(dataset)},
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == ["inspect_dataset"]
    registry = storage.read_artifact_registry("proj-exec", "r-confirmed-inspect")
    assert "dataset_profile" in registry
    assert "property_catalog" in registry


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
    snapshot = state.details["execution_snapshot"]
    assert snapshot["task_id"] == "train_model"
    assert snapshot["snapshot_id"].startswith("r-exec-gate:train_model:")
    assert snapshot["snapshot_hash"]
    assert snapshot["approved_gates"] == [GateName.TRAIN_CONFIG.value]
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


def test_run_plan_executor_resume_rejects_extraneous_future_gates(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-extra-gate",
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

    with pytest.raises(ValueError, match="unexpected gate approval"):
        executor.resume_after_gate(
            project_id="proj-exec",
            run_plan=run_plan,
            approved_gates=[GateName.TRAIN_CONFIG.value, GateName.FINAL_THRESHOLD.value],
            actor="user",
        )

    state = storage.read_stage_state("proj-exec", "r-resume-extra-gate")
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
    assert "model_diagnostics_report" in registry
    assert "model_package_review" in registry
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["trained_model"]).is_dir()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_metadata"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_manifest"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["domain_model_manifest"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_diagnostics_report"]).is_file()
    assert (storage.run_dir("proj-exec", "r-resume-train") / registry["model_package_review"]).is_file()
    run_dir = storage.run_dir("proj-exec", "r-resume-train")
    model_dir = run_dir / registry["trained_model"]
    model_metadata = json.loads((run_dir / registry["model_metadata"]).read_text(encoding="utf-8"))
    diagnostics = json.loads((run_dir / registry["model_diagnostics_report"]).read_text(encoding="utf-8"))
    review = json.loads((run_dir / registry["model_package_review"]).read_text(encoding="utf-8"))
    assert diagnostics["property_id"] == "plqy"
    assert diagnostics["model_id"] == "plqy_baseline_v001"
    assert review["model_id"] == "plqy_baseline_v001"
    assert review["executable"] is False
    assert review["decision"] in {"promote_candidate", "rerun_recommended", "memory_only", "blocked"}
    assert review["promotion_draft"] == {} or review["required_permissions"] == ["promote_asset"]
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
    assert decisions[0]["approved_snapshot_id"].startswith("r-resume-train:train_model:")
    assert decisions[0]["approved_snapshot_hash"]


def test_run_plan_executor_resume_rejects_changed_task_options_after_gate(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-snapshot-change",
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

    with pytest.raises(ValueError, match="execution snapshot changed"):
        executor.resume_after_gate(
            project_id="proj-exec",
            run_plan=run_plan,
            approved_gates=[GateName.TRAIN_CONFIG.value],
            actor="user",
            task_options={
                "train_model": {
                    "adapter": "train_model_unimol_legacy_adapter",
                    "execute": False,
                    "remote_host": "workstation2",
                }
            },
        )

    state = storage.read_stage_state("proj-exec", "r-resume-snapshot-change")
    assert state is not None
    assert state.stage == "train_model"
    assert state.status == RunStatus.WAITING_USER


def test_run_plan_executor_resume_rejects_changed_artifact_content_after_gate(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-resume-artifact-change",
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
    registry = storage.read_artifact_registry("proj-exec", "r-resume-artifact-change")
    run_dir = storage.run_dir("proj-exec", "r-resume-artifact-change")
    cleaned_path = run_dir / registry["cleaned_train_dataset"]
    with cleaned_path.open("a", encoding="utf-8") as f:
        f.write("CCN,0.999,777,train\n")

    with pytest.raises(ValueError, match="execution snapshot changed"):
        executor.resume_after_gate(
            project_id="proj-exec",
            run_plan=run_plan,
            approved_gates=[GateName.TRAIN_CONFIG.value],
            actor="user",
        )

    state = storage.read_stage_state("proj-exec", "r-resume-artifact-change")
    assert state is not None
    assert state.stage == "train_model"
    assert state.status == RunStatus.WAITING_USER


def test_run_plan_executor_same_gate_name_requires_new_snapshot_for_each_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path)
    monkeypatch.setattr(
        adapter_exports,
        "test_success_adapter",
        lambda payload: {"status": "success", "adapter": "test_success"},
        raising=False,
    )
    registry = AtomicTaskRegistry(
        [
            AtomicTaskSpec(
                task_id="first_data_gate",
                output_artifacts=["first_done"],
                risk_level=RiskLevel.HIGH,
                gates=[GateName.DATA_MINING.value],
                default_adapter="test_success_adapter",
            ),
            AtomicTaskSpec(
                task_id="second_data_gate",
                output_artifacts=["second_done"],
                risk_level=RiskLevel.HIGH,
                gates=[GateName.DATA_MINING.value],
                default_adapter="test_success_adapter",
            ),
        ]
    )
    run_plan = expand_run_plan(
        run_id="r-same-gate",
        requested_tasks=["first_data_gate", "second_data_gate"],
        registry=registry,
    )
    executor = RunPlanExecutor(storage=storage, registry=registry)
    first = executor.execute(project_id="proj-exec", run_plan=run_plan)
    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == "first_data_gate"

    after_first = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.DATA_MINING.value],
        actor="user",
    )

    assert after_first["status"] == RunStatus.WAITING_USER.value
    assert after_first["waiting_task"] == "second_data_gate"
    state = storage.read_stage_state("proj-exec", "r-same-gate")
    assert state is not None
    assert state.stage == "second_data_gate"
    assert state.details["execution_snapshot"]["task_id"] == "second_data_gate"


def test_training_review_promotion_and_prediction_preparation_acceptance(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-e2e-promote",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-e2e",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value

    trained = executor.resume_after_gate(
        project_id="proj-e2e",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
        note="approve local acceptance training",
    )

    assert trained["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-e2e", "r-e2e-promote")
    run_dir = storage.run_dir("proj-e2e", "r-e2e-promote")
    model_dir = run_dir / registry["trained_model"]
    metadata = json.loads((run_dir / registry["model_metadata"]).read_text(encoding="utf-8"))
    review = json.loads((run_dir / registry["model_package_review"]).read_text(encoding="utf-8"))
    assert review["model_id"] == "plqy_baseline_v001"
    assert review["decision"] in {"promote_candidate", "rerun_recommended", "memory_only", "blocked"}

    _, version_dir = storage.register_model_asset(
        "proj-e2e",
        "r-e2e-promote",
        model_dir,
        property_id="plqy",
        backend=metadata["backend"],
        content_hash="sha256:e2e-local-model",
        approved_by="user",
        approval_note="acceptance test registration",
    )
    draft = storage.build_promoted_model_asset_draft("proj-e2e", version_dir)
    promoted, promoted_path = storage.promote_registered_model_asset(
        "proj-e2e",
        "r-e2e-promote",
        version_dir,
        model_id=draft["model_id"],
        domain="oled",
        property_id=draft["property_id"],
        use_case=draft["use_case"],
        backend=draft["backend"],
        approved_by="user",
        metrics=draft["metrics"],
        applicability=draft["applicability"],
        feature_requirements=draft["feature_requirements"],
        input_columns=draft["input_columns"],
        limitations=draft["limitations"],
        note="acceptance test promotion",
    )
    assert promoted_path.is_file()
    assert promoted.status.value == "confirmed"

    candidate_csv = tmp_path / "input" / "candidates.csv"
    output_csv = tmp_path / "output" / "predictions.csv"
    candidate_csv.write_text("SMILES\nCCO\n", encoding="utf-8")
    preparation = PredictionPreparationAgent().prepare_prediction_for_project(
        storage=storage,
        project_id="proj-e2e",
        run_id="r-e2e-predict",
        goal="Predict quantum yield for OLED candidates with the approved model asset.",
        domain="oled",
        property_id="quantum_yield",
        use_case="scalar_prediction",
        available_inputs={"canonical_smiles"},
        input_columns={"canonical_smiles": "candidate_smiles"},
        candidate_csv=str(candidate_csv),
        output_csv=str(output_csv),
    )

    assert preparation.status == "needs_confirmation"
    assert preparation.promoted_model_asset is not None
    assert preparation.promoted_model_asset.asset_id == promoted.asset_id
    assert preparation.model_selection.selection_role == "prediction_asset"
    assert preparation.model_selection.selected_model.reuse_policy == "promoted_model_asset"
    assert preparation.requires_training is False
    assert preparation.reuse_requires_user_approval is False
    assert preparation.required_gates == [GateName.FINAL_THRESHOLD.value]
    assert "training_required_for_request" not in preparation.warnings
    assert preparation.adapter == "predict_candidates_baseline_adapter"
    assert preparation.adapter_payload["model_id"] == "plqy_baseline_v001"
    assert preparation.adapter_payload["model_backend"] == metadata["backend"]
    assert preparation.adapter_payload["model_dir"] == promoted.model_dir
    assert preparation.adapter_payload["input_columns"] == {"canonical_smiles": "SMILES"}
    assert preparation.adapter_payload["required_inputs"] == ["canonical_smiles"]


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
    task_options = {
        "train_model": {
            "adapter": "train_model_unimol_legacy_adapter",
            "execute": False,
            "remote_host": "workstation2",
            "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
            "remote_tmp_base": "/tmp/ai4s-agent",
        }
    }
    executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
        task_options=task_options,
    )

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
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
    execution_snapshot = state.details["execution_snapshot"]
    assert execution_snapshot["task_id"] == "train_model"
    assert execution_snapshot["task_options"]["execute"] is True
    assert execution_snapshot["payload"]["execute"] is True
    assert execution_snapshot["payload"]["remote_host"] == "workstation2"


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
    task_options = {
        "generate_candidates": {
            "backend": "reinvent4",
            "execute": False,
            "remote_host": "workstation2",
            "remote_python": "/home/lbh/miniconda3/envs/REINVENT4/bin/python",
        }
    }
    executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
        task_options=task_options,
    )
    after_train = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.TRAIN_CONFIG.value],
        actor="user",
        task_options=task_options,
    )
    assert after_train["waiting_task"] == "generate_candidates"

    result = executor.resume_after_gate(
        project_id="proj-exec",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="user",
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
    execution_snapshot = state.details["execution_snapshot"]
    assert execution_snapshot["task_id"] == "generate_candidates"
    assert execution_snapshot["task_options"]["execute"] is True
    assert execution_snapshot["payload"]["execute"] is True
    assert execution_snapshot["payload"]["remote_host"] == "workstation2"


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
    assert "execution_backend" not in resp.json
    storage = ProjectStorage(tmp_path)
    state = storage.read_stage_state("proj-exec-api", "r-exec-api")
    assert state is not None
    assert state.status == RunStatus.SUCCEEDED
    assert "baseline_report" in storage.read_artifact_registry("proj-exec-api", "r-exec-api")


def test_run_plan_execute_endpoint_sync_backend_logs_when_canary_disabled(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-exec-sync-backend", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-sync-backend",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    assert resp.status_code == 200
    assert resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in resp.json
    assert "queue_summary" not in resp.json
    messages = _log_messages(client, "r-exec-sync-backend")
    assert any("RunPlan execution started" in message for message in messages)
    assert any("RunPlan execution backend: sync" in message for message in messages)
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_run_plan_execute_endpoint_queued_canary_allowlisted_chain_uses_queue_bridge(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-exec-canary", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": True,
            "run_id": "r-exec-canary",
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"],
        },
        calls,
    )
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-canary",
            "run_plan": run_plan.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 200
    assert resp.json["ok"] is True
    assert resp.json["execution_backend"] == "queued_canary"
    assert resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert resp.json["queue_summary"]["ok"] is True
    assert resp.json["queue_summary"]["final_job"]["status"] == "succeeded"
    assert resp.json["queue_summary"]["final_lease"]["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["project_id"] == "proj-exec-canary"
    messages = _log_messages(client, "r-exec-canary")
    assert any("RunPlan execution backend: queued_canary" in message for message in messages)


def test_run_plan_execute_endpoint_canary_can_roll_back_to_sync_with_same_payload(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    canary_plan = expand_run_plan(run_id="r-exec-canary-rollback", requested_tasks=["run_baseline"], available_artifacts=[])
    sync_plan = expand_run_plan(run_id="r-exec-sync-rollback", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {"ok": True, "run_id": "r-exec-canary-rollback", "status": RunStatus.SUCCEEDED.value},
        calls,
    )
    client = app.test_client()

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    canary_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-rollback",
            "run_plan": canary_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = False
    sync_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-rollback",
            "run_plan": sync_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    assert canary_resp.status_code == 200
    assert canary_resp.json["execution_backend"] == "queued_canary"
    assert "queue_summary" in canary_resp.json
    assert sync_resp.status_code == 200
    assert "execution_backend" not in sync_resp.json
    assert "queue_summary" not in sync_resp.json
    assert sync_resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert (_default_queue_dir(tmp_path, "proj-exec-rollback", "r-exec-canary-rollback") / "worker_queue.json").exists()
    assert not _default_queue_dir(tmp_path, "proj-exec-rollback", "r-exec-sync-rollback").exists()
    assert client.post("/api/run-plan/resume", json={}).status_code == 400


def test_run_plan_execute_endpoint_queued_canary_falls_back_to_sync_for_train_model(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    sync_plan = expand_run_plan(run_id="r-exec-sync-waiting", requested_tasks=["train_model"], available_artifacts=[])
    fallback_plan = expand_run_plan(run_id="r-exec-canary-waiting", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {"ok": True, "run_id": "r-exec-canary-waiting", "status": RunStatus.WAITING_USER.value, "waiting_task": "train_model", "required_gates": [GateName.TRAIN_CONFIG.value]},
        calls,
    )
    client = app.test_client()

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = False
    sync_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-waiting",
            "run_plan": sync_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    fallback_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-waiting",
            "run_plan": fallback_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    assert sync_resp.status_code == 200
    assert fallback_resp.status_code == 200
    assert sync_resp.json["execution"]["status"] == fallback_resp.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert _waiting_task(sync_resp.json["execution"]) == _waiting_task(fallback_resp.json["execution"]) == "train_model"
    assert sync_resp.json["execution"]["required_gates"] == fallback_resp.json["execution"]["required_gates"] == [GateName.TRAIN_CONFIG.value]
    assert "execution_backend" not in fallback_resp.json
    assert "queue_summary" not in fallback_resp.json
    assert len(calls) == 0
    assert not _default_queue_dir(tmp_path, "proj-exec-waiting", "r-exec-canary-waiting").exists()
    messages = _log_messages(client, "r-exec-canary-waiting")
    assert any("RunPlan execution backend: sync_fallback_not_allowlisted" in message for message in messages)
    assert any("disallowed_tasks=train_model" in message for message in messages)


def test_run_plan_execute_endpoint_queued_canary_falls_back_to_sync_for_unknown_task(tmp_path: Path) -> None:
    run_plan = RunPlan(
        run_id="r-exec-canary-unknown",
        requested_tasks=["mystery_task"],
        tasks=[PlannedTask(task_id="mystery_task")],
        available_artifacts=[],
        missing_artifacts=[],
    )
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {"ok": True, "run_id": "r-exec-canary-unknown", "status": RunStatus.SUCCEEDED.value},
        calls,
    )
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-unknown",
            "run_plan": run_plan.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 400
    assert resp.json["ok"] is False
    assert "unknown atomic task: mystery_task" in resp.json["error"]
    assert "execution_backend" not in resp.json
    assert "queue_summary" not in resp.json
    assert len(calls) == 0
    assert not _default_queue_dir(tmp_path, "proj-exec-unknown", "r-exec-canary-unknown").exists()
    messages = _log_messages(client, "r-exec-canary-unknown")
    assert any("RunPlan execution backend: sync_fallback_not_allowlisted" in message for message in messages)
    assert any("disallowed_tasks=mystery_task" in message for message in messages)


def test_run_plan_execute_endpoint_queued_canary_does_not_consume_existing_job(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-exec-canary-existing", requested_tasks=["run_baseline"], available_artifacts=[])
    queue_dir = _default_queue_dir(tmp_path, "proj-exec-canary-existing", "r-exec-canary-existing")
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    old_job = queue.enqueue("proj-exec-canary-existing", "r-exec-canary-existing", {"task_id": "old_job"})
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {"ok": True, "run_id": "r-exec-canary-existing", "status": RunStatus.SUCCEEDED.value},
        calls,
    )
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-canary-existing",
            "run_plan": run_plan.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 200
    assert resp.json["execution_backend"] == "queued_canary"
    assert resp.json["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert queue.status(old_job["job_id"])["status"] == "queued"
    assert len(calls) == 1


def test_run_plan_execute_endpoint_queued_canary_failed_execution_is_not_success(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-exec-canary-failed", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": False,
            "run_id": "r-exec-canary-failed",
            "status": RunStatus.FAILED.value,
            "error": {"message": "adapter failed"},
        },
        calls,
    )
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-canary-failed",
            "run_plan": run_plan.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 400
    assert resp.json["ok"] is False
    assert resp.json["execution_backend"] == "queued_canary"
    assert resp.json["execution"]["status"] == RunStatus.FAILED.value
    assert resp.json["queue_summary"]["ok"] is False
    assert resp.json["queue_summary"]["final_job"]["status"] == "failed"
    assert len(calls) == 1


def test_run_plan_execute_endpoint_queued_canary_failed_execution_logs_backend_and_failed_task(tmp_path: Path) -> None:
    run_plan = expand_run_plan(run_id="r-exec-canary-failed-log", requested_tasks=["run_baseline"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": False,
            "run_id": "r-exec-canary-failed-log",
            "status": RunStatus.FAILED.value,
            "failed_task": "run_baseline",
            "error": {"message": "adapter failed"},
        },
        calls,
    )
    client = app.test_client()

    resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-canary-failed-log",
            "run_plan": run_plan.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 400
    assert resp.json["ok"] is False
    assert resp.json["execution_backend"] == "queued_canary"
    assert resp.json["execution"]["status"] == RunStatus.FAILED.value
    assert resp.json["execution"]["failed_task"] == "run_baseline"
    assert resp.json["queue_summary"]["ok"] is False
    assert resp.json["queue_summary"]["final_job"]["status"] == "failed"
    messages = _log_messages(client, "r-exec-canary-failed-log")
    assert any("RunPlan execution backend: queued_canary" in message for message in messages)
    assert any("RunPlan execution failed at run_baseline" in message for message in messages)
    assert not (
        _project_run_dir(tmp_path, "proj-exec-canary-failed-log", "r-exec-canary-failed-log")
        / "review"
        / "replan_resume_intent.json"
    ).exists()
    assert client.post("/api/run-plan/resume", json={}).status_code == 400
    assert len(calls) == 1


def test_run_plan_execute_endpoint_canary_non_empty_queue_then_sync_rollback_does_not_touch_old_job(tmp_path: Path) -> None:
    canary_plan = expand_run_plan(run_id="r-exec-canary-safety", requested_tasks=["run_baseline"], available_artifacts=[])
    sync_plan = expand_run_plan(run_id="r-exec-sync-safety", requested_tasks=["run_baseline"], available_artifacts=[])
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    queue_dir = _default_queue_dir(tmp_path, "proj-exec-safety", "r-exec-canary-safety")
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    old_job = queue.enqueue("proj-exec-safety", "r-exec-canary-safety", {"task_id": "old_job"})
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {"ok": True, "run_id": "r-exec-canary-safety", "status": RunStatus.SUCCEEDED.value},
        calls,
    )
    client = app.test_client()

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    canary_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-safety",
            "run_plan": canary_plan.model_dump(mode="json"),
        },
    )
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = False
    sync_resp = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-exec-safety",
            "run_plan": sync_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )

    assert canary_resp.status_code == 200
    assert canary_resp.json["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert sync_resp.status_code == 200
    assert "queue_summary" not in sync_resp.json
    assert queue.status(old_job["job_id"])["status"] == "queued"
    assert not _default_queue_dir(tmp_path, "proj-exec-safety", "r-exec-sync-safety").exists()
    assert len(calls) == 1


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


def test_run_plan_resume_endpoint_uses_preapproved_task_options_for_unimol_plan(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-resume-api-unimol", requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    task_options = {
        "train_model": {
            "adapter": "train_model_unimol_legacy_adapter",
            "execute": False,
            "remote_host": "workstation2",
            "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
            "remote_tmp_base": "/tmp/ai4s-agent",
        }
    }
    client.post(
        "/api/run-plan/execute",
        json={
            "project_id": "proj-resume-api",
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
            "task_options": task_options,
        },
    )

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
    assert resp.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert resp.json["execution"]["planned_task"] == "train_model"
    storage = ProjectStorage(tmp_path)
    assert "trained_model" not in storage.read_artifact_registry("proj-resume-api", "r-resume-api-unimol")


def test_run_plan_resume_endpoint_rejects_task_options_changed_after_gate(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id="r-resume-api-changed-options", requested_tasks=["train_model"], available_artifacts=[])
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
                }
            },
        },
    )

    assert resp.status_code == 400
    assert "execution snapshot changed" in resp.json["error"]


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


def test_task_options_rejects_artifact_identity_key_override(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-protected-keys",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    with pytest.raises(ValueError, match="cannot override artifact identity keys"):
        executor.execute(
            project_id="proj-exec",
            run_plan=run_plan,
            input_artifacts={"uploaded_dataset": str(dataset)},
            task_options={"train_model": {"train_csv": "/tmp/override.csv"}},
        )


def test_task_options_allows_non_protected_keys_and_safe_overrides(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-safe-options",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    result = executor.execute(
        project_id="proj-exec",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
        task_options={"train_model": {"epochs": 20, "batch_size": 16, "adapter": "train_model_unimol_legacy_adapter", "execute": False}},
    )
    assert result["status"] == RunStatus.WAITING_USER.value
