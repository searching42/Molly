from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from ai4s_agent.adapters.structured_dataset_canary import (
    package_private_unimol_model_v1_adapter,
    prepare_private_unimol_training_v1_adapter,
    verify_private_real_tool_harness_task_publication,
)
from ai4s_agent.planner import (
    AtomicTaskRegistry,
    private_structured_dataset_real_tool_task_registry_v3,
    private_structured_dataset_task_registry_v2,
)
from ai4s_agent.app import create_app
from ai4s_agent.scientific_agent_permissions import (
    IMPLEMENTATION_BOUND_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL,
    MODEL_INFERENCE_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import _molecule_identity
from ai4s_agent.structured_dataset_confirmation import (
    build_confirmation_authority,
    build_confirmed_dataset,
    canonical_json_bytes,
)
from tests.test_structured_dataset_confirmation_v2 import _raw_and_review, _row


def test_private_real_tool_v3_does_not_mutate_frozen_v1_or_v2_catalogs() -> None:
    default = AtomicTaskRegistry()
    private_v2 = private_structured_dataset_task_registry_v2()
    private_v3 = private_structured_dataset_real_tool_task_registry_v3()

    assert default.get("train_structured_dataset_canary").execution_route == "local_executor"
    assert private_v2.get("train_structured_dataset_canary").execution_route == "local_executor"
    assert private_v2.get("generate_structured_dataset_canary").execution_route == "local_executor"
    assert private_v3.get("train_private_unimol_v1").execution_route == "remote_execution_service"
    assert private_v3.get("train_private_unimol_v1").remote_task_type == "model_training"
    assert private_v3.get("generate_private_reinvent4_v1").remote_task_type == "molecular_generation"
    assert private_v3.get("predict_private_unimol_v1").remote_task_type == "model_inference"


def test_private_real_tool_v3_requires_remote_outputs_before_packaging() -> None:
    registry = private_structured_dataset_real_tool_task_registry_v3()

    model_package = registry.get("package_private_unimol_model_v1")
    assert {
        "unimol_model_config",
        "unimol_model_weights",
        "unimol_target_scaler",
        "unimol_training_audit",
        "unimol_training_metrics",
    }.issubset(model_package.required_artifacts)
    generation_package = registry.get("package_private_reinvent4_generation_v1")
    assert {
        "reinvent4_candidates",
        "reinvent4_generation_audit",
    }.issubset(generation_package.required_artifacts)
    prediction = registry.get("predict_private_unimol_v1")
    assert prediction.required_artifacts == [
        "candidate_dataset_csv",
        "unimol_model_config",
        "unimol_model_weights",
        "unimol_prediction_config",
        "unimol_target_scaler",
    ]


def test_model_inference_permission_policy_is_versioned_without_v4_drift() -> None:
    assert "model_inference" not in IMPLEMENTATION_BOUND_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL[
        "recognized_remote_task_types"
    ]
    assert "model_inference" in MODEL_INFERENCE_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL[
        "recognized_remote_task_types"
    ]


def test_private_registry_is_injected_through_one_server_bootstrap(
    tmp_path,
) -> None:
    registry = private_structured_dataset_real_tool_task_registry_v3()
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=registry,
    )

    proposal_store = app.extensions["scientific_agent_plan_proposal_store"]
    authorization = app.extensions["scientific_agent_authorization_service"]
    controller = app.extensions["scientific_agent_harness_controller"]
    assert proposal_store.registry is registry
    assert authorization.registry is registry
    assert controller.executor.registry is registry
    assert set(app.extensions["remote_execution_lifecycle"].profiles.execution_profiles) >= {
        "reinvent4-br1-v2",
        "unimol-predict-br1-v1",
        "unimol-train-br1-v2",
    }


def test_private_training_split_is_rederived_from_exact_confirmed_rows(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-v2", name="Private", created_at="2026-08-03T00:00:00Z")
    rows = [
        _row(f"r{index}", smiles, f"10.1000/example-{index}")
        for index, smiles in enumerate(
            ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF"], start=1
        )
    ]
    raw, review = _raw_and_review(rows)
    decision, receipt = build_confirmation_authority(
        raw=raw,
        review=review,
        actor="owner",
        actor_source="deterministic_test_fixture",
        trusted_actors={"owner"},
        project_id="project-v2",
        run_id="run-v2",
        decision_time="2026-08-03T00:00:00Z",
        rows=rows,
        molecule_inspector=_molecule_identity,
    )
    confirmed, confirmed_csv = build_confirmed_dataset(
        raw=raw,
        review=review,
        decision=decision.model_dump(mode="json"),
        receipt=receipt,
        rows=rows,
        trusted_actors={"owner"},
        project_id="project-v2",
        run_id="run-v2",
        created_at="2026-08-03T00:00:00Z",
        molecule_inspector=_molecule_identity,
    )
    run_dir = storage.run_dir("project-v2", "run-v2")
    inputs = run_dir / "inputs"
    inputs.mkdir()
    paths = {
        "confirmation_receipt": inputs / "receipt.json",
        "confirmed_training_dataset": inputs / "confirmed.json",
        "confirmed_training_dataset_csv": inputs / "confirmed.csv",
    }
    paths["confirmation_receipt"].write_bytes(canonical_json_bytes(receipt) + b"\n")
    paths["confirmed_training_dataset"].write_bytes(
        canonical_json_bytes(confirmed) + b"\n"
    )
    paths["confirmed_training_dataset_csv"].write_bytes(confirmed_csv)
    for artifact_id, artifact_path in paths.items():
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            artifact_path.relative_to(run_dir).as_posix(),
        )
    result = prepare_private_unimol_training_v1_adapter(
        {
            "project_id": "project-v2",
            "run_id": "run-v2",
            "output_root": str(run_dir / "structured_dataset_canary"),
            "created_at": "2026-08-03T00:00:00Z",
            "batch_size": 8,
            "early_stopping": 3,
            "epochs": 6,
            "gpu_device": 0,
            "learning_rate": 0.0001,
            "seed": 1729,
            **{
                f"{artifact_id}_path": str(artifact_path)
                for artifact_id, artifact_path in paths.items()
            },
        }
    )
    for artifact_id, artifact_path in result["outputs"].items():
        path_value = Path(artifact_path)
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            path_value.relative_to(run_dir).as_posix(),
        )
    registry = storage.read_artifact_registry("project-v2", "run-v2")
    current_paths = {
        artifact_id: str(run_dir / relative_path)
        for artifact_id, relative_path in registry.items()
    }
    verify_private_real_tool_harness_task_publication(
        storage=storage,
        project_id="project-v2",
        run_id="run-v2",
        task_id="prepare_private_unimol_training_v1",
        artifact_paths=current_paths,
    )

    remote = run_dir / "remote-training"
    remote.mkdir()
    training_config = json.loads(
        Path(result["outputs"]["unimol_training_config"]).read_text(
            encoding="utf-8"
        )
    )
    remote_paths = {
        "unimol_model_config": remote / "config.yaml",
        "unimol_model_weights": remote / "model_0.pth",
        "unimol_target_scaler": remote / "target_scaler.ss",
        "unimol_training_audit": remote / "training_audit.json",
        "unimol_training_metrics": remote / "training_metrics.json",
    }
    remote_paths["unimol_model_config"].write_text(
        "task: regression\ntarget_cols: target_value\n", encoding="utf-8"
    )
    remote_paths["unimol_model_weights"].write_bytes(b"fresh-model")
    remote_paths["unimol_target_scaler"].write_bytes(b"fresh-scaler")
    remote_paths["unimol_training_audit"].write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "unimol_training_audit.v1",
                "provider_version": "0.1.5",
                "config": training_config,
            }
        )
    )
    remote_paths["unimol_training_metrics"].write_bytes(
        canonical_json_bytes({"metrics": {"mae": 0.1, "row_count": 4}})
    )
    for artifact_id, artifact_path in remote_paths.items():
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            artifact_path.relative_to(run_dir).as_posix(),
        )
    registry = storage.read_artifact_registry("project-v2", "run-v2")
    package_inputs = {
        artifact_id: run_dir / registry[artifact_id]
        for artifact_id in [
            "confirmation_receipt",
            "confirmed_training_dataset",
            "unimol_training_request",
            "unimol_split_manifest",
            *remote_paths,
        ]
    }
    package_result = package_private_unimol_model_v1_adapter(
        {
            "project_id": "project-v2",
            "run_id": "run-v2",
            "output_root": str(run_dir / "structured_dataset_canary"),
            "created_at": "2026-08-03T00:00:00Z",
            **{
                f"{artifact_id}_path": str(artifact_path)
                for artifact_id, artifact_path in package_inputs.items()
            },
        }
    )
    for artifact_id, artifact_path in package_result["outputs"].items():
        path_value = Path(artifact_path)
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            path_value.relative_to(run_dir).as_posix(),
        )
    registry = storage.read_artifact_registry("project-v2", "run-v2")
    current_paths = {
        artifact_id: str(run_dir / relative_path)
        for artifact_id, relative_path in registry.items()
    }
    verify_private_real_tool_harness_task_publication(
        storage=storage,
        project_id="project-v2",
        run_id="run-v2",
        task_id="package_private_unimol_model_v1",
        artifact_paths=current_paths,
    )

    training_path = Path(result["outputs"]["unimol_training_dataset_csv"])
    os.chmod(training_path, 0o600)
    training_path.write_bytes(training_path.read_bytes().replace(b"CCF", b"CCI"))
    with pytest.raises(Exception, match="derivationally exact"):
        verify_private_real_tool_harness_task_publication(
            storage=storage,
            project_id="project-v2",
            run_id="run-v2",
            task_id="prepare_private_unimol_training_v1",
            artifact_paths=current_paths,
        )
