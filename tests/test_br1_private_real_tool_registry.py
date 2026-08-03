from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from ai4s_agent.adapters.structured_dataset_canary import (
    _training_rows_from_split,
    _verify_remote_execution_binding,
    _verify_exact_evaluation_publications,
    package_private_unimol_model_v1_adapter,
    prepare_private_unimol_training_v1_adapter,
    verify_private_real_tool_harness_task_publication,
)
from ai4s_agent.planner import (
    AtomicTaskRegistry,
    private_structured_dataset_real_tool_task_registry_v3,
    private_structured_dataset_task_registry_v2,
)
from ai4s_agent.remote_execution_lifecycle import (
    RemoteOutputArtifact,
    RemotePublication,
    build_remote_execution_request,
)
from ai4s_agent.resource_profiles import (
    EXECUTION_PROFILES,
    ConnectionProfile,
    build_transfer_manifest_from_payloads,
)
from ai4s_agent.app import create_app
from ai4s_agent.scientific_agent_permissions import (
    IMPLEMENTATION_BOUND_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL,
    MODEL_INFERENCE_RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryService,
    _molecule_identity,
    validate_candidates,
)
from ai4s_agent.structured_dataset_confirmation import (
    build_confirmation_authority,
    build_confirmed_dataset,
    bind_publication,
    canonical_json_bytes,
    digest_json,
    read_json_artifact,
)
from tests.test_structured_dataset_confirmation_v2 import _raw_and_review, _row


def _publish_test_remote_authority(
    *,
    run_dir: Path,
    project_id: str,
    run_id: str,
    task_id: str,
    profile_id: str,
    input_paths: list[Path],
    output_paths: dict[str, Path],
    audit_path: Path,
    audit_payload: dict[str, object],
) -> Path:
    profile = EXECUTION_PROFILES[profile_id]
    connection = ConnectionProfile(
        connection_id="private-compute",
        ssh_host_alias="private-compute",
        expected_hostname="private-compute",
        remote_root="/srv/molly-runs",
        known_hosts_path="/tmp/molly-known-hosts",
        declared_capabilities=list(profile.required_capabilities),
    )
    descriptors = []
    for index, input_path in enumerate(input_paths):
        suffix = input_path.suffix
        if suffix == ".csv":
            purpose, media_type = "training-data", "application/csv"
        else:
            purpose, media_type = "training-config", "application/json"
        descriptors.append(
            {
                "relative_path": f"input-{index:04d}{suffix}",
                "purpose": purpose,
                "media_type": media_type,
                "payload": input_path.read_bytes(),
            }
        )
    manifest = build_transfer_manifest_from_payloads(
        request_id=f"remote-{task_id}",
        artifacts=descriptors,
        connection=connection,
        execution_profile=profile,
        target_purpose=profile.task_type.replace("_", "-"),
    )
    request = build_remote_execution_request(
        project_id=project_id,
        run_id=run_id,
        task_id=task_id,
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=profile,
        requested_resources={
            "gpu_count": profile.resource_limits.gpu_count_max,
            "cpu_threads": min(8, profile.resource_limits.cpu_threads_max),
            "walltime_sec": min(3600, profile.resource_limits.walltime_sec_max),
        },
        created_at="2026-08-03T00:00:00Z",
    )
    audit_path.write_bytes(
        canonical_json_bytes(
            {
                **audit_payload,
                "remote_request": request.model_dump(mode="json"),
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "input_manifest_sha256": request.input_manifest.manifest_sha256,
            }
        )
    )
    artifacts = []
    media_types = {
        ".json": "application/json",
        ".pth": "application/octet-stream",
        ".ss": "application/octet-stream",
        ".yaml": "application/yaml",
    }
    relative_paths = {
        "unimol_model_config": "model/config.yaml",
        "unimol_model_weights": "model/model_0.pth",
        "unimol_target_scaler": "model/target_scaler.ss",
        "unimol_training_audit": "model/training_audit.json",
        "unimol_training_metrics": "model/training_metrics.json",
    }
    for artifact_id, output_path in sorted(output_paths.items()):
        payload = output_path.read_bytes()
        artifacts.append(
            RemoteOutputArtifact(
                artifact_id=artifact_id,
                relative_path=relative_paths[artifact_id],
                media_type=media_types[output_path.suffix],
                size_bytes=len(payload),
                sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            )
        )
    publication_payload = {
        "schema_version": "molly_remote_execution_publication.v1",
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "approval_sha256": "sha256:" + "a" * 64,
        "input_manifest_sha256": request.input_manifest.manifest_sha256,
        "output_contract": profile.output_contract,
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
        "published_at": "2026-08-03T00:00:00Z",
    }
    publication_payload["publication_sha256"] = digest_json(publication_payload)
    publication = RemotePublication.model_validate(publication_payload)
    publication_path = run_dir / f"{task_id}-remote-publication.json"
    publication_path.write_bytes(canonical_json_bytes(publication.model_dump(mode="json")))
    return publication_path


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


@pytest.mark.parametrize(
    ("artifact_id", "field", "replacement"),
    [
        ("prediction_publication", "prediction_roster", [{"predicted_property": 0.99}]),
        ("candidate_validation", "candidate_validation", [{"ad_status": "IN_DOMAIN"}]),
        ("ranking_publication", "ranked_candidates", [{"eligible": True, "rank": 1}]),
        ("computational_top_n", "candidates", [{"candidate_id": "replaced"}]),
        ("structured_dataset_canary_evidence", "bindings", {"topn": "sha256:" + "f" * 64}),
    ],
)
def test_final_publication_verifier_rejects_coherent_resign(
    artifact_id: str,
    field: str,
    replacement: object,
) -> None:
    expected = {
        "prediction_publication": bind_publication(
            {"prediction_roster": [{"predicted_property": 0.5}]},
            digest_field="publication_digest",
        ),
        "candidate_validation": bind_publication(
            {"candidate_validation": [{"ad_status": "OOD"}]},
            digest_field="publication_digest",
        ),
        "ranking_publication": bind_publication(
            {"ranked_candidates": [{"eligible": False, "rank": None}]},
            digest_field="publication_digest",
        ),
        "computational_top_n": bind_publication(
            {"candidates": []}, digest_field="publication_digest"
        ),
        "structured_dataset_canary_evidence": bind_publication(
            {"bindings": {}, "replay_digest": digest_json({})},
            digest_field="evidence_digest",
        ),
    }
    replaced = deepcopy(expected)
    digest_field = (
        "evidence_digest"
        if artifact_id == "structured_dataset_canary_evidence"
        else "publication_digest"
    )
    replaced[artifact_id][field] = replacement
    bind_publication(replaced[artifact_id], digest_field=digest_field)
    if artifact_id != "structured_dataset_canary_evidence":
        binding_name = {
            "prediction_publication": "prediction",
            "candidate_validation": "validation",
            "ranking_publication": "ranking",
            "computational_top_n": "topn",
        }[artifact_id]
        evidence = replaced["structured_dataset_canary_evidence"]
        evidence["bindings"] = {
            **evidence["bindings"],
            binding_name: replaced[artifact_id]["publication_digest"],
        }
        evidence["replay_digest"] = digest_json(evidence["bindings"])
        bind_publication(evidence, digest_field="evidence_digest")
    with pytest.raises(Exception, match="not derivationally exact"):
        _verify_exact_evaluation_publications(replaced, expected)


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
    split_value = read_json_artifact(
        Path(result["outputs"]["unimol_split_manifest"]),
        digest_field="split_manifest_digest",
    )
    canary_service = StructuredDatasetCanaryService(
        storage=storage,
        trusted_actors=set(),
        harness_authority_managed=True,
    )
    training_domain_rows = _training_rows_from_split(
        canary_service,
        confirmed_path=paths["confirmed_training_dataset_csv"],
        confirmed=confirmed,
        split=split_value,
    )
    reserved_id = next(
        item["row_id"]
        for item in split_value["assignments"]
        if item["split"] in {"test", "external"}
    )
    reserved_row = next(row for row in rows if row["row_id"] == reserved_id)
    candidate = [{"candidate_id": "reserved-only", "smiles": reserved_row["smiles"]}]
    training_domain_validation, _ = validate_candidates(
        candidate,
        training_domain_rows,
        seed=1729,
        ad_similarity_threshold=0.20,
    )
    full_dataset_validation, _ = validate_candidates(
        candidate,
        rows,
        seed=1729,
        ad_similarity_threshold=0.20,
    )
    assert training_domain_validation[0]["training_exact_duplicate"] is False
    assert full_dataset_validation[0]["training_exact_duplicate"] is True
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
    remote_paths["unimol_training_metrics"].write_bytes(
        canonical_json_bytes({"metrics": {"mae": 0.1, "row_count": 4}})
    )
    remote_publication_path = _publish_test_remote_authority(
        run_dir=run_dir,
        project_id="project-v2",
        run_id="run-v2",
        task_id="train_private_unimol_v1",
        profile_id="unimol-train-br1-v2",
        input_paths=[
            Path(result["outputs"]["unimol_training_dataset_csv"]),
            Path(result["outputs"]["unimol_training_config"]),
        ],
        output_paths=remote_paths,
        audit_path=remote_paths["unimol_training_audit"],
        audit_payload={
            "schema_version": "unimol_training_audit.v1",
            "provider_version": "0.1.5",
            "config": training_config,
        },
    )
    for artifact_id, artifact_path in remote_paths.items():
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            artifact_path.relative_to(run_dir).as_posix(),
        )
    storage.register_artifact_path(
        "project-v2",
        "run-v2",
        "remote_execution_publication_training",
        remote_publication_path.relative_to(run_dir).as_posix(),
    )
    registry = storage.read_artifact_registry("project-v2", "run-v2")
    package_inputs = {
        artifact_id: run_dir / registry[artifact_id]
        for artifact_id in [
            "confirmation_receipt",
            "confirmed_training_dataset",
            "unimol_training_request",
            "unimol_split_manifest",
            "unimol_training_dataset_csv",
            "unimol_training_config",
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
            "remote_execution_publication_paths": [
                str(remote_publication_path)
            ],
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

    audit_value = json.loads(
        remote_paths["unimol_training_audit"].read_text(encoding="utf-8")
    )
    with pytest.raises(Exception, match="expected request/profile"):
        _verify_remote_execution_binding(
            publication_paths=[remote_publication_path],
            project_id="project-v2",
            run_id="run-v2",
            task_id="train_private_unimol_v1",
            execution_profile_id="reinvent4-br1-v2",
            audit=audit_value,
            input_artifacts=[
                (
                    Path(result["outputs"]["unimol_training_dataset_csv"]),
                    "training-data",
                    "application/csv",
                ),
                (
                    Path(result["outputs"]["unimol_training_config"]),
                    "training-config",
                    "application/json",
                ),
            ],
            output_paths=remote_paths,
        )
    original_audit = remote_paths["unimol_training_audit"].read_bytes()
    replaced_audit = dict(audit_value)
    replaced_audit["request_id"] = "remote-cross-request"
    remote_paths["unimol_training_audit"].write_bytes(
        canonical_json_bytes(replaced_audit)
    )
    with pytest.raises(Exception, match="expected request/profile"):
        verify_private_real_tool_harness_task_publication(
            storage=storage,
            project_id="project-v2",
            run_id="run-v2",
            task_id="package_private_unimol_model_v1",
            artifact_paths=current_paths,
        )
    remote_paths["unimol_training_audit"].write_bytes(original_audit)

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
