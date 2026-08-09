from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.remote_execution_lifecycle import (
    PUBLICATION_VERSION,
    RemoteOutputArtifact,
    RemotePublication,
)
from ai4s_agent.schemas import AgentHarnessVerifiedOutputBinding
from ai4s_agent.scientific_agent_result_projection import (
    ScientificAgentResultProjectionError,
    ScientificAgentResultProjectionService,
)
from ai4s_agent.structured_dataset_confirmation import bind_publication, digest_json
from ai4s_agent.storage import ProjectStorage


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _audit_bytes() -> bytes:
    request = {
        "schema_version": "molly_remote_execution_request.v1",
        "request_id": "remote-request-001",
        "request_sha256": "sha256:" + "a" * 64,
        "execution_profile_digest": "sha256:" + "b" * 64,
        "input_manifest": {"manifest_sha256": "sha256:" + "c" * 64},
    }
    return (
        json.dumps(
            {
                "schema_version": "unimol_prediction_audit.v1",
                "provider_version": "0.1.5",
                "config": {},
                "remote_request": request,
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "input_manifest_sha256": request["input_manifest"]["manifest_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _publication(
    *,
    output_contract: str,
    payloads: dict[str, bytes],
) -> RemotePublication:
    artifact_specs = {
        "unimol-prediction-output-v1": (
            ("unimol_predictions", "predictions.csv", "text/csv"),
            ("unimol_prediction_audit", "prediction_audit.json", "application/json"),
        ),
        "reinvent4-generation-output-v1": (
            ("reinvent4_candidates", "candidates.csv", "text/csv"),
        ),
    }[output_contract]
    artifacts = sorted([
        RemoteOutputArtifact(
            artifact_id=artifact_id,
            relative_path=relative_path,
            media_type=media_type,
            size_bytes=len(payloads[relative_path]),
            sha256=_bytes_digest(payloads[relative_path]),
        ).model_dump(mode="json")
        for artifact_id, relative_path, media_type in artifact_specs
    ], key=lambda item: (item["artifact_id"], item["relative_path"]))
    material = {
        "schema_version": PUBLICATION_VERSION,
        "request_id": "remote-request-001",
        "request_sha256": "sha256:" + "a" * 64,
        "approval_sha256": "sha256:" + "d" * 64,
        "input_manifest_sha256": "sha256:" + "c" * 64,
        "output_contract": output_contract,
        "artifacts": artifacts,
        "published_at": "2026-08-09T00:00:00Z",
    }
    return RemotePublication.model_validate(
        {**material, "publication_sha256": _digest(material)}
    )


def _service(tmp_path: Path) -> ScientificAgentResultProjectionService:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Projection test", created_at="2026-08-09T00:00:00Z")
    return ScientificAgentResultProjectionService(projects=storage, top_n=2)


def _final_result_fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    project_id = "project-1"
    run_id = "run-1"
    evaluation_configuration = {"top_n": 2, "validation_seed": 1729}
    evaluation_configuration_digest = digest_json(evaluation_configuration)
    prediction = bind_publication(
        {
            "schema_version": "structured_dataset_prediction_publication.v1",
            "project_id": project_id,
            "run_id": run_id,
            "prediction_roster": [
                {"candidate_id": "candidate-b", "predicted_property": 0.80},
                {"candidate_id": "candidate-a", "predicted_property": 0.95},
            ],
        },
        digest_field="publication_digest",
    )
    validation = bind_publication(
        {
            "schema_version": "structured_dataset_candidate_validation.v1",
            "project_id": project_id,
            "run_id": run_id,
            "candidate_validation": [],
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
        },
        digest_field="publication_digest",
    )
    ranking_configuration = {
        "objective": "maximize_predicted_PLQY",
        "ranking_direction": "descending",
        "top_n_size": 2,
    }
    ranking_rows = [
        {
            "candidate_id": "candidate-b",
            "predicted_property": 0.80,
            "eligible": True,
            "rank": 1,
        },
        {
            "candidate_id": "candidate-a",
            "predicted_property": 0.95,
            "eligible": True,
            "rank": 2,
        },
        {
            "candidate_id": "candidate-excluded",
            "predicted_property": 0.99,
            "eligible": False,
            "rank": None,
        },
    ]
    ranking = bind_publication(
        {
            "schema_version": "structured_dataset_ranking_publication.v1",
            "project_id": project_id,
            "run_id": run_id,
            "prediction_publication_digest": prediction["publication_digest"],
            "validation_publication_digest": validation["publication_digest"],
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "ranking_configuration": ranking_configuration,
            "ranking_digest": digest_json(
                {"config": ranking_configuration, "rows": ranking_rows}
            ),
            "ranked_candidates": ranking_rows,
        },
        digest_field="publication_digest",
    )

    def topn_row(
        candidate_id: str,
        canonical_smiles: str,
        predicted_property: float,
        rank: int,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "canonical_smiles": canonical_smiles,
            "inchi": f"InChI=1S/{candidate_id}",
            "inchikey": f"KEY-{candidate_id}",
            "predicted_property": predicted_property,
            "rank": rank,
            "model_binding": _digest({"model": "model-1"}),
            "generation_binding": _digest({"generation": "generation-1"}),
            "nearest_neighbor_identity": "training-1",
            "nearest_neighbor_similarity": 0.42,
            "scaffold_novelty": "novel",
            "ad_ood_status": "AD",
            "validation_findings": [],
            "ranking_binding": ranking["ranking_digest"],
        }
        row["provenance_digest"] = digest_json(row)
        return row

    top_rows = [
        topn_row("candidate-b", "CCN", 0.80, 1),
        topn_row("candidate-a", "CCO", 0.95, 2),
    ]
    topn = bind_publication(
        {
            "schema_version": "structured_dataset_computational_topn.v1",
            "artifact_name": "Computational Top-N",
            "project_id": project_id,
            "run_id": run_id,
            "prediction_publication_digest": prediction["publication_digest"],
            "ranking_publication_digest": ranking["publication_digest"],
            "validation_publication_digest": validation["publication_digest"],
            "ranking_digest": ranking["ranking_digest"],
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "candidates": top_rows,
        },
        digest_field="publication_digest",
    )
    evidence_bindings = {
        "prediction": prediction["publication_digest"],
        "validation": validation["publication_digest"],
        "ranking": ranking["publication_digest"],
        "topn": topn["publication_digest"],
    }
    evidence = bind_publication(
        {
            "schema_version": "structured_dataset_private_runtime_chain_evidence.v1",
            "project_id": project_id,
            "run_id": run_id,
            "bindings": evidence_bindings,
            "replay_digest": digest_json(evidence_bindings),
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
        },
        digest_field="evidence_digest",
    )
    values = {
        "prediction_publication": prediction,
        "candidate_validation": validation,
        "ranking_publication": ranking,
        "computational_top_n": topn,
        "structured_dataset_canary_evidence": evidence,
    }
    registry = {
        artifact_id: f"structured_dataset_canary/{artifact_id}.json"
        for artifact_id in values
    }
    payloads = {
        registry[artifact_id]: json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for artifact_id, payload in values.items()
    }
    bindings = [
        AgentHarnessVerifiedOutputBinding(
            artifact_id=artifact_id,
            relative_path=registry[artifact_id],
            content_sha256=_bytes_digest(payloads[registry[artifact_id]]),
            size_bytes=len(payloads[registry[artifact_id]]),
            producer_task_id="evaluate_private_structured_dataset_canary_v1",
            verification_class="private-br1-output",
            verifier_version="structured-dataset-canary-v1",
            verifier_digest=_digest({"verifier": "structured-dataset-canary-v1"}),
        )
        for artifact_id in sorted(values)
    ]
    binding_payloads = [item.model_dump(mode="json") for item in bindings]
    terminal_result = {
        "task_id": "evaluate_private_structured_dataset_canary_v1",
        "task_options": evaluation_configuration,
        "source_publication_sha256": _digest({"local": "final-publication"}),
        "artifact_registry_digest": _digest(dict(sorted(registry.items()))),
        "verified_outputs_digest": _digest(binding_payloads),
        "artifact_registry": registry,
        "verified_outputs": binding_payloads,
    }
    return terminal_result, payloads


def test_valid_verified_artifacts_produce_ranked_projection(tmp_path: Path) -> None:
    payloads = {
        "predictions.csv": b"candidate_id,predicted_value\ncandidate-1,0.25\ncandidate-2,0.95\n",
        "prediction_audit.json": _audit_bytes(),
    }
    publication = _publication(
        output_contract="unimol-prediction-output-v1", payloads=payloads
    )
    registry = {
        "unimol_predictions": "remote-executions/slot-a/outputs/committed/payload/predictions.csv",
        "unimol_prediction_audit": "remote-executions/slot-a/outputs/committed/payload/prediction_audit.json",
    }

    projection = _service(tmp_path).project_verified_publication(
        project_id="project-1",
        run_id="run-1",
        publication=publication,
        artifact_registry=registry,
        artifact_reader=payloads,
    )

    assert projection.task_type == "predict_private_unimol_v1"
    assert projection.summary_statistics.candidate_count == 2
    assert [item.candidate_id for item in projection.ranked_candidates] == [
        "candidate-2",
        "candidate-1",
    ]
    assert projection.verification_status == "verified"
    assert "remote-executions" not in json.dumps(projection.model_dump(mode="json"))


def test_modified_verified_artifact_is_rejected(tmp_path: Path) -> None:
    payloads = {
        "predictions.csv": b"candidate_id,predicted_value\ncandidate-1,0.25\n",
        "prediction_audit.json": _audit_bytes(),
    }
    publication = _publication(
        output_contract="unimol-prediction-output-v1", payloads=payloads
    )
    tampered = dict(payloads)
    tampered["predictions.csv"] = b"candidate_id,predicted_value\ncandidate-1,9.99\n"
    service = _service(tmp_path)

    with pytest.raises(ScientificAgentResultProjectionError, match="digest changed"):
        service.project_verified_publication(
            project_id="project-1",
            run_id="run-1",
            publication=publication,
            artifact_registry={
                "unimol_predictions": "remote-executions/slot-a/outputs/committed/payload/predictions.csv",
                "unimol_prediction_audit": "remote-executions/slot-a/outputs/committed/payload/prediction_audit.json",
            },
            artifact_reader=tampered,
        )


def test_artifact_registry_path_must_match_publication_identity(tmp_path: Path) -> None:
    payloads = {
        "predictions.csv": b"candidate_id,predicted_value\ncandidate-1,0.25\n",
        "prediction_audit.json": _audit_bytes(),
    }
    publication = _publication(
        output_contract="unimol-prediction-output-v1", payloads=payloads
    )

    with pytest.raises(ScientificAgentResultProjectionError, match="Registry path"):
        _service(tmp_path).project_verified_publication(
            project_id="project-1",
            run_id="run-1",
            publication=publication,
            artifact_registry={
                "unimol_predictions": "remote-executions/slot-a/outputs/committed/payload/other.csv",
                "unimol_prediction_audit": "remote-executions/slot-a/outputs/committed/payload/prediction_audit.json",
            },
            artifact_reader=payloads,
        )


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    payloads = {
        "predictions.csv": b"candidate_id\ncandidate-1\n",
        "prediction_audit.json": _audit_bytes(),
    }
    publication = _publication(
        output_contract="unimol-prediction-output-v1", payloads=payloads
    )

    with pytest.raises(ScientificAgentResultProjectionError, match="result contract"):
        _service(tmp_path).project_verified_publication(
            project_id="project-1",
            run_id="run-1",
            publication=publication,
            artifact_registry={
                "unimol_predictions": "remote-executions/slot-a/outputs/committed/payload/predictions.csv",
                "unimol_prediction_audit": "remote-executions/slot-a/outputs/committed/payload/prediction_audit.json",
            },
            artifact_reader=payloads,
        )


def test_reinvent4_projection_replay_has_identical_digest(tmp_path: Path) -> None:
    payloads = {"candidates.csv": b"SMILES,score\nCCO,0.7\nCCN,0.9\n"}
    publication = _publication(
        output_contract="reinvent4-generation-output-v1", payloads=payloads
    )
    registry = {
        "reinvent4_candidates": "remote-executions/slot-b/outputs/committed/payload/candidates.csv",
    }
    service = _service(tmp_path)
    first = service.project_verified_publication(
        project_id="project-1",
        run_id="run-1",
        publication=publication,
        artifact_registry=registry,
        artifact_reader=payloads,
    )
    replay = service.project_verified_publication(
        project_id="project-1",
        run_id="run-1",
        publication=publication,
        artifact_registry=registry,
        artifact_reader=payloads,
    )

    assert replay.projection_id == first.projection_id
    assert replay.projection_digest == first.projection_digest
    assert [item.smiles for item in replay.ranked_candidates] == ["CCN", "CCO"]
    assert service.read_projection(
        project_id="project-1", run_id="run-1", projection_id=first.projection_id
    ).projection_digest == first.projection_digest


def test_br1_final_projection_uses_authoritative_topn_without_resorting(
    tmp_path: Path,
) -> None:
    terminal_result, payloads = _final_result_fixture()
    service = _service(tmp_path)

    first = service.project_verified_br1_final_result(
        project_id="project-1",
        run_id="run-1",
        terminal_result=terminal_result,
        artifact_reader=payloads,
    )
    replay = service.project_verified_br1_final_result(
        project_id="project-1",
        run_id="run-1",
        terminal_result=terminal_result,
        artifact_reader=payloads,
    )

    assert first.task_type == "evaluate_private_structured_dataset_canary_v1"
    assert first.output_contract == "computational-top-n-v1"
    assert first.summary_statistics.top_n == 2
    assert first.summary_statistics.candidate_count == 2
    assert [item.candidate_id for item in first.ranked_candidates] == [
        "candidate-b",
        "candidate-a",
    ]
    assert [item.rank for item in first.ranked_candidates] == [1, 2]
    assert first.projection_id == replay.projection_id
    assert first.projection_digest == replay.projection_digest
    assert "structured_dataset_canary/" not in json.dumps(
        first.model_dump(mode="json"), ensure_ascii=False
    )


def test_br1_final_projection_rejects_modified_artifact_and_topn_mismatch(
    tmp_path: Path,
) -> None:
    terminal_result, payloads = _final_result_fixture()
    service = _service(tmp_path)

    tampered = dict(payloads)
    topn_path = terminal_result["artifact_registry"]["computational_top_n"]
    tampered[topn_path] = tampered[topn_path] + b"\n"
    with pytest.raises(ScientificAgentResultProjectionError, match="digest changed"):
        service.project_verified_br1_final_result(
            project_id="project-1",
            run_id="run-1",
            terminal_result=terminal_result,
            artifact_reader=tampered,
            persist=False,
        )

    mismatch = dict(terminal_result)
    mismatch["task_options"] = {"top_n": 5, "validation_seed": 1729}
    with pytest.raises(ScientificAgentResultProjectionError, match="authorized top_n"):
        service.project_verified_br1_final_result(
            project_id="project-1",
            run_id="run-1",
            terminal_result=mismatch,
            artifact_reader=payloads,
            persist=False,
        )
