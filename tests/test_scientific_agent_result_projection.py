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
from ai4s_agent.scientific_agent_result_projection import (
    ScientificAgentResultProjectionError,
    ScientificAgentResultProjectionService,
)
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
