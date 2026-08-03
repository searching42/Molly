from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from ai4s_agent.remote_output_contracts import (
    verify_remote_output_contents,
    verify_remote_output_contract,
)


def _artifact(
    artifact_id: str,
    relative_path: str,
    media_type: str,
    payload: bytes,
) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id=artifact_id,
        relative_path=relative_path,
        media_type=media_type,
        size_bytes=len(payload),
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def test_unimol_training_v2_requires_prediction_capable_model_directory() -> None:
    payloads = {
        "model/config.yaml": b"task: regression\ntarget_cols: target_value\n",
        "model/model_0.pth": b"fresh-weights",
        "model/target_scaler.ss": b"fresh-scaler",
        "model/training_audit.json": (
            b'{"schema_version":"unimol_training_audit.v1",'
            b'"provider_version":"0.1.5","config":{"seed":1729}}\n'
        ),
        "model/training_metrics.json": b'{"metrics":{"mae":0.1}}\n',
    }
    artifacts = [
        _artifact("unimol_model_config", "model/config.yaml", "application/yaml", payloads["model/config.yaml"]),
        _artifact("unimol_model_weights", "model/model_0.pth", "application/octet-stream", payloads["model/model_0.pth"]),
        _artifact("unimol_target_scaler", "model/target_scaler.ss", "application/octet-stream", payloads["model/target_scaler.ss"]),
        _artifact("unimol_training_audit", "model/training_audit.json", "application/json", payloads["model/training_audit.json"]),
        _artifact("unimol_training_metrics", "model/training_metrics.json", "application/json", payloads["model/training_metrics.json"]),
    ]

    verify_remote_output_contract("unimol-training-output-v2", artifacts)
    verify_remote_output_contents(
        "unimol-training-output-v2", artifacts, payloads.__getitem__
    )

    with pytest.raises(ValueError, match="artifact roster"):
        verify_remote_output_contract("unimol-training-output-v2", artifacts[:-1])


def test_unimol_prediction_output_requires_exact_candidate_binding_header() -> None:
    payloads = {
        "predictions.csv": b"candidate_id,predicted_value\ncandidate-1,0.5\n",
        "prediction_audit.json": (
            b'{"schema_version":"unimol_prediction_audit.v1",'
            b'"provider_version":"0.1.5","config":{}}\n'
        ),
    }
    artifacts = [
        _artifact("unimol_predictions", "predictions.csv", "text/csv", payloads["predictions.csv"]),
        _artifact("unimol_prediction_audit", "prediction_audit.json", "application/json", payloads["prediction_audit.json"]),
    ]

    verify_remote_output_contents(
        "unimol-prediction-output-v1", artifacts, payloads.__getitem__
    )
    replaced = payloads | {
        "predictions.csv": b"SMILES,predicted_value\nCC,0.5\n"
    }
    with pytest.raises(ValueError, match="predictions CSV"):
        verify_remote_output_contents(
            "unimol-prediction-output-v1", artifacts, replaced.__getitem__
        )


def test_reinvent4_v2_requires_provider_and_effective_config_audit() -> None:
    payloads = {
        "candidates.csv": b"SMILES,score\nCC,0.5\n",
        "generation_audit.json": (
            b'{"schema_version":"reinvent4_generation_audit.v1",'
            b'"provider_version":"4.7.15",'
            b'"effective_config_digest":"sha256:' + b"a" * 64 + b'",'
            b'"seed":1729}\n'
        ),
    }
    artifacts = [
        _artifact(
            "reinvent4_candidates",
            "candidates.csv",
            "text/csv",
            payloads["candidates.csv"],
        ),
        _artifact(
            "reinvent4_generation_audit",
            "generation_audit.json",
            "application/json",
            payloads["generation_audit.json"],
        ),
    ]

    verify_remote_output_contents(
        "reinvent4-generation-output-v2", artifacts, payloads.__getitem__
    )
