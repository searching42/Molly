from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from ai4s_agent.structured_dataset_private_canary import (
    PrivateRealToolCanaryRequest,
    PrivateRealToolConfigurationError,
    validate_private_request,
)


def request() -> PrivateRealToolCanaryRequest:
    digest = "sha256:" + "a" * 64
    return PrivateRealToolCanaryRequest(
        project_id="project-1",
        run_id="run-1",
        raw_dataset_id="raw-1",
        raw_dataset_digest=digest,
        confirmation_receipt_id="confirmation-1",
        confirmation_receipt_digest=digest,
        confirmed_dataset_id="confirmed-1",
        confirmed_dataset_digest=digest,
        training_profile_id="unimol-training",
        generation_profile_id="reinvent4-generation",
        training_seed=7,
        generation_seed=11,
        unimol_provider_version="unimol-v1",
        reinvent4_version="reinvent4-v4",
        reinvent4_config_digest=digest,
    )


def test_private_request_freezes_real_tool_and_no_reuse_policy() -> None:
    payload = request().to_publication()

    assert payload["training"]["fresh_training_required"] is True
    assert payload["generation"]["real_execution_required"] is True
    assert payload["reuse_policy"] == {
        "old_model": False,
        "old_prediction": False,
        "old_generated_candidates": False,
        "existing_output": False,
    }
    assert payload["telemetry_authoritative"] is False
    assert "request_digest" in payload


@pytest.mark.parametrize(
    "mutation",
    [
        {"training": {"provider": "baseline", "fresh_training_required": True}},
        {"generation": {"provider": "stub", "real_execution_required": True}},
        {"reuse_policy": {"old_model": True}},
        {"private_note": "endpoint ssh stdout token /private/path"},
    ],
)
def test_private_request_rejects_fake_tools_reuse_and_environment_data(mutation: dict) -> None:
    payload = request().to_publication()
    payload.update(mutation)
    with pytest.raises(PrivateRealToolConfigurationError):
        validate_private_request(payload)


def test_ci_evidence_schema_accepts_real_canary_evidence(tmp_path: Path) -> None:
    from ai4s_agent.storage import ProjectStorage
    from ai4s_agent.structured_dataset_canary_harness import run_structured_dataset_ci_harness
    from tests.test_structured_dataset_confirmation import NOW, dataset_bytes

    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Fixture", created_at=NOW)
    evidence = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=3,
    ).evidence
    schema = json.loads(
        (Path("docs/schemas") / "structured_dataset_canary_evidence.schema.json").read_text()
    )

    jsonschema.validate(evidence, schema)
