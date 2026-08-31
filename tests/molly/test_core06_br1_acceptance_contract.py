"""Schema and binding regressions for the CORE-06 macro acceptance path."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from molly.core import ArtifactStore, RunLedger
from molly.core.ids import canonical_json_bytes
from molly.plugins.br1_inverse_design import Br1PluginConfig, DatasetGate, migrate_real_csv
from molly.plugins.br1_inverse_design.schema import EvaluationConfig
from molly.plugins.remote_compute import JobHandle
from tests.molly.core06_real_acceptance import HostConfig, _public_handle


def test_migrated_dataset_retains_exact_source_and_transformation_binding(tmp_path: Path) -> None:
    source = b"Chromophore,Quantum yield,Solvent,Reference\nCCO,0.2,water,doi:one\n"
    migrated = migrate_real_csv(
        source,
        historical_acceptance_id="br1-real-acceptance:historical",
        historical_review_basis="historical accepted real BR1 evidence; imported without recreating review history",
    )
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put(
        migrated.content,
        media_type="application/json",
        schema_name="molly.br1.migrated-reviewed-dataset",
        schema_version="1",
    )
    inspection = DatasetGate(store).inspect(artifact.artifact_id, target_property="quantum_yield")
    assert inspection.review_status == "MIGRATED_ACCEPTED_REAL_DATASET"
    assert inspection.source_content_digest == migrated.source_content_digest
    assert inspection.transformation_digest == migrated.transformation_digest
    assert inspection.row_count == 1
    assert "ReviewRecord" not in store.read(artifact.artifact_id).decode("utf-8")


def test_evaluation_config_digest_changes_when_execution_meaning_changes() -> None:
    maximum = EvaluationConfig(top_n=3, direction="MAX")
    minimum = EvaluationConfig(top_n=3, direction="MIN")
    larger = EvaluationConfig(top_n=4, direction="MAX")
    assert maximum.digest != minimum.digest
    assert maximum.digest != larger.digest
    assert b"claim_boundary" in canonical_json_bytes(maximum.to_dict())


def test_plugin_config_is_server_owned_and_target_catalog_is_closed() -> None:
    config = Br1PluginConfig(supported_target_properties=("quantum_yield", "emission_max_nm"))
    assert config.digest
    assert config.supported_target_properties == ("quantum_yield", "emission_max_nm")
    with pytest.raises(Exception):
        Br1PluginConfig(supported_target_properties=())


def test_real_acceptance_public_job_projection_is_secret_free(tmp_path: Path) -> None:
    handle = JobHandle(
        job_id="job_contract",
        profile_id="profile:br1-remote",
        profile_digest="a" * 64,
        task_digest="b" * 64,
        idempotency_key="c" * 64,
        input_artifact_ids=(),
        execution_config_digest="d" * 64,
        submitted_at="2026-08-31T00:00:00Z",
    )
    projected = _public_handle(handle)
    assert projected["job_id"] == handle.job_id
    assert "credential" not in json.dumps(projected).lower()
    assert "host" not in json.dumps(projected).lower()


def test_real_acceptance_requires_server_owned_remote_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("Chromophore,Quantum yield\nCCO,0.2\n", encoding="utf-8")
    with pytest.raises(Exception):
        HostConfig(
            source_path=source,
            ssh_target="",
            remote_root="",
            unimol_python="",
            reinvent_python="",
            reinvent_repository="",
        )
