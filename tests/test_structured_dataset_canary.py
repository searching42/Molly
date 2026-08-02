from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.harness_tracing import NoopHarnessTracer
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    RecoveryRequiredError,
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
    validate_candidates,
)
from tests.test_structured_dataset_confirmation import NOW, dataset_bytes, fixture_rows


def service(tmp_path: Path, *, tracer=None) -> tuple[StructuredDatasetCanaryService, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Fixture", created_at=NOW)
    return (
        StructuredDatasetCanaryService(
            storage=storage,
            trusted_actors={"test-actor"},
            tracer=tracer,
            clock=lambda: NOW,
        ),
        source,
    )


@pytest.mark.pr_fast
def test_ci_reference_canary_fresh_end_to_end_and_exact_replay(tmp_path: Path) -> None:
    canary, source = service(tmp_path)

    first = canary.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=7, top_n=5, created_at=NOW,
    )
    second = canary.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=7, top_n=5, created_at=NOW,
    )

    assert first.replayed is False
    assert second.replayed is True
    assert first.evidence["evidence_digest"] == second.evidence["evidence_digest"]
    assert first.computational_top_n["artifact_name"] == "Computational Top-N"
    assert first.computational_top_n["claim_boundary"].startswith("Model-ranked Computational Candidates")
    assert first.computational_top_n["scientific_scope"]["scope_downgraded"] is True
    assert first.evidence["private_real_tool_completed"] is False
    assert first.evidence["test_mode"] == "ci_reference"
    assert first.evidence["privacy_findings"]["environment_locator_count"] == 0
    assert len(first.computational_top_n["candidates"]) <= 5
    registry = canary.storage.read_artifact_registry("project-1", "run-1")
    assert {
        "raw_dataset", "review_snapshot", "confirmation_receipt", "confirmed_dataset",
        "model_package", "generation_publication", "prediction_publication",
        "candidate_validation", "ranking_publication", "computational_top_n",
    }.issubset(registry)
    model = json.loads(
        (canary.storage.run_dir("project-1", "run-1") / registry["model_package"]).read_text()
    )
    generation = json.loads(
        (canary.storage.run_dir("project-1", "run-1") / registry["generation_publication"]).read_text()
    )
    assert model["fresh_training"] is True
    assert model["run_id"] == "run-1"
    assert model["metrics"]["train_count"] >= 3
    assert generation["existing_output_used"] is False
    assert generation["model_package_digest"] == model["publication_digest"]


def test_fresh_process_determinism_and_seed_binding(tmp_path: Path) -> None:
    first, source1 = service(tmp_path / "a")
    second, source2 = service(tmp_path / "b")
    result1 = first.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source1,
        actor="test-actor", seed=19, created_at=NOW,
    )
    result2 = second.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source2,
        actor="test-actor", seed=19, created_at=NOW,
    )

    assert result1.evidence["replay_digest"] == result2.evidence["replay_digest"]
    assert result1.computational_top_n["publication_digest"] == result2.computational_top_n["publication_digest"]


def test_restart_after_training_checkpoint_does_not_refit_or_copy_old_model(tmp_path: Path) -> None:
    canary, source = service(tmp_path)
    with pytest.raises(StructuredDatasetCanaryError, match="training_checkpoint"):
        canary.run_ci_reference(
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor", seed=23, created_at=NOW, fault_after="training_checkpoint",
        )
    checkpoint = canary._path("project-1", "run-1", "model_checkpoint.json")
    original = checkpoint.read_bytes()

    result = canary.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=23, created_at=NOW,
    )

    assert result.computational_top_n["run_id"] == "run-1"
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("boundary", ["model_publication", "generation_publication"])
def test_restart_adopts_exact_publication_without_reexecution(
    tmp_path: Path, boundary: str
) -> None:
    canary, source = service(tmp_path)
    with pytest.raises(StructuredDatasetCanaryError, match=boundary):
        canary.run_ci_reference(
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor", seed=29, created_at=NOW, fault_after=boundary,
        )
    publication_name = "model_package.json" if boundary == "model_publication" else "generation.json"
    publication = canary._path("project-1", "run-1", publication_name)
    original = publication.read_bytes()

    result = canary.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=29, created_at=NOW,
    )

    assert result.computational_top_n["artifact_name"] == "Computational Top-N"
    assert publication.read_bytes() == original


def test_generation_unknown_outcome_never_redispatches(tmp_path: Path) -> None:
    canary, source = service(tmp_path)
    with pytest.raises(StructuredDatasetCanaryError, match="generation_request"):
        canary.run_ci_reference(
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor", seed=31, created_at=NOW, fault_after="generation_request",
        )
    dispatch = canary._path("project-1", "run-1", "generation_dispatch.json")
    dispatch.write_text(
        json.dumps(
            {
                "schema_version": "structured_dataset_generation_dispatch.v1",
                "request_digest": json.loads(canary._path("project-1", "run-1", "generation_request.json").read_text()),
                "dispatch_id": "dispatch-unknown",
                "outcome": "unknown",
            }
        )
    )
    before = dispatch.read_bytes()

    with pytest.raises(RecoveryRequiredError, match="exact reconciliation"):
        canary.run_ci_reference(
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor", seed=31, created_at=NOW,
        )
    assert dispatch.read_bytes() == before


def test_stale_model_and_existing_output_fail_closed(tmp_path: Path) -> None:
    canary, source = service(tmp_path)
    canary.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=37, created_at=NOW,
    )
    model_path = canary._path("project-1", "run-1", "model_package.json")
    payload = json.loads(model_path.read_text())
    payload["run_id"] = "old-run"
    model_path.chmod(0o600)
    model_path.write_text(json.dumps(payload))

    with pytest.raises(Exception):
        canary.run_ci_reference(
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor", seed=37, created_at=NOW,
        )


def test_chemical_validation_keeps_invalid_duplicate_and_ood_findings() -> None:
    candidates = [
        {"candidate_id": "c1", "smiles": "not-a-smiles"},
        {"candidate_id": "c2", "smiles": "CCO"},
        {"candidate_id": "c3", "smiles": "OCC"},
        {"candidate_id": "c4", "smiles": "c1ccccc1"},
    ]
    results, summary = validate_candidates(
        candidates, fixture_rows(), seed=41, ad_similarity_threshold=0.95,
    )

    assert len(results) == len(candidates)
    assert results[0]["valid"] is False
    assert results[2]["duplicate"] is True
    assert results[3]["training_exact_duplicate"] is True
    assert summary["no_silent_candidate_loss"] is True
    assert summary["ood_count"] >= 1


class FailingTracer(NoopHarnessTracer):
    def start_span(self, *args, **kwargs):
        raise RuntimeError("private /path host 192.0.2.1 token stdout stderr")


def test_telemetry_failure_does_not_change_authority(tmp_path: Path) -> None:
    baseline, source1 = service(tmp_path / "baseline")
    failing, source2 = service(tmp_path / "failing", tracer=FailingTracer())
    expected = baseline.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source1,
        actor="test-actor", seed=43, created_at=NOW,
    )

    actual = failing.run_ci_reference(
        project_id="project-1", run_id="run-1", raw_csv=source2,
        actor="test-actor", seed=43, created_at=NOW,
    )
    assert expected.evidence["replay_digest"] == actual.evidence["replay_digest"]
