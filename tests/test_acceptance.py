from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import ArtifactRef, RunStatus, StageHistoryItem, StageState
from ai4s_agent.storage import ProjectStorage


def test_stage_history_roundtrip_through_project_storage(tmp_path: Path) -> None:
    ps = ProjectStorage(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"

    history = [
        StageHistoryItem(stage="parse", status=RunStatus.SUCCEEDED, updated_at=now_iso()),
        StageHistoryItem(stage="clean", status=RunStatus.SUCCEEDED, updated_at=now_iso()),
        StageHistoryItem(stage="train", status=RunStatus.RUNNING, updated_at=now_iso(), note="waiting for GPU"),
    ]
    state = StageState(
        stage="train",
        next_stage="predict",
        status=RunStatus.RUNNING,
        started_at=now_iso(),
        updated_at=now_iso(),
        history=history,
        artifacts=[ArtifactRef(artifact_id="model", relative_path="03_training/model.pkl")],
    )
    ps.write_stage_state(project_id, run_id, state)

    loaded = ps.read_stage_state(project_id, run_id)
    assert loaded is not None
    assert loaded.stage == "train"
    assert loaded.next_stage == "predict"
    assert loaded.status == RunStatus.RUNNING
    assert len(loaded.history) == 3
    assert loaded.history[0].stage == "parse"
    assert loaded.history[0].status == RunStatus.SUCCEEDED
    assert loaded.history[2].note == "waiting for GPU"
    assert len(loaded.artifacts) == 1
    assert loaded.artifacts[0].artifact_id == "model"


def test_stage_history_no_unexpected_jumps(tmp_path: Path) -> None:
    """Stage history must not contain skipped statuses between completed stages."""
    ps = ProjectStorage(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"

    stages_in_order = [
        ("parse", RunStatus.SUCCEEDED),
        ("validate", RunStatus.SUCCEEDED),
        ("clean", RunStatus.SUCCEEDED),
        ("trainability", RunStatus.SUCCEEDED),
        ("train", RunStatus.SUCCEEDED),
    ]
    history = [
        StageHistoryItem(stage=name, status=status, updated_at=now_iso())
        for name, status in stages_in_order
    ]
    state = StageState(
        stage="train",
        next_stage="predict",
        status=RunStatus.SUCCEEDED,
        started_at=now_iso(),
        updated_at=now_iso(),
        history=history,
    )
    ps.write_stage_state(project_id, run_id, state)

    loaded = ps.read_stage_state(project_id, run_id)
    succeeded_stages = [h.stage for h in loaded.history if h.status == RunStatus.SUCCEEDED]
    # Verify forward-only progression: each stage appears at most once and in order
    assert succeeded_stages == ["parse", "validate", "clean", "trainability", "train"]


def test_report_artifact_paths_are_registered(tmp_path: Path) -> None:
    ps = ProjectStorage(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"

    report_artifacts = {
        "data_overview": "02_data/dataset_overview.md",
        "trainability_report": "02_data/trainability_report.json",
        "baseline_report": "03_training/baseline_report.json",
        "backend_recommendation": "03_training/backend_recommendation.md",
        "training_report": "03_training/training_report.json",
        "screening_report": "04_screening/screening_report.md",
        "final_summary": "05_report/final_summary.md",
    }
    for artifact_id, relative_path in report_artifacts.items():
        ps.register_artifact_path(project_id, run_id, artifact_id, relative_path)

    registry = ps.read_artifact_registry(project_id, run_id)
    assert len(registry) == len(report_artifacts)
    for artifact_id, relative_path in report_artifacts.items():
        assert registry[artifact_id] == relative_path


def test_report_json_artifacts_have_matching_md_counterpart(tmp_path: Path) -> None:
    """Every JSON report artifact should have a corresponding MD artifact."""
    ps = ProjectStorage(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"

    ps.register_artifact_path(project_id, run_id, "baseline_json", "03_training/baseline_report.json")
    ps.register_artifact_path(project_id, run_id, "baseline_md", "03_training/baseline_report.md")
    ps.register_artifact_path(project_id, run_id, "trainability_json", "02_data/trainability_report.json")
    ps.register_artifact_path(project_id, run_id, "trainability_md", "02_data/trainability_report.md")

    registry = ps.read_artifact_registry(project_id, run_id)
    json_artifacts = [path for path in registry.values() if path.endswith(".json")]
    md_artifacts = [path for path in registry.values() if path.endswith(".md")]

    # Each JSON should have a matching MD with same stem
    json_stems = {Path(p).stem for p in json_artifacts}
    md_stems = {Path(p).stem for p in md_artifacts}
    assert json_stems == md_stems, f"Mismatch: JSON stems {json_stems}, MD stems {md_stems}"


def test_asset_manifest_schema_version_is_set(tmp_path: Path) -> None:
    from ai4s_agent.schemas import AssetManifest, AssetStatus

    manifest = AssetManifest(
        asset_id="model/baseline_rf/logp",
        asset_type="trained_model",
        version="v001",
        status=AssetStatus.CANDIDATE,
        created_from_run_id="run-1",
        source_artifacts=["03_training/model.pkl"],
        content_hash="abc123",
    )
    assert manifest.schema_version == "1.0"
    data = manifest.model_dump()
    assert data["schema_version"] == "1.0"


def test_stage_state_error_structure_is_retained(tmp_path: Path) -> None:
    ps = ProjectStorage(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"

    state = StageState(
        stage="train",
        status=RunStatus.FAILED,
        started_at=now_iso(),
        updated_at=now_iso(),
        error={
            "category": "REMOTE",
            "reason": "GPU OOM",
            "retryable": True,
            "suggested_action": "reduce batch size",
        },
        history=[StageHistoryItem(stage="train", status=RunStatus.FAILED, updated_at=now_iso())],
    )
    ps.write_stage_state(project_id, run_id, state)

    loaded = ps.read_stage_state(project_id, run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.FAILED
    assert loaded.error is not None
    assert loaded.error["category"] == "REMOTE"
    assert loaded.error["retryable"] is True
    assert loaded.error["suggested_action"] == "reduce batch size"


def test_stage_state_empty_history_is_valid(tmp_path: Path) -> None:
    ps = ProjectStorage(workspace_dir=tmp_path)
    state = StageState(
        stage="parse",
        status=RunStatus.PENDING,
        started_at=now_iso(),
        updated_at=now_iso(),
    )
    ps.write_stage_state("proj-1", "run-1", state)

    loaded = ps.read_stage_state("proj-1", "run-1")
    assert loaded is not None
    assert loaded.history == []
    assert loaded.artifacts == []


def test_multiple_run_stage_histories_are_independent(tmp_path: Path) -> None:
    ps = ProjectStorage(workspace_dir=tmp_path)

    ps.write_stage_state("proj-1", "run-1", StageState(
        stage="train", status=RunStatus.SUCCEEDED,
        started_at=now_iso(), updated_at=now_iso(),
        history=[StageHistoryItem(stage="parse", status=RunStatus.SUCCEEDED, updated_at=now_iso())],
    ))
    ps.write_stage_state("proj-1", "run-2", StageState(
        stage="clean", status=RunStatus.FAILED,
        started_at=now_iso(), updated_at=now_iso(),
        history=[StageHistoryItem(stage="parse", status=RunStatus.SUCCEEDED, updated_at=now_iso())],
    ))

    r1 = ps.read_stage_state("proj-1", "run-1")
    r2 = ps.read_stage_state("proj-1", "run-2")
    assert r1.stage == "train"
    assert r1.status == RunStatus.SUCCEEDED
    assert r2.stage == "clean"
    assert r2.status == RunStatus.FAILED


def test_manual_real_unimol_acceptance_checklist_exists() -> None:
    checklist = Path(__file__).resolve().parents[1] / "docs" / "manual-real-unimol-acceptance.md"
    assert checklist.exists()
    text = checklist.read_text(encoding="utf-8")

    required_phrases = [
        "Manual Real Uni-Mol Acceptance Checklist",
        "AI4S_WORKSPACE",
        "train_model_unimol_legacy_adapter",
        "predict_candidates_unimol_legacy_adapter",
        "model_metadata",
        "stage.json",
        "artifact_registry.json",
        "asset promotion",
        "rollback",
        "acceptance sign-off",
    ]
    for phrase in required_phrases:
        assert phrase in text

    assert text.count("- [ ]") >= 20
