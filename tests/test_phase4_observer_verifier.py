from pathlib import Path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.agents.observer import ObserverAgent
from ai4s_agent.agents.verifier import VerifierAgent
from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import ArtifactRef, AssetManifest, AssetStatus, RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


def test_observer_agent_collects_stage_logs_reports_artifacts_and_manifests(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    jobs = JobManager(tmp_path / "runs")
    project_id = "proj-observe"
    run_id = "run-observe"
    now = now_iso()
    run_dir = storage.run_dir(project_id, run_id)
    report_path = write_json(
        run_dir / "outputs" / "extraction_confidence_report.json",
        {
            "run_id": run_id,
            "attempted_hit_count": 2,
            "extracted_record_count": 1,
            "rejected_record_count": 1,
            "high_confidence_count": 1,
            "medium_confidence_count": 0,
            "low_confidence_count": 0,
            "confidence_threshold": 0.7,
            "generated_at": now,
        },
    )
    storage.register_artifact_path(
        project_id,
        run_id,
        "extraction_confidence_report",
        str(report_path.relative_to(run_dir)),
    )
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="extract_records",
            status=RunStatus.RUNNING,
            started_at=now,
            updated_at=now,
            artifacts=[
                ArtifactRef(
                    artifact_id="extraction_confidence_report",
                    relative_path=str(report_path.relative_to(run_dir)),
                    producer_task_id="extract_records",
                )
            ],
        ),
    )
    jobs.add_log(run_id, "WARN", "extractor", "low confidence record rejected")
    manifest = AssetManifest(
        asset_id="dataset/confirmed",
        asset_type="training_dataset",
        version="v001",
        status=AssetStatus.CANDIDATE,
        created_from_run_id=run_id,
        source_artifacts=["outputs/extraction_confidence_report.json"],
        content_hash="hash-observe",
    )
    storage.write_asset_manifest(project_id, ["datasets", "confirmed"], "v001", manifest)

    observation = ObserverAgent(storage=storage, jobs=jobs).observe_run(project_id, run_id)

    assert observation.stage_state is not None
    assert observation.stage_state.stage == "extract_records"
    assert observation.artifacts[0].artifact_id == "extraction_confidence_report"
    assert observation.artifacts[0].exists is True
    assert observation.reports["extraction_confidence_report"]["extracted_record_count"] == 1
    assert observation.logs[0]["message"] == "low confidence record rejected"
    assert observation.asset_manifests[0].asset_id == "dataset/confirmed"


def test_observer_agent_treats_artifact_deleted_during_stat_as_missing(tmp_path, monkeypatch) -> None:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("proj-race", "run-race")
    artifact = write_json(run_dir / "volatile.json", {"ok": True})
    original_stat = Path.stat
    calls = {"count": 0}

    def flaky_stat(self: Path, *args, **kwargs):
        if self == artifact:
            calls["count"] += 1
            if calls["count"] >= 3:
                raise FileNotFoundError(str(self))
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    observed = ObserverAgent(storage=storage)._artifact_ref(run_dir, "volatile", "volatile.json", None)

    assert observed.exists is False
    assert observed.size_bytes == 0


def test_observer_agent_preserves_report_key_collisions(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("proj-collision", "run-collision")
    write_json(run_dir / "extraction_report.json", {"source": "plain"})
    write_json(run_dir / "run-collision_extraction_report.json", {"source": "prefixed"})

    observation = ObserverAgent(storage=storage).observe_run("proj-collision", "run-collision")

    assert observation.reports["extraction_report"]["source"] == "plain"
    assert observation.reports["run-collision_extraction_report"]["source"] == "prefixed"


def test_observer_agent_uses_canonical_report_key_even_when_prefixed_report_is_seen_first(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("proj-collision-subdir", "run-collision-subdir")
    write_json(run_dir / "00_prefixed" / "run-collision-subdir_extraction_report.json", {"source": "prefixed"})
    write_json(run_dir / "zz_plain" / "extraction_report.json", {"source": "plain"})

    observation = ObserverAgent(storage=storage).observe_run("proj-collision-subdir", "run-collision-subdir")

    assert observation.reports["extraction_report"]["source"] == "plain"
    assert observation.reports["run-collision-subdir_extraction_report"]["source"] == "prefixed"


def test_observer_agent_preserves_injected_approval_metadata(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("proj-approval-meta", "run-approval-meta")
    write_json(
        run_dir / "gate_decisions.json",
        {
            "decisions": [
                {
                    "gate": "gate_1_task_parse",
                    "approved": True,
                    "approval_type": "spoofed",
                    "source_file": "spoofed.json",
                }
            ]
        },
    )
    write_json(
        run_dir / "asset_promotion_records.json",
        {
            "records": [
                {
                    "asset_id": "dataset/confirmed",
                    "approved_at": "2026-06-04T10:00:00Z",
                    "approval_type": "spoofed",
                    "source_file": "spoofed.json",
                }
            ]
        },
    )

    observation = ObserverAgent(storage=storage).observe_run("proj-approval-meta", "run-approval-meta")

    assert observation.approval_records[0]["approval_type"] == "gate"
    assert observation.approval_records[0]["source_file"] == "gate_decisions.json"
    assert observation.approval_records[1]["approval_type"] == "asset_promotion"
    assert observation.approval_records[1]["source_file"] == "asset_promotion_records.json"


def test_verifier_agent_flags_common_failure_modes_and_writes_reports(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-verify"
    run_id = "run-verify"
    now = now_iso()
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "run_id": run_id,
            "attempted_hit_count": 8,
            "extracted_record_count": 0,
            "rejected_record_count": 8,
            "high_confidence_count": 0,
            "medium_confidence_count": 1,
            "low_confidence_count": 7,
            "confidence_threshold": 0.7,
            "generated_at": now,
        },
    )
    write_json(
        run_dir / "conflict_report.json",
        {
            "run_id": run_id,
            "input_record_count": 10,
            "merged_record_count": 6,
            "conflict_count": 4,
            "non_conflicting_record_count": 6,
            "conflicts": [],
            "generated_at": now,
        },
    )
    write_json(
        run_dir / "unit_normalization_report.json",
        {
            "run_id": run_id,
            "input_record_count": 10,
            "normalized_record_count": 8,
            "conversion_count": 2,
            "warning_count": 2,
            "conversions": [],
            "warnings": [{"record_id": "r1", "reason": "unknown unit"}],
            "generated_at": now,
        },
    )
    write_json(
        run_dir / "citation_license_report.json",
        {
            "run_id": run_id,
            "source_count": 1,
            "evidence_count": 4,
            "extracted_record_count": 0,
            "unknown_license_count": 1,
            "sources": [{"source_id": "s1", "paper_id": "p1", "license": "unknown"}],
            "generated_at": now,
        },
    )
    write_json(
        run_dir / "trainability_report.json",
        {
            "overall_status": "BLOCKED",
            "properties": [{"property_id": "plqy", "effective_labels": 2, "status": "INSUFFICIENT_LABELS"}],
        },
    )
    write_json(
        run_dir / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.2, "mae": 0.7}}]},
    )
    write_json(
        run_dir / "leakage_report.json",
        {
            "train_smiles_column": "smiles",
            "other_smiles_column": "smiles",
            "train_count": 10,
            "other_count": 5,
            "overlap_count": 2,
            "overlap_smiles": ["CCO", "CCN"],
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    verifier = VerifierAgent()
    report = verifier.verify(observation)
    report_json, report_md = verifier.write_reports(storage, project_id, run_id, report)

    categories = {finding.category for finding in report.findings}
    assert {
        "empty_extraction",
        "low_confidence",
        "high_conflict_rate",
        "invalid_units",
        "data_leakage",
        "poor_trainability",
        "abnormal_model_metrics",
        "missing_provenance",
    } <= categories
    assert report.overall_decision == "ask_user"
    assert report_json.exists()
    assert report_md.exists()
    assert "empty_extraction" in report_md.read_text(encoding="utf-8")


def test_verifier_agent_flags_malformed_numeric_report_values_without_crashing(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-malformed"
    run_id = "run-malformed"
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "attempted_hit_count": "N/A",
            "extracted_record_count": "N/A",
            "high_confidence_count": "N/A",
            "low_confidence_count": "N/A",
        },
    )
    write_json(run_dir / "conflict_report.json", {"input_record_count": "N/A", "conflict_count": "N/A"})
    write_json(run_dir / "unit_normalization_report.json", {"warning_count": "N/A"})
    write_json(run_dir / "leakage_report.json", {"overlap_count": "N/A"})
    write_json(run_dir / "citation_license_report.json", {"unknown_license_count": "N/A"})

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "malformed_report" in {finding.category for finding in report.findings}
    assert report.overall_decision == "ask_user"


def test_verifier_agent_flags_boolean_numeric_report_values_as_malformed(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-bool-number"
    run_id = "run-bool-number"
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "attempted_hit_count": 3,
            "extracted_record_count": 3,
            "high_confidence_count": True,
            "low_confidence_count": 0,
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "malformed_report" in {finding.category for finding in report.findings}


def test_verifier_agent_does_not_flag_all_medium_confidence_as_low_confidence(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-medium"
    run_id = "run-medium"
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "attempted_hit_count": 3,
            "extracted_record_count": 3,
            "high_confidence_count": 0,
            "medium_confidence_count": 3,
            "low_confidence_count": 0,
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "low_confidence" not in {finding.category for finding in report.findings}


def test_verifier_agent_flags_stale_approval_records(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-stale"
    run_id = "run-stale"
    run_dir = storage.run_dir(project_id, run_id)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="train_model",
            status=RunStatus.RUNNING,
            started_at="2026-06-04T10:00:00Z",
            updated_at="2026-06-04T10:05:00Z",
        ),
    )
    write_json(
        run_dir / "gate_decisions.json",
        {
            "run_id": run_id,
            "decisions": [
                {
                    "gate": "gate_3_train_config",
                    "approved": True,
                    "actor": "user",
                    "approved_at": "2026-06-04T09:55:00Z",
                }
            ],
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert observation.approval_records[0]["approved_at"] == "2026-06-04T09:55:00Z"
    assert "stale_approval" in {finding.category for finding in report.findings}
    assert report.overall_decision == "ask_user"


def test_verifier_agent_handles_naive_approval_timestamps(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-naive-approval"
    run_id = "run-naive-approval"
    run_dir = storage.run_dir(project_id, run_id)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="train_model",
            status=RunStatus.RUNNING,
            started_at="2026-06-04T10:00:00Z",
            updated_at="2026-06-04T10:05:00Z",
        ),
    )
    write_json(
        run_dir / "gate_decisions.json",
        {
            "run_id": run_id,
            "decisions": [
                {
                    "gate": "gate_3_train_config",
                    "approved": True,
                    "actor": "user",
                    "approved_at": "2026-06-04T09:55:00",
                }
            ],
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "stale_approval" in {finding.category for finding in report.findings}


def test_verifier_agent_flags_approval_when_stage_timestamp_is_missing(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-no-stage-time"
    run_id = "run-no-stage-time"
    run_dir = storage.run_dir(project_id, run_id)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="train_model",
            status=RunStatus.RUNNING,
            started_at="",
            updated_at="2026-06-04T10:05:00Z",
        ),
    )
    write_json(
        run_dir / "gate_decisions.json",
        {
            "run_id": run_id,
            "decisions": [
                {
                    "gate": "gate_3_train_config",
                    "approved": True,
                    "actor": "user",
                    "approved_at": "2026-06-04T09:55:00Z",
                }
            ],
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "stale_approval" in {finding.category for finding in report.findings}


def test_verifier_agent_flags_approval_with_missing_timestamp(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-missing-approval-time"
    run_id = "run-missing-approval-time"
    run_dir = storage.run_dir(project_id, run_id)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="train_model",
            status=RunStatus.RUNNING,
            started_at="2026-06-04T10:00:00Z",
            updated_at="2026-06-04T10:05:00Z",
        ),
    )
    write_json(
        run_dir / "gate_decisions.json",
        {
            "run_id": run_id,
            "decisions": [{"gate": "gate_3_train_config", "approved": True, "actor": "user"}],
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert "stale_approval" in {finding.category for finding in report.findings}
    assert report.findings[0].evidence["reason"] == "approval timestamp is missing or invalid"


def test_verifier_agent_allows_clean_observation_without_literature_provenance_report(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-clean"
    run_id = "run-clean"
    storage.run_dir(project_id, run_id)

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)

    assert report.overall_decision == "continue"
    assert report.findings == []
