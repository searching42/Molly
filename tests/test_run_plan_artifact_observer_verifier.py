from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.run_plan_artifact_verifier import (
    RunPlanArtifactVerification,
    verify_run_plan_artifacts,
)
from ai4s_agent.storage import ProjectStorage
from tests.test_phase1_queued_workflow_demo import (
    _audit_records as phase1_audit_records,
    _execute_demo as execute_phase1_demo,
)
from tests.test_oled_multiobjective_screening_demo import (
    PROJECT_ID as OLED_PROJECT_ID,
    RUN_ID as OLED_RUN_ID,
    _audit_records as oled_audit_records,
    _execute_demo as execute_oled_demo,
)
from tests.test_phase2_generation_screening_demo import (
    PROJECT_ID as PHASE2_PROJECT_ID,
    RUN_ID as PHASE2_RUN_ID,
    _audit_records as phase2_audit_records,
    _execute_demo as execute_phase2_demo,
)
from tests.test_phase3_literature_dataset_demo import _run_phase3_fixture


def test_artifact_verifier_continues_for_phase1_workflow_fixture(tmp_path: Path) -> None:
    project_id = "phase1-demo-project"
    run_id = "phase1-demo-run"
    _response, summary, _calls, app = execute_phase1_demo(tmp_path)
    client = app.test_client()
    status_response = client.get(
        f"/api/internal/run-plan/queue/status?project_id={project_id}&run_id={run_id}",
        headers={"X-Actor": "phase1-demo-user"},
    )

    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id=project_id,
        run_id=run_id,
        queue_summary=summary.model_dump(mode="json"),
        queue_status=status_response.get_json()["status"],
        audit_records=phase1_audit_records(tmp_path),
    )

    assert report.decision == "continue"
    assert report.observed["audit"]["terminal_outcome"] == "succeeded"
    assert "trainability_report" in report.observed["reports"]
    assert "model_metrics" in report.observed["reports"]


def test_artifact_verifier_recommends_rerun_for_oled_multiobjective_fixture_metrics(tmp_path: Path) -> None:
    _response, summary, _calls, app = execute_oled_demo(tmp_path)
    client = app.test_client()
    status_response = client.get(
        f"/api/internal/run-plan/queue/status?project_id={OLED_PROJECT_ID}&run_id={OLED_RUN_ID}",
        headers={"X-Actor": "oled-demo-user"},
    )

    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id=OLED_PROJECT_ID,
        run_id=OLED_RUN_ID,
        queue_summary=summary.model_dump(mode="json"),
        queue_status=status_response.get_json()["status"],
        audit_records=oled_audit_records(tmp_path),
    )

    assert isinstance(report, RunPlanArtifactVerification)
    assert report.decision == "rerun_recommended"
    assert report.observed["queue"]["summary"]["ok"] is True
    assert report.observed["audit"]["terminal_outcome"] == "succeeded"
    assert "multiobjective_ranking" in report.observed["reports"]
    assert not any(finding.decision == "blocked" for finding in report.findings)
    assert "poor_model_metrics" in {finding.category for finding in report.findings}


def test_artifact_verifier_continues_for_phase2_generation_fixture(tmp_path: Path) -> None:
    _response, summary, _calls, app = execute_phase2_demo(tmp_path)
    client = app.test_client()
    status_response = client.get(
        f"/api/internal/run-plan/queue/status?project_id={PHASE2_PROJECT_ID}&run_id={PHASE2_RUN_ID}",
        headers={"X-Actor": "phase2-demo-user"},
    )

    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id=PHASE2_PROJECT_ID,
        run_id=PHASE2_RUN_ID,
        queue_summary=summary.model_dump(mode="json"),
        queue_status=status_response.get_json()["status"],
        audit_records=phase2_audit_records(tmp_path),
    )

    assert report.decision == "continue"
    generation = report.observed["reports"]["generation_report"]
    assert generation["backend"] == "deterministic_stub"
    assert generation["generated_count"] > 0


def test_artifact_verifier_reads_phase3_extraction_benchmark_fixture(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)
    registry = {
        "extraction_benchmark_report": result["extraction_benchmark_report_json"],
        "confirmed_dataset": result["confirmed_dataset_csv"],
        "report_json": result["report_json"],
    }

    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id="phase3-fixture-project",
        run_id="phase3-literature-dataset-demo",
        artifact_registry=registry,
    )

    assert report.decision == "continue"
    benchmark = report.observed["reports"]["extraction_benchmark"]
    assert benchmark["trainable_labels_gained"] == 6
    assert benchmark["counts"]["confirmed_records"] == 6


def test_artifact_verifier_blocks_missing_registered_artifact(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-missing-artifact"
    run_id = "run-missing-artifact"
    storage.run_dir(project_id, run_id)
    storage.register_artifact_path(project_id, run_id, "model_metrics", "missing/model_metrics.json")

    report = verify_run_plan_artifacts(workspace_dir=tmp_path, project_id=project_id, run_id=run_id)

    assert report.decision == "blocked"
    assert "missing_artifact" in {finding.category for finding in report.findings}


def test_artifact_verifier_recommends_rerun_for_bad_model_metrics(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-bad-metrics"
    run_id = "run-bad-metrics"
    run_dir = storage.run_dir(project_id, run_id)
    metrics_path = write_json(
        run_dir / "metrics" / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.31, "mae": 0.42}}]},
    )
    storage.register_artifact_path(project_id, run_id, "model_metrics", str(metrics_path.relative_to(run_dir)))

    report = verify_run_plan_artifacts(workspace_dir=tmp_path, project_id=project_id, run_id=run_id)

    assert report.decision == "rerun_recommended"
    finding = next(item for item in report.findings if item.category == "poor_model_metrics")
    assert finding.evidence["weak_metrics"][0]["property_id"] == "plqy"


def test_artifact_verifier_needs_review_for_waiting_user_queue_summary(tmp_path: Path) -> None:
    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id="proj-waiting",
        run_id="run-waiting",
        queue_summary={
            "ok": True,
            "terminal": True,
            "queued_job_id": "job-waiting",
            "final_job": {"status": "succeeded"},
            "final_lease": {"status": "completed"},
            "loop_results": ["completed", "idle"],
            "waiting_user": True,
            "waiting_task": "approve_training",
            "required_gates": ["gate_3_train_config"],
            "error": None,
        },
    )

    assert report.decision == "needs_review"
    finding = next(item for item in report.findings if item.category == "waiting_user")
    assert finding.evidence["required_gates"] == ["gate_3_train_config"]


def test_artifact_verifier_blocks_failed_queue_summary(tmp_path: Path) -> None:
    report = verify_run_plan_artifacts(
        workspace_dir=tmp_path,
        project_id="proj-failed",
        run_id="run-failed",
        queue_summary={
            "ok": False,
            "terminal": True,
            "queued_job_id": "job-failed",
            "final_job": {"status": "failed", "error": {"reason": "adapter failed"}},
            "final_lease": {"status": "failed"},
            "loop_results": ["failed"],
            "error": {"type": "execution_failed", "message": "adapter failed"},
        },
    )

    assert report.decision == "blocked"
    assert "queue_failed" in {finding.category for finding in report.findings}


def test_artifact_verifier_needs_review_for_empty_generation_report(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-empty-generation"
    run_id = "run-empty-generation"
    run_dir = storage.run_dir(project_id, run_id)
    generation_path = write_json(
        run_dir / "generation" / "generation_report.json",
        {"backend": "deterministic_stub", "generated_count": 0},
    )
    storage.register_artifact_path(project_id, run_id, "generation_report", str(generation_path.relative_to(run_dir)))

    report = verify_run_plan_artifacts(workspace_dir=tmp_path, project_id=project_id, run_id=run_id)

    assert report.decision == "needs_review"
    assert "empty_generation" in {finding.category for finding in report.findings}


def test_artifact_verifier_needs_review_for_empty_multiobjective_ranking(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-empty-ranking"
    run_id = "run-empty-ranking"
    run_dir = storage.run_dir(project_id, run_id)
    ranking_path = run_dir / "rank" / "multiobjective_ranked_candidates.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "weighted_score"])
        writer.writeheader()
    storage.register_artifact_path(project_id, run_id, "multiobjective_ranked_candidates", str(ranking_path.relative_to(run_dir)))

    report = verify_run_plan_artifacts(workspace_dir=tmp_path, project_id=project_id, run_id=run_id)

    assert report.decision == "needs_review"
    assert "empty_ranking" in {finding.category for finding in report.findings}


def test_artifact_verifier_schema_rejects_unknown_decision() -> None:
    payload: dict[str, Any] = {
        "project_id": "proj",
        "run_id": "run",
        "generated_at": "2026-06-24T00:00:00Z",
        "decision": "ask_user",
        "summary": "bad decision",
        "findings": [],
        "observed": {},
    }

    try:
        RunPlanArtifactVerification.model_validate(payload)
    except ValueError as exc:
        assert "decision" in str(exc)
    else:
        raise AssertionError("unknown decision should be rejected")
