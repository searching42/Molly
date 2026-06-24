from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent._utils import write_json
from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactVerification
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_review_artifacts import (
    RunPlanReviewArtifactBundle,
    write_run_plan_review_artifacts,
)
from ai4s_agent.storage import ProjectStorage


def test_write_review_artifacts_links_verifier_and_replan_proposal(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-review"
    run_id = "run-review"
    run_dir = storage.run_dir(project_id, run_id)
    metrics_path = write_json(
        run_dir / "metrics" / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.42, "mae": 0.33}}]},
    )
    storage.register_artifact_path(project_id, run_id, "model_metrics", metrics_path.relative_to(run_dir).as_posix())

    bundle = write_run_plan_review_artifacts(
        workspace_dir=tmp_path,
        project_id=project_id,
        run_id=run_id,
    )

    assert isinstance(bundle, RunPlanReviewArtifactBundle)
    assert bundle.executable is False
    assert bundle.artifact_ids == [
        "observer_verification",
        "replan_proposal",
        "replan_review_markdown",
    ]
    verification_path = run_dir / "review" / "observer_verification.json"
    proposal_path = run_dir / "review" / "replan_proposal.json"
    markdown_path = run_dir / "review" / "replan_review.md"
    assert verification_path.exists()
    assert proposal_path.exists()
    assert markdown_path.exists()

    verification = RunPlanArtifactVerification.model_validate(json.loads(verification_path.read_text(encoding="utf-8")))
    proposal = RunPlanReplanProposal.model_validate(json.loads(proposal_path.read_text(encoding="utf-8")))
    assert verification.decision == "rerun_recommended"
    assert proposal.verifier_decision == "rerun_recommended"
    assert proposal.proposed_action == "rerun_task"
    assert proposal.executable is False
    assert proposal.proposed_run_plan_patch["applied"] is False

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Run Plan Review" in markdown
    assert "Observer-Verifier Decision: `rerun_recommended`" in markdown
    assert "Proposed Action: `rerun_task`" in markdown
    assert "Executable: `false`" in markdown
    assert "No adapter execution, RunPlan mutation, enqueue, patch apply, or auto-rerun was performed." in markdown

    registry = storage.read_artifact_registry(project_id, run_id)
    assert registry["observer_verification"] == "review/observer_verification.json"
    assert registry["replan_proposal"] == "review/replan_proposal.json"
    assert registry["replan_review_markdown"] == "review/replan_review.md"


def test_write_review_artifacts_accepts_precomputed_verification_without_rechecking_registry(tmp_path: Path) -> None:
    project_id = "proj-precomputed"
    run_id = "run-precomputed"
    verification = RunPlanArtifactVerification(
        project_id=project_id,
        run_id=run_id,
        generated_at="2026-06-24T00:00:00Z",
        decision="needs_review",
        summary="Queued execution is waiting for review.",
        findings=[],
        observed={"queue": {"summary": {"waiting_user": True}}},
    )

    bundle = write_run_plan_review_artifacts(
        workspace_dir=tmp_path,
        project_id=project_id,
        run_id=run_id,
        verification=verification,
    )

    run_dir = ProjectStorage(tmp_path).run_dir(project_id, run_id)
    saved_verification = RunPlanArtifactVerification.model_validate(
        json.loads((run_dir / "review" / "observer_verification.json").read_text(encoding="utf-8"))
    )
    proposal = RunPlanReplanProposal.model_validate(
        json.loads((run_dir / "review" / "replan_proposal.json").read_text(encoding="utf-8"))
    )
    assert saved_verification == verification
    assert proposal.verifier_decision == "needs_review"
    assert bundle.verification == verification
    assert bundle.proposal == proposal


def test_write_review_artifacts_rejects_mismatched_precomputed_verification(tmp_path: Path) -> None:
    verification = RunPlanArtifactVerification(
        project_id="other-project",
        run_id="run-a",
        generated_at="2026-06-24T00:00:00Z",
        decision="continue",
        summary="ok",
        findings=[],
        observed={},
    )

    try:
        write_run_plan_review_artifacts(
            workspace_dir=tmp_path,
            project_id="proj-a",
            run_id="run-a",
            verification=verification,
        )
    except ValueError as exc:
        assert "verification project_id/run_id mismatch" in str(exc)
    else:
        raise AssertionError("mismatched verification identity should fail")
