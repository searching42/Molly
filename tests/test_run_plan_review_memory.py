from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_review_artifacts import write_run_plan_review_artifacts
from ai4s_agent.run_plan_review_card import read_run_plan_review_card
from ai4s_agent.run_plan_review_memory import (
    RunPlanReviewMemorySave,
    build_run_plan_review_memory_record,
    save_run_plan_review_card_summary_to_memory,
)
from ai4s_agent.storage import ProjectStorage


def _write_review_card_inputs(workspace: Path, *, project_id: str = "proj-memory", run_id: str = "run-review") -> None:
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    metrics_path = write_json(
        run_dir / "metrics" / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.18, "mae": 0.31}}]},
    )
    storage.register_artifact_path(project_id, run_id, "model_metrics", metrics_path.relative_to(run_dir).as_posix())
    write_run_plan_review_artifacts(workspace_dir=workspace, project_id=project_id, run_id=run_id)


def test_build_run_plan_review_memory_record_uses_summary_only(tmp_path: Path) -> None:
    _write_review_card_inputs(tmp_path)
    card = read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-memory", run_id="run-review")

    record = build_run_plan_review_memory_record(card, confirmed_by="reviewer-a")

    assert record.record_id == "run-plan-review-run-review"
    assert record.category == "run_plan_review"
    assert record.decision == "run_plan_review_recorded"
    assert record.confirmed_by == "reviewer-a"
    assert "rerun_recommended" in record.summary
    assert record.value == {
        "kind": "run_plan_review_summary",
        "project_id": "proj-memory",
        "run_id": "run-review",
        "verifier_decision": "rerun_recommended",
        "proposed_action": "rerun_task",
        "affected_tasks": ["train_model"],
        "required_user_decisions": ["Approve rerun of train_model before any queued execution."],
        "artifact_refs": {
            "observer_verification": "review/observer_verification.json",
            "replan_proposal": "review/replan_proposal.json",
            "replan_review_markdown": "review/replan_review.md",
        },
        "executable": False,
    }
    assert record.source_refs == [
        "run:run-review:artifact:observer_verification",
        "run:run-review:artifact:replan_proposal",
        "run:run-review:artifact:replan_review_markdown",
    ]
    serialized = json.dumps(record.model_dump(mode="json"))
    assert "Observer-Verifier Decision" not in serialized
    assert "# Run Plan Review" not in serialized
    assert "proposed_run_plan_patch" not in serialized
    assert "properties" not in serialized
    assert "raw_data" not in serialized


def test_save_run_plan_review_card_summary_to_project_memory(tmp_path: Path) -> None:
    _write_review_card_inputs(tmp_path)

    result = save_run_plan_review_card_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-memory",
        run_id="run-review",
        confirmed_by="reviewer-a",
    )

    assert isinstance(result, RunPlanReviewMemorySave)
    assert result.saved is True
    assert result.executable is False
    assert result.record.record_id == "run-plan-review-run-review"
    records = ProjectMemory(tmp_path).list_project_records("proj-memory")
    assert [record.record_id for record in records] == ["run-plan-review-run-review"]
    assert records[0].value["artifact_refs"]["replan_proposal"] == "review/replan_proposal.json"


def test_save_run_plan_review_card_summary_is_idempotent_per_run(tmp_path: Path) -> None:
    _write_review_card_inputs(tmp_path)

    first = save_run_plan_review_card_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-memory",
        run_id="run-review",
        confirmed_by="reviewer-a",
    )
    second = save_run_plan_review_card_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-memory",
        run_id="run-review",
        confirmed_by="reviewer-b",
    )

    records = ProjectMemory(tmp_path).list_project_records("proj-memory")
    assert len(records) == 1
    assert first.record.record_id == second.record.record_id
    assert records[0].confirmed_by == "reviewer-b"


def test_save_run_plan_review_card_summary_accepts_preloaded_card_without_reading_artifacts(tmp_path: Path) -> None:
    _write_review_card_inputs(tmp_path)
    card = read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-memory", run_id="run-review")
    run_dir = ProjectStorage(tmp_path).run_dir("proj-memory", "run-review")
    (run_dir / "review" / "replan_review.md").unlink()

    result = save_run_plan_review_card_summary_to_memory(
        workspace_dir=tmp_path,
        project_id="proj-memory",
        run_id="run-review",
        card=card,
        confirmed_by="reviewer-a",
    )

    assert result.saved is True
    assert result.record.value["proposed_action"] == "rerun_task"


def test_save_run_plan_review_card_summary_rejects_mismatched_card(tmp_path: Path) -> None:
    _write_review_card_inputs(tmp_path)
    card = read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-memory", run_id="run-review")
    payload: dict[str, Any] = card.model_dump(mode="json")
    payload["run_id"] = "other-run"

    try:
        save_run_plan_review_card_summary_to_memory(
            workspace_dir=tmp_path,
            project_id="proj-memory",
            run_id="run-review",
            card=payload,
            confirmed_by="reviewer-a",
        )
    except ValueError as exc:
        assert "review card project_id/run_id mismatch" in str(exc)
    else:
        raise AssertionError("mismatched review card should fail")
