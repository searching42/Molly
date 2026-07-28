from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.oled_review_loop import OLEDDiscoveryReviewLoopAgent
from ai4s_agent.schemas import OLEDDiscoveryLoopReview, OLEDDiscoveryStage
from ai4s_agent.storage import ProjectStorage


def test_goal_only_builds_intent_stage_and_research_tool_recommendation() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="goal-only",
        goal="Find OLED emitters with high PLQY",
    )

    assert review.run_card.current_stage == OLEDDiscoveryStage.INTENT_CAPTURED.value
    assert "research_source_proposal" in {item.tool_id for item in review.tool_recommendations}
    assert review.critic_review.executable is False
    assert review.executable is False


def test_acceptable_diagnostics_recommends_candidate_generation() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="diagnostics-ok",
        goal="Find OLED emitters with high PLQY",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.run_card.current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert review.critic_review.decision.decision == "continue"
    assert "candidate_generation_or_prediction" in {item.tool_id for item in review.tool_recommendations}
    assert "candidate_generation_or_prediction" in review.ready_tool_ids
    assert review.recommended_next_action == "candidate_generation_or_prediction"


def test_weak_diagnostics_uses_critic_decision_as_next_action() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="weak",
        goal="Find OLED emitters with high PLQY",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "weak", "summary": "rerun recommended"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.critic_review.decision.decision == "rerun_baseline"
    assert review.recommended_next_action == "rerun_baseline"
    assert "weak_diagnostics" in review.risk_flags


def test_leakage_summary_revises_data() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="leakage",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        dataset_summary={"split_leakage": True},
        training_package_summary={"train_test_overlap": 1},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.critic_review.decision.decision == "revise_data"
    assert review.recommended_next_action == "revise_data"
    assert "split_leakage_risk" in review.risk_flags


def test_missing_provenance_after_model_stage_requests_more_evidence() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="provenance-gap",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 0, "evidence_count": 0},
    )

    assert review.critic_review.decision.decision == "request_more_evidence"
    assert review.recommended_next_action == "request_more_evidence"
    assert "insufficient_provenance" in review.risk_flags


def test_candidate_artifacts_recommend_candidate_review() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="candidate-review",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        candidate_artifacts={"candidate_rows": "candidates.jsonl"},
        candidate_summary={"candidate_count": 4},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.run_card.current_stage == OLEDDiscoveryStage.CANDIDATES_READY.value
    assert review.critic_review.decision.decision == "run_candidate_review"
    assert review.recommended_next_action == "run_candidate_review"


def test_overclaim_with_weak_diagnostics_blocks_promotion() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="overclaim",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "weak"},
        candidate_summary={"summary": "publish benchmark_validated result"},
        model_package_review={"recommendation": "promote"},
    )

    assert review.critic_review.decision.decision == "block_promotion"
    assert review.recommended_next_action == "block_promotion"
    assert "overclaim_risk" in review.risk_flags


def test_low_risk_budget_blocks_medium_high_tools() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="low-risk",
        goal="Find OLED sources",
        research_source_proposal={"query": "OLED PLQY"},
        research_acquisition_preparation={"required_gates": ["gate_2_data_mining"]},
        risk_budget="low",
    )

    acquisition = {item.tool_id: item for item in review.tool_recommendations}["acquire_literature_sources"]
    assert acquisition.ready is False
    assert "risk_level_exceeds_budget" in acquisition.blocked_reasons
    assert "acquire_literature_sources" in review.blocked_tool_ids


def test_disallow_gated_blocks_gated_tool_recommendations() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="no-gated",
        goal="Find OLED sources",
        research_source_proposal={"query": "OLED PLQY"},
        research_acquisition_preparation={"required_gates": ["gate_2_data_mining"]},
        allow_gated=False,
    )

    acquisition = {item.tool_id: item for item in review.tool_recommendations}["acquire_literature_sources"]
    assert acquisition.ready is False
    assert "gated_tool_not_allowed" in acquisition.blocked_reasons


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDDiscoveryReviewLoopAgent()
    review = agent.build_review(
        run_id="markdown",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )
    first = agent.render_markdown(review)
    second = agent.render_markdown(review)

    assert first == second
    assert "# OLED Discovery Loop Review" in first
    assert "## Tool Recommendations" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first


def test_json_writing_is_deterministic_and_nested(tmp_path: Path) -> None:
    agent = OLEDDiscoveryReviewLoopAgent()
    review = agent.build_review(
        run_id="write",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_review(storage, "project", "write", review)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_discovery_loop_review.json"
    assert md_path.name == "oled_discovery_loop_review.md"
    assert payload["run_card"]["current_stage"] == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert payload["tool_recommendations"]
    assert payload["critic_review"]["decision"]["decision"] == "continue"
    assert payload["executable"] is False
    assert "candidate_generation_or_prediction" in json_path.read_text(encoding="utf-8")


def test_schema_roundtrip_and_review_only_validation() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="roundtrip",
        goal="Find OLED emitters",
    )

    restored = OLEDDiscoveryLoopReview.model_validate_json(review.model_dump_json())
    assert restored.model_dump(mode="json") == review.model_dump(mode="json")
    with pytest.raises(ValidationError):
        OLEDDiscoveryLoopReview(
            run_id=review.run_id,
            run_card=review.run_card,
            critic_review=review.critic_review,
            executable=True,
        )


def test_integrated_loop_does_not_import_execution_or_governance_modules() -> None:
    before = set(sys.modules)
    OLEDDiscoveryReviewLoopAgent().build_review(run_id="imports", goal="Find OLED emitters")
    newly_loaded = set(sys.modules) - before
    forbidden_tokens = (
        "run_plan_executor",
        "registry_promotion_writer",
        "promoted_registry_publication_writer",
        "final_registry_global_append_writer",
        "global_append_release_writer",
        "external_publication",
        "openai",
        "mineru",
        "pdfplumber",
        "requests",
        "urllib3",
    )

    assert not any(any(token in module_name.lower() for token in forbidden_tokens) for module_name in newly_loaded)


def test_cli_outputs_compact_review_summary(capsys) -> None:
    from ai4s_agent.agents.oled_review_loop import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "demo",
            "--goal",
            "Find OLED emitters with high PLQY",
            "--diagnostics-status",
            "acceptable",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["run_id"] == "demo"
    assert summary["current_stage"] == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert summary["critic_decision"] == "continue"
    assert summary["recommended_next_action"] == "candidate_generation_or_prediction"
    assert summary["executable"] is False
    assert "tool_recommendations" not in summary
