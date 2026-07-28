from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai4s_agent.agents.oled_discovery import OLEDDiscoveryLoopAgent
from ai4s_agent.schemas import (
    OLEDDiscoveryNextAction,
    OLEDDiscoveryRunCard,
    OLEDDiscoveryStage,
    OLEDDiscoveryStageStatus,
)
from ai4s_agent.storage import ProjectStorage


def _agent() -> OLEDDiscoveryLoopAgent:
    return OLEDDiscoveryLoopAgent()


def _stage_statuses(card: OLEDDiscoveryRunCard) -> dict[str, OLEDDiscoveryStageStatus]:
    return {status.stage: status for status in card.stage_statuses}


def test_empty_goal_blocks_and_asks_for_objective() -> None:
    card = _agent().build_run_card(run_id="run-empty")

    assert card.current_stage == OLEDDiscoveryStage.BLOCKED.value
    assert card.executable is False
    assert "missing_discovery_objective" in card.blocked_reasons
    assert card.recommended_next_actions[0].action_id == "provide_oled_discovery_objective"
    assert card.recommended_next_actions[0].target_stage == OLEDDiscoveryStage.INTENT_CAPTURED.value


def test_goal_only_captures_intent_and_recommends_research_source_proposal() -> None:
    card = _agent().build_run_card(
        run_id="run-goal",
        goal="Find OLED emitters with high PLQY and red-shifted emission",
    )

    assert card.current_stage == OLEDDiscoveryStage.INTENT_CAPTURED.value
    assert card.executable is False
    assert card.recommended_next_actions[0].action_id == "prepare_research_source_proposal"
    assert _stage_statuses(card)[OLEDDiscoveryStage.INTENT_CAPTURED.value].status == "complete"
    assert _stage_statuses(card)[OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value].status == "missing"


def test_research_proposal_present_produces_research_plan_proposed() -> None:
    card = _agent().build_run_card(
        run_id="run-research",
        goal="Find OLED PLQY literature",
        research_source_proposal={"proposal_id": "proposal-1", "sources": ["doi:10.1/test"]},
    )

    assert card.current_stage == OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value
    assert card.recommended_next_actions[0].action_id == "prepare_acquisition_scope"
    assert "research_source_proposal" in card.available_artifacts
    assert "research_acquisition_preparation" in card.missing_artifacts


def test_acquisition_prepared_without_dataset_recommends_data_workflow_with_gate() -> None:
    card = _agent().build_run_card(
        run_id="run-acq",
        goal="Find OLED PLQY literature",
        research_source_proposal={"proposal_id": "proposal-1"},
        research_acquisition_preparation={"status": "needs_confirmation", "requires_external_acquisition": True},
    )

    assert card.current_stage == OLEDDiscoveryStage.ACQUISITION_PREPARED.value
    action = card.recommended_next_actions[0]
    assert action.action_id == "run_or_provide_dataset_artifacts"
    assert action.requires_gate is True
    assert action.suggested_task == "data_mining"
    assert "data_mining_gate_required" in card.risk_flags


def test_dataset_training_baseline_diagnostics_transitions_to_diagnostics_ready() -> None:
    card = _agent().build_run_card(
        run_id="run-diag",
        goal="Screen OLED emitters",
        research_source_proposal={"proposal_id": "proposal-1"},
        research_acquisition_preparation={"status": "prepared"},
        dataset_artifacts={"curated_gold_records": "gold.jsonl", "split_rows": "splits.jsonl"},
        training_package_artifacts={"training_rows": "rows.jsonl", "schema": "schema.json"},
        baseline_artifacts={"predictions": "pred.jsonl", "metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable", "summary": "Baseline is stable enough for candidate screening."},
    )

    assert card.current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert card.recommended_next_actions[0].action_id == "prepare_candidate_generation_or_prediction"
    assert "candidate_artifacts" in card.missing_artifacts
    assert card.executable is False


@pytest.mark.parametrize(
    "diagnostics_report",
    [
        {"status": "weak", "summary": "High error"},
        {"status": "blocked", "summary": "Missing eval split"},
        {"recommendation": "rerun baseline with revised features"},
    ],
)
def test_weak_diagnostics_recommend_revision_not_candidate_generation(diagnostics_report: dict) -> None:
    card = _agent().build_run_card(
        run_id="run-weak",
        goal="Screen OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "rows.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report=diagnostics_report,
    )

    action = card.recommended_next_actions[0]
    assert card.current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert action.action_id == "revise_data_or_model_before_candidates"
    assert action.target_stage == OLEDDiscoveryStage.BASELINE_READY.value
    assert "diagnostics_not_ready_for_candidate_screening" in card.risk_flags


def test_candidate_artifacts_without_critic_review_recommend_critic_review() -> None:
    card = _agent().build_run_card(
        run_id="run-candidates",
        goal="Screen OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "rows.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        candidate_artifacts={"candidate_table": "candidates.jsonl"},
    )

    assert card.current_stage == OLEDDiscoveryStage.CANDIDATES_READY.value
    assert card.recommended_next_actions[0].action_id == "run_critic_review"
    assert card.recommended_next_actions[0].target_stage == OLEDDiscoveryStage.CRITIC_REVIEWED.value


def test_critic_review_produces_reviewed_stage_and_next_action_proposal() -> None:
    card = _agent().build_run_card(
        run_id="run-reviewed",
        goal="Screen OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "rows.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        candidate_artifacts={"candidate_table": "candidates.jsonl"},
        critic_review={"decision": "revise_features", "summary": "Candidates need feature audit."},
    )

    assert card.current_stage == OLEDDiscoveryStage.CRITIC_REVIEWED.value
    action_ids = [action.action_id for action in card.recommended_next_actions]
    assert "propose_next_research_model_or_data_action" in action_ids
    assert _stage_statuses(card)[OLEDDiscoveryStage.NEXT_ACTION_PROPOSED.value].status == "ready"


def test_markdown_run_card_is_deterministic_and_includes_safety_boundary(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    card = _agent().build_run_card(
        run_id="run-write",
        project_id="project-oled",
        goal="Find OLED emitters",
    )

    json_path, md_path = _agent().write_run_card(storage, "project-oled", "run-write", card)
    first_md = md_path.read_text(encoding="utf-8")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    _json_path_2, md_path_2 = _agent().write_run_card(storage, "project-oled", "run-write", card)

    assert first_md == md_path_2.read_text(encoding="utf-8")
    assert "# OLED Discovery Run Card" in first_md
    assert "## Safety Boundary" in first_md
    assert "Executable: false" in first_md
    assert json_payload["executable"] is False


def test_run_card_is_always_non_executable() -> None:
    cards = [
        _agent().build_run_card(run_id="a"),
        _agent().build_run_card(run_id="b", goal="Find OLED emitters"),
        _agent().build_run_card(
            run_id="c",
            goal="Find OLED emitters",
            dataset_artifacts={"rows": "rows.jsonl"},
            training_package_artifacts={"rows": "training.jsonl"},
            baseline_artifacts={"metrics": "metrics.json"},
            diagnostics_report={"status": "acceptable"},
            candidate_artifacts={"candidates": "candidates.jsonl"},
            critic_review={"decision": "continue"},
        ),
    ]

    assert all(card.executable is False for card in cards)


def test_oled_discovery_module_does_not_import_governance_writer_modules() -> None:
    module_names_before = set(sys.modules)
    _agent().build_run_card(run_id="run-import-guard", goal="Find OLED emitters")
    newly_loaded = set(sys.modules) - module_names_before
    forbidden_tokens = (
        "registry_promotion_writer",
        "promoted_registry_publication_writer",
        "final_registry_global_append_writer",
        "global_append_release_writer",
        "external_publication_preflight",
    )

    assert not any(any(token in name for token in forbidden_tokens) for name in newly_loaded)


def test_schema_models_roundtrip() -> None:
    action = OLEDDiscoveryNextAction(
        action_id="prepare_research_source_proposal",
        label="Prepare research source proposal",
        reason="Intent is captured and sources are missing.",
        target_stage=OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value,
        requires_gate=False,
        suggested_task=None,
    )
    card = OLEDDiscoveryRunCard(
        run_id="run-schema",
        project_id="project-oled",
        goal="Find OLED emitters",
        current_stage=OLEDDiscoveryStage.INTENT_CAPTURED.value,
        stage_statuses=[
            OLEDDiscoveryStageStatus(
                stage=OLEDDiscoveryStage.INTENT_CAPTURED.value,
                status="complete",
                evidence=["goal"],
                missing=[],
                summary="Intent captured.",
            )
        ],
        available_artifacts=["goal"],
        missing_artifacts=["research_source_proposal"],
        blocked_reasons=[],
        risk_flags=[],
        recommended_next_actions=[action],
        assumptions=["Review-only run card."],
        executable=False,
    )

    restored = OLEDDiscoveryRunCard.model_validate_json(card.model_dump_json())

    assert restored.model_dump(mode="json") == card.model_dump(mode="json")
