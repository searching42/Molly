from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.oled_discovery import OLEDDiscoveryLoopAgent
from ai4s_agent.agents.tool_registry import AgentToolRegistry
from ai4s_agent.schemas import (
    AgentToolRecommendation,
    AgentToolRegistrySnapshot,
    AgentToolSpec,
    OLEDDiscoveryStage,
)


def _registry() -> AgentToolRegistry:
    return AgentToolRegistry()


def test_registry_snapshot_is_non_executable_and_contains_expected_tools() -> None:
    snapshot = _registry().snapshot()
    tool_ids = {tool.tool_id for tool in snapshot.tools}

    assert snapshot.registry_id == "agent-tool-registry:oLED-discovery:v1".lower()
    assert snapshot.executable is False
    assert snapshot.tool_count == len(snapshot.tools)
    assert "research_source_proposal" in tool_ids
    assert "acquire_literature_sources" in tool_ids
    assert "training_package" in tool_ids
    assert "baseline_runner" in tool_ids
    assert "critic_review" in tool_ids
    assert all(tool.executable is False for tool in snapshot.tools)


def test_tools_for_diagnostics_ready_returns_candidate_related_tools() -> None:
    tools = _registry().tools_for_stage(OLEDDiscoveryStage.DIAGNOSTICS_READY.value)
    tool_ids = {tool.tool_id for tool in tools}

    assert "candidate_generation_or_prediction" in tool_ids
    assert "candidate_ranking" in tool_ids


def test_recommend_tools_marks_ready_when_inputs_available() -> None:
    recommendations = _registry().recommend_tools(
        current_stage=OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
        available_artifacts=["diagnostics_report", "training_package_artifacts"],
        risk_budget="medium",
    )
    by_id = {recommendation.tool_id: recommendation for recommendation in recommendations}

    assert by_id["candidate_generation_or_prediction"].ready is True
    assert by_id["candidate_generation_or_prediction"].missing_inputs == []
    assert by_id["candidate_generation_or_prediction"].executable is False


def test_missing_input_artifacts_block_readiness() -> None:
    recommendations = _registry().recommend_tools(
        current_stage=OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
        available_artifacts=[],
    )
    recommendation = {item.tool_id: item for item in recommendations}["candidate_generation_or_prediction"]

    assert recommendation.ready is False
    assert "diagnostics_report" in recommendation.missing_inputs
    assert "missing_required_inputs" in recommendation.blocked_reasons


def test_low_risk_budget_blocks_medium_and_high_tools() -> None:
    recommendations = _registry().recommend_tools(
        current_stage=OLEDDiscoveryStage.ACQUISITION_PREPARED.value,
        available_artifacts=["research_acquisition_preparation"],
        risk_budget="low",
    )
    acquisition = {item.tool_id: item for item in recommendations}["acquire_literature_sources"]

    assert acquisition.ready is False
    assert "risk_level_exceeds_budget" in acquisition.blocked_reasons


def test_allow_gated_false_blocks_tools_requiring_gates() -> None:
    recommendations = _registry().recommend_tools(
        current_stage=OLEDDiscoveryStage.ACQUISITION_PREPARED.value,
        available_artifacts=["research_acquisition_preparation"],
        allow_gated=False,
    )
    acquisition = {item.tool_id: item for item in recommendations}["acquire_literature_sources"]

    assert acquisition.ready is False
    assert acquisition.required_gates == ["gate_2_data_mining"]
    assert "gated_tool_not_allowed" in acquisition.blocked_reasons


def test_data_acquisition_tool_requires_data_mining_gate() -> None:
    tool = {tool.tool_id: tool for tool in _registry().list_tools()}["acquire_literature_sources"]

    assert tool.required_gates == ["gate_2_data_mining"]
    assert tool.risk_level == "high"
    assert "external_acquisition_scope" in tool.required_permissions


def test_recommended_tools_for_run_card_uses_current_stage_and_available_artifacts() -> None:
    run_card = OLEDDiscoveryLoopAgent().build_run_card(
        run_id="run-card",
        goal="Screen OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
    )

    recommendations = _registry().recommended_tools_for_run_card(run_card)
    by_id = {recommendation.tool_id: recommendation for recommendation in recommendations}

    assert run_card.current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert by_id["candidate_generation_or_prediction"].target_stage == OLEDDiscoveryStage.CANDIDATES_READY.value


def test_registry_does_not_import_governance_writer_modules() -> None:
    before = set(sys.modules)
    _registry().recommend_tools(current_stage=OLEDDiscoveryStage.DIAGNOSTICS_READY.value)
    newly_loaded = set(sys.modules) - before
    forbidden_tokens = (
        "registry_promotion_writer",
        "promoted_registry_publication_writer",
        "final_registry_global_append_writer",
        "global_append_release_writer",
        "external_publication_preflight",
    )

    assert not any(any(token in module_name for token in forbidden_tokens) for module_name in newly_loaded)


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary(tmp_path: Path) -> None:
    snapshot = _registry().snapshot()
    first = _registry().render_markdown(snapshot)
    second = _registry().render_markdown(snapshot)
    output_path = tmp_path / "registry.md"
    output_path.write_text(first, encoding="utf-8")

    assert first == second
    assert "# Agent Tool Registry" in first
    assert "## Safety Boundary" in first
    assert "candidate_generation_or_prediction" in first
    assert "Executable: false" in first


def test_schema_roundtrip_and_review_only_validation() -> None:
    spec = AgentToolSpec(
        tool_id="demo_tool",
        label="Demo Tool",
        discovery_stages=[OLEDDiscoveryStage.INTENT_CAPTURED.value],
        risk_level="low",
    )
    recommendation = AgentToolRecommendation(
        tool_id="demo_tool",
        reason="Demo",
        target_stage=OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value,
        ready=True,
    )
    snapshot = AgentToolRegistrySnapshot(registry_id="demo", tool_count=1, tools=[spec])

    assert AgentToolSpec.model_validate_json(spec.model_dump_json()).model_dump(mode="json") == spec.model_dump(mode="json")
    assert AgentToolRecommendation.model_validate_json(recommendation.model_dump_json()).model_dump(mode="json") == recommendation.model_dump(mode="json")
    assert AgentToolRegistrySnapshot.model_validate_json(snapshot.model_dump_json()).model_dump(mode="json") == snapshot.model_dump(mode="json")
    with pytest.raises(ValidationError):
        AgentToolSpec(tool_id="bad", label="Bad", risk_level="extreme")
    with pytest.raises(ValidationError):
        AgentToolRecommendation(
            tool_id="bad",
            reason="Bad",
            target_stage=OLEDDiscoveryStage.INTENT_CAPTURED.value,
            ready=True,
            executable=True,
        )


def test_cli_outputs_compact_json_summary(capsys) -> None:
    from ai4s_agent.agents.tool_registry import main  # noqa: PLC0415

    exit_code = main(
        [
            "--stage",
            OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
            "--available-artifact",
            "diagnostics_report",
            "--available-artifact",
            "training_package_artifacts",
            "--risk-budget",
            "medium",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["stage"] == OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    assert "candidate_generation_or_prediction" in summary["recommended_tool_ids"]
    assert "tools" not in summary
