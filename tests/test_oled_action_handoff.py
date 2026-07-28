from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.action_handoff import OLEDDiscoveryActionHandoffAgent
from ai4s_agent.agents.oled_review_loop import OLEDDiscoveryReviewLoopAgent
from ai4s_agent.schemas import OLEDDiscoveryActionHandoff, OLEDDiscoveryStage
from ai4s_agent.storage import ProjectStorage


def _diagnostics_review():
    return OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="diagnostics",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )


def _acquisition_review(*, allow_gated: bool = True):
    return OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="acquisition",
        goal="Find OLED papers",
        research_source_proposal={"query": "OLED PLQY"},
        research_acquisition_preparation={"required_gates": ["gate_2_data_mining"]},
        risk_budget="high",
        allow_gated=allow_gated,
    )


def test_candidate_generation_maps_to_selected_tool_and_placeholder_payload() -> None:
    review = _diagnostics_review()
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "candidate_generation_or_prediction"
    assert handoff.selected_tool_id == "candidate_generation_or_prediction"
    assert handoff.selected_task_id == "candidate_generation_or_prediction"
    assert handoff.ready is True
    assert handoff.executable is False
    assert handoff.payload_template == {
        "run_id": "diagnostics",
        "diagnostics_report": "<diagnostics_report>",
        "training_package_artifacts": "<training_package_artifacts>",
        "review_only": True,
    }


def test_rerun_baseline_maps_to_baseline_runner() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="rerun",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "weak"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "rerun_baseline"
    assert handoff.selected_tool_id == "baseline_runner"
    assert handoff.payload_template["training_package_artifacts"] == "<training_package_artifacts>"


def test_request_more_evidence_maps_to_evidence_or_research_tool() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="evidence",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 0, "evidence_count": 0},
    )
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "request_more_evidence"
    assert handoff.selected_tool_id in {"retrieve_evidence", "research_source_proposal"}
    assert handoff.executable is False


def test_revise_data_maps_to_leakage_split_or_training_package() -> None:
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id="revise-data",
        goal="Find OLED emitters",
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        dataset_summary={"split_leakage": True},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "revise_data"
    assert handoff.selected_tool_id in {"leakage_split", "training_package"}
    assert "split_leakage_risk" in handoff.risk_flags


def test_run_candidate_review_maps_to_critic_review() -> None:
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
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "run_candidate_review"
    assert handoff.selected_tool_id == "critic_review"
    assert handoff.payload_template["candidate_summary"] == "<candidate_summary>"


def test_block_promotion_maps_to_review_only_critic_handling() -> None:
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
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.recommended_next_action == "block_promotion"
    assert handoff.selected_tool_id == "critic_review"
    assert handoff.executable is False
    assert not any("promotion" in str(value).lower() and "write" in str(value).lower() for value in handoff.payload_template.values())


def test_resolve_missing_inputs_produces_no_selected_executable_tool() -> None:
    review = _diagnostics_review().model_copy(update={"recommended_next_action": "resolve_missing_inputs:training_package_artifacts"})
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.selected_tool_id == ""
    assert handoff.ready is False
    assert handoff.missing_inputs == ["training_package_artifacts"]
    assert "missing_input:training_package_artifacts" in handoff.blocked_reasons


def test_gated_selected_tool_preserves_gates_and_remains_non_executable() -> None:
    review = _acquisition_review()
    review = review.model_copy(update={"recommended_next_action": "acquire_literature_sources"})
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.selected_tool_id == "acquire_literature_sources"
    assert handoff.ready is True
    assert handoff.required_gates == ["gate_2_data_mining"]
    assert "external_acquisition_scope" in handoff.required_permissions
    assert "ready for gated review, not execution" in handoff.rationale
    assert handoff.executable is False


def test_not_ready_selected_tool_copies_missing_inputs_and_blocked_reasons() -> None:
    review = _diagnostics_review()
    review = review.model_copy(update={"recommended_next_action": "candidate_ranking"})
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.selected_tool_id == "candidate_ranking"
    assert handoff.ready is False
    assert "candidate_artifacts" in handoff.missing_inputs
    assert "missing_required_inputs" in handoff.blocked_reasons


def test_no_selected_tool_produces_no_selected_tool_blocker() -> None:
    review = _diagnostics_review().model_copy(update={"recommended_next_action": "human_review_required"})
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)

    assert handoff.selected_tool_id == ""
    assert handoff.ready is False
    assert handoff.blocked_reasons == ["no_selected_tool"]


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=_diagnostics_review())
    first = OLEDDiscoveryActionHandoffAgent().render_markdown(handoff)
    second = OLEDDiscoveryActionHandoffAgent().render_markdown(handoff)

    assert first == second
    assert "# OLED Discovery Action Handoff" in first
    assert "## Payload Template" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first


def test_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDDiscoveryActionHandoffAgent()
    handoff = agent.build_handoff(loop_review=_diagnostics_review())
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_handoff(storage, "project", "diagnostics", handoff)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_discovery_action_handoff.json"
    assert md_path.name == "oled_discovery_action_handoff.md"
    assert payload["selected_tool_id"] == "candidate_generation_or_prediction"
    assert payload["payload_template"]["review_only"] is True
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_schema_roundtrip_and_validation() -> None:
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=_diagnostics_review())

    restored = OLEDDiscoveryActionHandoff.model_validate_json(handoff.model_dump_json())
    assert restored.model_dump(mode="json") == handoff.model_dump(mode="json")
    with pytest.raises(ValidationError):
        OLEDDiscoveryActionHandoff(
            run_id="bad",
            recommended_next_action="candidate_generation_or_prediction",
            critic_decision="continue",
            ready=True,
            missing_inputs=["diagnostics_report"],
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryActionHandoff(
            run_id="bad",
            recommended_next_action="candidate_generation_or_prediction",
            critic_decision="continue",
            executable=True,
        )


def test_module_does_not_import_execution_or_governance_modules() -> None:
    before = set(sys.modules)
    OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=_diagnostics_review())
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


def test_cli_outputs_compact_json_summary(capsys) -> None:
    from ai4s_agent.agents.action_handoff import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "demo",
            "--goal",
            "Find OLED emitters with high PLQY",
            "--recommended-next-action",
            "candidate_generation_or_prediction",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["run_id"] == "demo"
    assert summary["recommended_next_action"] == "candidate_generation_or_prediction"
    assert summary["selected_tool_id"] == "candidate_generation_or_prediction"
    assert summary["executable"] is False
    assert "payload_template" not in summary
