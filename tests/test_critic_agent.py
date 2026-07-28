from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.critic import CriticAgent
from ai4s_agent.agents.oled_discovery import OLEDDiscoveryLoopAgent
from ai4s_agent.agents.tool_registry import AgentToolRegistry
from ai4s_agent.schemas import (
    AgentToolRecommendation,
    CriticDecision,
    CriticFinding,
    CriticReview,
    OLEDDiscoveryStage,
)
from ai4s_agent.storage import ProjectStorage


def _run_card(stage: OLEDDiscoveryStage, *, goal: str = "Screen OLED emitters"):
    agent = OLEDDiscoveryLoopAgent()
    kwargs = {
        "run_id": "run-critic",
        "goal": goal,
        "project_id": "project-critic",
    }
    if stage in {
        OLEDDiscoveryStage.DATASET_READY,
        OLEDDiscoveryStage.TRAINING_PACKAGE_READY,
        OLEDDiscoveryStage.BASELINE_READY,
        OLEDDiscoveryStage.DIAGNOSTICS_READY,
        OLEDDiscoveryStage.CANDIDATES_READY,
        OLEDDiscoveryStage.CRITIC_REVIEWED,
    }:
        kwargs["dataset_artifacts"] = {"dataset_view_rows": "rows.jsonl"}
    if stage in {
        OLEDDiscoveryStage.TRAINING_PACKAGE_READY,
        OLEDDiscoveryStage.BASELINE_READY,
        OLEDDiscoveryStage.DIAGNOSTICS_READY,
        OLEDDiscoveryStage.CANDIDATES_READY,
        OLEDDiscoveryStage.CRITIC_REVIEWED,
    }:
        kwargs["training_package_artifacts"] = {"training_rows": "training.jsonl"}
    if stage in {
        OLEDDiscoveryStage.BASELINE_READY,
        OLEDDiscoveryStage.DIAGNOSTICS_READY,
        OLEDDiscoveryStage.CANDIDATES_READY,
        OLEDDiscoveryStage.CRITIC_REVIEWED,
    }:
        kwargs["baseline_artifacts"] = {"metrics": "metrics.json"}
    if stage in {
        OLEDDiscoveryStage.DIAGNOSTICS_READY,
        OLEDDiscoveryStage.CANDIDATES_READY,
        OLEDDiscoveryStage.CRITIC_REVIEWED,
    }:
        kwargs["diagnostics_report"] = {"status": "acceptable"}
    if stage in {OLEDDiscoveryStage.CANDIDATES_READY, OLEDDiscoveryStage.CRITIC_REVIEWED}:
        kwargs["candidate_artifacts"] = {"candidate_rows": "candidates.jsonl"}
    if stage == OLEDDiscoveryStage.CRITIC_REVIEWED:
        kwargs["critic_review"] = {"decision": "continue"}
    return agent.build_run_card(**kwargs)


def test_missing_objective_or_blocked_run_card_requests_more_evidence() -> None:
    run_card = OLEDDiscoveryLoopAgent().build_run_card(run_id="blocked")

    review = CriticAgent().review(run_id="blocked", run_card=run_card)

    assert review.decision.decision == "request_more_evidence"
    assert review.decision.target_stage == OLEDDiscoveryStage.INTENT_CAPTURED.value
    assert "missing_discovery_objective" in review.blocked_reasons
    assert any(finding.severity == "critical" and finding.category == "objective" for finding in review.findings)
    assert review.executable is False


def test_missing_provenance_after_dataset_stage_requests_more_evidence() -> None:
    review = CriticAgent().review(
        run_id="provenance-gap",
        run_card=_run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY),
        diagnostics_report={"status": "acceptable"},
        dataset_summary={"row_count": 12},
        provenance_summary={"source_count": 0, "evidence_count": 0},
    )

    assert review.decision.decision == "request_more_evidence"
    assert "insufficient_provenance" in review.risk_flags
    assert any(finding.category == "provenance" for finding in review.findings)


def test_leakage_summary_revises_data_with_critical_finding() -> None:
    review = CriticAgent().review(
        run_id="leakage",
        run_card=_run_card(OLEDDiscoveryStage.TRAINING_PACKAGE_READY),
        dataset_summary={"row_count": 20, "split_leakage": True},
        training_package_summary={"train_test_overlap": 2},
        provenance_summary={"source_count": 3, "evidence_count": 10},
    )

    assert review.decision.decision == "revise_data"
    assert "split_leakage_risk" in review.risk_flags
    assert any(finding.severity == "critical" and finding.category == "data_leakage" for finding in review.findings)


def test_weak_diagnostics_recommends_baseline_rerun_or_model_revision() -> None:
    review = CriticAgent().review(
        run_id="weak-diagnostics",
        run_card=_run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY),
        diagnostics_report={"status": "weak", "summary": "rerun recommended"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.decision.decision == "rerun_baseline"
    assert "weak_diagnostics" in review.risk_flags
    assert any(finding.category == "diagnostics" for finding in review.findings)


def test_diagnostics_specific_risks_are_preserved() -> None:
    review = CriticAgent().review(
        run_id="diagnostic-risks",
        run_card=_run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY),
        diagnostics_report={
            "status": "weak",
            "risk_flags": ["high_value_underprediction", "prediction_range_compression"],
        },
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert "high_value_underprediction" in review.risk_flags
    assert "prediction_range_compression" in review.risk_flags


def test_candidate_domain_risk_recommends_candidate_review() -> None:
    review = CriticAgent().review(
        run_id="candidate-risk",
        run_card=_run_card(OLEDDiscoveryStage.CANDIDATES_READY),
        diagnostics_report={"status": "acceptable"},
        candidate_summary={"out_of_domain_count": 2, "invalid_smiles_count": 1, "missing_prediction_count": 0},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.decision.decision == "run_candidate_review"
    assert "candidate_domain_risk" in review.risk_flags
    assert any(finding.category == "candidate_domain" for finding in review.findings)


def test_overclaim_terms_with_weak_diagnostics_or_missing_provenance_block_promotion() -> None:
    review = CriticAgent().review(
        run_id="overclaim",
        run_card=_run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY),
        diagnostics_report={"status": "weak"},
        candidate_summary={"summary": "publish benchmark_validated candidates"},
        model_package_review={"recommendation": "promote"},
    )

    assert review.decision.decision == "block_promotion"
    assert "overclaim_risk" in review.risk_flags
    assert any(finding.category == "overclaim" and finding.severity == "critical" for finding in review.findings)


def test_acceptable_diagnostics_without_candidates_continues_to_candidate_generation() -> None:
    run_card = _run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY)
    recommendations = AgentToolRegistry().recommended_tools_for_run_card(run_card)

    review = CriticAgent().review(
        run_id="continue",
        run_card=run_card,
        tool_recommendations=recommendations,
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.decision.decision == "continue"
    assert "candidate_generation_or_prediction" in review.decision.suggested_tools
    assert "candidate_generation_or_prediction" in review.recommended_next_actions


def test_candidate_artifacts_present_recommends_critic_review() -> None:
    review = CriticAgent().review(
        run_id="candidate-review",
        run_card=_run_card(OLEDDiscoveryStage.CANDIDATES_READY),
        diagnostics_report={"status": "acceptable"},
        candidate_summary={"candidate_count": 5},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    assert review.decision.decision == "run_candidate_review"
    assert review.decision.target_stage == OLEDDiscoveryStage.CRITIC_REVIEWED.value
    assert "critic_review" in review.recommended_next_actions


def test_markdown_rendering_and_write_review_are_deterministic(tmp_path: Path) -> None:
    agent = CriticAgent()
    review = agent.review(
        run_id="write",
        goal="Find OLED emitters",
        run_card=_run_card(OLEDDiscoveryStage.DIAGNOSTICS_READY),
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 2, "evidence_count": 8},
    )

    first = agent.render_markdown(review)
    second = agent.render_markdown(review)
    storage = ProjectStorage(tmp_path)
    json_path, md_path = agent.write_review(storage, "project", "write", review)

    assert first == second
    assert "# Critic Review" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first
    assert json_path.name == "critic_review.json"
    assert md_path.read_text(encoding="utf-8") == first
    assert json.loads(json_path.read_text(encoding="utf-8"))["executable"] is False


def test_critic_review_executable_is_always_false() -> None:
    review = CriticAgent().review(
        run_id="nonexec",
        goal="Find OLED emitters",
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 1, "evidence_count": 1},
    )

    assert review.executable is False
    with pytest.raises(ValidationError):
        CriticReview(
            run_id="bad",
            current_stage=OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
            decision=CriticDecision(decision="continue", reason="Bad"),
            executable=True,
        )


def test_schema_roundtrip_and_validation() -> None:
    finding = CriticFinding(
        finding_id="finding-1",
        severity="warning",
        category="diagnostics",
        summary="Weak generalization.",
        evidence_refs=[" diagnostics_report ", "diagnostics_report"],
        recommended_actions=["rerun_baseline", "rerun_baseline"],
    )
    decision = CriticDecision(
        decision="rerun_baseline",
        reason="Diagnostics need revision.",
        target_stage=OLEDDiscoveryStage.BASELINE_READY.value,
        suggested_tools=["baseline_runner"],
    )
    review = CriticReview(
        run_id="roundtrip",
        current_stage=OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
        decision=decision,
        findings=[finding],
        risk_flags=["weak_diagnostics", "weak_diagnostics"],
    )

    assert finding.evidence_refs == ["diagnostics_report"]
    assert finding.recommended_actions == ["rerun_baseline"]
    assert review.risk_flags == ["weak_diagnostics"]
    assert CriticReview.model_validate_json(review.model_dump_json()).model_dump(mode="json") == review.model_dump(mode="json")
    with pytest.raises(ValidationError):
        CriticFinding(finding_id="bad", severity="notice", category="bad", summary="Bad")
    with pytest.raises(ValidationError):
        CriticDecision(decision="ship_it", reason="Bad")


def test_critic_module_does_not_import_execution_or_governance_modules() -> None:
    before = set(sys.modules)
    CriticAgent().review(
        run_id="imports",
        goal="Find OLED emitters",
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 1, "evidence_count": 1},
    )
    newly_loaded = set(sys.modules) - before
    forbidden_tokens = (
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
    from ai4s_agent.agents.critic import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "demo",
            "--goal",
            "Find OLED emitters with high PLQY",
            "--current-stage",
            OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
            "--diagnostics-status",
            "weak",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["run_id"] == "demo"
    assert summary["decision"] == "rerun_baseline"
    assert summary["executable"] is False
    assert "findings" not in summary
