from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.schemas import (
    AgentToolRecommendation,
    CriticDecision,
    CriticFinding,
    CriticReview,
    OLEDDiscoveryRunCard,
    OLEDDiscoveryStage,
)
from ai4s_agent.storage import ProjectStorage


_STAGE_RANK = {
    OLEDDiscoveryStage.BLOCKED.value: 0,
    OLEDDiscoveryStage.INTENT_CAPTURED.value: 1,
    OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value: 2,
    OLEDDiscoveryStage.ACQUISITION_PREPARED.value: 3,
    OLEDDiscoveryStage.DATASET_READY.value: 4,
    OLEDDiscoveryStage.TRAINING_PACKAGE_READY.value: 5,
    OLEDDiscoveryStage.BASELINE_READY.value: 6,
    OLEDDiscoveryStage.DIAGNOSTICS_READY.value: 7,
    OLEDDiscoveryStage.CANDIDATES_READY.value: 8,
    OLEDDiscoveryStage.CRITIC_REVIEWED.value: 9,
    OLEDDiscoveryStage.NEXT_ACTION_PROPOSED.value: 10,
}
_DIAGNOSTIC_RISK_TOKENS = (
    "weak",
    "blocked",
    "rerun",
    "failed",
    "high_value_underprediction",
    "prediction_range_compression",
    "weak_generalization",
)
_OVERCLAIM_TOKENS = (
    "promote",
    "publish",
    "validated",
    "external_publication",
    "benchmark_validated",
)
_SAFETY_BOUNDARY = [
    "Review-only critic artifact.",
    "Does not execute adapters or tools.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class CriticAgent:
    """Review-only critic for OLED discovery run decisions."""

    def review(
        self,
        *,
        run_id: str,
        goal: str = "",
        project_id: str | None = None,
        run_card: OLEDDiscoveryRunCard | dict[str, Any] | None = None,
        tool_recommendations: list[AgentToolRecommendation | dict[str, Any]] | None = None,
        dataset_summary: dict[str, Any] | None = None,
        training_package_summary: dict[str, Any] | None = None,
        baseline_summary: dict[str, Any] | None = None,
        diagnostics_report: dict[str, Any] | None = None,
        candidate_summary: dict[str, Any] | None = None,
        provenance_summary: dict[str, Any] | None = None,
        model_package_review: dict[str, Any] | None = None,
    ) -> CriticReview:
        card = _coerce_run_card(run_card)
        recommendations = _coerce_recommendations(tool_recommendations)
        clean_goal = str(goal or (card.goal if card else "") or "").strip()
        clean_project_id = str(project_id or (card.project_id if card else "") or "").strip() or None
        current_stage = str(card.current_stage if card else _infer_stage(diagnostics_report, candidate_summary)).strip()

        findings: list[CriticFinding] = []
        risk_flags = list(card.risk_flags if card else [])
        blocked_reasons = list(card.blocked_reasons if card else [])

        objective_missing = not clean_goal or current_stage == OLEDDiscoveryStage.BLOCKED.value or "missing_discovery_objective" in blocked_reasons
        if objective_missing:
            blocked_reasons.append("missing_discovery_objective")
            findings.append(
                _finding(
                    "objective",
                    "critical",
                    "objective",
                    "OLED discovery objective is missing or the run card is blocked.",
                    ["run_card"],
                    ["provide_oled_discovery_objective"],
                )
            )

        stage_after_dataset = _STAGE_RANK.get(current_stage, 0) >= _STAGE_RANK[OLEDDiscoveryStage.TRAINING_PACKAGE_READY.value]
        dataset_available = _has_summary(dataset_summary) or (card is not None and "dataset_artifacts" in card.available_artifacts)
        provenance_gap = stage_after_dataset and (not dataset_available or not _has_positive_counts(provenance_summary, ("source_count", "evidence_count")))
        if provenance_gap:
            risk_flags.append("insufficient_provenance")
            findings.append(
                _finding(
                    "provenance",
                    "critical" if not provenance_summary else "warning",
                    "provenance",
                    "Dataset/model stage requires nonempty provenance and evidence summaries before stronger claims.",
                    ["dataset_summary", "provenance_summary"],
                    ["request_more_evidence"],
                )
            )

        leakage = _has_leakage_signal(dataset_summary) or _has_leakage_signal(training_package_summary)
        if leakage:
            risk_flags.append("split_leakage_risk")
            findings.append(
                _finding(
                    "data_leakage",
                    "critical",
                    "data_leakage",
                    "Dataset or training package summary indicates split leakage or train/test overlap.",
                    ["dataset_summary", "training_package_summary"],
                    ["revise_data"],
                )
            )

        diagnostic_flags = _diagnostic_flags(diagnostics_report)
        if diagnostic_flags:
            risk_flags.extend(diagnostic_flags)
            if "weak_diagnostics" not in risk_flags:
                risk_flags.append("weak_diagnostics")
            findings.append(
                _finding(
                    "diagnostics",
                    "warning",
                    "diagnostics",
                    "Diagnostics indicate weak, blocked, failed, compressed, or rerun-needed model behavior.",
                    ["diagnostics_report"],
                    ["rerun_baseline", "revise_model"],
                )
            )

        candidate_domain_risk = _candidate_domain_risk(candidate_summary)
        if candidate_domain_risk:
            risk_flags.append("candidate_domain_risk")
            findings.append(
                _finding(
                    "candidate_domain",
                    "warning",
                    "candidate_domain",
                    "Candidate summary indicates out-of-domain candidates, invalid SMILES, or missing predictions.",
                    ["candidate_summary"],
                    ["run_candidate_review", "revise_model"],
                )
            )

        overclaim = _contains_terms(candidate_summary, _OVERCLAIM_TOKENS) or _contains_terms(model_package_review, _OVERCLAIM_TOKENS)
        if overclaim and (diagnostic_flags or provenance_gap):
            risk_flags.append("overclaim_risk")
            findings.append(
                _finding(
                    "overclaim",
                    "critical",
                    "overclaim",
                    "Promotion, publication, or validation wording appears before diagnostics/provenance support it.",
                    ["candidate_summary", "model_package_review", "diagnostics_report", "provenance_summary"],
                    ["block_promotion"],
                )
            )

        decision = _select_decision(
            current_stage=current_stage,
            objective_missing=objective_missing,
            provenance_gap=provenance_gap,
            leakage=leakage,
            diagnostic_flags=diagnostic_flags,
            candidate_domain_risk=candidate_domain_risk,
            overclaim=overclaim and (diagnostic_flags or provenance_gap),
            candidate_summary=candidate_summary,
            recommendations=recommendations,
        )

        return CriticReview(
            run_id=str(run_id or (card.run_id if card else "") or "").strip(),
            project_id=clean_project_id,
            goal=clean_goal,
            current_stage=current_stage,
            decision=decision,
            findings=findings,
            risk_flags=sorted(set(risk_flags)),
            blocked_reasons=sorted(set(blocked_reasons)),
            recommended_next_actions=list(decision.suggested_tools),
            assumptions=[
                "Critic review is deterministic and review-only.",
                "Supplied summaries are treated as already-redacted dictionaries, not files to inspect.",
                "No tools, adapters, LLMs, MinerU, PDF/image readers, model backends, or network calls are executed.",
            ],
            executable=False,
        )

    def review_run_card(
        self,
        run_card: OLEDDiscoveryRunCard,
        *,
        tool_recommendations: list[AgentToolRecommendation] | None = None,
        dataset_summary: dict[str, Any] | None = None,
        training_package_summary: dict[str, Any] | None = None,
        baseline_summary: dict[str, Any] | None = None,
        diagnostics_report: dict[str, Any] | None = None,
        candidate_summary: dict[str, Any] | None = None,
        provenance_summary: dict[str, Any] | None = None,
        model_package_review: dict[str, Any] | None = None,
    ) -> CriticReview:
        return self.review(
            run_id=run_card.run_id,
            goal=run_card.goal,
            project_id=run_card.project_id,
            run_card=run_card,
            tool_recommendations=tool_recommendations,
            dataset_summary=dataset_summary,
            training_package_summary=training_package_summary,
            baseline_summary=baseline_summary,
            diagnostics_report=diagnostics_report,
            candidate_summary=candidate_summary,
            provenance_summary=provenance_summary,
            model_package_review=model_package_review,
        )

    def write_review(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        review: CriticReview,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "critic_review.json", review.model_dump(mode="json"))
        md_path = run_dir / "critic_review.md"
        md_path.write_text(self.render_markdown(review), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "critic_review_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "critic_review_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(review: CriticReview) -> str:
        lines = [
            "# Critic Review",
            "",
            f"- Run ID: {review.run_id}",
            f"- Project ID: {review.project_id or ''}",
            f"- Goal: {review.goal}",
            f"- Current Stage: {review.current_stage}",
            f"- Decision: {review.decision.decision}",
            f"- Executable: {str(review.executable).lower()}",
            "",
            "## Findings",
            "",
            "| Severity | Category | Summary | Recommended Actions |",
            "| --- | --- | --- | --- |",
        ]
        if review.findings:
            for finding in review.findings:
                lines.append(
                    f"| {finding.severity} | {finding.category} | {finding.summary} | {', '.join(finding.recommended_actions)} |"
                )
        else:
            lines.append("| info | none | No critic findings. |  |")
        lines.extend(["", "## Risk Flags", ""])
        lines.extend(_markdown_list(review.risk_flags))
        lines.extend(["", "## Blocked Reasons", ""])
        lines.extend(_markdown_list(review.blocked_reasons))
        lines.extend(["", "## Recommended Next Actions", ""])
        lines.extend(_markdown_list(review.recommended_next_actions))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery critic review.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--current-stage", default=OLEDDiscoveryStage.INTENT_CAPTURED.value)
    parser.add_argument("--diagnostics-status", default="")
    args = parser.parse_args(argv)

    run_card = {
        "run_id": args.run_id,
        "project_id": args.project_id,
        "goal": args.goal,
        "current_stage": args.current_stage,
        "available_artifacts": ["dataset_artifacts", "diagnostics_report"] if args.diagnostics_status else [],
        "executable": False,
    }
    diagnostics = {"status": args.diagnostics_status} if args.diagnostics_status else None
    provenance = {"source_count": 1, "evidence_count": 1} if args.diagnostics_status else None
    review = CriticAgent().review(
        run_id=args.run_id,
        goal=args.goal,
        project_id=args.project_id,
        run_card=run_card,
        diagnostics_report=diagnostics,
        provenance_summary=provenance,
    )
    summary = {
        "run_id": review.run_id,
        "current_stage": review.current_stage,
        "decision": review.decision.decision,
        "risk_flags": review.risk_flags,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _coerce_run_card(value: OLEDDiscoveryRunCard | dict[str, Any] | None) -> OLEDDiscoveryRunCard | None:
    if value is None:
        return None
    if isinstance(value, OLEDDiscoveryRunCard):
        return value
    return OLEDDiscoveryRunCard.model_validate(value)


def _coerce_recommendations(values: list[AgentToolRecommendation | dict[str, Any]] | None) -> list[AgentToolRecommendation]:
    recommendations: list[AgentToolRecommendation] = []
    for value in values or []:
        recommendations.append(value if isinstance(value, AgentToolRecommendation) else AgentToolRecommendation.model_validate(value))
    return recommendations


def _infer_stage(diagnostics_report: dict[str, Any] | None, candidate_summary: dict[str, Any] | None) -> str:
    if _has_summary(candidate_summary):
        return OLEDDiscoveryStage.CANDIDATES_READY.value
    if _has_summary(diagnostics_report):
        return OLEDDiscoveryStage.DIAGNOSTICS_READY.value
    return OLEDDiscoveryStage.INTENT_CAPTURED.value


def _has_summary(value: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(_usable_value(item) for item in value.values())


def _usable_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return True
    if isinstance(value, Mapping):
        return _has_summary(value)
    if isinstance(value, list | tuple | set):
        return any(_usable_value(item) for item in value)
    return True


def _has_positive_counts(summary: Mapping[str, Any] | None, keys: tuple[str, ...]) -> bool:
    if not isinstance(summary, Mapping):
        return False
    return all(_numeric(summary.get(key)) > 0 for key in keys)


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, list | tuple | set | dict):
            return float(len(value))
        return 0.0


def _has_leakage_signal(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    leakage_terms = ("leakage", "split_contamination", "train_test_overlap", "train/test_overlap", "scaffold_leakage")
    for key, value in summary.items():
        normalized_key = str(key).lower()
        if any(term in normalized_key for term in leakage_terms) and _truthy_or_positive(value):
            return True
    return False


def _truthy_or_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null", "no", "n/a"}
    if isinstance(value, list | tuple | set | dict):
        return len(value) > 0
    return bool(value)


def _diagnostic_flags(summary: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(summary, Mapping):
        return []
    text = _flatten_text(summary)
    flags: list[str] = []
    for token in _DIAGNOSTIC_RISK_TOKENS:
        if token in text:
            flags.append(token if token not in {"weak", "blocked", "rerun", "failed"} else "weak_diagnostics")
    raw_flags = summary.get("risk_flags")
    if isinstance(raw_flags, list):
        flags.extend(str(item).strip() for item in raw_flags if str(item).strip())
    return sorted(set(flags))


def _candidate_domain_risk(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    for key in ("out_of_domain_count", "invalid_smiles_count", "missing_prediction_count"):
        if _numeric(summary.get(key)) > 0:
            return True
    return False


def _contains_terms(summary: Mapping[str, Any] | None, terms: tuple[str, ...]) -> bool:
    if not isinstance(summary, Mapping):
        return False
    text = _flatten_text(summary)
    return any(term in text for term in terms)


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values()).lower()
    if isinstance(value, list | tuple | set):
        return " ".join(_flatten_text(item) for item in value).lower()
    return str(value).lower()


def _select_decision(
    *,
    current_stage: str,
    objective_missing: bool,
    provenance_gap: bool,
    leakage: bool,
    diagnostic_flags: list[str],
    candidate_domain_risk: bool,
    overclaim: bool,
    candidate_summary: dict[str, Any] | None,
    recommendations: list[AgentToolRecommendation],
) -> CriticDecision:
    if objective_missing:
        return CriticDecision(
            decision="request_more_evidence",
            reason="OLED discovery objective or unblocked run-card state is required before planning can proceed.",
            target_stage=OLEDDiscoveryStage.INTENT_CAPTURED.value,
            suggested_tools=["conversation_modeling_payload"],
        )
    if overclaim:
        return CriticDecision(
            decision="block_promotion",
            reason="Promotion/publication/validation wording is unsupported by diagnostics or provenance.",
            target_stage=current_stage,
            suggested_tools=["critic_review"],
        )
    if leakage:
        return CriticDecision(
            decision="revise_data",
            reason="Split leakage or train/test overlap must be resolved before model/candidate decisions.",
            target_stage=OLEDDiscoveryStage.DATASET_READY.value,
            suggested_tools=["leakage_split", "training_package"],
        )
    if provenance_gap:
        return CriticDecision(
            decision="request_more_evidence",
            reason="Provenance and evidence summaries are insufficient for the current discovery stage.",
            target_stage=OLEDDiscoveryStage.ACQUISITION_PREPARED.value,
            suggested_tools=["retrieve_evidence", "track_citation_provenance"],
        )
    if diagnostic_flags:
        return CriticDecision(
            decision="rerun_baseline",
            reason="Diagnostics indicate weak or rerun-needed baseline/model behavior.",
            target_stage=OLEDDiscoveryStage.BASELINE_READY.value,
            suggested_tools=["baseline_runner", "diagnostics_report"],
        )
    if candidate_domain_risk:
        return CriticDecision(
            decision="run_candidate_review",
            reason="Candidate artifacts need review for domain validity and missing predictions.",
            target_stage=OLEDDiscoveryStage.CRITIC_REVIEWED.value,
            suggested_tools=["critic_review"],
        )
    if current_stage == OLEDDiscoveryStage.CANDIDATES_READY.value or _has_summary(candidate_summary):
        return CriticDecision(
            decision="run_candidate_review",
            reason="Candidate artifacts are available and need critic review.",
            target_stage=OLEDDiscoveryStage.CRITIC_REVIEWED.value,
            suggested_tools=["critic_review"],
        )
    suggested_tools = [recommendation.tool_id for recommendation in recommendations if recommendation.ready]
    if not suggested_tools and current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value:
        suggested_tools = ["candidate_generation_or_prediction"]
    return CriticDecision(
        decision="continue",
        reason="No blocking critic findings were detected for the supplied summaries.",
        target_stage=OLEDDiscoveryStage.CANDIDATES_READY.value if current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY.value else current_stage,
        suggested_tools=suggested_tools,
    )


def _finding(
    suffix: str,
    severity: str,
    category: str,
    summary: str,
    evidence_refs: list[str],
    recommended_actions: list[str],
) -> CriticFinding:
    return CriticFinding(
        finding_id=f"critic:{suffix}",
        severity=severity,
        category=category,
        summary=summary,
        evidence_refs=evidence_refs,
        recommended_actions=recommended_actions,
    )


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
