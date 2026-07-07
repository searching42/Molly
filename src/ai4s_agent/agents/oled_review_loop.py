from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.critic import CriticAgent
from ai4s_agent.agents.oled_discovery import OLEDDiscoveryLoopAgent
from ai4s_agent.agents.tool_registry import AgentToolRegistry
from ai4s_agent.schemas import AgentToolRecommendation, CriticReview, OLEDDiscoveryLoopReview, OLEDDiscoveryStage
from ai4s_agent.storage import ProjectStorage


_SAFETY_BOUNDARY = [
    "Review-only integrated OLED discovery loop artifact.",
    "Does not execute adapters, tools, or RunPlanExecutor.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class OLEDDiscoveryReviewLoopAgent:
    """Review-only integrated OLED discovery loop harness."""

    def __init__(
        self,
        *,
        discovery_agent: OLEDDiscoveryLoopAgent | None = None,
        tool_registry: AgentToolRegistry | None = None,
        critic_agent: CriticAgent | None = None,
    ) -> None:
        self.discovery_agent = discovery_agent or OLEDDiscoveryLoopAgent()
        self.tool_registry = tool_registry or AgentToolRegistry()
        self.critic_agent = critic_agent or CriticAgent()

    def build_review(
        self,
        *,
        run_id: str,
        goal: str = "",
        project_id: str | None = None,
        conversation_decision: dict[str, Any] | None = None,
        research_source_proposal: dict[str, Any] | None = None,
        research_acquisition_preparation: dict[str, Any] | None = None,
        target_modeling_brief: dict[str, Any] | None = None,
        dataset_artifacts: dict[str, Any] | None = None,
        training_package_artifacts: dict[str, Any] | None = None,
        baseline_artifacts: dict[str, Any] | None = None,
        diagnostics_report: dict[str, Any] | None = None,
        candidate_artifacts: dict[str, Any] | None = None,
        critic_review_summary: dict[str, Any] | None = None,
        dataset_summary: dict[str, Any] | None = None,
        training_package_summary: dict[str, Any] | None = None,
        baseline_summary: dict[str, Any] | None = None,
        candidate_summary: dict[str, Any] | None = None,
        provenance_summary: dict[str, Any] | None = None,
        model_package_review: dict[str, Any] | None = None,
        risk_budget: str = "medium",
        allow_gated: bool = True,
    ) -> OLEDDiscoveryLoopReview:
        run_card = self.discovery_agent.build_run_card(
            run_id=run_id,
            goal=goal,
            project_id=project_id,
            conversation_decision=conversation_decision,
            research_source_proposal=research_source_proposal,
            research_acquisition_preparation=research_acquisition_preparation,
            target_modeling_brief=target_modeling_brief,
            dataset_artifacts=dataset_artifacts,
            training_package_artifacts=training_package_artifacts,
            baseline_artifacts=baseline_artifacts,
            diagnostics_report=diagnostics_report,
            candidate_artifacts=candidate_artifacts,
            critic_review=critic_review_summary,
        )
        tool_recommendations = self.tool_registry.recommended_tools_for_run_card(
            run_card,
            risk_budget=risk_budget,
            allow_gated=allow_gated,
        )
        critic_review = self.critic_agent.review_run_card(
            run_card,
            tool_recommendations=tool_recommendations,
            dataset_summary=dataset_summary,
            training_package_summary=training_package_summary,
            baseline_summary=baseline_summary,
            diagnostics_report=diagnostics_report,
            candidate_summary=candidate_summary,
            provenance_summary=provenance_summary,
            model_package_review=model_package_review,
        )
        ready_tool_ids = [item.tool_id for item in tool_recommendations if item.ready]
        blocked_tool_ids = [item.tool_id for item in tool_recommendations if not item.ready]
        blocked_reasons = _combined_blocked_reasons(run_card.blocked_reasons, tool_recommendations, critic_review)
        risk_flags = sorted(set([*run_card.risk_flags, *critic_review.risk_flags]))
        recommended_next_action = _select_recommended_next_action(
            run_card_blocked=bool(run_card.blocked_reasons),
            critic_review=critic_review,
            tool_recommendations=tool_recommendations,
            run_card_blocked_reasons=run_card.blocked_reasons,
        )

        return OLEDDiscoveryLoopReview(
            run_id=run_card.run_id,
            project_id=run_card.project_id,
            goal=run_card.goal,
            run_card=run_card,
            tool_recommendations=tool_recommendations,
            critic_review=critic_review,
            recommended_next_action=recommended_next_action,
            ready_tool_ids=ready_tool_ids,
            blocked_tool_ids=blocked_tool_ids,
            blocked_reasons=blocked_reasons,
            risk_flags=risk_flags,
            review_summary=_review_summary(run_card.current_stage, critic_review, recommended_next_action),
            assumptions=[
                "Integrated loop review composes run-card, tool-registry, and critic review artifacts only.",
                "No adapters, execution plans, model backends, LLMs, MinerU, PDF/image readers, or network calls are executed.",
                "Supplied dictionaries are treated as synthetic summaries, not files to inspect.",
            ],
            executable=False,
        )

    def write_review(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        review: OLEDDiscoveryLoopReview,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_loop_review.json", review.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_loop_review.md"
        md_path.write_text(self.render_markdown(review), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_loop_review_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_loop_review_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(review: OLEDDiscoveryLoopReview) -> str:
        lines = [
            "# OLED Discovery Loop Review",
            "",
            f"- Run ID: {review.run_id}",
            f"- Project ID: {review.project_id or ''}",
            f"- Goal: {review.goal}",
            f"- Current Stage: {review.run_card.current_stage}",
            f"- Critic Decision: {review.critic_review.decision.decision}",
            f"- Recommended Next Action: {review.recommended_next_action}",
            f"- Executable: {str(review.executable).lower()}",
            "",
            "## Run Card Summary",
            "",
            f"- Current Stage: {review.run_card.current_stage}",
            f"- Available Artifacts: {', '.join(review.run_card.available_artifacts) or 'None'}",
            f"- Missing Artifacts: {', '.join(review.run_card.missing_artifacts) or 'None'}",
            "",
            "## Tool Recommendations",
            "",
            "| Tool ID | Ready | Missing Inputs | Blocked Reasons | Gates |",
            "| --- | --- | --- | --- | --- |",
        ]
        if review.tool_recommendations:
            for item in review.tool_recommendations:
                lines.append(
                    f"| {item.tool_id} | {str(item.ready).lower()} | {', '.join(item.missing_inputs)} | {', '.join(item.blocked_reasons)} | {', '.join(item.required_gates)} |"
                )
        else:
            lines.append("| None | false |  |  |  |")
        lines.extend(
            [
                "",
                "## Critic Findings",
                "",
                "| Severity | Category | Summary | Recommended Actions |",
                "| --- | --- | --- | --- |",
            ]
        )
        if review.critic_review.findings:
            for finding in review.critic_review.findings:
                lines.append(
                    f"| {finding.severity} | {finding.category} | {finding.summary} | {', '.join(finding.recommended_actions)} |"
                )
        else:
            lines.append("| info | none | No critic findings. |  |")
        lines.extend(["", "## Risk Flags", ""])
        lines.extend(_markdown_list(review.risk_flags))
        lines.extend(["", "## Blocked Reasons", ""])
        lines.extend(_markdown_list(review.blocked_reasons))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only integrated OLED discovery loop review.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--diagnostics-status", default="")
    parser.add_argument("--risk-budget", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--disallow-gated", action="store_true")
    args = parser.parse_args(argv)

    diagnostics = {"status": args.diagnostics_status} if args.diagnostics_status else None
    synthetic_artifacts = bool(args.diagnostics_status)
    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id=args.run_id,
        goal=args.goal,
        project_id=args.project_id,
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"} if synthetic_artifacts else None,
        training_package_artifacts={"training_rows": "training.jsonl"} if synthetic_artifacts else None,
        baseline_artifacts={"metrics": "metrics.json"} if synthetic_artifacts else None,
        diagnostics_report=diagnostics,
        provenance_summary={"source_count": 1, "evidence_count": 1} if synthetic_artifacts else None,
        risk_budget=args.risk_budget,
        allow_gated=not args.disallow_gated,
    )
    summary = {
        "run_id": review.run_id,
        "current_stage": review.run_card.current_stage,
        "critic_decision": review.critic_review.decision.decision,
        "recommended_next_action": review.recommended_next_action,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _select_recommended_next_action(
    *,
    run_card_blocked: bool,
    critic_review: CriticReview,
    tool_recommendations: list[AgentToolRecommendation],
    run_card_blocked_reasons: list[str],
) -> str:
    if critic_review.decision.decision != "continue":
        return critic_review.decision.decision
    ready = [item.tool_id for item in tool_recommendations if item.ready]
    if ready:
        return sorted(ready)[0]
    blocked = [item for item in tool_recommendations if not item.ready]
    if blocked:
        first = sorted(blocked, key=lambda item: item.tool_id)[0]
        if first.missing_inputs:
            return f"resolve_missing_inputs:{first.missing_inputs[0]}"
        if first.required_gates:
            return f"resolve_gate:{first.required_gates[0]}"
        if first.blocked_reasons:
            return f"resolve_blocked_tool:{first.blocked_reasons[0]}"
    if run_card_blocked and run_card_blocked_reasons:
        return f"resolve_run_card_blocker:{run_card_blocked_reasons[0]}"
    return "human_review_required"


def _combined_blocked_reasons(
    run_card_reasons: list[str],
    recommendations: list[AgentToolRecommendation],
    critic_review: CriticReview,
) -> list[str]:
    reasons = [*run_card_reasons, *critic_review.blocked_reasons]
    for item in recommendations:
        reasons.extend(item.blocked_reasons)
        reasons.extend(f"missing_input:{missing}" for missing in item.missing_inputs)
    return sorted(set(reason for reason in reasons if reason))


def _review_summary(stage: str, critic_review: CriticReview, recommended_next_action: str) -> str:
    return (
        f"OLED discovery loop is at `{stage}`; critic decision is "
        f"`{critic_review.decision.decision}` and next action is `{recommended_next_action}`."
    )


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
