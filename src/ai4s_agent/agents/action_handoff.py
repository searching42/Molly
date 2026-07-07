from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.oled_review_loop import OLEDDiscoveryReviewLoopAgent
from ai4s_agent.agents.tool_registry import AgentToolRegistry
from ai4s_agent.schemas import (
    AgentToolRecommendation,
    AgentToolSpec,
    OLEDDiscoveryActionHandoff,
    OLEDDiscoveryLoopReview,
    OLEDDiscoveryStage,
)
from ai4s_agent.storage import ProjectStorage


_SAFETY_BOUNDARY = [
    "Review-only action handoff artifact.",
    "Does not execute adapters, tools, gates, or RunPlanExecutor.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]
_CRITIC_ACTION_TOOL_MAP = {
    "request_more_evidence": ["retrieve_evidence", "research_source_proposal"],
    "revise_data": ["leakage_split", "training_package"],
    "revise_model": ["baseline_runner", "diagnostics_report"],
    "rerun_baseline": ["baseline_runner"],
    "run_candidate_review": ["critic_review"],
    "block_promotion": ["critic_review"],
}


class OLEDDiscoveryActionHandoffAgent:
    """Review-only handoff from OLED discovery loop review to controlled action planning."""

    def __init__(self, *, tool_registry: AgentToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or AgentToolRegistry()

    def build_handoff(
        self,
        *,
        loop_review: OLEDDiscoveryLoopReview | dict[str, Any],
        risk_budget: str = "medium",
        allow_gated: bool = True,
    ) -> OLEDDiscoveryActionHandoff:
        review = loop_review if isinstance(loop_review, OLEDDiscoveryLoopReview) else OLEDDiscoveryLoopReview.model_validate(loop_review)
        specs = {tool.tool_id: tool for tool in self.tool_registry.list_tools()}
        recommendations = {item.tool_id: item for item in review.tool_recommendations}
        action = str(review.recommended_next_action or "").strip()

        resolver = _resolver_blocker(action)
        selected_tool_id = "" if resolver else _map_action_to_tool_id(action, review, recommendations, specs)
        spec = specs.get(selected_tool_id)
        recommendation = recommendations.get(selected_tool_id)

        input_artifacts: list[str] = []
        output_artifacts: list[str] = []
        missing_inputs: list[str] = []
        required_gates: list[str] = []
        required_permissions: list[str] = []
        blocked_reasons: list[str] = []
        target_stage = ""
        selected_task_id = ""

        if resolver:
            missing_inputs, blocked_reasons = resolver
        elif spec is None:
            blocked_reasons = ["no_selected_tool"]
        else:
            input_artifacts = list(spec.input_artifacts)
            output_artifacts = list(spec.output_artifacts)
            required_gates = list(spec.required_gates)
            required_permissions = list(spec.required_permissions)
            selected_task_id = spec.suggested_tasks[0] if spec.suggested_tasks else spec.tool_id
            target_stage = recommendation.target_stage if recommendation else _target_stage_from_spec(spec)
            missing_inputs = list(recommendation.missing_inputs) if recommendation else _missing_inputs_from_run_card(spec, review)
            blocked_reasons = list(recommendation.blocked_reasons) if recommendation else []
            if missing_inputs and "missing_required_inputs" not in blocked_reasons:
                blocked_reasons.append("missing_required_inputs")

        ready = bool(spec) and not missing_inputs and not blocked_reasons
        rationale = _rationale(action, selected_tool_id, ready, required_gates, missing_inputs, blocked_reasons)

        return OLEDDiscoveryActionHandoff(
            run_id=review.run_id,
            project_id=review.project_id,
            goal=review.goal,
            source_review_id=f"oled_discovery_loop_review:{review.run_id}",
            recommended_next_action=action,
            critic_decision=review.critic_review.decision.decision,
            selected_tool_id=selected_tool_id if spec else "",
            selected_task_id=selected_task_id,
            target_stage=target_stage,
            ready=ready,
            executable=False,
            input_artifacts=input_artifacts,
            missing_inputs=missing_inputs,
            output_artifacts=output_artifacts,
            required_gates=required_gates,
            required_permissions=required_permissions,
            blocked_reasons=blocked_reasons,
            risk_flags=review.risk_flags,
            payload_template=_payload_template(selected_tool_id, review.run_id, review.goal),
            rationale=rationale,
            assumptions=[
                "Action handoff is review-only and prepares future controlled planning only.",
                "Payload templates contain placeholders and do not read artifact contents.",
                "Ready means ready for human review or future gated planner handoff, not execution.",
                "No adapters, RunPlanExecutor, gate approval, LLMs, MinerU, PDF/image readers, model backends, or network calls are executed.",
            ],
        )

    def write_handoff(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        handoff: OLEDDiscoveryActionHandoff,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_action_handoff.json", handoff.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_action_handoff.md"
        md_path.write_text(self.render_markdown(handoff), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_action_handoff_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_action_handoff_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(handoff: OLEDDiscoveryActionHandoff) -> str:
        lines = [
            "# OLED Discovery Action Handoff",
            "",
            f"- Run ID: {handoff.run_id}",
            f"- Project ID: {handoff.project_id or ''}",
            f"- Goal: {handoff.goal}",
            f"- Recommended Next Action: {handoff.recommended_next_action}",
            f"- Critic Decision: {handoff.critic_decision}",
            f"- Selected Tool: {handoff.selected_tool_id}",
            f"- Selected Task: {handoff.selected_task_id}",
            f"- Target Stage: {handoff.target_stage}",
            f"- Ready: {str(handoff.ready).lower()}",
            f"- Executable: {str(handoff.executable).lower()}",
            "",
            "## Inputs",
            "",
        ]
        lines.extend(_markdown_list(handoff.input_artifacts))
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(_markdown_list(handoff.missing_inputs))
        lines.extend(["", "## Required Gates", ""])
        lines.extend(_markdown_list(handoff.required_gates))
        lines.extend(["", "## Required Permissions", ""])
        lines.extend(_markdown_list(handoff.required_permissions))
        lines.extend(["", "## Payload Template", "", "```json"])
        lines.append(json.dumps(handoff.payload_template, sort_keys=True, indent=2))
        lines.extend(["```", "", "## Rationale", ""])
        lines.extend(_markdown_list(handoff.rationale))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery action handoff.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--recommended-next-action", required=True)
    args = parser.parse_args(argv)

    review = OLEDDiscoveryReviewLoopAgent().build_review(
        run_id=args.run_id,
        goal=args.goal,
        project_id=args.project_id,
        dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
        training_package_artifacts={"training_rows": "training.jsonl"},
        baseline_artifacts={"metrics": "metrics.json"},
        diagnostics_report={"status": "acceptable"},
        provenance_summary={"source_count": 1, "evidence_count": 1},
    )
    review = review.model_copy(update={"recommended_next_action": args.recommended_next_action})
    handoff = OLEDDiscoveryActionHandoffAgent().build_handoff(loop_review=review)
    summary = {
        "run_id": handoff.run_id,
        "recommended_next_action": handoff.recommended_next_action,
        "selected_tool_id": handoff.selected_tool_id,
        "ready": handoff.ready,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _map_action_to_tool_id(
    action: str,
    review: OLEDDiscoveryLoopReview,
    recommendations: dict[str, AgentToolRecommendation],
    specs: dict[str, AgentToolSpec],
) -> str:
    if action in recommendations or action in specs:
        return action
    if action == "continue":
        ready = sorted((item.tool_id for item in review.tool_recommendations if item.ready))
        return ready[0] if ready else ""
    if action == "stop":
        return ""
    for candidate in _CRITIC_ACTION_TOOL_MAP.get(action, []):
        if candidate in recommendations or candidate in specs:
            return candidate
    return ""


def _resolver_blocker(action: str) -> tuple[list[str], list[str]] | None:
    resolvers = {
        "resolve_missing_inputs:": lambda value: ([value], [f"missing_input:{value}"]),
        "resolve_gate:": lambda value: ([], [f"gate_approval_required:{value}"]),
        "resolve_blocked_tool:": lambda value: ([], [value]),
        "resolve_run_card_blocker:": lambda value: ([], [value]),
    }
    for prefix, builder in resolvers.items():
        if action.startswith(prefix):
            value = action[len(prefix) :].strip()
            return builder(value)
    if action == "human_review_required":
        return ([], ["no_selected_tool"])
    return None


def _missing_inputs_from_run_card(spec: AgentToolSpec, review: OLEDDiscoveryLoopReview) -> list[str]:
    available = set(review.run_card.available_artifacts)
    return [artifact for artifact in spec.input_artifacts if artifact not in available]


def _target_stage_from_spec(spec: AgentToolSpec) -> str:
    mapping = {
        "research_source_proposal": OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value,
        "research_acquisition_preparation": OLEDDiscoveryStage.ACQUISITION_PREPARED.value,
        "dataset_artifacts": OLEDDiscoveryStage.DATASET_READY.value,
        "training_package_artifacts": OLEDDiscoveryStage.TRAINING_PACKAGE_READY.value,
        "baseline_artifacts": OLEDDiscoveryStage.BASELINE_READY.value,
        "diagnostics_report": OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
        "candidate_artifacts": OLEDDiscoveryStage.CANDIDATES_READY.value,
        "critic_review": OLEDDiscoveryStage.CRITIC_REVIEWED.value,
    }
    for output in reversed(spec.output_artifacts):
        if output in mapping:
            return mapping[output]
    return spec.discovery_stages[-1] if spec.discovery_stages else ""


def _payload_template(tool_id: str, run_id: str, goal: str) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {
        "research_source_proposal": {"run_id": run_id, "goal": goal, "review_only": True},
        "research_acquisition_preparation": {
            "run_id": run_id,
            "research_source_proposal": "<research_source_proposal>",
            "review_only": True,
        },
        "baseline_runner": {
            "run_id": run_id,
            "training_package_artifacts": "<training_package_artifacts>",
            "review_only": True,
        },
        "candidate_generation_or_prediction": {
            "run_id": run_id,
            "diagnostics_report": "<diagnostics_report>",
            "training_package_artifacts": "<training_package_artifacts>",
            "review_only": True,
        },
        "critic_review": {"run_id": run_id, "candidate_summary": "<candidate_summary>", "review_only": True},
        "retrieve_evidence": {"run_id": run_id, "corpus_index": "<corpus_index>", "review_only": True},
        "leakage_split": {"run_id": run_id, "dataset_artifacts": "<dataset_artifacts>", "review_only": True},
        "training_package": {"run_id": run_id, "dataset_artifacts": "<dataset_artifacts>", "review_only": True},
        "acquire_literature_sources": {
            "run_id": run_id,
            "research_acquisition_preparation": "<research_acquisition_preparation>",
            "review_only": True,
        },
    }
    if tool_id in templates:
        return templates[tool_id]
    if tool_id:
        return {"run_id": run_id, "selected_tool_id": tool_id, "review_only": True}
    return {"run_id": run_id, "review_only": True}


def _rationale(
    action: str,
    selected_tool_id: str,
    ready: bool,
    required_gates: list[str],
    missing_inputs: list[str],
    blocked_reasons: list[str],
) -> list[str]:
    rationale = [f"recommended_next_action:{action}"]
    if selected_tool_id:
        rationale.append(f"selected_tool:{selected_tool_id}")
    else:
        rationale.append("no executable tool selected; handoff remains review-only")
    if ready and required_gates:
        rationale.append("ready for gated review, not execution")
    elif ready:
        rationale.append("ready for human review, not execution")
    if missing_inputs:
        rationale.append(f"missing_inputs:{', '.join(missing_inputs)}")
    if blocked_reasons:
        rationale.append(f"blocked_reasons:{', '.join(blocked_reasons)}")
    rationale.extend(_SAFETY_BOUNDARY)
    return rationale


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
