from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence

from ai4s_agent.schemas import (
    AgentToolRecommendation,
    AgentToolRegistrySnapshot,
    AgentToolSpec,
    OLEDDiscoveryRunCard,
    OLEDDiscoveryStage,
)


_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_REGISTRY_ID = "agent-tool-registry:oled-discovery:v1"
_SAFETY_BOUNDARY = [
    "Review-only capability map.",
    "Does not execute adapters or tools.",
    "Does not call LLMs, MinerU, PDFs/images, model backends, or external network services.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class AgentToolRegistry:
    """Review-only tool capability registry for Agent planning."""

    def __init__(self, tools: Iterable[AgentToolSpec] | None = None) -> None:
        self._tools = sorted(list(tools) if tools is not None else _built_in_tools(), key=lambda tool: tool.tool_id)

    def list_tools(self) -> list[AgentToolSpec]:
        return list(self._tools)

    def tools_for_stage(self, stage: str) -> list[AgentToolSpec]:
        normalized = str(stage or "").strip()
        return [tool for tool in self._tools if normalized in tool.discovery_stages]

    def recommend_tools(
        self,
        *,
        current_stage: str,
        available_artifacts: list[str] | None = None,
        risk_budget: str = "medium",
        allow_gated: bool = True,
    ) -> list[AgentToolRecommendation]:
        available = {str(item).strip() for item in (available_artifacts or []) if str(item).strip()}
        budget = _normalize_risk_budget(risk_budget)
        recommendations: list[AgentToolRecommendation] = []
        for tool in self.tools_for_stage(current_stage):
            missing_inputs = [artifact for artifact in tool.input_artifacts if artifact not in available]
            blocked_reasons: list[str] = []
            if missing_inputs:
                blocked_reasons.append("missing_required_inputs")
            if _RISK_RANK[tool.risk_level] > _RISK_RANK[budget]:
                blocked_reasons.append("risk_level_exceeds_budget")
            if tool.required_gates and not allow_gated:
                blocked_reasons.append("gated_tool_not_allowed")
            ready = not missing_inputs and not blocked_reasons
            recommendations.append(
                AgentToolRecommendation(
                    tool_id=tool.tool_id,
                    reason=_recommendation_reason(tool, ready, blocked_reasons),
                    target_stage=_target_stage_for_tool(tool),
                    ready=ready,
                    missing_inputs=missing_inputs,
                    blocked_reasons=blocked_reasons,
                    required_gates=list(tool.required_gates),
                    executable=False,
                )
            )
        return recommendations

    def recommended_tools_for_run_card(
        self,
        run_card: OLEDDiscoveryRunCard,
        *,
        risk_budget: str = "medium",
        allow_gated: bool = True,
    ) -> list[AgentToolRecommendation]:
        return self.recommend_tools(
            current_stage=run_card.current_stage,
            available_artifacts=run_card.available_artifacts,
            risk_budget=risk_budget,
            allow_gated=allow_gated,
        )

    def snapshot(self) -> AgentToolRegistrySnapshot:
        tools = self.list_tools()
        return AgentToolRegistrySnapshot(
            registry_id=_REGISTRY_ID,
            tool_count=len(tools),
            tools=tools,
            assumptions=[
                "AgentToolRegistry is a review-only capability map, not an execution registry.",
                "Readiness is inferred from supplied artifact names only.",
                "Gated or higher-risk tools still require explicit review before any executor action.",
            ],
            executable=False,
        )

    @staticmethod
    def render_markdown(snapshot: AgentToolRegistrySnapshot) -> str:
        lines = [
            "# Agent Tool Registry",
            "",
            f"- Registry ID: {snapshot.registry_id}",
            f"- Tool Count: {snapshot.tool_count}",
            f"- Executable: {str(snapshot.executable).lower()}",
            "",
            "## Tools",
            "",
            "| Tool ID | Stages | Risk | Gates | Inputs | Outputs |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for tool in sorted(snapshot.tools, key=lambda item: item.tool_id):
            lines.append(
                "| "
                + " | ".join(
                    [
                        tool.tool_id,
                        ", ".join(tool.discovery_stages),
                        tool.risk_level,
                        ", ".join(tool.required_gates),
                        ", ".join(tool.input_artifacts),
                        ", ".join(tool.output_artifacts),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Safety Boundary",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review-only Agent tool registry for OLED discovery planning.")
    parser.add_argument("--stage", default="", help="Discovery stage for recommendations.")
    parser.add_argument("--available-artifact", action="append", default=[], help="Available artifact name; repeatable.")
    parser.add_argument("--risk-budget", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--disallow-gated", action="store_true", help="Mark gated tools as not ready.")
    args = parser.parse_args(argv)

    registry = AgentToolRegistry()
    recommendations = registry.recommend_tools(
        current_stage=args.stage,
        available_artifacts=args.available_artifact,
        risk_budget=args.risk_budget,
        allow_gated=not args.disallow_gated,
    )
    summary = {
        "stage": args.stage,
        "recommendation_count": len(recommendations),
        "recommended_tool_ids": [recommendation.tool_id for recommendation in recommendations],
        "ready_tool_ids": [recommendation.tool_id for recommendation in recommendations if recommendation.ready],
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _tool(
    tool_id: str,
    label: str,
    stages: list[OLEDDiscoveryStage],
    *,
    suggested_tasks: list[str] | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    risk: str = "low",
    gates: list[str] | None = None,
    permissions: list[str] | None = None,
    failure_modes: list[str] | None = None,
    description: str = "",
) -> AgentToolSpec:
    return AgentToolSpec(
        tool_id=tool_id,
        label=label,
        description=description,
        discovery_stages=[stage.value for stage in stages],
        suggested_tasks=suggested_tasks or [tool_id],
        input_artifacts=inputs or [],
        output_artifacts=outputs or [],
        risk_level=risk,
        required_gates=gates or [],
        required_permissions=permissions or [],
        failure_modes=failure_modes or [],
        safety_boundary=list(_SAFETY_BOUNDARY),
        executable=False,
    )


def _built_in_tools() -> list[AgentToolSpec]:
    return [
        _tool(
            "conversation_modeling_payload",
            "Conversation modeling payload",
            [OLEDDiscoveryStage.BLOCKED, OLEDDiscoveryStage.INTENT_CAPTURED],
            inputs=["goal"],
            outputs=["conversation_decision"],
            description="Summarize user intent into a review-only modeling or research handoff.",
        ),
        _tool(
            "research_source_proposal",
            "Research source proposal",
            [OLEDDiscoveryStage.INTENT_CAPTURED],
            inputs=["goal"],
            outputs=["research_source_proposal"],
            failure_modes=["underspecified_goal", "weak_seed_sources"],
        ),
        _tool(
            "research_acquisition_preparation",
            "Research acquisition preparation",
            [OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED],
            inputs=["research_source_proposal"],
            outputs=["research_acquisition_preparation"],
            risk="medium",
            failure_modes=["unconfirmed_external_scope"],
        ),
        _tool(
            "target_modeling_brief",
            "Target modeling brief",
            [OLEDDiscoveryStage.DATASET_READY, OLEDDiscoveryStage.TRAINING_PACKAGE_READY],
            inputs=["dataset_artifacts"],
            outputs=["target_modeling_brief"],
            risk="low",
        ),
        _tool(
            "prepare_literature_corpus_sources",
            "Prepare literature corpus sources",
            [OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED],
            inputs=["research_source_proposal"],
            outputs=["literature_corpus_source_manifest"],
            risk="medium",
        ),
        _tool(
            "acquire_literature_sources",
            "Acquire literature sources",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["research_acquisition_preparation"],
            outputs=["literature_acquisition_manifest"],
            risk="high",
            gates=["gate_2_data_mining"],
            permissions=["external_acquisition_scope"],
            failure_modes=["network_unavailable", "license_restriction", "source_not_found"],
        ),
        _tool(
            "parse_document_mineru",
            "Parse document with MinerU",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["literature_acquisition_manifest"],
            outputs=["parsed_document"],
            risk="high",
            gates=["gate_2_data_mining"],
            failure_modes=["mineru_unavailable", "bad_pdf_parse"],
        ),
        _tool(
            "parse_document_pdfplumber",
            "Parse document with pdfplumber",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["literature_acquisition_manifest"],
            outputs=["parsed_document"],
            risk="medium",
            gates=["gate_2_data_mining"],
            failure_modes=["bad_pdf_parse", "table_extraction_failed"],
        ),
        _tool(
            "index_corpus",
            "Index corpus",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["parsed_document"],
            outputs=["corpus_index"],
            risk="medium",
        ),
        _tool(
            "retrieve_evidence",
            "Retrieve evidence",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["corpus_index"],
            outputs=["evidence_hits"],
            risk="medium",
        ),
        _tool(
            "extract_records",
            "Extract records",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["evidence_hits"],
            outputs=["extracted_records"],
            risk="medium",
            failure_modes=["low_confidence_extraction"],
        ),
        _tool(
            "normalize_extracted_units",
            "Normalize extracted units",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["extracted_records"],
            outputs=["unit_normalization_report"],
            risk="low",
        ),
        _tool(
            "track_citation_provenance",
            "Track citation provenance",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["extracted_records"],
            outputs=["citation_provenance"],
            risk="low",
        ),
        _tool(
            "merge_extracted_records",
            "Merge extracted records",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["extracted_records", "unit_normalization_report"],
            outputs=["merged_records"],
            risk="medium",
            failure_modes=["conflicting_records"],
        ),
        _tool(
            "confirm_extracted_dataset",
            "Confirm extracted dataset",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["merged_records", "citation_provenance"],
            outputs=["dataset_artifacts"],
            risk="medium",
            gates=["gate_2_data_mining"],
        ),
        _tool(
            "curated_gold_view",
            "Curated gold view",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["curated_gold_records"],
            outputs=["curated_gold_view"],
            risk="low",
        ),
        _tool(
            "dataset_view",
            "Dataset view",
            [OLEDDiscoveryStage.ACQUISITION_PREPARED],
            inputs=["curated_gold_view"],
            outputs=["dataset_artifacts"],
            risk="low",
        ),
        _tool(
            "leakage_split",
            "Leakage split",
            [OLEDDiscoveryStage.DATASET_READY],
            inputs=["dataset_artifacts"],
            outputs=["split_rows"],
            risk="low",
        ),
        _tool(
            "feature_materialization",
            "Feature materialization",
            [OLEDDiscoveryStage.DATASET_READY],
            inputs=["split_rows"],
            outputs=["feature_rows"],
            risk="low",
        ),
        _tool(
            "training_package",
            "Training package",
            [OLEDDiscoveryStage.DATASET_READY],
            inputs=["dataset_artifacts"],
            outputs=["training_package_artifacts"],
            risk="low",
        ),
        _tool(
            "baseline_runner",
            "Baseline runner",
            [OLEDDiscoveryStage.TRAINING_PACKAGE_READY],
            inputs=["training_package_artifacts"],
            outputs=["baseline_artifacts"],
            risk="medium",
            gates=["gate_3_train_config"],
            failure_modes=["insufficient_train_rows", "nonnumeric_targets"],
        ),
        _tool(
            "diagnostics_report",
            "Diagnostics report",
            [OLEDDiscoveryStage.BASELINE_READY],
            inputs=["baseline_artifacts"],
            outputs=["diagnostics_report"],
            risk="low",
        ),
        _tool(
            "candidate_generation_or_prediction",
            "Candidate generation or prediction",
            [OLEDDiscoveryStage.DIAGNOSTICS_READY],
            inputs=["diagnostics_report", "training_package_artifacts"],
            outputs=["candidate_artifacts"],
            risk="medium",
            gates=["gate_5_final_threshold"],
            failure_modes=["weak_diagnostics", "out_of_domain_candidates"],
        ),
        _tool(
            "candidate_ranking",
            "Candidate ranking",
            [OLEDDiscoveryStage.DIAGNOSTICS_READY, OLEDDiscoveryStage.CANDIDATES_READY],
            inputs=["candidate_artifacts"],
            outputs=["ranked_candidates"],
            risk="low",
        ),
        _tool(
            "critic_review",
            "Critic review",
            [OLEDDiscoveryStage.CANDIDATES_READY],
            inputs=["candidate_artifacts"],
            outputs=["critic_review"],
            risk="low",
            failure_modes=["insufficient_provenance", "model_overclaim"],
        ),
        _tool(
            "next_action_proposal",
            "Next action proposal",
            [OLEDDiscoveryStage.CRITIC_REVIEWED],
            inputs=["critic_review"],
            outputs=["next_action_proposal"],
            risk="low",
        ),
    ]


def _recommendation_reason(tool: AgentToolSpec, ready: bool, blocked_reasons: list[str]) -> str:
    if ready:
        return f"`{tool.tool_id}` has required inputs for the current discovery stage."
    if blocked_reasons:
        return f"`{tool.tool_id}` is not ready: {', '.join(blocked_reasons)}."
    return f"`{tool.tool_id}` is not ready for the current discovery stage."


def _target_stage_for_tool(tool: AgentToolSpec) -> str:
    if tool.output_artifacts:
        output = tool.output_artifacts[-1]
        mapping = {
            "conversation_decision": OLEDDiscoveryStage.INTENT_CAPTURED.value,
            "research_source_proposal": OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED.value,
            "research_acquisition_preparation": OLEDDiscoveryStage.ACQUISITION_PREPARED.value,
            "dataset_artifacts": OLEDDiscoveryStage.DATASET_READY.value,
            "training_package_artifacts": OLEDDiscoveryStage.TRAINING_PACKAGE_READY.value,
            "baseline_artifacts": OLEDDiscoveryStage.BASELINE_READY.value,
            "diagnostics_report": OLEDDiscoveryStage.DIAGNOSTICS_READY.value,
            "candidate_artifacts": OLEDDiscoveryStage.CANDIDATES_READY.value,
            "critic_review": OLEDDiscoveryStage.CRITIC_REVIEWED.value,
            "next_action_proposal": OLEDDiscoveryStage.NEXT_ACTION_PROPOSED.value,
        }
        if output in mapping:
            return mapping[output]
    return tool.discovery_stages[-1] if tool.discovery_stages else ""


def _normalize_risk_budget(risk_budget: str) -> str:
    normalized = str(risk_budget or "").strip().lower()
    if normalized not in _RISK_RANK:
        raise ValueError("risk_budget must be low, medium, or high")
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
