from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.action_handoff import OLEDDiscoveryActionHandoffAgent
from ai4s_agent.agents.dry_run_bridge import OLEDDiscoveryDryRunBridgeRequestAgent
from ai4s_agent.agents.dry_run_packet import OLEDDiscoveryDryRunPacketAgent
from ai4s_agent.agents.execution_preview import OLEDDiscoveryExecutionPreviewAgent
from ai4s_agent.agents.oled_review_loop import OLEDDiscoveryReviewLoopAgent
from ai4s_agent.storage import ProjectStorage


_SUPPORTED_SCENARIOS = {
    "acceptable_diagnostics",
    "weak_diagnostics",
    "missing_provenance",
    "candidate_review_needed",
}
_SAFETY_BOUNDARY = [
    "Review-only OLED Agent MVP demo artifact.",
    "Does not execute adapters, call or instantiate RunPlanExecutor, approve gates, or mutate run state.",
    "Does not read or hash artifact paths.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class OLEDAgentMVPDemoRunner:
    """Deterministic review-only OLED Agent MVP demo runner."""

    def __init__(
        self,
        *,
        review_loop_agent: OLEDDiscoveryReviewLoopAgent | None = None,
        handoff_agent: OLEDDiscoveryActionHandoffAgent | None = None,
        execution_preview_agent: OLEDDiscoveryExecutionPreviewAgent | None = None,
        dry_run_packet_agent: OLEDDiscoveryDryRunPacketAgent | None = None,
        bridge_request_agent: OLEDDiscoveryDryRunBridgeRequestAgent | None = None,
    ) -> None:
        self.review_loop_agent = review_loop_agent or OLEDDiscoveryReviewLoopAgent()
        self.handoff_agent = handoff_agent or OLEDDiscoveryActionHandoffAgent()
        self.execution_preview_agent = execution_preview_agent or OLEDDiscoveryExecutionPreviewAgent()
        self.dry_run_packet_agent = dry_run_packet_agent or OLEDDiscoveryDryRunPacketAgent()
        self.bridge_request_agent = bridge_request_agent or OLEDDiscoveryDryRunBridgeRequestAgent()

    def run_demo(
        self,
        *,
        run_id: str,
        goal: str,
        project_id: str | None = None,
        scenario: str = "acceptable_diagnostics",
    ) -> dict[str, Any]:
        scenario_payload = _scenario_payload(scenario)
        review = self.review_loop_agent.build_review(
            run_id=run_id,
            goal=goal,
            project_id=project_id,
            **scenario_payload,
        )
        handoff = self.handoff_agent.build_handoff(loop_review=review)
        execution_preview = self.execution_preview_agent.build_preview(
            handoff=handoff,
            risk_budget="high",
        )
        dry_run_packet = self.dry_run_packet_agent.build_packet(preview=execution_preview)
        bridge_request = self.bridge_request_agent.build_request(
            packet=dry_run_packet,
            require_confirmed_reviewer=False,
        )

        return {
            "run_id": run_id,
            "project_id": project_id,
            "goal": goal,
            "scenario": scenario,
            "current_stage": review.run_card.current_stage,
            "critic_decision": review.critic_review.decision.decision,
            "recommended_next_action": review.recommended_next_action,
            "selected_tool_id": handoff.selected_tool_id,
            "resolved_atomic_task_id": execution_preview.resolved_atomic_task_id,
            "approval_mode": execution_preview.approval_mode,
            "dry_run_mode": dry_run_packet.dry_run_mode,
            "bridge_mode": bridge_request.bridge_mode,
            "eligible_for_bridge": bridge_request.eligible_for_bridge,
            "risk_flags": _clean_unique([*review.risk_flags, *handoff.risk_flags]),
            "blocked_reasons": _clean_unique(
                [
                    *review.blocked_reasons,
                    *handoff.blocked_reasons,
                    *execution_preview.blocked_reasons,
                    *dry_run_packet.blocked_reasons,
                    *bridge_request.blocked_reasons,
                ]
            ),
            "executable": False,
            "pipeline_summary": {
                "review_loop": f"{review.run_card.current_stage}:{review.recommended_next_action}",
                "action_handoff": handoff.selected_tool_id or "no_selected_tool",
                "execution_preview": f"{execution_preview.resolved_atomic_task_id or 'unmapped'}:{execution_preview.approval_mode}",
                "dry_run_packet": dry_run_packet.dry_run_mode,
                "bridge_request": bridge_request.bridge_mode,
            },
        }

    def write_demo_report(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        result: dict[str, Any],
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_agent_mvp_demo.json", result)
        md_path = run_dir / "oled_agent_mvp_demo.md"
        md_path.write_text(self.render_markdown(result), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(result: dict[str, Any]) -> str:
        pipeline_summary = result.get("pipeline_summary") if isinstance(result.get("pipeline_summary"), dict) else {}
        lines = [
            "# OLED Agent MVP Demo",
            "",
            f"- Run ID: {result.get('run_id', '')}",
            f"- Project ID: {result.get('project_id') or ''}",
            f"- Scenario: {result.get('scenario', '')}",
            f"- Goal: {result.get('goal', '')}",
            f"- Current Stage: {result.get('current_stage', '')}",
            f"- Critic Decision: {result.get('critic_decision', '')}",
            f"- Recommended Next Action: {result.get('recommended_next_action', '')}",
            f"- Selected Tool: {result.get('selected_tool_id', '')}",
            f"- Atomic Task: {result.get('resolved_atomic_task_id', '')}",
            f"- Approval Mode: {result.get('approval_mode', '')}",
            f"- Dry-Run Mode: {result.get('dry_run_mode', '')}",
            f"- Bridge Mode: {result.get('bridge_mode', '')}",
            f"- Executable: {str(result.get('executable', False)).lower()}",
            "",
            "## Pipeline Summary",
            "",
            "| Component | Key Result |",
            "| --- | --- |",
        ]
        for component in ["review_loop", "action_handoff", "execution_preview", "dry_run_packet", "bridge_request"]:
            lines.append(f"| {component} | {pipeline_summary.get(component, '')} |")
        lines.extend(["", "## Risk Flags", ""])
        lines.extend(_markdown_list(list(result.get("risk_flags") or [])))
        lines.extend(["", "## Blocked Reasons", ""])
        lines.extend(_markdown_list(list(result.get("blocked_reasons") or [])))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a review-only OLED Agent MVP demo.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--project-id")
    parser.add_argument("--scenario", default="acceptable_diagnostics", choices=sorted(_SUPPORTED_SCENARIOS))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    runner = OLEDAgentMVPDemoRunner()
    result = runner.run_demo(
        run_id=args.run_id,
        goal=args.goal,
        project_id=args.project_id,
        scenario=args.scenario,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "oled_agent_mvp_demo.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "oled_agent_mvp_demo.md").write_text(runner.render_markdown(result), encoding="utf-8")

    summary = {key: result[key] for key in sorted(REQUIRED_COMPACT_KEYS) if key in result}
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


REQUIRED_COMPACT_KEYS = {
    "approval_mode",
    "bridge_mode",
    "critic_decision",
    "current_stage",
    "dry_run_mode",
    "eligible_for_bridge",
    "executable",
    "goal",
    "project_id",
    "recommended_next_action",
    "resolved_atomic_task_id",
    "risk_flags",
    "run_id",
    "scenario",
    "selected_tool_id",
}


def _scenario_payload(scenario: str) -> dict[str, Any]:
    clean = str(scenario or "").strip()
    if clean not in _SUPPORTED_SCENARIOS:
        raise ValueError(f"unsupported OLED Agent MVP demo scenario: {scenario}")

    base = {
        "dataset_artifacts": {"dataset_view_rows": "synthetic_dataset_view_rows"},
        "training_package_artifacts": {"training_rows": "synthetic_training_rows"},
        "baseline_artifacts": {"metrics": "synthetic_metrics"},
        "provenance_summary": {"source_count": 2, "evidence_count": 8},
    }
    if clean == "acceptable_diagnostics":
        return {**base, "diagnostics_report": {"status": "acceptable"}}
    if clean == "weak_diagnostics":
        return {**base, "diagnostics_report": {"status": "weak", "summary": "rerun recommended"}}
    if clean == "missing_provenance":
        return {
            **base,
            "diagnostics_report": {"status": "acceptable"},
            "provenance_summary": {"source_count": 0, "evidence_count": 0},
        }
    if clean == "candidate_review_needed":
        return {
            **base,
            "diagnostics_report": {"status": "acceptable"},
            "candidate_artifacts": {"candidate_rows": "synthetic_candidates"},
            "candidate_summary": {"candidate_count": 4},
        }
    raise AssertionError("unreachable")


def _clean_unique(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
