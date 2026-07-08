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


_DEFAULT_SCENARIOS = [
    "acceptable_diagnostics",
    "weak_diagnostics",
    "missing_provenance",
    "candidate_review_needed",
]
_SUPPORTED_SCENARIOS = set(_DEFAULT_SCENARIOS)
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
        return self._run_payload_demo(
            run_id=run_id,
            goal=goal,
            project_id=project_id,
            scenario=scenario,
            payload=_scenario_payload(scenario),
        )

    def _run_payload_demo(
        self,
        *,
        run_id: str,
        goal: str,
        project_id: str | None,
        scenario: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        review = self.review_loop_agent.build_review(
            run_id=run_id,
            goal=goal,
            project_id=project_id,
            **payload,
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

    def run_scenario_matrix(
        self,
        *,
        run_id: str,
        goal: str,
        project_id: str | None = None,
        scenarios: list[str] | None = None,
    ) -> dict[str, Any]:
        scenario_names = list(scenarios or _DEFAULT_SCENARIOS)
        rows = [
            _matrix_row(
                self.run_demo(
                    run_id=f"{run_id}:{scenario}",
                    goal=goal,
                    project_id=project_id,
                    scenario=scenario,
                )
            )
            for scenario in scenario_names
        ]
        return {
            "run_id": run_id,
            "project_id": project_id,
            "goal": goal,
            "scenario_count": len(rows),
            "scenarios": rows,
            "summary": _matrix_summary(rows),
            "executable": False,
        }

    def write_scenario_matrix_report(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        matrix: dict[str, Any],
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_agent_mvp_demo_matrix.json", matrix)
        md_path = run_dir / "oled_agent_mvp_demo_matrix.md"
        md_path.write_text(self.render_matrix_markdown(matrix), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_matrix_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_matrix_md", md_path.name)
        return json_path, md_path

    def run_local_bundle(
        self,
        *,
        run_id: str,
        bundle_path: Path | str,
        goal: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        bundle = load_local_input_bundle(bundle_path)
        effective_goal = str(goal if goal is not None else bundle.get("goal") or "")
        effective_project_id = project_id if project_id is not None else bundle.get("project_id")
        scenario_rows = []
        for scenario in bundle["scenarios"]:
            name = str(scenario["name"])
            scenario_rows.append(
                _matrix_row(
                    self._run_payload_demo(
                        run_id=f"{run_id}:{name}",
                        goal=effective_goal,
                        project_id=effective_project_id,
                        scenario=name,
                        payload=dict(scenario["payload"]),
                    )
                )
            )
        return {
            "run_id": run_id,
            "project_id": effective_project_id,
            "goal": effective_goal,
            "source": "local_input_bundle",
            "bundle_path": _sanitize_bundle_path(bundle_path),
            "scenario_count": len(scenario_rows),
            "scenarios": scenario_rows,
            "summary": _matrix_summary(scenario_rows),
            "executable": False,
        }

    def write_local_bundle_report(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        result: dict[str, Any],
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_agent_mvp_demo_bundle.json", result)
        md_path = run_dir / "oled_agent_mvp_demo_bundle.md"
        md_path.write_text(self.render_local_bundle_markdown(result), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_bundle_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_agent_mvp_demo_bundle_md", md_path.name)
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

    @staticmethod
    def render_matrix_markdown(matrix: dict[str, Any]) -> str:
        summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
        lines = [
            "# OLED Agent MVP Demo Matrix",
            "",
            f"- Run ID: {matrix.get('run_id', '')}",
            f"- Project ID: {matrix.get('project_id') or ''}",
            f"- Goal: {matrix.get('goal', '')}",
            f"- Scenario Count: {matrix.get('scenario_count', 0)}",
            f"- Executable: {str(matrix.get('executable', False)).lower()}",
            "",
            "## Scenario Matrix",
            "",
            "| Scenario | Stage | Critic Decision | Next Action | Tool | Atomic Task | Approval | Dry-Run | Bridge | Risk Flags | Blockers |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in list(matrix.get("scenarios") or []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("scenario", "")),
                        str(row.get("current_stage", "")),
                        str(row.get("critic_decision", "")),
                        str(row.get("recommended_next_action", "")),
                        str(row.get("selected_tool_id", "")),
                        str(row.get("resolved_atomic_task_id", "")),
                        str(row.get("approval_mode", "")),
                        str(row.get("dry_run_mode", "")),
                        str(row.get("bridge_mode", "")),
                        ", ".join(row.get("risk_flags") or []),
                        ", ".join(row.get("blocked_reasons") or []),
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Decision Counts", ""])
        lines.extend(_markdown_count_list(dict(summary.get("critic_decision_counts") or {})))
        lines.extend(["", "## Bridge Mode Counts", ""])
        lines.extend(_markdown_count_list(dict(summary.get("bridge_mode_counts") or {})))
        lines.extend(["", "## Scenarios With Blockers", ""])
        lines.extend(_markdown_list(list(summary.get("scenarios_with_blockers") or [])))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def render_local_bundle_markdown(result: dict[str, Any]) -> str:
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        lines = [
            "# OLED Agent MVP Demo Local Bundle",
            "",
            f"- Run ID: {result.get('run_id', '')}",
            f"- Project ID: {result.get('project_id') or ''}",
            f"- Goal: {result.get('goal', '')}",
            f"- Source: {result.get('source', '')}",
            f"- Scenario Count: {result.get('scenario_count', 0)}",
            f"- Executable: {str(result.get('executable', False)).lower()}",
            "",
            "## Scenario Matrix",
            "",
            "| Scenario | Stage | Critic Decision | Next Action | Tool | Atomic Task | Approval | Dry-Run | Bridge | Risk Flags | Blockers |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in list(result.get("scenarios") or []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("scenario", "")),
                        str(row.get("current_stage", "")),
                        str(row.get("critic_decision", "")),
                        str(row.get("recommended_next_action", "")),
                        str(row.get("selected_tool_id", "")),
                        str(row.get("resolved_atomic_task_id", "")),
                        str(row.get("approval_mode", "")),
                        str(row.get("dry_run_mode", "")),
                        str(row.get("bridge_mode", "")),
                        ", ".join(row.get("risk_flags") or []),
                        ", ".join(row.get("blocked_reasons") or []),
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Decision Counts", ""])
        lines.extend(_markdown_count_list(dict(summary.get("critic_decision_counts") or {})))
        lines.extend(["", "## Bridge Mode Counts", ""])
        lines.extend(_markdown_count_list(dict(summary.get("bridge_mode_counts") or {})))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a review-only OLED Agent MVP demo.")
    parser.add_argument("--run-id")
    parser.add_argument("--goal")
    parser.add_argument("--project-id")
    parser.add_argument("--scenario", default="acceptable_diagnostics", choices=sorted(_SUPPORTED_SCENARIOS))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--input-bundle")
    parser.add_argument("--print-input-bundle-template", action="store_true")
    parser.add_argument("--write-input-bundle-template")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.print_input_bundle_template:
        print(json.dumps(local_input_bundle_template(), sort_keys=True, indent=2))
        return 0
    if args.write_input_bundle_template:
        template_path = write_local_input_bundle_template(args.write_input_bundle_template)
        summary = {
            "template_path": template_path.name,
            "scenario_count": len(local_input_bundle_template()["scenarios"]),
            "executable": False,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.run_id:
        parser.error("--run-id is required unless printing or writing an input bundle template")

    runner = OLEDAgentMVPDemoRunner()
    if args.input_bundle:
        result = runner.run_local_bundle(
            run_id=args.run_id,
            bundle_path=args.input_bundle,
            goal=args.goal,
            project_id=args.project_id,
        )
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "oled_agent_mvp_demo_bundle.json").write_text(
                json.dumps(result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            (output_dir / "oled_agent_mvp_demo_bundle.md").write_text(
                runner.render_local_bundle_markdown(result),
                encoding="utf-8",
            )
        summary = {
            "run_id": result["run_id"],
            "source": result["source"],
            "scenario_count": result["scenario_count"],
            "critic_decision_counts": result["summary"]["critic_decision_counts"],
            "executable": False,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.goal:
        parser.error("--goal is required unless --input-bundle is supplied")

    if args.all_scenarios:
        matrix = runner.run_scenario_matrix(
            run_id=args.run_id,
            goal=args.goal,
            project_id=args.project_id,
        )
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "oled_agent_mvp_demo_matrix.json").write_text(
                json.dumps(matrix, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            (output_dir / "oled_agent_mvp_demo_matrix.md").write_text(
                runner.render_matrix_markdown(matrix),
                encoding="utf-8",
            )
        summary = {
            "run_id": matrix["run_id"],
            "scenario_count": matrix["scenario_count"],
            "critic_decision_counts": matrix["summary"]["critic_decision_counts"],
            "executable": False,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0

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


def local_input_bundle_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "project_id": "demo-project",
        "notes": [
            "Summary-only bundle for OLEDAgentMVPDemoRunner.",
            "Artifact values are labels/placeholders; they are not opened or read.",
        ],
        "scenarios": [
            {
                "name": "local_acceptable",
                "description": "Acceptable diagnostics with provenance present.",
                "payload": {
                    "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
                    "training_package_artifacts": {"training_rows": "local_training_rows"},
                    "baseline_artifacts": {"metrics": "local_metrics"},
                    "diagnostics_report": {"status": "acceptable"},
                    "provenance_summary": {"source_count": 2, "evidence_count": 8},
                    "dataset_summary": {"row_count": 86, "property_count": 3},
                },
            },
            {
                "name": "local_weak_diagnostics",
                "description": "Weak diagnostics example that should trigger baseline rerun.",
                "payload": {
                    "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
                    "training_package_artifacts": {"training_rows": "local_training_rows"},
                    "baseline_artifacts": {"metrics": "local_metrics"},
                    "diagnostics_report": {"status": "weak", "summary": "rerun recommended"},
                    "provenance_summary": {"source_count": 2, "evidence_count": 8},
                },
            },
            {
                "name": "local_missing_provenance",
                "description": "Acceptable diagnostics but empty provenance.",
                "payload": {
                    "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
                    "training_package_artifacts": {"training_rows": "local_training_rows"},
                    "baseline_artifacts": {"metrics": "local_metrics"},
                    "diagnostics_report": {"status": "acceptable"},
                    "provenance_summary": {"source_count": 0, "evidence_count": 0},
                },
            },
        ],
    }


def write_local_input_bundle_template(path: Path | str) -> Path:
    template_path = Path(path)
    template_path.write_text(json.dumps(local_input_bundle_template(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return template_path


def load_local_input_bundle(path: Path | str) -> dict[str, Any]:
    bundle_path = Path(path)
    try:
        with open(bundle_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"missing_local_input_bundle:{bundle_path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_local_input_bundle_json:{bundle_path.name}") from exc

    if not isinstance(payload, dict):
        raise ValueError("invalid_local_input_bundle:top_level_object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported_local_input_bundle_schema_version")
    if "scenarios" not in payload:
        raise ValueError("missing_local_input_bundle_scenarios")
    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list):
        raise ValueError("invalid_local_input_bundle_scenarios")
    if not scenarios:
        raise ValueError("empty_local_input_bundle_scenarios")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"invalid_local_input_bundle_scenario:{index}")
        name = str(scenario.get("name") or "").strip()
        if not name:
            raise ValueError(f"missing_local_input_bundle_scenario_name:{index}")
        if not isinstance(scenario.get("payload"), dict):
            raise ValueError(f"missing_local_input_bundle_scenario_payload:{name}")
    return payload


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


def _matrix_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": result["scenario"],
        "current_stage": result["current_stage"],
        "critic_decision": result["critic_decision"],
        "recommended_next_action": result["recommended_next_action"],
        "selected_tool_id": result["selected_tool_id"],
        "resolved_atomic_task_id": result["resolved_atomic_task_id"],
        "approval_mode": result["approval_mode"],
        "dry_run_mode": result["dry_run_mode"],
        "bridge_mode": result["bridge_mode"],
        "eligible_for_bridge": result["eligible_for_bridge"],
        "risk_flags": list(result["risk_flags"]),
        "blocked_reasons": list(result["blocked_reasons"]),
        "executable": False,
    }


def _matrix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "critic_decision_counts": _count_values(row["critic_decision"] for row in rows),
        "bridge_mode_counts": _count_values(row["bridge_mode"] for row in rows),
        "scenarios_with_blockers": [row["scenario"] for row in rows if row["blocked_reasons"]],
    }


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


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


def _markdown_count_list(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- None"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]


def _sanitize_bundle_path(path: Path | str) -> str:
    return Path(path).name


if __name__ == "__main__":
    raise SystemExit(main())
