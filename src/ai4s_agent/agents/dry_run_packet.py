from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.execution_preview import OLEDDiscoveryExecutionPreviewAgent
from ai4s_agent.schemas import OLEDDiscoveryActionHandoff, OLEDDiscoveryDryRunPacket, OLEDDiscoveryExecutionPreview
from ai4s_agent.storage import ProjectStorage


_BASE_REVIEW_CHECKLIST = [
    "confirm selected tool/task mapping",
    "confirm payload placeholders are correct",
    "confirm required artifacts are available",
    "confirm missing inputs are resolved",
    "confirm required gates are understood",
    "confirm no registry/promotion/publication mutation",
    "confirm no adapter execution in this PR",
    "confirm future executor will bind approval to execution snapshot",
]
_SAFETY_BOUNDARY = [
    "Review-only dry-run packet artifact.",
    "Does not execute adapters, instantiate or call RunPlanExecutor, approve gates, or mutate run state.",
    "Does not read or hash artifact paths.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class OLEDDiscoveryDryRunPacketAgent:
    """Review-only dry-run packet builder for OLED execution previews."""

    def build_packet(
        self,
        *,
        preview: OLEDDiscoveryExecutionPreview | dict[str, Any],
        allow_auto_eligible: bool = True,
        allow_gated: bool = True,
    ) -> OLEDDiscoveryDryRunPacket:
        execution_preview = preview if isinstance(preview, OLEDDiscoveryExecutionPreview) else OLEDDiscoveryExecutionPreview.model_validate(preview)
        blocked_reasons = list(execution_preview.blocked_reasons)

        if execution_preview.approval_mode == "blocked" and "preview_blocked" not in blocked_reasons:
            blocked_reasons.append("preview_blocked")
        if not execution_preview.ready_for_controlled_planning and "preview_not_ready_for_controlled_planning" not in blocked_reasons:
            blocked_reasons.append("preview_not_ready_for_controlled_planning")
        if execution_preview.missing_inputs and "missing_required_inputs" not in blocked_reasons:
            blocked_reasons.append("missing_required_inputs")
        if execution_preview.selected_tool_id and not execution_preview.resolved_atomic_task_id and "no_atomic_task_mapping" not in blocked_reasons:
            blocked_reasons.append("no_atomic_task_mapping")
        if execution_preview.required_gates and not allow_gated and "gated_packet_not_allowed" not in blocked_reasons:
            blocked_reasons.append("gated_packet_not_allowed")
        if (
            execution_preview.approval_mode == "auto_eligible"
            and not allow_auto_eligible
            and "auto_eligible_packet_not_allowed" not in blocked_reasons
        ):
            blocked_reasons.append("auto_eligible_packet_not_allowed")

        dry_run_mode = _dry_run_mode(
            preview=execution_preview,
            blocked_reasons=blocked_reasons,
            allow_auto_eligible=allow_auto_eligible,
            allow_gated=allow_gated,
        )
        ready_for_dry_run_review = dry_run_mode != "blocked" and not execution_preview.missing_inputs
        checklist = _review_checklist(
            required_gates=execution_preview.required_gates,
            dry_run_mode=dry_run_mode,
        )

        return OLEDDiscoveryDryRunPacket(
            run_id=execution_preview.run_id,
            project_id=execution_preview.project_id,
            goal=execution_preview.goal,
            source_preview_id=execution_preview.source_handoff_id or f"oled_discovery_execution_preview:{execution_preview.run_id}",
            recommended_next_action=execution_preview.recommended_next_action,
            selected_tool_id=execution_preview.selected_tool_id,
            resolved_atomic_task_id=execution_preview.resolved_atomic_task_id,
            resolved_adapter_name=execution_preview.resolved_adapter_name,
            approval_mode=execution_preview.approval_mode,
            dry_run_mode=dry_run_mode,
            ready_for_dry_run_review=ready_for_dry_run_review,
            executable=False,
            would_execute=False,
            risk_level=execution_preview.risk_level,
            input_artifacts=execution_preview.input_artifacts,
            missing_inputs=execution_preview.missing_inputs,
            output_artifacts=execution_preview.output_artifacts,
            required_gates=execution_preview.required_gates,
            required_permissions=execution_preview.required_permissions,
            blocked_reasons=blocked_reasons,
            execution_preconditions=execution_preview.execution_preconditions,
            payload_template=_json_copy(execution_preview.payload_template),
            dry_run_snapshot_material=_snapshot_material(execution_preview, dry_run_mode),
            review_checklist=checklist,
            policy_notes=execution_preview.policy_notes,
            assumptions=[
                "Dry-run packets are review-only planning artifacts.",
                "would_execute=false and executable=false are enforced for this PR.",
                "Snapshot material contains references and placeholders only; artifact paths are not opened or hashed.",
                "No adapters, RunPlanExecutor calls, gate approval, stage mutation, model backends, LLMs, MinerU, PDF/image readers, or network calls are executed.",
                *execution_preview.assumptions,
            ],
        )

    def write_packet(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        packet: OLEDDiscoveryDryRunPacket,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_dry_run_packet.json", packet.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_dry_run_packet.md"
        md_path.write_text(self.render_markdown(packet), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_dry_run_packet_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_dry_run_packet_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(packet: OLEDDiscoveryDryRunPacket) -> str:
        lines = [
            "# OLED Discovery Dry-Run Packet",
            "",
            f"- Run ID: {packet.run_id}",
            f"- Project ID: {packet.project_id or ''}",
            f"- Goal: {packet.goal}",
            f"- Recommended Next Action: {packet.recommended_next_action}",
            f"- Selected Tool: {packet.selected_tool_id}",
            f"- Atomic Task: {packet.resolved_atomic_task_id}",
            f"- Adapter: {packet.resolved_adapter_name}",
            f"- Approval Mode: {packet.approval_mode}",
            f"- Dry-Run Mode: {packet.dry_run_mode}",
            f"- Ready For Dry-Run Review: {str(packet.ready_for_dry_run_review).lower()}",
            f"- Would Execute: {str(packet.would_execute).lower()}",
            f"- Executable: {str(packet.executable).lower()}",
            "",
            "## Inputs",
            "",
        ]
        lines.extend(_markdown_list(packet.input_artifacts))
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(_markdown_list(packet.missing_inputs))
        lines.extend(["", "## Required Gates", ""])
        lines.extend(_markdown_list(packet.required_gates))
        lines.extend(["", "## Execution Preconditions", ""])
        lines.extend(_markdown_list(packet.execution_preconditions))
        lines.extend(["", "## Dry-Run Snapshot Material", "", "```json"])
        lines.append(json.dumps(packet.dry_run_snapshot_material, sort_keys=True, indent=2))
        lines.extend(["```", "", "## Review Checklist", ""])
        lines.extend(_markdown_list(packet.review_checklist))
        lines.extend(["", "## Policy Notes", ""])
        lines.extend(_markdown_list(packet.policy_notes))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery dry-run packet.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--selected-tool", required=True)
    parser.add_argument("--allow-gated", action="store_true", default=True)
    parser.add_argument("--disallow-gated", action="store_false", dest="allow_gated")
    parser.add_argument("--allow-auto-eligible", action="store_true", default=True)
    parser.add_argument("--disallow-auto-eligible", action="store_false", dest="allow_auto_eligible")
    args = parser.parse_args(argv)

    handoff = OLEDDiscoveryActionHandoff(
        run_id=args.run_id,
        project_id=args.project_id,
        goal=args.goal,
        source_review_id=f"oled_discovery_action_handoff:{args.run_id}",
        recommended_next_action=args.selected_tool,
        critic_decision="continue",
        selected_tool_id=args.selected_tool,
        selected_task_id=args.selected_tool,
        ready=True,
        executable=False,
        payload_template={"run_id": args.run_id, "selected_tool_id": args.selected_tool, "review_only": True},
        assumptions=["Synthetic CLI handoff for review-only dry-run packet."],
    )
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(
        handoff=handoff,
        risk_budget="high",
        allow_auto_eligible=args.allow_auto_eligible,
        allow_gated=args.allow_gated,
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(
        preview=preview,
        allow_auto_eligible=args.allow_auto_eligible,
        allow_gated=args.allow_gated,
    )
    summary = {
        "run_id": packet.run_id,
        "dry_run_mode": packet.dry_run_mode,
        "would_execute": False,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _dry_run_mode(
    *,
    preview: OLEDDiscoveryExecutionPreview,
    blocked_reasons: list[str],
    allow_auto_eligible: bool,
    allow_gated: bool,
) -> str:
    if blocked_reasons or preview.missing_inputs:
        return "blocked"
    if preview.approval_mode == "auto_eligible":
        if allow_auto_eligible and preview.risk_level == "low" and not preview.required_gates:
            return "auto_eligible_preview"
        return "blocked"
    if preview.approval_mode == "gated_review_required":
        if allow_gated and preview.required_gates:
            return "gated_review_packet"
        return "blocked"
    if preview.approval_mode == "manual_review_required":
        return "manual_review_packet"
    return "blocked"


def _snapshot_material(preview: OLEDDiscoveryExecutionPreview, dry_run_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": preview.run_id,
        "source_preview_id": preview.source_handoff_id,
        "selected_tool_id": preview.selected_tool_id,
        "resolved_atomic_task_id": preview.resolved_atomic_task_id,
        "resolved_adapter_name": preview.resolved_adapter_name,
        "approval_mode": preview.approval_mode,
        "dry_run_mode": dry_run_mode,
        "risk_level": preview.risk_level,
        "required_gates": list(preview.required_gates),
        "input_artifacts": list(preview.input_artifacts),
        "missing_inputs": list(preview.missing_inputs),
        "payload_template": _json_copy(preview.payload_template),
    }


def _review_checklist(*, required_gates: list[str], dry_run_mode: str) -> list[str]:
    checklist = list(_BASE_REVIEW_CHECKLIST)
    if required_gates:
        checklist.append("confirm human gate approval before future execution")
    if dry_run_mode == "auto_eligible_preview":
        checklist.append("confirm auto eligibility only applies to future dry-run bridge, not this packet")
    return checklist


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True))


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
