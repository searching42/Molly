from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.dry_run_packet import OLEDDiscoveryDryRunPacketAgent
from ai4s_agent.agents.execution_preview import OLEDDiscoveryExecutionPreviewAgent
from ai4s_agent.schemas import (
    OLEDDiscoveryActionHandoff,
    OLEDDiscoveryDryRunBridgeRequest,
    OLEDDiscoveryDryRunPacket,
)
from ai4s_agent.storage import ProjectStorage


_BASE_SNAPSHOT_BINDING_REQUIREMENTS = [
    "bind bridge request to dry-run packet id",
    "bind payload template to reviewed snapshot material",
    "verify artifact paths before future execution",
    "verify gate approval snapshot before future execution",
    "verify adapter policy has not changed",
    "verify no registry/promotion/publication mutation",
    "verify dry-run bridge uses non-mutating mode",
]
_BASE_REVIEWER_CONFIRMATIONS = [
    "confirm_packet_reviewed",
    "confirm_payload_placeholders",
    "confirm_no_execution_requested",
]
_SAFETY_BOUNDARY = [
    "Review-only dry-run bridge request artifact.",
    "Does not execute adapters, call or instantiate RunPlanExecutor, approve gates, or mutate run state.",
    "Does not read or hash artifact paths.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class OLEDDiscoveryDryRunBridgeRequestAgent:
    """Review-only bridge request builder for OLED dry-run packets."""

    def build_request(
        self,
        *,
        packet: OLEDDiscoveryDryRunPacket | dict[str, Any],
        allow_auto_eligible: bool = True,
        allow_gated: bool = True,
        require_confirmed_reviewer: bool = True,
    ) -> OLEDDiscoveryDryRunBridgeRequest:
        dry_run_packet = packet if isinstance(packet, OLEDDiscoveryDryRunPacket) else OLEDDiscoveryDryRunPacket.model_validate(packet)
        blocked_reasons = list(dry_run_packet.blocked_reasons)

        if dry_run_packet.dry_run_mode == "blocked" and "packet_blocked" not in blocked_reasons:
            blocked_reasons.append("packet_blocked")
        if not dry_run_packet.ready_for_dry_run_review and "packet_not_ready_for_dry_run_review" not in blocked_reasons:
            blocked_reasons.append("packet_not_ready_for_dry_run_review")
        if dry_run_packet.missing_inputs and "missing_required_inputs" not in blocked_reasons:
            blocked_reasons.append("missing_required_inputs")
        if not dry_run_packet.resolved_atomic_task_id or not dry_run_packet.resolved_adapter_name:
            blocked_reasons.append("missing_atomic_task_or_adapter")
        if dry_run_packet.required_gates and not allow_gated:
            blocked_reasons.append("gated_bridge_not_allowed")
        if dry_run_packet.dry_run_mode == "auto_eligible_preview" and not allow_auto_eligible:
            blocked_reasons.append("auto_eligible_bridge_not_allowed")

        reviewer_confirmations = list(_BASE_REVIEWER_CONFIRMATIONS)
        if require_confirmed_reviewer:
            reviewer_confirmations.insert(0, "reviewer_confirmation_required")
            blocked_reasons.append("reviewer_confirmation_required")

        bridge_mode = _bridge_mode(
            packet=dry_run_packet,
            blocked_reasons=blocked_reasons,
            allow_auto_eligible=allow_auto_eligible,
            allow_gated=allow_gated,
        )
        eligible_for_bridge = bridge_mode != "blocked" and not dry_run_packet.missing_inputs and not blocked_reasons

        return OLEDDiscoveryDryRunBridgeRequest(
            run_id=dry_run_packet.run_id,
            project_id=dry_run_packet.project_id,
            goal=dry_run_packet.goal,
            source_packet_id=f"oled_discovery_dry_run_packet:{dry_run_packet.run_id}",
            selected_tool_id=dry_run_packet.selected_tool_id,
            resolved_atomic_task_id=dry_run_packet.resolved_atomic_task_id,
            resolved_adapter_name=dry_run_packet.resolved_adapter_name,
            bridge_mode=bridge_mode,
            dry_run_mode=dry_run_packet.dry_run_mode,
            approval_mode=dry_run_packet.approval_mode,
            eligible_for_bridge=eligible_for_bridge,
            executable=False,
            would_execute=False,
            adapter_invocation=_adapter_invocation(dry_run_packet),
            payload_template=_json_copy(dry_run_packet.payload_template),
            required_gates=dry_run_packet.required_gates,
            required_permissions=dry_run_packet.required_permissions,
            missing_inputs=dry_run_packet.missing_inputs,
            blocked_reasons=blocked_reasons,
            snapshot_binding_requirements=_snapshot_binding_requirements(dry_run_packet),
            reviewer_confirmations=reviewer_confirmations,
            dry_run_snapshot_material=_json_copy(dry_run_packet.dry_run_snapshot_material),
            audit_notes=_audit_notes(
                bridge_mode=bridge_mode,
                require_confirmed_reviewer=require_confirmed_reviewer,
            ),
            assumptions=[
                "Dry-run bridge requests are review-only planning artifacts.",
                "eligible_for_bridge=true never means execution in this PR.",
                "would_execute=false and executable=false are enforced.",
                "Adapter invocation is a placeholder payload only.",
                "No adapters, RunPlanExecutor calls, gate approval, stage mutation, artifact reads/hashes, model backends, LLMs, MinerU, PDF/image readers, or network calls are executed.",
                *dry_run_packet.assumptions,
            ],
        )

    def write_request(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        request: OLEDDiscoveryDryRunBridgeRequest,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_dry_run_bridge_request.json", request.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_dry_run_bridge_request.md"
        md_path.write_text(self.render_markdown(request), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_dry_run_bridge_request_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_dry_run_bridge_request_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(request: OLEDDiscoveryDryRunBridgeRequest) -> str:
        lines = [
            "# OLED Discovery Dry-Run Bridge Request",
            "",
            f"- Run ID: {request.run_id}",
            f"- Project ID: {request.project_id or ''}",
            f"- Goal: {request.goal}",
            f"- Selected Tool: {request.selected_tool_id}",
            f"- Atomic Task: {request.resolved_atomic_task_id}",
            f"- Adapter: {request.resolved_adapter_name}",
            f"- Dry-Run Mode: {request.dry_run_mode}",
            f"- Bridge Mode: {request.bridge_mode}",
            f"- Eligible For Bridge: {str(request.eligible_for_bridge).lower()}",
            f"- Would Execute: {str(request.would_execute).lower()}",
            f"- Executable: {str(request.executable).lower()}",
            "",
            "## Adapter Invocation",
            "",
            "```json",
            json.dumps(request.adapter_invocation, sort_keys=True, indent=2),
            "```",
            "",
            "## Required Gates",
            "",
        ]
        lines.extend(_markdown_list(request.required_gates))
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(_markdown_list(request.missing_inputs))
        lines.extend(["", "## Blocked Reasons", ""])
        lines.extend(_markdown_list(request.blocked_reasons))
        lines.extend(["", "## Snapshot Binding Requirements", ""])
        lines.extend(_markdown_list(request.snapshot_binding_requirements))
        lines.extend(["", "## Reviewer Confirmations", ""])
        lines.extend(_markdown_list(request.reviewer_confirmations))
        lines.extend(["", "## Audit Notes", ""])
        lines.extend(_markdown_list(request.audit_notes))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery dry-run bridge request.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--selected-tool", required=True)
    parser.add_argument("--allow-gated", action="store_true", default=True)
    parser.add_argument("--disallow-gated", action="store_false", dest="allow_gated")
    parser.add_argument("--allow-auto-eligible", action="store_true", default=True)
    parser.add_argument("--disallow-auto-eligible", action="store_false", dest="allow_auto_eligible")
    parser.add_argument("--require-confirmed-reviewer", action="store_true", default=True)
    parser.add_argument("--no-require-confirmed-reviewer", action="store_false", dest="require_confirmed_reviewer")
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
        assumptions=["Synthetic CLI handoff for review-only dry-run bridge request."],
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
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        allow_auto_eligible=args.allow_auto_eligible,
        allow_gated=args.allow_gated,
        require_confirmed_reviewer=args.require_confirmed_reviewer,
    )
    summary = {
        "run_id": request.run_id,
        "bridge_mode": request.bridge_mode,
        "eligible_for_bridge": request.eligible_for_bridge,
        "would_execute": False,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _bridge_mode(
    *,
    packet: OLEDDiscoveryDryRunPacket,
    blocked_reasons: list[str],
    allow_auto_eligible: bool,
    allow_gated: bool,
) -> str:
    if blocked_reasons or packet.missing_inputs:
        return "blocked"
    if packet.dry_run_mode == "auto_eligible_preview":
        return "auto_eligible_bridge_request" if allow_auto_eligible and not packet.required_gates else "blocked"
    if packet.dry_run_mode == "gated_review_packet":
        return "gated_bridge_request" if allow_gated and packet.required_gates else "blocked"
    if packet.dry_run_mode == "manual_review_packet":
        return "manual_bridge_request"
    return "blocked"


def _adapter_invocation(packet: OLEDDiscoveryDryRunPacket) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": packet.run_id,
        "task_id": packet.resolved_atomic_task_id,
        "adapter": packet.resolved_adapter_name,
        "payload_template": _json_copy(packet.payload_template),
        "dry_run": True,
        "review_only": True,
        "would_execute": False,
    }


def _snapshot_binding_requirements(packet: OLEDDiscoveryDryRunPacket) -> list[str]:
    requirements = list(_BASE_SNAPSHOT_BINDING_REQUIREMENTS)
    if packet.required_gates:
        requirements.append("verify human gate approval before future bridge execution")
    if packet.dry_run_mode == "auto_eligible_preview":
        requirements.append("verify auto eligibility still holds at bridge time")
    return requirements


def _audit_notes(*, bridge_mode: str, require_confirmed_reviewer: bool) -> list[str]:
    notes = [
        f"bridge_mode:{bridge_mode}",
        "placeholder_adapter_invocation_only",
        "no_execution_requested",
    ]
    if require_confirmed_reviewer:
        notes.append("reviewer_confirmation_required_by_default")
    return notes


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True))


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
