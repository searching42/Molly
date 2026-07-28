from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.execution_policy import ExecutionPolicyRegistry
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import OLEDDiscoveryActionHandoff, OLEDDiscoveryExecutionPreview
from ai4s_agent.storage import ProjectStorage


TOOL_TO_ATOMIC_TASK: dict[str, str] = {
    "research_acquisition_preparation": "prepare_literature_corpus_sources",
    "acquire_literature_sources": "acquire_literature_sources",
    "parse_document_mineru": "parse_document",
    "parse_document_pdfplumber": "parse_document_pdfplumber",
    "baseline_runner": "run_baseline",
    "candidate_generation_or_prediction": "generate_candidates",
    "candidate_ranking": "filter_rank",
    "critic_review": "",
    "research_source_proposal": "",
    "target_modeling_brief": "",
    "leakage_split": "",
    "training_package": "",
    "retrieve_evidence": "retrieve_evidence",
}

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_BASE_EXECUTION_PRECONDITIONS = [
    "verify artifact paths exist",
    "verify payload still matches handoff",
    "verify gate approvals bind to snapshot",
    "verify no registry/promotion/publication mutation",
    "verify executor dry-run mode if used later",
]
_SAFETY_BOUNDARY = [
    "Review-only execution preview artifact.",
    "Does not execute adapters, call RunPlanExecutor, approve gates, or mutate stage state.",
    "Does not train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, or use external network access.",
    "Does not mutate registry, promotion, publication, release, or global append artifacts.",
]


class OLEDDiscoveryExecutionPreviewAgent:
    """Review-only execution preview for OLED discovery action handoffs."""

    def __init__(
        self,
        *,
        task_registry: AtomicTaskRegistry | None = None,
        policy_registry: ExecutionPolicyRegistry | None = None,
    ) -> None:
        self.task_registry = task_registry or AtomicTaskRegistry()
        self.policy_registry = policy_registry or ExecutionPolicyRegistry(self.task_registry)

    def build_preview(
        self,
        *,
        handoff: OLEDDiscoveryActionHandoff | dict[str, Any],
        risk_budget: str = "medium",
        allow_auto_eligible: bool = True,
        allow_gated: bool = True,
    ) -> OLEDDiscoveryExecutionPreview:
        action_handoff = handoff if isinstance(handoff, OLEDDiscoveryActionHandoff) else OLEDDiscoveryActionHandoff.model_validate(handoff)
        normalized_budget = _normalize_risk(risk_budget, field_name="risk_budget")
        task_id = self._resolve_atomic_task_id(action_handoff)
        task = self._task_or_none(task_id)
        adapter_name = str(task.default_adapter or "") if task else ""
        task_risk = _normalize_risk(str(task.risk_level.value if hasattr(task.risk_level, "value") else task.risk_level), field_name="risk_level") if task else "low"

        policy = self.policy_registry.adapter_policy(adapter_name, action_handoff.payload_template) if adapter_name else None
        policy_gates = list(policy.required_gates) if policy else []
        task_gates = list(task.gates) if task else []
        required_gates = _clean_unique([*action_handoff.required_gates, *task_gates, *policy_gates])
        input_artifacts = _clean_unique([*action_handoff.input_artifacts, *(task.required_artifacts if task else [])])
        output_artifacts = _clean_unique([*action_handoff.output_artifacts, *(task.output_artifacts if task else [])])
        missing_inputs = list(action_handoff.missing_inputs)
        blocked_reasons = list(action_handoff.blocked_reasons)
        policy_notes = self._policy_notes(task_id=task_id, adapter_name=adapter_name, policy=policy)
        execution_preconditions = list(_BASE_EXECUTION_PRECONDITIONS)

        if required_gates:
            execution_preconditions.append("human gate approval required before execution")
        if not action_handoff.ready and "handoff_not_ready" not in blocked_reasons:
            blocked_reasons.append("handoff_not_ready")
        if not action_handoff.selected_tool_id and not action_handoff.selected_task_id:
            blocked_reasons.append("no_selected_tool")
        if action_handoff.selected_tool_id and not task:
            blocked_reasons.append("no_atomic_task_mapping")
            execution_preconditions.append("manual planner mapping required")
        if _RISK_ORDER[task_risk] > _RISK_ORDER[normalized_budget]:
            blocked_reasons.append("risk_level_exceeds_budget")
        if required_gates and not allow_gated:
            blocked_reasons.append("gated_tool_not_allowed")

        approval_mode = _approval_mode(
            handoff=action_handoff,
            task_resolved=task is not None,
            risk_level=task_risk,
            required_gates=required_gates,
            blocked_reasons=blocked_reasons,
            allow_auto_eligible=allow_auto_eligible,
            allow_gated=allow_gated,
        )
        ready_for_controlled_planning = (
            approval_mode != "blocked"
            and not missing_inputs
            and task is not None
            and "risk_level_exceeds_budget" not in blocked_reasons
            and "gated_tool_not_allowed" not in blocked_reasons
        )

        return OLEDDiscoveryExecutionPreview(
            run_id=action_handoff.run_id,
            project_id=action_handoff.project_id,
            goal=action_handoff.goal,
            source_handoff_id=action_handoff.source_review_id or f"oled_discovery_action_handoff:{action_handoff.run_id}",
            recommended_next_action=action_handoff.recommended_next_action,
            selected_tool_id=action_handoff.selected_tool_id,
            selected_task_id=action_handoff.selected_task_id,
            resolved_atomic_task_id=task.task_id if task else "",
            resolved_adapter_name=adapter_name,
            risk_level=task_risk,
            approval_mode=approval_mode,
            ready_for_controlled_planning=ready_for_controlled_planning,
            executable=False,
            input_artifacts=input_artifacts,
            missing_inputs=missing_inputs,
            output_artifacts=output_artifacts,
            required_gates=required_gates,
            required_permissions=action_handoff.required_permissions,
            blocked_reasons=blocked_reasons,
            execution_preconditions=execution_preconditions,
            payload_template=_json_copy(action_handoff.payload_template),
            policy_notes=policy_notes,
            assumptions=[
                "Execution previews are review-only planning artifacts.",
                "Auto-eligible means a future controlled dry-run may consider auto approval; it is not execution.",
                "Adapter and atomic task metadata are read only.",
                "No adapters, RunPlanExecutor calls, gate approval, stage mutation, model backends, LLMs, MinerU, PDF/image readers, or network calls are executed.",
                *action_handoff.assumptions,
            ],
        )

    def write_preview(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        preview: OLEDDiscoveryExecutionPreview,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_execution_preview.json", preview.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_execution_preview.md"
        md_path.write_text(self.render_markdown(preview), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_execution_preview_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_execution_preview_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(preview: OLEDDiscoveryExecutionPreview) -> str:
        lines = [
            "# OLED Discovery Execution Preview",
            "",
            f"- Run ID: {preview.run_id}",
            f"- Project ID: {preview.project_id or ''}",
            f"- Goal: {preview.goal}",
            f"- Recommended Next Action: {preview.recommended_next_action}",
            f"- Selected Tool: {preview.selected_tool_id}",
            f"- Selected Task: {preview.selected_task_id}",
            f"- Resolved Atomic Task: {preview.resolved_atomic_task_id}",
            f"- Adapter: {preview.resolved_adapter_name}",
            f"- Risk Level: {preview.risk_level}",
            f"- Approval Mode: {preview.approval_mode}",
            f"- Ready For Controlled Planning: {str(preview.ready_for_controlled_planning).lower()}",
            f"- Executable: {str(preview.executable).lower()}",
            "",
            "## Inputs",
            "",
        ]
        lines.extend(_markdown_list(preview.input_artifacts))
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(_markdown_list(preview.missing_inputs))
        lines.extend(["", "## Required Gates", ""])
        lines.extend(_markdown_list(preview.required_gates))
        lines.extend(["", "## Required Permissions", ""])
        lines.extend(_markdown_list(preview.required_permissions))
        lines.extend(["", "## Execution Preconditions", ""])
        lines.extend(_markdown_list(preview.execution_preconditions))
        lines.extend(["", "## Payload Template", "", "```json"])
        lines.append(json.dumps(preview.payload_template, sort_keys=True, indent=2))
        lines.extend(["```", "", "## Policy Notes", ""])
        lines.extend(_markdown_list(preview.policy_notes))
        lines.extend(["", "## Safety Boundary", ""])
        lines.extend(f"- {item}" for item in _SAFETY_BOUNDARY)
        lines.append("")
        return "\n".join(lines)

    def _resolve_atomic_task_id(self, handoff: OLEDDiscoveryActionHandoff) -> str:
        candidates = [
            str(handoff.selected_task_id or "").strip(),
            TOOL_TO_ATOMIC_TASK.get(str(handoff.selected_tool_id or "").strip(), ""),
            str(handoff.selected_tool_id or "").strip(),
        ]
        for candidate in candidates:
            if candidate and self._task_or_none(candidate) is not None:
                return candidate
        return ""

    def _task_or_none(self, task_id: str) -> Any | None:
        if not task_id:
            return None
        try:
            return self.task_registry.get(task_id)
        except ValueError:
            return None

    @staticmethod
    def _policy_notes(*, task_id: str, adapter_name: str, policy: Any | None) -> list[str]:
        notes: list[str] = []
        if task_id:
            notes.append(f"atomic_task_metadata_resolved:{task_id}")
        else:
            notes.append("atomic_task_metadata_unresolved")
        if adapter_name:
            notes.append(f"adapter_metadata_resolved:{adapter_name}")
        if policy is not None:
            notes.append(f"adapter_policy_action:{policy.action}")
            if policy.snapshot_required_execute:
                notes.append("adapter_requires_snapshot_for_execute")
            if policy.validate_execute_boolean:
                notes.append("adapter_validates_execute_boolean")
        return notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery execution preview.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--selected-tool", required=True)
    parser.add_argument("--risk-budget", default="medium", choices=["low", "medium", "high"])
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
        assumptions=["Synthetic CLI handoff for review-only preview."],
    )
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(
        handoff=handoff,
        risk_budget=args.risk_budget,
        allow_auto_eligible=args.allow_auto_eligible,
        allow_gated=args.allow_gated,
    )
    summary = {
        "run_id": preview.run_id,
        "selected_tool_id": preview.selected_tool_id,
        "resolved_atomic_task_id": preview.resolved_atomic_task_id,
        "approval_mode": preview.approval_mode,
        "executable": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def _approval_mode(
    *,
    handoff: OLEDDiscoveryActionHandoff,
    task_resolved: bool,
    risk_level: str,
    required_gates: list[str],
    blocked_reasons: list[str],
    allow_auto_eligible: bool,
    allow_gated: bool,
) -> str:
    hard_blockers = {
        "gated_tool_not_allowed",
        "handoff_not_ready",
        "missing_required_inputs",
        "no_selected_tool",
        "risk_level_exceeds_budget",
    }
    if handoff.missing_inputs or any(reason in hard_blockers for reason in blocked_reasons):
        return "blocked"
    if required_gates:
        return "gated_review_required" if allow_gated else "blocked"
    if task_resolved and risk_level == "low" and allow_auto_eligible:
        return "auto_eligible"
    if task_resolved or handoff.selected_tool_id:
        return "manual_review_required"
    return "blocked"


def _normalize_risk(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _RISK_ORDER:
        raise ValueError(f"{field_name} must be low, medium, or high")
    return normalized


def _clean_unique(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True))


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
