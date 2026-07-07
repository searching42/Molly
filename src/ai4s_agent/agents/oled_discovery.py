from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.schemas import (
    OLEDDiscoveryNextAction,
    OLEDDiscoveryRunCard,
    OLEDDiscoveryStage,
    OLEDDiscoveryStageStatus,
)
from ai4s_agent.storage import ProjectStorage


_STAGE_ORDER = [
    OLEDDiscoveryStage.INTENT_CAPTURED,
    OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED,
    OLEDDiscoveryStage.ACQUISITION_PREPARED,
    OLEDDiscoveryStage.DATASET_READY,
    OLEDDiscoveryStage.TRAINING_PACKAGE_READY,
    OLEDDiscoveryStage.BASELINE_READY,
    OLEDDiscoveryStage.DIAGNOSTICS_READY,
    OLEDDiscoveryStage.CANDIDATES_READY,
    OLEDDiscoveryStage.CRITIC_REVIEWED,
    OLEDDiscoveryStage.NEXT_ACTION_PROPOSED,
]


class OLEDDiscoveryLoopAgent:
    """Review-only state machine for OLED discovery runs."""

    def build_run_card(
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
        critic_review: dict[str, Any] | None = None,
    ) -> OLEDDiscoveryRunCard:
        clean_goal = str(goal or "").strip()
        payloads = {
            "conversation_decision": _clean_payload(conversation_decision),
            "research_source_proposal": _clean_payload(research_source_proposal),
            "research_acquisition_preparation": _clean_payload(research_acquisition_preparation),
            "target_modeling_brief": _clean_payload(target_modeling_brief),
            "dataset_artifacts": _clean_payload(dataset_artifacts),
            "training_package_artifacts": _clean_payload(training_package_artifacts),
            "baseline_artifacts": _clean_payload(baseline_artifacts),
            "diagnostics_report": _clean_payload(diagnostics_report),
            "candidate_artifacts": _clean_payload(candidate_artifacts),
            "critic_review": _clean_payload(critic_review),
        }
        intent_ready = bool(clean_goal) or _has_payload(payloads["conversation_decision"])
        research_ready = _has_payload(payloads["research_source_proposal"])
        acquisition_ready = _has_payload(payloads["research_acquisition_preparation"])
        dataset_ready = _has_payload(payloads["dataset_artifacts"])
        training_ready = _has_payload(payloads["training_package_artifacts"])
        baseline_ready = _has_payload(payloads["baseline_artifacts"])
        diagnostics_ready = _has_payload(payloads["diagnostics_report"])
        candidates_ready = _has_payload(payloads["candidate_artifacts"])
        critic_ready = _has_payload(payloads["critic_review"])

        blocked_reasons: list[str] = []
        risk_flags: list[str] = []
        if not intent_ready:
            blocked_reasons.append("missing_discovery_objective")
        if acquisition_ready and not dataset_ready and _requires_external_acquisition(payloads["research_acquisition_preparation"]):
            risk_flags.append("data_mining_gate_required")
        diagnostics_weak = diagnostics_ready and _diagnostics_need_revision(payloads["diagnostics_report"])
        if diagnostics_weak:
            risk_flags.append("diagnostics_not_ready_for_candidate_screening")

        current_stage = self._current_stage(
            intent_ready=intent_ready,
            research_ready=research_ready,
            acquisition_ready=acquisition_ready,
            dataset_ready=dataset_ready,
            training_ready=training_ready,
            baseline_ready=baseline_ready,
            diagnostics_ready=diagnostics_ready,
            candidates_ready=candidates_ready,
            critic_ready=critic_ready,
        )
        available_artifacts = self._available_artifacts(clean_goal, payloads)
        missing_artifacts = self._missing_artifacts(
            intent_ready=intent_ready,
            research_ready=research_ready,
            acquisition_ready=acquisition_ready,
            dataset_ready=dataset_ready,
            training_ready=training_ready,
            baseline_ready=baseline_ready,
            diagnostics_ready=diagnostics_ready,
            candidates_ready=candidates_ready,
            critic_ready=critic_ready,
        )
        actions = self._next_actions(
            current_stage=current_stage,
            diagnostics_weak=diagnostics_weak,
            acquisition_preparation=payloads["research_acquisition_preparation"],
        )

        stage_statuses = self._stage_statuses(
            intent_ready=intent_ready,
            research_ready=research_ready,
            acquisition_ready=acquisition_ready,
            dataset_ready=dataset_ready,
            training_ready=training_ready,
            baseline_ready=baseline_ready,
            diagnostics_ready=diagnostics_ready,
            candidates_ready=candidates_ready,
            critic_ready=critic_ready,
            diagnostics_weak=diagnostics_weak,
            blocked=bool(blocked_reasons),
        )
        return OLEDDiscoveryRunCard(
            run_id=str(run_id or "").strip(),
            project_id=str(project_id).strip() if project_id else None,
            goal=clean_goal,
            current_stage=current_stage.value,
            stage_statuses=stage_statuses,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
            blocked_reasons=blocked_reasons,
            risk_flags=sorted(set(risk_flags)),
            recommended_next_actions=actions,
            assumptions=[
                "Run card is a deterministic review artifact only.",
                "No research acquisition, parsing, modeling, prediction, or registry mutation is executed.",
                "Supplied artifact dictionaries are treated as summaries, not as file paths to inspect.",
            ],
            executable=False,
        )

    def write_run_card(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        run_card: OLEDDiscoveryRunCard,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "oled_discovery_run_card.json", run_card.model_dump(mode="json"))
        md_path = run_dir / "oled_discovery_run_card.md"
        md_path.write_text(self.render_markdown(run_card), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "oled_discovery_run_card_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "oled_discovery_run_card_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def render_markdown(run_card: OLEDDiscoveryRunCard) -> str:
        lines = [
            "# OLED Discovery Run Card",
            "",
            f"- Run ID: {run_card.run_id}",
            f"- Project ID: {run_card.project_id or ''}",
            f"- Goal: {run_card.goal}",
            f"- Current Stage: {run_card.current_stage}",
            f"- Executable: {str(run_card.executable).lower()}",
            "",
            "## Stage Status",
            "",
            "| Stage | Status | Summary | Missing |",
            "| --- | --- | --- | --- |",
        ]
        for status in run_card.stage_statuses:
            missing = ", ".join(status.missing)
            lines.append(f"| {status.stage} | {status.status} | {status.summary} | {missing} |")
        lines.extend(["", "## Available Artifacts", ""])
        lines.extend(_markdown_list(run_card.available_artifacts))
        lines.extend(["", "## Blocked Reasons", ""])
        lines.extend(_markdown_list(run_card.blocked_reasons))
        lines.extend(["", "## Risk Flags", ""])
        lines.extend(_markdown_list(run_card.risk_flags))
        lines.extend(["", "## Recommended Next Actions", ""])
        if run_card.recommended_next_actions:
            for action in run_card.recommended_next_actions:
                gate = "requires gate" if action.requires_gate else "review only"
                task = f"; suggested task: {action.suggested_task}" if action.suggested_task else ""
                lines.append(f"- `{action.action_id}`: {action.label} ({gate}{task}) - {action.reason}")
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Safety Boundary",
                "",
                "- Review-only run card.",
                "- No registry, promotion, publication, release, or global append mutation.",
                "- No backend/model execution, model training, prediction, LLM calls, MinerU calls, PDF/image reads, or external network access.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _current_stage(
        *,
        intent_ready: bool,
        research_ready: bool,
        acquisition_ready: bool,
        dataset_ready: bool,
        training_ready: bool,
        baseline_ready: bool,
        diagnostics_ready: bool,
        candidates_ready: bool,
        critic_ready: bool,
    ) -> OLEDDiscoveryStage:
        if not intent_ready:
            return OLEDDiscoveryStage.BLOCKED
        if critic_ready:
            return OLEDDiscoveryStage.CRITIC_REVIEWED
        if candidates_ready:
            return OLEDDiscoveryStage.CANDIDATES_READY
        if diagnostics_ready:
            return OLEDDiscoveryStage.DIAGNOSTICS_READY
        if baseline_ready:
            return OLEDDiscoveryStage.BASELINE_READY
        if training_ready:
            return OLEDDiscoveryStage.TRAINING_PACKAGE_READY
        if dataset_ready:
            return OLEDDiscoveryStage.DATASET_READY
        if acquisition_ready:
            return OLEDDiscoveryStage.ACQUISITION_PREPARED
        if research_ready:
            return OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED
        return OLEDDiscoveryStage.INTENT_CAPTURED

    @staticmethod
    def _available_artifacts(goal: str, payloads: dict[str, dict[str, Any]]) -> list[str]:
        names: list[str] = []
        if goal:
            names.append("goal")
        names.extend(name for name, payload in payloads.items() if _has_payload(payload))
        return sorted(names)

    @staticmethod
    def _missing_artifacts(
        *,
        intent_ready: bool,
        research_ready: bool,
        acquisition_ready: bool,
        dataset_ready: bool,
        training_ready: bool,
        baseline_ready: bool,
        diagnostics_ready: bool,
        candidates_ready: bool,
        critic_ready: bool,
    ) -> list[str]:
        checks = [
            ("goal_or_conversation_decision", intent_ready),
            ("research_source_proposal", research_ready),
            ("research_acquisition_preparation", acquisition_ready),
            ("dataset_artifacts", dataset_ready),
            ("training_package_artifacts", training_ready),
            ("baseline_artifacts", baseline_ready),
            ("diagnostics_report", diagnostics_ready),
            ("candidate_artifacts", candidates_ready),
            ("critic_review", critic_ready),
        ]
        return [name for name, ready in checks if not ready]

    @staticmethod
    def _next_actions(
        *,
        current_stage: OLEDDiscoveryStage,
        diagnostics_weak: bool,
        acquisition_preparation: dict[str, Any],
    ) -> list[OLEDDiscoveryNextAction]:
        if current_stage == OLEDDiscoveryStage.BLOCKED:
            return [
                _action(
                    "provide_oled_discovery_objective",
                    "Provide OLED discovery objective",
                    "No usable OLED discovery objective or conversation decision was supplied.",
                    OLEDDiscoveryStage.INTENT_CAPTURED,
                    False,
                    None,
                )
            ]
        if current_stage == OLEDDiscoveryStage.INTENT_CAPTURED:
            return [
                _action(
                    "prepare_research_source_proposal",
                    "Prepare research source proposal",
                    "Intent is captured; source planning is the next review-only step.",
                    OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED,
                    False,
                    "research_source_proposal",
                )
            ]
        if current_stage == OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED:
            return [
                _action(
                    "prepare_acquisition_scope",
                    "Prepare acquisition scope",
                    "Research source proposal exists but acquisition preparation is missing.",
                    OLEDDiscoveryStage.ACQUISITION_PREPARED,
                    False,
                    "research_acquisition_preparation",
                )
            ]
        if current_stage == OLEDDiscoveryStage.ACQUISITION_PREPARED:
            requires_gate = _requires_external_acquisition(acquisition_preparation)
            return [
                _action(
                    "run_or_provide_dataset_artifacts",
                    "Run approved local data workflow or provide dataset artifacts",
                    "Acquisition is prepared but curated dataset artifacts are not available.",
                    OLEDDiscoveryStage.DATASET_READY,
                    requires_gate,
                    "data_mining" if requires_gate else "dataset_materialization",
                )
            ]
        if current_stage == OLEDDiscoveryStage.DATASET_READY:
            return [
                _action(
                    "prepare_training_package",
                    "Prepare split training package",
                    "Dataset artifacts exist but training package artifacts are missing.",
                    OLEDDiscoveryStage.TRAINING_PACKAGE_READY,
                    True,
                    "training_package_preparation",
                )
            ]
        if current_stage == OLEDDiscoveryStage.TRAINING_PACKAGE_READY:
            return [
                _action(
                    "run_backend_readiness_or_baseline",
                    "Run backend readiness or low-risk baseline",
                    "Training package artifacts exist but baseline artifacts are missing.",
                    OLEDDiscoveryStage.BASELINE_READY,
                    True,
                    "baseline_or_backend_readiness",
                )
            ]
        if current_stage == OLEDDiscoveryStage.BASELINE_READY:
            return [
                _action(
                    "generate_diagnostics_report",
                    "Generate diagnostics report",
                    "Baseline artifacts exist but diagnostics have not been reviewed.",
                    OLEDDiscoveryStage.DIAGNOSTICS_READY,
                    False,
                    "diagnostics_report",
                )
            ]
        if current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY and diagnostics_weak:
            return [
                _action(
                    "revise_data_or_model_before_candidates",
                    "Revise data or model before candidate screening",
                    "Diagnostics indicate weak, blocked, or rerun-needed model state.",
                    OLEDDiscoveryStage.BASELINE_READY,
                    True,
                    "data_or_model_revision",
                )
            ]
        if current_stage == OLEDDiscoveryStage.DIAGNOSTICS_READY:
            return [
                _action(
                    "prepare_candidate_generation_or_prediction",
                    "Prepare candidate generation or prediction",
                    "Diagnostics are acceptable and candidate artifacts are missing.",
                    OLEDDiscoveryStage.CANDIDATES_READY,
                    True,
                    "candidate_screening_preparation",
                )
            ]
        if current_stage == OLEDDiscoveryStage.CANDIDATES_READY:
            return [
                _action(
                    "run_critic_review",
                    "Run critic review",
                    "Candidate artifacts exist and need independent review.",
                    OLEDDiscoveryStage.CRITIC_REVIEWED,
                    False,
                    "critic_review",
                )
            ]
        return [
            _action(
                "propose_next_research_model_or_data_action",
                "Propose next research, model, or data action",
                "Critic review exists; next action should be selected by a reviewed plan.",
                OLEDDiscoveryStage.NEXT_ACTION_PROPOSED,
                False,
                "next_action_proposal",
            )
        ]

    @staticmethod
    def _stage_statuses(
        *,
        intent_ready: bool,
        research_ready: bool,
        acquisition_ready: bool,
        dataset_ready: bool,
        training_ready: bool,
        baseline_ready: bool,
        diagnostics_ready: bool,
        candidates_ready: bool,
        critic_ready: bool,
        diagnostics_weak: bool,
        blocked: bool,
    ) -> list[OLEDDiscoveryStageStatus]:
        readiness = {
            OLEDDiscoveryStage.INTENT_CAPTURED: intent_ready,
            OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED: research_ready,
            OLEDDiscoveryStage.ACQUISITION_PREPARED: acquisition_ready,
            OLEDDiscoveryStage.DATASET_READY: dataset_ready,
            OLEDDiscoveryStage.TRAINING_PACKAGE_READY: training_ready,
            OLEDDiscoveryStage.BASELINE_READY: baseline_ready,
            OLEDDiscoveryStage.DIAGNOSTICS_READY: diagnostics_ready,
            OLEDDiscoveryStage.CANDIDATES_READY: candidates_ready,
            OLEDDiscoveryStage.CRITIC_REVIEWED: critic_ready,
            OLEDDiscoveryStage.NEXT_ACTION_PROPOSED: critic_ready,
        }
        evidence = {
            OLEDDiscoveryStage.INTENT_CAPTURED: ["goal_or_conversation_decision"],
            OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED: ["research_source_proposal"],
            OLEDDiscoveryStage.ACQUISITION_PREPARED: ["research_acquisition_preparation"],
            OLEDDiscoveryStage.DATASET_READY: ["dataset_artifacts"],
            OLEDDiscoveryStage.TRAINING_PACKAGE_READY: ["training_package_artifacts"],
            OLEDDiscoveryStage.BASELINE_READY: ["baseline_artifacts"],
            OLEDDiscoveryStage.DIAGNOSTICS_READY: ["diagnostics_report"],
            OLEDDiscoveryStage.CANDIDATES_READY: ["candidate_artifacts"],
            OLEDDiscoveryStage.CRITIC_REVIEWED: ["critic_review"],
            OLEDDiscoveryStage.NEXT_ACTION_PROPOSED: ["critic_review"],
        }
        statuses: list[OLEDDiscoveryStageStatus] = []
        prior_ready = True
        for stage in _STAGE_ORDER:
            is_ready = readiness[stage]
            if blocked and stage == OLEDDiscoveryStage.INTENT_CAPTURED:
                status = "blocked"
                summary = "Discovery objective is missing."
            elif is_ready and stage == OLEDDiscoveryStage.DIAGNOSTICS_READY and diagnostics_weak:
                status = "needs_review"
                summary = "Diagnostics are available but indicate revision before candidate screening."
            elif is_ready and stage == OLEDDiscoveryStage.NEXT_ACTION_PROPOSED:
                status = "ready"
                summary = "Critic review is available for next-action planning."
            elif is_ready:
                status = "complete"
                summary = _stage_summary(stage, True)
            elif prior_ready:
                status = "missing"
                summary = _stage_summary(stage, False)
            else:
                status = "missing"
                summary = "Waiting on earlier discovery-loop stages."
            statuses.append(
                OLEDDiscoveryStageStatus(
                    stage=stage.value,
                    status=status,
                    evidence=evidence[stage] if is_ready else [],
                    missing=[] if is_ready else evidence[stage],
                    summary=summary,
                )
            )
            prior_ready = prior_ready and is_ready
        return statuses


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a review-only OLED discovery loop run card.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    agent = OLEDDiscoveryLoopAgent()
    card = agent.build_run_card(run_id=args.run_id, goal=args.goal, project_id=args.project_id)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "oled_discovery_run_card.json", card.model_dump(mode="json"))
        (output_dir / "oled_discovery_run_card.md").write_text(agent.render_markdown(card), encoding="utf-8")
    print(json.dumps({"run_id": card.run_id, "current_stage": card.current_stage, "executable": card.executable}, sort_keys=True))
    return 0


def _clean_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _has_payload(value: Mapping[str, Any]) -> bool:
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
        return _has_payload(value)
    if isinstance(value, list | tuple | set):
        return any(_usable_value(item) for item in value)
    return True


def _requires_external_acquisition(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    direct_flags = (
        "requires_external_acquisition",
        "external_acquisition_required",
        "needs_external_acquisition",
        "requires_data_mining_gate",
    )
    if any(bool(payload.get(key)) for key in direct_flags):
        return True
    gates = payload.get("required_gates")
    if isinstance(gates, list) and any("data_mining" in str(item).lower() or "gate_2" in str(item).lower() for item in gates):
        return True
    permissions = payload.get("required_permissions")
    return isinstance(permissions, list) and any("external" in str(item).lower() for item in permissions)


def _diagnostics_need_revision(payload: Mapping[str, Any]) -> bool:
    text = " ".join(str(value).lower() for value in payload.values() if isinstance(value, str))
    status = str(payload.get("status") or payload.get("decision") or payload.get("overall_decision") or "").lower()
    return any(token in status or token in text for token in ("weak", "blocked", "rerun", "revise", "revision", "failed"))


def _action(
    action_id: str,
    label: str,
    reason: str,
    target_stage: OLEDDiscoveryStage,
    requires_gate: bool,
    suggested_task: str | None,
) -> OLEDDiscoveryNextAction:
    return OLEDDiscoveryNextAction(
        action_id=action_id,
        label=label,
        reason=reason,
        target_stage=target_stage.value,
        requires_gate=requires_gate,
        suggested_task=suggested_task,
    )


def _stage_summary(stage: OLEDDiscoveryStage, ready: bool) -> str:
    ready_summaries = {
        OLEDDiscoveryStage.INTENT_CAPTURED: "Discovery intent is captured.",
        OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED: "Research/source plan is available.",
        OLEDDiscoveryStage.ACQUISITION_PREPARED: "Acquisition scope is prepared for review.",
        OLEDDiscoveryStage.DATASET_READY: "Dataset artifacts are available.",
        OLEDDiscoveryStage.TRAINING_PACKAGE_READY: "Training package artifacts are available.",
        OLEDDiscoveryStage.BASELINE_READY: "Baseline artifacts are available.",
        OLEDDiscoveryStage.DIAGNOSTICS_READY: "Diagnostics report is available.",
        OLEDDiscoveryStage.CANDIDATES_READY: "Candidate artifacts are available.",
        OLEDDiscoveryStage.CRITIC_REVIEWED: "Critic review is available.",
        OLEDDiscoveryStage.NEXT_ACTION_PROPOSED: "Next action proposal can be reviewed.",
    }
    missing_summaries = {
        OLEDDiscoveryStage.INTENT_CAPTURED: "Discovery intent has not been captured.",
        OLEDDiscoveryStage.RESEARCH_PLAN_PROPOSED: "Research/source proposal is missing.",
        OLEDDiscoveryStage.ACQUISITION_PREPARED: "Acquisition preparation is missing.",
        OLEDDiscoveryStage.DATASET_READY: "Dataset artifacts are missing.",
        OLEDDiscoveryStage.TRAINING_PACKAGE_READY: "Training package artifacts are missing.",
        OLEDDiscoveryStage.BASELINE_READY: "Baseline artifacts are missing.",
        OLEDDiscoveryStage.DIAGNOSTICS_READY: "Diagnostics report is missing.",
        OLEDDiscoveryStage.CANDIDATES_READY: "Candidate artifacts are missing.",
        OLEDDiscoveryStage.CRITIC_REVIEWED: "Critic review is missing.",
        OLEDDiscoveryStage.NEXT_ACTION_PROPOSED: "Next action proposal is not ready.",
    }
    return (ready_summaries if ready else missing_summaries)[stage]


def _markdown_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
