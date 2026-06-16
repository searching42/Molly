from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.schemas import (
    PlanQuestion,
    ReportNextStep,
    ReportSection,
    ReportSynthesisProposal,
    VerificationReport,
)
from ai4s_agent.storage import ProjectStorage


class ReportAgent:
    """Dry-run report synthesizer for audited run review and next-step planning."""

    def synthesize_run(
        self,
        *,
        run_id: str,
        goal: str,
        observation: dict[str, Any] | None = None,
        verification_report: VerificationReport | dict[str, Any] | None = None,
        run_plan_revision: dict[str, Any] | None = None,
        research_proposal: dict[str, Any] | None = None,
        modeling_proposal: dict[str, Any] | None = None,
        generation_proposal: dict[str, Any] | None = None,
    ) -> ReportSynthesisProposal:
        clean_goal = str(goal or "").strip()
        verification = self._coerce_verification_report(verification_report)
        sections = self._sections(
            observation=observation or {},
            verification=verification,
            run_plan_revision=run_plan_revision or {},
            research_proposal=research_proposal or {},
            modeling_proposal=modeling_proposal or {},
            generation_proposal=generation_proposal or {},
        )
        limitations = self._limitations(
            observation=observation or {},
            verification=verification,
            research_proposal=research_proposal or {},
            modeling_proposal=modeling_proposal or {},
            generation_proposal=generation_proposal or {},
        )
        next_steps = self._next_steps(
            verification=verification,
            research_proposal=research_proposal or {},
            modeling_proposal=modeling_proposal or {},
            generation_proposal=generation_proposal or {},
        )
        questions = self._questions(next_steps)
        status = "needs_clarification" if questions else "needs_confirmation"

        return ReportSynthesisProposal(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            status=status,
            executive_summary=self._executive_summary(verification, limitations),
            sections=sections,
            limitations=limitations,
            next_steps=next_steps,
            paper_audit_outline=[
                "Objective and scope",
                "Methods and adapters used",
                "Artifacts and provenance",
                "Verification findings",
                "Limitations and next steps",
            ],
            assumptions=[
                "ReportAgent summarizes supplied structured artifacts only; it does not execute tasks.",
                "Recommendations must be reviewed before triggering adapters or asset promotion.",
            ],
            questions=questions,
            executable=False,
        )

    def write_proposal(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        proposal: ReportSynthesisProposal,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "report_synthesis_proposal.json", proposal.model_dump(mode="json"))
        md_path = run_dir / "report_synthesis_proposal.md"
        md_path.write_text(self._render_markdown(proposal), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "report_synthesis_proposal_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "report_synthesis_proposal_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def _coerce_verification_report(value: VerificationReport | dict[str, Any] | None) -> VerificationReport | None:
        if value is None:
            return None
        if isinstance(value, VerificationReport):
            return value
        if isinstance(value, dict):
            return VerificationReport.model_validate(value)
        raise ValueError("verification_report must be an object")

    @staticmethod
    def _sections(
        *,
        observation: dict[str, Any],
        verification: VerificationReport | None,
        run_plan_revision: dict[str, Any],
        research_proposal: dict[str, Any],
        modeling_proposal: dict[str, Any],
        generation_proposal: dict[str, Any],
    ) -> list[ReportSection]:
        sections: list[ReportSection] = []
        stage_state = observation.get("stage_state", {}) if isinstance(observation, dict) else {}
        if isinstance(stage_state, dict) and stage_state:
            sections.append(
                ReportSection(
                    title="Run State",
                    summary=f"Observed stage `{stage_state.get('stage', '')}` with status `{stage_state.get('status', '')}`.",
                    evidence_refs=["run_observation"],
                    details={"artifact_count": len(ReportAgent._list_payload(observation.get("artifacts")))},
                )
            )
        if verification is not None:
            categories = sorted({finding.category for finding in verification.findings})
            sections.append(
                ReportSection(
                    title="Verification",
                    summary=verification.summary or f"Verifier decision: {verification.overall_decision}.",
                    evidence_refs=["verification_report_json"],
                    risk_flags=categories,
                    details={
                        "overall_decision": verification.overall_decision,
                        "finding_count": len(verification.findings),
                        "categories": categories,
                    },
                )
            )
        for title, payload in (
            ("Research Sources", research_proposal),
            ("Modeling Plan", modeling_proposal),
            ("Generation Strategy", generation_proposal),
            ("Run Plan Revision", run_plan_revision),
        ):
            if payload:
                sections.append(
                    ReportSection(
                        title=title,
                        summary=ReportAgent._proposal_summary(title, payload),
                        evidence_refs=[ReportAgent._artifact_ref_for_title(title)],
                        details=ReportAgent._compact_payload_details(payload),
                    )
                )
        return sections

    @staticmethod
    def _limitations(
        *,
        observation: dict[str, Any],
        verification: VerificationReport | None,
        research_proposal: dict[str, Any],
        modeling_proposal: dict[str, Any],
        generation_proposal: dict[str, Any],
    ) -> list[str]:
        limitations: list[str] = []
        for note in ReportAgent._list_payload(observation.get("notes") if isinstance(observation, dict) else []):
            limitations.append(str(note))
        if verification is None:
            limitations.append("No verification report was supplied.")
        else:
            for finding in verification.findings:
                limitations.append(f"{finding.category}: {finding.message}")
        for label, payload in (
            ("research", research_proposal),
            ("modeling", modeling_proposal),
            ("generation", generation_proposal),
        ):
            if payload.get("status") == "needs_clarification":
                limitations.append(f"{label} proposal still has blocking questions.")
            for permission in ReportAgent._list_payload(
                payload.get("required_permissions") if isinstance(payload, dict) else []
            ):
                limitations.append(f"{label} proposal requires permission: {permission}")
        return ReportAgent._dedup(limitations)

    @staticmethod
    def _next_steps(
        *,
        verification: VerificationReport | None,
        research_proposal: dict[str, Any],
        modeling_proposal: dict[str, Any],
        generation_proposal: dict[str, Any],
    ) -> list[ReportNextStep]:
        steps: list[ReportNextStep] = []
        if verification is not None:
            if verification.overall_decision == "replan":
                steps.append(
                    ReportNextStep(
                        action="propose_replan",
                        reason="Verifier requested a revised plan before continuing.",
                        priority="high",
                        required_approval=True,
                        related_artifacts=["verification_report_json"],
                    )
                )
            elif verification.overall_decision == "retry":
                steps.append(
                    ReportNextStep(
                        action="retry_failed_stage",
                        reason="Verifier found a retryable issue.",
                        priority="high",
                        required_approval=True,
                        related_artifacts=["verification_report_json"],
                    )
                )
            elif verification.overall_decision == "ask_user":
                steps.append(
                    ReportNextStep(
                        action="answer_verifier_questions",
                        reason="Verifier requires human input before advancement.",
                        priority="high",
                        required_approval=True,
                        related_artifacts=["verification_report_json"],
                    )
                )
            elif verification.overall_decision == "abort":
                steps.append(
                    ReportNextStep(
                        action="abort_run",
                        reason="Verifier found a critical issue.",
                        priority="high",
                        required_approval=True,
                        related_artifacts=["verification_report_json"],
                    )
                )
        if research_proposal.get("status") == "needs_clarification":
            steps.append(ReportNextStep(action="add_research_sources", reason="Research proposal lacks approved sources.", priority="medium", required_approval=True))
        if ReportAgent._list_payload(modeling_proposal.get("retry_proposals")):
            steps.append(ReportNextStep(action="review_modeling_retry", reason="Modeling proposal includes retry options.", priority="medium", required_approval=True))
        if "generate_candidates_expensive" in ReportAgent._list_payload(generation_proposal.get("required_permissions")):
            steps.append(ReportNextStep(action="confirm_generation_budget", reason="Generation proposal requires expensive-generation permission.", priority="high", required_approval=True))
        if not steps:
            steps.append(ReportNextStep(action="review_and_confirm_report", reason="No blocking finding was supplied.", priority="medium", required_approval=False))
        return ReportAgent._dedup_steps(steps)

    @staticmethod
    def _questions(next_steps: list[ReportNextStep]) -> list[PlanQuestion]:
        blocking = [step for step in next_steps if step.required_approval]
        if not blocking:
            return []
        return [
            PlanQuestion(
                question_id="q_report_next_step_confirmation",
                prompt="Which recommended next step should be approved before execution?",
                reason="The report summary includes review-gated next steps.",
                choices=[step.action for step in blocking],
                blocks_execution=True,
            )
        ]

    @staticmethod
    def _executive_summary(verification: VerificationReport | None, limitations: list[str]) -> str:
        if verification is not None and verification.summary:
            return verification.summary
        if verification is not None:
            return f"Verifier decision is `{verification.overall_decision}` with {len(verification.findings)} findings."
        return f"Run summary is based on supplied artifacts; {len(limitations)} limitations were identified."

    @staticmethod
    def _proposal_summary(title: str, payload: dict[str, Any]) -> str:
        status = str(payload.get("status") or "").strip()
        if title == "Generation Strategy":
            backend = str(payload.get("backend") or "").strip()
            count = str(payload.get("requested_count") or "").strip()
            return f"Generation proposal status `{status}` using `{backend}` for {count or 'unknown'} candidates."
        if title == "Modeling Plan":
            return f"Modeling proposal status `{status}` with {len(ReportAgent._list_payload(payload.get('retry_proposals')))} retry proposals."
        if title == "Research Sources":
            return f"Research proposal status `{status}`."
        return f"{title} artifact was supplied for audit synthesis."

    @staticmethod
    def _artifact_ref_for_title(title: str) -> str:
        return {
            "Research Sources": "research_source_proposal_json",
            "Modeling Plan": "modeling_plan_proposal_json",
            "Generation Strategy": "generation_strategy_proposal_json",
            "Run Plan Revision": "run_plan_revision_json",
        }.get(title, title.lower().replace(" ", "_"))

    @staticmethod
    def _compact_payload_details(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: payload[key]
            for key in ("status", "backend", "requested_count", "required_permissions", "overall_decision")
            if key in payload
        }

    @staticmethod
    def _dedup(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _dedup_steps(steps: list[ReportNextStep]) -> list[ReportNextStep]:
        seen: set[str] = set()
        result: list[ReportNextStep] = []
        for step in steps:
            if step.action in seen:
                continue
            seen.add(step.action)
            result.append(step)
        return result

    @staticmethod
    def _list_payload(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _render_markdown(proposal: ReportSynthesisProposal) -> str:
        lines = [
            "# Report Synthesis Proposal",
            "",
            f"- Run: `{proposal.run_id}`",
            f"- Status: `{proposal.status}`",
            "",
            "## Executive Summary",
            proposal.executive_summary,
            "",
            "## Sections",
        ]
        lines.extend(f"- **{section.title}:** {section.summary}" for section in proposal.sections)
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in proposal.limitations)
        lines.extend(["", "## Next Steps"])
        lines.extend(f"- `{step.action}` ({step.priority}): {step.reason}" for step in proposal.next_steps)
        return "\n".join(lines) + "\n"
