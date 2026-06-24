from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactVerification, verify_run_plan_artifacts
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal, propose_replan_from_verification
from ai4s_agent.storage import ProjectStorage


OBSERVER_VERIFICATION_ARTIFACT_ID = "observer_verification"
REPLAN_PROPOSAL_ARTIFACT_ID = "replan_proposal"
REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID = "replan_review_markdown"


class RunPlanReviewArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    generated_at: str = Field(default_factory=now_iso)
    verification: RunPlanArtifactVerification
    proposal: RunPlanReplanProposal
    artifact_ids: list[str]
    artifacts: dict[str, str]
    executable: bool = False

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("project_id and run_id are required")
        return clean

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @model_validator(mode="after")
    def validate_non_executable(self) -> RunPlanReviewArtifactBundle:
        if self.executable is not False:
            raise ValueError("review artifacts are not executable")
        if self.proposal.executable is not False:
            raise ValueError("replan proposal must remain non-executable")
        return self


def write_run_plan_review_artifacts(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    queue_summary: dict[str, Any] | None = None,
    queue_status: dict[str, Any] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
    artifact_registry: dict[str, str] | None = None,
    verification: RunPlanArtifactVerification | dict[str, Any] | None = None,
) -> RunPlanReviewArtifactBundle:
    """Write read-only verifier/replan review artifacts under a run directory.

    This function only materializes review artifacts. It does not execute
    adapters, call LLMs, mutate a RunPlan, enqueue jobs, apply proposal patches,
    or automatically rerun tasks.
    """

    storage = ProjectStorage(Path(workspace_dir))
    run_dir = storage.run_dir(project_id, run_id)
    verified = _verification_or_run(
        verification=verification,
        workspace_dir=Path(workspace_dir),
        project_id=project_id,
        run_id=run_id,
        queue_summary=queue_summary,
        queue_status=queue_status,
        audit_records=audit_records,
        artifact_registry=artifact_registry,
    )
    if verified.project_id != project_id or verified.run_id != run_id:
        raise ValueError("verification project_id/run_id mismatch")

    proposal = propose_replan_from_verification(verified)
    verification_rel = "review/observer_verification.json"
    proposal_rel = "review/replan_proposal.json"
    markdown_rel = "review/replan_review.md"

    write_json(run_dir / verification_rel, verified.model_dump(mode="json"))
    write_json(run_dir / proposal_rel, proposal.model_dump(mode="json"))
    markdown_path = run_dir / markdown_rel
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_review_markdown(verified, proposal), encoding="utf-8")

    artifacts = {
        OBSERVER_VERIFICATION_ARTIFACT_ID: verification_rel,
        REPLAN_PROPOSAL_ARTIFACT_ID: proposal_rel,
        REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID: markdown_rel,
    }
    for artifact_id, relative_path in artifacts.items():
        storage.register_artifact_path(project_id, run_id, artifact_id, relative_path)

    return RunPlanReviewArtifactBundle(
        project_id=project_id,
        run_id=run_id,
        verification=verified,
        proposal=proposal,
        artifact_ids=list(artifacts),
        artifacts=artifacts,
        executable=False,
    )


def _verification_or_run(
    *,
    verification: RunPlanArtifactVerification | dict[str, Any] | None,
    workspace_dir: Path,
    project_id: str,
    run_id: str,
    queue_summary: dict[str, Any] | None,
    queue_status: dict[str, Any] | None,
    audit_records: list[dict[str, Any]] | None,
    artifact_registry: dict[str, str] | None,
) -> RunPlanArtifactVerification:
    if verification is not None:
        return (
            verification
            if isinstance(verification, RunPlanArtifactVerification)
            else RunPlanArtifactVerification.model_validate(verification)
        )
    return verify_run_plan_artifacts(
        workspace_dir=workspace_dir,
        project_id=project_id,
        run_id=run_id,
        queue_summary=queue_summary,
        queue_status=queue_status,
        audit_records=audit_records,
        artifact_registry=artifact_registry,
    )


def _render_review_markdown(
    verification: RunPlanArtifactVerification,
    proposal: RunPlanReplanProposal,
) -> str:
    lines = [
        "# Run Plan Review",
        "",
        f"- Project: `{verification.project_id}`",
        f"- Run: `{verification.run_id}`",
        f"- Observer-Verifier Decision: `{verification.decision}`",
        f"- Proposed Action: `{proposal.proposed_action}`",
        f"- Executable: `{str(proposal.executable).lower()}`",
        "",
        "No adapter execution, RunPlan mutation, enqueue, patch apply, or auto-rerun was performed.",
        "",
        "## Observer Summary",
        "",
        verification.summary,
        "",
        "## Findings",
        "",
    ]
    if verification.findings:
        for finding in verification.findings:
            lines.append(
                f"- `{finding.finding_id}` `{finding.category}` `{finding.severity}` "
                f"`{finding.decision}`: {finding.message}"
            )
    else:
        lines.append("- No verifier findings.")

    lines.extend(
        [
            "",
            "## Replan Proposal",
            "",
            f"- Decision source: `{proposal.decision_source}`",
            f"- Verifier decision: `{proposal.verifier_decision}`",
            f"- Proposed action: `{proposal.proposed_action}`",
            f"- Affected tasks: {_markdown_list_value(proposal.affected_tasks)}",
            f"- Source findings: {_markdown_list_value(proposal.source_finding_ids)}",
            "",
            "## Required User Decisions",
            "",
        ]
    )
    if proposal.required_user_decisions:
        lines.extend(f"- {item}" for item in proposal.required_user_decisions)
    else:
        lines.append("- No user decision required before continuing.")

    operations = proposal.proposed_run_plan_patch.get("operations", [])
    lines.extend(
        [
            "",
            "## Advisory Patch",
            "",
            f"- Applied: `{str(proposal.proposed_run_plan_patch.get('applied') is True).lower()}`",
            f"- Operation count: `{len(operations) if isinstance(operations, list) else 0}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_list_value(values: list[str]) -> str:
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"
