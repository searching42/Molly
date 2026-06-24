from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactDecision, RunPlanArtifactVerification
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanAction, RunPlanReplanProposal
from ai4s_agent.run_plan_review_artifacts import (
    OBSERVER_VERIFICATION_ARTIFACT_ID,
    REPLAN_PROPOSAL_ARTIFACT_ID,
    REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID,
)


DEFAULT_REVIEW_ARTIFACTS: dict[str, str] = {
    OBSERVER_VERIFICATION_ARTIFACT_ID: "review/observer_verification.json",
    REPLAN_PROPOSAL_ARTIFACT_ID: "review/replan_proposal.json",
    REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID: "review/replan_review.md",
}


class RunPlanReviewCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    generated_at: str = Field(default_factory=now_iso)
    verifier_decision: RunPlanArtifactDecision
    proposed_action: RunPlanReplanAction
    summary: str
    required_user_decisions: list[str] = Field(default_factory=list)
    affected_tasks: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    proposal: dict[str, Any] = Field(default_factory=dict)
    review_markdown: str = ""
    executable: bool = False

    @field_validator("project_id", "run_id", "summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("review card text fields are required")
        return clean

    @field_validator("required_user_decisions", "affected_tasks")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("executable")
    @classmethod
    def validate_executable_false(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("review card is read-only and executable must remain false")
        return False

    @model_validator(mode="after")
    def validate_read_only(self) -> RunPlanReviewCard:
        if self.executable is not False:
            raise ValueError("review card is read-only and executable must remain false")
        if self.proposal and self.proposal.get("executable") is not False:
            raise ValueError("review card proposal must remain non-executable")
        patch = self.proposal.get("proposed_run_plan_patch") if isinstance(self.proposal, dict) else None
        if isinstance(patch, dict) and patch.get("applied") is not False:
            raise ValueError("review card proposal patch must remain unapplied")
        return self


def read_run_plan_review_card(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
) -> RunPlanReviewCard:
    """Read previously written verifier/replan artifacts into one review card.

    This is a read-only aggregation helper. It does not create artifacts, call
    the verifier, execute adapters, call LLMs, mutate a RunPlan, enqueue jobs,
    apply patches, or auto-rerun tasks.
    """

    workspace = Path(workspace_dir).expanduser().resolve()
    clean_project = _safe_component(project_id, "project_id")
    clean_run = _safe_component(run_id, "run_id")
    run_dir = _run_dir(workspace, clean_project, clean_run)
    artifacts = _review_artifact_paths(run_dir)

    verification_path = _resolve_artifact_path(run_dir, artifacts[OBSERVER_VERIFICATION_ARTIFACT_ID], OBSERVER_VERIFICATION_ARTIFACT_ID)
    proposal_path = _resolve_artifact_path(run_dir, artifacts[REPLAN_PROPOSAL_ARTIFACT_ID], REPLAN_PROPOSAL_ARTIFACT_ID)
    markdown_path = _resolve_artifact_path(run_dir, artifacts[REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID], REPLAN_REVIEW_MARKDOWN_ARTIFACT_ID)

    verification = RunPlanArtifactVerification.model_validate(_read_json_object(verification_path, OBSERVER_VERIFICATION_ARTIFACT_ID))
    proposal = RunPlanReplanProposal.model_validate(_read_json_object(proposal_path, REPLAN_PROPOSAL_ARTIFACT_ID))
    review_markdown = markdown_path.read_text(encoding="utf-8")
    if verification.project_id != clean_project or verification.run_id != clean_run:
        raise ValueError("observer verification project_id/run_id mismatch")
    return RunPlanReviewCard(
        project_id=clean_project,
        run_id=clean_run,
        verifier_decision=verification.decision,
        proposed_action=proposal.proposed_action,
        summary=verification.summary,
        required_user_decisions=proposal.required_user_decisions,
        affected_tasks=proposal.affected_tasks,
        artifacts=artifacts,
        verification=verification.model_dump(mode="json"),
        proposal=proposal.model_dump(mode="json"),
        review_markdown=review_markdown,
        executable=False,
    )


def _run_dir(workspace: Path, project_id: str, run_id: str) -> Path:
    projects_root = (workspace / "projects").resolve()
    run_dir = (projects_root / project_id / "runs" / run_id).resolve()
    if not run_dir.is_relative_to(projects_root):
        raise ValueError("run review path must stay under workspace projects")
    return run_dir


def _review_artifact_paths(run_dir: Path) -> dict[str, str]:
    registry_path = run_dir / "artifact_registry.json"
    artifacts = dict(DEFAULT_REVIEW_ARTIFACTS)
    if not registry_path.exists():
        return artifacts
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("artifact_registry.json is not valid JSON") from exc
    registry = payload.get("artifacts") if isinstance(payload, dict) else None
    if isinstance(registry, dict):
        for artifact_id in DEFAULT_REVIEW_ARTIFACTS:
            raw_path = str(registry.get(artifact_id) or "").strip()
            if raw_path:
                artifacts[artifact_id] = raw_path
    return artifacts


def _resolve_artifact_path(run_dir: Path, relative_path: str, artifact_id: str) -> Path:
    path = (run_dir / relative_path).resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError(f"{artifact_id} path escapes run directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{artifact_id} artifact not found: {relative_path}")
    return path


def _read_json_object(path: Path, artifact_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_id} artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_id} artifact JSON root must be an object")
    return payload


def _safe_component(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    if clean in {".", ".."} or "/" in clean or "\\" in clean:
        raise ValueError(f"{label} must be a safe path component")
    return clean
