from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_review_card import RunPlanReviewCard, read_run_plan_review_card
from ai4s_agent.schemas import ProjectMemoryRecord


class RunPlanReviewMemorySave(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    saved: bool
    record: ProjectMemoryRecord
    executable: bool = False

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("project_id and run_id are required")
        return clean

    @model_validator(mode="after")
    def validate_non_executable(self) -> RunPlanReviewMemorySave:
        if self.executable is not False:
            raise ValueError("run-plan review memory save is not executable")
        return self


def save_run_plan_review_card_summary_to_memory(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    card: RunPlanReviewCard | dict[str, Any] | None = None,
    confirmed_by: str = "",
) -> RunPlanReviewMemorySave:
    """Save a compact run-plan review summary as a project memory record.

    This function stores only review decisions and artifact references. It does
    not store raw datasets, full artifact contents, markdown bodies, complete
    verifier/proposal payloads, execute proposals, apply patches, call LLMs,
    enqueue work, or mutate a RunPlan.
    """

    review_card = _card_or_read(
        card=card,
        workspace_dir=workspace_dir,
        project_id=project_id,
        run_id=run_id,
    )
    if review_card.project_id != project_id or review_card.run_id != run_id:
        raise ValueError("review card project_id/run_id mismatch")
    record = build_run_plan_review_memory_record(review_card, confirmed_by=confirmed_by)
    saved = ProjectMemory(Path(workspace_dir)).save_project_record(project_id, record)
    return RunPlanReviewMemorySave(
        project_id=project_id,
        run_id=run_id,
        saved=True,
        record=saved,
        executable=False,
    )


def build_run_plan_review_memory_record(
    card: RunPlanReviewCard | dict[str, Any],
    *,
    confirmed_by: str = "",
) -> ProjectMemoryRecord:
    review_card = card if isinstance(card, RunPlanReviewCard) else RunPlanReviewCard.model_validate(card)
    artifact_refs = {
        artifact_id: relative_path
        for artifact_id, relative_path in review_card.artifacts.items()
        if artifact_id in {"observer_verification", "replan_proposal", "replan_review_markdown"}
    }
    source_refs = [f"run:{review_card.run_id}:artifact:{artifact_id}" for artifact_id in artifact_refs]
    value = {
        "kind": "run_plan_review_summary",
        "project_id": review_card.project_id,
        "run_id": review_card.run_id,
        "verifier_decision": review_card.verifier_decision,
        "proposed_action": review_card.proposed_action,
        "affected_tasks": list(review_card.affected_tasks),
        "required_user_decisions": list(review_card.required_user_decisions),
        "artifact_refs": artifact_refs,
        "executable": False,
    }
    return ProjectMemoryRecord(
        record_id=f"run-plan-review-{review_card.run_id}",
        category="run_plan_review",
        summary=_summary(review_card),
        value=value,
        source_refs=source_refs,
        decision="run_plan_review_recorded",
        confirmed_by=str(confirmed_by or "").strip(),
        metadata={
            "source": "run_plan_review_card",
            "content_policy": "summary_and_artifact_refs_only",
        },
    )


def _card_or_read(
    *,
    card: RunPlanReviewCard | dict[str, Any] | None,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
) -> RunPlanReviewCard:
    if card is not None:
        return card if isinstance(card, RunPlanReviewCard) else RunPlanReviewCard.model_validate(card)
    return read_run_plan_review_card(
        workspace_dir=workspace_dir,
        project_id=project_id,
        run_id=run_id,
    )


def _summary(card: RunPlanReviewCard) -> str:
    decision_count = len(card.required_user_decisions)
    return (
        f"Run {card.run_id} review: verifier={card.verifier_decision}; "
        f"proposed_action={card.proposed_action}; required_user_decisions={decision_count}."
    )
