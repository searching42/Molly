from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.app import create_app
from ai4s_agent.run_plan_review_artifacts import write_run_plan_review_artifacts
from ai4s_agent.run_plan_review_card import RunPlanReviewCard, read_run_plan_review_card
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


PERMISSION_ACTION = "run_plan_queue_execute"


def _grant_run_plan_queue_permission(
    workspace: Path,
    *,
    project_id: str = "proj-card",
    run_id: str = "run-card",
    actor: str = "admin",
) -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        project_id,
        PERMISSION_ACTION,
        actor=actor,
        actor_source="test",
        run_id=run_id,
        reason="test grant",
    )


def _write_review_artifacts(workspace: Path, *, project_id: str = "proj-card", run_id: str = "run-card") -> None:
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    metrics_path = write_json(
        run_dir / "metrics" / "model_metrics.json",
        {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.23, "mae": 0.35}}]},
    )
    storage.register_artifact_path(project_id, run_id, "model_metrics", metrics_path.relative_to(run_dir).as_posix())
    write_run_plan_review_artifacts(workspace_dir=workspace, project_id=project_id, run_id=run_id)


def test_read_run_plan_review_card_aggregates_review_artifacts(tmp_path: Path) -> None:
    _write_review_artifacts(tmp_path)

    card = read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-card", run_id="run-card")

    assert isinstance(card, RunPlanReviewCard)
    assert card.project_id == "proj-card"
    assert card.run_id == "run-card"
    assert card.verifier_decision == "rerun_recommended"
    assert card.proposed_action == "rerun_task"
    assert card.executable is False
    assert card.proposal["executable"] is False
    assert card.proposal["proposed_run_plan_patch"]["applied"] is False
    assert card.required_user_decisions
    assert card.review_markdown.startswith("# Run Plan Review")
    assert card.artifacts == {
        "observer_verification": "review/observer_verification.json",
        "replan_proposal": "review/replan_proposal.json",
        "replan_review_markdown": "review/replan_review.md",
    }


def test_read_run_plan_review_card_rejects_missing_artifacts(tmp_path: Path) -> None:
    ProjectStorage(tmp_path).run_dir("proj-card", "run-card")

    try:
        read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-card", run_id="run-card")
    except FileNotFoundError as exc:
        assert "observer_verification" in str(exc)
    else:
        raise AssertionError("missing review artifacts should fail closed")


def test_read_run_plan_review_card_does_not_rewrite_review_artifacts(tmp_path: Path) -> None:
    _write_review_artifacts(tmp_path)
    run_dir = ProjectStorage(tmp_path).run_dir("proj-card", "run-card")
    proposal_path = run_dir / "review" / "replan_proposal.json"
    before = proposal_path.read_text(encoding="utf-8")

    read_run_plan_review_card(workspace_dir=tmp_path, project_id="proj-card", run_id="run-card")

    assert proposal_path.read_text(encoding="utf-8") == before


def test_internal_run_plan_review_card_route_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.get("/api/internal/run-plan/review-card?project_id=proj-card&run_id=run-card")

    assert response.status_code == 404


def test_internal_run_plan_review_card_route_requires_actor_and_permission(tmp_path: Path) -> None:
    _write_review_artifacts(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    client = app.test_client()

    missing_actor = client.get("/api/internal/run-plan/review-card?project_id=proj-card&run_id=run-card")
    assert missing_actor.status_code == 403
    assert missing_actor.get_json()["error"]["type"] == "validation_error"

    missing_permission = client.get(
        "/api/internal/run-plan/review-card?project_id=proj-card&run_id=run-card",
        headers={"X-Actor": "review-user"},
    )
    assert missing_permission.status_code == 403
    assert missing_permission.get_json()["error"]["type"] == "permission_denied"


def test_internal_run_plan_review_card_route_returns_card_without_executor(tmp_path: Path) -> None:
    _write_review_artifacts(tmp_path)
    _grant_run_plan_queue_permission(tmp_path)
    calls: list[dict[str, Any]] = []
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = lambda storage: calls.append({"storage": storage})
    client = app.test_client()

    response = client.get(
        "/api/internal/run-plan/review-card?project_id=proj-card&run_id=run-card",
        headers={"X-Actor": "review-user"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    card = RunPlanReviewCard.model_validate(payload["card"])
    assert card.verifier_decision == "rerun_recommended"
    assert card.proposed_action == "rerun_task"
    assert card.executable is False
    assert payload["permission"]["allowed"] is True
    assert calls == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues" / "proj-card" / "run-card" / "worker_queue.json").exists()


def test_internal_run_plan_review_card_route_rejects_path_components(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    client = app.test_client()

    response = client.get(
        "/api/internal/run-plan/review-card?project_id=../outside&run_id=run-card",
        headers={"X-Actor": "review-user"},
    )

    assert response.status_code == 400
    assert "project_id must be a safe path component" in response.get_json()["error"]["message"]


def test_run_plan_review_card_schema_rejects_executable_true() -> None:
    payload = {
        "project_id": "proj-card",
        "run_id": "run-card",
        "generated_at": "2026-06-24T00:00:00Z",
        "verifier_decision": "continue",
        "proposed_action": "continue",
        "summary": "ok",
        "required_user_decisions": [],
        "affected_tasks": [],
        "artifacts": {},
        "verification": {},
        "proposal": {},
        "review_markdown": "# Review\n",
        "executable": True,
    }

    try:
        RunPlanReviewCard.model_validate(payload)
    except ValueError as exc:
        assert "review card is read-only" in str(exc)
    else:
        raise AssertionError("review card executable=true should be rejected")
