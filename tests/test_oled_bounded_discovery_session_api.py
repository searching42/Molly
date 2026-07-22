from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

import ai4s_agent.oled_bounded_discovery_session_actions as action_module
from ai4s_agent.app import create_app
from ai4s_agent.oled_bounded_discovery_session import (
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
    create_oled_bounded_discovery_session,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    build_oled_bounded_discovery_session_view,
)
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_bounded_discovery_session import _spec


def _poll_action(client: Any, project_id: str, action_id: str) -> dict[str, Any]:
    for _ in range(400):
        response = client.get(
            f"/api/projects/{project_id}/oled-bounded-session-actions/{action_id}"
        )
        assert response.status_code == 200
        action = response.get_json()["action"]
        if action["status"] in {"SUCCEEDED", "FAILED", "RECOVERY_REQUIRED"}:
            return action
        time.sleep(0.01)
    raise AssertionError("bounded-session action did not reach a terminal state")


def test_bounded_session_api_creates_advances_and_approves_without_blocking_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "api-bounded-session"

    created = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions",
        json={"session_spec": _spec(tmp_path, monkeypatch, target_top_n=1)},
    )
    assert created.status_code == 201
    session = created.get_json()["session"]
    assert (session["status"], session["current_step"], session["revision"]) == (
        "ACTIVE",
        "screening",
        0,
    )
    assert session["claims"]["experimental_validation_claimed"] is False
    assert "session_dir" not in session

    queued = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}/actions/advance",
        json={"expected_revision": 0},
    )
    assert queued.status_code == 202
    action = _poll_action(
        client, project_id, queued.get_json()["action"]["action_id"]
    )
    assert action["status"] == "SUCCEEDED"
    waiting = action["result"]
    assert waiting["status"] == "WAITING_USER"
    assert waiting["gate"]["required"] is True

    # The mutable action record is not allowed to become a candidate-result
    # trust anchor.  A fully rewritten stored result is ignored and the API
    # exact-replays the authoritative PR-AV session instead.
    action_id = queued.get_json()["action"]["action_id"]
    action_path = (
        tmp_path
        / "runs"
        / "oled-bounded-session-actions"
        / project_id
        / action_id
        / "action.json"
    )
    persisted = json.loads(action_path.read_text(encoding="utf-8"))
    persisted["result"] = {
        "status": "COMPLETED_TOP_N",
        "terminal": {"top_candidates": [{"candidate_id": "forged"}]},
    }
    action_path.write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    replayed = client.get(
        f"/api/projects/{project_id}/oled-bounded-session-actions/{action_id}"
    ).get_json()["action"]
    assert replayed["result"]["status"] == "WAITING_USER"
    assert replayed["result"]["terminal"] is None

    stale = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}/actions/advance",
        json={"expected_revision": 0},
    )
    assert stale.status_code == 409
    assert "revision conflict" in stale.get_json()["error"]

    missing_actor = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}/actions/approve",
        json={"expected_revision": waiting["revision"]},
    )
    assert missing_actor.status_code == 400

    approved = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}/actions/approve",
        json={
            "expected_revision": waiting["revision"],
            "actor": "api-reviewer",
            "note": "approve exact-bound screening",
        },
    )
    assert approved.status_code == 202
    approval = _poll_action(
        client, project_id, approved.get_json()["action"]["action_id"]
    )
    assert approval["status"] == "SUCCEEDED"
    assert approval["result"]["status"] == "ACTIVE"
    assert approval["result"]["current_step"] == "initial_decision"


def test_bounded_session_api_rejects_duplicate_active_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=workspace)
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "duplicate-action"
    created = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions",
        json={"session_spec": _spec(tmp_path, monkeypatch, target_top_n=1)},
    ).get_json()["session"]
    started = threading.Event()
    release = threading.Event()
    original = action_module.advance_oled_bounded_discovery_session

    def blocked_advance(**kwargs: Any) -> Any:
        started.set()
        assert release.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(
        action_module,
        "advance_oled_bounded_discovery_session",
        blocked_advance,
    )
    try:
        first = client.post(
            f"/api/projects/{project_id}/oled-bounded-sessions/{created['session_id']}/actions/advance",
            json={"expected_revision": 0},
        )
        assert first.status_code == 202
        assert started.wait(timeout=2)
        second = client.post(
            f"/api/projects/{project_id}/oled-bounded-sessions/{created['session_id']}/actions/advance",
            json={"expected_revision": 0},
        )
        assert second.status_code == 409
        assert "already has an active" in second.get_json()["error"]
    finally:
        release.set()
    assert _poll_action(
        client, project_id, first.get_json()["action"]["action_id"]
    )["status"] == "SUCCEEDED"


def test_interrupted_action_is_reported_recovery_required_and_not_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "interrupted-action"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    holding = _HoldingExecutor()
    root = tmp_path / "actions"
    first = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=root,
        executor=holding,  # type: ignore[arg-type]
    )
    queued = first.enqueue_advance(
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=0,
    )
    second = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=root,
    )
    recovered = second.get_action(
        project_id=project_id,
        action_id=queued["action_id"],
    )
    assert recovered["status"] == "RECOVERY_REQUIRED"
    assert recovered["persisted_status"] == "QUEUED"
    assert build_oled_bounded_discovery_session_view(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
    )["revision"] == 0


def test_reconciled_revision_is_not_permanently_blocked_by_old_action_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "reconciled-action"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    root = tmp_path / "actions"
    interrupted = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=root,
        executor=_HoldingExecutor(),  # type: ignore[arg-type]
    )
    interrupted.enqueue_advance(
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=0,
    )

    # Model an externally reconciled durable transition.  The old action is
    # still useful history, but cannot lock revision 1 forever.
    waiting = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=0,
    )
    restarted = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=root,
        executor=_HoldingExecutor(),  # type: ignore[arg-type]
    )
    approval = restarted.enqueue_approval(
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=waiting.revision,
        actor="recovery-reviewer",
    )
    assert approval["status"] == "QUEUED"


def test_completed_session_view_presents_exact_replayed_top_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "presented-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    current = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
    )
    current = approve_oled_bounded_discovery_session_gate(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        actor="view-reviewer",
    )
    current = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
    )
    current = approve_oled_bounded_discovery_session_gate(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        actor="view-reviewer",
    )
    current = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
    )
    view = build_oled_bounded_discovery_session_view(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
    )
    assert view["status"] == "COMPLETED_TOP_N"
    assert view["terminal"]["result_source"] == "pr_arb_v1"
    assert len(view["terminal"]["top_candidates"]) == 1
    assert view["terminal"]["top_candidates"][0]["source_kind"] == "registry"
    assert view["claims"] == {
        "recommendation_only": True,
        "experimental_validation_claimed": False,
        "computational_validation_claimed": False,
        "registry_mutated": False,
        "human_candidate_adjudication_performed": False,
    }


def test_bounded_session_page_has_safe_async_controls_and_top_n_surface(
    tmp_path: Path,
) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    response = client.get("/oled-bounded-sessions")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "OLED 有界发现会话" in html
    assert 'id="advance-button"' in html
    assert 'id="approval-form"' in html
    assert 'id="top-n-panel"' in html
    assert "pollAction" in html
    assert "escapeHTML" in html
    assert "accept" not in html.lower()
    assert "defer" not in html.lower()
    assert "reject" not in html.lower()


class _HoldingExecutor:
    def submit(self, function: Any, *args: Any) -> Future[Any]:
        del function, args
        return Future()
