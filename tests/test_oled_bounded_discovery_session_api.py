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
    inspect_oled_bounded_discovery_session,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    build_oled_bounded_discovery_session_view,
)
from ai4s_agent.oled_real_phase1_execution import _stable_hash
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

    # Candidate results are absent from mutable action state. A successful poll
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
    assert "result" not in persisted
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


def test_queued_action_request_rewrite_fails_closed_without_cross_session_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "frozen-action"
    first_spec = _spec(tmp_path, monkeypatch, target_top_n=1)
    first = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=first_spec,
    )
    second_spec = json.loads(json.dumps(first_spec))
    second_spec["candidate_decision"]["target_top_n"] = 2
    second_spec["candidate_decision"]["max_pairwise_tanimoto"] = 1.0
    second = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=second_spec,
    )
    second = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=second.session_id,
        expected_revision=0,
    )
    assert second.status == "WAITING_USER"

    deferred = _DeferredExecutor()
    root = tmp_path / "actions"
    service = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=root,
        executor=deferred,  # type: ignore[arg-type]
    )
    queued = service.enqueue_advance(
        project_id=project_id,
        session_id=first.session_id,
        expected_revision=0,
    )
    action_dir = root / project_id / queued["action_id"]

    # Fully re-sign the immutable-envelope payload to target the waiting second
    # session, and also inject the old control fields into mutable state.
    request_path = action_dir / "request.json"
    forged_request = json.loads(request_path.read_text(encoding="utf-8"))
    forged_request.update(
        {
            "action": "approve",
            "session_id": second.session_id,
            "expected_revision": second.revision,
            "actor": "forged-reviewer",
            "note": "forged approval",
        }
    )
    unsigned = dict(forged_request)
    unsigned.pop("request_digest")
    forged_request["request_digest"] = "sha256:" + _stable_hash(unsigned)
    request_path.write_text(
        json.dumps(forged_request, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    state_path = action_dir / "action.json"
    forged_state = json.loads(state_path.read_text(encoding="utf-8"))
    forged_state.update(
        {
            "action": "approve",
            "session_id": second.session_id,
            "expected_revision": second.revision,
            "actor": "forged-reviewer",
        }
    )
    state_path.write_text(
        json.dumps(forged_state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    deferred.run()
    failed_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert failed_state["status"] == "FAILED"
    assert failed_state["error"]["code"] == "action_request_integrity_failed"
    assert inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=first.session_id,
    ).revision == 0
    unchanged_second = inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=second.session_id,
    )
    assert unchanged_second.revision == second.revision
    assert unchanged_second.status == "WAITING_USER"


@pytest.mark.parametrize("noncanonical", ["canonical-project ", " canonical-project"])
def test_project_id_whitespace_is_rejected_before_cross_project_session_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    noncanonical: str,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    spec = _spec(tmp_path, monkeypatch, target_top_n=1)
    canonical = create_oled_bounded_discovery_session(
        storage=storage,
        project_id="canonical-project",
        session_spec=spec,
    )
    padded = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=noncanonical,
        session_spec=spec,
    )
    assert canonical.session_id == padded.session_id
    service = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=tmp_path / "actions",
        executor=_HoldingExecutor(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="project_id must be canonical"):
        service.enqueue_advance(
            project_id=noncanonical,
            session_id=canonical.session_id,
            expected_revision=0,
        )
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
    )
    encoded_project = noncanonical.replace(" ", "%20")
    response = app.test_client().post(
        f"/api/projects/{encoded_project}/oled-bounded-sessions/"
        f"{canonical.session_id}/actions/advance",
        json={"expected_revision": 0},
    )
    assert response.status_code == 400
    assert "project_id must be canonical" in response.get_json()["error"]
    assert inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id="canonical-project",
        session_id=canonical.session_id,
    ).revision == 0
    assert inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=noncanonical,
        session_id=padded.session_id,
    ).revision == 0


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


class _DeferredExecutor:
    def __init__(self) -> None:
        self.call: tuple[Any, tuple[Any, ...]] | None = None
        self.future: Future[Any] = Future()

    def submit(self, function: Any, *args: Any) -> Future[Any]:
        self.call = (function, args)
        return self.future

    def run(self) -> None:
        assert self.call is not None
        function, args = self.call
        try:
            result = function(*args)
        except Exception as exc:  # pragma: no cover - worker catches expected failure.
            self.future.set_exception(exc)
            raise
        else:
            self.future.set_result(result)
