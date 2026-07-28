from __future__ import annotations

import json
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

import ai4s_agent.oled_bounded_discovery_session as session_module
from ai4s_agent.app import create_app
from ai4s_agent.control_plane_events import ControlPlaneEventProjector
from ai4s_agent.oled_bounded_discovery_session import (
    create_oled_bounded_discovery_session,
    inspect_oled_bounded_discovery_session,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_bounded_discovery_session import (
    _CountingExecutor,
    _advance_with_executor,
    _approve_with_executor,
    _spec,
)


class _HoldingExecutor:
    def __init__(self) -> None:
        self.future: Future[Any] = Future()

    def submit(self, *args: Any, **kwargs: Any) -> Future[Any]:
        return self.future


def _projector(tmp_path: Path) -> tuple[
    ProjectStorage,
    OledBoundedDiscoverySessionActionService,
    ControlPlaneEventProjector,
]:
    storage = ProjectStorage(tmp_path / "workspace")
    actions = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=tmp_path / "runs" / "oled-bounded-session-actions",
        executor=_HoldingExecutor(),  # type: ignore[arg-type]
    )
    projector = ControlPlaneEventProjector(
        storage=storage,
        actions=actions,
        events_root=tmp_path / "runs" / "control-plane-event-projections",
    )
    return storage, actions, projector


def _session_tree_bytes(session_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(session_dir)): path.read_bytes()
        for path in sorted(session_dir.rglob("*"))
        if path.is_file()
    }


def _exercise_snapshot_sse_and_repeat(client: Any, base: str) -> None:
    snapshot = client.get(base)
    assert snapshot.status_code == 200
    stream = client.get(base + "/events?once=1")
    assert stream.status_code == 200
    assert "event: snapshot" in stream.get_data(as_text=True)
    repeated = client.get(base)
    assert repeated.status_code == 200


def test_projector_replays_monotonic_durable_events_without_advancing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, actions, projector = _projector(tmp_path)
    project_id = "projector-replay"
    created = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )

    initial = projector.project(
        project_id=project_id,
        session_id=created.session_id,
    )
    assert initial["snapshot"]["revision"] == 0
    assert initial["authority"]["projector_is_authoritative"] is False
    assert initial["authority"]["events_may_drive_execution"] is False
    assert [(item["event_id"], item["event_type"]) for item in initial["durable_events"]] == [
        (1, "session.created")
    ]

    repeated = projector.project(
        project_id=project_id,
        session_id=created.session_id,
        after_event_id=1,
    )
    assert repeated["durable_events"] == []
    assert repeated["cursor"]["latest_event_id"] == 1

    queued = actions.enqueue_advance(
        project_id=project_id,
        session_id=created.session_id,
        expected_revision=0,
    )
    replay = projector.project(
        project_id=project_id,
        session_id=created.session_id,
        after_event_id=1,
    )
    assert [(item["event_id"], item["event_type"]) for item in replay["durable_events"]] == [
        (2, "action.queued")
    ]
    assert replay["durable_events"][0]["data"]["action_id"] == queued["action_id"]
    assert inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=created.session_id,
    ).revision == 0


def test_projector_uses_newline_commit_marker_and_fails_closed_on_complete_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _, projector = _projector(tmp_path)
    project_id = "projector-journal"
    created = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    first = projector.project(project_id=project_id, session_id=created.session_id)
    journal = (
        tmp_path
        / "runs"
        / "control-plane-event-projections"
        / project_id
        / created.session_id
        / "events.jsonl"
    )
    committed = journal.read_bytes()
    journal.write_bytes(committed.removesuffix(b"\n"))
    regenerated = projector.project(project_id=project_id, session_id=created.session_id)
    assert [(item["event_id"], item["event_type"]) for item in regenerated["durable_events"]] == [
        (1, "session.created")
    ]
    assert journal.read_bytes().endswith(b"\n")
    committed = journal.read_bytes()
    journal.write_bytes(committed + b'{"partial"')

    recovered = projector.project(
        project_id=project_id,
        session_id=created.session_id,
        after_event_id=first["cursor"]["latest_event_id"],
    )
    assert recovered["durable_events"] == []
    assert journal.read_bytes() == committed

    journal.write_bytes(committed + b"{not-json}\n")
    with pytest.raises(ValueError, match="corrupt record"):
        projector.project(project_id=project_id, session_id=created.session_id)


def test_projector_derives_gate_stage_and_artifact_events_from_immutable_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _, projector = _projector(tmp_path)
    project_id = "projector-session-facts"
    created = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    initial = projector.project(project_id=project_id, session_id=created.session_id)
    executor = _CountingExecutor(storage)

    waiting = _advance_with_executor(storage, project_id, created, executor)
    waiting_projection = projector.project(
        project_id=project_id,
        session_id=created.session_id,
        after_event_id=initial["cursor"]["latest_event_id"],
    )
    waiting_types = [item["event_type"] for item in waiting_projection["durable_events"]]
    assert "session.stage_changed" in waiting_types
    assert "gate.waiting" in waiting_types

    active = _approve_with_executor(storage, project_id, waiting, executor)
    approved_projection = projector.project(
        project_id=project_id,
        session_id=created.session_id,
        after_event_id=waiting_projection["cursor"]["latest_event_id"],
    )
    approved_types = [item["event_type"] for item in approved_projection["durable_events"]]
    assert "gate.approved" in approved_types
    assert "artifact.registered" in approved_types
    assert approved_projection["snapshot"]["revision"] == active.revision


def test_event_projection_api_supports_snapshot_last_event_id_and_ephemeral_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "projector-api"
    created = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions",
        json={"session_spec": _spec(tmp_path, monkeypatch, target_top_n=1)},
    )
    assert created.status_code == 201
    session = created.get_json()["session"]
    base = (
        f"/api/projects/{project_id}/oled-bounded-sessions/"
        f"{session['session_id']}/event-projection"
    )

    projection = client.get(base)
    assert projection.status_code == 200
    payload = projection.get_json()
    assert payload["snapshot"]["revision"] == 0
    assert payload["durable_events"][0]["event_type"] == "session.created"
    assert projection.headers["Cache-Control"] == "no-store"

    stream = client.get(base + "/events?once=1", headers={"Last-Event-ID": "0"})
    assert stream.status_code == 200
    assert stream.mimetype == "text/event-stream"
    body = stream.get_data(as_text=True)
    assert "event: snapshot\n" in body
    assert "id: 1\nevent: session.created\n" in body
    heartbeat = next(block for block in body.split("\n\n") if "event: heartbeat" in block)
    assert "id:" not in heartbeat
    heartbeat_payload = json.loads(
        next(line.removeprefix("data: ") for line in heartbeat.splitlines() if line.startswith("data: "))
    )
    assert heartbeat_payload["durable"] is False

    resumed = client.get(base + "/events?once=1", headers={"Last-Event-ID": "1"})
    resumed_body = resumed.get_data(as_text=True)
    assert "event: session.created" not in resumed_body
    assert "event: snapshot" in resumed_body
    assert "event: heartbeat" in resumed_body

    unchanged = client.get(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}"
    ).get_json()["session"]
    assert unchanged["revision"] == 0

    queued = client.post(
        f"/api/projects/{project_id}/oled-bounded-sessions/{session['session_id']}/actions/advance",
        json={"expected_revision": 0},
    )
    assert queued.status_code == 202
    action_id = queued.get_json()["action"]["action_id"]
    for _ in range(1000):
        action = client.get(
            f"/api/projects/{project_id}/oled-bounded-session-actions/{action_id}"
        ).get_json()["action"]
        if action["status"] in {"SUCCEEDED", "FAILED", "RECOVERY_REQUIRED"}:
            break
        time.sleep(0.01)
    assert action["status"] == "SUCCEEDED"

    after_action = client.get(base, headers={"Last-Event-ID": "1"}).get_json()
    action_types = [item["event_type"] for item in after_action["durable_events"]]
    assert "action.queued" in action_types
    assert "action.running" in action_types
    assert "action.succeeded" in action_types
    assert "gate.waiting" in action_types


def test_snapshot_and_sse_do_not_reconcile_child_fact_ahead_of_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    projector = app.extensions["control_plane_event_projector"]
    storage = projector.storage
    project_id = "projector-read-only-recovery-window"
    created = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    executor = _CountingExecutor(storage)
    waiting = _advance_with_executor(storage, project_id, created, executor)

    def stop_before_session_revision(path: Path) -> None:
        raise SystemExit(f"simulated exit before publishing {path.name}")

    monkeypatch.setattr(
        session_module,
        "_REVISION_PUBLISH_FAULT_HOOK",
        stop_before_session_revision,
    )
    with pytest.raises(SystemExit, match="simulated exit"):
        _approve_with_executor(storage, project_id, waiting, executor)

    session_dir = waiting.session_dir
    session_tree_before = _session_tree_bytes(session_dir)
    action_root = tmp_path / "runs" / "oled-bounded-session-actions" / project_id
    action_files_before = (
        {
            str(path.relative_to(action_root)): path.read_bytes()
            for path in action_root.rglob("*")
            if path.is_file()
        }
        if action_root.exists()
        else {}
    )
    assert not (session_dir / "session_result.json").exists()

    base = (
        f"/api/projects/{project_id}/oled-bounded-sessions/"
        f"{waiting.session_id}/event-projection"
    )
    snapshot = client.get(base)
    assert snapshot.status_code == 200
    payload = snapshot.get_json()
    assert payload["snapshot"]["revision"] == waiting.revision
    assert payload["snapshot"]["status"] == "WAITING_USER"
    assert payload["snapshot"]["reconciliation"] == {
        "available": True,
        "session_revision": waiting.revision,
        "run_id": waiting.waiting_run_id,
        "task_id": waiting.waiting_task_id,
        "session_child_status": "waiting_user",
        "observed_stage_status": "SUCCEEDED",
        "requires_explicit_recovery": True,
    }
    assert "session.reconciliation_available" in {
        item["event_type"] for item in payload["durable_events"]
    }

    stream = client.get(base + "/events?once=1")
    assert stream.status_code == 200
    assert "event: session.reconciliation_available" in stream.get_data(as_text=True)
    repeated = client.get(base)
    assert repeated.status_code == 200
    assert repeated.get_json()["snapshot"]["revision"] == waiting.revision

    assert _session_tree_bytes(session_dir) == session_tree_before
    assert not (session_dir / "session_result.json").exists()
    action_files_after = (
        {
            str(path.relative_to(action_root)): path.read_bytes()
            for path in action_root.rglob("*")
            if path.is_file()
        }
        if action_root.exists()
        else {}
    )
    assert action_files_after == action_files_before


def test_projection_does_not_materialize_pending_controller_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    storage = app.extensions["control_plane_event_projector"].storage
    project_id = "projector-no-controller-materialization"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=4),
    )
    executor = _CountingExecutor(storage)
    for _ in range(3):
        current = _approve_with_executor(
            storage,
            project_id,
            _advance_with_executor(storage, project_id, current, executor),
            executor,
        )
    current = _advance_with_executor(storage, project_id, current, executor)
    current = _advance_with_executor(storage, project_id, current, executor)
    assert (current.status, current.current_step) == ("ACTIVE", "controller")
    controller_request = current.session_dir / "controller_request_01.json"
    assert not controller_request.exists()
    session_tree_before = _session_tree_bytes(current.session_dir)

    base = (
        f"/api/projects/{project_id}/oled-bounded-sessions/"
        f"{current.session_id}/event-projection"
    )
    _exercise_snapshot_sse_and_repeat(client, base)

    assert not controller_request.exists()
    assert _session_tree_bytes(current.session_dir) == session_tree_before


def test_projection_does_not_materialize_second_round_generation_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    storage = app.extensions["control_plane_event_projector"].storage
    project_id = "projector-no-roster-materialization"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=4),
    )
    executor = _CountingExecutor(storage)
    for _ in range(3):
        current = _approve_with_executor(
            storage,
            project_id,
            _advance_with_executor(storage, project_id, current, executor),
            executor,
        )
    current = _advance_with_executor(storage, project_id, current, executor)
    current = _advance_with_executor(storage, project_id, current, executor)
    current = _advance_with_executor(storage, project_id, current, executor)
    assert (current.status, current.current_step) == ("ACTIVE", "generation")
    current = _approve_with_executor(
        storage,
        project_id,
        _advance_with_executor(storage, project_id, current, executor),
        executor,
    )
    assert (current.status, current.current_step) == ("ACTIVE", "evaluation")
    generation_roster = current.session_dir / "generation_roster_02.json"
    assert not generation_roster.exists()
    session_tree_before = _session_tree_bytes(current.session_dir)

    base = (
        f"/api/projects/{project_id}/oled-bounded-sessions/"
        f"{current.session_id}/event-projection"
    )
    _exercise_snapshot_sse_and_repeat(client, base)

    assert not generation_roster.exists()
    assert _session_tree_bytes(current.session_dir) == session_tree_before


def test_bounded_session_ui_connects_to_read_only_event_projection(tmp_path: Path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    html = app.test_client().get("/oled-bounded-sessions").get_data(as_text=True)
    assert "new EventSource" in html
    assert "/event-projection/events" in html
    assert 'id="event-cursor"' in html
    assert "heartbeat 不占 cursor" in html
