from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai4s_agent.oled_bounded_discovery_session import (
    COMPLETED_TOP_N,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
    create_oled_bounded_discovery_session,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.storage import ProjectStorage
from test_oled_bounded_discovery_session import _spec


def _advance(storage: ProjectStorage, project_id: str, current: object) -> object:
    return advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        expected_revision=current.revision,  # type: ignore[attr-defined]
    )


def _approve(storage: ProjectStorage, project_id: str, current: object) -> object:
    return approve_oled_bounded_discovery_session_gate(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        expected_revision=current.revision,  # type: ignore[attr-defined]
        actor="trajectory-test-reviewer",
    )


def _terminal_single_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStorage, str, object]:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "trajectory-single-round"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _advance(storage, project_id, current)
    assert current.status == COMPLETED_TOP_N  # type: ignore[attr-defined]
    return storage, project_id, current


def _terminal_two_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStorage, str, object]:
    storage = ProjectStorage(tmp_path / "workspace-two")
    project_id = "trajectory-two-round"
    inputs_root = tmp_path / "two"
    inputs_root.mkdir()
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(inputs_root, monkeypatch, target_top_n=4),
    )
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    assert current.status == COMPLETED_TOP_N  # type: ignore[attr-defined]
    return storage, project_id, current


def _write_action_pair(
    actions_root: Path,
    *,
    project_id: str,
    session_id: str,
    completed_revision: int,
) -> tuple[Path, dict[str, object]]:
    identity: dict[str, object] = {
        "request_version": "oled_bounded_discovery_session_action_request.v1",
        "project_id": project_id,
        "session_id": session_id,
        "action": "advance",
        "expected_revision": 0,
        "actor": "",
        "note": "",
        "created_at": "2026-07-24T00:00:00Z",
        "request_nonce": "1" * 32,
    }
    action_id = "oled-session-action-" + _stable_hash(identity)
    base = {**identity, "action_id": action_id}
    request = {**base, "request_digest": "sha256:" + _stable_hash(base)}
    state: dict[str, object] = {
        "state_version": "oled_bounded_discovery_session_action_state.v2",
        "action_id": action_id,
        "project_id": project_id,
        "status": "SUCCEEDED",
        "updated_at": "2026-07-24T00:00:01Z",
        "instance_id": "historical-worker",
        "request_digest": request["request_digest"],
        "completed_revision": completed_revision,
        "error": None,
    }
    action_dir = actions_root / project_id / action_id
    action_dir.mkdir(parents=True)
    (action_dir / "request.json").write_bytes(_json_bytes(request))
    (action_dir / "action.json").write_bytes(_json_bytes(state))
    return action_dir, state


def test_projection_rejects_nonterminal_session_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "trajectory-active"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    output_root = tmp_path / "projections"

    with pytest.raises(ValueError, match="only projects terminal"):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
            actions_root=tmp_path / "actions",
            output_root=output_root,
        )

    assert not output_root.exists()


def test_terminal_projection_is_deterministic_observer_only_and_path_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    session_before = {
        path.relative_to(current.session_dir): path.read_bytes()  # type: ignore[attr-defined]
        for path in current.session_dir.rglob("*")  # type: ignore[attr-defined]
        if path.is_file()
    }
    registries_before = {
        path: path.read_bytes()
        for path in storage.project_dir(project_id).glob("runs/*/artifact_registry.json")
    }

    first = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-a",
    )
    second = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-b",
    )

    assert first.trajectory_id == second.trajectory_id
    assert first.publication_id == second.publication_id
    assert sorted(path.name for path in first.output_dir.iterdir()) == [
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
        "trajectory.json",
    ]
    for name in (
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
        "trajectory.json",
    ):
        assert (first.output_dir / name).read_bytes() == (second.output_dir / name).read_bytes()
        assert str(tmp_path).encode() not in (first.output_dir / name).read_bytes()

    receipt = json.loads(first.receipt_json.read_text(encoding="utf-8"))
    assert receipt["claims"] == {
        "counterfactual_alternatives_invented": False,
        "mutable_telemetry_authoritative": False,
        "observer_only": True,
        "post_hoc_projection": True,
        "private_chain_of_thought_recorded": False,
        "scientific_execution_modified": False,
        "scientific_trust_anchor_created": False,
    }
    events = [json.loads(line) for line in first.events_jsonl.read_text().splitlines()]
    assert [item["sequence_index"] for item in events] == list(range(len(events)))
    assert events[-1]["event_kind"] == "terminal_result_committed"
    authorizations = [
        item for item in events if item["event_kind"] == "action_authorized"
    ]
    assert len(authorizations) == 2
    assert all(
        item["source"]["logical_role"] == "gate_decision"
        and item["outcome"]["approved"] is True
        for item in authorizations
    )
    assert {
        path.relative_to(current.session_dir): path.read_bytes()  # type: ignore[attr-defined]
        for path in current.session_dir.rglob("*")  # type: ignore[attr-defined]
        if path.is_file()
    } == session_before
    assert {
        path: path.read_bytes()
        for path in storage.project_dir(project_id).glob("runs/*/artifact_registry.json")
    } == registries_before


def test_multi_round_projection_contains_cumulative_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_two_rounds(tmp_path, monkeypatch)

    result = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection",
    )

    events = [json.loads(line) for line in result.events_jsonl.read_text().splitlines()]
    dispatched = {
        item["child_run_id"]
        for item in events
        if item["event_kind"] == "task_dispatched"
    }
    assert any(str(run_id).endswith("evaluation-02") for run_id in dispatched)
    assert sum(str(run_id).endswith(("generation-01", "generation-02")) for run_id in dispatched) == 2
    receipt = json.loads(result.receipt_json.read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == COMPLETED_TOP_N


def test_mutable_action_telemetry_changes_findings_not_scientific_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    action_dir, state = _write_action_pair(
        tmp_path / "actions",
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )
    first = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-good-telemetry",
    )
    assert first.telemetry_findings_jsonl.read_bytes() == b""

    contradictory = {
        **state,
        "status": "RUNNING",
        "updated_at": "2026-07-24T00:00:02Z",
        "completed_revision": None,
    }
    (action_dir / "action.json").write_bytes(_json_bytes(contradictory))
    second = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-conflicting-telemetry",
    )

    assert first.trajectory_id == second.trajectory_id
    assert first.events_jsonl.read_bytes() == second.events_jsonl.read_bytes()
    assert first.source_bindings_json.read_bytes() == second.source_bindings_json.read_bytes()
    assert first.publication_id != second.publication_id
    finding = json.loads(second.telemetry_findings_jsonl.read_text().strip())
    assert finding["reason_code"] == "telemetry_conflicts_with_session_history"
    assert finding["authority_effect"] == "ignored_for_scientific_facts"


def test_immutable_action_request_tamper_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    action_dir, _ = _write_action_pair(
        tmp_path / "actions",
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )
    request = json.loads((action_dir / "request.json").read_text(encoding="utf-8"))
    request["session_id"] = "forged-session"
    (action_dir / "request.json").write_bytes(_json_bytes(request))
    output_root = tmp_path / "projection"

    with pytest.raises(ValueError):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions",
            output_root=output_root,
        )

    assert not output_root.exists()


def test_projection_publication_is_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    output_root = tmp_path / "projection"
    first = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=output_root,
    )
    before = {
        path.name: path.read_bytes() for path in first.output_dir.iterdir()
    }

    with pytest.raises(ValueError, match="already exists"):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions",
            output_root=output_root,
        )

    assert {path.name: path.read_bytes() for path in first.output_dir.iterdir()} == before


def test_authoritative_child_state_conflict_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    run_id = terminal["children"][0]["run_id"]
    stage_path = storage.run_dir(project_id, run_id) / "stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["status"] = "FAILED"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    output_root = tmp_path / "projection"

    with pytest.raises(ValueError):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions",
            output_root=output_root,
        )

    assert not output_root.exists()


def test_same_terminal_inputs_are_byte_identical_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    local = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-local",
    )
    script = """
import json
import sys
from pathlib import Path
from ai4s_agent.oled_scientific_agent_trajectory_projection import publish_oled_scientific_agent_trajectory_projection
from ai4s_agent.storage import ProjectStorage
result = publish_oled_scientific_agent_trajectory_projection(
    storage=ProjectStorage(Path(sys.argv[1])),
    project_id=sys.argv[2],
    session_id=sys.argv[3],
    actions_root=Path(sys.argv[4]),
    output_root=Path(sys.argv[5]),
)
print(json.dumps({"trajectory_id": result.trajectory_id, "publication_id": result.publication_id}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(storage.workspace_dir),
            project_id,
            current.session_id,  # type: ignore[attr-defined]
            str(tmp_path / "actions"),
            str(tmp_path / "projection-subprocess"),
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src:."},
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(completed.stdout)
    remote_dir = tmp_path / "projection-subprocess" / identity["publication_id"]

    assert identity == {
        "trajectory_id": local.trajectory_id,
        "publication_id": local.publication_id,
    }
    assert {
        path.name: path.read_bytes() for path in local.output_dir.iterdir()
    } == {path.name: path.read_bytes() for path in remote_dir.iterdir()}
