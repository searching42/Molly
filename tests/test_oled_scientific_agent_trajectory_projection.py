from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai4s_agent import adapters
import ai4s_agent.oled_scientific_agent_trajectory_projection as projection_module
from ai4s_agent.oled_bounded_discovery_session import (
    COMPLETED_TOP_N,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
    create_oled_bounded_discovery_session,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_scientific_agent_source_evidence import (
    ScientificAgentTypedFailure,
    build_failure_evidence,
    publish_dispatch_receipt,
    publish_recovery_receipt,
    read_dispatch_receipts,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.storage import ProjectStorage
from test_oled_bounded_discovery_session import _spec


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]]:
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path).encode())
        elif path.is_dir():
            snapshot[relative] = ("directory", None)
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        else:
            snapshot[relative] = ("other", None)
    return snapshot


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


@pytest.mark.pr_fast
@pytest.mark.parametrize(
    "reasons",
    (
        ("known_hosts_verification_failed",),
        ("gate_snapshot_mismatch", "ssh_connection_failed"),
    ),
)
def test_typed_stage_failure_reasons_project_from_authoritative_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasons: tuple[str, ...],
) -> None:
    input_root = tmp_path / "typed"
    input_root.mkdir()
    session_spec = _spec(input_root, monkeypatch, target_top_n=1)

    def typed_failure(_: dict[str, object]) -> dict[str, object]:
        raise ScientificAgentTypedFailure(*reasons)

    monkeypatch.setattr(
        adapters,
        "execute_oled_registry_candidate_screening_adapter",
        typed_failure,
    )
    storage = ProjectStorage(tmp_path / "workspace-typed-failure")
    project_id = "trajectory-typed-failure"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=session_spec,
    )
    current = _advance(storage, project_id, current)
    current = _approve(storage, project_id, current)
    assert current.status == "FAILED"  # type: ignore[attr-defined]
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    failed_child = terminal["children"][-1]
    failed_stage_path = storage.run_dir(
        project_id, failed_child["run_id"]
    ) / "stage.json"
    failed_stage = json.loads(failed_stage_path.read_text(encoding="utf-8"))
    persisted_evidence = failed_stage["details"]["failure_evidence"]
    assert persisted_evidence["reason_codes"] == sorted(reasons)
    assert persisted_evidence["causal_link"] is None
    failed_stage["details"]["failure_evidence"] = build_failure_evidence(
        reason_codes=reasons,
        cause_child_run_id=failed_child["run_id"],
    )
    failed_stage_path.write_bytes(_json_bytes(failed_stage))

    result = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions",
        output_root=tmp_path / "projection-typed-failure",
    )
    events = [json.loads(line) for line in result.events_jsonl.read_text().splitlines()]
    failed = next(item for item in events if item["event_kind"] == "stage_failed")
    dispatched = next(
        item for item in events if item["event_kind"] == "task_dispatched"
    )

    assert set(reasons).issubset(failed["reason_codes"])
    assert failed["outcome"]["recovery_disposition"] == "unrecovered"
    assert failed["outcome"]["causal_link"] == {
        "version": "scientific_agent_failure_causal_link.v1",
        "cause_child_run_id": failed_child["run_id"],
    }
    assert dispatched["source"]["logical_role"] == "child_dispatch_receipt"
    assert dispatched["outcome"]["execution_started"] is True
    output = result.events_jsonl.read_bytes() + result.source_bindings_json.read_bytes()
    assert b"known_hosts_path" not in output
    assert str(tmp_path).encode() not in output


@pytest.mark.pr_fast
def test_free_text_known_hosts_exception_stays_tool_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "untyped"
    input_root.mkdir()
    session_spec = _spec(input_root, monkeypatch, target_top_n=1)

    def untyped_failure(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("known_hosts /private/.ssh/known_hosts private.compute.invalid")

    monkeypatch.setattr(
        adapters,
        "execute_oled_registry_candidate_screening_adapter",
        untyped_failure,
    )
    storage = ProjectStorage(tmp_path / "workspace-untyped-failure")
    project_id = "trajectory-untyped-failure"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=session_spec,
    )
    current = _advance(storage, project_id, current)
    current = _approve(storage, project_id, current)
    assert current.status == "FAILED"  # type: ignore[attr-defined]
    result = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions-untyped",
        output_root=tmp_path / "projection-untyped-failure",
    )
    events = [json.loads(line) for line in result.events_jsonl.read_text().splitlines()]
    failed = next(item for item in events if item["event_kind"] == "stage_failed")

    assert "adapter_runtime_failed" in failed["reason_codes"]
    assert "known_hosts_verification_failed" not in failed["reason_codes"]
    assert b"private.compute.invalid" not in result.events_jsonl.read_bytes()


@pytest.mark.pr_fast
def test_distinct_dispatch_receipts_and_causal_link_are_representable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    child = terminal["children"][0]
    run_dir = storage.run_dir(project_id, child["run_id"])
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=child["run_id"],
        task_id=child["task_id"],
        dispatch_kind="duplicate_rejected",
        request_or_stage_digest="sha256:" + "9" * 64,
        attempt_id="8" * 32,
    )
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=child["run_id"],
        task_id=child["task_id"],
        dispatch_kind="idempotent_replay",
        request_or_stage_digest="sha256:" + "7" * 64,
        attempt_id="7" * 32,
    )
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=child["run_id"],
        task_id=child["task_id"],
        dispatch_kind="recovery_adoption",
        request_or_stage_digest="sha256:" + "6" * 64,
        attempt_id="6" * 32,
    )
    result = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=tmp_path / "actions-capability",
        output_root=tmp_path / "projection-capability",
    )
    events = [json.loads(line) for line in result.events_jsonl.read_text().splitlines()]
    dispatches = [
        item
        for item in events
        if item["event_kind"] == "task_dispatched"
        and item["child_run_id"] == child["run_id"]
    ]
    assert len(dispatches) == 2
    assert len({item["source"]["source_artifact_id"] for item in dispatches}) == 2
    assert dispatches[1]["reason_codes"] == ["duplicate_dispatch_detected"]
    assert dispatches[1]["outcome"]["execution_started"] is False
    bindings = json.loads(result.source_bindings_json.read_text(encoding="utf-8"))
    child_receipt_ids = {
        str(item.payload["receipt_id"])
        for item in read_dispatch_receipts(run_dir=run_dir)
    }
    projected_receipt_ids = {
        item["source_artifact_id"]
        for item in bindings["sources"]
        if item["logical_role"] == "child_dispatch_receipt"
    }
    assert child_receipt_ids.issubset(projected_receipt_ids)
    assert len(child_receipt_ids) == 4


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


@pytest.mark.pr_fast
def test_recovery_receipt_is_exact_bound_projection_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions-recovery-receipt"
    action_dir, state = _write_action_pair(
        actions_root,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )
    state["status"] = "RECOVERED"
    (action_dir / "action.json").write_bytes(_json_bytes(state))
    request = json.loads((action_dir / "request.json").read_text(encoding="utf-8"))
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    child = terminal["children"][0]
    run_dir = storage.run_dir(project_id, child["run_id"])
    stage_bytes = (run_dir / "stage.json").read_bytes()
    dispatch_ids = [
        str(item.payload["receipt_id"])
        for item in read_dispatch_receipts(run_dir=run_dir)
    ]
    receipt = publish_recovery_receipt(
        action_dir=action_dir,
        action_id=request["action_id"],
        request_digest=request["request_digest"],
        recovered_child_run_id=child["run_id"],
        recovered_stage_sha256="sha256:" + hashlib.sha256(stage_bytes).hexdigest(),
        source_dispatch_receipt_ids=dispatch_ids,
        expected_revision=request["expected_revision"],
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )
    scientific_before = _tree_snapshot(storage.workspace_dir)
    recovery_before = _tree_snapshot(actions_root)

    result = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        output_root=tmp_path / "projection-recovery-receipt",
    )
    assert _tree_snapshot(storage.workspace_dir) == scientific_before
    assert _tree_snapshot(actions_root) == recovery_before
    bindings = json.loads(result.source_bindings_json.read_text(encoding="utf-8"))
    recovery = [
        item
        for item in bindings["sources"]
        if item["logical_role"] == "action_recovery_receipt"
    ]
    assert recovery == [
        {
            "logical_role": "action_recovery_receipt",
            "source_artifact_id": receipt.payload["receipt_id"],
            "source_publication_id": "scientific_agent_recovery_receipt.v1",
            "sha256": receipt.sha256,
            "manifest_sha256": receipt.sha256,
        }
    ]


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


@pytest.mark.adversarial
@pytest.mark.parametrize("attack", ("content", "full-resign", "extra-roster"))
def test_dispatch_receipt_tampering_fails_before_projection_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    run_dir = storage.run_dir(project_id, terminal["children"][0]["run_id"])
    receipt_root = run_dir / "dispatch-receipts"
    receipt_dir = next(receipt_root.iterdir())
    receipt_path = receipt_dir / "receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if attack == "content":
        payload["execution_started"] = False
        receipt_path.write_bytes(_json_bytes(payload))
    elif attack == "extra-roster":
        (receipt_dir / "unexpected.json").write_bytes(b"{}\n")
    else:
        payload["task_id"] = "execute_oled_experiment_batch_selection"
        identity = {key: value for key, value in payload.items() if key != "receipt_id"}
        new_id = "scientific-agent-dispatch-receipt:" + _stable_hash(identity)
        payload["receipt_id"] = new_id
        receipt_path.write_bytes(_json_bytes(payload))
        receipt_dir.rename(receipt_root / new_id)

    output_root = tmp_path / "projection-receipt-tamper"
    with pytest.raises(ValueError):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions-receipt-tamper",
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.adversarial
@pytest.mark.pr_fast
def test_same_byte_dispatch_receipt_inode_replacement_during_projection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    run_dir = storage.run_dir(project_id, terminal["children"][0]["run_id"])
    receipt_path = next((run_dir / "dispatch-receipts").glob("*/receipt.json"))
    original_recheck = projection_module._recheck_captures

    def replace_then_recheck(captures: object) -> None:
        replacement = receipt_path.with_name("replacement.json")
        replacement.write_bytes(receipt_path.read_bytes())
        os.replace(replacement, receipt_path)
        original_recheck(captures)  # type: ignore[arg-type]

    monkeypatch.setattr(
        projection_module,
        "_recheck_captures",
        replace_then_recheck,
    )
    output_root = tmp_path / "projection-receipt-replacement"
    with pytest.raises(ValueError, match="source changed"):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions-receipt-replacement",
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.adversarial
def test_same_roster_dispatch_directory_replacement_during_projection_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    run_dir = storage.run_dir(project_id, terminal["children"][0]["run_id"])
    receipt_root = run_dir / "dispatch-receipts"
    original_recheck = projection_module._recheck_directory_rosters
    replaced = False

    def replace_before_recheck(
        rosters: list[projection_module._CapturedDirectoryRoster],
    ) -> None:
        nonlocal replaced
        if not replaced:
            replacement = run_dir / "dispatch-receipts-replacement"
            displaced = run_dir / "dispatch-receipts-displaced"
            shutil.copytree(receipt_root, replacement)
            receipt_root.rename(displaced)
            replacement.rename(receipt_root)
            replaced = True
        original_recheck(rosters)

    monkeypatch.setattr(
        projection_module,
        "_recheck_directory_rosters",
        replace_before_recheck,
    )
    output_root = tmp_path / "projection-receipt-directory-replacement"
    with pytest.raises(ValueError, match="directory changed"):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions-directory-replacement",
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


def test_missing_child_run_fails_without_creating_or_changing_workspace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    missing_run_id = terminal["children"][0]["run_id"]
    missing_run_dir = storage.run_dir(project_id, missing_run_id)
    shutil.rmtree(missing_run_dir)
    before = _tree_snapshot(storage.workspace_dir)
    output_root = tmp_path / "projection"

    with pytest.raises(ValueError, match="child run"):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions",
            output_root=output_root,
        )

    assert _tree_snapshot(storage.workspace_dir) == before
    assert not missing_run_dir.exists()
    assert not output_root.exists()


def test_output_root_cannot_overlap_scientific_or_action_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
    _write_action_pair(
        actions_root,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )
    terminal = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    child_run_dir = storage.run_dir(project_id, terminal["children"][0]["run_id"])
    sources = [
        current.session_dir,  # type: ignore[attr-defined]
        child_run_dir,
        actions_root / project_id,
    ]
    before = _tree_snapshot(tmp_path)

    for source in sources:
        with pytest.raises(ValueError, match="overlaps"):
            publish_oled_scientific_agent_trajectory_projection(
                storage=storage,
                project_id=project_id,
                session_id=current.session_id,  # type: ignore[attr-defined]
                actions_root=actions_root,
                output_root=source,
            )

    assert _tree_snapshot(tmp_path) == before


def test_output_symlink_redirect_cannot_create_inside_session_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    redirected = tmp_path / "redirected-source"
    redirected.symlink_to(current.session_dir, target_is_directory=True)  # type: ignore[attr-defined]
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ValueError):
        publish_oled_scientific_agent_trajectory_projection(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=tmp_path / "actions",
            output_root=redirected / "injected-projection",
        )

    assert _tree_snapshot(tmp_path) == before
    assert not (current.session_dir / "injected-projection").exists()  # type: ignore[attr-defined]
