from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ai4s_agent.oled_scientific_agent_trajectory_audit_metrics as audit_module
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    _prepare_audit_publication_from_verified_bytes,
    publish_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    _canonical_json_bytes,
    _canonical_jsonl_bytes,
    _sha256,
    publish_oled_scientific_agent_trajectory_projection,
)
from test_oled_scientific_agent_trajectory_projection import (
    _terminal_two_rounds,
    _tree_snapshot,
)
from test_oled_scientific_agent_trajectory_verifier import _publication


_SOURCE_NAMES = (
    "events.jsonl",
    "source_bindings.json",
    "telemetry_findings.jsonl",
    "trajectory.json",
)
_AUDIT_NAMES = (
    "audit_findings.jsonl",
    "audit_manifest.json",
    "audit_metrics.json",
    "report.md",
    "source_binding.json",
)


def _source_payloads(publication: Path) -> dict[str, bytes]:
    return {name: (publication / name).read_bytes() for name in _SOURCE_NAMES}


def _audit_json(prepared: object, name: str) -> dict[str, object]:
    payloads = prepared.payloads  # type: ignore[attr-defined]
    return json.loads(payloads[name].decode("utf-8"))


def _audit_findings(prepared: object) -> list[dict[str, object]]:
    payloads = prepared.payloads  # type: ignore[attr-defined]
    return [
        json.loads(line)
        for line in payloads["audit_findings.jsonl"].decode("utf-8").splitlines()
    ]


def _prepare(payloads: dict[str, bytes]) -> object:
    receipt = json.loads(payloads["trajectory.json"])
    return _prepare_audit_publication_from_verified_bytes(
        payloads=payloads,
        verified_trajectory_id=receipt["trajectory_id"],
        verified_publication_id=receipt["publication_id"],
    )


def test_single_round_audit_is_source_bound_observer_only_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    workspace_before = _tree_snapshot(storage.workspace_dir)  # type: ignore[attr-defined]
    projection_before = _source_payloads(publication)

    result = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=publication,
        output_root=tmp_path / "audit",
    )

    assert tuple(sorted(path.name for path in result.output_dir.iterdir())) == _AUDIT_NAMES
    metrics = json.loads(result.audit_metrics_json.read_text(encoding="utf-8"))
    manifest = json.loads(result.audit_manifest_json.read_text(encoding="utf-8"))
    binding = json.loads(result.source_binding_json.read_text(encoding="utf-8"))
    coverage = metrics["metrics"]["provenance_coverage"]
    assert coverage["action"]["status"] == "complete"
    assert coverage["evidence"]["status"] == "complete"
    assert coverage["authorization"]["status"] == "complete"
    assert coverage["observation_to_decision"]["status"] == "complete"
    assert coverage["recovery"]["status"] == "not_applicable"
    assert coverage["terminal"]["status"] == "complete"
    assert metrics["metrics"]["latency"] == {
        "reason_code": "projection_v1_has_no_wall_clock_event_fields",
        "status": "unavailable",
        "value_milliseconds": None,
    }
    assert metrics["metrics"]["wasted_computation"]["status"] == "not_derivable"
    assert metrics["metrics"]["top_n_completion"] == {
        "has_complete_top_n": True,
        "status": "available",
    }
    assert metrics["metrics"]["bounded_search_correct_stop"]["correct_stop"] is True
    assert manifest["claims"] == {
        "context_bound_verified_bytes_consumed": True,
        "counterfactual_alternatives_invented": False,
        "observer_only": True,
        "root_cause_inferred": False,
        "scientific_execution_modified": False,
        "scientific_trust_anchor_created": False,
        "session_or_control_plane_modified": False,
        "trajectory_projection_modified": False,
    }
    assert binding["source_artifacts"] == {
        name: _sha256(payload) for name, payload in sorted(projection_before.items())
    }
    assert binding["verification"] == {
        "context_bound_verified_bytes": True,
        "exact_bytes": True,
        "exact_external_replay": True,
        "exact_file_roster": True,
    }
    assert result.audit_findings_jsonl.read_bytes() == b""
    for path in result.output_dir.iterdir():
        assert str(tmp_path).encode() not in path.read_bytes()
    assert _tree_snapshot(storage.workspace_dir) == workspace_before  # type: ignore[attr-defined]
    assert _source_payloads(publication) == projection_before


def test_multi_round_audit_reports_cumulative_budget_and_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_two_rounds(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
    projected = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        output_root=tmp_path / "projection",
    )

    result = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=projected.output_dir,
        output_root=tmp_path / "audit",
    )

    metrics = json.loads(result.audit_metrics_json.read_text(encoding="utf-8"))[
        "metrics"
    ]
    assert metrics["budget_consumption"] == {
        "generated_candidates": 2,
        "generation_rounds": 2,
        "iterations": 2,
        "reason_code": None,
        "status": "available",
    }
    assert metrics["gate"]["authorization_event_count"] == 4
    assert metrics["gate"]["approved_count"] == 4
    assert metrics["action_outcome"]["succeeded_child_count"] > 4
    assert metrics["retry"]["duplicate_dispatch_count"] == 0
    assert metrics["bounded_search_correct_stop"]["correct_stop"] is True


def test_terminal_failure_verified_bytes_use_terminal_state_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, publication = _publication(tmp_path, monkeypatch)
    payloads = _source_payloads(publication)
    receipt = json.loads(payloads["trajectory.json"])
    events = [json.loads(line) for line in payloads["events.jsonl"].splitlines()]
    terminal = events.pop()
    assert terminal["event_kind"] == "terminal_result_committed"
    final_state = next(
        event for event in reversed(events) if event["event_kind"] == "state_committed"
    )
    final_state["outcome"]["status"] = "FAILED"
    completed = next(
        event for event in reversed(events) if event["event_kind"] == "stage_completed"
    )
    completed["event_kind"] = "stage_failed"
    completed["outcome"]["child_status"] = "failed"
    completed["reason_codes"] = ["failed"]
    receipt["terminal_status"] = "FAILED"
    receipt["counts"]["event_count"] = len(events)
    payloads["events.jsonl"] = _canonical_jsonl_bytes(events)
    receipt["artifacts"]["events.jsonl"] = _sha256(payloads["events.jsonl"])
    payloads["trajectory.json"] = _canonical_json_bytes(receipt)

    prepared = _prepare(payloads)
    metrics = _audit_json(prepared, "audit_metrics.json")["metrics"]

    assert metrics["provenance_coverage"]["terminal"]["status"] == "complete"
    assert metrics["tool_failure"]["stage_failed_event_count"] == 1
    assert metrics["action_outcome"]["failed_child_count"] == 1
    assert metrics["budget_consumption"]["status"] == "unavailable"
    assert metrics["top_n_completion"] == {
        "has_complete_top_n": None,
        "status": "not_applicable",
    }
    assert metrics["bounded_search_correct_stop"]["status"] == "not_applicable"
    assert metrics["wasted_computation"] == {
        "failed_terminal_child_count": 1,
        "reason_code": "projection_v1_has_no_cost_or_reuse_evidence",
        "status": "not_derivable",
        "value": None,
    }


def test_bounded_no_solution_stop_is_evaluated_from_frozen_reason_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, publication = _publication(tmp_path, monkeypatch)
    payloads = _source_payloads(publication)
    receipt = json.loads(payloads["trajectory.json"])
    events = [json.loads(line) for line in payloads["events.jsonl"].splitlines()]
    terminal = events[-1]
    terminal["outcome"].update(
        {
            "status": "STOPPED_BOUNDED_NO_SOLUTION",
            "stop_reason": "max_generation_rounds_reached",
            "has_complete_top_n": False,
        }
    )
    terminal["reason_codes"] = ["max_generation_rounds_reached"]
    receipt["terminal_status"] = "STOPPED_BOUNDED_NO_SOLUTION"
    payloads["events.jsonl"] = _canonical_jsonl_bytes(events)
    receipt["artifacts"]["events.jsonl"] = _sha256(payloads["events.jsonl"])
    payloads["trajectory.json"] = _canonical_json_bytes(receipt)

    metrics = _audit_json(_prepare(payloads), "audit_metrics.json")["metrics"]

    assert metrics["top_n_completion"] == {
        "has_complete_top_n": False,
        "status": "available",
    }
    assert metrics["bounded_search_correct_stop"] == {
        "correct_stop": True,
        "status": "available",
        "stop_reason": "max_generation_rounds_reached",
        "terminal_status": "STOPPED_BOUNDED_NO_SOLUTION",
    }


def test_retry_and_reconciliation_metrics_count_only_explicit_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, publication = _publication(tmp_path, monkeypatch)
    payloads = _source_payloads(publication)
    receipt = json.loads(payloads["trajectory.json"])
    events = [json.loads(line) for line in payloads["events.jsonl"].splitlines()]
    dispatch = next(event for event in events if event["event_kind"] == "task_dispatched")
    events.insert(-1, dict(dispatch))
    recovery_state = next(
        event
        for event in events
        if event["event_kind"] == "state_committed"
        and event["session_revision"] > 0
    )
    recovery_state["outcome"] = {**recovery_state["outcome"], "status": "RECOVERY_REQUIRED"}
    for index, event in enumerate(events):
        event["sequence_index"] = index
    receipt["counts"]["event_count"] = len(events)
    payloads["events.jsonl"] = _canonical_jsonl_bytes(events)
    receipt["artifacts"]["events.jsonl"] = _sha256(payloads["events.jsonl"])
    payloads["trajectory.json"] = _canonical_json_bytes(receipt)

    metrics = _audit_json(_prepare(payloads), "audit_metrics.json")["metrics"]

    assert metrics["retry"]["duplicate_dispatch_count"] == 1
    assert metrics["retry"]["duplicate_child_run_ids"] == [dispatch["child_run_id"]]
    assert metrics["reconciliation"] == {
        "explicit_recovery_required_state_count": 1,
        "inferred_reconciliation_count": 0,
        "session_revisions": [recovery_state["session_revision"]],
    }
    assert metrics["provenance_coverage"]["recovery"] == {
        "coverage_basis_points": 10_000,
        "eligible_event_count": 1,
        "source_bound_event_count": 1,
        "status": "complete",
    }


@pytest.mark.parametrize("attack", ["event_deleted", "count_mismatch", "binding_missing"])
def test_inconsistent_verified_byte_views_produce_source_backed_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _, _, _, _, publication = _publication(tmp_path, monkeypatch)
    payloads = _source_payloads(publication)
    receipt = json.loads(payloads["trajectory.json"])
    events = [json.loads(line) for line in payloads["events.jsonl"].splitlines()]
    if attack == "event_deleted":
        events.pop(1)
        payloads["events.jsonl"] = _canonical_jsonl_bytes(events)
    elif attack == "count_mismatch":
        receipt["counts"]["event_count"] += 1
        payloads["trajectory.json"] = _canonical_json_bytes(receipt)
    else:
        source_payload = json.loads(payloads["source_bindings.json"])
        missing = next(
            event["source"]
            for event in events
            if event["event_kind"] == "task_dispatched"
        )
        source_payload["sources"].remove(missing)
        payloads["source_bindings.json"] = _canonical_json_bytes(source_payload)

    prepared = _prepare(payloads)
    findings = _audit_findings(prepared)
    codes = {finding["reason_code"] for finding in findings}

    if attack == "event_deleted":
        assert "event_count_mismatch" in codes
        assert "source_artifact_digest_mismatch" in codes
    elif attack == "count_mismatch":
        assert "event_count_mismatch" in codes
    else:
        assert "event_source_binding_missing" in codes
        metrics = _audit_json(prepared, "audit_metrics.json")["metrics"]
        assert metrics["provenance_coverage"]["action"]["status"] == "partial"
    assert findings
    assert all(finding["root_cause_claimed"] is False for finding in findings)
    assert all(finding["source_refs"] for finding in findings)
    assert all(
        ref["artifact_name"] in _SOURCE_NAMES
        and ref["sha256"].startswith("sha256:")
        for finding in findings
        for ref in finding["source_refs"]
    )


def test_same_verified_projection_is_byte_identical_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    local = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=publication,
        output_root=tmp_path / "audit-local",
    )
    script = """
import json
import sys
from pathlib import Path
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    publish_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.storage import ProjectStorage
result = publish_oled_scientific_agent_trajectory_audit_metrics(
    storage=ProjectStorage(Path(sys.argv[1])),
    project_id=sys.argv[2],
    session_id=sys.argv[3],
    actions_root=Path(sys.argv[4]),
    trajectory_publication_dir=Path(sys.argv[5]),
    output_root=Path(sys.argv[6]),
)
print(json.dumps({"audit_id": result.audit_id, "publication_id": result.publication_id}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(storage.workspace_dir),  # type: ignore[attr-defined]
            project_id,
            current.session_id,  # type: ignore[attr-defined]
            str(actions_root),
            str(publication),
            str(tmp_path / "audit-subprocess"),
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src:."},
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(completed.stdout)
    remote = tmp_path / "audit-subprocess" / identity["publication_id"]

    assert identity == {
        "audit_id": local.audit_id,
        "publication_id": local.publication_id,
    }
    assert {
        path.name: path.read_bytes() for path in local.output_dir.iterdir()
    } == {path.name: path.read_bytes() for path in remote.iterdir()}


def test_source_named_file_replacement_during_audit_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    original = audit_module._prepare_audit_publication

    def prepare_then_replace(bound: object) -> object:
        prepared = original(bound)  # type: ignore[arg-type]
        replacement = publication / "replacement.tmp"
        replacement.write_bytes(b"\n")
        replacement.replace(publication / "events.jsonl")
        return prepared

    monkeypatch.setattr(audit_module, "_prepare_audit_publication", prepare_then_replace)
    output_root = tmp_path / "audit"

    with pytest.raises(ValueError, match="changed during verification"):
        publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=publication,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_source_directory_replacement_during_audit_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    original = audit_module._prepare_audit_publication
    detached_root = tmp_path / "detached"
    detached_root.mkdir()

    def prepare_then_replace_directory(bound: object) -> object:
        prepared = original(bound)  # type: ignore[arg-type]
        detached = detached_root / publication.name
        publication.rename(detached)
        shutil.copytree(detached, publication)
        return prepared

    monkeypatch.setattr(
        audit_module,
        "_prepare_audit_publication",
        prepare_then_replace_directory,
    )
    output_root = tmp_path / "audit"

    with pytest.raises(ValueError, match="changed during verification"):
        publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=publication,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_source_mutation_chains_consumer_exception_and_creates_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )

    def mutate_then_fail(bound: object) -> object:
        replacement = publication / "replacement.tmp"
        replacement.write_bytes(b"\n")
        replacement.replace(publication / "events.jsonl")
        raise RuntimeError("synthetic audit consumer failure")

    monkeypatch.setattr(audit_module, "_prepare_audit_publication", mutate_then_fail)
    output_root = tmp_path / "audit"

    with pytest.raises(ValueError, match="changed during verification") as caught:
        publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=publication,
            output_root=output_root,
        )

    chain: list[BaseException] = []
    error: BaseException | None = caught.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__
    assert any(
        isinstance(item, RuntimeError)
        and str(item) == "synthetic audit consumer failure"
        for item in chain
    )
    assert not output_root.exists()


def test_audit_on_or_off_does_not_change_science_or_projection_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    workspace_before = _tree_snapshot(storage.workspace_dir)  # type: ignore[attr-defined]
    projection_before = _source_payloads(publication)

    publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=publication,
        output_root=tmp_path / "audit",
    )

    assert _tree_snapshot(storage.workspace_dir) == workspace_before  # type: ignore[attr-defined]
    assert _source_payloads(publication) == projection_before


def test_audit_publication_is_atomic_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    output_root = tmp_path / "audit"
    first = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=publication,
        output_root=output_root,
    )
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir()}

    with pytest.raises(ValueError, match="already exists"):
        publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=publication,
            output_root=output_root,
        )

    assert {path.name: path.read_bytes() for path in first.output_dir.iterdir()} == before


def test_audit_output_root_cannot_overlap_verified_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    before = _source_payloads(publication)

    with pytest.raises(ValueError, match="overlaps"):
        publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=publication,
            output_root=publication,
        )

    assert _source_payloads(publication) == before


def test_audit_manifest_binds_every_non_manifest_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    result = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=publication,
        output_root=tmp_path / "audit",
    )

    manifest = json.loads(result.audit_manifest_json.read_text(encoding="utf-8"))
    expected = {
        name: "sha256:" + hashlib.sha256((result.output_dir / name).read_bytes()).hexdigest()
        for name in _AUDIT_NAMES
        if name != "audit_manifest.json"
    }
    assert manifest["artifacts"] == expected
    assert result.output_dir.name == manifest["publication_id"]
