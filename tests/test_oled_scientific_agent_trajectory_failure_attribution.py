from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

import ai4s_agent.oled_scientific_agent_trajectory_failure_attribution as attribution_module
from ai4s_agent.oled_real_phase1_execution import _stable_hash
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    _prepare_audit_publication_from_verified_bytes,
    publish_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    _assert_exact_attribution_payloads,
    _prepare_failure_attribution_from_verified_bytes,
    publish_oled_scientific_agent_failure_attribution,
    verify_oled_scientific_agent_failure_attribution,
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


_TRAJECTORY_NAMES = (
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
_ATTRIBUTION_NAMES = (
    "attribution_manifest.json",
    "failure_attributions.jsonl",
    "failure_taxonomy.json",
    "report.md",
    "source_binding.json",
)


@dataclass(frozen=True)
class _SourceBundle:
    storage: object
    project_id: str
    current: object
    actions_root: Path
    trajectory_dir: Path
    audit_dir: Path


@pytest.fixture(scope="module")
def verified_source_bundle(tmp_path_factory: pytest.TempPathFactory) -> _SourceBundle:
    root = tmp_path_factory.mktemp("failure-attribution-source")
    patcher = pytest.MonkeyPatch()
    try:
        storage, project_id, current, actions_root, trajectory_dir = _publication(
            root, patcher
        )
        audit = publish_oled_scientific_agent_trajectory_audit_metrics(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            trajectory_publication_dir=trajectory_dir,
            output_root=root / "audit",
        )
        yield _SourceBundle(
            storage=storage,
            project_id=project_id,
            current=current,
            actions_root=actions_root,
            trajectory_dir=trajectory_dir,
            audit_dir=audit.output_dir,
        )
    finally:
        patcher.undo()


def _payloads(directory: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in names}


def _trajectory_payloads(bundle: _SourceBundle) -> dict[str, bytes]:
    return _payloads(bundle.trajectory_dir, _TRAJECTORY_NAMES)


def _audit_payloads(bundle: _SourceBundle) -> dict[str, bytes]:
    return _payloads(bundle.audit_dir, _AUDIT_NAMES)


def _resign_event(event: dict[str, object]) -> dict[str, object]:
    unsigned = {
        key: value
        for key, value in event.items()
        if key not in {"event_id", "sequence_index"}
    }
    return {
        **event,
        "event_id": "scientific-agent-trajectory-event:" + _stable_hash(unsigned),
    }


def _refresh_trajectory(
    payloads: dict[str, bytes],
    *,
    events: list[dict[str, object]],
    telemetry_findings: list[dict[str, object]] | None = None,
) -> dict[str, bytes]:
    result = dict(payloads)
    resigned: list[dict[str, object]] = []
    for index, event in enumerate(events):
        resigned.append(_resign_event({**event, "sequence_index": index}))
    result["events.jsonl"] = _canonical_jsonl_bytes(resigned)
    if telemetry_findings is not None:
        result["telemetry_findings.jsonl"] = _canonical_jsonl_bytes(
            telemetry_findings
        )
    receipt = json.loads(result["trajectory.json"])
    receipt["counts"]["event_count"] = len(resigned)
    receipt["counts"]["telemetry_finding_count"] = len(
        result["telemetry_findings.jsonl"].splitlines()
    )
    for name in (
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
    ):
        receipt["artifacts"][name] = _sha256(result[name])
    result["trajectory.json"] = _canonical_json_bytes(receipt)
    return result


def _prepare_direct(trajectory_payloads: dict[str, bytes]) -> object:
    receipt = json.loads(trajectory_payloads["trajectory.json"])
    audit = _prepare_audit_publication_from_verified_bytes(
        payloads=trajectory_payloads,
        verified_trajectory_id=receipt["trajectory_id"],
        verified_publication_id=receipt["publication_id"],
    )
    return _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=trajectory_payloads,
        audit_payloads=audit.payloads,
        verified_trajectory_id=receipt["trajectory_id"],
        verified_trajectory_publication_id=receipt["publication_id"],
        verified_audit_id=audit.audit_id,
        verified_audit_publication_id=audit.publication_id,
    )


def _prepared_rows(prepared: object) -> list[dict[str, object]]:
    payloads = prepared.payloads  # type: ignore[attr-defined]
    return [
        json.loads(line)
        for line in payloads["failure_attributions.jsonl"].splitlines()
    ]


def _prepared_manifest(prepared: object) -> dict[str, object]:
    payloads = prepared.payloads  # type: ignore[attr-defined]
    return json.loads(payloads["attribution_manifest.json"])


def _failure_payloads(
    base: dict[str, bytes],
    *,
    reason_code: str | tuple[str, ...],
    sensitive_outcome: dict[str, object] | None = None,
    telemetry_findings: list[dict[str, object]] | None = None,
) -> dict[str, bytes]:
    receipt = json.loads(base["trajectory.json"])
    events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
    completed = next(
        event for event in events if event["event_kind"] == "stage_completed"
    )
    child_id = completed["child_run_id"]
    completed["event_kind"] = "stage_failed"
    completed["outcome"] = {
        "child_status": "failed",
        **(sensitive_outcome or {}),
    }
    completed["reason_codes"] = (
        [reason_code] if isinstance(reason_code, str) else list(reason_code)
    )
    events = [
        event
        for event in events
        if not (
            event["event_kind"] == "publication_verified"
            and event["child_run_id"] == child_id
        )
    ]
    final_state = next(
        event
        for event in reversed(events)
        if event["event_kind"] == "state_committed"
    )
    final_state["outcome"]["status"] = "FAILED"
    terminal = next(
        event
        for event in events
        if event["event_kind"] == "terminal_result_committed"
    )
    terminal["outcome"].update(
        {
            "status": "FAILED",
            "stop_reason": "stage_failure",
            "has_complete_top_n": None,
        }
    )
    terminal["reason_codes"] = ["stage_failure"]
    receipt["terminal_status"] = "FAILED"
    result = dict(base)
    result["trajectory.json"] = _canonical_json_bytes(receipt)
    return _refresh_trajectory(
        result,
        events=events,
        telemetry_findings=telemetry_findings,
    )


def _bounded_payloads(base: dict[str, bytes], stop_reason: str) -> dict[str, bytes]:
    receipt = json.loads(base["trajectory.json"])
    events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
    terminal = next(
        event
        for event in events
        if event["event_kind"] == "terminal_result_committed"
    )
    terminal["outcome"].update(
        {
            "status": "STOPPED_BOUNDED_NO_SOLUTION",
            "stop_reason": stop_reason,
            "has_complete_top_n": False,
        }
    )
    terminal["reason_codes"] = [stop_reason]
    receipt["terminal_status"] = "STOPPED_BOUNDED_NO_SOLUTION"
    result = dict(base)
    result["trajectory.json"] = _canonical_json_bytes(receipt)
    return _refresh_trajectory(result, events=events)


def _recovered_then_bounded_payloads(
    base: dict[str, bytes], stop_reason: str
) -> dict[str, bytes]:
    receipt = json.loads(base["trajectory.json"])
    events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
    completed_index = next(
        index
        for index, event in enumerate(events)
        if event["event_kind"] == "stage_completed"
    )
    failed = deepcopy(events[completed_index])
    failed["event_kind"] = "stage_failed"
    failed["outcome"] = {"child_status": "failed"}
    failed["reason_codes"] = ["tool_runtime_failure"]
    events.insert(completed_index, failed)
    terminal = next(
        event
        for event in events
        if event["event_kind"] == "terminal_result_committed"
    )
    terminal["outcome"].update(
        {
            "status": "STOPPED_BOUNDED_NO_SOLUTION",
            "stop_reason": stop_reason,
            "has_complete_top_n": False,
        }
    )
    terminal["reason_codes"] = [stop_reason]
    receipt["terminal_status"] = "STOPPED_BOUNDED_NO_SOLUTION"
    result = dict(base)
    result["trajectory.json"] = _canonical_json_bytes(receipt)
    return _refresh_trajectory(result, events=events)


def _standard_case(
    base: dict[str, bytes], case: str
) -> tuple[object, dict[str, object]]:
    expected: dict[str, object]
    if case == "known-hosts-propagation":
        payloads = _failure_payloads(
            base,
            reason_code="known_hosts_verification_failed",
            sensitive_outcome={
                "expected_hostname": "private.compute.invalid",
                "known_hosts_path": "/private/runtime/known_hosts",
                "username": "private-operator",
            },
        )
        expected = {
            "first": "transport",
            "code": "REVIEW_RECOMMENDED",
            "symptoms": ["tool_runtime"],
            "status": "determined",
        }
    elif case == "history-truncation":
        payloads = dict(base)
        events = [json.loads(line) for line in payloads["events.jsonl"].splitlines()]
        events.pop(1)
        payloads["events.jsonl"] = _canonical_jsonl_bytes(events)
        expected = {
            "first": "audit_integrity",
            "code": "INTEGRITY_FAILURE",
            "symptoms": [],
            "status": "determined",
        }
    elif case == "duplicate-dispatch":
        receipt = json.loads(base["trajectory.json"])
        events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
        dispatch_index = next(
            index
            for index, event in enumerate(events)
            if event["event_kind"] == "task_dispatched"
        )
        duplicate = dict(events[dispatch_index])
        duplicate["reason_codes"] = ["duplicate_dispatch_detected"]
        events.insert(dispatch_index + 1, duplicate)
        terminal = next(
            event
            for event in events
            if event["event_kind"] == "terminal_result_committed"
        )
        terminal["outcome"].update(
            {
                "status": "FAILED",
                "stop_reason": "duplicate_dispatch",
                "has_complete_top_n": None,
            }
        )
        terminal["reason_codes"] = ["duplicate_dispatch"]
        receipt["terminal_status"] = "FAILED"
        payloads = dict(base)
        payloads["trajectory.json"] = _canonical_json_bytes(receipt)
        payloads = _refresh_trajectory(payloads, events=events)
        expected = {
            "first": "recovery",
            "code": "REVIEW_RECOMMENDED",
            "symptoms": ["tool_runtime"],
            "status": "determined",
        }
    elif case == "stale-state":
        telemetry = [
            {
                "finding_version": "scientific_agent_trajectory_projection.v1",
                "action_id": "oled-session-action-stale",
                "reason_code": "telemetry_conflicts_with_session_history",
                "telemetry_sha256": "sha256:" + "2" * 64,
                "authority_effect": "ignored_for_scientific_facts",
                "finding_id": "trajectory-telemetry-finding:stale",
                "runtime_path": "/private/runtime/action.json",
                "hostname": "private.compute.invalid",
            }
        ]
        payloads = _failure_payloads(
            base,
            reason_code="tool_runtime_failure",
            telemetry_findings=telemetry,
        )
        expected = {
            "first": "tool_runtime",
            "code": "REVIEW_RECOMMENDED",
            "symptoms": ["tool_runtime", "recovery"],
            "status": "determined",
        }
    else:
        raise AssertionError(case)
    return _prepare_direct(payloads), expected


@pytest.mark.pr_fast
@pytest.mark.parametrize(
    "case",
    (
        "known-hosts-propagation",
        "history-truncation",
        "duplicate-dispatch",
        "stale-state",
    ),
)
def test_standard_failure_cases_freeze_first_cause_symptoms_and_sources(
    verified_source_bundle: _SourceBundle,
    case: str,
) -> None:
    prepared, expected = _standard_case(_trajectory_payloads(verified_source_bundle), case)
    rows = _prepared_rows(prepared)
    manifest = _prepared_manifest(prepared)
    first = [row for row in rows if row["attribution_role"] == "first_cause"]
    symptoms = [
        row for row in rows if row["attribution_role"] == "downstream_symptom"
    ]

    assert manifest["result"]["attribution_status"] == expected["status"]
    assert len(first) == 1
    assert first[0]["taxonomy_family"] == expected["first"]
    assert first[0]["finding_code"] == expected["code"]
    assert [row["taxonomy_family"] for row in symptoms] == expected["symptoms"]
    assert all(row["source_refs"] for row in rows)
    assert all(
        ref["artifact_name"] in {
            "events.jsonl",
            "audit_findings.jsonl",
            "telemetry_findings.jsonl",
        }
        and ref["sha256"].startswith("sha256:")
        for row in rows
        for ref in row["source_refs"]
    )
    output = b"".join(prepared.payloads.values())  # type: ignore[attr-defined]
    assert b"known_hosts_verification_failed" not in output
    assert b"private.compute.invalid" not in output
    assert b"private-operator" not in output
    assert b"/private/runtime" not in output
    if case == "stale-state":
        stale = next(row for row in rows if row["taxonomy_family"] == "recovery")
        assert stale["attribution_status"] == "undetermined"
        assert stale["evidence_sufficiency"] == "insufficient"


@pytest.mark.pr_fast
def test_multi_family_stage_failure_publishes_ambiguity_without_priority(
    verified_source_bundle: _SourceBundle,
) -> None:
    base = _trajectory_payloads(verified_source_bundle)
    prepared_by_order = []
    for reasons in (
        ("gate_snapshot_mismatch", "ssh_connection_failed"),
        ("ssh_connection_failed", "gate_snapshot_mismatch"),
    ):
        prepared_by_order.append(
            _prepare_direct(_failure_payloads(base, reason_code=reasons))
        )

    for prepared in prepared_by_order:
        manifest = _prepared_manifest(prepared)
        rows = _prepared_rows(prepared)
        assert manifest["result"] == {
            "ambiguity_reason": "multiple_equal_first_cause_candidates",
            "attribution_status": "undetermined",
            "primary_first_cause_id": None,
        }
        candidates = [
            row
            for row in rows
            if row["taxonomy_family"]
            in {"authorization_mismatch", "transport"}
        ]
        assert {row["taxonomy_family"] for row in candidates} == {
            "authorization_mismatch",
            "transport",
        }
        assert all(
            row["deterministic_reason_code"]
            == "ambiguous_equal_first_cause_candidates"
            and row["attribution_status"] == "undetermined"
            for row in candidates
        )


@pytest.mark.pr_fast
@pytest.mark.parametrize(
    ("stop_reason", "family"),
    (
        ("max_generation_rounds_reached", "policy_constraint"),
        ("candidate_supply_exhausted", "candidate_supply"),
    ),
)
def test_recovered_early_failure_is_not_linked_to_independent_terminal_stop(
    verified_source_bundle: _SourceBundle,
    stop_reason: str,
    family: str,
) -> None:
    prepared = _prepare_direct(
        _recovered_then_bounded_payloads(
            _trajectory_payloads(verified_source_bundle), stop_reason
        )
    )
    rows = _prepared_rows(prepared)
    first = next(row for row in rows if row["attribution_role"] == "first_cause")
    terminal = next(
        row
        for row in rows
        if row["affected"]["event_kind"] == "terminal_result_committed"
    )

    assert first["taxonomy_family"] == "tool_runtime"
    assert terminal["taxonomy_family"] == family
    assert terminal["attribution_role"] == "downstream_symptom"
    assert terminal["attribution_status"] == "undetermined"
    assert terminal["deterministic_reason_code"] == "causal_link_not_proven"
    assert terminal["finding_code"] == "REVIEW_RECOMMENDED"


@pytest.mark.parametrize(
    ("case", "family", "code", "status"),
    (
        (
            "bounded",
            "policy_constraint",
            "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            "determined",
        ),
        (
            "budget",
            "policy_constraint",
            "BUDGET_LIMIT_REACHED",
            "determined",
        ),
        (
            "model",
            "model_inadequacy",
            "MODEL_INADEQUACY_DETECTED",
            "determined",
        ),
        (
            "candidate",
            "candidate_supply",
            "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            "determined",
        ),
        (
            "insufficient",
            "policy_constraint",
            "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            "undetermined",
        ),
    ),
)
def test_normal_failure_classification_respects_claim_boundaries(
    verified_source_bundle: _SourceBundle,
    case: str,
    family: str,
    code: str,
    status: str,
) -> None:
    base = _trajectory_payloads(verified_source_bundle)
    if case == "bounded":
        payloads = _bounded_payloads(
            base, "non_supply_policy_prevented_complete_top_n"
        )
    elif case == "budget":
        payloads = _bounded_payloads(base, "max_generation_rounds_reached")
    elif case == "model":
        payloads = _failure_payloads(base, reason_code="model_inadequacy_detected")
    elif case == "candidate":
        payloads = _bounded_payloads(base, "candidate_supply_exhausted")
    else:
        payloads = _bounded_payloads(base, "bounded_search_ended")
    prepared = _prepare_direct(payloads)
    rows = _prepared_rows(prepared)
    manifest = _prepared_manifest(prepared)

    assert rows[0]["taxonomy_family"] == family
    assert rows[0]["finding_code"] == code
    assert manifest["result"]["attribution_status"] == status
    if status == "undetermined":
        assert manifest["result"]["primary_first_cause_id"] is None
        assert rows[0]["evidence_sufficiency"] == "insufficient"


@pytest.mark.pr_fast
def test_insufficient_stage_evidence_is_undetermined_and_review_recommended(
    verified_source_bundle: _SourceBundle,
) -> None:
    prepared = _prepare_direct(
        _failure_payloads(
            _trajectory_payloads(verified_source_bundle), reason_code="failed"
        )
    )
    rows = _prepared_rows(prepared)
    manifest = _prepared_manifest(prepared)

    assert manifest["result"] == {
        "ambiguity_reason": "insufficient_causal_evidence",
        "attribution_status": "undetermined",
        "primary_first_cause_id": None,
    }
    assert all(row["finding_code"] == "REVIEW_RECOMMENDED" for row in rows)
    assert all(row["attribution_role"] == "downstream_symptom" for row in rows)


def test_successful_recovery_marker_does_not_create_failure_attribution(
    verified_source_bundle: _SourceBundle,
) -> None:
    base = _trajectory_payloads(verified_source_bundle)
    events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
    state = next(
        event
        for event in events
        if event["event_kind"] == "state_committed"
        and event["session_revision"] > 0
    )
    state["outcome"] = {**state["outcome"], "status": "RECOVERY_REQUIRED"}
    prepared = _prepare_direct(_refresh_trajectory(base, events=events))

    assert _prepared_rows(prepared) == []
    assert _prepared_manifest(prepared)["result"]["attribution_status"] == "no_failure"


@pytest.mark.parametrize("replay", ("idempotent-dispatch", "recovery-adoption", "terminal-replay"))
def test_replay_forms_are_not_misclassified_as_duplicate_dispatch(
    verified_source_bundle: _SourceBundle,
    replay: str,
) -> None:
    base = _trajectory_payloads(verified_source_bundle)
    events = [json.loads(line) for line in base["events.jsonl"].splitlines()]
    if replay == "idempotent-dispatch":
        index = next(
            i for i, event in enumerate(events) if event["event_kind"] == "task_dispatched"
        )
        events.insert(index + 1, dict(events[index]))
    elif replay == "recovery-adoption":
        completed = next(
            event for event in events if event["event_kind"] == "stage_completed"
        )
        completed["reason_codes"] = ["recovered_existing_publication"]
    else:
        terminal = next(
            event
            for event in events
            if event["event_kind"] == "terminal_result_committed"
        )
        terminal["reason_codes"] = ["terminal_exact_replay"]
    rows = _prepared_rows(_prepare_direct(_refresh_trajectory(base, events=events)))

    assert not any(
        row["deterministic_reason_code"] == "duplicate_dispatch_persisted"
        for row in rows
    )
    assert not any(
        row["taxonomy_family"] == "recovery"
        and row["attribution_role"] == "first_cause"
        for row in rows
    )


def _copy_bound_sources(
    bundle: _SourceBundle, root: Path
) -> tuple[Path, Path]:
    trajectory = root / "trajectory" / bundle.trajectory_dir.name
    audit = root / "audit" / bundle.audit_dir.name
    trajectory.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    shutil.copytree(bundle.trajectory_dir, trajectory)
    shutil.copytree(bundle.audit_dir, audit)
    return trajectory, audit


@pytest.mark.pr_fast
@pytest.mark.parametrize(
    "attack",
    (
        "projection-named-inode-replacement",
        "audit-named-inode-replacement",
        "projection-directory-replacement",
        "audit-directory-replacement",
    ),
)
def test_source_named_inode_and_directory_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_source_bundle: _SourceBundle,
    attack: str,
) -> None:
    trajectory, audit = _copy_bound_sources(verified_source_bundle, tmp_path)
    original = attribution_module._prepare_failure_attribution

    def prepare_then_replace(bound: object) -> object:
        prepared = original(bound)  # type: ignore[arg-type]
        target = trajectory if attack.startswith("projection") else audit
        if "directory" in attack:
            detached = tmp_path / "detached" / target.name
            detached.parent.mkdir()
            target.rename(detached)
            shutil.copytree(detached, target)
        else:
            name = "events.jsonl" if target == trajectory else "audit_metrics.json"
            replacement = target / "replacement.tmp"
            replacement.write_bytes((target / name).read_bytes())
            replacement.replace(target / name)
        return prepared

    monkeypatch.setattr(
        attribution_module, "_prepare_failure_attribution", prepare_then_replace
    )
    output_root = tmp_path / "attribution"

    with pytest.raises(ValueError, match="changed during verification"):
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=trajectory,
            audit_publication_dir=audit,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_source_mutation_preserves_consumer_error_cause_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_source_bundle: _SourceBundle,
) -> None:
    trajectory, audit = _copy_bound_sources(verified_source_bundle, tmp_path)

    def mutate_then_fail(bound: object) -> object:
        replacement = trajectory / "replacement.tmp"
        replacement.write_bytes((trajectory / "events.jsonl").read_bytes())
        replacement.replace(trajectory / "events.jsonl")
        raise RuntimeError("synthetic attribution consumer failure")

    monkeypatch.setattr(
        attribution_module, "_prepare_failure_attribution", mutate_then_fail
    )
    with pytest.raises(ValueError, match="changed during verification") as caught:
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=trajectory,
            audit_publication_dir=audit,
            output_root=tmp_path / "attribution",
        )
    chain: list[BaseException] = []
    error: BaseException | None = caught.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__
    assert any(
        isinstance(item, RuntimeError)
        and str(item) == "synthetic attribution consumer failure"
        for item in chain
    )


@pytest.mark.parametrize(
    "attack",
    (
        "projection-roster-add",
        "audit-roster-delete",
        "projection-event-delete-resign",
        "projection-event-reorder-resign",
        "audit-source-sha-resign",
        "trajectory-source-symlink",
    ),
)
def test_source_roster_resigning_event_and_symlink_attacks_fail_closed(
    tmp_path: Path,
    verified_source_bundle: _SourceBundle,
    attack: str,
) -> None:
    trajectory, audit = _copy_bound_sources(verified_source_bundle, tmp_path)
    if attack == "projection-roster-add":
        (trajectory / "extra.json").write_text("{}\n", encoding="utf-8")
    elif attack == "audit-roster-delete":
        (audit / "report.md").unlink()
    elif attack.startswith("projection-event"):
        events = [
            json.loads(line)
            for line in (trajectory / "events.jsonl").read_text().splitlines()
        ]
        if "delete" in attack:
            events.pop(1)
        else:
            events[1], events[2] = events[2], events[1]
        for index, event in enumerate(events):
            event["sequence_index"] = index
        (trajectory / "events.jsonl").write_bytes(_canonical_jsonl_bytes(events))
        receipt = json.loads((trajectory / "trajectory.json").read_text())
        receipt["counts"]["event_count"] = len(events)
        receipt["artifacts"]["events.jsonl"] = _sha256(
            (trajectory / "events.jsonl").read_bytes()
        )
        (trajectory / "trajectory.json").write_bytes(_canonical_json_bytes(receipt))
    elif attack == "audit-source-sha-resign":
        binding = json.loads((audit / "source_binding.json").read_text())
        binding["source_artifacts"]["events.jsonl"] = "sha256:" + "f" * 64
        (audit / "source_binding.json").write_bytes(_canonical_json_bytes(binding))
        manifest = json.loads((audit / "audit_manifest.json").read_text())
        manifest["artifacts"]["source_binding.json"] = _sha256(
            (audit / "source_binding.json").read_bytes()
        )
        (audit / "audit_manifest.json").write_bytes(_canonical_json_bytes(manifest))
    else:
        original = trajectory
        link = tmp_path / "trajectory-link"
        link.symlink_to(original, target_is_directory=True)
        trajectory = link

    with pytest.raises(ValueError):
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=trajectory,
            audit_publication_dir=audit,
            output_root=tmp_path / "attribution",
        )


@pytest.mark.parametrize(
    "attack",
    (
        "taxonomy-family",
        "finding-code",
        "first-cause-source-rebind",
        "first-cause-symptom-order",
        "artifact-roster",
        "publication-content",
        "manifest-resign",
    ),
)
def test_attribution_tampering_and_resigned_manifest_fail_exact_replay(
    verified_source_bundle: _SourceBundle,
    attack: str,
) -> None:
    prepared, _ = _standard_case(
        _trajectory_payloads(verified_source_bundle), "known-hosts-propagation"
    )
    tampered = dict(prepared.payloads)  # type: ignore[attr-defined]
    if attack in {
        "taxonomy-family",
        "finding-code",
        "first-cause-source-rebind",
        "first-cause-symptom-order",
    }:
        rows = [
            json.loads(line)
            for line in tampered["failure_attributions.jsonl"].splitlines()
        ]
        if attack == "taxonomy-family":
            rows[0]["taxonomy_family"] = "model_inadequacy"
        elif attack == "finding-code":
            rows[0]["finding_code"] = "UNFROZEN_CODE"
        elif attack == "first-cause-source-rebind":
            rows[0]["source_refs"][0]["sha256"] = "sha256:" + "f" * 64
        else:
            rows[0]["attribution_role"] = "downstream_symptom"
            rows[1]["attribution_role"] = "first_cause"
            rows.reverse()
        tampered["failure_attributions.jsonl"] = _canonical_jsonl_bytes(rows)
    elif attack == "artifact-roster":
        tampered.pop("report.md")
    elif attack == "publication-content":
        tampered["report.md"] += b"tampered\n"
    else:
        taxonomy = json.loads(tampered["failure_taxonomy.json"])
        taxonomy["families"][0]["meaning"] = "resigned forgery"
        tampered["failure_taxonomy.json"] = _canonical_json_bytes(taxonomy)
        manifest = json.loads(tampered["attribution_manifest.json"])
        manifest["artifacts"]["failure_taxonomy.json"] = _sha256(
            tampered["failure_taxonomy.json"]
        )
        tampered["attribution_manifest.json"] = _canonical_json_bytes(manifest)

    with pytest.raises(ValueError, match="roster|exact replay"):
        _assert_exact_attribution_payloads(
            payloads=tampered,
            prepared=prepared,  # type: ignore[arg-type]
            directory_name=prepared.publication_id,  # type: ignore[attr-defined]
        )


def test_publication_is_atomic_no_replace_and_output_symlink_fails_closed(
    tmp_path: Path,
    verified_source_bundle: _SourceBundle,
) -> None:
    root = tmp_path / "attribution"
    first = publish_oled_scientific_agent_failure_attribution(
        storage=verified_source_bundle.storage,  # type: ignore[arg-type]
        project_id=verified_source_bundle.project_id,
        session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
        actions_root=verified_source_bundle.actions_root,
        trajectory_publication_dir=verified_source_bundle.trajectory_dir,
        audit_publication_dir=verified_source_bundle.audit_dir,
        output_root=root,
    )
    before = _payloads(first.output_dir, _ATTRIBUTION_NAMES)
    with pytest.raises(ValueError, match="already exists"):
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=verified_source_bundle.trajectory_dir,
            audit_publication_dir=verified_source_bundle.audit_dir,
            output_root=root,
        )
    assert _payloads(first.output_dir, _ATTRIBUTION_NAMES) == before

    symlink_root = tmp_path / "symlink-root"
    target = tmp_path / "target"
    target.mkdir()
    symlink_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|symbolic"):
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=verified_source_bundle.trajectory_dir,
            audit_publication_dir=verified_source_bundle.audit_dir,
            output_root=symlink_root,
        )


def test_attribution_failure_does_not_damage_existing_source_publications(
    tmp_path: Path,
    verified_source_bundle: _SourceBundle,
) -> None:
    projection_before = _trajectory_payloads(verified_source_bundle)
    audit_before = _audit_payloads(verified_source_bundle)
    with pytest.raises(ValueError, match="overlaps"):
        publish_oled_scientific_agent_failure_attribution(
            storage=verified_source_bundle.storage,  # type: ignore[arg-type]
            project_id=verified_source_bundle.project_id,
            session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
            actions_root=verified_source_bundle.actions_root,
            trajectory_publication_dir=verified_source_bundle.trajectory_dir,
            audit_publication_dir=verified_source_bundle.audit_dir,
            output_root=verified_source_bundle.audit_dir,
        )
    assert _trajectory_payloads(verified_source_bundle) == projection_before
    assert _audit_payloads(verified_source_bundle) == audit_before


def test_same_verified_bytes_and_reversed_input_maps_are_byte_identical(
    verified_source_bundle: _SourceBundle,
) -> None:
    trajectory = _trajectory_payloads(verified_source_bundle)
    audit = _audit_payloads(verified_source_bundle)
    manifest = json.loads(audit["audit_manifest.json"])
    receipt = json.loads(trajectory["trajectory.json"])
    kwargs = {
        "verified_trajectory_id": receipt["trajectory_id"],
        "verified_trajectory_publication_id": receipt["publication_id"],
        "verified_audit_id": manifest["audit_id"],
        "verified_audit_publication_id": manifest["publication_id"],
    }
    first = _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=trajectory,
        audit_payloads=audit,
        **kwargs,
    )
    second = _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=dict(reversed(tuple(trajectory.items()))),
        audit_payloads=dict(reversed(tuple(audit.items()))),
        **kwargs,
    )

    assert first.attribution_id == second.attribution_id
    assert first.publication_id == second.publication_id
    assert dict(first.payloads) == dict(second.payloads)


@pytest.mark.slow
def test_cross_process_and_hash_seed_attribution_is_byte_identical(
    tmp_path: Path,
    verified_source_bundle: _SourceBundle,
) -> None:
    trajectory = _failure_payloads(
        _trajectory_payloads(verified_source_bundle),
        reason_code=("gate_snapshot_mismatch", "ssh_connection_failed"),
    )
    receipt = json.loads(trajectory["trajectory.json"])
    audit = _prepare_audit_publication_from_verified_bytes(
        payloads=trajectory,
        verified_trajectory_id=receipt["trajectory_id"],
        verified_publication_id=receipt["publication_id"],
    )
    prepared = _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=trajectory,
        audit_payloads=audit.payloads,
        verified_trajectory_id=receipt["trajectory_id"],
        verified_trajectory_publication_id=receipt["publication_id"],
        verified_audit_id=audit.audit_id,
        verified_audit_publication_id=audit.publication_id,
    )
    trajectory_dir = tmp_path / "trajectory"
    audit_dir = tmp_path / "audit"
    trajectory_dir.mkdir()
    audit_dir.mkdir()
    for name, payload in trajectory.items():
        (trajectory_dir / name).write_bytes(payload)
    for name, payload in audit.payloads.items():
        (audit_dir / name).write_bytes(payload)
    script = r'''
import hashlib
import json
import sys
from pathlib import Path
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import _prepare_failure_attribution_from_verified_bytes
t = {p.name: p.read_bytes() for p in reversed(sorted(Path(sys.argv[1]).iterdir()))}
a = {p.name: p.read_bytes() for p in reversed(sorted(Path(sys.argv[2]).iterdir()))}
receipt = json.loads(t["trajectory.json"])
manifest = json.loads(a["audit_manifest.json"])
prepared = _prepare_failure_attribution_from_verified_bytes(
    trajectory_payloads=t,
    audit_payloads=a,
    verified_trajectory_id=receipt["trajectory_id"],
    verified_trajectory_publication_id=receipt["publication_id"],
    verified_audit_id=manifest["audit_id"],
    verified_audit_publication_id=manifest["publication_id"],
)
digest = hashlib.sha256()
for name in sorted(prepared.payloads):
    digest.update(name.encode())
    digest.update(prepared.payloads[name])
print(json.dumps([prepared.attribution_id, prepared.publication_id, digest.hexdigest()]))
'''
    expected_digest = hashlib.sha256()
    for name in sorted(prepared.payloads):  # type: ignore[attr-defined]
        expected_digest.update(name.encode())
        expected_digest.update(prepared.payloads[name])  # type: ignore[attr-defined]
    expected = [
        prepared.attribution_id,  # type: ignore[attr-defined]
        prepared.publication_id,  # type: ignore[attr-defined]
        expected_digest.hexdigest(),
    ]
    observed = []
    for seed in ("1", "8675309"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(trajectory_dir), str(audit_dir)],
            cwd=Path(__file__).parents[1],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src:."},
            check=True,
            capture_output=True,
            text=True,
        )
        observed.append(json.loads(completed.stdout))
    assert observed == [expected, expected]


@pytest.mark.pr_fast
def test_exact_verifier_replays_publication_and_observer_only_bytes(
    tmp_path: Path,
    verified_source_bundle: _SourceBundle,
) -> None:
    workspace_before = _tree_snapshot(
        verified_source_bundle.storage.workspace_dir  # type: ignore[attr-defined]
    )
    projection_before = _trajectory_payloads(verified_source_bundle)
    audit_before = _audit_payloads(verified_source_bundle)
    result = publish_oled_scientific_agent_failure_attribution(
        storage=verified_source_bundle.storage,  # type: ignore[arg-type]
        project_id=verified_source_bundle.project_id,
        session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
        actions_root=verified_source_bundle.actions_root,
        trajectory_publication_dir=verified_source_bundle.trajectory_dir,
        audit_publication_dir=verified_source_bundle.audit_dir,
        output_root=tmp_path / "attribution",
    )
    verification = verify_oled_scientific_agent_failure_attribution(
        storage=verified_source_bundle.storage,  # type: ignore[arg-type]
        project_id=verified_source_bundle.project_id,
        session_id=verified_source_bundle.current.session_id,  # type: ignore[attr-defined]
        actions_root=verified_source_bundle.actions_root,
        trajectory_publication_dir=verified_source_bundle.trajectory_dir,
        audit_publication_dir=verified_source_bundle.audit_dir,
        attribution_publication_dir=result.output_dir,
    )

    assert verification.publication_id == result.publication_id
    assert verification.attribution_id == result.attribution_id
    assert result.failure_attributions_jsonl.read_bytes() == b""
    assert tuple(sorted(path.name for path in result.output_dir.iterdir())) == _ATTRIBUTION_NAMES
    assert _tree_snapshot(
        verified_source_bundle.storage.workspace_dir  # type: ignore[attr-defined]
    ) == workspace_before
    assert _trajectory_payloads(verified_source_bundle) == projection_before
    assert _audit_payloads(verified_source_bundle) == audit_before
    for path in result.output_dir.iterdir():
        assert str(tmp_path).encode() not in path.read_bytes()


@pytest.mark.slow
def test_multi_round_success_has_no_invented_failure_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, project_id, current = _terminal_two_rounds(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
    trajectory = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        output_root=tmp_path / "trajectory",
    )
    audit = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=trajectory.output_dir,
        output_root=tmp_path / "audit",
    )
    projection_before = _payloads(trajectory.output_dir, _TRAJECTORY_NAMES)
    audit_before = _payloads(audit.output_dir, _AUDIT_NAMES)
    result = publish_oled_scientific_agent_failure_attribution(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        trajectory_publication_dir=trajectory.output_dir,
        audit_publication_dir=audit.output_dir,
        output_root=tmp_path / "attribution",
    )

    assert result.failure_attributions_jsonl.read_bytes() == b""
    assert json.loads(result.attribution_manifest_json.read_text())["result"] == {
        "ambiguity_reason": None,
        "attribution_status": "no_failure",
        "primary_first_cause_id": None,
    }
    assert _payloads(trajectory.output_dir, _TRAJECTORY_NAMES) == projection_before
    assert _payloads(audit.output_dir, _AUDIT_NAMES) == audit_before
