from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.oled_scientific_agent_source_evidence import (
    CAUSAL_LINK_VERSION,
    DISPATCH_RECEIPT_VERSION,
    FAILURE_EVIDENCE_VERSION,
    RECOVERY_RECEIPT_VERSION,
    ScientificAgentTypedFailure,
    build_failure_evidence,
    dispatch_authority_roster,
    failure_reason_codes_for_error_code,
    publish_actual_dispatch_receipt,
    publish_dispatch_receipt,
    publish_recovery_receipt,
    read_dispatch_receipts,
    read_recovery_receipts,
    validate_dispatch_receipt,
    validate_failure_evidence,
    validate_recovery_receipt,
)


_CHILD_ID = "oled-bounded-session-" + "a" * 64 + "-generation-01"
_TASK_ID = "execute_oled_inverse_design"
_SOURCE_SHA = "sha256:" + "1" * 64


def _publish_dispatch_after_barrier(
    run_dir: str,
    attempt_id: str,
    barrier: Any,
    results: Any,
) -> None:
    barrier.wait()
    try:
        receipt = publish_actual_dispatch_receipt(
            run_dir=Path(run_dir),
            child_run_id=_CHILD_ID,
            task_id=_TASK_ID,
            request_or_stage_digest=_SOURCE_SHA,
            attempt_id=attempt_id,
        )
        results.put(("ok", dict(receipt.payload)))
    except Exception as exc:  # pragma: no cover - surfaced by the parent assertion
        results.put(("error", type(exc).__name__))


@pytest.mark.pr_fast
@pytest.mark.unit
def test_failure_source_evidence_is_semantic_canonical_and_multi_reason() -> None:
    first = build_failure_evidence(
        reason_codes=("ssh_connection_failed", "gate_snapshot_mismatch"),
        cause_child_run_id=_CHILD_ID,
    )
    second = build_failure_evidence(
        reason_codes=("gate_snapshot_mismatch", "ssh_connection_failed"),
        cause_child_run_id=_CHILD_ID,
    )

    assert first == second == {
        "evidence_version": FAILURE_EVIDENCE_VERSION,
        "reason_codes": ["gate_snapshot_mismatch", "ssh_connection_failed"],
        "recovery_disposition": "unrecovered",
        "recovery_receipt_id": None,
        "causal_link": {
            "version": CAUSAL_LINK_VERSION,
            "cause_child_run_id": _CHILD_ID,
        },
        "source_record_digests": [],
    }
    assert validate_failure_evidence(first) == first


@pytest.mark.pr_fast
@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe",
    (
        "/private/operator/project",
        "/private/.ssh/known_hosts",
        "private.compute.invalid",
        "internal-node_42",
        "192.0.2.1",
        "user@example.invalid",
        "Authorization: Bearer secret-token",
        "MOLLY_LLM_API_KEY=secret",
        "ssh user@private-host",
    ),
)
def test_failure_reason_allowlist_rejects_private_or_free_text(unsafe: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_failure_evidence(reason_codes=(unsafe,))


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe_child",
    (
        "oled-bounded-session-" + "a" * 64 + "-internal-node_42",
        "oled-bounded-session-private.compute.invalid-screening",
        "oled-bounded-session-" + "a" * 64 + "-generation-user@example.invalid",
    ),
)
def test_source_ids_reject_infrastructure_shaped_suffixes(unsafe_child: str) -> None:
    with pytest.raises(ValueError, match="child run ID"):
        build_failure_evidence(
            reason_codes=("tool_runtime_failure",),
            cause_child_run_id=unsafe_child,
        )


@pytest.mark.unit
def test_typed_failure_and_error_mapping_never_classify_from_message_text() -> None:
    error = ScientificAgentTypedFailure("known_hosts_verification_failed")
    assert error.reason_codes == ("known_hosts_verification_failed",)
    assert str(error) == "scientific_agent_typed_failure"
    assert failure_reason_codes_for_error_code(
        "generic failure mentioning known_hosts and private.compute.invalid",
        fallback="tool_runtime_failure",
    ) == ("tool_runtime_failure",)
    assert failure_reason_codes_for_error_code(
        "known_hosts_verification_failed"
    ) == ("known_hosts_verification_failed",)


@pytest.mark.pr_fast
def test_dispatch_receipts_distinguish_real_duplicate_and_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial = publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="initial",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="1" * 32,
    )
    duplicate = publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="duplicate_rejected",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="2" * 32,
    )
    replay = publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="idempotent_replay",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="3" * 32,
    )

    receipts = read_dispatch_receipts(run_dir=run_dir)
    assert [item.payload["dispatch_kind"] for item in receipts] == [
        "initial",
        "duplicate_rejected",
        "idempotent_replay",
    ]
    assert [item.payload["execution_started"] for item in receipts] == [
        True,
        False,
        False,
    ]
    assert duplicate.payload["reason_codes"] == ["duplicate_dispatch_detected"]
    assert replay.payload["reason_codes"] == []
    assert duplicate.payload["predecessor_receipt_id"] == initial.payload["receipt_id"]
    assert replay.payload["predecessor_receipt_id"] == duplicate.payload["receipt_id"]
    assert all(item.payload["receipt_version"] == DISPATCH_RECEIPT_VERSION for item in receipts)
    assert all(item.authority_payload is not None for item in receipts)
    assert all(item.authority_sha256 is not None for item in receipts)


@pytest.mark.adversarial
def test_dispatch_ordinal_and_predecessor_allocation_is_cross_process_atomic(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_publish_dispatch_after_barrier,
            args=(str(run_dir), str(index) * 32, barrier, results),
        )
        for index in (1, 2)
    ]
    for process in processes:
        process.start()
    observed = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert [status for status, _ in observed] == ["ok", "ok"], observed
    receipts = read_dispatch_receipts(run_dir=run_dir)
    assert [item.payload["dispatch_ordinal"] for item in receipts] == [1, 2]
    assert [item.payload["dispatch_kind"] for item in receipts] == [
        "initial",
        "retry",
    ]
    assert receipts[0].payload["predecessor_receipt_id"] is None
    assert (
        receipts[1].payload["predecessor_receipt_id"]
        == receipts[0].payload["receipt_id"]
    )
    assert dispatch_authority_roster(run_dir=run_dir) == [
        {
            "receipt_id": str(item.payload["receipt_id"]),
            "dispatch_authority_id": str(item.payload["dispatch_authority_id"]),
            "authority_sha256": str(item.authority_sha256),
        }
        for item in receipts
    ]


@pytest.mark.adversarial
def test_dispatch_receipts_reject_reused_attempt_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="initial",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="7" * 32,
    )
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="retry",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="7" * 32,
    )

    with pytest.raises(ValueError, match="attempt IDs"):
        read_dispatch_receipts(run_dir=run_dir)


def test_recovery_receipt_is_content_bound_and_idempotent(tmp_path: Path) -> None:
    action_dir = tmp_path / "action"
    action_dir.mkdir()
    first = publish_recovery_receipt(
        action_dir=action_dir,
        action_id="oled-session-action-" + "2" * 64,
        request_digest="sha256:" + "3" * 64,
        recovered_child_run_id=_CHILD_ID,
        recovered_stage_sha256="sha256:" + "4" * 64,
        source_dispatch_receipt_ids=[],
        expected_revision=7,
        completed_revision=8,
    )
    second = publish_recovery_receipt(
        action_dir=action_dir,
        action_id="oled-session-action-" + "2" * 64,
        request_digest="sha256:" + "3" * 64,
        recovered_child_run_id=_CHILD_ID,
        recovered_stage_sha256="sha256:" + "4" * 64,
        source_dispatch_receipt_ids=[],
        expected_revision=7,
        completed_revision=8,
    )

    assert first.payload == second.payload
    assert first.sha256 == second.sha256
    assert first.payload["receipt_version"] == RECOVERY_RECEIPT_VERSION
    assert read_recovery_receipts(action_dir=action_dir)[0].payload == first.payload


@pytest.mark.adversarial
@pytest.mark.pr_fast
@pytest.mark.parametrize("attack", ("content", "authority", "roster", "symlink"))
def test_dispatch_receipt_tampering_fails_closed(
    tmp_path: Path, attack: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    receipt = publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=_CHILD_ID,
        task_id=_TASK_ID,
        dispatch_kind="initial",
        request_or_stage_digest=_SOURCE_SHA,
        attempt_id="5" * 32,
    )
    if attack == "content":
        payload = dict(receipt.payload)
        payload["dispatch_ordinal"] = 2
        receipt.receipt_json.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif attack == "authority":
        assert receipt.authority_json is not None
        authority = json.loads(receipt.authority_json.read_text(encoding="utf-8"))
        authority["boundary_material_sha256"] = "sha256:" + "9" * 64
        receipt.authority_json.write_text(
            json.dumps(authority, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif attack == "roster":
        (receipt.receipt_dir / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        receipt.receipt_json.unlink()
        receipt.receipt_json.symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError):
        read_dispatch_receipts(run_dir=run_dir)


@pytest.mark.unit
def test_receipt_validators_reject_resigned_identity_and_invalid_recovery() -> None:
    dispatch = {
        "receipt_version": DISPATCH_RECEIPT_VERSION,
        "receipt_id": "scientific-agent-dispatch-receipt:" + "0" * 64,
        "child_run_id": _CHILD_ID,
        "task_id": _TASK_ID,
        "attempt_id": "6" * 32,
        "dispatch_ordinal": 1,
        "dispatch_kind": "initial",
        "execution_started": True,
        "reason_codes": [],
        "predecessor_receipt_id": None,
        "dispatch_authority_id": "scientific-agent-dispatch-authority:" + "1" * 64,
        "request_or_stage_digest": _SOURCE_SHA,
    }
    with pytest.raises(ValueError, match="identity"):
        validate_dispatch_receipt(dispatch)

    recovery = {
        "receipt_version": RECOVERY_RECEIPT_VERSION,
        "receipt_id": "scientific-agent-recovery-receipt:" + "0" * 64,
        "action_id": "oled-session-action-" + "2" * 64,
        "request_digest": "sha256:" + "3" * 64,
        "recovery_kind": "adopt_completed_child",
        "recovered_child_run_id": _CHILD_ID,
        "recovered_stage_sha256": "sha256:" + "4" * 64,
        "source_dispatch_receipt_ids": [],
        "expected_revision": 9,
        "completed_revision": 8,
    }
    with pytest.raises(ValueError, match="revision"):
        validate_recovery_receipt(recovery)
