from __future__ import annotations

import hashlib
import json

from ai4s_agent.state_machine.execution_state import ExecutionState
from ai4s_agent.state_machine.provenance_chain import (
    ROOT_PARENT_HASH,
    append_transition,
    hash_evidence,
    validate_provenance_chain,
)
from ai4s_agent.state_machine.transitions import (
    REQUIRED_EVIDENCE_BY_TARGET_STATE,
    validate_transition,
)


FIXED_TIMESTAMP = "2026-01-01T00:00:00Z"


def test_hash_mismatch_blocks_transition() -> None:
    context = _context_for(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="hash-mismatch-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    context["evidence_hash"] = _safe_hash("forged-evidence")

    result = validate_transition(ExecutionState.QUARANTINED, ExecutionState.ADMITTED, context)

    assert not result.allowed
    assert "evidence_hash_mismatch" in result.errors


def test_forged_timestamp_blocks_transition() -> None:
    context = _context_for(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="forged-timestamp-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    context["timestamp"] = "2026-01-01 00:00:00"

    result = validate_transition(ExecutionState.QUARANTINED, ExecutionState.ADMITTED, context)

    assert not result.allowed
    assert "invalid_transition_timestamp" in result.errors


def test_redaction_invariant_blocks_sensitive_evidence() -> None:
    context = _context_for(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="redaction-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    evidence = dict(context["evidence"])
    evidence["raw_value"] = "0.72"
    context["evidence"] = evidence
    context["evidence_hash"] = hash_evidence(evidence)

    result = validate_transition(ExecutionState.QUARANTINED, ExecutionState.ADMITTED, context)

    assert not result.allowed
    assert "redaction_invariant_failed" in result.errors


def test_state_regression_blocks_transition() -> None:
    result = validate_transition(
        ExecutionState.DOMAIN_VALIDATED,
        ExecutionState.ADMITTED,
        _context_for(
            ExecutionState.DOMAIN_VALIDATED,
            ExecutionState.ADMITTED,
            transition_id="regression-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )

    assert not result.allowed
    assert "state_transition_not_adjacent" in result.errors


def test_parallel_state_conflict_blocks_transition() -> None:
    admitted = validate_transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        _context_for(
            ExecutionState.QUARANTINED,
            ExecutionState.ADMITTED,
            transition_id="parallel-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )
    assert admitted.allowed
    chain = append_transition((), admitted)

    result = validate_transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        _context_for(
            ExecutionState.QUARANTINED,
            ExecutionState.ADMITTED,
            transition_id="parallel-002",
            parent_hash=admitted.transition_hash,
            chain=chain,
        ),
    )

    assert not result.allowed
    assert "parallel_state_conflict" in result.errors


def test_terminal_state_conflict_blocks_multiple_executed_states() -> None:
    chain: tuple[dict[str, str], ...] = ()
    parent_hash = ROOT_PARENT_HASH
    state = ExecutionState.QUARANTINED
    for index, next_state in enumerate(
        (
            ExecutionState.ADMITTED,
            ExecutionState.DOMAIN_VALIDATED,
            ExecutionState.MATERIALIZATION_PREPARED,
            ExecutionState.REQUEST_CREATED,
            ExecutionState.REQUEST_PRECHECKED,
            ExecutionState.REQUEST_APPROVED,
            ExecutionState.EXECUTION_AUTHORIZED,
            ExecutionState.EXECUTED,
        ),
        start=1,
    ):
        result = validate_transition(
            state,
            next_state,
            _context_for(
                state,
                next_state,
                transition_id=f"terminal-{index:03d}",
                parent_hash=parent_hash,
                chain=chain,
            ),
        )
        assert result.allowed, result.errors
        chain = append_transition(chain, result)
        parent_hash = result.transition_hash
        state = next_state

    result = validate_transition(
        ExecutionState.EXECUTION_AUTHORIZED,
        ExecutionState.EXECUTED,
        _context_for(
            ExecutionState.EXECUTION_AUTHORIZED,
            ExecutionState.EXECUTED,
            transition_id="terminal-replay-001",
            parent_hash=parent_hash,
            chain=chain,
        ),
    )

    assert not result.allowed
    assert "terminal_state_already_reached" in result.errors


def test_filename_alone_cannot_infer_state() -> None:
    context = _context_for(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="filename-inference-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    context.pop("state_before")
    evidence = dict(context["evidence"])
    evidence["artifact_basename"] = "manually-claimed-executed-state.json"
    context["evidence"] = evidence
    context["evidence_hash"] = hash_evidence(evidence)

    result = validate_transition(ExecutionState.QUARANTINED, ExecutionState.ADMITTED, context)

    assert not result.allowed
    assert "missing_explicit_state_before" in result.errors


def test_provenance_chain_rejects_missing_intermediate_state() -> None:
    admitted = validate_transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        _context_for(
            ExecutionState.QUARANTINED,
            ExecutionState.ADMITTED,
            transition_id="missing-chain-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )
    assert admitted.allowed
    domain_validated = validate_transition(
        ExecutionState.ADMITTED,
        ExecutionState.DOMAIN_VALIDATED,
        _context_for(
            ExecutionState.ADMITTED,
            ExecutionState.DOMAIN_VALIDATED,
            transition_id="missing-chain-002",
            parent_hash=admitted.transition_hash,
            chain=append_transition((), admitted),
        ),
    )
    assert domain_validated.allowed

    broken_chain = (admitted.transition_record, domain_validated.transition_record)

    result = validate_provenance_chain(broken_chain, require_terminal_state=ExecutionState.EXECUTED)

    assert not result.allowed
    assert "provenance_chain_missing_terminal_state" in result.errors


def _context_for(
    from_state: ExecutionState,
    to_state: ExecutionState,
    *,
    transition_id: str,
    parent_hash: str,
    chain: tuple[dict[str, str], ...],
) -> dict[str, object]:
    evidence = {
        "redaction_status": "passed",
        "upstream_evidence": {
            label: _safe_hash(f"{transition_id}:{label}")
            for label in REQUIRED_EVIDENCE_BY_TARGET_STATE[to_state]
        },
        "artifact_basename": f"{transition_id}-evidence.json",
    }
    return {
        "transition_id": transition_id,
        "parent_hash": parent_hash,
        "state_before": from_state.value,
        "state_after": to_state.value,
        "evidence": evidence,
        "evidence_hash": hash_evidence(evidence),
        "timestamp": FIXED_TIMESTAMP,
        "chain": chain,
    }


def _safe_hash(value: str) -> str:
    encoded = json.dumps(value, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
