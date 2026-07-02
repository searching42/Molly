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
    PIPELINE_STAGE_STATE_MAPPING,
    REQUIRED_EVIDENCE_BY_TARGET_STATE,
    validate_transition,
)


FIXED_TIMESTAMP = "2026-01-01T00:00:00Z"


def test_valid_linear_progression_reaches_executed_with_hash_chain() -> None:
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
                transition_id=f"transition-{index:03d}",
                parent_hash=parent_hash,
                chain=chain,
            ),
        )

        assert result.allowed, result.errors
        assert result.transition_record is not None
        assert result.transition_hash.startswith("sha256:")

        chain = append_transition(chain, result)
        parent_hash = result.transition_hash
        state = next_state

    assert chain[-1]["state_after"] == "EXECUTED"
    assert validate_provenance_chain(chain).allowed


def test_forbidden_jump_from_quarantined_to_executed_fails() -> None:
    result = validate_transition(
        ExecutionState.QUARANTINED,
        ExecutionState.EXECUTED,
        _context_for(
            ExecutionState.QUARANTINED,
            ExecutionState.EXECUTED,
            transition_id="forbidden-jump-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )

    assert not result.allowed
    assert "state_transition_not_adjacent" in result.errors


def test_forged_executed_state_without_intermediate_chain_fails() -> None:
    result = validate_transition(
        ExecutionState.EXECUTION_AUTHORIZED,
        ExecutionState.EXECUTED,
        _context_for(
            ExecutionState.EXECUTION_AUTHORIZED,
            ExecutionState.EXECUTED,
            transition_id="forged-executed-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )

    assert not result.allowed
    assert "missing_intermediate_provenance_chain" in result.errors


def test_replayed_transition_id_fails() -> None:
    admitted = validate_transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        _context_for(
            ExecutionState.QUARANTINED,
            ExecutionState.ADMITTED,
            transition_id="replayed-transition-001",
            parent_hash=ROOT_PARENT_HASH,
            chain=(),
        ),
    )
    assert admitted.allowed
    chain = append_transition((), admitted)

    replay = validate_transition(
        ExecutionState.ADMITTED,
        ExecutionState.DOMAIN_VALIDATED,
        _context_for(
            ExecutionState.ADMITTED,
            ExecutionState.DOMAIN_VALIDATED,
            transition_id="replayed-transition-001",
            parent_hash=admitted.transition_hash,
            chain=chain,
        ),
    )

    assert not replay.allowed
    assert "transition_id_replayed" in replay.errors


def test_missing_provenance_hash_chain_fails() -> None:
    context = _context_for(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="missing-provenance-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    context.pop("parent_hash")

    result = validate_transition(ExecutionState.QUARANTINED, ExecutionState.ADMITTED, context)

    assert not result.allowed
    assert "missing_parent_hash" in result.errors


def test_existing_pipeline_stages_map_to_execution_states() -> None:
    assert PIPELINE_STAGE_STATE_MAPPING["dry-run"] is ExecutionState.MATERIALIZATION_PREPARED
    assert PIPELINE_STAGE_STATE_MAPPING["precheck"] is ExecutionState.REQUEST_PRECHECKED
    assert PIPELINE_STAGE_STATE_MAPPING["execution request"] is ExecutionState.REQUEST_CREATED
    assert PIPELINE_STAGE_STATE_MAPPING["execution preflight"] is ExecutionState.REQUEST_APPROVED


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
