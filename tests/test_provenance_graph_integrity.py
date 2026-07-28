from __future__ import annotations

import hashlib
import json

from ai4s_agent.provenance.artifact_registry import ArtifactRegistry
from ai4s_agent.provenance.execution_binding import create_execution_binding
from ai4s_agent.provenance.provenance_graph import validate_provenance_graph
from ai4s_agent.state_machine.execution_state import ExecutionState
from ai4s_agent.state_machine.provenance_chain import (
    ROOT_PARENT_HASH,
    append_transition,
    hash_evidence,
)
from ai4s_agent.state_machine.transitions import (
    REQUIRED_EVIDENCE_BY_TARGET_STATE,
    validate_transition,
)


FIXED_TIMESTAMP = "2026-01-01T00:00:00Z"


def test_terminal_executed_transition_requires_bound_artifact() -> None:
    transitions = _linear_executed_chain()

    result = validate_provenance_graph(
        transitions=transitions,
        artifact_registry=ArtifactRegistry.empty(),
        require_terminal_artifact=True,
    )

    assert not result.allowed
    assert "terminal_transition_missing_artifact" in result.errors


def test_terminal_executed_transition_with_bound_artifact_is_valid() -> None:
    transitions = _linear_executed_chain()
    terminal_transition = transitions[-1]
    binding_result = create_execution_binding(
        transition_record=terminal_transition,
        artifact_payload={"artifact_id": "terminal-execution-report", "status": "recorded"},
        artifact_type="execution_report",
        timestamp=FIXED_TIMESTAMP,
    )
    assert binding_result.allowed, binding_result.errors
    assert binding_result.binding is not None
    registry = ArtifactRegistry.empty().register(binding_result.binding).registry

    result = validate_provenance_graph(
        transitions=transitions,
        artifact_registry=registry,
        require_terminal_artifact=True,
    )

    assert result.allowed, result.errors


def test_broken_parent_chain_blocks_graph() -> None:
    transitions = list(_linear_executed_chain())
    transitions[2] = {**transitions[2], "parent_hash": _safe_hash("forged-parent")}

    result = validate_provenance_graph(
        transitions=tuple(transitions),
        artifact_registry=ArtifactRegistry.empty(),
        require_terminal_artifact=False,
    )

    assert not result.allowed
    assert "parent_hash_mismatch" in result.errors


def test_duplicate_artifact_hash_binding_is_rejected() -> None:
    first_transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="duplicate-artifact-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    chain = append_transition((), first_transition)
    second_transition = _transition(
        ExecutionState.ADMITTED,
        ExecutionState.DOMAIN_VALIDATED,
        transition_id="duplicate-artifact-002",
        parent_hash=first_transition.transition_hash,
        chain=chain,
    )
    artifact = {"artifact_id": "same-artifact-content"}
    first_binding = create_execution_binding(
        transition_result=first_transition,
        artifact_payload=artifact,
        artifact_type="precheck_summary",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    second_binding = create_execution_binding(
        transition_result=second_transition,
        artifact_payload=artifact,
        artifact_type="precheck_summary",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert first_binding is not None
    assert second_binding is not None

    registry = ArtifactRegistry.empty().register(first_binding).registry
    duplicate = registry.register(second_binding)

    assert not duplicate.allowed
    assert "artifact_hash_already_bound" in duplicate.errors


def test_registry_registration_is_immutable() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="immutable-registry-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    binding = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "immutable-artifact"},
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert binding is not None

    empty_registry = ArtifactRegistry.empty()
    registered = empty_registry.register(binding)

    assert len(empty_registry.records) == 0
    assert len(registered.registry.records) == 1


def test_artifact_binding_must_match_transition_hash() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="transition-hash-match-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    binding = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "hash-linked-artifact"},
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert binding is not None
    forged_binding = binding.__class__(
        **{
            **binding.to_record(),
            "transition_hash": _safe_hash("forged-transition-hash"),
        }
    )
    registry = ArtifactRegistry.empty().register(forged_binding).registry

    result = validate_provenance_graph(
        transitions=append_transition((), transition),
        artifact_registry=registry,
    )

    assert not result.allowed
    assert "artifact_transition_hash_mismatch" in result.errors


def _linear_executed_chain() -> tuple[dict[str, str], ...]:
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
        result = _transition(
            state,
            next_state,
            transition_id=f"graph-transition-{index:03d}",
            parent_hash=parent_hash,
            chain=chain,
        )
        chain = append_transition(chain, result)
        parent_hash = result.transition_hash
        state = next_state
    return chain


def _transition(
    from_state: ExecutionState,
    to_state: ExecutionState,
    *,
    transition_id: str,
    parent_hash: str,
    chain: tuple[dict[str, str], ...],
):
    result = validate_transition(
        from_state,
        to_state,
        _context_for(
            from_state,
            to_state,
            transition_id=transition_id,
            parent_hash=parent_hash,
            chain=chain,
        ),
    )
    assert result.allowed, result.errors
    return result


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
