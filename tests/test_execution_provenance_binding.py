from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from ai4s_agent.provenance.artifact_registry import ArtifactRegistry
from ai4s_agent.provenance.execution_binding import create_execution_binding
from ai4s_agent.provenance.hash_linker import hash_artifact_payload
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


def test_valid_binding_chain_links_state_artifact_hash_and_graph() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="binding-transition-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    artifact = {"artifact_id": "dry-run-report-001", "status": "passed", "count": 3}

    binding_result = create_execution_binding(
        transition_result=transition,
        artifact_payload=artifact,
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
    )

    assert binding_result.allowed, binding_result.errors
    binding = binding_result.binding
    assert binding is not None
    assert binding.transition_id == "binding-transition-001"
    assert binding.artifact_hash == hash_artifact_payload(artifact)
    assert binding.parent_transition_hash == ROOT_PARENT_HASH

    registry_result = ArtifactRegistry.empty().register(binding)
    assert registry_result.allowed, registry_result.errors
    graph_result = validate_provenance_graph(
        transitions=append_transition((), transition),
        artifact_registry=registry_result.registry,
    )

    assert graph_result.allowed, graph_result.errors


def test_orphan_artifact_without_transition_fails_graph_validation() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="orphan-transition-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    binding = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "orphan-artifact-001"},
        artifact_type="precheck_summary",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert binding is not None
    registry = ArtifactRegistry.empty().register(binding).registry

    graph_result = validate_provenance_graph(transitions=(), artifact_registry=registry)

    assert not graph_result.allowed
    assert "artifact_transition_not_found" in graph_result.errors


def test_forged_transition_id_with_valid_artifact_fails_graph_validation() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="real-transition-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    binding = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "request-artifact-001"},
        artifact_type="execution_request",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert binding is not None
    forged_binding = replace(binding, transition_id="forged-transition-001")
    registry = ArtifactRegistry.empty().register(forged_binding).registry

    graph_result = validate_provenance_graph(
        transitions=append_transition((), transition),
        artifact_registry=registry,
    )

    assert not graph_result.allowed
    assert "artifact_transition_not_found" in graph_result.errors


def test_hash_mismatch_attack_fails_binding_creation() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="hash-attack-transition-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )

    result = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "hash-attack-artifact-001"},
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
        declared_artifact_hash=_safe_hash("wrong-artifact"),
    )

    assert not result.allowed
    assert "artifact_hash_mismatch" in result.errors


def test_replay_attack_reusing_transition_with_different_artifact_fails() -> None:
    transition = _transition(
        ExecutionState.QUARANTINED,
        ExecutionState.ADMITTED,
        transition_id="replay-transition-001",
        parent_hash=ROOT_PARENT_HASH,
        chain=(),
    )
    first = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "first-artifact"},
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    second = create_execution_binding(
        transition_result=transition,
        artifact_payload={"artifact_id": "second-artifact"},
        artifact_type="dry_run_report",
        timestamp=FIXED_TIMESTAMP,
    ).binding
    assert first is not None
    assert second is not None

    registry_result = ArtifactRegistry.empty().register(first)
    replay_result = registry_result.registry.register(second)

    assert not replay_result.allowed
    assert "transition_already_bound_to_artifact" in replay_result.errors


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
