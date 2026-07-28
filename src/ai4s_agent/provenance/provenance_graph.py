from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ai4s_agent.state_machine.execution_state import ExecutionState
from ai4s_agent.state_machine.invariants import is_sha256, redaction_errors
from ai4s_agent.state_machine.provenance_chain import validate_provenance_chain

from .artifact_registry import ArtifactRegistry
from .execution_binding import ExecutionBinding


@dataclass(frozen=True)
class ProvenanceGraphValidationResult:
    allowed: bool
    errors: tuple[str, ...] = ()


def validate_provenance_graph(
    *,
    transitions: Iterable[dict[str, Any]],
    artifact_registry: ArtifactRegistry,
    require_terminal_artifact: bool = False,
) -> ProvenanceGraphValidationResult:
    errors: list[str] = []
    transition_records = tuple(transitions)

    chain_result = validate_provenance_chain(transition_records)
    if not chain_result.allowed:
        errors.extend(chain_result.errors)

    if not isinstance(artifact_registry, ArtifactRegistry):
        errors.append("invalid_artifact_registry")
        return _result(errors)

    transition_counts: dict[str, int] = {}
    transitions_by_id: dict[str, dict[str, Any]] = {}
    for record in transition_records:
        transition_id = str(record.get("transition_id", ""))
        transition_counts[transition_id] = transition_counts.get(transition_id, 0) + 1
        transitions_by_id.setdefault(transition_id, record)

    _validate_registry_records(artifact_registry.records, transitions_by_id, transition_counts, errors)

    if require_terminal_artifact:
        terminal_transition_ids = {
            str(record.get("transition_id"))
            for record in transition_records
            if record.get("state_after") == ExecutionState.EXECUTED.value
        }
        bound_transition_ids = {record.transition_id for record in artifact_registry.records}
        if terminal_transition_ids and not terminal_transition_ids.issubset(bound_transition_ids):
            errors.append("terminal_transition_missing_artifact")

    return _result(errors)


def _validate_registry_records(
    records: tuple[ExecutionBinding, ...],
    transitions_by_id: dict[str, dict[str, Any]],
    transition_counts: dict[str, int],
    errors: list[str],
) -> None:
    seen_artifact_hashes: set[str] = set()
    seen_transition_ids: set[str] = set()

    for binding in records:
        if not isinstance(binding, ExecutionBinding):
            errors.append("invalid_artifact_binding")
            continue
        if redaction_errors(binding.to_record()):
            errors.append("artifact_binding_redaction_failed")
        if not is_sha256(binding.artifact_hash):
            errors.append("invalid_artifact_hash")
        if not is_sha256(binding.transition_hash):
            errors.append("invalid_artifact_transition_hash")
        if not is_sha256(binding.parent_transition_hash):
            errors.append("invalid_parent_transition_hash")

        if binding.artifact_hash in seen_artifact_hashes:
            errors.append("duplicate_artifact_binding")
        seen_artifact_hashes.add(binding.artifact_hash)

        if binding.transition_id in seen_transition_ids:
            errors.append("transition_already_bound_to_artifact")
        seen_transition_ids.add(binding.transition_id)

        transition_count = transition_counts.get(binding.transition_id, 0)
        if transition_count == 0:
            errors.append("artifact_transition_not_found")
            continue
        if transition_count > 1:
            errors.append("artifact_transition_not_unique")
            continue

        transition = transitions_by_id[binding.transition_id]
        if binding.transition_hash != transition.get("transition_hash"):
            errors.append("artifact_transition_hash_mismatch")
        if binding.parent_transition_hash != transition.get("parent_hash"):
            errors.append("artifact_parent_transition_hash_mismatch")
        if binding.state_before != transition.get("state_before"):
            errors.append("artifact_state_before_mismatch")
        if binding.state_after != transition.get("state_after"):
            errors.append("artifact_state_after_mismatch")


def _result(errors: list[str]) -> ProvenanceGraphValidationResult:
    return ProvenanceGraphValidationResult(not errors, tuple(dict.fromkeys(errors)))
