from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .execution_state import ExecutionState, NEXT_STATE_BY_STATE
from .invariants import is_sha256, is_valid_transition_timestamp


ROOT_PARENT_HASH = "sha256:" + ("0" * 64)
TRANSITION_RECORD_FIELDS = (
    "transition_id",
    "parent_hash",
    "state_before",
    "state_after",
    "evidence_hash",
    "timestamp",
)


@dataclass(frozen=True)
class ProvenanceValidationResult:
    allowed: bool
    errors: tuple[str, ...] = ()


def hash_evidence(evidence: Any) -> str:
    return _hash_payload(evidence)


def hash_transition_record(record: dict[str, Any]) -> str:
    payload = {field: record[field] for field in TRANSITION_RECORD_FIELDS}
    return _hash_payload(payload)


def make_transition_record(
    *,
    transition_id: str,
    parent_hash: str,
    state_before: ExecutionState,
    state_after: ExecutionState,
    evidence_hash: str,
    timestamp: str,
) -> dict[str, str]:
    record = {
        "transition_id": transition_id,
        "parent_hash": parent_hash,
        "state_before": state_before.value,
        "state_after": state_after.value,
        "evidence_hash": evidence_hash,
        "timestamp": timestamp,
    }
    record["transition_hash"] = hash_transition_record(record)
    return record


def append_transition(chain: Iterable[dict[str, str]], result: Any) -> tuple[dict[str, str], ...]:
    if not getattr(result, "allowed", False) or result.transition_record is None:
        raise ValueError("cannot_append_failed_transition")
    return (*tuple(chain), dict(result.transition_record))


def validate_provenance_chain(
    chain: Iterable[dict[str, Any]],
    *,
    require_terminal_state: ExecutionState | None = None,
) -> ProvenanceValidationResult:
    records = tuple(chain)
    errors: list[str] = []
    if not records:
        if require_terminal_state is not None:
            return ProvenanceValidationResult(False, ("provenance_chain_missing_terminal_state",))
        return ProvenanceValidationResult(True)

    expected_parent_hash = ROOT_PARENT_HASH
    expected_state_before = ExecutionState.QUARANTINED
    seen_transition_ids: set[str] = set()

    for index, record in enumerate(records):
        missing = [field for field in (*TRANSITION_RECORD_FIELDS, "transition_hash") if field not in record]
        if missing:
            errors.append("provenance_record_missing_required_fields")
            continue

        transition_id = record["transition_id"]
        if transition_id in seen_transition_ids:
            errors.append("transition_id_replayed")
        seen_transition_ids.add(transition_id)

        parent_hash = record["parent_hash"]
        if not is_sha256(parent_hash):
            errors.append("invalid_parent_hash")
        elif parent_hash != expected_parent_hash:
            errors.append("parent_hash_mismatch")

        if not is_sha256(record["evidence_hash"]):
            errors.append("invalid_evidence_hash")
        if not is_valid_transition_timestamp(record["timestamp"]):
            errors.append("invalid_transition_timestamp")

        try:
            state_before = ExecutionState(record["state_before"])
            state_after = ExecutionState(record["state_after"])
        except ValueError:
            errors.append("invalid_execution_state")
            continue

        if index == 0 and parent_hash != ROOT_PARENT_HASH:
            errors.append("missing_root_parent_hash")
        if state_before is not expected_state_before:
            errors.append("provenance_chain_state_gap")
        if NEXT_STATE_BY_STATE.get(state_before) is not state_after:
            errors.append("provenance_chain_not_adjacent")

        transition_hash = record["transition_hash"]
        expected_transition_hash = hash_transition_record(dict(record))
        if not is_sha256(transition_hash):
            errors.append("invalid_transition_hash")
        elif transition_hash != expected_transition_hash:
            errors.append("transition_hash_mismatch")

        expected_parent_hash = transition_hash
        expected_state_before = state_after

    if require_terminal_state is not None and records[-1].get("state_after") != require_terminal_state.value:
        errors.append("provenance_chain_missing_terminal_state")

    return ProvenanceValidationResult(not errors, tuple(dict.fromkeys(errors)))


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
