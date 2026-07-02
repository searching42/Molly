from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai4s_agent.state_machine.invariants import (
    is_safe_id,
    is_sha256,
    is_valid_transition_timestamp,
    redaction_errors,
)
from ai4s_agent.state_machine.provenance_chain import hash_transition_record

from .hash_linker import hash_artifact_payload


@dataclass(frozen=True)
class ExecutionBinding:
    transition_id: str
    state_before: str
    state_after: str
    artifact_hash: str
    artifact_type: str
    parent_transition_hash: str
    timestamp: str
    transition_hash: str

    def to_record(self) -> dict[str, str]:
        return {
            "transition_id": self.transition_id,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "artifact_hash": self.artifact_hash,
            "artifact_type": self.artifact_type,
            "parent_transition_hash": self.parent_transition_hash,
            "timestamp": self.timestamp,
            "transition_hash": self.transition_hash,
        }


@dataclass(frozen=True)
class BindingResult:
    allowed: bool
    binding: ExecutionBinding | None = None
    errors: tuple[str, ...] = ()


def create_execution_binding(
    *,
    artifact_payload: Any,
    artifact_type: str,
    timestamp: str,
    transition_result: Any | None = None,
    transition_record: dict[str, Any] | None = None,
    declared_artifact_hash: str | None = None,
) -> BindingResult:
    errors: list[str] = []
    record = _transition_record_from(transition_result, transition_record, errors)
    if record is None:
        return BindingResult(False, errors=tuple(dict.fromkeys(errors)))

    if not is_safe_id(artifact_type):
        errors.append("unsafe_artifact_type")
    if not is_valid_transition_timestamp(timestamp):
        errors.append("invalid_binding_timestamp")
    if redaction_errors({"artifact_type": artifact_type, "artifact_payload": artifact_payload}):
        errors.append("artifact_redaction_invariant_failed")

    artifact_hash = hash_artifact_payload(artifact_payload)
    if declared_artifact_hash is not None:
        if not is_sha256(declared_artifact_hash):
            errors.append("invalid_declared_artifact_hash")
        elif declared_artifact_hash != artifact_hash:
            errors.append("artifact_hash_mismatch")

    transition_hash = record.get("transition_hash")
    if not transition_hash:
        transition_hash = hash_transition_record(record)
    if not is_sha256(transition_hash):
        errors.append("invalid_transition_hash")

    parent_hash = record.get("parent_hash")
    if not is_sha256(parent_hash):
        errors.append("invalid_parent_transition_hash")

    required_fields = ("transition_id", "state_before", "state_after")
    if any(not record.get(field) for field in required_fields):
        errors.append("transition_record_missing_required_fields")

    if errors:
        return BindingResult(False, errors=tuple(dict.fromkeys(errors)))

    binding = ExecutionBinding(
        transition_id=str(record["transition_id"]),
        state_before=str(record["state_before"]),
        state_after=str(record["state_after"]),
        artifact_hash=artifact_hash,
        artifact_type=artifact_type,
        parent_transition_hash=str(parent_hash),
        timestamp=timestamp,
        transition_hash=str(transition_hash),
    )
    return BindingResult(True, binding=binding)


def _transition_record_from(
    transition_result: Any | None,
    transition_record: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any] | None:
    if transition_record is not None:
        return dict(transition_record)

    if transition_result is None:
        errors.append("missing_transition_record")
        return None

    if not getattr(transition_result, "allowed", False):
        errors.append("transition_result_not_allowed")
        return None

    result_record = getattr(transition_result, "transition_record", None)
    if not isinstance(result_record, dict):
        errors.append("missing_transition_record")
        return None
    return dict(result_record)
