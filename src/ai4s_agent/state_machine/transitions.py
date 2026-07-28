from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_state import ExecutionState, NEXT_STATE_BY_STATE
from .invariants import (
    is_safe_id,
    is_sha256,
    is_valid_transition_timestamp,
    redaction_errors,
)
from .provenance_chain import (
    ROOT_PARENT_HASH,
    hash_evidence,
    hash_transition_record,
    make_transition_record,
    validate_provenance_chain,
)


REQUIRED_EVIDENCE_BY_TARGET_STATE: dict[ExecutionState, tuple[str, ...]] = {
    ExecutionState.ADMITTED: ("quarantine_evidence",),
    ExecutionState.DOMAIN_VALIDATED: ("admission_evidence", "domain_review"),
    ExecutionState.MATERIALIZATION_PREPARED: ("domain_validation", "dry_run"),
    ExecutionState.REQUEST_CREATED: ("materialization_prepared", "execution_request"),
    ExecutionState.REQUEST_PRECHECKED: ("execution_request", "precheck"),
    ExecutionState.REQUEST_APPROVED: ("request_precheck", "approval_policy"),
    ExecutionState.EXECUTION_AUTHORIZED: ("request_approval", "explicit_confirmation"),
    ExecutionState.EXECUTED: ("execution_gate",),
}

PIPELINE_STAGE_STATE_MAPPING: dict[str, ExecutionState] = {
    "dry-run": ExecutionState.MATERIALIZATION_PREPARED,
    "precheck": ExecutionState.REQUEST_PRECHECKED,
    "execution request": ExecutionState.REQUEST_CREATED,
    "execution preflight": ExecutionState.REQUEST_APPROVED,
}


@dataclass(frozen=True)
class TransitionResult:
    allowed: bool
    from_state: ExecutionState
    to_state: ExecutionState
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    transition_record: dict[str, str] | None = None
    transition_hash: str = ""


def validate_transition(
    from_state: ExecutionState,
    to_state: ExecutionState,
    context: dict[str, Any],
) -> TransitionResult:
    """Validate one explicit, provenance-bound state transition."""

    errors: list[str] = []
    if not isinstance(from_state, ExecutionState) or not isinstance(to_state, ExecutionState):
        errors.append("invalid_execution_state")
        return _blocked(from_state, to_state, errors)

    if NEXT_STATE_BY_STATE.get(from_state) is not to_state:
        errors.append("state_transition_not_adjacent")

    if not isinstance(context, dict):
        errors.append("invalid_transition_context")
        return _blocked(from_state, to_state, errors)

    transition_id = context.get("transition_id")
    if transition_id is None:
        errors.append("missing_transition_id")
    elif not is_safe_id(transition_id):
        errors.append("unsafe_transition_id")

    if "state_before" not in context:
        errors.append("missing_explicit_state_before")
    elif context["state_before"] != from_state.value:
        errors.append("explicit_state_before_mismatch")

    if "state_after" not in context:
        errors.append("missing_explicit_state_after")
    elif context["state_after"] != to_state.value:
        errors.append("explicit_state_after_mismatch")

    parent_hash = context.get("parent_hash")
    if parent_hash is None:
        errors.append("missing_parent_hash")
    elif not is_sha256(parent_hash):
        errors.append("invalid_parent_hash")

    timestamp = context.get("timestamp")
    if timestamp is None:
        errors.append("missing_transition_timestamp")
    elif not is_valid_transition_timestamp(timestamp):
        errors.append("invalid_transition_timestamp")

    evidence = context.get("evidence")
    evidence_hash = context.get("evidence_hash")
    if not isinstance(evidence, dict):
        errors.append("missing_transition_evidence")
    else:
        _validate_required_evidence(to_state, evidence, errors)
        if redaction_errors({"transition_id": transition_id, "evidence": evidence}):
            errors.append("redaction_invariant_failed")

    if evidence_hash is None:
        errors.append("missing_evidence_hash")
    elif not is_sha256(evidence_hash):
        errors.append("invalid_evidence_hash")
    elif isinstance(evidence, dict) and evidence_hash != hash_evidence(evidence):
        errors.append("evidence_hash_mismatch")

    chain = context.get("chain", ())
    if not isinstance(chain, (list, tuple)):
        errors.append("invalid_provenance_chain")
        chain_records: tuple[dict[str, Any], ...] = ()
    else:
        chain_records = tuple(chain)

    if chain_records:
        chain_result = validate_provenance_chain(chain_records)
        if not chain_result.allowed:
            errors.extend(chain_result.errors)

        if any(record.get("state_after") == ExecutionState.EXECUTED.value for record in chain_records):
            errors.append("terminal_state_already_reached")

        if any(record.get("transition_id") == transition_id for record in chain_records):
            errors.append("transition_id_replayed")

        last_record = chain_records[-1]
        if last_record.get("state_after") != from_state.value:
            errors.append("parallel_state_conflict")

        last_hash = last_record.get("transition_hash")
        if not last_hash and all(field in last_record for field in ("transition_id", "parent_hash", "state_before", "state_after", "evidence_hash", "timestamp")):
            last_hash = hash_transition_record(dict(last_record))
        if parent_hash is not None and is_sha256(parent_hash) and last_hash != parent_hash:
            errors.append("parent_hash_mismatch")
    else:
        if from_state is not ExecutionState.QUARANTINED:
            errors.append("missing_intermediate_provenance_chain")
        if parent_hash is not None and is_sha256(parent_hash) and parent_hash != ROOT_PARENT_HASH:
            errors.append("parent_hash_mismatch")

    if errors:
        return _blocked(from_state, to_state, errors)

    transition_record = make_transition_record(
        transition_id=str(transition_id),
        parent_hash=str(parent_hash),
        state_before=from_state,
        state_after=to_state,
        evidence_hash=str(evidence_hash),
        timestamp=str(timestamp),
    )
    return TransitionResult(
        True,
        from_state,
        to_state,
        transition_record=transition_record,
        transition_hash=transition_record["transition_hash"],
    )


def _validate_required_evidence(
    to_state: ExecutionState,
    evidence: dict[str, Any],
    errors: list[str],
) -> None:
    required = REQUIRED_EVIDENCE_BY_TARGET_STATE.get(to_state, ())
    upstream_evidence = evidence.get("upstream_evidence")
    if not isinstance(upstream_evidence, dict):
        errors.append("missing_required_upstream_evidence")
        return

    missing = [label for label in required if label not in upstream_evidence]
    if missing:
        errors.append("missing_required_upstream_evidence")

    invalid_hashes = [
        label
        for label in required
        if label in upstream_evidence and not is_sha256(upstream_evidence[label])
    ]
    if invalid_hashes:
        errors.append("invalid_upstream_evidence_hash")

    if evidence.get("redaction_status") != "passed":
        errors.append("redaction_status_not_passed")


def _blocked(
    from_state: ExecutionState,
    to_state: ExecutionState,
    errors: list[str],
) -> TransitionResult:
    return TransitionResult(
        False,
        from_state,
        to_state,
        errors=tuple(dict.fromkeys(errors)),
    )
