from __future__ import annotations

from enum import Enum


class ExecutionState(str, Enum):
    """Immutable controlled-writer execution states."""

    QUARANTINED = "QUARANTINED"
    ADMITTED = "ADMITTED"
    DOMAIN_VALIDATED = "DOMAIN_VALIDATED"
    MATERIALIZATION_PREPARED = "MATERIALIZATION_PREPARED"
    REQUEST_CREATED = "REQUEST_CREATED"
    REQUEST_PRECHECKED = "REQUEST_PRECHECKED"
    REQUEST_APPROVED = "REQUEST_APPROVED"
    EXECUTION_AUTHORIZED = "EXECUTION_AUTHORIZED"
    EXECUTED = "EXECUTED"


EXECUTION_STATE_ORDER: tuple[ExecutionState, ...] = (
    ExecutionState.QUARANTINED,
    ExecutionState.ADMITTED,
    ExecutionState.DOMAIN_VALIDATED,
    ExecutionState.MATERIALIZATION_PREPARED,
    ExecutionState.REQUEST_CREATED,
    ExecutionState.REQUEST_PRECHECKED,
    ExecutionState.REQUEST_APPROVED,
    ExecutionState.EXECUTION_AUTHORIZED,
    ExecutionState.EXECUTED,
)

NEXT_STATE_BY_STATE: dict[ExecutionState, ExecutionState] = {
    state: EXECUTION_STATE_ORDER[index + 1]
    for index, state in enumerate(EXECUTION_STATE_ORDER[:-1])
}
