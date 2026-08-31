"""Current-run binding helpers based on factual RunLedger occurrences."""

from __future__ import annotations

from collections.abc import Iterable

from molly.core.agent_loop import TOOL_EXECUTION_SUCCEEDED
from molly.core.ids import validate_artifact_id, validate_identifier
from molly.core.ledger import LedgerEvent, RunLedger

from .errors import Br1BindingError


def successful_output_event(
    ledger: RunLedger,
    *,
    run_id: str,
    tool_name: str,
    artifact_id: str,
    required_inputs: Iterable[str] = (),
) -> LedgerEvent:
    """Return a success occurrence in ``run_id`` proving an output binding.

    Aggregate lineage is intentionally not used for current-run authority.
    The RunLedger success fact, including its declared inputs, is the source
    of truth for this check.
    """

    validate_identifier(run_id, field="run_id")
    validate_identifier(tool_name, field="tool_name")
    validate_artifact_id(artifact_id)
    expected_inputs = tuple(required_inputs)
    for item in expected_inputs:
        validate_artifact_id(item)
    matches = [
        event
        for event in ledger.for_run(run_id)
        if event.event_type == TOOL_EXECUTION_SUCCEEDED
        and event.tool_name == tool_name
        and artifact_id in event.output_artifact_ids
        and all(item in event.input_artifact_ids for item in expected_inputs)
    ]
    if not matches:
        raise Br1BindingError(
            f"artifact {artifact_id} is not a successful {tool_name} output in the current run"
        )
    return matches[-1]


def require_current_run_chain(
    ledger: RunLedger,
    *,
    run_id: str,
    artifact_tool_pairs: Iterable[tuple[str, str]],
) -> tuple[LedgerEvent, ...]:
    events = []
    for artifact_id, tool_name in artifact_tool_pairs:
        events.append(
            successful_output_event(
                ledger,
                run_id=run_id,
                tool_name=tool_name,
                artifact_id=artifact_id,
            )
        )
    return tuple(events)


__all__ = ["require_current_run_chain", "successful_output_event"]
