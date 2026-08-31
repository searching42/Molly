"""Deterministic projection from Core inspection facts to a safe trace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molly.core.agent_loop import (
    APPROVAL_RECORDED,
    APPROVAL_REQUIRED,
    DECISION_RECORDED,
    REVIEW_REQUESTED,
    RUN_FAILED,
    RUN_STOPPED,
    TOOL_CALL_MATERIALIZED,
    TOOL_CALL_REJECTED,
    TOOL_EXECUTION_FAILED,
    TOOL_EXECUTION_STARTED,
    TOOL_EXECUTION_SUCCEEDED,
)
from molly.core.ids import canonical_json_bytes, sha256_bytes
from molly.core.inspection import RunInspection, RunInspector, ToolCallInspection
from molly.core.ledger import LedgerEvent

from .model import RunTrace, TraceEvent, TraceSpan


_SAFE_PROFILE_KEYS = frozenset(
    {
        "profile_ref",
        "provider_profile_ref",
        "model_identifier",
        "model_version",
        "job_id",
        "job_handle_id",
        "config_digest",
        "task_digest",
        "execution_config_digest",
        "seed",
        "seed_value",
        "resource_profile_ref",
    }
)
_CALL_EVENT_TYPES = frozenset(
    {
        TOOL_CALL_MATERIALIZED,
        APPROVAL_REQUIRED,
        APPROVAL_RECORDED,
        TOOL_CALL_REJECTED,
        TOOL_EXECUTION_STARTED,
        TOOL_EXECUTION_SUCCEEDED,
        TOOL_EXECUTION_FAILED,
    }
)


def trace_id_for_run(run_id: str) -> str:
    return sha256_bytes(canonical_json_bytes({"namespace": "molly.trace.v1", "run_id": run_id}))[:32]


def span_id_for(trace_id: str, kind: str, identity: str) -> str:
    return sha256_bytes(
        canonical_json_bytes({"trace_id": trace_id, "kind": kind, "identity": identity})
    )[:16]


def _safe_mapping_attributes(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or (key not in _SAFE_PROFILE_KEYS and not key.endswith("_digest")):
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
        elif isinstance(item, (list, tuple)):
            result[key] = [child for child in item if isinstance(child, (str, int, float, bool)) or child is None]
    return result


def _event_attributes(event: LedgerEvent) -> dict[str, Any]:
    attributes: dict[str, Any] = {"event_type": event.event_type}
    if event.status is not None:
        attributes["status"] = event.status
    call_id = event.metadata.get("call_id")
    if isinstance(call_id, str):
        attributes["call_id"] = call_id
    call_digest = event.metadata.get("tool_call_digest")
    if isinstance(call_digest, str):
        attributes["tool_call_digest"] = call_digest
    if event.tool_name is not None:
        attributes["tool_name"] = event.tool_name
    if event.tool_version is not None:
        attributes["tool_version"] = event.tool_version
    for mapping in (event.model_profile, event.provider_profile, event.metadata):
        attributes.update(_safe_mapping_attributes(mapping))
    for name, value in (
        ("prompt_digest", event.prompt_digest),
        ("config_digest", event.config_digest),
    ):
        if value is not None:
            attributes[name] = value
    if event.seed_metadata:
        attributes.update(_safe_mapping_attributes(event.seed_metadata))
    if event.event_type == APPROVAL_RECORDED:
        approval = event.metadata.get("approval")
        if isinstance(approval, Mapping):
            for key in ("decision", "reviewer_ref"):
                value = approval.get(key)
                if isinstance(value, str):
                    attributes[key] = value
    return attributes


def _root_attributes(inspection: RunInspection) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "run_id": inspection.run_id,
        "request_digest": inspection.request_digest,
        "policy_digest": inspection.policy_digest,
        "status": inspection.status,
        "decision_count": inspection.decision_count,
        "tool_call_count": inspection.tool_call_count,
        "step_count": inspection.step_count,
        "initial_artifact_ids": list(inspection.initial_artifact_ids),
        "final_artifact_ids": list(inspection.final_artifact_ids),
        "ledger_sha256": inspection.ledger_sha256,
        "lineage_sha256": inspection.lineage_sha256,
    }
    if inspection.runtime_profile_ref is not None:
        attributes["runtime_profile_ref"] = inspection.runtime_profile_ref
    if inspection.runtime_profile_digest is not None:
        attributes["runtime_profile_digest"] = inspection.runtime_profile_digest
    return attributes


def _call_span_attributes(call: ToolCallInspection) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "call_id": call.call_id,
        "step_id": call.step_id,
        "tool_name": call.tool_name,
        "tool_version": call.tool_version,
        "tool_spec_digest": call.tool_spec_digest,
        "tool_call_digest": call.tool_call_digest,
        "input_artifact_ids": list(call.input_artifact_ids),
        "output_artifact_ids": list(call.output_artifact_ids),
        "approval_required": call.approval_required,
        "execution_status": call.execution_status,
    }
    if call.approval_status is not None:
        attributes["approval_status"] = call.approval_status
    if call.result_data_sha256 is not None:
        attributes["result_data_sha256"] = call.result_data_sha256
    if call.failure_type is not None:
        attributes["failure_type"] = call.failure_type
    return attributes


class RunTraceProjector:
    """Build a deterministic trace without creating any runtime side effect."""

    def __init__(self, inspector: RunInspector) -> None:
        if not isinstance(inspector, RunInspector):
            raise TypeError("RunTraceProjector requires a RunInspector")
        self.inspector = inspector

    def project_run(self, run_id: str) -> RunTrace:
        inspection = self.inspector.inspect_run(run_id)
        events = self.inspector.events_for_run(run_id)
        trace_id = trace_id_for_run(inspection.run_id)
        root_id = span_id_for(trace_id, "run", inspection.run_id)
        root_events: list[TraceEvent] = []
        events_by_id = {event.event_id: event for event in events}
        for event in events:
            if event.event_type not in _CALL_EVENT_TYPES:
                root_events.append(
                    TraceEvent(
                        name=f"molly.event.{event.event_type.lower()}",
                        timestamp=event.timestamp,
                        attributes=_event_attributes(event),
                    )
                )
        spans = [
            TraceSpan(
                span_id=root_id,
                parent_span_id=None,
                name="molly.run",
                start_time=events[0].timestamp,
                end_time=events[-1].timestamp,
                attributes=_root_attributes(inspection),
                events=tuple(root_events),
            )
        ]
        for call in inspection.materialized_calls:
            call_events = [events_by_id[event_id] for event_id in call.event_ids]
            span_events = tuple(
                TraceEvent(
                    name=f"molly.event.{event.event_type.lower()}",
                    timestamp=event.timestamp,
                    attributes=_event_attributes(event),
                )
                for event in call_events
            )
            span_id = span_id_for(trace_id, "call", call.call_id)
            spans.append(
                TraceSpan(
                    span_id=span_id,
                    parent_span_id=root_id,
                    name=f"molly.tool.{call.tool_name}",
                    start_time=call_events[0].timestamp,
                    end_time=call_events[-1].timestamp,
                    attributes=_call_span_attributes(call),
                    events=span_events,
                )
            )
        return RunTrace(
            trace_id=trace_id,
            run_id=inspection.run_id,
            status=inspection.status,
            source_ledger_sha256=inspection.ledger_sha256,
            source_lineage_sha256=inspection.lineage_sha256,
            spans=tuple(spans),
        )


__all__ = ["RunTraceProjector", "span_id_for", "trace_id_for_run"]
