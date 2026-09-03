"""Read-only, deterministic projections of authoritative Molly Core facts.

Inspection deliberately lives beside the Core contracts but never writes to
the ledger, artifact store, or lineage.  It validates the three authoritative
stores before deriving a bounded operator-facing view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .agent_loop import (
    APPROVAL_RECORDED,
    APPROVAL_REQUIRED,
    BUDGET_EXHAUSTED,
    DECISION_RECORDED,
    REVIEW_REQUESTED,
    RUN_FAILED,
    RUN_REJECTED,
    RUN_STARTED,
    RUN_STOPPED,
    TOOL_CALL_MATERIALIZED,
    TOOL_CALL_REJECTED,
    TOOL_EXECUTION_FAILED,
    TOOL_EXECUTION_STARTED,
    TOOL_EXECUTION_SUCCEEDED,
)
from .approvals import ApprovalRecord
from .artifacts import ArtifactStore
from .errors import InspectionError, InspectionIntegrityError
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    freeze_json_value,
    sha256_bytes,
    thaw_json,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
)
from .ledger import LedgerEvent, RunLedger
from .lineage import ArtifactLineage, LineageRelation, RelationType
from .runs import RunRequest, RunStatus
from .tools import MAX_TOOL_RESULT_DATA_BYTES, MaterializedToolCall


_TERMINAL_RUN_EVENTS = frozenset({RUN_STOPPED, RUN_REJECTED, RUN_FAILED, BUDGET_EXHAUSTED})
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
_SAFE_METADATA_KEYS = frozenset(
    {
        "run_id",
        "call_id",
        "success_event_id",
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
        "status",
    }
)


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded logical references from occurrence metadata.

    Core metadata is already canonical JSON, but occurrence metadata may be
    extended by plugins.  Inspection exposes an allowlist rather than
    treating arbitrary plugin metadata as operator-safe output.
    """

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if key not in _SAFE_METADATA_KEYS and not (
            key.endswith("_id") or key.endswith("_ref") or key.endswith("_digest")
        ):
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
        elif isinstance(item, (list, tuple)):
            safe_items = [
                child
                for child in item
                if isinstance(child, (str, int, float, bool)) or child is None
            ]
            result[key] = safe_items
    return result


def _digest_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _artifact_ids_from_event(event: LedgerEvent) -> tuple[str, ...]:
    return (*event.input_artifact_ids, *event.output_artifact_ids)


@dataclass(frozen=True, slots=True)
class ToolCallInspection:
    """A read-only projection of one exact materialized tool call."""

    call_id: str
    step_id: str
    tool_name: str
    tool_version: str
    tool_spec_digest: str
    tool_call_digest: str
    arguments: Mapping[str, Any]
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    approval_required: bool
    approval_status: str | None
    execution_status: str
    reason_summary: str = ""
    result_data: Any = None
    result_data_sha256: str | None = None
    failure_type: str | None = None
    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.call_id, "call_id"),
            (self.step_id, "step_id"),
            (self.tool_name, "tool_name"),
            (self.tool_version, "tool_version"),
        ):
            validate_identifier(value, field=field_name)
        validate_digest_reference(self.tool_spec_digest, field="tool_spec_digest")
        validate_digest_reference(self.tool_call_digest, field="tool_call_digest")
        object.__setattr__(self, "arguments", freeze_json_mapping(self.arguments, field="inspection arguments"))
        object.__setattr__(
            self,
            "input_artifact_ids",
            tuple(validate_artifact_id(value, field="inspection input artifact") for value in self.input_artifact_ids),
        )
        object.__setattr__(
            self,
            "output_artifact_ids",
            tuple(validate_artifact_id(value, field="inspection output artifact") for value in self.output_artifact_ids),
        )
        if not isinstance(self.approval_required, bool):
            raise InspectionError("inspection approval_required must be boolean")
        if self.approval_status is not None:
            validate_identifier(self.approval_status, field="approval_status")
        validate_identifier(self.execution_status, field="execution_status")
        if not isinstance(self.reason_summary, str) or len(self.reason_summary) > 2_000:
            raise InspectionError("inspection reason_summary must be bounded text")
        if "\x00" in self.reason_summary:
            raise InspectionError("inspection reason_summary contains NUL")
        if self.result_data_sha256 is not None:
            validate_digest_reference(self.result_data_sha256, field="result_data_sha256")
            freeze_json_value(self.result_data, field="inspection result data")
        if self.failure_type is not None:
            validate_identifier(self.failure_type, field="failure_type")
        object.__setattr__(
            self,
            "event_ids",
            tuple(validate_identifier(value, field="inspection event_id") for value in self.event_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "call_id": self.call_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_spec_digest": self.tool_spec_digest,
            "tool_call_digest": self.tool_call_digest,
            "arguments": thaw_json(self.arguments),
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "approval_required": self.approval_required,
            "approval_status": self.approval_status,
            "execution_status": self.execution_status,
            "reason_summary": self.reason_summary,
            "event_ids": list(self.event_ids),
        }
        if self.result_data_sha256 is not None:
            value["result_data"] = thaw_json(self.result_data)
            value["result_data_sha256"] = self.result_data_sha256
        if self.failure_type is not None:
            value["failure_type"] = self.failure_type
        return value


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    """A read-only projection of one verified immutable artifact."""

    artifact_id: str
    sha256: str
    media_type: str
    schema_name: str | None
    schema_version: str | None
    size_bytes: int
    stored_at: str
    producer_occurrences: tuple[Mapping[str, Any], ...] = ()
    consumer_occurrences: tuple[Mapping[str, Any], ...] = ()
    derived_from: tuple[str, ...] = ()
    supported_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_artifact_id(self.artifact_id)
        validate_digest_reference(self.sha256, field="artifact inspection sha256")
        if self.artifact_id != f"sha256:{self.sha256}":
            raise InspectionIntegrityError("artifact inspection identity does not match its SHA-256")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise InspectionError("artifact inspection media_type is required")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise InspectionError("artifact inspection size is invalid")
        for field_name, values in (
            ("derived_from", self.derived_from),
            ("supported_by", self.supported_by),
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(validate_artifact_id(value, field=f"{field_name} artifact") for value in values),
            )
        object.__setattr__(
            self,
            "producer_occurrences",
            tuple(freeze_json_mapping(item, field="producer occurrence") for item in self.producer_occurrences),
        )
        object.__setattr__(
            self,
            "consumer_occurrences",
            tuple(freeze_json_mapping(item, field="consumer occurrence") for item in self.consumer_occurrences),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "size_bytes": self.size_bytes,
            "stored_at": self.stored_at,
            "producer_occurrences": [thaw_json(item) for item in self.producer_occurrences],
            "consumer_occurrences": [thaw_json(item) for item in self.consumer_occurrences],
            "derived_from": list(self.derived_from),
            "supported_by": list(self.supported_by),
        }


@dataclass(frozen=True, slots=True)
class RunInspection:
    """Stable canonical projection derived only from Core authoritative data."""

    run_id: str
    request_digest: str
    goal: str
    initial_artifact_ids: tuple[str, ...]
    policy_digest: str
    status: str
    decision_count: int
    tool_call_count: int
    step_count: int
    materialized_calls: tuple[ToolCallInspection, ...]
    pending_call: Mapping[str, Any] | None
    review_request_state: str | None
    failure_summary: tuple[Mapping[str, Any], ...]
    lineage_relations: tuple[Mapping[str, Any], ...]
    final_artifact_ids: tuple[str, ...]
    referenced_artifact_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    ledger_sha256: str
    lineage_sha256: str
    runtime_profile_ref: str | None = None
    runtime_profile_digest: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field="inspection run_id")
        validate_digest_reference(self.request_digest, field="inspection request_digest")
        validate_digest_reference(self.policy_digest, field="inspection policy_digest")
        validate_identifier(self.status, field="inspection status")
        for field_name in (
            "initial_artifact_ids",
            "final_artifact_ids",
            "referenced_artifact_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(validate_artifact_id(value, field=f"inspection {field_name}") for value in getattr(self, field_name)),
            )
        for field_name in ("decision_count", "tool_call_count", "step_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise InspectionError(f"inspection {field_name} is invalid")
        object.__setattr__(
            self,
            "materialized_calls",
            tuple(item if isinstance(item, ToolCallInspection) else ToolCallInspection(**item) for item in self.materialized_calls),
        )
        if self.pending_call is not None:
            object.__setattr__(self, "pending_call", freeze_json_mapping(self.pending_call, field="pending inspection call"))
        if self.review_request_state is not None:
            validate_identifier(self.review_request_state, field="review_request_state")
        object.__setattr__(
            self,
            "failure_summary",
            tuple(freeze_json_mapping(item, field="failure summary") for item in self.failure_summary),
        )
        object.__setattr__(
            self,
            "lineage_relations",
            tuple(freeze_json_mapping(item, field="inspection lineage relation") for item in self.lineage_relations),
        )
        object.__setattr__(
            self,
            "event_ids",
            tuple(validate_identifier(value, field="inspection event_id") for value in self.event_ids),
        )
        validate_digest_reference(self.ledger_sha256, field="inspection ledger_sha256")
        validate_digest_reference(self.lineage_sha256, field="inspection lineage_sha256")
        if self.runtime_profile_ref is not None:
            validate_identifier(self.runtime_profile_ref, field="runtime_profile_ref")
        if self.runtime_profile_digest is not None:
            validate_digest_reference(self.runtime_profile_digest, field="runtime_profile_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "molly.core.run-inspection",
            "schema_version": "1",
            "run_id": self.run_id,
            "request_digest": self.request_digest,
            "goal": self.goal,
            "initial_artifact_ids": list(self.initial_artifact_ids),
            "policy_digest": self.policy_digest,
            "runtime_profile_ref": self.runtime_profile_ref,
            "runtime_profile_digest": self.runtime_profile_digest,
            "status": self.status,
            "decision_count": self.decision_count,
            "tool_call_count": self.tool_call_count,
            "step_count": self.step_count,
            "materialized_calls": [item.to_dict() for item in self.materialized_calls],
            "pending_call": thaw_json(self.pending_call),
            "review_request_state": self.review_request_state,
            "failure_summary": [thaw_json(item) for item in self.failure_summary],
            "lineage_relations": [thaw_json(item) for item in self.lineage_relations],
            "final_artifact_ids": list(self.final_artifact_ids),
            "referenced_artifact_ids": list(self.referenced_artifact_ids),
            "event_ids": list(self.event_ids),
            "ledger_sha256": self.ledger_sha256,
            "lineage_sha256": self.lineage_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return sha256_bytes(self.canonical_bytes())


class RunInspector:
    """Validate and project Core facts without changing any authoritative store."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        ledger: RunLedger,
        lineage: ArtifactLineage,
    ) -> None:
        if not isinstance(store, ArtifactStore):
            raise InspectionError("RunInspector requires an ArtifactStore")
        if not isinstance(ledger, RunLedger):
            raise InspectionError("RunInspector requires a RunLedger")
        if not isinstance(lineage, ArtifactLineage):
            raise InspectionError("RunInspector requires an ArtifactLineage")
        self.store = store
        self.ledger = ledger
        self.lineage = lineage

    def _all_events(self) -> tuple[LedgerEvent, ...]:
        try:
            return self.ledger.events
        except Exception as exc:
            raise InspectionIntegrityError("RunLedger integrity verification failed") from exc

    def _all_relations(self) -> tuple[LineageRelation, ...]:
        try:
            return self.lineage.relations
        except Exception as exc:
            raise InspectionIntegrityError("ArtifactLineage integrity verification failed") from exc

    def _verify_artifact(self, artifact_id: str) -> None:
        try:
            self.store.verify(artifact_id)
        except Exception as exc:
            raise InspectionIntegrityError(
                "inspection references an unavailable or corrupt artifact"
            ) from exc

    def _verify_all_artifact_references(
        self,
        events: tuple[LedgerEvent, ...],
        relations: tuple[LineageRelation, ...],
    ) -> None:
        seen: set[str] = set()
        for event in events:
            for artifact_id in _artifact_ids_from_event(event):
                seen.add(artifact_id)
        for relation in relations:
            for identity in (relation.subject_id, relation.object_id):
                if identity.startswith("sha256:"):
                    seen.add(identity)
        for artifact_id in sorted(seen):
            self._verify_artifact(artifact_id)

    @staticmethod
    def _request_from_start(start: LedgerEvent) -> RunRequest:
        raw_request = start.metadata.get("request")
        if not isinstance(raw_request, Mapping):
            raise InspectionIntegrityError("RUN_STARTED is missing its immutable request")
        try:
            request = RunRequest.from_dict(raw_request)
            digest = validate_digest_reference(
                str(start.metadata.get("request_digest", "")), field="request_digest"
            )
        except Exception as exc:
            raise InspectionIntegrityError("RUN_STARTED request binding is malformed") from exc
        if request.request_sha256 != digest or request.run_id != start.run_id:
            raise InspectionIntegrityError("RUN_STARTED request binding is inconsistent")
        recorded_inputs = tuple(start.metadata.get("initial_artifact_ids", ()))
        if recorded_inputs != request.input_artifact_ids:
            raise InspectionIntegrityError("RUN_STARTED input binding is inconsistent")
        recorded_policy = start.metadata.get("policy_digest")
        if recorded_policy != request.tool_policy_digest:
            raise InspectionIntegrityError("RUN_STARTED policy binding is inconsistent")
        return request

    @staticmethod
    def _call_id(event: LedgerEvent) -> str | None:
        value = event.metadata.get("call_id")
        if value is None:
            return None
        if not isinstance(value, str):
            raise InspectionIntegrityError("call event has a malformed call_id")
        try:
            return validate_identifier(value, field="call_id")
        except Exception as exc:
            raise InspectionIntegrityError("call event has an invalid call_id") from exc

    @staticmethod
    def _validate_result_data(event: LedgerEvent) -> tuple[Any, str]:
        metadata = event.metadata
        if "result_data" not in metadata or "result_data_sha256" not in metadata:
            raise InspectionIntegrityError("successful execution is missing durable result data")
        value = metadata["result_data"]
        try:
            encoded = canonical_json_bytes(value)
            digest = sha256_bytes(encoded)
            recorded = validate_digest_reference(
                str(metadata["result_data_sha256"]), field="result_data_sha256"
            )
        except Exception as exc:
            raise InspectionIntegrityError("successful execution result data is malformed") from exc
        if len(encoded) > MAX_TOOL_RESULT_DATA_BYTES or digest != recorded:
            raise InspectionIntegrityError("successful execution result data integrity failed")
        return value, digest

    def _project_calls(
        self,
        events: tuple[LedgerEvent, ...],
    ) -> tuple[tuple[ToolCallInspection, ...], Mapping[str, Any] | None, bool]:
        materialized: dict[str, MaterializedToolCall] = {}
        materialized_events: dict[str, LedgerEvent] = {}
        related: dict[str, list[LedgerEvent]] = {}
        for event in events:
            call_id = self._call_id(event)
            if event.event_type == TOOL_CALL_MATERIALIZED:
                raw = event.metadata.get("materialized_call")
                if not isinstance(raw, Mapping):
                    raise InspectionIntegrityError("materialized call event is missing its call")
                try:
                    call = MaterializedToolCall.from_dict(raw)
                except Exception as exc:
                    raise InspectionIntegrityError("materialized call is malformed") from exc
                expected_call_id = call_id
                if expected_call_id != call.call_id or event.step_id != call.step_id:
                    raise InspectionIntegrityError("materialized call identity is inconsistent")
                if event.tool_name != call.tool_name or event.tool_version != call.tool_version:
                    raise InspectionIntegrityError("materialized call tool identity is inconsistent")
                if tuple(event.input_artifact_ids) != call.input_artifact_ids:
                    raise InspectionIntegrityError("materialized call input identity is inconsistent")
                if event.metadata.get("tool_call_digest") != call.digest:
                    raise InspectionIntegrityError("materialized call digest is inconsistent")
                if call.call_id in materialized:
                    raise InspectionIntegrityError("run contains duplicate materialized call IDs")
                materialized[call.call_id] = call
                materialized_events[call.call_id] = event
                related[call.call_id] = []
                continue
            if call_id is not None:
                if event.event_type not in _CALL_EVENT_TYPES:
                    raise InspectionIntegrityError("non-call event contains a call_id")
                if call_id not in materialized:
                    raise InspectionIntegrityError("call event precedes or omits materialization")
                related[call_id].append(event)

        projections: list[ToolCallInspection] = []
        pending_approval: Mapping[str, Any] | None = None
        unresolved = 0
        for call_id, call in materialized.items():
            call_events = related[call_id]
            for event in call_events:
                if event.step_id != call.step_id:
                    raise InspectionIntegrityError("call event step identity is inconsistent")
                if event.tool_name != call.tool_name or event.tool_version != call.tool_version:
                    raise InspectionIntegrityError("call event tool identity is inconsistent")
                if tuple(event.input_artifact_ids) != call.input_artifact_ids:
                    raise InspectionIntegrityError("call event input identity is inconsistent")
                recorded_digest = event.metadata.get("tool_call_digest")
                if recorded_digest is not None and recorded_digest != call.digest:
                    raise InspectionIntegrityError("call event digest is inconsistent")

            approval_events = [event for event in call_events if event.event_type == APPROVAL_RECORDED]
            if len(approval_events) > 1:
                raise InspectionIntegrityError("call has multiple approval records")
            approval: ApprovalRecord | None = None
            if approval_events:
                raw_approval = approval_events[0].metadata.get("approval")
                if not isinstance(raw_approval, Mapping):
                    raise InspectionIntegrityError("approval event is missing its record")
                try:
                    approval = ApprovalRecord.from_dict(raw_approval)
                    approval.assert_binds_to(call)
                except Exception as exc:
                    raise InspectionIntegrityError("approval is not bound to its call") from exc

            required = any(event.event_type == APPROVAL_REQUIRED for event in call_events)
            rejected = [event for event in call_events if event.event_type == TOOL_CALL_REJECTED]
            started = [event for event in call_events if event.event_type == TOOL_EXECUTION_STARTED]
            terminal = [
                event
                for event in call_events
                if event.event_type in {TOOL_EXECUTION_SUCCEEDED, TOOL_EXECUTION_FAILED}
            ]
            if len(rejected) > 1 or len(started) > 1 or len(terminal) > 1:
                raise InspectionIntegrityError("call has duplicate lifecycle events")
            if rejected and approval is None:
                raise InspectionIntegrityError("rejected call has no approval record")
            if rejected and approval is not None and approval.decision != "REJECTED":
                raise InspectionIntegrityError("rejected call approval decision is inconsistent")
            if approval is not None and approval.decision == "REJECTED":
                execution_status = "REJECTED"
            elif started and not terminal:
                execution_status = "INTERRUPTED"
            elif terminal and terminal[0].event_type == TOOL_EXECUTION_SUCCEEDED:
                execution_status = "SUCCEEDED"
            elif terminal:
                execution_status = "FAILED"
            else:
                execution_status = "PENDING"

            result_data: Any = None
            result_digest: str | None = None
            failure_type: str | None = None
            if terminal and terminal[0].event_type == TOOL_EXECUTION_SUCCEEDED:
                result_data, result_digest = self._validate_result_data(terminal[0])
            if terminal and terminal[0].event_type == TOOL_EXECUTION_FAILED:
                raw_error = terminal[0].metadata.get("error_type")
                if raw_error is not None:
                    if not isinstance(raw_error, str):
                        raise InspectionIntegrityError("failure event error_type is malformed")
                    failure_type = validate_identifier(raw_error, field="failure_type")

            approval_status = None
            if required:
                approval_status = approval.decision if approval is not None else "REQUIRED"
            elif approval is not None:
                approval_status = approval.decision

            if execution_status == "PENDING":
                unresolved += 1
                if required and approval is None:
                    pending_approval = call.to_dict()
            projections.append(
                ToolCallInspection(
                    call_id=call.call_id,
                    step_id=call.step_id,
                    tool_name=call.tool_name,
                    tool_version=call.tool_version,
                    tool_spec_digest=call.tool_spec_digest,
                    tool_call_digest=call.digest,
                    arguments=call.arguments,
                    input_artifact_ids=call.input_artifact_ids,
                    output_artifact_ids=(
                        terminal[0].output_artifact_ids
                        if terminal and terminal[0].event_type == TOOL_EXECUTION_SUCCEEDED
                        else ()
                    ),
                    approval_required=required,
                    approval_status=approval_status,
                    execution_status=execution_status,
                    reason_summary=call.reason_summary,
                    result_data=result_data,
                    result_data_sha256=result_digest,
                    failure_type=failure_type,
                    event_ids=(
                        materialized_events[call_id].event_id,
                        *(event.event_id for event in call_events),
                    ),
                )
            )
        if unresolved > 1:
            raise InspectionIntegrityError("run contains multiple unresolved tool calls")
        return tuple(projections), pending_approval, unresolved > 0

    @staticmethod
    def _status(
        events: tuple[LedgerEvent, ...],
        *,
        pending_approval: Mapping[str, Any] | None,
        unresolved: bool,
    ) -> str:
        terminal = [event for event in events if event.event_type in _TERMINAL_RUN_EVENTS]
        if terminal:
            if events[-1].event_id != terminal[-1].event_id:
                raise InspectionIntegrityError("events follow a terminal run event")
            return {
                RUN_STOPPED: RunStatus.STOPPED.value,
                RUN_REJECTED: RunStatus.REJECTED.value,
                RUN_FAILED: RunStatus.FAILED.value,
                BUDGET_EXHAUSTED: RunStatus.BUDGET_EXHAUSTED.value,
            }[terminal[-1].event_type]
        if any(
            event.event_type == TOOL_EXECUTION_STARTED
            and not any(
                later.event_type in {TOOL_EXECUTION_SUCCEEDED, TOOL_EXECUTION_FAILED}
                and later.metadata.get("call_id") == event.metadata.get("call_id")
                for later in events
            )
            for event in events
        ):
            return RunStatus.INTERRUPTED.value
        if pending_approval is not None:
            return RunStatus.WAITING_APPROVAL.value
        if events and events[-1].event_type == REVIEW_REQUESTED:
            return RunStatus.WAITING_REVIEW.value
        if unresolved:
            return RunStatus.ACTIVE.value
        return RunStatus.ACTIVE.value

    def _lineage_for_run(
        self,
        run_id: str,
        events: tuple[LedgerEvent, ...],
        relations: tuple[LineageRelation, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        step_identities: set[str] = {
            event.step_id for event in events if event.step_id is not None
        }
        projected: list[Mapping[str, Any]] = []
        for relation in relations:
            relation_run_id = relation.metadata.get("run_id")
            if relation_run_id != run_id and not (
                relation_run_id is None
                and (
                    relation.subject_id in step_identities
                    or relation.object_id in step_identities
                )
            ):
                continue
            value = relation.to_dict()
            value["metadata"] = _safe_metadata(relation.metadata)
            projected.append(value)
        return tuple(projected)

    def inspect_run(self, run_id: str) -> RunInspection:
        """Return a canonical projection; this method never appends or repairs."""

        try:
            validate_identifier(run_id, field="run_id")
        except Exception as exc:
            raise InspectionError("run_id is invalid") from exc
        all_events = self._all_events()
        events = tuple(event for event in all_events if event.run_id == run_id)
        if not events:
            raise InspectionError("run was not found")
        starts = [event for event in events if event.event_type == RUN_STARTED]
        if len(starts) != 1 or events[0].event_type != RUN_STARTED:
            raise InspectionIntegrityError("run must start with exactly one RUN_STARTED event")
        request = self._request_from_start(starts[0])
        relations = self._all_relations()
        for artifact_id in request.input_artifact_ids:
            self._verify_artifact(artifact_id)
        self._verify_all_artifact_references(all_events, relations)
        calls, pending_call, unresolved = self._project_calls(events)
        status = self._status(events, pending_approval=pending_call, unresolved=unresolved)
        final_ids: list[str] = list(request.input_artifact_ids)
        for call in calls:
            for artifact_id in call.output_artifact_ids:
                if artifact_id not in final_ids:
                    final_ids.append(artifact_id)
        referenced_ids: list[str] = []
        for event in events:
            for artifact_id in _artifact_ids_from_event(event):
                if artifact_id not in referenced_ids:
                    referenced_ids.append(artifact_id)
        for artifact_id in final_ids:
            if artifact_id not in referenced_ids:
                referenced_ids.append(artifact_id)
        failures: list[Mapping[str, Any]] = []
        for event in events:
            if event.event_type != TOOL_EXECUTION_FAILED:
                continue
            value: dict[str, Any] = {
                "event_id": event.event_id,
                "call_id": event.metadata.get("call_id"),
                "tool_name": event.tool_name,
                "error_type": event.metadata.get("error_type"),
            }
            failures.append(value)
        review_state = None
        if any(event.event_type == REVIEW_REQUESTED for event in events):
            review_state = "WAITING_REVIEW" if events[-1].event_type == REVIEW_REQUESTED else "REQUESTED"
        runtime_profile_ref = request.metadata.get("runtime_profile_ref")
        runtime_profile_digest = request.metadata.get("runtime_profile_digest")
        if runtime_profile_ref is not None and not isinstance(runtime_profile_ref, str):
            raise InspectionIntegrityError("runtime profile reference is malformed")
        if runtime_profile_digest is not None:
            try:
                runtime_profile_digest = validate_digest_reference(
                    str(runtime_profile_digest), field="runtime_profile_digest"
                )
            except Exception as exc:
                raise InspectionIntegrityError("runtime profile digest is malformed") from exc
        ledger_digest = _digest_json([event.to_dict() for event in all_events])
        lineage_digest = _digest_json([relation.to_dict() for relation in relations])
        return RunInspection(
            run_id=request.run_id,
            request_digest=request.request_sha256,
            goal=request.goal,
            initial_artifact_ids=request.input_artifact_ids,
            policy_digest=request.tool_policy_digest,
            runtime_profile_ref=runtime_profile_ref,
            runtime_profile_digest=runtime_profile_digest,
            status=status,
            decision_count=sum(event.event_type == DECISION_RECORDED for event in events),
            tool_call_count=sum(event.event_type == TOOL_CALL_MATERIALIZED for event in events),
            step_count=len({event.step_id for event in events if event.event_type == TOOL_CALL_MATERIALIZED and event.step_id}),
            materialized_calls=calls,
            pending_call=pending_call,
            review_request_state=review_state,
            failure_summary=tuple(failures),
            lineage_relations=self._lineage_for_run(run_id, events, relations),
            final_artifact_ids=tuple(final_ids),
            referenced_artifact_ids=tuple(referenced_ids),
            event_ids=tuple(event.event_id for event in events),
            ledger_sha256=ledger_digest,
            lineage_sha256=lineage_digest,
        )

    def inspect_artifact(self, artifact_id: str) -> ArtifactInspection:
        """Verify and inspect one artifact without changing any store."""

        try:
            record = self.store.verify(artifact_id)
        except Exception as exc:
            raise InspectionIntegrityError("artifact inspection could not verify the artifact") from exc
        relations = self._all_relations()
        self._verify_all_artifact_references(self._all_events(), relations)
        producer: list[Mapping[str, Any]] = []
        consumer: list[Mapping[str, Any]] = []
        derived: list[str] = []
        supported: list[str] = []
        for relation in relations:
            if relation.subject_id != record.artifact_id:
                continue
            value = relation.to_dict()
            value["metadata"] = _safe_metadata(relation.metadata)
            if relation.relation_type == RelationType.PRODUCED_BY.value:
                producer.append(value)
            elif relation.relation_type == RelationType.CONSUMED_BY.value:
                consumer.append(value)
            elif relation.relation_type == RelationType.DERIVED_FROM.value:
                derived.append(relation.object_id)
            elif relation.relation_type == RelationType.SUPPORTED_BY.value:
                supported.append(relation.object_id)
        return ArtifactInspection(
            artifact_id=record.artifact_id,
            sha256=record.sha256,
            media_type=record.media_type,
            schema_name=record.schema_name,
            schema_version=record.schema_version,
            size_bytes=record.size_bytes,
            stored_at=record.stored_at,
            producer_occurrences=tuple(producer),
            consumer_occurrences=tuple(consumer),
            derived_from=tuple(derived),
            supported_by=tuple(supported),
        )

    def events_for_run(self, run_id: str) -> tuple[LedgerEvent, ...]:
        """Return validated authoritative events for trace projection."""

        self.inspect_run(run_id)
        return tuple(event for event in self._all_events() if event.run_id == run_id)


__all__ = ["ArtifactInspection", "RunInspection", "RunInspector", "ToolCallInspection"]
