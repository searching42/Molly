"""The single bounded execution authority for Molly Core v2 CORE-02."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from .approvals import ApprovalDecision, ApprovalRecord
from .artifacts import ArtifactStore
from .errors import (
    ActionError,
    ApprovalError,
    CoreContractError,
    ReconciliationError,
    RunBindingError,
    RunStateError,
    SchemaValidationError,
    ToolContractError,
    ToolExecutionError,
    ToolPolicyError,
)
from .ids import (
    canonical_json_bytes,
    new_server_id,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_digest_reference,
    validate_identifier,
)
from .ledger import LedgerEvent, RunLedger
from .lineage import ArtifactLineage, RelationType
from .runs import (
    RunContext,
    RunRequest,
    RunResult,
    RunStatus,
    TERMINAL_RUN_STATUSES,
)
from .tools import (
    DecisionProvider,
    MAX_TOOL_RESULT_DATA_BYTES,
    MaterializedToolCall,
    RequestReviewAction,
    StopAction,
    StructuredAction,
    ToolCallProposal,
    ToolExecutionContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    action_from_value,
)


RUN_STARTED = "RUN_STARTED"
DECISION_RECORDED = "DECISION_RECORDED"
TOOL_CALL_MATERIALIZED = "TOOL_CALL_MATERIALIZED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
APPROVAL_RECORDED = "APPROVAL_RECORDED"
TOOL_CALL_REJECTED = "TOOL_CALL_REJECTED"
TOOL_EXECUTION_STARTED = "TOOL_EXECUTION_STARTED"
TOOL_EXECUTION_SUCCEEDED = "TOOL_EXECUTION_SUCCEEDED"
TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
REVIEW_REQUESTED = "REVIEW_REQUESTED"
RUN_STOPPED = "RUN_STOPPED"
RUN_FAILED = "RUN_FAILED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
INTENT_FROZEN = "INTENT_FROZEN"

EVENT_TYPES = frozenset(
    {
        RUN_STARTED,
        DECISION_RECORDED,
        TOOL_CALL_MATERIALIZED,
        APPROVAL_REQUIRED,
        APPROVAL_RECORDED,
        TOOL_CALL_REJECTED,
        TOOL_EXECUTION_STARTED,
        TOOL_EXECUTION_SUCCEEDED,
        TOOL_EXECUTION_FAILED,
        REVIEW_REQUESTED,
        RUN_STOPPED,
        RUN_FAILED,
        BUDGET_EXHAUSTED,
        INTENT_FROZEN,
    }
)


@dataclass(frozen=True, slots=True)
class _CallState:
    call: MaterializedToolCall
    approval_required: bool
    approval: ApprovalRecord | None
    approval_event: LedgerEvent | None
    rejected_event: LedgerEvent | None
    started_event: LedgerEvent | None
    terminal_event: LedgerEvent | None


@dataclass(frozen=True, slots=True)
class _RunProjection:
    events: tuple[LedgerEvent, ...]
    call_states: tuple[_CallState, ...]
    pending_approval: _CallState | None
    pending_execution: _CallState | None
    interrupted: bool
    waiting_review: bool
    terminal_event: LedgerEvent | None


@dataclass(frozen=True, slots=True)
class _ServerRunHardLimits:
    """Non-configurable server safety limits for one AgentLoop run."""

    max_decisions: int = 12
    max_tool_calls: int = 8
    max_steps: int = 8


SERVER_RUN_HARD_LIMITS = _ServerRunHardLimits()


class AgentLoop:
    """One validate-policy-materialize-execute-publish execution authority.

    The loop deliberately has no planner, controller, recovery agent, network
    adapter, or model implementation.  Its only external decision source is a
    ``DecisionProvider``; all authority and persistence remain host-owned.
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        ledger: RunLedger,
        lineage: ArtifactLineage,
        registry: ToolRegistry,
        policy: ToolPolicy,
        decision_provider: DecisionProvider,
        clock: Callable[[], str] = utc_timestamp,
    ) -> None:
        if not isinstance(store, ArtifactStore):
            raise CoreContractError("AgentLoop requires an ArtifactStore")
        if not isinstance(ledger, RunLedger):
            raise CoreContractError("AgentLoop requires a RunLedger")
        if not isinstance(lineage, ArtifactLineage):
            raise CoreContractError("AgentLoop requires an ArtifactLineage")
        if not isinstance(registry, ToolRegistry):
            raise CoreContractError("AgentLoop requires a ToolRegistry")
        if not isinstance(policy, ToolPolicy):
            raise CoreContractError("AgentLoop requires a ToolPolicy")
        if not hasattr(decision_provider, "next_action") and not callable(decision_provider):
            raise CoreContractError("decision_provider must expose next_action")
        if not callable(clock):
            raise CoreContractError("clock must be callable")
        self.store = store
        self.ledger = ledger
        self.lineage = lineage
        self.registry = registry
        self.policy = policy
        self.decision_provider = decision_provider
        self._clock = clock

    def _now(self) -> str:
        return normalize_timestamp(self._clock(), field="event timestamp")

    def _new_unique_id(self, prefix: str, *, field: str) -> str:
        used: set[str] = set()
        for event in self.ledger.events:
            if field == "step_id" and event.step_id:
                used.add(event.step_id)
            elif field == "call_id":
                call_id = event.metadata.get("call_id")
                if isinstance(call_id, str):
                    used.add(call_id)
            elif field == "event_id":
                used.add(event.event_id)
        while True:
            candidate = new_server_id(prefix)
            if candidate not in used:
                return candidate

    @staticmethod
    def _metadata_call_id(event: LedgerEvent) -> str | None:
        value = event.metadata.get("call_id")
        return value if isinstance(value, str) else None

    @staticmethod
    def _metadata_digest(event: LedgerEvent, name: str) -> str | None:
        value = event.metadata.get(name)
        return value if isinstance(value, str) else None

    def _append(
        self,
        request: RunRequest,
        event_type: str,
        *,
        step_id: str | None = None,
        status: str | None = None,
        tool_name: str | None = None,
        tool_version: str | None = None,
        input_artifact_ids: tuple[str, ...] = (),
        output_artifact_ids: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> LedgerEvent:
        if event_type not in EVENT_TYPES:
            raise RunStateError(f"unknown AgentLoop event type: {event_type}")
        return self.ledger.append(
            event_id=self._new_unique_id("evt", field="event_id"),
            run_id=request.run_id,
            event_type=event_type,
            step_id=step_id,
            status=status,
            tool_name=tool_name,
            tool_version=tool_version,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=output_artifact_ids,
            timestamp=self._now(),
            metadata={} if metadata is None else metadata,
        )

    def _ensure_request_policy(self, request: RunRequest) -> None:
        expected = validate_digest_reference(self.policy.policy_digest, field="policy_digest")
        if request.tool_policy_digest != expected:
            raise RunBindingError("RunRequest policy digest does not match the active ToolPolicy")

    def _start_metadata(self, request: RunRequest) -> dict[str, Any]:
        return {
            "request": request.to_dict(),
            "request_digest": request.request_sha256,
            "policy_digest": request.tool_policy_digest,
            "initial_artifact_ids": list(request.input_artifact_ids),
        }

    def _initialize_or_validate(self, request: RunRequest) -> tuple[LedgerEvent, ...]:
        if not isinstance(request, RunRequest):
            raise RunBindingError("AgentLoop requires a RunRequest")
        self._ensure_request_policy(request)
        existing = self.ledger.for_run(request.run_id)
        if not existing:
            for artifact_id in request.input_artifact_ids:
                self.store.verify(artifact_id)
            self._append(
                request,
                RUN_STARTED,
                status="STARTED",
                metadata=self._start_metadata(request),
            )
            return self.ledger.for_run(request.run_id)

        starts = [event for event in existing if event.event_type == RUN_STARTED]
        if len(starts) != 1:
            raise RunStateError("run ledger must contain exactly one RUN_STARTED event")
        start = starts[0]
        recorded_digest = start.metadata.get("request_digest")
        if recorded_digest != request.request_sha256:
            raise RunBindingError("same run_id was supplied with a different request digest")
        recorded_policy = start.metadata.get("policy_digest")
        if recorded_policy != request.tool_policy_digest:
            raise RunBindingError("run policy binding changed during resume")
        recorded_request = start.metadata.get("request")
        if not isinstance(recorded_request, Mapping):
            raise RunStateError("RUN_STARTED does not contain the immutable request binding")
        try:
            reconstructed = RunRequest.from_dict(recorded_request)
        except Exception as exc:
            raise RunStateError("RUN_STARTED request binding is malformed") from exc
        if reconstructed.request_sha256 != request.request_sha256:
            raise RunBindingError("persisted request binding does not match the supplied request")
        for artifact_id in request.input_artifact_ids:
            self.store.verify(artifact_id)
        return existing

    @staticmethod
    def _call_id_for_event(event: LedgerEvent) -> str:
        call_id = AgentLoop._metadata_call_id(event)
        if call_id is None:
            raise RunStateError(f"{event.event_type} is missing call_id")
        validate_identifier(call_id, field="call_id")
        return call_id

    def _project(self, events: tuple[LedgerEvent, ...]) -> _RunProjection:
        materialized: list[tuple[LedgerEvent, MaterializedToolCall]] = []
        by_call: dict[str, list[LedgerEvent]] = {}
        for event in events:
            if event.event_type == TOOL_EXECUTION_SUCCEEDED:
                self._validated_success_result_data(event)
            if event.event_type == TOOL_CALL_MATERIALIZED:
                raw = event.metadata.get("materialized_call")
                if not isinstance(raw, Mapping):
                    raise RunStateError("materialization event is missing its exact call")
                try:
                    call = MaterializedToolCall.from_dict(raw)
                except Exception as exc:
                    raise RunStateError("materialization event contains an invalid call") from exc
                call_id = self._call_id_for_event(event)
                if call.call_id != call_id or event.step_id != call.step_id:
                    raise RunStateError("materialization event identity disagrees with its call")
                if event.tool_name != call.tool_name or event.tool_version != call.tool_version:
                    raise RunStateError("materialization event tool identity disagrees with its call")
                if tuple(event.input_artifact_ids) != call.input_artifact_ids:
                    raise RunStateError("materialization event input identity disagrees with its call")
                materialized.append((event, call))
            if self._metadata_call_id(event) is not None:
                by_call.setdefault(self._call_id_for_event(event), []).append(event)

        call_states: list[_CallState] = []
        for materialized_event, call in materialized:
            related = by_call.get(call.call_id, [])
            for related_event in related:
                if related_event.event_type == TOOL_CALL_MATERIALIZED:
                    continue
                recorded_digest = related_event.metadata.get("tool_call_digest")
                if recorded_digest is not None and recorded_digest != call.digest:
                    raise RunStateError("call event digest disagrees with its materialized call")
                if related_event.step_id != call.step_id:
                    raise RunStateError("call event step identity disagrees with its materialized call")
            approval_events = [item for item in related if item.event_type == APPROVAL_RECORDED]
            if len(approval_events) > 1:
                raise RunStateError("a materialized call has multiple approval records")
            approval_event = approval_events[0] if approval_events else None
            approval: ApprovalRecord | None = None
            if approval_event is not None:
                raw_approval = approval_event.metadata.get("approval")
                if not isinstance(raw_approval, Mapping):
                    raise RunStateError("approval event is missing its immutable record")
                try:
                    approval = ApprovalRecord.from_dict(raw_approval)
                    approval.assert_binds_to(call)
                except Exception as exc:
                    raise RunStateError("approval event is not bound to its materialized call") from exc
            rejected_events = [item for item in related if item.event_type == TOOL_CALL_REJECTED]
            if len(rejected_events) > 1:
                raise RunStateError("a materialized call has multiple rejection events")
            started_events = [item for item in related if item.event_type == TOOL_EXECUTION_STARTED]
            if len(started_events) > 1:
                raise RunStateError("a materialized call has multiple execution starts")
            terminal_events = [
                item
                for item in related
                if item.event_type in {TOOL_EXECUTION_SUCCEEDED, TOOL_EXECUTION_FAILED}
            ]
            if len(terminal_events) > 1:
                raise RunStateError("a materialized call has multiple execution terminals")
            approval_required = any(item.event_type == APPROVAL_REQUIRED for item in related)
            if approval_required and approval is None and rejected_events:
                raise RunStateError("rejected call has no approval record")
            if approval is not None and approval.decision == ApprovalDecision.REJECTED.value:
                terminal = rejected_events[0] if rejected_events else None
            else:
                terminal = terminal_events[0] if terminal_events else None
            call_states.append(
                _CallState(
                    call=call,
                    approval_required=approval_required,
                    approval=approval,
                    approval_event=approval_event,
                    rejected_event=rejected_events[0] if rejected_events else None,
                    started_event=started_events[0] if started_events else None,
                    terminal_event=terminal,
                )
            )

        terminal_run_events = [
            event
            for event in events
            if event.event_type in {RUN_STOPPED, RUN_FAILED, BUDGET_EXHAUSTED}
        ]
        terminal_run = terminal_run_events[-1] if terminal_run_events else None
        if terminal_run is not None and events[-1].event_id != terminal_run.event_id:
            raise RunStateError("events were appended after a terminal run state")

        interrupted = any(
            state.started_event is not None and state.terminal_event is None
            for state in call_states
        )
        pending_approval_states = [
            state
            for state in call_states
            if state.terminal_event is None
            and state.started_event is None
            and state.approval_required
            and state.approval is None
        ]
        pending_execution_states = [
            state
            for state in call_states
            if state.terminal_event is None
            and state.started_event is None
            and (
                not state.approval_required
                or (
                    state.approval is not None
                    and state.approval.decision == ApprovalDecision.APPROVED.value
                )
            )
        ]
        unresolved = pending_approval_states + pending_execution_states
        if len(unresolved) > 1:
            raise RunStateError("run contains multiple unresolved materialized calls")
        review_events = [event for event in events if event.event_type == REVIEW_REQUESTED]
        waiting_review = bool(review_events) and (
            not materialized or events[-1].event_id == review_events[-1].event_id
        )
        return _RunProjection(
            events=events,
            call_states=tuple(call_states),
            pending_approval=pending_approval_states[0] if pending_approval_states else None,
            pending_execution=pending_execution_states[0] if pending_execution_states else None,
            interrupted=interrupted,
            waiting_review=waiting_review,
            terminal_event=terminal_run,
        )

    @staticmethod
    def _counts(events: tuple[LedgerEvent, ...]) -> tuple[int, int, int]:
        decisions = sum(event.event_type == DECISION_RECORDED for event in events)
        tool_calls = sum(event.event_type == TOOL_CALL_MATERIALIZED for event in events)
        steps = len(
            {
                event.step_id
                for event in events
                if event.event_type == TOOL_CALL_MATERIALIZED and event.step_id is not None
            }
        )
        return decisions, tool_calls, steps

    @staticmethod
    def _effective_hard_limits(request: RunRequest) -> _ServerRunHardLimits:
        """Apply the strictest limit when resuming a legacy request.

        The removed request-level budget is retained only as a compatibility
        binding. It must never expand the current server safety ceiling, but a
        persisted lower value remains part of the authority of that old run.
        """

        legacy = getattr(request, "_legacy_budget", None)
        if legacy is None:
            return SERVER_RUN_HARD_LIMITS
        return _ServerRunHardLimits(
            max_decisions=min(SERVER_RUN_HARD_LIMITS.max_decisions, legacy["max_decisions"]),
            max_tool_calls=min(SERVER_RUN_HARD_LIMITS.max_tool_calls, legacy["max_tool_calls"]),
            max_steps=min(SERVER_RUN_HARD_LIMITS.max_steps, legacy["max_steps"]),
        )

    def _hard_limit_reason(
        self,
        request: RunRequest,
        events: tuple[LedgerEvent, ...],
        *,
        pending_execution: bool = False,
    ) -> tuple[str, tuple[int, int, int]] | None:
        """Return a server-limit reason without consulting new request input.

        A materialized call already admitted before the cap is allowed to
        finish, including after approval. The strict pre-provider check
        prevents another model decision or tool materialization; the relaxed
        pending-execution check only rejects legacy/corrupt states that are
        already over the effective server ceiling.
        """

        counts = self._counts(events)
        decisions, tool_calls, steps = counts
        limits = self._effective_hard_limits(request)
        if pending_execution:
            if decisions > limits.max_decisions:
                return "server decision safety limit exceeded", counts
            if tool_calls > limits.max_tool_calls:
                return "server tool-call safety limit exceeded", counts
            if steps > limits.max_steps:
                return "server step safety limit exceeded", counts
            return None
        if decisions >= limits.max_decisions:
            return "server decision safety limit reached", counts
        if tool_calls >= limits.max_tool_calls:
            return "server tool-call safety limit reached", counts
        if steps >= limits.max_steps:
            return "server step safety limit reached", counts
        return None

    def _append_hard_limit(
        self,
        request: RunRequest,
        reason: str,
        counts: tuple[int, int, int],
    ) -> None:
        events = self.ledger.for_run(request.run_id)
        if any(event.event_type == BUDGET_EXHAUSTED for event in events):
            return
        limits = self._effective_hard_limits(request)
        self._append(
            request,
            BUDGET_EXHAUSTED,
            status="EXHAUSTED",
            metadata={
                "reason": reason,
                "counts": {
                    "decisions": counts[0],
                    "tool_calls": counts[1],
                    "steps": counts[2],
                },
                "server_limits": {
                    "max_decisions": limits.max_decisions,
                    "max_tool_calls": limits.max_tool_calls,
                    "max_steps": limits.max_steps,
                },
            },
        )

    def _visible_artifacts(
        self, request: RunRequest, events: tuple[LedgerEvent, ...]
    ) -> tuple[str, ...]:
        visible = list(request.input_artifact_ids)
        for event in events:
            if event.event_type != TOOL_EXECUTION_SUCCEEDED:
                continue
            for artifact_id in event.output_artifact_ids:
                if artifact_id not in visible:
                    visible.append(artifact_id)
        return tuple(visible)

    @staticmethod
    def _recent_events(events: tuple[LedgerEvent, ...]) -> tuple[Mapping[str, Any], ...]:
        summaries: list[Mapping[str, Any]] = []
        for event in events[-12:]:
            summaries.append(
                {
                    "event_type": event.event_type,
                    "status": event.status,
                    "step_id": event.step_id,
                    "tool_name": event.tool_name,
                    "timestamp": event.timestamp,
                }
            )
        return tuple(summaries)

    def _result(
        self,
        request: RunRequest,
        *,
        projection: _RunProjection | None = None,
        message: str = "",
    ) -> RunResult:
        events = self.ledger.for_run(request.run_id)
        projection = projection or self._project(events)
        if projection.terminal_event is not None:
            status = {
                RUN_STOPPED: RunStatus.STOPPED,
                RUN_FAILED: RunStatus.FAILED,
                BUDGET_EXHAUSTED: RunStatus.BUDGET_EXHAUSTED,
            }[projection.terminal_event.event_type]
        elif projection.interrupted:
            status = RunStatus.INTERRUPTED
        elif projection.pending_approval is not None:
            status = RunStatus.WAITING_APPROVAL
        elif projection.waiting_review:
            status = RunStatus.WAITING_REVIEW
        else:
            status = RunStatus.ACTIVE
        pending = None
        if projection.pending_approval is not None:
            pending = projection.pending_approval.call.to_dict()
        return RunResult(
            run_id=request.run_id,
            status=status,
            visible_artifact_ids=self._visible_artifacts(request, events),
            last_event_id=events[-1].event_id if events else None,
            pending_call=pending,
            message=message,
        )

    def status(self, run_id: str) -> RunResult:
        """Project a run without appending or repairing any authoritative data."""

        validate_identifier(run_id, field="run_id")
        events = self.ledger.for_run(run_id)
        if not events:
            return RunResult(
                run_id=run_id,
                status=RunStatus.NEW,
                visible_artifact_ids=(),
            )
        start = next((event for event in events if event.event_type == RUN_STARTED), None)
        if start is None or not isinstance(start.metadata.get("request"), Mapping):
            raise RunStateError("cannot project a run without its immutable request")
        request = RunRequest.from_dict(start.metadata["request"])
        return self._result(request)

    def _context(self, request: RunRequest, events: tuple[LedgerEvent, ...]) -> RunContext:
        previous: Mapping[str, Any] | None = None
        for event in reversed(events):
            if event.event_type == TOOL_EXECUTION_SUCCEEDED:
                result_data, result_data_sha256 = self._validated_success_result_data(event)
                previous = {
                    "event_type": event.event_type,
                    "status": event.status,
                    "tool_name": event.tool_name,
                    "tool_version": event.tool_version,
                    "output_artifact_ids": list(event.output_artifact_ids),
                    "data": thaw_json(result_data),
                    "data_sha256": result_data_sha256,
                }
                break
            if event.event_type == TOOL_EXECUTION_FAILED:
                previous = {
                    "event_type": event.event_type,
                    "status": event.status,
                    "tool_name": event.tool_name,
                    "tool_version": event.tool_version,
                    "output_artifact_ids": list(event.output_artifact_ids),
                }
                break
        return RunContext(
            run_id=request.run_id,
            goal=request.goal,
            visible_artifact_ids=self._visible_artifacts(request, events),
            initial_artifact_ids=request.input_artifact_ids,
            request_metadata={
                key: request.metadata[key]
                for key in ("llm_profile_ref", "llm_profile_digest")
                if key in request.metadata
            },
            recent_events=self._recent_events(events),
            previous_tool_outcome=previous,
        )

    def _reconcile_success_lineage(
        self, request: RunRequest, events: tuple[LedgerEvent, ...]
    ) -> None:
        for event in events:
            if event.event_type == TOOL_EXECUTION_SUCCEEDED:
                self._project_success(event)

    @staticmethod
    def _execution_relation_id(
        success_event: LedgerEvent, relation_type: RelationType, subject_id: str, object_id: str
    ) -> str:
        body = {
            "success_event_id": success_event.event_id,
            "relation_type": relation_type.value,
            "subject_id": subject_id,
            "object_id": object_id,
        }
        return "rel_exec_" + sha256_bytes(canonical_json_bytes(body))

    def _project_success(self, success_event: LedgerEvent) -> None:
        if success_event.event_type != TOOL_EXECUTION_SUCCEEDED:
            raise ReconciliationError("only successful execution events can be projected")
        self._validated_success_result_data(success_event)
        if success_event.step_id is None:
            raise ReconciliationError("successful execution is missing step_id")
        call_id = self._metadata_call_id(success_event)
        if call_id is None:
            raise ReconciliationError("successful execution is missing call_id")
        metadata = {
            "run_id": success_event.run_id,
            "call_id": call_id,
            "success_event_id": success_event.event_id,
        }
        self.lineage.register_step(success_event.step_id)
        for artifact_id in (*success_event.input_artifact_ids, *success_event.output_artifact_ids):
            try:
                self.store.verify(artifact_id)
            except Exception as exc:
                raise ReconciliationError(
                    "successful execution references an unavailable or corrupt artifact"
                ) from exc
            self.lineage.register_artifact(artifact_id)

        def add(relation_type: RelationType, subject_id: str, object_id: str) -> None:
            relation_id = self._execution_relation_id(
                success_event, relation_type, subject_id, object_id
            )
            try:
                self.lineage.add_relation_idempotent(
                    relation_type,
                    subject_id,
                    object_id,
                    relation_id=relation_id,
                    created_at=success_event.timestamp,
                    metadata=metadata,
                )
            except Exception as exc:
                if isinstance(exc, ReconciliationError):
                    raise
                raise ReconciliationError("execution lineage projection failed") from exc

        for output_id in success_event.output_artifact_ids:
            add(RelationType.PRODUCED_BY, output_id, success_event.step_id)
            for input_id in success_event.input_artifact_ids:
                add(RelationType.DERIVED_FROM, output_id, input_id)
        for input_id in success_event.input_artifact_ids:
            add(RelationType.CONSUMED_BY, input_id, success_event.step_id)

    def _materialize(
        self,
        request: RunRequest,
        proposal: ToolCallProposal,
        spec: ToolSpec,
    ) -> MaterializedToolCall:
        step_id = self._new_unique_id("step", field="step_id")
        call_id = self._new_unique_id("call", field="call_id")
        return MaterializedToolCall(
            run_id=request.run_id,
            step_id=step_id,
            call_id=call_id,
            tool_name=spec.name,
            tool_version=spec.version,
            tool_spec_digest=spec.spec_digest,
            policy_digest=request.tool_policy_digest,
            arguments=proposal.arguments,
            input_artifact_ids=proposal.input_artifact_ids,
            created_at=self._now(),
            tool_call_digest=None,
        )

    def _validate_proposal_inputs(
        self, proposal: ToolCallProposal, visible_artifacts: tuple[str, ...]
    ) -> None:
        visible = set(visible_artifacts)
        for artifact_id in proposal.input_artifact_ids:
            if artifact_id not in visible:
                raise ToolContractError("tool proposal references an artifact not visible in this run")
            self.store.verify(artifact_id)

    def _append_materialized(
        self, request: RunRequest, call: MaterializedToolCall
    ) -> LedgerEvent:
        return self._append(
            request,
            TOOL_CALL_MATERIALIZED,
            step_id=call.step_id,
            status="MATERIALIZED",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_spec_digest": call.tool_spec_digest,
                "policy_digest": call.policy_digest,
                "tool_call_digest": call.tool_call_digest,
                "materialized_call": call.to_dict(),
            },
        )

    def _append_approval_required(
        self, request: RunRequest, call: MaterializedToolCall
    ) -> LedgerEvent:
        return self._append(
            request,
            APPROVAL_REQUIRED,
            step_id=call.step_id,
            status="WAITING_APPROVAL",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
            },
        )

    def _record_approval(
        self,
        request: RunRequest,
        call: MaterializedToolCall,
        approval: ApprovalRecord,
    ) -> LedgerEvent:
        if not isinstance(approval, ApprovalRecord):
            raise ApprovalError("resume approval must be an ApprovalRecord")
        approval.assert_binds_to(call)
        return self._append(
            request,
            APPROVAL_RECORDED,
            step_id=call.step_id,
            status=approval.decision,
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
                "approval": approval.to_dict(),
            },
        )

    def _append_rejection(
        self, request: RunRequest, call: MaterializedToolCall, approval: ApprovalRecord
    ) -> LedgerEvent:
        return self._append(
            request,
            TOOL_CALL_REJECTED,
            step_id=call.step_id,
            status="REJECTED",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
                "approval_id": approval.approval_id,
            },
        )

    @staticmethod
    def _safe_error_type(error: BaseException) -> str:
        candidate = type(error).__name__
        return candidate if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", candidate) else "ToolError"

    def _append_execution_failed(
        self, request: RunRequest, call: MaterializedToolCall, error: BaseException
    ) -> LedgerEvent:
        return self._append(
            request,
            TOOL_EXECUTION_FAILED,
            step_id=call.step_id,
            status="FAILED",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
                "error_type": self._safe_error_type(error),
            },
        )

    @staticmethod
    def _validated_success_result_data(event: LedgerEvent) -> tuple[Any, str]:
        """Validate the exact bounded observation stored by a success event."""

        if event.event_type != TOOL_EXECUTION_SUCCEEDED:
            raise RunStateError("only successful execution events contain result data")
        metadata = event.metadata
        if "result_data" not in metadata or "result_data_sha256" not in metadata:
            raise RunStateError("successful execution is missing durable result data")
        result_data = metadata["result_data"]
        recorded_digest = metadata["result_data_sha256"]
        try:
            encoded = canonical_json_bytes(result_data)
            computed_digest = sha256_bytes(encoded)
            normalized_recorded = validate_digest_reference(
                recorded_digest, field="result_data_sha256"
            )
        except Exception as exc:
            raise RunStateError("successful execution contains invalid result data") from exc
        if len(encoded) > MAX_TOOL_RESULT_DATA_BYTES:
            raise RunStateError("successful execution result data exceeds its bounded size")
        if normalized_recorded != computed_digest:
            raise RunStateError("successful execution result data digest mismatch")
        return result_data, computed_digest

    def _assert_current_call(self, request: RunRequest, call: MaterializedToolCall) -> ToolSpec:
        try:
            spec = self.registry.resolve_exact(
                call.tool_name, call.tool_version, call.tool_spec_digest
            )
            if call.policy_digest != request.tool_policy_digest:
                raise RunBindingError("materialized call policy digest does not match the run")
            self.policy.check(spec)
            return spec
        except Exception as exc:
            if isinstance(exc, (RunBindingError, ToolContractError)):
                raise
            raise ToolContractError("materialized call no longer resolves to the exact tool") from exc

    def _execute_call(self, request: RunRequest, call: MaterializedToolCall) -> LedgerEvent:
        spec = self._assert_current_call(request, call)
        executor = self.registry.executor_for(spec)

        self._append(
            request,
            TOOL_EXECUTION_STARTED,
            step_id=call.step_id,
            status="STARTED",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
            },
        )
        context = ToolExecutionContext(
            run_id=call.run_id,
            step_id=call.step_id,
            call_id=call.call_id,
            idempotency_key=call.idempotency_key,
            arguments=call.arguments,
            input_artifact_ids=call.input_artifact_ids,
            reader=self.store.read,
        )
        try:
            raw_result = executor(context)
            result = raw_result if isinstance(raw_result, ToolResult) else ToolResult(**raw_result)
            result_data_bytes = canonical_json_bytes(result.data)
            if len(result_data_bytes) > MAX_TOOL_RESULT_DATA_BYTES:
                raise ToolContractError(
                    "tool result data exceeds the bounded canonical size; "
                    "publish larger content through ArtifactDraft"
                )
            spec.validate_output(result.data)
            output_records = tuple(
                self.store.put(
                    draft.content,
                    media_type=draft.media_type,
                    schema_name=draft.schema_name,
                    schema_version=draft.schema_version,
                )
                for draft in result.artifacts
            )
        except Exception as exc:
            failed = self._append_execution_failed(request, call, exc)
            return failed

        success = self._append(
            request,
            TOOL_EXECUTION_SUCCEEDED,
            step_id=call.step_id,
            status="SUCCEEDED",
            tool_name=call.tool_name,
            tool_version=call.tool_version,
            input_artifact_ids=call.input_artifact_ids,
            output_artifact_ids=tuple(record.artifact_id for record in output_records),
            metadata={
                "call_id": call.call_id,
                "tool_call_digest": call.tool_call_digest,
                "result_data": thaw_json(result.data),
                "result_data_sha256": sha256_bytes(result_data_bytes),
            },
        )
        # The durable success fact is deliberately appended before this
        # projection.  A later restart can repair a missing projection.
        self._project_success(success)
        return success

    def _invoke_provider(self, context: RunContext) -> StructuredAction:
        try:
            if hasattr(self.decision_provider, "next_action"):
                raw_action = self.decision_provider.next_action(
                    context, self.registry.model_visible_tools()
                )
            else:
                raw_action = self.decision_provider(
                    context, self.registry.model_visible_tools()
                )
            return action_from_value(raw_action)
        except StopIteration:
            # Deterministic scripted providers may intentionally expose only
            # one action for a turn.  No authoritative state is changed.
            raise
        except Exception as exc:
            if isinstance(exc, ActionError):
                raise
            raise ActionError("DecisionProvider returned an invalid action") from exc

    def run(
        self, request: RunRequest, *, approval: ApprovalRecord | None = None
    ) -> RunResult:
        """Drive one run until it stops, waits, interrupts, or needs input."""

        events = self._initialize_or_validate(request)
        self._reconcile_success_lineage(request, events)
        approval_resume = False

        while True:
            events = self.ledger.for_run(request.run_id)
            projection = self._project(events)
            if projection.terminal_event is not None:
                if approval is not None:
                    raise ApprovalError("terminal runs cannot accept a new approval")
                return self._result(request, projection=projection)
            if projection.interrupted:
                if approval is not None:
                    raise ApprovalError("interrupted calls require reconciliation before approval")
                return self._result(request, projection=projection, message="call needs reconciliation")

            if projection.pending_approval is not None:
                pending = projection.pending_approval
                self._assert_current_call(request, pending.call)
                if approval is None:
                    return self._result(request, projection=projection)
                if pending.approval is not None:
                    if approval.digest != pending.approval.digest:
                        raise ApprovalError("a different approval was supplied for the pending call")
                    approval = None
                    continue
                self._record_approval(request, pending.call, approval)
                if approval.decision == ApprovalDecision.REJECTED.value:
                    self._append_rejection(request, pending.call, approval)
                    return self._result(request, message="approval rejected")
                # The exact approved call is executed below without asking
                # the DecisionProvider to recreate it.
                approval = None
                approval_resume = True
                continue

            if approval is not None:
                approved_pending = projection.pending_execution
                if (
                    approved_pending is not None
                    and approved_pending.approval is not None
                    and approved_pending.approval.decision == ApprovalDecision.APPROVED.value
                ):
                    if approval.digest != approved_pending.approval.digest:
                        raise ApprovalError("a different approval was supplied for the approved call")
                    approval = None
                else:
                    raise ApprovalError("no pending approval matches the supplied record")
            if projection.waiting_review:
                return self._result(request, projection=projection)

            if projection.pending_execution is not None:
                hard_limit = self._hard_limit_reason(
                    request, events, pending_execution=True
                )
                if hard_limit is not None:
                    reason, counts = hard_limit
                    self._append_hard_limit(request, reason, counts)
                    return self._result(request)
                approval_resume = approval_resume or (
                    projection.pending_execution.approval is not None
                    and projection.pending_execution.approval.decision
                    == ApprovalDecision.APPROVED.value
                )
                self._execute_call(request, projection.pending_execution.call)
                if approval_resume:
                    return self._result(request)
                continue

            hard_limit = self._hard_limit_reason(request, events)
            if hard_limit is not None:
                reason, counts = hard_limit
                self._append_hard_limit(request, reason, counts)
                return self._result(request)

            context = self._context(request, events)
            try:
                action = self._invoke_provider(context)
            except StopIteration:
                return self._result(request, message="DecisionProvider has no further action")
            self._append(
                request,
                DECISION_RECORDED,
                status="PROPOSED",
                metadata={"action": action.to_dict()},
            )

            if isinstance(action, StopAction):
                self._append(
                    request,
                    RUN_STOPPED,
                    status="STOPPED",
                    metadata={"reason": action.reason},
                )
                return self._result(request)
            if isinstance(action, RequestReviewAction):
                self._append(
                    request,
                    REVIEW_REQUESTED,
                    status="WAITING_REVIEW",
                    metadata={"action": action.to_dict()},
                )
                return self._result(request)
            if not isinstance(action, ToolCallProposal):
                raise ActionError("unknown structured action")

            try:
                spec = self.registry.resolve(action.tool_name)
                self._validate_proposal_inputs(action, self._visible_artifacts(request, events))
                spec.validate_arguments(action.arguments)
                self.policy.check(spec)
            except Exception as exc:
                if isinstance(exc, (ToolContractError, SchemaValidationError, ToolPolicyError)):
                    raise
                raise ToolContractError("tool proposal failed closed") from exc

            # The server, never the provider, creates the step/call identity
            # and all execution-bound fields.
            call = self._materialize(request, action, spec)
            self._append_materialized(request, call)
            if self.policy.requires_approval(spec):
                self._append_approval_required(request, call)
                return self._result(request)
            self._execute_call(request, call)


RunEngine = AgentLoop


__all__ = [
    "APPROVAL_RECORDED",
    "APPROVAL_REQUIRED",
    "AgentLoop",
    "BUDGET_EXHAUSTED",
    "DECISION_RECORDED",
    "EVENT_TYPES",
    "INTENT_FROZEN",
    "REVIEW_REQUESTED",
    "RUN_FAILED",
    "RUN_STARTED",
    "RUN_STOPPED",
    "SERVER_RUN_HARD_LIMITS",
    "RunEngine",
    "TOOL_CALL_MATERIALIZED",
    "TOOL_CALL_REJECTED",
    "TOOL_EXECUTION_FAILED",
    "TOOL_EXECUTION_STARTED",
    "TOOL_EXECUTION_SUCCEEDED",
]
