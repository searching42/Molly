"""Deterministic one-action Scientific Agent Harness Controller v1.

The Controller consumes an exact start intent, re-verifies every authority,
selects one server-owned action in RunPlan order, commits the decision before
the effect, and commits an immutable evidence receipt afterwards.  It never
accepts a client-selected task, route, adapter, profile, resource, or action.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.harness_tracing import HarnessTracer, NoopHarnessTracer
from ai4s_agent.resource_profiles import build_transfer_manifest
from ai4s_agent.schemas import (
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerDecision,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    AgentHarnessControllerInspectionFact,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessControllerSourceBinding,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerStatus,
    AgentHarnessControllerTaskSlot,
    AgentHarnessGateApprovalRequest,
    AgentHarnessRemoteApprovalRequest,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPlanAuthorization,
    RunStatus,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanPublication,
    ScientificAgentPlanSourceChanged,
    _exclusive_process_lock,
    _fsync_directory,
    _pretty_json_bytes,
    _read_exact_bytes,
    _read_stable_file,
    _safe_artifact_path,
    _safe_relative_artifact_path,
    _safe_scope_id,
    _write_exclusive,
)


CONTROLLER_POLICY_VERSION = "scientific-agent-harness-controller-policy.v1"
CONTROLLER_REQUEST_VERSION = "agent_harness_controller_request_checkpoint.v1"
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_CONTROLLER_REQUEST_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)
_POLICY_MATERIAL: Mapping[str, Any] = {
    "schema_version": CONTROLLER_POLICY_VERSION,
    "recognized_action_kinds": sorted(item.value for item in AgentHarnessControllerAction),
    "next_action_precedence": [
        "verify_authority",
        "observe_prior_receipt",
        "observe_task_outputs",
        "gate",
        "remote_request",
        "remote_approval",
        "remote_dispatch",
        "remote_refresh",
        "remote_recovery",
        "remote_adoption",
        "local_execution",
        "terminal_observation",
    ],
    "exact_authority_verification": [
        "start_intent",
        "authorization",
        "authorized_start_permission",
        "proposal_publication",
        "remote_authority_set_when_required",
    ],
    "route_separation": {
        "local": "run_plan_executor.execute_one_task",
        "remote": "remote_execution_lifecycle.task_slot",
    },
    "gate_policy": "exact_current_snapshot_positive_server_actor_decision",
    "remote_approval_policy": "record_only_dispatch_is_separate",
    "effect_policy": "at_most_one_bounded_effect_per_decision",
    "attempt_policy": "explicit_zero_based_attempt_ordinal_no_auto_retry",
    "recovery_policy": "reconcile_committed_authority_never_rerun_unknown_local_effect",
    "adoption_policy": "verified_stage_registry_or_remote_publication_only",
    "terminal_policy": "authoritative_sources_plus_committed_controller_receipt",
    "source_binding_policy": "ids_and_sha256_digests_only",
    "privacy_allowlist": "no_paths_hosts_commands_prompts_exceptions_or_trace_identity",
    "tracing_policy": "optional_fail_open_non_authoritative",
    "authority_expansion": "forbidden",
    "reason_codes": sorted(
        {
            "ALL_TASKS_COMPLETED",
            "GATE_APPROVAL_REQUIRED",
            "GATE_DECISION_COMMITTED",
            "GATE_SNAPSHOT_READY",
            "LOCAL_TASK_COMPLETED",
            "LOCAL_TASK_FAILED",
            "LOCAL_TASK_READY",
            "REMOTE_APPROVAL_REQUIRED",
            "REMOTE_APPROVAL_RECORDED",
            "REMOTE_DISPATCH_READY",
            "REMOTE_EXECUTION_CANCELLED",
            "REMOTE_EXECUTION_FAILED",
            "REMOTE_EXECUTION_RUNNING",
            "REMOTE_OUTPUTS_ADOPTED",
            "REMOTE_RECOVERY_ATTEMPTED",
            "REMOTE_RECOVERY_REQUIRED",
            "REMOTE_REQUEST_PREPARED",
            "REMOTE_REQUEST_READY",
            "TASK_COMPLETED",
            "TASK_INPUTS_UNAVAILABLE",
            "TERMINAL_OBSERVED",
        }
    ),
}
CONTROLLER_POLICY_DIGEST = _agent_digest(_POLICY_MATERIAL)


class ScientificAgentHarnessControllerError(ValueError):
    """Base privacy-safe Controller failure."""


class ScientificAgentHarnessControllerConflict(ScientificAgentHarnessControllerError):
    """An immutable request, decision, or receipt is bound to other bytes."""


class ScientificAgentHarnessControllerVerificationError(
    ScientificAgentHarnessControllerError
):
    """A current authority or result failed exact verification."""


class ScientificAgentHarnessControllerRecoveryRequired(
    ScientificAgentHarnessControllerError
):
    """A prior effect cannot be safely repeated automatically."""


@dataclass(frozen=True)
class ControllerRequestSession:
    project_id: str
    operation: str
    scope_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class ControllerAdvanceResult:
    execution: AgentHarnessControllerExecution
    inspection: AgentHarnessControllerInspection
    decision: AgentHarnessControllerDecision | None = None
    receipt: AgentHarnessControllerActionReceipt | None = None


class AgentHarnessControllerStore:
    """Request checkpoints layered over the existing immutable control store."""

    def __init__(self, *, control_store: AgentPlanControlStore) -> None:
        self.control_store = control_store

    @contextmanager
    def scope_session(
        self,
        *,
        project_id: str,
        operation: str,
        scope_id: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        operation_id = _safe_scope_id(operation, field="operation")
        clean_scope = _safe_scope_id(scope_id, field="scope_id")
        control = self.control_store._control_root(project_id=project, create=True)
        if control is None:  # pragma: no cover
            raise ScientificAgentHarnessControllerError("Controller storage unavailable")
        root = self._directory(control, "controller_requests")
        operation_root = self._directory(root, operation_id)
        scope_root = self._directory(operation_root, clean_scope)
        lock = scope_root / "scope.lock"
        if lock.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller scope lock is unsafe")
        with _CONTROLLER_REQUEST_LOCKS[str(lock.resolve())], _exclusive_process_lock(lock):
            yield

    @contextmanager
    def request_session(
        self,
        *,
        project_id: str,
        operation: str,
        scope_id: str,
        client_request_id: str,
        request_digest: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        operation_id = _safe_scope_id(operation, field="operation")
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        clean_scope = _safe_scope_id(scope_id, field="scope_id")
        control = self.control_store._control_root(project_id=project, create=True)
        if control is None:  # pragma: no cover
            raise ScientificAgentHarnessControllerError("Controller storage unavailable")
        root = self._directory(control, "controller_requests")
        operation_root = self._directory(root, operation_id)
        scope_root = self._directory(operation_root, clean_scope)
        request_dir = self._directory(scope_root, request_id)
        lock = request_dir / "request.lock"
        if lock.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller request lock is unsafe")
        with _CONTROLLER_REQUEST_LOCKS[str(lock.resolve())], _exclusive_process_lock(lock):
            session = ControllerRequestSession(
                project_id=project,
                operation=operation_id,
                scope_id=clean_scope,
                client_request_id=request_id,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            self.write_or_verify(
                request_dir / "reservation.json",
                {
                    "schema_version": CONTROLLER_REQUEST_VERSION,
                    "status": "RESERVED",
                    "project_id": project,
                    "operation": operation_id,
                    "scope_id": clean_scope,
                    "client_request_id": request_id,
                    "request_digest": request_digest,
                },
            )
            yield session

    def write_marker(
        self,
        session: ControllerRequestSession,
        *,
        filename: str,
        status: str,
        values: Mapping[str, Any],
    ) -> None:
        self.write_or_verify(
            session.request_dir / filename,
            {
                "schema_version": CONTROLLER_REQUEST_VERSION,
                "status": status,
                "project_id": session.project_id,
                "operation": session.operation,
                "scope_id": session.scope_id,
                "client_request_id": session.client_request_id,
                "request_digest": session.request_digest,
                **dict(values),
            },
        )

    @staticmethod
    def read_marker(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is unsafe")
        try:
            payload = json.loads(
                _read_exact_bytes(path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES)
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ScientificAgentHarnessControllerConflict(
                "Controller checkpoint failed exact verification"
            ) from exc
        if not isinstance(payload, dict):
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is invalid")
        return payload

    @staticmethod
    def write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
        expected = _pretty_json_bytes(dict(payload))
        if path.is_symlink():
            raise ScientificAgentHarnessControllerConflict("Controller checkpoint is unsafe")
        if path.exists():
            actual = _read_exact_bytes(
                path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES
            )
            if actual != expected:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller request ID is bound to different content"
                )
            return
        try:
            _write_exclusive(path, expected)
        except FileExistsError:
            actual = _read_exact_bytes(
                path, label="Controller checkpoint", max_bytes=_MAX_REQUEST_BYTES
            )
            if actual != expected:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller request ID is bound to different content"
                )

    @staticmethod
    def _directory(parent: Path, name: str) -> Path:
        path = parent / name
        if path.is_symlink():
            raise ScientificAgentHarnessControllerError("Controller storage is unsafe")
        if not path.exists():
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(parent)
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentHarnessControllerError("Controller storage is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(parent.resolve()):
            raise ScientificAgentHarnessControllerError("Controller storage escapes project scope")
        return resolved


class ScientificAgentHarnessController:
    """Server-owned deterministic Controller with an at-most-one-action API."""

    def __init__(
        self,
        *,
        storage: Any,
        proposal_store: Any,
        authorization_service: Any,
        control_store: AgentPlanControlStore,
        resource_authority_service: Any,
        executor: RunPlanExecutor,
        remote_executions: Any,
        tracer: HarnessTracer | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.control_store = control_store
        self.resource_authority_service = resource_authority_service
        self.executor = executor
        self.remote_executions = remote_executions
        self.tracer = tracer or NoopHarnessTracer()
        self.clock = clock
        self.requests = AgentHarnessControllerStore(control_store=control_store)

    def create(
        self,
        *,
        project_id: str,
        start_intent_id: str,
        request: AgentHarnessControllerStartRequest,
        actor: str,
        actor_source: str,
    ) -> ControllerAdvanceResult:
        request_digest = self._request_digest(
            project_id=project_id,
            operation="create",
            scope_id=start_intent_id,
            request=request.model_dump(mode="json"),
            actor=actor,
            actor_source=actor_source,
        )
        scope = self._scope_id("create", start_intent_id)
        with self.tracer.start_span(
            "controller.execution",
            attributes={"controller_policy_version": CONTROLLER_POLICY_VERSION},
        ):
            with self.requests.scope_session(
                project_id=project_id,
                operation="create",
                scope_id=scope,
            ), self.requests.request_session(
                project_id=project_id,
                operation="create",
                scope_id=scope,
                client_request_id=request.client_request_id,
                request_digest=request_digest,
            ) as session:
                marker = self.requests.read_marker(session.request_dir / "execution.json")
                if marker is not None:
                    execution = self.verify_execution(
                        project_id=project_id,
                        controller_execution_id=str(marker.get("controller_execution_id") or ""),
                    )
                else:
                    existing = self.control_store.list_harness_controller_executions(
                        project_id=project_id,
                        start_intent_id=start_intent_id,
                    )
                    if len(existing) > 1:
                        raise ScientificAgentHarnessControllerVerificationError(
                            "start intent has multiple Controller executions"
                        )
                    if existing:
                        execution = existing[0]
                        if (
                            execution.request_digest != request_digest
                            or execution.client_request_id != request.client_request_id
                            or execution.actor != actor
                            or execution.actor_source != actor_source
                        ):
                            raise ScientificAgentHarnessControllerConflict(
                                "start intent is already consumed by another request"
                            )
                        execution = self.verify_execution(
                            project_id=project_id,
                            controller_execution_id=execution.controller_execution_id,
                        )
                    else:
                        intent = self.authorization_service.verify_start_intent(
                            project_id=project_id,
                            start_intent_id=start_intent_id,
                            verify_current=True,
                        )
                        if intent.start_intent_digest != request.expected_start_intent_digest:
                            raise ScientificAgentHarnessControllerConflict(
                                "start intent digest does not match the current authority"
                            )
                        authorization = self.authorization_service.verify_authorization(
                            project_id=project_id,
                            authorization_id=intent.authorization_id,
                            verify_current=True,
                        )
                        publication = self.proposal_store.read(
                            project_id=project_id,
                            proposal_id=intent.proposal_id,
                            verify_current=True,
                        )
                        permission = self.control_store.read_permission_decision(
                            project_id=project_id,
                            decision_id=intent.permission_decision_id,
                        )
                        execution = self._build_execution(
                            intent=intent,
                            authorization=authorization,
                            publication=publication,
                            permission=permission,
                            actor=actor,
                            actor_source=actor_source,
                            client_request_id=request.client_request_id,
                            request_digest=request_digest,
                            created_at=self.clock(),
                        )
                        self.control_store.publish_harness_controller_execution(execution)
                    self.requests.write_marker(
                        session,
                        filename="execution.json",
                        status="EXECUTION_COMMITTED",
                        values={
                            "controller_execution_id": execution.controller_execution_id,
                            "controller_execution_digest": execution.execution_digest,
                        },
                    )
                return self._advance_in_session(execution=execution, session=session)

    def get(
        self, *, project_id: str, controller_execution_id: str
    ) -> ControllerAdvanceResult:
        execution = self.verify_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        return ControllerAdvanceResult(execution=execution, inspection=self._inspect(execution))

    def advance(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
    ) -> ControllerAdvanceResult:
        execution = self.verify_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        if execution.execution_digest != request.expected_controller_execution_digest:
            raise ScientificAgentHarnessControllerConflict(
                "Controller execution digest does not match the current authority"
            )
        request_digest = self._request_digest(
            project_id=project_id,
            operation="advance",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self.tracer.start_span(
            "controller.advance",
            attributes={"controller_execution_id": execution.controller_execution_id},
        ):
            with self.requests.request_session(
                project_id=project_id,
                operation="advance",
                scope_id=self._scope_id("advance", controller_execution_id),
                client_request_id=request.client_request_id,
                request_digest=request_digest,
            ) as session:
                return self._advance_in_session(execution=execution, session=session)

    def approve_gate(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        gate_id: str,
        request: AgentHarnessGateApprovalRequest,
        actor: str,
    ) -> ControllerAdvanceResult:
        execution = self.verify_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        request_digest = self._request_digest(
            project_id=project_id,
            operation="gate-approval",
            scope_id=f"{controller_execution_id}:{gate_id}",
            request=request.model_dump(mode="json"),
            actor=actor,
        )
        with self.requests.request_session(
            project_id=project_id,
            operation="gate-approval",
            scope_id=self._scope_id("gate-approval", f"{controller_execution_id}:{gate_id}"),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            marker = self.requests.read_marker(session.request_dir / "side_effect_observed.json")
            if marker is not None:
                return ControllerAdvanceResult(
                    execution=execution, inspection=self._inspect(execution)
                )
            inspection = self._inspect(execution)
            if (
                inspection.status != AgentHarnessControllerStatus.WAITING_GATE
                and inspection.next_action
                != AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller is not waiting for a Gate"
                )
            slot = self._current_slot(execution, inspection)
            authorization = self._authorization(execution, verify_current=False)
            if gate_id not in authorization.pending_gates:
                raise ScientificAgentHarnessControllerVerificationError(
                    "Gate is not pending in the exact authorization"
                )
            if gate_id in authorization.preauthorized_operational_gates:
                raise ScientificAgentHarnessControllerVerificationError(
                    "preauthorized Gate cannot be approved through the stepwise route"
                )
            gate_decision = self.executor.commit_one_task_gate_decision(
                project_id=project_id,
                run_plan=authorization.run_plan,
                task_index=slot.planned_task_index,
                task_id=slot.task_id,
                gate_id=gate_id,
                approved=True,
                actor=actor,
                note=request.note,
                expected_snapshot_id=request.expected_snapshot_id,
                expected_snapshot_digest=request.expected_snapshot_hash,
                task_options=authorization.compiled_task_options[slot.task_id],
            )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={
                    "gate_decision_digest": _agent_digest(
                        gate_decision.model_dump(mode="json")
                    )
                },
            )
            return ControllerAdvanceResult(
                execution=execution, inspection=self._inspect(execution)
            )

    def approve_remote(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessRemoteApprovalRequest,
        actor: str,
    ) -> ControllerAdvanceResult:
        execution = self.verify_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        request_digest = self._request_digest(
            project_id=project_id,
            operation="remote-approval",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
            actor=actor,
        )
        with self.requests.request_session(
            project_id=project_id,
            operation="remote-approval",
            scope_id=self._scope_id("remote-approval", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            marker = self.requests.read_marker(session.request_dir / "side_effect_observed.json")
            if marker is not None:
                return ControllerAdvanceResult(
                    execution=execution, inspection=self._inspect(execution)
                )
            inspection = self._inspect(execution)
            if (
                inspection.status
                != AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL
                and inspection.next_action
                != AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller is not waiting for remote approval"
                )
            slot = self._current_slot(execution, inspection)
            binding = self._remote_slot_binding(execution, slot)
            if binding.request_sha256 != request.expected_remote_request_sha256:
                raise ScientificAgentHarnessControllerConflict(
                    "remote approval does not bind the current exact request"
                )
            result = self.remote_executions.record_approval(
                project_id=project_id,
                run_id=execution.run_id,
                request_sha256=binding.request_sha256,
                actor=actor,
                note=request.note,
                slot_id=slot.slot_id,
                expected_slot_binding_digest=binding.slot_binding_digest,
            )
            approval = result.get("approval")
            if not isinstance(approval, dict) or not approval.get("approval_sha256"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "remote approval commit is unavailable"
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"remote_approval_digest": approval["approval_sha256"]},
            )
            return ControllerAdvanceResult(
                execution=execution, inspection=self._inspect(execution)
            )

    def cancel(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
    ) -> ControllerAdvanceResult:
        execution = self._verified_request_execution(
            project_id, controller_execution_id, request
        )
        request_digest = self._request_digest(
            project_id=project_id,
            operation="cancel",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self.requests.request_session(
            project_id=project_id,
            operation="cancel",
            scope_id=self._scope_id("cancel", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            return self._explicit_remote_action_in_session(
                execution=execution,
                session=session,
                action=AgentHarnessControllerAction.CANCEL_EXECUTION,
                require_recovery=False,
            )

    def recover(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentHarnessControllerAdvanceRequest,
    ) -> ControllerAdvanceResult:
        execution = self._verified_request_execution(
            project_id, controller_execution_id, request
        )
        request_digest = self._request_digest(
            project_id=project_id,
            operation="recover",
            scope_id=controller_execution_id,
            request=request.model_dump(mode="json"),
        )
        with self.requests.request_session(
            project_id=project_id,
            operation="recover",
            scope_id=self._scope_id("recover", controller_execution_id),
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            return self._explicit_remote_action_in_session(
                execution=execution,
                session=session,
                action=AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
                require_recovery=True,
            )

    def verify_execution(
        self, *, project_id: str, controller_execution_id: str
    ) -> AgentHarnessControllerExecution:
        execution = self.control_store.read_harness_controller_execution(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
        )
        try:
            intent = self.authorization_service.verify_start_intent(
                project_id=project_id,
                start_intent_id=execution.start_intent_id,
                verify_current=True,
            )
            authorization = self.authorization_service.verify_authorization(
                project_id=project_id,
                authorization_id=execution.authorization_id,
                verify_current=True,
            )
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=execution.proposal_id,
                verify_current=True,
            )
        except ScientificAgentPlanSourceChanged:
            # A committed Controller action intentionally changes StageState
            # and/or Registry, so the pre-execution planner snapshot cannot
            # remain byte-identical.  Accept only those dynamic changes that
            # are anchored by the latest immutable receipt (or one committed
            # decision awaiting crash reconciliation), while independently
            # rechecking static sources and exact input contents.
            intent = self.authorization_service.verify_start_intent(
                project_id=project_id,
                start_intent_id=execution.start_intent_id,
                verify_current=False,
            )
            authorization = self.authorization_service.verify_authorization(
                project_id=project_id,
                authorization_id=execution.authorization_id,
                verify_current=False,
            )
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=execution.proposal_id,
                verify_current=False,
            )
            self._verify_post_start_sources(execution, authorization, publication)
        permission = self.control_store.read_permission_decision(
            project_id=project_id,
            decision_id=execution.permission_decision_id,
        )
        expected = self._build_execution(
            intent=intent,
            authorization=authorization,
            publication=publication,
            permission=permission,
            actor=execution.actor,
            actor_source=execution.actor_source,
            client_request_id=execution.client_request_id,
            request_digest=execution.request_digest,
            created_at=execution.created_at,
        )
        if expected.model_dump(mode="json") != execution.model_dump(mode="json"):
            raise ScientificAgentHarnessControllerVerificationError(
                "Controller execution no longer matches current exact authority"
            )
        return execution

    def _build_execution(
        self,
        *,
        intent: Any,
        authorization: AgentPlanAuthorization,
        publication: ScientificAgentPlanPublication,
        permission: Any,
        actor: str,
        actor_source: str,
        client_request_id: str,
        request_digest: str,
        created_at: str,
    ) -> AgentHarnessControllerExecution:
        if (
            permission.phase != AgentPermissionPhase.AUTHORIZED_START
            or permission.outcome != AgentPermissionOutcome.ALLOW
            or permission.decision_digest != intent.permission_decision_digest
            or permission.authorization_id != authorization.authorization_id
            or permission.authorization_digest != authorization.authorization_digest
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "start permission is not an exact authorized-start ALLOW"
            )
        if publication.proposal.proposal_digest != authorization.proposal_digest:
            raise ScientificAgentHarnessControllerVerificationError(
                "proposal does not match authorization"
            )
        dispatch_by_task = {item.task_id: item for item in authorization.dispatch_intents}
        ordered = [item.task_id for item in authorization.run_plan.tasks]
        remote_bindings: dict[str, Any] = {}
        authority_set = None
        for task_id in ordered:
            intent_binding = dispatch_by_task[task_id]
            if intent_binding.execution_route != "remote_execution_service":
                continue
            current = self.resource_authority_service.current_authority(
                publication=publication,
                task_id=task_id,
            )
            if authority_set is None:
                authority_set = current.authority_set
            elif current.authority_set.model_dump(mode="json") != authority_set.model_dump(
                mode="json"
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "remote tasks do not share one complete current AuthoritySet"
                )
            remote_bindings[task_id] = current.authority
        slots: list[AgentHarnessControllerTaskSlot] = []
        dispatch_digests: dict[str, str] = {}
        for index, task in enumerate(authorization.run_plan.tasks):
            dispatch = dispatch_by_task[task.task_id]
            dispatch_digest = _agent_digest(dispatch.model_dump(mode="json"))
            dispatch_digests[task.task_id] = dispatch_digest
            remote = remote_bindings.get(task.task_id)
            slot_identity = _agent_digest(
                {
                    "schema_version": "agent_harness_task_slot_identity.v1",
                    "start_intent_id": intent.start_intent_id,
                    "task_index": index,
                    "task_id": task.task_id,
                    "attempt_ordinal": 0,
                }
            )
            slots.append(
                AgentHarnessControllerTaskSlot(
                    planned_task_index=index,
                    task_id=task.task_id,
                    attempt=0,
                    execution_route=dispatch.execution_route,
                    slot_id=f"harness-slot-{slot_identity.split(':', 1)[1][:32]}",
                    task_authority_digest=authorization.task_authority_digests[task.task_id],
                    dispatch_intent_digest=dispatch_digest,
                    compiled_options_digest=_agent_digest(
                        authorization.compiled_task_options[task.task_id]
                    ),
                    input_artifacts_digest=_agent_digest(
                        {"required_artifact_ids": list(task.required_artifacts)}
                    ),
                    output_contract_digest=_agent_digest(
                        {
                            "task_id": task.task_id,
                            "output_artifact_ids": list(task.output_artifacts),
                        }
                    ),
                    remote_authority_id=remote.authority_id if remote else "",
                    remote_authority_digest=remote.authority_digest if remote else "",
                )
            )
        sources = [
            self._source("start_intent", intent.start_intent_id, intent.start_intent_digest),
            self._source(
                "authorization", authorization.authorization_id, authorization.authorization_digest
            ),
            self._source("permission", permission.decision_id, permission.decision_digest),
            self._source(
                "proposal", publication.proposal.proposal_id, publication.proposal.proposal_digest
            ),
        ]
        if authority_set is not None:
            sources.append(
                self._source(
                    "remote_authority_set",
                    authority_set.authority_set_id,
                    authority_set.authority_set_digest,
                )
            )
        unavailable_budget = _agent_digest(
            {"schema_version": "agent_harness_remote_budget_unavailable.v1", "reason": "local-only-plan"}
        )
        return AgentHarnessControllerExecution(
            project_id=authorization.project_id,
            run_id=authorization.run_id,
            start_intent_id=intent.start_intent_id,
            start_intent_digest=intent.start_intent_digest,
            authorization_id=authorization.authorization_id,
            authorization_digest=authorization.authorization_digest,
            authorization_mode=authorization.authorization_mode,
            permission_decision_id=permission.decision_id,
            permission_decision_digest=permission.decision_digest,
            permission_policy_version=permission.policy_version,
            permission_policy_digest=permission.policy_digest,
            proposal_id=publication.proposal.proposal_id,
            proposal_digest=publication.proposal.proposal_digest,
            semantic_plan_id=publication.proposal.semantic_plan_id,
            semantic_plan_digest=publication.proposal.semantic_plan_digest,
            observation_id=publication.observation.observation_id,
            observation_digest=publication.observation.observation_digest,
            tool_catalog_digest=publication.catalog.catalog_digest,
            run_plan_digest=authorization.run_plan_digest,
            ordered_task_ids=ordered,
            task_roster_digest=_agent_digest(
                {"schema_version": "agent_harness_controller_task_roster.v1", "task_ids": ordered}
            ),
            task_authority_digests=authorization.task_authority_digests,
            dispatch_intent_digests=dispatch_digests,
            compiled_task_options_digest=_agent_digest(authorization.compiled_task_options),
            artifact_binding_digest=_agent_digest(
                [item.model_dump(mode="json") for item in authorization.artifact_bindings]
            ),
            gate_binding_digest=_agent_digest(
                [item.model_dump(mode="json") for item in authorization.gate_bindings]
            ),
            budget_binding_digest=_agent_digest(authorization.limits),
            remote_authority_set_id=authority_set.authority_set_id if authority_set else "",
            remote_authority_set_digest=authority_set.authority_set_digest if authority_set else "",
            remote_authority_roster_digest=authority_set.complete_roster_digest if authority_set else "",
            aggregate_budget_digest=(
                authority_set.aggregate_budget_digest if authority_set else unavailable_budget
            ),
            task_slots=slots,
            source_bindings=sources,
            source_bindings_digest=_agent_digest(
                [item.model_dump(mode="json") for item in sources]
            ),
            controller_policy_digest=CONTROLLER_POLICY_DIGEST,
            actor=actor,
            actor_source=actor_source,
            client_request_id=client_request_id,
            request_digest=request_digest,
            created_at=created_at,
        )

    def _inspect(
        self,
        execution: AgentHarnessControllerExecution,
        *,
        verify_authority: bool = False,
    ) -> AgentHarnessControllerInspection:
        authorization = self._authorization(execution, verify_current=verify_authority)
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        receipts = self.control_store.list_harness_controller_action_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        facts = [
            AgentHarnessControllerInspectionFact(
                name="controller_execution",
                authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                source_id=execution.controller_execution_id,
                source_digest=execution.execution_digest,
                state="verified",
            ),
            AgentHarnessControllerInspectionFact(
                name="artifact_registry",
                authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                source_id=f"registry-{execution.run_id}",
                source_digest=_agent_digest(registry),
                state="verified",
            ),
        ]
        if stage is not None:
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="stage_state",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    source_id=f"stage-{execution.run_id}",
                    source_digest=_agent_digest(stage.model_dump(mode="json")),
                    state=stage.status.value.lower(),
                )
            )
        completed = 0
        for slot in execution.task_slots:
            task = authorization.run_plan.tasks[slot.planned_task_index]
            if self._local_task_completed(slot, task, stage, registry, receipts):
                completed += 1
                continue
            if slot.execution_route == "remote_execution_service":
                remote = self._remote_inspection_or_none(execution, slot)
                if remote is not None and self._remote_task_completed(
                    execution, remote, receipts, slot, registry
                ):
                    completed += 1
                    continue
            break
        if completed == len(execution.task_slots):
            return self._inspection(
                execution, AgentHarnessControllerStatus.SUCCEEDED, None,
                AgentHarnessControllerAction.COMPLETE_EXECUTION, facts
            )
        slot = execution.task_slots[completed]
        task = authorization.run_plan.tasks[slot.planned_task_index]
        missing = [item for item in task.required_artifacts if item not in registry]
        if missing:
            facts.append(
                AgentHarnessControllerInspectionFact(
                    name="task_inputs",
                    authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
                    state="unavailable",
                    detail="required logical artifact IDs are not registered",
                )
            )
            return self._inspection(
                execution, AgentHarnessControllerStatus.FAILED, slot,
                AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts
            )
        if slot.execution_route == "remote_execution_service":
            return self._inspect_remote_action(execution, slot, facts)
        return self._inspect_local_action(execution, slot, stage, registry, facts)

    def _inspect_local_action(self, execution: Any, slot: Any, stage: Any, registry: Any, facts: list[Any]):
        task = self._authorization(execution, verify_current=False).run_plan.tasks[slot.planned_task_index]
        spec = self.executor.registry.get(task.task_id)
        if stage is not None and stage.stage == task.task_id:
            if stage.status == RunStatus.RUNNING:
                return self._inspection(execution, AgentHarnessControllerStatus.RECOVERY_REQUIRED, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)
            if stage.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
                status = AgentHarnessControllerStatus.FAILED if stage.status == RunStatus.FAILED else AgentHarnessControllerStatus.CANCELLED
                return self._inspection(execution, status, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)
        if spec.gates:
            if stage is None or stage.stage != task.task_id or stage.status != RunStatus.WAITING_USER:
                return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.PREPARE_LOCAL_GATE, facts)
            snapshot = stage.details.get("execution_snapshot")
            if not isinstance(snapshot, dict):
                raise ScientificAgentHarnessControllerVerificationError("waiting Gate snapshot is unavailable")
            decisions = self._gate_decisions(execution, slot, snapshot, spec.gates)
            if len(decisions) != len(spec.gates):
                return self._inspection(execution, AgentHarnessControllerStatus.WAITING_GATE, slot, AgentHarnessControllerAction.WAIT_FOR_GATE, facts)
            if any(not item.approved for item in decisions.values()):
                return self._inspection(execution, AgentHarnessControllerStatus.FAILED, slot, AgentHarnessControllerAction.STOP_GATE_REJECTED, facts)
        return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.EXECUTE_LOCAL_TASK, facts)

    def _inspect_remote_action(self, execution: Any, slot: Any, facts: list[Any]):
        remote = self._remote_inspection_or_none(execution, slot)
        if remote is None:
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST, facts)
        state = str(remote["state"]["status"])
        request = remote["request"]
        facts.append(AgentHarnessControllerInspectionFact(
            name="remote_request", authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
            source_id=str(request["request_id"]), source_digest=str(request["request_sha256"]), state=state.lower()
        ))
        if state == "WAITING_APPROVAL":
            return self._inspection(execution, AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL, slot, AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL, facts)
        if state == "APPROVED":
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.DISPATCH_REMOTE_TASK, facts)
        if state in {"ACCEPTED", "RUNNING", "CANCEL_REQUESTED"}:
            return self._inspection(execution, AgentHarnessControllerStatus.RUNNING_REMOTE, slot, AgentHarnessControllerAction.REFRESH_REMOTE_TASK, facts)
        if state == "RECOVERY_REQUIRED":
            return self._inspection(execution, AgentHarnessControllerStatus.RECOVERY_REQUIRED, slot, AgentHarnessControllerAction.RECOVER_REMOTE_TASK, facts)
        if state == "SUCCEEDED":
            return self._inspection(execution, AgentHarnessControllerStatus.ACTIVE, slot, AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS, facts)
        status = AgentHarnessControllerStatus.CANCELLED if state == "CANCELLED" else AgentHarnessControllerStatus.FAILED
        return self._inspection(execution, status, slot, AgentHarnessControllerAction.STOP_TASK_TERMINAL, facts)

    def _advance_in_session(self, *, execution: Any, session: ControllerRequestSession) -> ControllerAdvanceResult:
        receipt_marker = self.requests.read_marker(session.request_dir / "receipt.json")
        if receipt_marker is not None:
            receipt = self.control_store.read_harness_controller_action_receipt(
                project_id=execution.project_id,
                receipt_id=str(receipt_marker.get("receipt_id") or ""),
            )
            return ControllerAdvanceResult(execution, self._inspect(execution), receipt=receipt)
        decision_marker = self.requests.read_marker(session.request_dir / "decision.json")
        if decision_marker is not None:
            decision = self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=str(decision_marker.get("decision_id") or ""),
            )
        else:
            inspection = self._inspect(execution)
            predecessor = self._latest_receipt(execution)
            sources = self._bindings_from_facts(inspection.facts)
            slot = None if inspection.current_task_index is None else execution.task_slots[inspection.current_task_index]
            decision = AgentHarnessControllerDecision(
                controller_execution_id=execution.controller_execution_id,
                controller_execution_digest=execution.execution_digest,
                client_request_id=session.client_request_id,
                inspection_digest=inspection.inspection_digest,
                action_kind=inspection.next_action,
                task_id=slot.task_id if slot else "",
                task_index=slot.planned_task_index if slot else None,
                attempt_ordinal=slot.attempt if slot else 0,
                slot_id=slot.slot_id if slot else "",
                source_bindings=sources,
                source_bindings_digest=_agent_digest([item.model_dump(mode="json") for item in sources]),
                predecessor_receipt_id=predecessor.receipt_id if predecessor else "",
                reason_codes=[self._decision_reason(inspection.next_action)],
                created_at=self.clock(),
                executable=inspection.next_action not in {
                    AgentHarnessControllerAction.WAIT_FOR_GATE,
                    AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
                    AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
                },
            )
            self.control_store.publish_harness_controller_decision(
                project_id=execution.project_id, decision=decision
            )
            self.requests.write_marker(
                session, filename="decision.json", status="DECISION_COMMITTED",
                values={"decision_id": decision.decision_id, "decision_digest": decision.decision_digest}
            )
        effect_marker = self.requests.read_marker(
            session.request_dir / "side_effect_observed.json"
        )
        if effect_marker is not None:
            receipt_payload = effect_marker.get("receipt")
            if not isinstance(receipt_payload, dict):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint lacks its exact receipt"
                )
            receipt = AgentHarnessControllerActionReceipt.model_validate(receipt_payload)
            if (
                receipt.controller_execution_id != execution.controller_execution_id
                or receipt.decision_id != decision.decision_id
                or receipt.decision_digest != decision.decision_digest
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint authority mismatch"
                )
        else:
            execution = self.verify_execution(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            action_attributes: dict[str, str | int] = {
                "controller_execution_id": execution.controller_execution_id,
                "action_id": decision.decision_id,
                "decision_id": decision.decision_id,
                "action": decision.action_kind.value,
            }
            if decision.task_id:
                action_attributes["task_id"] = decision.task_id
                action_attributes["task_index"] = decision.task_index or 0
                action_attributes["attempt"] = decision.attempt_ordinal
                action_attributes["slot_id"] = decision.slot_id
            with self.tracer.start_span(
                "controller.action",
                attributes=action_attributes,
            ) as span:
                span.add_event(
                    "controller.decision",
                    {
                        "decision_id": decision.decision_id,
                        "action": decision.action_kind.value,
                    },
                )
                try:
                    receipt = self._execute_decision(execution, decision)
                except Exception:
                    span.record_error("CONTROLLER_ACTION_FAILED")
                    raise
                span.set_attribute("outcome", receipt.outcome.value)
                span.set_attribute("receipt_digest", receipt.receipt_digest)
                span.add_event(
                    "controller.receipt",
                    {
                        "outcome": receipt.outcome.value,
                        "receipt_digest": receipt.receipt_digest,
                    },
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
        self.control_store.publish_harness_controller_action_receipt(
            project_id=execution.project_id, receipt=receipt
        )
        self.requests.write_marker(
            session, filename="receipt.json", status="RECEIPT_COMMITTED",
            values={"receipt_id": receipt.receipt_id, "receipt_digest": receipt.receipt_digest}
        )
        return ControllerAdvanceResult(execution, self._inspect(execution), decision, receipt)

    def _explicit_remote_action_in_session(
        self,
        *,
        execution: AgentHarnessControllerExecution,
        session: ControllerRequestSession,
        action: AgentHarnessControllerAction,
        require_recovery: bool,
    ) -> ControllerAdvanceResult:
        receipt_marker = self.requests.read_marker(session.request_dir / "receipt.json")
        if receipt_marker is not None:
            receipt = self.control_store.read_harness_controller_action_receipt(
                project_id=execution.project_id,
                receipt_id=str(receipt_marker.get("receipt_id") or ""),
            )
            return ControllerAdvanceResult(
                execution=execution,
                inspection=self._inspect(execution),
                receipt=receipt,
            )
        decision_marker = self.requests.read_marker(session.request_dir / "decision.json")
        if decision_marker is not None:
            decision = self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=str(decision_marker.get("decision_id") or ""),
            )
            if decision.action_kind != action:
                raise ScientificAgentHarnessControllerConflict(
                    "Controller control request is bound to another action"
                )
        else:
            inspection = self._inspect(execution)
            if require_recovery and inspection.status != AgentHarnessControllerStatus.RECOVERY_REQUIRED:
                raise ScientificAgentHarnessControllerConflict("Controller is not in recovery")
            if (
                not require_recovery
                and inspection.status
                in {
                    AgentHarnessControllerStatus.SUCCEEDED,
                    AgentHarnessControllerStatus.FAILED,
                    AgentHarnessControllerStatus.CANCELLED,
                }
            ):
                raise ScientificAgentHarnessControllerConflict(
                    "terminal Controller execution cannot be cancelled"
                )
            slot = self._current_slot(execution, inspection)
            if slot.execution_route != "remote_execution_service":
                if require_recovery:
                    raise ScientificAgentHarnessControllerRecoveryRequired(
                        "local unknown outcomes cannot be rerun automatically"
                    )
                raise ScientificAgentHarnessControllerConflict(
                    "only the current exact remote slot can be cancelled"
                )
            predecessor = self._latest_receipt(execution)
            sources = self._bindings_from_facts(inspection.facts)
            decision = AgentHarnessControllerDecision(
                controller_execution_id=execution.controller_execution_id,
                controller_execution_digest=execution.execution_digest,
                client_request_id=session.client_request_id,
                inspection_digest=inspection.inspection_digest,
                action_kind=action,
                task_id=slot.task_id,
                task_index=slot.planned_task_index,
                attempt_ordinal=slot.attempt,
                slot_id=slot.slot_id,
                source_bindings=sources,
                source_bindings_digest=_agent_digest(
                    [item.model_dump(mode="json") for item in sources]
                ),
                predecessor_receipt_id=predecessor.receipt_id if predecessor else "",
                reason_codes=[self._decision_reason(action)],
                created_at=self.clock(),
                executable=True,
            )
            self.control_store.publish_harness_controller_decision(
                project_id=execution.project_id,
                decision=decision,
            )
            self.requests.write_marker(
                session,
                filename="decision.json",
                status="DECISION_COMMITTED",
                values={
                    "decision_id": decision.decision_id,
                    "decision_digest": decision.decision_digest,
                },
            )
        effect_marker = self.requests.read_marker(
            session.request_dir / "side_effect_observed.json"
        )
        if effect_marker is not None:
            receipt_payload = effect_marker.get("receipt")
            if not isinstance(receipt_payload, dict):
                raise ScientificAgentHarnessControllerConflict(
                    "Controller effect checkpoint lacks its exact receipt"
                )
            receipt = AgentHarnessControllerActionReceipt.model_validate(receipt_payload)
        else:
            execution = self.verify_execution(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
            with self.tracer.start_span(
                "controller.action",
                attributes={
                    "controller_execution_id": execution.controller_execution_id,
                    "action_id": decision.decision_id,
                    "decision_id": decision.decision_id,
                    "action": decision.action_kind.value,
                    "task_id": decision.task_id,
                    "task_index": decision.task_index or 0,
                    "attempt": decision.attempt_ordinal,
                    "slot_id": decision.slot_id,
                },
            ) as span:
                span.add_event(
                    "controller.decision",
                    {
                        "decision_id": decision.decision_id,
                        "action": decision.action_kind.value,
                    },
                )
                try:
                    receipt = self._execute_decision(execution, decision)
                except Exception:
                    span.record_error("CONTROLLER_ACTION_FAILED")
                    raise
                span.set_attribute("outcome", receipt.outcome.value)
                span.set_attribute("receipt_digest", receipt.receipt_digest)
                span.add_event(
                    "controller.receipt",
                    {
                        "outcome": receipt.outcome.value,
                        "receipt_digest": receipt.receipt_digest,
                    },
                )
            self.requests.write_marker(
                session,
                filename="side_effect_observed.json",
                status="SIDE_EFFECT_OBSERVED",
                values={"receipt": receipt.model_dump(mode="json")},
            )
        if (
            receipt.controller_execution_id != execution.controller_execution_id
            or receipt.decision_id != decision.decision_id
            or receipt.decision_digest != decision.decision_digest
        ):
            raise ScientificAgentHarnessControllerConflict(
                "Controller control receipt authority mismatch"
            )
        self.control_store.publish_harness_controller_action_receipt(
            project_id=execution.project_id,
            receipt=receipt,
        )
        self.requests.write_marker(
            session,
            filename="receipt.json",
            status="RECEIPT_COMMITTED",
            values={
                "receipt_id": receipt.receipt_id,
                "receipt_digest": receipt.receipt_digest,
            },
        )
        return ControllerAdvanceResult(
            execution=execution,
            inspection=self._inspect(execution),
            decision=decision,
            receipt=receipt,
        )

    def _execute_decision(self, execution: Any, decision: AgentHarnessControllerDecision) -> AgentHarnessControllerActionReceipt:
        before_stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        before_registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        slot = execution.task_slots[decision.task_index] if decision.task_index is not None else None
        action = decision.action_kind
        result: dict[str, Any] = {}
        outcome = AgentHarnessControllerReceiptOutcome.COMMITTED
        execution_started = False
        dispatch_occurred = False
        reason = self._receipt_reason(action)
        if action in {AgentHarnessControllerAction.WAIT_FOR_GATE, AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL}:
            outcome = AgentHarnessControllerReceiptOutcome.WAITING
        elif action == AgentHarnessControllerAction.PREPARE_LOCAL_GATE:
            assert slot is not None
            current = self.storage.read_stage_state(execution.project_id, execution.run_id)
            if current is not None and current.stage == slot.task_id and current.status == RunStatus.WAITING_USER:
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                result = self._prepare_local_gate(execution, slot)
        elif action == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK:
            assert slot is not None
            if self._local_outputs_committed(execution, slot):
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                current = self.storage.read_stage_state(execution.project_id, execution.run_id)
                if current is not None and current.stage == slot.task_id and current.status == RunStatus.RUNNING:
                    raise ScientificAgentHarnessControllerRecoveryRequired("local task outcome is unknown")
                execution_started = True
                dispatch_occurred = True
                with self.tracer.start_span(
                    "executor.local_task",
                    attributes={
                        "controller_execution_id": execution.controller_execution_id,
                        "task_id": slot.task_id,
                        "task_index": slot.planned_task_index,
                        "attempt": slot.attempt,
                        "execution_route": slot.execution_route,
                    },
                ):
                    result = self._execute_local(execution, slot)
                if result.get("status") != RunStatus.SUCCEEDED.value:
                    outcome = AgentHarnessControllerReceiptOutcome.FAILED
                    reason = "LOCAL_TASK_FAILED"
                else:
                    reason = "TASK_COMPLETED"
        elif action == AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST:
            assert slot is not None
            remote = self._remote_inspection_or_none(execution, slot)
            if remote is not None:
                result = remote
                outcome = AgentHarnessControllerReceiptOutcome.RECONCILED
            else:
                with self.tracer.start_span(
                    "remote.prepare",
                    attributes={
                        "controller_execution_id": execution.controller_execution_id,
                        "task_id": slot.task_id,
                        "task_index": slot.planned_task_index,
                        "attempt": slot.attempt,
                        "slot_id": slot.slot_id,
                    },
                ):
                    result = self._prepare_remote(execution, slot)
        elif action == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            execution_started = True
            dispatch_occurred = True
            with self.tracer.start_span(
                "remote.dispatch",
                attributes={
                    "controller_execution_id": execution.controller_execution_id,
                    "task_id": slot.task_id,
                    "task_index": slot.planned_task_index,
                    "attempt": slot.attempt,
                    "slot_id": slot.slot_id,
                },
            ):
                result = self.remote_executions.dispatch(
                    project_id=execution.project_id, run_id=execution.run_id,
                    request_sha256=binding.request_sha256, slot_id=slot.slot_id,
                    expected_slot_binding_digest=binding.slot_binding_digest,
                )
        elif action == AgentHarnessControllerAction.REFRESH_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            with self.tracer.start_span(
                "remote.refresh",
                attributes={
                    "controller_execution_id": execution.controller_execution_id,
                    "task_id": slot.task_id,
                    "task_index": slot.planned_task_index,
                    "attempt": slot.attempt,
                    "slot_id": slot.slot_id,
                },
            ):
                result = self.remote_executions.refresh(
                    project_id=execution.project_id, run_id=execution.run_id,
                    slot_id=slot.slot_id, expected_slot_binding_digest=binding.slot_binding_digest,
                )
        elif action == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS:
            assert slot is not None
            result = self._remote_inspection(execution, slot)
            if str(result["state"]["status"]) != "SUCCEEDED" or result.get("publication") is None:
                raise ScientificAgentHarnessControllerVerificationError("remote success publication is unavailable")
            reason = "TASK_COMPLETED"
        elif action == AgentHarnessControllerAction.RECOVER_REMOTE_TASK:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            with self.tracer.start_span(
                "remote.recover",
                attributes={
                    "controller_execution_id": execution.controller_execution_id,
                    "task_id": slot.task_id,
                    "task_index": slot.planned_task_index,
                    "attempt": slot.attempt,
                    "slot_id": slot.slot_id,
                },
            ):
                result = self.remote_executions.recover(
                    project_id=execution.project_id,
                    run_id=execution.run_id,
                    slot_id=slot.slot_id,
                    expected_slot_binding_digest=binding.slot_binding_digest,
                )
            reason = "REMOTE_RECOVERY_ATTEMPTED"
        elif action == AgentHarnessControllerAction.CANCEL_EXECUTION:
            assert slot is not None
            binding = self._remote_slot_binding(execution, slot)
            result = self.remote_executions.cancel(
                project_id=execution.project_id,
                run_id=execution.run_id,
                request_sha256=binding.request_sha256,
                slot_id=slot.slot_id,
                expected_slot_binding_digest=binding.slot_binding_digest,
            )
            reason = "REMOTE_EXECUTION_CANCELLED"
        elif action in {AgentHarnessControllerAction.STOP_TASK_TERMINAL, AgentHarnessControllerAction.STOP_GATE_REJECTED, AgentHarnessControllerAction.STOP_REMOTE_REJECTED}:
            outcome = AgentHarnessControllerReceiptOutcome.FAILED
            reason = "TERMINAL_OBSERVED"
        after_stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        after_registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        after_inspection = self._inspect(execution, verify_authority=False)
        sources = self._bindings_from_facts(after_inspection.facts)
        remote_request = result.get("request") if isinstance(result, dict) else None
        remote_approval = result.get("approval") if isinstance(result, dict) else None
        remote_publication = result.get("publication") if isinstance(result, dict) else None
        gate_snapshot = self._stage_snapshot(after_stage, slot.task_id if slot else "") or self._stage_snapshot(
            before_stage, slot.task_id if slot else ""
        )
        return AgentHarnessControllerActionReceipt(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            decision_id=decision.decision_id,
            decision_digest=decision.decision_digest,
            action_kind=action,
            task_id=slot.task_id if slot else "",
            task_index=slot.planned_task_index if slot else None,
            attempt_ordinal=slot.attempt if slot else 0,
            slot_id=slot.slot_id if slot else "",
            execution_started=execution_started,
            dispatch_occurred=dispatch_occurred,
            before_stage_digest=self._stage_digest(before_stage),
            after_stage_digest=self._stage_digest(after_stage),
            before_artifact_registry_digest=_agent_digest(before_registry),
            after_artifact_registry_digest=_agent_digest(after_registry),
            local_dispatch_receipt_ids=[],
            remote_execution_slot_id=slot.slot_id if slot and slot.execution_route == "remote_execution_service" else "",
            remote_request_id=str((remote_request or {}).get("request_id") or ""),
            remote_request_sha256=str((remote_request or {}).get("request_sha256") or ""),
            remote_approval_digest=str((remote_approval or {}).get("approval_sha256") or ""),
            remote_publication_digest=str((remote_publication or {}).get("publication_sha256") or ""),
            gate_snapshot_id=str(gate_snapshot.get("snapshot_id") or ""),
            gate_snapshot_hash=str(gate_snapshot.get("snapshot_digest") or ""),
            gate_decision_digest=self._gate_decision_roster_digest(execution, slot, gate_snapshot),
            outcome=outcome,
            status_after=after_inspection.status,
            source_bindings=sources,
            source_bindings_digest=_agent_digest([item.model_dump(mode="json") for item in sources]),
            reason_codes=[reason],
            created_at=self.clock(),
        )

    def _prepare_local_gate(self, execution: Any, slot: Any) -> dict[str, Any]:
        authorization = self._authorization(execution, verify_current=False)
        binding = self.executor.derive_one_task_server_binding(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index,
            task_options=authorization.compiled_task_options[slot.task_id],
        )
        return self.executor.prepare_one_task_gate(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_id=slot.task_id,
            task_options=authorization.compiled_task_options[slot.task_id],
            expected_local_adapter_execution_binding_digest=binding["local_adapter_execution_binding_digest"],
            expected_compiled_options_digest=slot.compiled_options_digest,
            expected_input_artifacts_digest=binding["input_artifacts_digest"],
            expected_output_contract_digest=slot.output_contract_digest,
        )

    def _execute_local(self, execution: Any, slot: Any) -> dict[str, Any]:
        authorization = self._authorization(execution, verify_current=False)
        options = authorization.compiled_task_options[slot.task_id]
        binding = self.executor.derive_one_task_server_binding(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_options=options,
        )
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        snapshot = self._stage_snapshot(stage, slot.task_id)
        common = dict(
            project_id=execution.project_id, run_plan=authorization.run_plan,
            task_index=slot.planned_task_index, task_id=slot.task_id,
            task_options=options,
            expected_local_adapter_execution_binding_digest=binding["local_adapter_execution_binding_digest"],
            expected_compiled_options_digest=slot.compiled_options_digest,
            expected_input_artifacts_digest=binding["input_artifacts_digest"],
            expected_output_contract_digest=slot.output_contract_digest,
        )
        if snapshot:
            return self.executor.execute_one_task_after_committed_gate(
                **common, actor=execution.actor,
                expected_snapshot_id=snapshot["snapshot_id"],
                expected_snapshot_digest=snapshot["snapshot_digest"],
            )
        return self.executor.execute_one_task(**common)

    def _prepare_remote(self, execution: Any, slot: Any) -> dict[str, Any]:
        authorization = self._authorization(execution, verify_current=False)
        publication = self.proposal_store.read(
            project_id=execution.project_id, proposal_id=execution.proposal_id, verify_current=False
        )
        current = self.resource_authority_service.current_authority(
            publication=publication, task_id=slot.task_id
        )
        authority = current.authority
        if authority.authority_id != slot.remote_authority_id or authority.authority_digest != slot.remote_authority_digest:
            raise ScientificAgentHarnessControllerVerificationError("remote task authority changed")
        connection = self.remote_executions.profiles.get_connection(authority.connection_id)
        profile = self.remote_executions.profiles.resolve_execution_profile(authority.execution_profile_id)
        task = authorization.run_plan.tasks[slot.planned_task_index]
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        input_root = run_dir / "agent-harness-controller-inputs" / slot.slot_id
        input_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptors: list[dict[str, str]] = []
        bindings: dict[str, str] = {}
        for index, artifact_id in enumerate(task.required_artifacts):
            registered = registry.get(artifact_id)
            if not registered:
                raise ScientificAgentHarnessControllerVerificationError("remote input is unavailable")
            relative = _safe_relative_artifact_path(registered)
            source = _safe_artifact_path(run_dir, relative, label="remote input")
            payload, exists = _read_stable_file(source, label="remote input", max_bytes=2 * 1024 * 1024 * 1024)
            if not exists:
                raise ScientificAgentHarnessControllerVerificationError("remote input is unavailable")
            suffix = source.suffix.lower() or ".json"
            destination_name = f"input-{index:04d}{suffix}"
            purpose, media_type = self._remote_input_contract(profile.task_type, suffix)
            self._write_exact(input_root / destination_name, payload)
            descriptors.append({"relative_path": destination_name, "purpose": purpose, "media_type": media_type})
            bindings[destination_name] = artifact_id
        if not descriptors:
            payload = _pretty_json_bytes({
                "schema_version": "agent_harness_remote_execution_input.v1",
                "task_id": slot.task_id,
                "compiled_task_options": authorization.compiled_task_options[slot.task_id],
            })
            destination_name = "execution-request.json"
            purpose, media_type = self._remote_input_contract(profile.task_type, ".json")
            self._write_exact(input_root / destination_name, payload)
            artifact_identity = _agent_digest({"slot_id": slot.slot_id, "payload": json.loads(payload)})
            artifact_id = f"harness-input-{artifact_identity.split(':', 1)[1][:32]}"
            relative = str((input_root / destination_name).relative_to(run_dir))
            current_path = registry.get(artifact_id)
            if current_path is None:
                self.storage.register_new_artifact_registry_paths(
                    execution.project_id, execution.run_id, {artifact_id: relative}
                )
            elif current_path != relative:
                raise ScientificAgentHarnessControllerConflict("remote synthetic input binding changed")
            descriptors.append({"relative_path": destination_name, "purpose": purpose, "media_type": media_type})
            bindings[destination_name] = artifact_id
        request_identity = _agent_digest({"controller_execution_id": execution.controller_execution_id, "slot_id": slot.slot_id})
        manifest = build_transfer_manifest(
            request_id=f"remote-{request_identity.split(':', 1)[1][:32]}",
            input_root=input_root,
            artifacts=descriptors,
            connection=connection,
            execution_profile=profile,
            target_purpose=profile.task_type.replace("_", "-"),
        )
        return self.remote_executions.prepare(
            project_id=execution.project_id, run_id=execution.run_id, task_id=slot.task_id,
            transfer_manifest=manifest,
            requested_resources=authority.configured_resources.model_dump(mode="json"),
            input_artifacts=bindings, slot_id=slot.slot_id,
            slot_binding_authority={
                "controller_execution_id": execution.controller_execution_id,
                "controller_execution_digest": execution.execution_digest,
                "planned_task_index": slot.planned_task_index,
                "attempt": slot.attempt,
                "task_authority_digest": slot.task_authority_digest,
                "dispatch_intent_digest": slot.dispatch_intent_digest,
                "compiled_options_digest": slot.compiled_options_digest,
                "input_artifacts_digest": slot.input_artifacts_digest,
                "output_contract_digest": slot.output_contract_digest,
                "remote_authority_id": slot.remote_authority_id,
                "remote_authority_digest": slot.remote_authority_digest,
                "remote_authority_set_id": execution.remote_authority_set_id,
                "remote_authority_set_digest": execution.remote_authority_set_digest,
            },
        )

    def _remote_inspection_or_none(self, execution: Any, slot: Any) -> dict[str, Any] | None:
        slot_path = (
            self.storage.run_dir(execution.project_id, execution.run_id)
            / "remote-executions"
            / slot.slot_id
        )
        if not slot_path.exists():
            return None
        if slot_path.is_symlink() or not slot_path.is_dir():
            raise ScientificAgentHarnessControllerVerificationError(
                "remote task slot storage is unsafe"
            )
        try:
            return self._remote_inspection(execution, slot)
        except FileNotFoundError:
            return None

    def _remote_inspection(self, execution: Any, slot: Any) -> dict[str, Any]:
        binding = self._remote_slot_binding(execution, slot)
        return self.remote_executions.inspect(
            project_id=execution.project_id, run_id=execution.run_id,
            slot_id=slot.slot_id, expected_slot_binding_digest=binding.slot_binding_digest,
        )

    def _remote_slot_binding(self, execution: Any, slot: Any):
        binding = self.remote_executions.inspect_slot_binding(
            project_id=execution.project_id, run_id=execution.run_id, slot_id=slot.slot_id
        )
        expected = {
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "planned_task_index": slot.planned_task_index,
            "task_id": slot.task_id,
            "attempt": slot.attempt,
            "task_authority_digest": slot.task_authority_digest,
            "dispatch_intent_digest": slot.dispatch_intent_digest,
            "compiled_options_digest": slot.compiled_options_digest,
            "input_artifacts_digest": slot.input_artifacts_digest,
            "output_contract_digest": slot.output_contract_digest,
            "remote_authority_id": slot.remote_authority_id,
            "remote_authority_digest": slot.remote_authority_digest,
            "remote_authority_set_id": execution.remote_authority_set_id,
            "remote_authority_set_digest": execution.remote_authority_set_digest,
        }
        if any(getattr(binding, key) != value for key, value in expected.items()):
            raise ScientificAgentHarnessControllerVerificationError("remote slot authority mismatch")
        return binding

    def _authorization(
        self, execution: Any, *, verify_current: bool = True
    ) -> AgentPlanAuthorization:
        return self.authorization_service.verify_authorization(
            project_id=execution.project_id, authorization_id=execution.authorization_id,
            verify_current=verify_current,
        )

    @staticmethod
    def _inspection(execution: Any, status: Any, slot: Any, action: Any, facts: list[Any]):
        return AgentHarnessControllerInspection(
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            status=status,
            current_task_index=slot.planned_task_index if slot else None,
            current_task_id=slot.task_id if slot else "",
            current_slot_id=slot.slot_id if slot else "",
            next_action=action,
            facts=facts,
            source_roster_digest=_agent_digest([item.model_dump(mode="json") for item in facts]),
            inspected_at=now_iso(),
        )

    @staticmethod
    def _source(name: str, source_id: str, source_digest: str) -> AgentHarnessControllerSourceBinding:
        return AgentHarnessControllerSourceBinding(name=name, source_id=source_id, source_digest=source_digest)

    @staticmethod
    def _bindings_from_facts(facts: list[AgentHarnessControllerInspectionFact]) -> list[AgentHarnessControllerSourceBinding]:
        return [
            AgentHarnessControllerSourceBinding(
                name=item.name, source_id=item.source_id, source_digest=item.source_digest,
                authority_class=item.authority_class,
            )
            for item in facts if item.source_id and item.source_digest
        ]

    def _latest_receipt(self, execution: Any):
        receipts = self.control_store.list_harness_controller_action_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        if not receipts:
            return None
        predecessor_ids = {
            self.control_store.read_harness_controller_decision(
                project_id=execution.project_id,
                decision_id=item.decision_id,
            ).predecessor_receipt_id
            for item in receipts
        }
        candidates = [item for item in receipts if item.receipt_id not in predecessor_ids]
        if len(candidates) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "Controller receipt chain is forked or incomplete"
            )
        return candidates[0]

    def _verify_post_start_sources(
        self,
        execution: AgentHarnessControllerExecution,
        authorization: AgentPlanAuthorization,
        publication: ScientificAgentPlanPublication,
    ) -> None:
        current_observation = self.proposal_store.observation_builder.build(
            project_id=publication.observation.project_id,
            run_id=publication.observation.run_id,
            goal=publication.observation.goal_context,
            user_constraints=publication.observation.explicit_constraints,
        )
        original_sources = {
            item.source_id: item for item in publication.observation.source_bindings
        }
        current_sources = {
            item.source_id: item for item in current_observation.source_bindings
        }
        for source_id in (
            "existing_run_plan",
            "scientific_tool_catalog",
            "resource_profile_snapshot",
        ):
            original = original_sources.get(source_id)
            current = current_sources.get(source_id)
            if original is None or current is None or current.model_dump(mode="json") != original.model_dump(mode="json"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "non-execution authority changed after Controller start"
                )
        original_artifacts = {
            item.artifact_id: item for item in publication.observation.available_artifacts
        }
        current_artifacts = {
            item.artifact_id: item for item in current_observation.available_artifacts
        }
        for artifact_id, original in original_artifacts.items():
            current = current_artifacts.get(artifact_id)
            if current is None or current.model_dump(mode="json") != original.model_dump(mode="json"):
                raise ScientificAgentHarnessControllerVerificationError(
                    "pre-existing artifact authority changed after Controller start"
                )
        run_dir = self.storage.run_dir(execution.project_id, execution.run_id)
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        for binding in authorization.artifact_bindings:
            relative = registry.get(binding.artifact_id)
            if not relative:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized input artifact is no longer registered"
                )
            source = _safe_artifact_path(
                run_dir,
                _safe_relative_artifact_path(relative),
                label="authorized Controller input",
            )
            payload, present = _read_stable_file(
                source,
                label="authorized Controller input",
                max_bytes=2 * 1024 * 1024 * 1024,
            )
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            if not present or digest != binding.content_digest:
                raise ScientificAgentHarnessControllerVerificationError(
                    "authorized input artifact content changed"
                )
        latest = self._latest_receipt(execution)
        decisions = self.control_store.list_harness_controller_decisions(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        receipted_decisions = {
            item.decision_id
            for item in self.control_store.list_harness_controller_action_receipts(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
        }
        unreceipted = [item for item in decisions if item.decision_id not in receipted_decisions]
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        stage_digest = self._stage_digest(stage)
        registry_digest = _agent_digest(registry)
        if not unreceipted:
            if latest is None:
                raise ScientificAgentHarnessControllerVerificationError(
                    "unanchored execution source change"
                )
            if (
                latest.after_stage_digest != stage_digest
                or latest.after_artifact_registry_digest != registry_digest
            ):
                raise ScientificAgentHarnessControllerVerificationError(
                    "execution sources differ from the latest immutable receipt"
                )
            return
        if len(unreceipted) != 1:
            raise ScientificAgentHarnessControllerVerificationError(
                "multiple Controller effects lack receipts"
            )
        decision = unreceipted[0]
        if stage is not None and decision.task_id and stage.stage != decision.task_id:
            raise ScientificAgentHarnessControllerVerificationError(
                "unreceipted StageState belongs to another task"
            )
        allowed_new_ids = {
            artifact_id
            for task in authorization.run_plan.tasks[: (decision.task_index or 0) + 1]
            for artifact_id in task.output_artifacts
        }
        unexpected = set(registry).difference(original_artifacts).difference(allowed_new_ids)
        if any(
            not item.startswith("harness-input-")
            and not item.startswith("remote_execution_publication_")
            for item in unexpected
        ):
            raise ScientificAgentHarnessControllerVerificationError(
                "unreceipted Registry mutation is outside the selected task contract"
            )

    @staticmethod
    def _local_task_completed(slot: Any, task: Any, stage: Any, registry: dict[str, str], receipts: list[Any]) -> bool:
        if slot.execution_route != "local_executor":
            return False
        receipt = any(item.task_id == slot.task_id and "TASK_COMPLETED" in item.reason_codes and item.outcome in {AgentHarnessControllerReceiptOutcome.COMMITTED, AgentHarnessControllerReceiptOutcome.RECONCILED} for item in receipts)
        history_ok = bool(stage and any(item.stage == slot.task_id and item.status == RunStatus.SUCCEEDED for item in stage.history))
        current_ok = bool(stage and stage.stage == slot.task_id and stage.status == RunStatus.SUCCEEDED)
        outputs_ok = all(item in registry for item in task.output_artifacts)
        return receipt and outputs_ok and (history_ok or current_ok)

    def _local_outputs_committed(self, execution: Any, slot: Any) -> bool:
        task = self._authorization(execution, verify_current=False).run_plan.tasks[slot.planned_task_index]
        registry = self.storage.read_artifact_registry(execution.project_id, execution.run_id)
        stage = self.storage.read_stage_state(execution.project_id, execution.run_id)
        return bool(stage and stage.stage == slot.task_id and stage.status == RunStatus.SUCCEEDED and all(item in registry for item in task.output_artifacts))

    def _remote_task_completed(
        self,
        execution: Any,
        remote: dict[str, Any],
        receipts: list[Any],
        slot: Any,
        registry: dict[str, str],
    ) -> bool:
        task = self._authorization(
            execution, verify_current=False
        ).run_plan.tasks[slot.planned_task_index]
        publication = remote.get("publication")
        publication_ids = {
            str(item.get("artifact_id") or "")
            for item in (publication or {}).get("artifacts", [])
        }
        outputs_ok = all(
            item in registry and item in publication_ids for item in task.output_artifacts
        )
        return str(remote["state"]["status"]) == "SUCCEEDED" and publication is not None and outputs_ok and any(
            item.task_id == slot.task_id and "TASK_COMPLETED" in item.reason_codes and item.outcome in {AgentHarnessControllerReceiptOutcome.COMMITTED, AgentHarnessControllerReceiptOutcome.RECONCILED}
            for item in receipts
        )

    def _gate_decisions(self, execution: Any, slot: Any, snapshot: Mapping[str, Any], gates: list[str]):
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        snapshot_hash = str(snapshot.get("snapshot_hash") or "")
        result = {}
        for raw in self.storage.read_gate_decisions(execution.project_id, execution.run_id):
            from ai4s_agent.schemas import GateDecision
            item = GateDecision.model_validate(raw)
            if item.gate.value in gates and item.approved_snapshot_id == snapshot_id and item.approved_snapshot_hash == snapshot_hash:
                if item.gate.value in result and result[item.gate.value].model_dump() != item.model_dump():
                    raise ScientificAgentHarnessControllerVerificationError("Gate snapshot has conflicting decisions")
                result[item.gate.value] = item
        return result

    def _gate_decision_roster_digest(self, execution: Any, slot: Any, snapshot: Mapping[str, Any]) -> str:
        if slot is None or not snapshot:
            return ""
        spec = self.executor.registry.get(slot.task_id)
        decisions = self._gate_decisions(execution, slot, {"snapshot_id": snapshot["snapshot_id"], "snapshot_hash": snapshot["snapshot_digest"].split(":", 1)[1]}, spec.gates)
        if not decisions:
            return ""
        return _agent_digest([decisions[key].model_dump(mode="json") for key in sorted(decisions)])

    @staticmethod
    def _stage_snapshot(stage: Any, task_id: str) -> dict[str, str]:
        if stage is None or stage.stage != task_id:
            return {}
        raw = stage.details.get("execution_snapshot")
        if not isinstance(raw, dict):
            return {}
        snapshot_id = str(raw.get("snapshot_id") or "")
        snapshot_hash = str(raw.get("snapshot_hash") or "")
        if not snapshot_id or len(snapshot_hash) != 64:
            return {}
        return {"snapshot_id": snapshot_id, "snapshot_digest": f"sha256:{snapshot_hash}"}

    @staticmethod
    def _stage_digest(stage: Any) -> str:
        return "" if stage is None else _agent_digest(stage.model_dump(mode="json"))

    @staticmethod
    def _decision_reason(action: AgentHarnessControllerAction) -> str:
        return {
            AgentHarnessControllerAction.PREPARE_LOCAL_GATE: "GATE_SNAPSHOT_READY",
            AgentHarnessControllerAction.WAIT_FOR_GATE: "GATE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.EXECUTE_LOCAL_TASK: "LOCAL_TASK_READY",
            AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: "REMOTE_REQUEST_READY",
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: "REMOTE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: "REMOTE_DISPATCH_READY",
            AgentHarnessControllerAction.REFRESH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.RECOVER_REMOTE_TASK: "REMOTE_RECOVERY_REQUIRED",
            AgentHarnessControllerAction.CANCEL_EXECUTION: "REMOTE_EXECUTION_CANCELLED",
            AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: "REMOTE_OUTPUTS_ADOPTED",
            AgentHarnessControllerAction.COMPLETE_EXECUTION: "ALL_TASKS_COMPLETED",
        }.get(action, "TERMINAL_OBSERVED")

    @staticmethod
    def _receipt_reason(action: AgentHarnessControllerAction) -> str:
        return {
            AgentHarnessControllerAction.PREPARE_LOCAL_GATE: "GATE_SNAPSHOT_READY",
            AgentHarnessControllerAction.WAIT_FOR_GATE: "GATE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: "REMOTE_REQUEST_PREPARED",
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: "REMOTE_APPROVAL_REQUIRED",
            AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.REFRESH_REMOTE_TASK: "REMOTE_EXECUTION_RUNNING",
            AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: "TASK_COMPLETED",
            AgentHarnessControllerAction.COMPLETE_EXECUTION: "ALL_TASKS_COMPLETED",
        }.get(action, "TERMINAL_OBSERVED")

    @staticmethod
    def _remote_input_contract(task_type: str, suffix: str) -> tuple[str, str]:
        if suffix == ".pdf":
            return "source-pdf", "application/pdf"
        if suffix == ".csv":
            return "training-data", "application/csv"
        if suffix in {".parquet", ".pq"}:
            return "training-data", "application/parquet"
        if suffix == ".toml":
            return "generator-config", "application/toml"
        if suffix == ".json":
            if task_type == "document_parsing":
                return "corpus-manifest", "application/json"
            if task_type == "model_training":
                return "training-config", "application/json"
            return "execution-request", "application/json"
        raise ScientificAgentHarnessControllerVerificationError("remote input media type is not allowed")

    @staticmethod
    def _write_exact(path: Path, payload: bytes) -> None:
        if path.exists():
            actual, exists = _read_stable_file(path, label="Controller remote input", max_bytes=2 * 1024 * 1024 * 1024)
            if not exists or actual != payload:
                raise ScientificAgentHarnessControllerConflict("Controller remote input changed")
            return
        _write_exclusive(path, payload)

    def _verified_request_execution(self, project_id: str, execution_id: str, request: AgentHarnessControllerAdvanceRequest):
        execution = self.verify_execution(project_id=project_id, controller_execution_id=execution_id)
        if execution.execution_digest != request.expected_controller_execution_digest:
            raise ScientificAgentHarnessControllerConflict("Controller execution digest does not match")
        return execution

    @staticmethod
    def _current_slot(execution: Any, inspection: AgentHarnessControllerInspection):
        if inspection.current_task_index is None:
            raise ScientificAgentHarnessControllerConflict("Controller has no current task")
        return execution.task_slots[inspection.current_task_index]

    @staticmethod
    def _scope_id(operation: str, value: str) -> str:
        digest = _agent_digest({"operation": operation, "scope": value})
        return f"scope-{digest.split(':', 1)[1][:32]}"

    @staticmethod
    def _request_digest(*, project_id: str, operation: str, scope_id: str, request: Mapping[str, Any], actor: str = "", actor_source: str = "") -> str:
        return _agent_digest({
            "schema_version": CONTROLLER_REQUEST_VERSION,
            "project_id": project_id,
            "operation": operation,
            "scope_id": scope_id,
            "request": dict(request),
            "actor": actor,
            "actor_source": actor_source,
        })
