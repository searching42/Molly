"""Immutable Scientific Agent authorization and start-intent control plane v1.

The public service sequence is intentionally fixed:

verified PR-BL proposal -> deterministic permission decision -> immutable
authorization publication -> exact re-verification -> immutable start intent.

No function in this module imports or calls the Executor, remote lifecycle,
adapter registry, worker queue, GateDecision storage, or StageState storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.harness_tracing import HarnessTracer, NoopHarnessTracer
from ai4s_agent.observability_correlation import (
    build_harness_telemetry_correlation,
    privacy_safe_telemetry_attributes,
)
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AgentAuthorizationArtifactBinding,
    AgentAuthorizationGateBinding,
    AgentAuthorizationMode,
    AgentAuthorizationProfileBinding,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerDecision,
    AgentHarnessControllerExecution,
    AGENT_EXECUTION_PLAN_PROPOSAL_V2,
    AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2,
    AgentHarnessLocalDispatchReceipt,
    AgentHarnessLocalExecutionPublication,
    AgentPermissionDecision,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPermissionShadowRecord,
    AgentRemoteResourceAuthority,
    AgentRemoteResourceAuthorityDecision,
    AgentRemoteResourceAuthoritySet,
    AgentPlanAuthorization,
    AGENT_PLAN_AUTHORIZATION_V1,
    AGENT_PLAN_AUTHORIZATION_V2,
    AgentPlanAuthorizationRequest,
    AgentPlanStartIntent,
    _agent_digest,
)
from ai4s_agent.scientific_agent_permissions import (
    ScientificAgentPermissionEngine,
    compare_permission_outcomes,
    derive_legacy_route_expectation,
)
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanError,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanPublication,
    ScientificAgentPlanPublicationConflict,
    ScientificAgentPlanSourceChanged,
    _exclusive_process_lock,
    _existing_project_dir,
    _fsync_directory,
    _pretty_json_bytes,
    _read_exact_bytes,
    _safe_scope_id,
    _write_exclusive,
)


CONTROL_ROOT_NAME = "agent_plan_control"
CONTROL_PUBLICATION_MANIFEST_VERSION = "agent_plan_control_publication_manifest.v1"
CONTROL_VERIFICATION_VERSION = "agent_plan_control_verification.v1"
AUTHORIZATION_REQUEST_BINDING_VERSION = "agent_plan_authorization_request_binding.v1"
AUTHORIZATION_CHECKPOINT_VERSION = "agent_plan_authorization_checkpoint.v1"
START_INTENT_CHECKPOINT_VERSION = "agent_plan_start_intent_checkpoint.v1"
AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG = "AI4S_ENABLE_AGENT_PERMISSION_SHADOW_OBSERVATION"

_MAX_CONTROL_BYTES = 16 * 1024 * 1024
_CONTROL_LAYOUT: Mapping[str, tuple[str, str, str, type[BaseModel]]] = {
    "permission_decision": (
        "permission_decisions",
        "permission_decision.json",
        "decision_digest",
        AgentPermissionDecision,
    ),
    "authorization": (
        "authorizations",
        "authorization.json",
        "authorization_digest",
        AgentPlanAuthorization,
    ),
    "start_intent": (
        "start_intents",
        "start_intent.json",
        "start_intent_digest",
        AgentPlanStartIntent,
    ),
    "shadow_record": (
        "shadow_records",
        "shadow_record.json",
        "shadow_record_digest",
        AgentPermissionShadowRecord,
    ),
    "remote_resource_authority_decision": (
        "remote_resource_authority_decisions",
        "remote_resource_authority_decision.json",
        "decision_digest",
        AgentRemoteResourceAuthorityDecision,
    ),
    "remote_resource_authority": (
        "remote_resource_authorities",
        "remote_resource_authority.json",
        "authority_digest",
        AgentRemoteResourceAuthority,
    ),
    "remote_resource_authority_set": (
        "remote_resource_authority_sets",
        "remote_resource_authority_set.json",
        "authority_set_digest",
        AgentRemoteResourceAuthoritySet,
    ),
    "harness_controller_execution": (
        "harness_controller_executions",
        "controller_execution.json",
        "execution_digest",
        AgentHarnessControllerExecution,
    ),
    "harness_controller_decision": (
        "harness_controller_decisions",
        "controller_decision.json",
        "decision_digest",
        AgentHarnessControllerDecision,
    ),
    "harness_controller_action_receipt": (
        "harness_controller_action_receipts",
        "controller_action_receipt.json",
        "receipt_digest",
        AgentHarnessControllerActionReceipt,
    ),
    "harness_local_dispatch_receipt": (
        "harness_local_dispatch_receipts",
        "local_dispatch_receipt.json",
        "dispatch_receipt_digest",
        AgentHarnessLocalDispatchReceipt,
    ),
    "harness_local_execution_publication": (
        "harness_local_execution_publications",
        "local_execution_publication.json",
        "publication_digest",
        AgentHarnessLocalExecutionPublication,
    ),
}
_CONTROL_ID_FIELDS = {
    "permission_decision": "decision_id",
    "authorization": "authorization_id",
    "start_intent": "start_intent_id",
    "shadow_record": "shadow_record_id",
    "remote_resource_authority_decision": "decision_id",
    "remote_resource_authority": "authority_id",
    "remote_resource_authority_set": "authority_set_id",
    "harness_controller_execution": "controller_execution_id",
    "harness_controller_decision": "decision_id",
    "harness_controller_action_receipt": "receipt_id",
    "harness_local_dispatch_receipt": "dispatch_receipt_id",
    "harness_local_execution_publication": "publication_id",
}

ModelT = TypeVar("ModelT", bound=BaseModel)


class ScientificAgentAuthorizationError(ValueError):
    """Base fail-closed control-plane error."""


class ScientificAgentAuthorizationConflict(ScientificAgentAuthorizationError):
    """An immutable request or authority ID is bound to different bytes."""


class ScientificAgentAuthorizationDenied(ScientificAgentAuthorizationError):
    """A deterministic permission decision denied the requested authority."""

    def __init__(self, decision: AgentPermissionDecision) -> None:
        self.decision = decision
        super().__init__("scientific agent plan authorization was denied")


class ScientificAgentAuthorizationVerificationError(ScientificAgentAuthorizationError):
    """Persisted control authority failed exact re-verification."""


@dataclass(frozen=True)
class AgentPlanControlRequestSession:
    project_id: str
    proposal_id: str
    client_request_id: str
    operation: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class ApproveAndStartResult:
    authorization: AgentPlanAuthorization
    start_intent: AgentPlanStartIntent
    authorization_decision: AgentPermissionDecision
    start_decision: AgentPermissionDecision


def _authorization_request_digest(
    *,
    project_id: str,
    proposal_id: str,
    operation: str,
    request: AgentPlanAuthorizationRequest,
    actor: str,
    actor_source: str,
) -> str:
    return _agent_digest(
        {
            "schema_version": AUTHORIZATION_REQUEST_BINDING_VERSION,
            "project_id": project_id,
            "proposal_id": proposal_id,
            "operation": operation,
            "request": request.model_dump(mode="json"),
            "actor": actor,
            "actor_source": actor_source,
        }
    )


def _start_intent_slot_id(
    *,
    project_id: str,
    proposal_id: str,
    proposal_digest: str,
) -> str:
    identity = _agent_digest(
        {
            "schema_version": "agent_plan_start_intent.v1",
            "project_id": project_id,
            "proposal_id": proposal_id,
            "proposal_digest": proposal_digest,
            "intent_type": "start_authorized_plan",
        }
    )
    return f"start-intent-{identity.split(':', 1)[1][:32]}"


class AgentPlanControlStore:
    """Project-scoped no-replace storage for all PR-BM control artifacts."""

    def __init__(
        self,
        *,
        storage: Any,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.fault_injector = fault_injector

    def _fault(self, phase: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(phase)

    def publish_permission_decision(
        self, decision: AgentPermissionDecision
    ) -> AgentPermissionDecision:
        return self._publish_model(
            project_id=decision.project_id,
            kind="permission_decision",
            artifact_id=decision.decision_id,
            model=decision,
        )

    def read_permission_decision(
        self, *, project_id: str, decision_id: str
    ) -> AgentPermissionDecision:
        return self._read_model(
            project_id=project_id,
            kind="permission_decision",
            artifact_id=decision_id,
            expected_type=AgentPermissionDecision,
        )

    def publish_authorization(
        self, authorization: AgentPlanAuthorization, *, staging_parent: Path | None = None
    ) -> AgentPlanAuthorization:
        return self._publish_model(
            project_id=authorization.project_id,
            kind="authorization",
            artifact_id=authorization.authorization_id,
            model=authorization,
            staging_parent=staging_parent,
        )

    def read_authorization(
        self, *, project_id: str, authorization_id: str
    ) -> AgentPlanAuthorization:
        return self._read_model(
            project_id=project_id,
            kind="authorization",
            artifact_id=authorization_id,
            expected_type=AgentPlanAuthorization,
        )

    def publish_start_intent(
        self, start_intent: AgentPlanStartIntent, *, staging_parent: Path | None = None
    ) -> AgentPlanStartIntent:
        return self._publish_model(
            project_id=start_intent.project_id,
            kind="start_intent",
            artifact_id=start_intent.start_intent_id,
            model=start_intent,
            staging_parent=staging_parent,
        )

    def read_start_intent(
        self, *, project_id: str, start_intent_id: str
    ) -> AgentPlanStartIntent:
        return self._read_model(
            project_id=project_id,
            kind="start_intent",
            artifact_id=start_intent_id,
            expected_type=AgentPlanStartIntent,
        )

    def publish_shadow_record(
        self, record: AgentPermissionShadowRecord
    ) -> AgentPermissionShadowRecord:
        return self._publish_model(
            project_id=record.project_id,
            kind="shadow_record",
            artifact_id=record.shadow_record_id,
            model=record,
        )

    def publish_remote_resource_authority_decision(
        self, decision: AgentRemoteResourceAuthorityDecision
    ) -> AgentRemoteResourceAuthorityDecision:
        return self._publish_model(
            project_id=decision.project_id,
            kind="remote_resource_authority_decision",
            artifact_id=decision.decision_id,
            model=decision,
        )

    def read_remote_resource_authority_decision(
        self, *, project_id: str, decision_id: str
    ) -> AgentRemoteResourceAuthorityDecision:
        return self._read_model(
            project_id=project_id,
            kind="remote_resource_authority_decision",
            artifact_id=decision_id,
            expected_type=AgentRemoteResourceAuthorityDecision,
        )

    def publish_remote_resource_authority(
        self,
        authority: AgentRemoteResourceAuthority,
        *,
        staging_parent: Path | None = None,
    ) -> AgentRemoteResourceAuthority:
        return self._publish_model(
            project_id=authority.project_id,
            kind="remote_resource_authority",
            artifact_id=authority.authority_id,
            model=authority,
            staging_parent=staging_parent,
        )

    def read_remote_resource_authority(
        self, *, project_id: str, authority_id: str
    ) -> AgentRemoteResourceAuthority:
        return self._read_model(
            project_id=project_id,
            kind="remote_resource_authority",
            artifact_id=authority_id,
            expected_type=AgentRemoteResourceAuthority,
        )

    def publish_remote_resource_authority_set(
        self,
        authority_set: AgentRemoteResourceAuthoritySet,
        *,
        staging_parent: Path | None = None,
    ) -> AgentRemoteResourceAuthoritySet:
        return self._publish_model(
            project_id=authority_set.project_id,
            kind="remote_resource_authority_set",
            artifact_id=authority_set.authority_set_id,
            model=authority_set,
            staging_parent=staging_parent,
        )

    def read_remote_resource_authority_set(
        self, *, project_id: str, authority_set_id: str
    ) -> AgentRemoteResourceAuthoritySet:
        return self._read_model(
            project_id=project_id,
            kind="remote_resource_authority_set",
            artifact_id=authority_set_id,
            expected_type=AgentRemoteResourceAuthoritySet,
        )

    def read_shadow_record(
        self, *, project_id: str, shadow_record_id: str
    ) -> AgentPermissionShadowRecord:
        return self._read_model(
            project_id=project_id,
            kind="shadow_record",
            artifact_id=shadow_record_id,
            expected_type=AgentPermissionShadowRecord,
        )

    def publish_harness_controller_execution(
        self, execution: AgentHarnessControllerExecution
    ) -> AgentHarnessControllerExecution:
        if execution.controller_policy_version != AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2:
            raise ScientificAgentAuthorizationVerificationError(
                "historical Controller execution is read-only and cannot be published"
            )
        return self._publish_model(
            project_id=execution.project_id,
            kind="harness_controller_execution",
            artifact_id=execution.controller_execution_id,
            model=execution,
        )

    def read_harness_controller_execution(
        self, *, project_id: str, controller_execution_id: str
    ) -> AgentHarnessControllerExecution:
        return self._read_model(
            project_id=project_id,
            kind="harness_controller_execution",
            artifact_id=controller_execution_id,
            expected_type=AgentHarnessControllerExecution,
        )

    def list_harness_controller_executions(
        self, *, project_id: str, start_intent_id: str | None = None
    ) -> list[AgentHarnessControllerExecution]:
        root = self._collection_root(
            project_id=project_id,
            kind="harness_controller_execution",
            create=False,
        )
        if root is None:
            return []
        executions: list[AgentHarnessControllerExecution] = []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ScientificAgentAuthorizationVerificationError(
                "controller execution collection exceeds its bounded roster"
            )
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ScientificAgentAuthorizationVerificationError(
                    "controller execution collection contains an unsafe entry"
                )
            execution = self.read_harness_controller_execution(
                project_id=project_id,
                controller_execution_id=child.name,
            )
            if start_intent_id is None or execution.start_intent_id == start_intent_id:
                executions.append(execution)
        return sorted(
            executions,
            key=lambda item: (item.created_at, item.controller_execution_id),
        )

    def publish_harness_controller_decision(
        self, *, project_id: str, decision: AgentHarnessControllerDecision
    ) -> AgentHarnessControllerDecision:
        return self._publish_model(
            project_id=project_id,
            kind="harness_controller_decision",
            artifact_id=decision.decision_id,
            model=decision,
        )

    def read_harness_controller_decision(
        self, *, project_id: str, decision_id: str
    ) -> AgentHarnessControllerDecision:
        return self._read_model(
            project_id=project_id,
            kind="harness_controller_decision",
            artifact_id=decision_id,
            expected_type=AgentHarnessControllerDecision,
        )

    def list_harness_controller_decisions(
        self, *, project_id: str, controller_execution_id: str
    ) -> list[AgentHarnessControllerDecision]:
        root = self._collection_root(
            project_id=project_id,
            kind="harness_controller_decision",
            create=False,
        )
        if root is None:
            return []
        decisions: list[AgentHarnessControllerDecision] = []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ScientificAgentAuthorizationVerificationError(
                "controller decision collection exceeds its bounded roster"
            )
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ScientificAgentAuthorizationVerificationError(
                    "controller decision collection contains an unsafe entry"
                )
            decision = self.read_harness_controller_decision(
                project_id=project_id,
                decision_id=child.name,
            )
            if decision.controller_execution_id == controller_execution_id:
                decisions.append(decision)
        return sorted(decisions, key=lambda item: (item.created_at, item.decision_id))

    def publish_harness_controller_action_receipt(
        self, *, project_id: str, receipt: AgentHarnessControllerActionReceipt
    ) -> AgentHarnessControllerActionReceipt:
        return self._publish_model(
            project_id=project_id,
            kind="harness_controller_action_receipt",
            artifact_id=receipt.receipt_id,
            model=receipt,
        )

    def read_harness_controller_action_receipt(
        self, *, project_id: str, receipt_id: str
    ) -> AgentHarnessControllerActionReceipt:
        return self._read_model(
            project_id=project_id,
            kind="harness_controller_action_receipt",
            artifact_id=receipt_id,
            expected_type=AgentHarnessControllerActionReceipt,
        )

    def list_harness_controller_action_receipts(
        self, *, project_id: str, controller_execution_id: str
    ) -> list[AgentHarnessControllerActionReceipt]:
        root = self._collection_root(
            project_id=project_id,
            kind="harness_controller_action_receipt",
            create=False,
        )
        if root is None:
            return []
        receipts: list[AgentHarnessControllerActionReceipt] = []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ScientificAgentAuthorizationVerificationError(
                "controller receipt collection exceeds its bounded roster"
            )
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ScientificAgentAuthorizationVerificationError(
                    "controller receipt collection contains an unsafe entry"
                )
            receipt = self.read_harness_controller_action_receipt(
                project_id=project_id,
                receipt_id=child.name,
            )
            if receipt.controller_execution_id == controller_execution_id:
                receipts.append(receipt)
        return sorted(receipts, key=lambda item: (item.created_at, item.receipt_id))

    def publish_harness_local_dispatch_receipt(
        self,
        *,
        project_id: str,
        receipt: AgentHarnessLocalDispatchReceipt,
    ) -> AgentHarnessLocalDispatchReceipt:
        return self._publish_model(
            project_id=project_id,
            kind="harness_local_dispatch_receipt",
            artifact_id=receipt.dispatch_receipt_id,
            model=receipt,
        )

    def read_harness_local_dispatch_receipt(
        self,
        *,
        project_id: str,
        dispatch_receipt_id: str,
    ) -> AgentHarnessLocalDispatchReceipt:
        return self._read_model(
            project_id=project_id,
            kind="harness_local_dispatch_receipt",
            artifact_id=dispatch_receipt_id,
            expected_type=AgentHarnessLocalDispatchReceipt,
        )

    def list_harness_local_dispatch_receipts(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
    ) -> list[AgentHarnessLocalDispatchReceipt]:
        root = self._collection_root(
            project_id=project_id,
            kind="harness_local_dispatch_receipt",
            create=False,
        )
        if root is None:
            return []
        result: list[AgentHarnessLocalDispatchReceipt] = []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ScientificAgentAuthorizationVerificationError(
                "local dispatch receipt collection exceeds its bounded roster"
            )
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ScientificAgentAuthorizationVerificationError(
                    "local dispatch receipt collection contains an unsafe entry"
                )
            receipt = self.read_harness_local_dispatch_receipt(
                project_id=project_id,
                dispatch_receipt_id=child.name,
            )
            if receipt.controller_execution_id == controller_execution_id:
                result.append(receipt)
        return sorted(
            result,
            key=lambda item: (item.created_at, item.dispatch_receipt_id),
        )

    def publish_harness_local_execution_publication(
        self,
        *,
        project_id: str,
        publication: AgentHarnessLocalExecutionPublication,
    ) -> AgentHarnessLocalExecutionPublication:
        return self._publish_model(
            project_id=project_id,
            kind="harness_local_execution_publication",
            artifact_id=publication.publication_id,
            model=publication,
        )

    def read_harness_local_execution_publication(
        self,
        *,
        project_id: str,
        publication_id: str,
    ) -> AgentHarnessLocalExecutionPublication:
        return self._read_model(
            project_id=project_id,
            kind="harness_local_execution_publication",
            artifact_id=publication_id,
            expected_type=AgentHarnessLocalExecutionPublication,
        )

    def list_harness_local_execution_publications(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
    ) -> list[AgentHarnessLocalExecutionPublication]:
        root = self._collection_root(
            project_id=project_id,
            kind="harness_local_execution_publication",
            create=False,
        )
        if root is None:
            return []
        result: list[AgentHarnessLocalExecutionPublication] = []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ScientificAgentAuthorizationVerificationError(
                "local execution publication collection exceeds its bounded roster"
            )
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ScientificAgentAuthorizationVerificationError(
                    "local execution publication collection contains an unsafe entry"
                )
            publication = self.read_harness_local_execution_publication(
                project_id=project_id,
                publication_id=child.name,
            )
            if publication.controller_execution_id == controller_execution_id:
                result.append(publication)
        return sorted(
            result,
            key=lambda item: (item.created_at, item.publication_id),
        )

    @contextmanager
    def request_session(
        self,
        *,
        project_id: str,
        proposal_id: str,
        client_request_id: str,
        operation: str,
        request_digest: str,
    ):
        clean_project = _safe_scope_id(project_id, field="project_id")
        clean_proposal = _safe_scope_id(proposal_id, field="proposal_id")
        clean_request = _safe_scope_id(client_request_id, field="client_request_id")
        clean_operation = _safe_scope_id(operation, field="operation")
        request_dir = self._request_dir(
            project_id=clean_project,
            client_request_id=clean_request,
            create=True,
        )
        if request_dir is None:  # pragma: no cover
            raise ScientificAgentAuthorizationError("request storage unavailable")
        lock_path = request_dir / "request.lock"
        if lock_path.is_symlink():
            raise ScientificAgentAuthorizationError("authorization request lock is a symbolic link")
        with _exclusive_process_lock(lock_path):
            session = AgentPlanControlRequestSession(
                project_id=clean_project,
                proposal_id=clean_proposal,
                client_request_id=clean_request,
                operation=clean_operation,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            reservation = {
                "schema_version": AUTHORIZATION_REQUEST_BINDING_VERSION,
                "status": "RESERVED",
                "project_id": clean_project,
                "proposal_id": clean_proposal,
                "client_request_id": clean_request,
                "operation": clean_operation,
                "request_digest": request_digest,
            }
            self.write_or_verify_request_file(
                request_dir / "reservation.json",
                _pretty_json_bytes(reservation),
                conflict="client request ID is bound to different authorization content",
            )
            yield session

    def write_request_marker(
        self,
        session: AgentPlanControlRequestSession,
        *,
        filename: str,
        status: str,
        values: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema_version": AUTHORIZATION_REQUEST_BINDING_VERSION,
            "status": status,
            "project_id": session.project_id,
            "proposal_id": session.proposal_id,
            "client_request_id": session.client_request_id,
            "operation": session.operation,
            "request_digest": session.request_digest,
            **dict(values),
        }
        self.write_or_verify_request_file(
            session.request_dir / filename,
            _pretty_json_bytes(payload),
            conflict=f"{status} marker differs from the immutable request",
        )

    def read_request_json(self, path: Path, *, label: str) -> dict[str, Any]:
        if path.is_symlink():
            raise ScientificAgentAuthorizationConflict(f"{label} is a symbolic link")
        try:
            loaded = json.loads(
                _read_exact_bytes(path, label=label, max_bytes=_MAX_CONTROL_BYTES)
            )
        except json.JSONDecodeError as exc:
            raise ScientificAgentAuthorizationConflict(f"{label} is invalid JSON") from exc
        if not isinstance(loaded, dict):
            raise ScientificAgentAuthorizationConflict(f"{label} must be an object")
        return loaded

    @staticmethod
    def write_or_verify_request_file(
        path: Path,
        payload: bytes,
        *,
        conflict: str,
    ) -> None:
        if path.is_symlink():
            raise ScientificAgentAuthorizationError("authorization request state is a symbolic link")
        if path.exists():
            actual = _read_exact_bytes(
                path,
                label="authorization request state",
                max_bytes=_MAX_CONTROL_BYTES,
            )
            if actual != payload:
                raise ScientificAgentAuthorizationConflict(conflict)
            return
        try:
            _write_exclusive(path, payload)
        except FileExistsError:
            actual = _read_exact_bytes(
                path,
                label="authorization request state",
                max_bytes=_MAX_CONTROL_BYTES,
            )
            if actual != payload:
                raise ScientificAgentAuthorizationConflict(conflict)

    def _publish_model(
        self,
        *,
        project_id: str,
        kind: str,
        artifact_id: str,
        model: ModelT,
        staging_parent: Path | None = None,
    ) -> ModelT:
        expected = self._publication_payloads(
            kind=kind,
            artifact_id=artifact_id,
            model=model,
        )
        root = self._collection_root(project_id=project_id, kind=kind, create=True)
        if root is None:  # pragma: no cover
            raise ScientificAgentAuthorizationError("control collection unavailable")
        target = self._safe_target(root=root, artifact_id=artifact_id)
        if target.exists() or target.is_symlink():
            self._verify_publication_bytes(target, expected=expected)
            return model

        parent = staging_parent or root
        resolved_parent = parent.resolve()
        control_root = self._control_root(project_id=project_id, create=True)
        if control_root is None or not resolved_parent.is_relative_to(control_root):
            raise ScientificAgentAuthorizationError("control staging parent escapes project scope")
        staging = resolved_parent / f"{kind}-staging-{uuid.uuid4().hex}"
        if staging.exists() or staging.is_symlink():  # pragma: no cover
            raise ScientificAgentAuthorizationConflict("control staging path already exists")
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_directory(resolved_parent)
        data_filename = _CONTROL_LAYOUT[kind][1]
        ordered_files = (data_filename, "verification.json", "publication_manifest.json")
        for index, filename in enumerate(ordered_files, start=1):
            _write_exclusive(staging / filename, expected[filename])
            self._fault(f"after_{kind}_file_{index}")
        _fsync_directory(staging)
        try:
            os.rename(staging, target)
        except OSError as exc:
            if not target.exists() or target.is_symlink():
                raise ScientificAgentAuthorizationConflict(
                    f"{kind} publication could not be atomically committed"
                ) from exc
            self._verify_publication_bytes(target, expected=expected)
        else:
            _fsync_directory(root)
        self._fault(f"after_{kind}_rename")
        self._verify_publication_bytes(target, expected=expected)
        return model

    def _read_model(
        self,
        *,
        project_id: str,
        kind: str,
        artifact_id: str,
        expected_type: type[ModelT],
    ) -> ModelT:
        root = self._collection_root(project_id=project_id, kind=kind, create=False)
        if root is None:
            raise FileNotFoundError(f"{kind} not found")
        target = self._safe_target(root=root, artifact_id=artifact_id)
        if target.is_symlink() or not target.is_dir():
            raise FileNotFoundError(f"{kind} not found")
        data_filename = _CONTROL_LAYOUT[kind][1]
        try:
            model = expected_type.model_validate_json(
                _read_exact_bytes(
                    target / data_filename,
                    label=f"{kind} artifact",
                    max_bytes=_MAX_CONTROL_BYTES,
                )
            )
        except (ValidationError, ValueError) as exc:
            raise ScientificAgentAuthorizationVerificationError(
                f"persisted {kind} failed strict validation"
            ) from exc
        if str(getattr(model, _CONTROL_ID_FIELDS[kind], "")) != artifact_id:
            raise ScientificAgentAuthorizationVerificationError(
                f"persisted {kind} identity does not match its publication directory"
            )
        expected = self._publication_payloads(
            kind=kind,
            artifact_id=artifact_id,
            model=model,
        )
        self._verify_publication_bytes(target, expected=expected)
        return model

    @staticmethod
    def _publication_payloads(
        *,
        kind: str,
        artifact_id: str,
        model: BaseModel,
    ) -> dict[str, bytes]:
        _, data_filename, digest_field, _ = _CONTROL_LAYOUT[kind]
        model_payload = model.model_dump(mode="json")
        digest = str(model_payload.get(digest_field) or "")
        data = _pretty_json_bytes(model_payload)
        verification = {
            "schema_version": CONTROL_VERIFICATION_VERSION,
            "artifact_type": kind,
            "artifact_id": artifact_id,
            "artifact_digest": digest,
            "executable": False,
            "verified": True,
        }
        payloads = {
            data_filename: data,
            "verification.json": _pretty_json_bytes(verification),
        }
        manifest = {
            "schema_version": CONTROL_PUBLICATION_MANIFEST_VERSION,
            "artifact_type": kind,
            "artifact_id": artifact_id,
            "artifact_digest": digest,
            "files": {
                filename: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for filename, payload in sorted(payloads.items())
            },
            "complete": True,
        }
        payloads["publication_manifest.json"] = _pretty_json_bytes(manifest)
        return payloads

    @staticmethod
    def _verify_publication_bytes(
        target: Path,
        *,
        expected: Mapping[str, bytes],
    ) -> None:
        if target.is_symlink() or not target.is_dir():
            raise ScientificAgentAuthorizationConflict("control publication is not a safe directory")
        try:
            names = {item.name for item in target.iterdir()}
        except OSError as exc:
            raise ScientificAgentAuthorizationConflict("control publication cannot be inspected") from exc
        if names != set(expected):
            raise ScientificAgentAuthorizationConflict("control publication has an incomplete file roster")
        for filename, expected_bytes in expected.items():
            path = target / filename
            if path.is_symlink() or not path.is_file():
                raise ScientificAgentAuthorizationConflict("control publication contains an unsafe file")
            actual = _read_exact_bytes(
                path,
                label=f"control publication {filename}",
                max_bytes=_MAX_CONTROL_BYTES,
            )
            if actual != expected_bytes:
                raise ScientificAgentAuthorizationConflict(
                    "control artifact ID is already bound to different bytes"
                )

    def _control_root(self, *, project_id: str, create: bool) -> Path | None:
        clean_project = _safe_scope_id(project_id, field="project_id")
        project_dir = _existing_project_dir(self.storage, clean_project)
        path = project_dir / CONTROL_ROOT_NAME
        if path.is_symlink():
            raise ScientificAgentAuthorizationError("control root is a symbolic link")
        if path.exists() and not path.is_dir():
            raise ScientificAgentAuthorizationError("control root is not a directory")
        if not path.exists():
            if not create:
                return None
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(project_dir)
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentAuthorizationError("control root is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(project_dir):
            raise ScientificAgentAuthorizationError("control root escapes project scope")
        return resolved

    def _collection_root(
        self,
        *,
        project_id: str,
        kind: str,
        create: bool,
    ) -> Path | None:
        collection_name = _CONTROL_LAYOUT[kind][0]
        control = self._control_root(project_id=project_id, create=create)
        if control is None:
            return None
        path = control / collection_name
        if path.is_symlink():
            raise ScientificAgentAuthorizationError("control collection is a symbolic link")
        if path.exists() and not path.is_dir():
            raise ScientificAgentAuthorizationError("control collection is not a directory")
        if not path.exists():
            if not create:
                return None
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(control)
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentAuthorizationError("control collection is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(control):
            raise ScientificAgentAuthorizationError("control collection escapes project scope")
        return resolved

    def _request_dir(
        self,
        *,
        project_id: str,
        client_request_id: str,
        create: bool,
    ) -> Path | None:
        control = self._control_root(project_id=project_id, create=create)
        if control is None:
            return None
        requests = control / "requests"
        if requests.is_symlink():
            raise ScientificAgentAuthorizationError("control request root is a symbolic link")
        if requests.exists() and not requests.is_dir():
            raise ScientificAgentAuthorizationError("control request root is not a directory")
        if not requests.exists():
            if not create:
                return None
            try:
                requests.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(control)
        clean_request = _safe_scope_id(client_request_id, field="client_request_id")
        path = requests / clean_request
        if path.is_symlink():
            raise ScientificAgentAuthorizationError("control request directory is a symbolic link")
        if path.exists() and not path.is_dir():
            raise ScientificAgentAuthorizationConflict("control request path is not a directory")
        if not path.exists():
            if not create:
                return None
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(requests)
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentAuthorizationConflict("control request directory is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(requests.resolve()):
            raise ScientificAgentAuthorizationError("control request directory escapes project scope")
        return resolved

    @staticmethod
    def _safe_target(*, root: Path, artifact_id: str) -> Path:
        clean_id = _safe_scope_id(artifact_id, field="control artifact ID")
        path = root / clean_id
        if path.is_symlink():
            raise ScientificAgentAuthorizationConflict("control target is a symbolic link")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ScientificAgentAuthorizationError("control target escapes collection scope")
        return resolved


class ScientificAgentAuthorizationService:
    """Evaluate, authorize, reverify, and create non-dispatched start intents."""

    def __init__(
        self,
        *,
        storage: Any,
        proposal_store: ScientificAgentPlanProposalStore,
        registry: AtomicTaskRegistry | None = None,
        control_store: AgentPlanControlStore | None = None,
        permission_engine: ScientificAgentPermissionEngine | None = None,
        resource_authority_resolver: Callable[[ScientificAgentPlanPublication, str], Any]
        | None = None,
        tracer: HarnessTracer | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.proposal_store = proposal_store
        self.registry = registry or proposal_store.registry
        self.control_store = control_store or AgentPlanControlStore(storage=storage)
        self.permission_engine = permission_engine or ScientificAgentPermissionEngine(
            registry=self.registry,
            resource_authority_resolver=resource_authority_resolver,
            clock=clock,
        )
        self.tracer = tracer or NoopHarnessTracer()
        self.clock = clock

    def evaluate_permission(
        self,
        *,
        project_id: str,
        proposal_id: str,
        expected_proposal_digest: str | None = None,
    ) -> AgentPermissionDecision:
        publication = self._verified_publication(project_id, proposal_id)
        proposal = publication.proposal
        correlation = build_harness_telemetry_correlation(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            semantic_plan_id=proposal.semantic_plan_id,
            semantic_plan_digest=proposal.semantic_plan_digest,
            operation="agent.permission.evaluate",
            component="permission",
            phase="evaluate",
        )
        with self.tracer.start_span(
            "permission.evaluate",
            attributes=privacy_safe_telemetry_attributes(correlation),
        ) as span:
            decision = self.permission_engine.evaluate(
                publication=publication,
                phase=AgentPermissionPhase.PROPOSAL_REVIEW,
                expected_proposal_digest=expected_proposal_digest,
            )
            committed = self.control_store.publish_permission_decision(decision)
            span.set_attribute("permission_decision_id", committed.decision_id)
            span.set_attribute("decision_digest", committed.decision_digest)
            span.add_event(
                "permission.decision",
                {"outcome": committed.outcome.value.lower()},
            )
            return committed

    def authorize(
        self,
        *,
        project_id: str,
        proposal_id: str,
        request: AgentPlanAuthorizationRequest,
        actor: str,
        actor_source: str,
    ) -> AgentPlanAuthorization:
        correlation = build_harness_telemetry_correlation(
            project_id=project_id,
            proposal_id=proposal_id,
            proposal_digest=request.expected_proposal_digest,
            operation="agent.authorization.create",
            component="authorization",
            phase="commit",
        )
        with self.tracer.start_span(
            "authorization.create",
            attributes=privacy_safe_telemetry_attributes(correlation),
        ) as span:
            result = self._commit_authority_chain(
                project_id=project_id,
                proposal_id=proposal_id,
                request=request,
                actor=actor,
                actor_source=actor_source,
                operation="authorize",
            )
            authorization = (
                result.authorization
                if isinstance(result, ApproveAndStartResult)
                else result
            )
            span.set_attribute("authorization_id", authorization.authorization_id)
            span.add_event("authorization.committed", {"outcome": "committed"})
        if isinstance(result, ApproveAndStartResult):  # pragma: no cover
            return result.authorization
        return result

    def approve_and_start(
        self,
        *,
        project_id: str,
        proposal_id: str,
        request: AgentPlanAuthorizationRequest,
        actor: str,
        actor_source: str,
    ) -> ApproveAndStartResult:
        correlation = build_harness_telemetry_correlation(
            project_id=project_id,
            proposal_id=proposal_id,
            proposal_digest=request.expected_proposal_digest,
            operation="agent.authorization.create",
            component="authorization",
            phase="approve_and_start",
        )
        with self.tracer.start_span(
            "authorization.create",
            attributes=privacy_safe_telemetry_attributes(correlation),
        ) as span:
            result = self._commit_authority_chain(
                project_id=project_id,
                proposal_id=proposal_id,
                request=request,
                actor=actor,
                actor_source=actor_source,
                operation="approve-and-start",
            )
            if isinstance(result, ApproveAndStartResult):
                span.set_attribute(
                    "authorization_id", result.authorization.authorization_id
                )
                span.add_event(
                    "authorization.committed", {"outcome": "committed"}
                )
        if not isinstance(result, ApproveAndStartResult):  # pragma: no cover
            raise ScientificAgentAuthorizationError("approve-and-start did not create an intent")
        start_correlation = build_harness_telemetry_correlation(
            project_id=project_id,
            run_id=result.start_intent.run_id,
            proposal_id=proposal_id,
            proposal_digest=request.expected_proposal_digest,
            authorization_id=result.authorization.authorization_id,
            start_intent_id=result.start_intent.start_intent_id,
            operation="agent.start_intent.create",
            component="authorization",
            phase="committed",
        )
        with self.tracer.start_span(
            "start_intent.create",
            attributes=privacy_safe_telemetry_attributes(start_correlation),
        ) as start_span:
            start_span.add_event("start_intent.committed", {"outcome": "committed"})
        return result

    def verify_authorization(
        self,
        *,
        project_id: str,
        authorization_id: str,
        verify_current: bool = True,
    ) -> AgentPlanAuthorization:
        authorization = self.control_store.read_authorization(
            project_id=project_id,
            authorization_id=authorization_id,
        )
        if verify_current:
            publication = self._verified_publication(
                project_id,
                authorization.proposal_id,
            )
        else:
            try:
                publication = self.proposal_store.read(
                    project_id=project_id,
                    proposal_id=authorization.proposal_id,
                    verify_current=False,
                )
            except ScientificAgentPlanError as exc:
                raise ScientificAgentAuthorizationVerificationError(
                    "authorization proposal publication failed exact verification"
                ) from exc
        decision = self.control_store.read_permission_decision(
            project_id=project_id,
            decision_id=authorization.permission_decision_id,
        )
        if (
            decision.phase != AgentPermissionPhase.AUTHORIZATION_CANDIDATE
            or decision.outcome == AgentPermissionOutcome.DENY
            or decision.decision_digest != authorization.permission_decision_digest
        ):
            raise ScientificAgentAuthorizationVerificationError(
                "authorization permission decision binding is invalid"
            )
        if authorization.authorization_scope_digest != (
            publication.proposal.authorization_scope_digest
        ):
            raise ScientificAgentAuthorizationVerificationError(
                "authorization scope no longer matches the verified proposal"
            )
        regenerated_decision = self.permission_engine.evaluate(
            publication=publication,
            phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
            expected_proposal_digest=authorization.proposal_digest,
            expected_authorization_scope_digest=(
                authorization.authorization_scope_digest
            ),
            authorization_mode=authorization.authorization_mode,
            requested_preauthorized_gate_ids=authorization.preauthorized_operational_gates,
            actor=authorization.actor,
            actor_source=authorization.actor_source,
            client_request_id=authorization.client_request_id,
            policy_version=decision.policy_version,
        )
        if regenerated_decision.decision_digest != decision.decision_digest:
            raise ScientificAgentAuthorizationVerificationError(
                "authorization permission decision is stale"
            )
        expected = self._build_authorization(
            publication=publication,
            request=AgentPlanAuthorizationRequest(
                expected_proposal_digest=authorization.proposal_digest,
                authorization_mode=authorization.authorization_mode,
                requested_preauthorized_gate_ids=authorization.preauthorized_operational_gates,
                confirmed=True,
                client_request_id=authorization.client_request_id,
                note=authorization.note,
            ),
            actor=authorization.actor,
            actor_source=authorization.actor_source,
            decision=decision,
            created_at=authorization.created_at,
        )
        if expected.model_dump(mode="json") != authorization.model_dump(mode="json"):
            raise ScientificAgentAuthorizationVerificationError(
                "authorization no longer exactly binds the verified proposal"
            )
        return authorization

    def verify_start_intent(
        self,
        *,
        project_id: str,
        start_intent_id: str,
        verify_current: bool = True,
    ) -> AgentPlanStartIntent:
        intent = self.control_store.read_start_intent(
            project_id=project_id,
            start_intent_id=start_intent_id,
        )
        authorization = self.verify_authorization(
            project_id=project_id,
            authorization_id=intent.authorization_id,
            verify_current=verify_current,
        )
        decision = self.control_store.read_permission_decision(
            project_id=project_id,
            decision_id=intent.permission_decision_id,
        )
        if (
            intent.proposal_id != authorization.proposal_id
            or intent.proposal_digest != authorization.proposal_digest
            or intent.authorization_digest != authorization.authorization_digest
            or decision.phase != AgentPermissionPhase.AUTHORIZED_START
            or decision.outcome != AgentPermissionOutcome.ALLOW
            or decision.decision_digest != intent.permission_decision_digest
            or decision.proposal_id != authorization.proposal_id
            or decision.proposal_digest != authorization.proposal_digest
            or decision.authorization_id != authorization.authorization_id
            or decision.authorization_digest != authorization.authorization_digest
            or decision.authorization_mode != authorization.authorization_mode
            or decision.requested_preauthorized_gate_ids
            != authorization.preauthorized_operational_gates
            or decision.actor != authorization.actor
            or decision.actor_source != authorization.actor_source
            or decision.client_request_id != intent.client_request_id
        ):
            raise ScientificAgentAuthorizationVerificationError(
                "start intent authority binding is invalid"
            )
        publication = (
            self._verified_publication(project_id, authorization.proposal_id)
            if verify_current
            else self.proposal_store.read(
                project_id=project_id,
                proposal_id=authorization.proposal_id,
                verify_current=False,
            )
        )
        regenerated = self.permission_engine.evaluate(
            publication=publication,
            phase=AgentPermissionPhase.AUTHORIZED_START,
            expected_proposal_digest=authorization.proposal_digest,
            authorization_mode=authorization.authorization_mode,
            requested_preauthorized_gate_ids=authorization.preauthorized_operational_gates,
            actor=authorization.actor,
            actor_source=authorization.actor_source,
            client_request_id=intent.client_request_id,
            authorization_id=authorization.authorization_id,
            authorization_digest=authorization.authorization_digest,
            authorization_verified=True,
            start_intent_slot_available=True,
            policy_version=decision.policy_version,
        )
        if regenerated.decision_digest != decision.decision_digest:
            raise ScientificAgentAuthorizationVerificationError(
                "start intent permission decision is stale"
            )
        expected = self._build_start_intent(
            authorization=authorization,
            permission_decision=decision,
            client_request_id=intent.client_request_id,
            created_at=intent.created_at,
        )
        if expected.model_dump(mode="json") != intent.model_dump(mode="json"):
            raise ScientificAgentAuthorizationVerificationError(
                "start intent does not match its exact immutable authority"
            )
        return intent

    def evaluate_shadow(
        self,
        *,
        project_id: str,
        proposal_id: str,
        expected_proposal_digest: str | None = None,
    ) -> AgentPermissionShadowRecord:
        publication = self._verified_publication(project_id, proposal_id)
        decision = self.permission_engine.evaluate(
            publication=publication,
            phase=AgentPermissionPhase.SHADOW_COMPARISON,
            expected_proposal_digest=expected_proposal_digest,
        )
        decision = self.control_store.publish_permission_decision(decision)
        legacy_action, legacy_outcome, legacy_reasons = derive_legacy_route_expectation(publication)
        alignment = compare_permission_outcomes(decision.outcome, legacy_outcome)
        source_digest = _agent_digest(
            {
                "proposal_digest": publication.proposal.proposal_digest,
                "observation_digest": publication.observation.observation_digest,
                "tool_catalog_digest": publication.catalog.catalog_digest,
                "legacy_action": legacy_action,
                "legacy_outcome": None if legacy_outcome is None else legacy_outcome.value,
            }
        )
        record = AgentPermissionShadowRecord(
            project_id=publication.proposal.project_id,
            run_id=publication.proposal.run_id,
            proposal_id=publication.proposal.proposal_id,
            permission_decision_id=decision.decision_id,
            new_outcome=decision.outcome,
            legacy_action=legacy_action,
            legacy_outcome=legacy_outcome,
            alignment=alignment,
            reason_codes=sorted({*decision.reason_codes, *legacy_reasons}),
            policy_digest=decision.policy_digest,
            source_digest=source_digest,
            created_at=publication.proposal.created_at,
            executable=False,
        )
        return self.control_store.publish_shadow_record(record)

    def _commit_authority_chain(
        self,
        *,
        project_id: str,
        proposal_id: str,
        request: AgentPlanAuthorizationRequest,
        actor: str,
        actor_source: str,
        operation: str,
    ) -> AgentPlanAuthorization | ApproveAndStartResult:
        request_digest = _authorization_request_digest(
            project_id=project_id,
            proposal_id=proposal_id,
            operation=operation,
            request=request,
            actor=actor,
            actor_source=actor_source,
        )
        with self.control_store.request_session(
            project_id=project_id,
            proposal_id=proposal_id,
            client_request_id=request.client_request_id,
            operation=operation,
            request_digest=request_digest,
        ) as session:
            # The first current-source read occurs only after the immutable
            # request reservation holds the cross-process request lock.
            publication = self._verified_publication(project_id, proposal_id)
            self.control_store._fault("after_initial_proposal_read")
            authorization_decision = self.permission_engine.evaluate(
                publication=publication,
                phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
                expected_proposal_digest=request.expected_proposal_digest,
                authorization_mode=request.authorization_mode,
                requested_preauthorized_gate_ids=request.requested_preauthorized_gate_ids,
                actor=actor,
                actor_source=actor_source,
                client_request_id=request.client_request_id,
            )
            authorization_decision = self.control_store.publish_permission_decision(
                authorization_decision
            )
            self.control_store._fault("after_authorization_candidate_decision")
            if authorization_decision.outcome == AgentPermissionOutcome.DENY:
                raise ScientificAgentAuthorizationDenied(authorization_decision)

            authorization = self._checkpointed_authorization(
                session=session,
                publication=publication,
                request=request,
                actor=actor,
                actor_source=actor_source,
                decision=authorization_decision,
            )
            self.control_store._fault("after_authorization_checkpoint")
            # Fail closed if any authoritative source changed after the
            # candidate decision or while the authorization was staged.
            current_publication = self._verified_publication(project_id, proposal_id)
            current_candidate = self.permission_engine.evaluate(
                publication=current_publication,
                phase=AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
                expected_proposal_digest=request.expected_proposal_digest,
                authorization_mode=request.authorization_mode,
                requested_preauthorized_gate_ids=request.requested_preauthorized_gate_ids,
                actor=actor,
                actor_source=actor_source,
                client_request_id=request.client_request_id,
            )
            if current_candidate.decision_digest != authorization_decision.decision_digest:
                raise ScientificAgentAuthorizationVerificationError(
                    "authorization candidate changed before immutable commit"
                )
            self.control_store._fault("before_authorization_commit")
            authorization = self.control_store.publish_authorization(
                authorization,
                staging_parent=session.request_dir,
            )
            self.control_store._fault("after_authorization_commit")
            verified_authorization = self.verify_authorization(
                project_id=publication.proposal.project_id,
                authorization_id=authorization.authorization_id,
                verify_current=True,
            )
            self.control_store._fault("after_authorization_verification")
            verified_authorization = self.verify_authorization(
                project_id=publication.proposal.project_id,
                authorization_id=authorization.authorization_id,
                verify_current=True,
            )
            self.control_store.write_request_marker(
                session,
                filename="authorization_committed.json",
                status="AUTHORIZATION_COMMITTED",
                values={
                    "authorization_id": authorization.authorization_id,
                    "authorization_digest": authorization.authorization_digest,
                    "permission_decision_id": authorization_decision.decision_id,
                    "permission_decision_digest": authorization_decision.decision_digest,
                },
            )
            if operation == "authorize":
                return verified_authorization

            # Durability boundary: the authorization publication and marker
            # are fsynced before this exact re-read can create a start intent.
            verified_authorization = self.verify_authorization(
                project_id=publication.proposal.project_id,
                authorization_id=authorization.authorization_id,
                verify_current=True,
            )
            current_publication = self._verified_publication(project_id, proposal_id)
            slot_available = self._start_slot_available(
                authorization=verified_authorization,
                client_request_id=request.client_request_id,
            )
            start_decision = self.permission_engine.evaluate(
                publication=current_publication,
                phase=AgentPermissionPhase.AUTHORIZED_START,
                expected_proposal_digest=request.expected_proposal_digest,
                authorization_mode=request.authorization_mode,
                requested_preauthorized_gate_ids=request.requested_preauthorized_gate_ids,
                actor=actor,
                actor_source=actor_source,
                client_request_id=request.client_request_id,
                authorization_id=verified_authorization.authorization_id,
                authorization_digest=verified_authorization.authorization_digest,
                authorization_verified=True,
                start_intent_slot_available=slot_available,
            )
            start_decision = self.control_store.publish_permission_decision(start_decision)
            if start_decision.outcome != AgentPermissionOutcome.ALLOW:
                raise ScientificAgentAuthorizationDenied(start_decision)

            start_intent = self._checkpointed_start_intent(
                session=session,
                authorization=verified_authorization,
                permission_decision=start_decision,
            )
            self.control_store._fault("after_start_intent_checkpoint")
            # Reverify the complete authorization/source binding immediately
            # before the second authority commit.
            verified_authorization = self.verify_authorization(
                project_id=publication.proposal.project_id,
                authorization_id=authorization.authorization_id,
                verify_current=True,
            )
            self.control_store._fault("before_start_intent_commit")
            start_intent = self.control_store.publish_start_intent(
                start_intent,
                staging_parent=session.request_dir,
            )
            self.control_store._fault("after_start_intent_commit")
            verified_start_intent = self.verify_start_intent(
                project_id=publication.proposal.project_id,
                start_intent_id=start_intent.start_intent_id,
                verify_current=True,
            )
            self.control_store._fault("after_start_intent_verification")
            verified_start_intent = self.verify_start_intent(
                project_id=publication.proposal.project_id,
                start_intent_id=start_intent.start_intent_id,
                verify_current=True,
            )
            self.control_store.write_request_marker(
                session,
                filename="start_intent_committed.json",
                status="START_INTENT_COMMITTED",
                values={
                    "authorization_id": verified_authorization.authorization_id,
                    "authorization_digest": verified_authorization.authorization_digest,
                    "start_intent_id": start_intent.start_intent_id,
                    "start_intent_digest": start_intent.start_intent_digest,
                    "permission_decision_id": start_decision.decision_id,
                    "permission_decision_digest": start_decision.decision_digest,
                },
            )
            return ApproveAndStartResult(
                authorization=verified_authorization,
                start_intent=verified_start_intent,
                authorization_decision=authorization_decision,
                start_decision=start_decision,
            )

    def _checkpointed_authorization(
        self,
        *,
        session: AgentPlanControlRequestSession,
        publication: ScientificAgentPlanPublication,
        request: AgentPlanAuthorizationRequest,
        actor: str,
        actor_source: str,
        decision: AgentPermissionDecision,
    ) -> AgentPlanAuthorization:
        path = session.request_dir / "authorization_checkpoint.json"
        if path.exists() or path.is_symlink():
            payload = self.control_store.read_request_json(path, label="authorization checkpoint")
            self._verify_checkpoint_identity(
                payload,
                session=session,
                schema_version=AUTHORIZATION_CHECKPOINT_VERSION,
            )
            try:
                authorization = AgentPlanAuthorization.model_validate(payload.get("authorization"))
            except ValidationError as exc:
                raise ScientificAgentAuthorizationConflict(
                    "authorization checkpoint failed strict validation"
                ) from exc
            expected = self._build_authorization(
                publication=publication,
                request=request,
                actor=actor,
                actor_source=actor_source,
                decision=decision,
                created_at=authorization.created_at,
            )
            if expected.model_dump(mode="json") != authorization.model_dump(mode="json"):
                raise ScientificAgentAuthorizationConflict(
                    "authorization checkpoint differs from the verified request"
                )
            return authorization
        authorization = self._build_authorization(
            publication=publication,
            request=request,
            actor=actor,
            actor_source=actor_source,
            decision=decision,
            created_at=self.clock(),
        )
        checkpoint = {
            "schema_version": AUTHORIZATION_CHECKPOINT_VERSION,
            "project_id": session.project_id,
            "proposal_id": session.proposal_id,
            "client_request_id": session.client_request_id,
            "operation": session.operation,
            "request_digest": session.request_digest,
            "authorization": authorization.model_dump(mode="json"),
        }
        self.control_store.write_or_verify_request_file(
            path,
            _pretty_json_bytes(checkpoint),
            conflict="authorization checkpoint differs from the immutable request",
        )
        return authorization

    def _checkpointed_start_intent(
        self,
        *,
        session: AgentPlanControlRequestSession,
        authorization: AgentPlanAuthorization,
        permission_decision: AgentPermissionDecision,
    ) -> AgentPlanStartIntent:
        path = session.request_dir / "start_intent_checkpoint.json"
        if path.exists() or path.is_symlink():
            payload = self.control_store.read_request_json(path, label="start intent checkpoint")
            self._verify_checkpoint_identity(
                payload,
                session=session,
                schema_version=START_INTENT_CHECKPOINT_VERSION,
            )
            try:
                intent = AgentPlanStartIntent.model_validate(payload.get("start_intent"))
            except ValidationError as exc:
                raise ScientificAgentAuthorizationConflict(
                    "start intent checkpoint failed strict validation"
                ) from exc
            expected = self._build_start_intent(
                authorization=authorization,
                permission_decision=permission_decision,
                client_request_id=session.client_request_id,
                created_at=intent.created_at,
            )
            if expected.model_dump(mode="json") != intent.model_dump(mode="json"):
                raise ScientificAgentAuthorizationConflict(
                    "start intent checkpoint differs from the verified authorization"
                )
            return intent
        intent = self._build_start_intent(
            authorization=authorization,
            permission_decision=permission_decision,
            client_request_id=session.client_request_id,
            created_at=self.clock(),
        )
        checkpoint = {
            "schema_version": START_INTENT_CHECKPOINT_VERSION,
            "project_id": session.project_id,
            "proposal_id": session.proposal_id,
            "client_request_id": session.client_request_id,
            "operation": session.operation,
            "request_digest": session.request_digest,
            "start_intent": intent.model_dump(mode="json"),
        }
        self.control_store.write_or_verify_request_file(
            path,
            _pretty_json_bytes(checkpoint),
            conflict="start intent checkpoint differs from the immutable request",
        )
        return intent

    @staticmethod
    def _verify_checkpoint_identity(
        payload: Mapping[str, Any],
        *,
        session: AgentPlanControlRequestSession,
        schema_version: str,
    ) -> None:
        if (
            payload.get("schema_version") != schema_version
            or payload.get("project_id") != session.project_id
            or payload.get("proposal_id") != session.proposal_id
            or payload.get("client_request_id") != session.client_request_id
            or payload.get("operation") != session.operation
            or payload.get("request_digest") != session.request_digest
        ):
            raise ScientificAgentAuthorizationConflict("control checkpoint identity mismatch")

    def _build_authorization(
        self,
        *,
        publication: ScientificAgentPlanPublication,
        request: AgentPlanAuthorizationRequest,
        actor: str,
        actor_source: str,
        decision: AgentPermissionDecision,
        created_at: str,
    ) -> AgentPlanAuthorization:
        proposal = publication.proposal
        observation = publication.observation
        artifacts_by_id = {item.artifact_id: item for item in observation.available_artifacts}
        artifact_bindings = [
            AgentAuthorizationArtifactBinding(
                artifact_id=artifact_id,
                content_digest=artifacts_by_id[artifact_id].content_digest,
                trust_class=artifacts_by_id[artifact_id].trust_class,
                verification_state=artifacts_by_id[artifact_id].verification_state,
                producer_task_id=artifacts_by_id[artifact_id].producer_task_id,
            )
            for artifact_id in proposal.selected_artifacts
        ]
        profiles_by_id = {
            item.profile_id: item for item in observation.logical_execution_profiles
        }
        profile_bindings = [
            AgentAuthorizationProfileBinding(
                profile_id=profile_id,
                profile_type=profiles_by_id[profile_id].profile_type,
                capability_digest=profiles_by_id[profile_id].capability_digest,
                availability_state="available",
                verified_capabilities=profiles_by_id[profile_id].verified_capabilities,
                supported_logical_task_types=profiles_by_id[
                    profile_id
                ].supported_logical_task_types,
            )
            for profile_id in proposal.selected_profiles
        ]
        tools_by_task = {item.task_id: item for item in publication.catalog.tools}
        gate_bindings: list[AgentAuthorizationGateBinding] = []
        for task in proposal.run_plan.tasks:
            tool = tools_by_task.get(task.task_id)
            if tool is None:
                internal = self.registry.get(task.task_id)
                effect_class = str(internal.effect_class or "")
                required_gates = list(internal.gates)
                supports_plan_preapproval = internal.supports_plan_preapproval
            else:
                effect_class = tool.effect_class
                required_gates = list(tool.required_gates)
                supports_plan_preapproval = tool.supports_plan_preapproval
            gate_class = (
                "semantic"
                if effect_class
                in {"scientific_confirm", "change_objective", "publish_or_promote"}
                else "operational"
            )
            gate_bindings.extend(
                AgentAuthorizationGateBinding(
                    task_id=task.task_id,
                    gate_id=gate_id,
                    effect_class=effect_class,
                    gate_class=gate_class,
                    supports_plan_preapproval=supports_plan_preapproval,
                )
                for gate_id in required_gates
            )
        preauthorized = sorted(request.requested_preauthorized_gate_ids)
        pending = sorted(set(proposal.required_gates).difference(preauthorized))
        return AgentPlanAuthorization(
            schema_version=(
                AGENT_PLAN_AUTHORIZATION_V2
                if proposal.schema_version == AGENT_EXECUTION_PLAN_PROPOSAL_V2
                else AGENT_PLAN_AUTHORIZATION_V1
            ),
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            semantic_plan_id=proposal.semantic_plan_id,
            semantic_plan_digest=proposal.semantic_plan_digest,
            observation_id=proposal.observation_id,
            observation_digest=proposal.observation_digest,
            tool_catalog_digest=proposal.tool_catalog_digest,
            run_plan_digest=_agent_digest(proposal.run_plan.model_dump(mode="json")),
            authorization_scope_digest=proposal.authorization_scope_digest,
            run_plan=proposal.run_plan,
            task_ids=[item.task_id for item in proposal.run_plan.tasks],
            task_authority_digests={
                item.task_id: item.task_authority_digest
                for item in decision.task_decisions
            },
            effective_planner_options=proposal.effective_planner_options,
            compiled_task_options=proposal.compiled_task_options,
            dispatch_intents=proposal.dispatch_intents,
            artifact_bindings=artifact_bindings,
            profile_bindings=profile_bindings,
            limits=proposal.limits,
            stop_conditions=proposal.stop_conditions,
            success_criteria=proposal.success_criteria,
            required_gates=proposal.required_gates,
            gate_bindings=gate_bindings,
            preauthorized_operational_gates=preauthorized,
            pending_gates=pending,
            permission_policy_version=decision.policy_version,
            permission_policy_digest=decision.policy_digest,
            permission_decision_id=decision.decision_id,
            permission_decision_digest=decision.decision_digest,
            authorization_mode=request.authorization_mode,
            actor=actor,
            actor_source=actor_source,
            note=request.note,
            client_request_id=request.client_request_id,
            created_at=created_at,
            executable=False,
        )

    @staticmethod
    def _build_start_intent(
        *,
        authorization: AgentPlanAuthorization,
        permission_decision: AgentPermissionDecision,
        client_request_id: str,
        created_at: str,
    ) -> AgentPlanStartIntent:
        return AgentPlanStartIntent(
            project_id=authorization.project_id,
            run_id=authorization.run_id,
            proposal_id=authorization.proposal_id,
            proposal_digest=authorization.proposal_digest,
            authorization_id=authorization.authorization_id,
            authorization_digest=authorization.authorization_digest,
            permission_decision_id=permission_decision.decision_id,
            permission_decision_digest=permission_decision.decision_digest,
            authorization_mode=authorization.authorization_mode,
            requested_by=authorization.actor,
            requested_by_source=authorization.actor_source,
            client_request_id=client_request_id,
            created_at=created_at,
            executable=False,
        )

    def _start_slot_available(
        self,
        *,
        authorization: AgentPlanAuthorization,
        client_request_id: str,
    ) -> bool:
        start_intent_id = _start_intent_slot_id(
            project_id=authorization.project_id,
            proposal_id=authorization.proposal_id,
            proposal_digest=authorization.proposal_digest,
        )
        try:
            existing = self.control_store.read_start_intent(
                project_id=authorization.project_id,
                start_intent_id=start_intent_id,
            )
        except FileNotFoundError:
            return True
        return bool(
            existing.authorization_id == authorization.authorization_id
            and existing.authorization_digest == authorization.authorization_digest
            and existing.client_request_id == client_request_id
        )

    def _verified_publication(
        self, project_id: str, proposal_id: str
    ) -> ScientificAgentPlanPublication:
        try:
            return self.proposal_store.read(
                project_id=project_id,
                proposal_id=proposal_id,
                verify_current=True,
            )
        except ScientificAgentPlanSourceChanged:
            raise
        except ScientificAgentPlanPublicationConflict as exc:
            raise ScientificAgentAuthorizationVerificationError(
                "proposal publication bytes failed no-replace verification"
            ) from exc
        except ScientificAgentPlanError as exc:
            raise ScientificAgentAuthorizationVerificationError(
                "proposal publication or current source failed verification"
            ) from exc


__all__ = [
    "CONTROL_ROOT_NAME",
    "AGENT_PERMISSION_SHADOW_OBSERVATION_FLAG",
    "ScientificAgentAuthorizationError",
    "ScientificAgentAuthorizationConflict",
    "ScientificAgentAuthorizationDenied",
    "ScientificAgentAuthorizationVerificationError",
    "AgentPlanControlRequestSession",
    "ApproveAndStartResult",
    "AgentPlanControlStore",
    "ScientificAgentAuthorizationService",
]
