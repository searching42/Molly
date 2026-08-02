"""Crash-safe immutable storage for bounded Execution Agent proposals."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from ai4s_agent.schemas import (
    AgentExecutionAgentObservation,
    AgentExecutionToolCatalog,
    AgentToolCallApplicationReceipt,
    AgentToolCallProposal,
)
from ai4s_agent.scientific_agent_plan import (
    _exclusive_process_lock,
    _existing_project_dir,
    _fsync_directory,
    _pretty_json_bytes,
    _read_exact_bytes,
    _safe_scope_id,
    _write_exclusive,
)


EXECUTION_AGENT_REQUEST_VERSION = "execution_agent_request_checkpoint.v1"
EXECUTION_AGENT_PUBLICATION_MANIFEST_VERSION = (
    "execution_agent_publication_manifest.v1"
)
EXECUTION_AGENT_VERIFICATION_VERSION = "execution_agent_publication_verification.v1"
_MAX_EXECUTION_AGENT_BYTES = 16 * 1024 * 1024


class ExecutionAgentStoreError(ValueError):
    """Base privacy-safe storage failure."""


class ExecutionAgentStoreConflict(ExecutionAgentStoreError):
    """A no-replace identity is already bound to other bytes."""


class ExecutionAgentStoreVerificationError(ExecutionAgentStoreError):
    """Persisted bytes failed exact strict verification."""


class ExecutionAgentStoreRecoveryRequired(ExecutionAgentStoreError):
    """A request stopped at an ambiguous external-call boundary."""

    def __init__(self, state: str) -> None:
        super().__init__(state)
        self.state = state


@dataclass(frozen=True)
class ExecutionAgentRequestSession:
    project_id: str
    controller_execution_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class ExecutionAgentApplicationSession:
    project_id: str
    tool_call_proposal_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class ExecutionAgentProposalPublication:
    observation: AgentExecutionAgentObservation
    tool_catalog: AgentExecutionToolCatalog
    proposal: AgentToolCallProposal


class ExecutionAgentStore:
    """Project-scoped request checkpoints and manifest-last publications."""

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

    @contextmanager
    def proposal_request_session(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        client_request_id: str,
        request_digest: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        execution_id = _safe_scope_id(
            controller_execution_id,
            field="controller_execution_id",
        )
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        request_dir = self._nested_request_dir(
            project_id=project,
            root_name="agent_execution_agent_requests",
            scope_id=execution_id,
            client_request_id=request_id,
            create=True,
        )
        if request_dir is None:  # pragma: no cover
            raise ExecutionAgentStoreError("execution agent request storage unavailable")
        lock = request_dir / "request.lock"
        if lock.is_symlink():
            raise ExecutionAgentStoreError("execution agent request lock is unsafe")
        with _exclusive_process_lock(lock):
            session = ExecutionAgentRequestSession(
                project_id=project,
                controller_execution_id=execution_id,
                client_request_id=request_id,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            self.write_marker(
                session,
                filename="reservation.json",
                status="RESERVED",
                values={},
            )
            yield session

    @contextmanager
    def application_session(
        self,
        *,
        project_id: str,
        tool_call_proposal_id: str,
        client_request_id: str,
        request_digest: str,
    ):
        project = _safe_scope_id(project_id, field="project_id")
        proposal_id = _safe_scope_id(
            tool_call_proposal_id,
            field="tool_call_proposal_id",
        )
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        application_root = self._nested_scope_root(
            project_id=project,
            root_name="agent_execution_agent_applications",
            scope_id=proposal_id,
            create=True,
        )
        if application_root is None:  # pragma: no cover
            raise ExecutionAgentStoreError("application storage unavailable")
        lock = application_root / "application.lock"
        if lock.is_symlink():
            raise ExecutionAgentStoreError("application lock is unsafe")
        # This proposal-level lock is deliberately acquired before Controller
        # execution/request locks in the service layer.
        with _exclusive_process_lock(lock):
            requests = self._directory(application_root, "requests")
            request_dir = self._directory(requests, request_id)
            session = ExecutionAgentApplicationSession(
                project_id=project,
                tool_call_proposal_id=proposal_id,
                client_request_id=request_id,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            self.write_marker(
                session,
                filename="reservation.json",
                status="RESERVED",
                values={},
            )
            yield session

    def write_marker(
        self,
        session: ExecutionAgentRequestSession | ExecutionAgentApplicationSession,
        *,
        filename: str,
        status: str,
        values: Mapping[str, Any],
    ) -> None:
        scope_key = (
            "controller_execution_id"
            if isinstance(session, ExecutionAgentRequestSession)
            else "tool_call_proposal_id"
        )
        scope_value = (
            session.controller_execution_id
            if isinstance(session, ExecutionAgentRequestSession)
            else session.tool_call_proposal_id
        )
        payload = {
            "schema_version": EXECUTION_AGENT_REQUEST_VERSION,
            "status": status,
            "project_id": session.project_id,
            scope_key: scope_value,
            "client_request_id": session.client_request_id,
            "request_digest": session.request_digest,
            **dict(values),
        }
        self.write_or_verify(
            session.request_dir / filename,
            _pretty_json_bytes(payload),
        )

    @staticmethod
    def read_marker(path: Path) -> dict[str, Any] | None:
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            raise ExecutionAgentStoreConflict("execution agent checkpoint is unsafe")
        try:
            payload = json.loads(
                _read_exact_bytes(
                    path,
                    label="execution agent checkpoint",
                    max_bytes=_MAX_EXECUTION_AGENT_BYTES,
                )
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ExecutionAgentStoreConflict(
                "execution agent checkpoint failed exact verification"
            ) from exc
        if not isinstance(payload, dict):
            raise ExecutionAgentStoreConflict("execution agent checkpoint is invalid")
        return payload

    @staticmethod
    def write_or_verify(path: Path, payload: bytes) -> None:
        if path.is_symlink():
            raise ExecutionAgentStoreConflict("execution agent checkpoint is unsafe")
        if path.exists():
            actual = _read_exact_bytes(
                path,
                label="execution agent checkpoint",
                max_bytes=_MAX_EXECUTION_AGENT_BYTES,
            )
            if actual != payload:
                raise ExecutionAgentStoreConflict(
                    "execution agent request ID is bound to different content"
                )
            return
        try:
            _write_exclusive(path, payload)
        except FileExistsError:
            actual = _read_exact_bytes(
                path,
                label="execution agent checkpoint",
                max_bytes=_MAX_EXECUTION_AGENT_BYTES,
            )
            if actual != payload:
                raise ExecutionAgentStoreConflict(
                    "execution agent request ID is bound to different content"
                )

    def publish_proposal(
        self,
        *,
        publication: ExecutionAgentProposalPublication,
        staging_parent: Path,
    ) -> ExecutionAgentProposalPublication:
        expected = self._proposal_payloads(publication)
        self._publish_directory(
            project_id=publication.proposal.project_id,
            root_name="agent_execution_agent_proposals",
            artifact_id=publication.proposal.tool_call_proposal_id,
            expected=expected,
            staging_parent=staging_parent,
            fault_prefix="execution_agent_proposal",
        )
        return publication

    def read_proposal(
        self,
        *,
        project_id: str,
        tool_call_proposal_id: str,
    ) -> ExecutionAgentProposalPublication:
        target = self._publication_target(
            project_id=project_id,
            root_name="agent_execution_agent_proposals",
            artifact_id=tool_call_proposal_id,
            create_root=False,
        )
        if target is None or target.is_symlink() or not target.is_dir():
            raise FileNotFoundError("execution agent proposal not found")
        payloads = self._read_publication_files(target)
        try:
            observation = AgentExecutionAgentObservation.model_validate_json(
                payloads["observation.json"]
            )
            catalog = AgentExecutionToolCatalog.model_validate_json(
                payloads["tool_catalog.json"]
            )
            proposal = AgentToolCallProposal.model_validate_json(
                payloads["tool_call_proposal.json"]
            )
        except (KeyError, ValidationError, ValueError) as exc:
            raise ExecutionAgentStoreVerificationError(
                "execution agent proposal failed strict validation"
            ) from exc
        publication = ExecutionAgentProposalPublication(observation, catalog, proposal)
        selected = next(
            (item for item in catalog.tools if item.tool_id == proposal.selected_tool_id),
            None,
        )
        if (
            proposal.project_id != _safe_scope_id(project_id, field="project_id")
            or proposal.tool_call_proposal_id
            != _safe_scope_id(tool_call_proposal_id, field="tool_call_proposal_id")
            or proposal.observation_id != observation.observation_id
            or proposal.observation_digest != observation.observation_digest
            or proposal.tool_catalog_id != catalog.tool_catalog_id
            or proposal.tool_catalog_digest != catalog.tool_catalog_digest
            or observation.tool_catalog_id != catalog.tool_catalog_id
            or observation.tool_catalog_digest != catalog.tool_catalog_digest
            or selected is None
            or proposal.current_task_id != observation.current_task_id
            or proposal.current_task_index != observation.current_task_index
            or proposal.current_attempt_ordinal != observation.current_attempt_ordinal
            or proposal.current_slot_id != observation.current_slot_id
            or proposal.next_controller_action != observation.next_controller_action
            or proposal.controller_action_boundary_class
            != observation.controller_action_boundary_class
            or proposal.server_compiled_operation != selected.server_compiled_operation
            or proposal.application_eligible != selected.application_eligible
            or proposal.user_boundary_kind != selected.user_boundary_kind
            or proposal.execution_agent_policy_version
            != observation.execution_agent_policy_version
            or proposal.execution_agent_policy_digest
            != observation.execution_agent_policy_digest
        ):
            raise ExecutionAgentStoreVerificationError(
                "execution agent proposal authority binding mismatch"
            )
        expected = self._proposal_payloads(publication)
        self._verify_publication_bytes(target, expected)
        return publication

    def publish_application_receipt(
        self,
        *,
        project_id: str,
        receipt: AgentToolCallApplicationReceipt,
        staging_parent: Path,
    ) -> AgentToolCallApplicationReceipt:
        expected = self._receipt_payloads(receipt)
        self._publish_directory(
            project_id=project_id,
            root_name="agent_execution_agent_application_receipts",
            artifact_id=receipt.application_receipt_id,
            expected=expected,
            staging_parent=staging_parent,
            fault_prefix="execution_agent_application_receipt",
        )
        return receipt

    def read_application_receipt(
        self,
        *,
        project_id: str,
        application_receipt_id: str,
    ) -> AgentToolCallApplicationReceipt:
        target = self._publication_target(
            project_id=project_id,
            root_name="agent_execution_agent_application_receipts",
            artifact_id=application_receipt_id,
            create_root=False,
        )
        if target is None or target.is_symlink() or not target.is_dir():
            raise FileNotFoundError("application receipt not found")
        payloads = self._read_publication_files(target)
        try:
            receipt = AgentToolCallApplicationReceipt.model_validate_json(
                payloads["application_receipt.json"]
            )
        except (KeyError, ValidationError, ValueError) as exc:
            raise ExecutionAgentStoreVerificationError(
                "application receipt failed strict validation"
            ) from exc
        if receipt.application_receipt_id != _safe_scope_id(
            application_receipt_id,
            field="application_receipt_id",
        ):
            raise ExecutionAgentStoreVerificationError(
                "application receipt identity mismatch"
            )
        self._verify_publication_bytes(target, self._receipt_payloads(receipt))
        return receipt

    def application_receipts_for_proposal(
        self,
        *,
        project_id: str,
        tool_call_proposal_id: str,
    ) -> list[AgentToolCallApplicationReceipt]:
        root = self._root(
            project_id=project_id,
            name="agent_execution_agent_application_receipts",
            create=False,
        )
        if root is None:
            return []
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ExecutionAgentStoreVerificationError(
                "application receipt roster exceeds its bounded limit"
            )
        result: list[AgentToolCallApplicationReceipt] = []
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ExecutionAgentStoreVerificationError(
                    "application receipt roster contains an unsafe entry"
                )
            receipt = self.read_application_receipt(
                project_id=project_id,
                application_receipt_id=child.name,
            )
            if receipt.tool_call_proposal_id == tool_call_proposal_id:
                result.append(receipt)
        return sorted(result, key=lambda item: item.application_receipt_id)

    def _proposal_payloads(
        self,
        publication: ExecutionAgentProposalPublication,
    ) -> dict[str, bytes]:
        proposal = publication.proposal
        payloads = {
            "observation.json": _pretty_json_bytes(
                publication.observation.model_dump(mode="json")
            ),
            "tool_catalog.json": _pretty_json_bytes(
                publication.tool_catalog.model_dump(mode="json")
            ),
            "tool_call_proposal.json": _pretty_json_bytes(
                proposal.model_dump(mode="json")
            ),
        }
        return self._with_verification(
            payloads=payloads,
            artifact_type="tool_call_proposal",
            artifact_id=proposal.tool_call_proposal_id,
            artifact_digest=proposal.tool_call_proposal_digest,
        )

    def _receipt_payloads(
        self,
        receipt: AgentToolCallApplicationReceipt,
    ) -> dict[str, bytes]:
        return self._with_verification(
            payloads={
                "application_receipt.json": _pretty_json_bytes(
                    receipt.model_dump(mode="json")
                )
            },
            artifact_type="tool_call_application_receipt",
            artifact_id=receipt.application_receipt_id,
            artifact_digest=receipt.application_receipt_digest,
        )

    @staticmethod
    def _with_verification(
        *,
        payloads: dict[str, bytes],
        artifact_type: str,
        artifact_id: str,
        artifact_digest: str,
    ) -> dict[str, bytes]:
        result = dict(payloads)
        result["verification.json"] = _pretty_json_bytes(
            {
                "schema_version": EXECUTION_AGENT_VERIFICATION_VERSION,
                "artifact_type": artifact_type,
                "artifact_id": artifact_id,
                "artifact_digest": artifact_digest,
                "executable": False,
                "verified": True,
            }
        )
        manifest = {
            "schema_version": EXECUTION_AGENT_PUBLICATION_MANIFEST_VERSION,
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "artifact_digest": artifact_digest,
            "files": {
                filename: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for filename, payload in sorted(result.items())
            },
            "complete": True,
        }
        result["publication_manifest.json"] = _pretty_json_bytes(manifest)
        return result

    def _publish_directory(
        self,
        *,
        project_id: str,
        root_name: str,
        artifact_id: str,
        expected: Mapping[str, bytes],
        staging_parent: Path,
        fault_prefix: str,
    ) -> None:
        root = self._root(project_id=project_id, name=root_name, create=True)
        if root is None:  # pragma: no cover
            raise ExecutionAgentStoreError("execution agent publication root unavailable")
        target = self._safe_target(root, artifact_id)
        if target.exists() or target.is_symlink():
            self._verify_publication_bytes(target, expected)
            return
        parent = staging_parent.resolve()
        project_dir = _existing_project_dir(self.storage, project_id)
        if not parent.is_relative_to(project_dir):
            raise ExecutionAgentStoreError("execution agent staging escapes project scope")
        staging = parent / f"{fault_prefix}-staging-{uuid.uuid4().hex}"
        if staging.exists() or staging.is_symlink():  # pragma: no cover
            raise ExecutionAgentStoreConflict("execution agent staging path already exists")
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_directory(parent)
        ordered = [name for name in sorted(expected) if name != "publication_manifest.json"]
        ordered.append("publication_manifest.json")
        for index, filename in enumerate(ordered, start=1):
            _write_exclusive(staging / filename, expected[filename])
            self._fault(f"after_{fault_prefix}_file_{index}")
        _fsync_directory(staging)
        try:
            os.rename(staging, target)
        except OSError as exc:
            if not target.exists() or target.is_symlink():
                raise ExecutionAgentStoreConflict(
                    "execution agent publication could not be atomically committed"
                ) from exc
            self._verify_publication_bytes(target, expected)
        else:
            _fsync_directory(root)
        self._fault(f"after_{fault_prefix}_rename")
        self._verify_publication_bytes(target, expected)

    @staticmethod
    def _verify_publication_bytes(
        target: Path,
        expected: Mapping[str, bytes],
    ) -> None:
        if target.is_symlink() or not target.is_dir():
            raise ExecutionAgentStoreConflict("execution agent publication is unsafe")
        try:
            names = {item.name for item in target.iterdir()}
        except OSError as exc:
            raise ExecutionAgentStoreConflict(
                "execution agent publication cannot be inspected"
            ) from exc
        if names != set(expected):
            raise ExecutionAgentStoreConflict(
                "execution agent publication file roster is incomplete"
            )
        for filename, expected_bytes in expected.items():
            path = target / filename
            if path.is_symlink() or not path.is_file():
                raise ExecutionAgentStoreConflict(
                    "execution agent publication contains an unsafe file"
                )
            actual = _read_exact_bytes(
                path,
                label=f"execution agent publication {filename}",
                max_bytes=_MAX_EXECUTION_AGENT_BYTES,
            )
            if actual != expected_bytes:
                raise ExecutionAgentStoreConflict(
                    "execution agent artifact ID is bound to different bytes"
                )

    @staticmethod
    def _read_publication_files(target: Path) -> dict[str, bytes]:
        if target.is_symlink() or not target.is_dir():
            raise ExecutionAgentStoreVerificationError(
                "execution agent publication is unsafe"
            )
        result: dict[str, bytes] = {}
        for child in target.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ExecutionAgentStoreVerificationError(
                    "execution agent publication contains an unsafe entry"
                )
            result[child.name] = _read_exact_bytes(
                child,
                label="execution agent publication",
                max_bytes=_MAX_EXECUTION_AGENT_BYTES,
            )
        return result

    def _publication_target(
        self,
        *,
        project_id: str,
        root_name: str,
        artifact_id: str,
        create_root: bool,
    ) -> Path | None:
        root = self._root(project_id=project_id, name=root_name, create=create_root)
        if root is None:
            return None
        return self._safe_target(root, artifact_id)

    def _nested_request_dir(
        self,
        *,
        project_id: str,
        root_name: str,
        scope_id: str,
        client_request_id: str,
        create: bool,
    ) -> Path | None:
        scope = self._nested_scope_root(
            project_id=project_id,
            root_name=root_name,
            scope_id=scope_id,
            create=create,
        )
        if scope is None:
            return None
        requests = self._directory(scope, "requests") if create else scope / "requests"
        if not requests.exists():
            return None
        return (
            self._directory(requests, client_request_id)
            if create
            else self._existing_directory(requests, client_request_id)
        )

    def _nested_scope_root(
        self,
        *,
        project_id: str,
        root_name: str,
        scope_id: str,
        create: bool,
    ) -> Path | None:
        root = self._root(project_id=project_id, name=root_name, create=create)
        if root is None:
            return None
        return self._directory(root, scope_id) if create else self._existing_directory(root, scope_id)

    def _root(self, *, project_id: str, name: str, create: bool) -> Path | None:
        project = _existing_project_dir(
            self.storage,
            _safe_scope_id(project_id, field="project_id"),
        )
        path = project / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ExecutionAgentStoreError("execution agent root is unsafe")
        if not path.exists():
            if not create:
                return None
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(project)
        if path.is_symlink() or not path.is_dir():
            raise ExecutionAgentStoreError("execution agent root is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(project):
            raise ExecutionAgentStoreError("execution agent root escapes project scope")
        return resolved

    @staticmethod
    def _safe_target(root: Path, artifact_id: str) -> Path:
        clean = _safe_scope_id(artifact_id, field="execution agent artifact ID")
        path = root / clean
        if path.is_symlink():
            raise ExecutionAgentStoreConflict("execution agent target is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ExecutionAgentStoreError("execution agent target escapes collection scope")
        return resolved

    @staticmethod
    def _directory(parent: Path, name: str) -> Path:
        clean = _safe_scope_id(name, field="execution agent directory")
        path = parent / clean
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ExecutionAgentStoreError("execution agent directory is unsafe")
        if not path.exists():
            try:
                path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(parent)
        if path.is_symlink() or not path.is_dir():
            raise ExecutionAgentStoreError("execution agent directory is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(parent.resolve()):
            raise ExecutionAgentStoreError("execution agent directory escapes scope")
        return resolved

    @staticmethod
    def _existing_directory(parent: Path, name: str) -> Path | None:
        clean = _safe_scope_id(name, field="execution agent directory")
        path = parent / clean
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_dir():
            raise ExecutionAgentStoreError("execution agent directory is unsafe")
        resolved = path.resolve()
        if not resolved.is_relative_to(parent.resolve()):
            raise ExecutionAgentStoreError("execution agent directory escapes scope")
        return resolved


__all__ = [
    "ExecutionAgentApplicationSession",
    "ExecutionAgentProposalPublication",
    "ExecutionAgentRequestSession",
    "ExecutionAgentStore",
    "ExecutionAgentStoreConflict",
    "ExecutionAgentStoreError",
    "ExecutionAgentStoreRecoveryRequired",
    "ExecutionAgentStoreVerificationError",
]
