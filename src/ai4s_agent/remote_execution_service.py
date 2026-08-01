from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.remote_execution_storage import PinnedExecutionTree
from ai4s_agent.remote_output_contracts import (
    verify_remote_output_contents,
    verify_remote_output_contract,
)
from ai4s_agent.resource_profiles import (
    CapabilityProbeService,
    ConnectionProfile,
    ExecutionProfile,
    ResourceProfileStore,
    TransferManifest,
    verify_transfer_manifest_binding,
)
from ai4s_agent.schemas import (
    AgentHarnessRemoteExecutionSlotBinding,
    ArtifactRef,
    RunStatus,
    StageHistoryItem,
    StageState,
    _agent_digest,
)
from ai4s_agent.storage import ProjectStorage


class DescriptorRemoteExecutionLifecycleService:
    """Remote lifecycle whose local authority IO is pinned to directory fds."""

    _SLOT_AUTHORITY_FIELDS = frozenset(
        {
            "controller_execution_id",
            "controller_execution_digest",
            "planned_task_index",
            "attempt",
            "task_authority_digest",
            "dispatch_intent_digest",
            "compiled_options_digest",
            "input_artifacts_digest",
            "output_contract_digest",
            "remote_authority_id",
            "remote_authority_digest",
            "remote_authority_set_id",
            "remote_authority_set_digest",
        }
    )

    def __init__(
        self,
        *,
        projects: ProjectStorage,
        profiles: ResourceProfileStore,
        transport: Any | None = None,
        capability_probe: Any | None = None,
    ) -> None:
        from ai4s_agent.remote_execution_lifecycle import PinnedWorkerTransport

        self.projects = projects
        self.profiles = profiles
        self.transport = transport or PinnedWorkerTransport()
        self.capability_probe = capability_probe or CapabilityProbeService(store=profiles)
        self._lock = threading.RLock()

    @staticmethod
    def _types():
        import ai4s_agent.remote_execution_lifecycle as lifecycle

        return lifecycle

    @contextmanager
    def _tree(
        self,
        project_id: str,
        run_id: str,
        *,
        create: bool,
        slot_id: str | None = None,
    ):
        lifecycle = self._types()
        project = lifecycle._identifier(project_id, "project_id")
        run = lifecycle._identifier(run_id, "run_id")
        with PinnedExecutionTree.open(
            projects_root=self.projects.projects_root,
            project_id=project,
            run_id=run,
            create_remote=create,
            slot_id=slot_id,
        ) as tree:
            yield tree

    def prepare(
        self,
        *,
        project_id: str,
        run_id: str,
        task_id: str,
        transfer_manifest: TransferManifest | Mapping[str, Any],
        requested_resources: Mapping[str, Any] | Any,
        input_artifacts: Mapping[str, str],
        slot_id: str | None = None,
        slot_binding_authority: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        lifecycle = self._types()
        manifest = TransferManifest.model_validate(transfer_manifest)
        connection = self.profiles.get_connection(manifest.connection_id)
        profile = self.profiles.resolve_execution_profile(manifest.execution_profile_id)
        candidate = lifecycle.build_remote_execution_request(
            project_id=project_id,
            run_id=run_id,
            task_id=task_id,
            transfer_manifest=manifest,
            connection=connection,
            execution_profile=profile,
            requested_resources=requested_resources,
        )
        with self._lock, self._tree(
            project_id, run_id, create=True, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            lifecycle._local_io_boundary("prepare.tree_pinned")
            tree.assert_named_identity()
            if tree.exists("remote", "execution_request.json"):
                request = self._read_request(tree, project_id, run_id)
                if not self._same_preparation(request, candidate):
                    raise ValueError("remote execution run already has a different request")
                self._publish_or_verify_slot_binding(
                    tree,
                    request,
                    slot_binding_authority,
                )
                if not tree.exists("remote", "approval.json") and not tree.exists("remote", "publication.json"):
                    self._repair_waiting_authority(tree, request)
                return self._inspect_tree(tree, request)
            if tree.exists("remote", "approval.json") or tree.exists("remote", "publication.json"):
                raise ValueError("remote execution authority exists without its request")
            self._stage_registered_inputs(tree, candidate, input_artifacts)
            lifecycle._commit_boundary("prepare.inputs")
            tree.assert_named_identity()
            tree.publish_immutable_json("remote", "execution_request.json", candidate.model_dump(mode="json"))
            lifecycle._commit_boundary("prepare.request")
            tree.assert_named_identity()
            self._publish_or_verify_slot_binding(
                tree,
                candidate,
                slot_binding_authority,
            )
            lifecycle._commit_boundary("prepare.slot_binding")
            tree.assert_named_identity()
            self._write_stage(tree, candidate, RunStatus.WAITING_USER, "remote_execution_approval")
            lifecycle._commit_boundary("prepare.stage")
            tree.assert_named_identity()
            self._write_state(tree, candidate, status="WAITING_APPROVAL")
            lifecycle._commit_boundary("prepare.telemetry")
            tree.assert_named_identity()
            return self._inspect_tree(tree, candidate)

    def approve(
        self,
        *,
        project_id: str,
        run_id: str,
        request_sha256: str,
        actor: str,
        note: str = "",
        slot_id: str | None = None,
        expected_slot_binding_digest: str | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            if request.request_sha256 != request_sha256:
                raise ValueError("approval does not bind the exact execution request")
            self._record_approval_tree(
                tree,
                request,
                request_sha256=request_sha256,
                actor=actor,
                note=note,
            )
            return self._dispatch_tree(tree, request)

    def record_approval(
        self,
        *,
        project_id: str,
        run_id: str,
        request_sha256: str,
        actor: str,
        note: str = "",
        slot_id: str,
        expected_slot_binding_digest: str,
    ) -> dict[str, Any]:
        """Commit approval for one exact task slot without dispatching it."""

        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            if request.request_sha256 != request_sha256:
                raise ValueError("approval does not bind the exact execution request")
            self._record_approval_tree(
                tree,
                request,
                request_sha256=request_sha256,
                actor=actor,
                note=note,
            )
            return self._inspect_tree(tree, request)

    def dispatch(
        self,
        *,
        project_id: str,
        run_id: str,
        request_sha256: str,
        slot_id: str,
        expected_slot_binding_digest: str,
    ) -> dict[str, Any]:
        """Dispatch one already-approved exact task slot at most once."""

        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            if request.request_sha256 != request_sha256:
                raise ValueError("dispatch does not bind the exact execution request")
            return self._dispatch_tree(tree, request)

    def _record_approval_tree(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        *,
        request_sha256: str,
        actor: str,
        note: str,
    ) -> Any:
        lifecycle = self._types()
        lifecycle._local_io_boundary("approval.tree_pinned")
        if tree.exists("remote", "publication.json"):
            self._recover_success(tree, request)
            return self._read_approval(tree, request)
        if tree.exists("remote", "approval.json"):
            return self._read_approval(tree, request)
        approval = lifecycle.build_remote_execution_approval(
            request,
            request_sha256=request_sha256,
            actor=actor,
            note=note,
        )
        connection, _ = self._verify_current_profiles(request)
        self._run_submission_preflight(request, connection)
        self._verify_staged_inputs(tree, request)
        lifecycle._local_io_boundary("approval.before_record")
        tree.assert_named_identity()
        tree.publish_immutable_json(
            "remote",
            "approval.json",
            approval.model_dump(mode="json"),
        )
        lifecycle._commit_boundary("approval.record")
        tree.assert_named_identity()
        self._write_stage(
            tree,
            request,
            RunStatus.WAITING_USER,
            "remote_execution_dispatch",
        )
        self._write_state(tree, request, status="APPROVED")
        return approval

    def _dispatch_tree(
        self,
        tree: PinnedExecutionTree,
        request: Any,
    ) -> dict[str, Any]:
        lifecycle = self._types()
        if tree.exists("remote", "publication.json"):
            self._recover_success(tree, request)
            return self._inspect_tree(tree, request)
        approval = self._read_approval(tree, request)
        stage = self._read_stage(tree)
        if stage is not None and stage.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return self._inspect_tree(tree, request)
        state = self._read_state(tree, request, required=False)
        if state is not None and state["status"] in {
            "ACCEPTED",
            "RUNNING",
            "CANCEL_REQUESTED",
            "RECOVERY_REQUIRED",
        }:
            return self._inspect_tree(tree, request)
        if state is None or state["status"] == "WAITING_APPROVAL":
            self._write_state(tree, request, status="APPROVED")
        connection, _ = self._verify_current_profiles(request)
        self._run_submission_preflight(request, connection)
        self._verify_staged_inputs(tree, request)
        tree.assert_named_identity()
        try:
            observation = self.transport.dispatch(
                connection=connection,
                request=request,
                approval=approval,
                tree=tree,
            )
            self._validate_observation(request, observation)
        except lifecycle.RemoteTransportError:
            self._mark_recovery(tree, request, "dispatch_outcome_unknown")
            return self._inspect_tree(tree, request)
        self._apply_observation(tree, request, approval, observation)
        return self._inspect_tree(tree, request)

    def refresh(
        self,
        *,
        project_id: str,
        run_id: str,
        slot_id: str | None = None,
        expected_slot_binding_digest: str | None = None,
    ) -> dict[str, Any]:
        lifecycle = self._types()
        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            stage = self._read_stage(tree)
            if stage is not None and stage.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return self._inspect_tree(tree, request)
            if tree.exists("remote", "publication.json"):
                self._recover_success(tree, request)
                return self._inspect_tree(tree, request)
            approval = self._read_approval(tree, request)
            state = self._read_state(tree, request, required=False)
            connection, _ = self._verify_current_profiles(request)
            try:
                observation = self.transport.inspect(connection=connection, request=request)
                self._validate_observation(request, observation)
            except lifecycle.RemoteTransportError:
                self._mark_recovery(tree, request, "remote_status_unavailable")
                return self._inspect_tree(tree, request)
            self._apply_observation(
                tree,
                request,
                approval,
                observation,
                cancellation_pending=(
                    tree.exists("remote", "cancellation.json")
                    or bool(state and state["status"] == "CANCEL_REQUESTED")
                ),
            )
            return self._inspect_tree(tree, request)

    def cancel(
        self,
        *,
        project_id: str,
        run_id: str,
        request_sha256: str,
        slot_id: str | None = None,
        expected_slot_binding_digest: str | None = None,
    ) -> dict[str, Any]:
        lifecycle = self._types()
        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            if request.request_sha256 != request_sha256:
                raise ValueError("cancel does not bind the exact execution request")
            approval = self._read_approval(tree, request)
            state = self._read_state(tree, request, required=False)
            stage = self._read_stage(tree)
            if tree.exists("remote", "publication.json") or (
                stage is not None
                and stage.status
                in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
            ):
                return self._inspect_tree(tree, request)
            if tree.exists("remote", "cancellation.json"):
                self._read_cancellation(tree, request)
            else:
                cancellation = lifecycle.build_remote_execution_cancellation(request)
                tree.publish_immutable_json(
                    "remote", "cancellation.json", cancellation.model_dump(mode="json")
                )
                lifecycle._commit_boundary("cancel.record")
                tree.assert_named_identity()
            remote_job_id = str(state.get("remote_job_id") or "") if state else ""
            self._write_state(tree, request, status="CANCEL_REQUESTED", remote_job_id=remote_job_id)
            connection, _ = self._verify_current_profiles(request)
            try:
                observation = self.transport.cancel(connection=connection, request=request)
                self._validate_observation(request, observation)
            except lifecycle.RemoteTransportError:
                self._mark_recovery(tree, request, "cancel_outcome_unknown", remote_job_id)
                return self._inspect_tree(tree, request)
            self._apply_observation(tree, request, approval, observation, cancellation_pending=True)
            return self._inspect_tree(tree, request)

    def recover(
        self,
        *,
        project_id: str,
        run_id: str,
        slot_id: str | None = None,
        expected_slot_binding_digest: str | None = None,
    ) -> dict[str, Any]:
        lifecycle = self._types()
        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree, tree.lifecycle_lock():
            tree.assert_named_identity()
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            stage = self._read_stage(tree)
            if stage is not None and stage.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return self._inspect_tree(tree, request)
            if tree.exists("remote", "publication.json"):
                self._recover_success(tree, request)
                return self._inspect_tree(tree, request)
            if not tree.exists("remote", "approval.json"):
                self._repair_waiting_authority(tree, request)
                return self._inspect_tree(tree, request)
            approval = self._read_approval(tree, request)
            cancellation_pending = tree.exists("remote", "cancellation.json")
            if cancellation_pending:
                self._read_cancellation(tree, request)
            connection, _ = self._verify_current_profiles(request)
            try:
                observation = self.transport.inspect(connection=connection, request=request)
                self._validate_observation(request, observation)
            except lifecycle.RemoteTransportError:
                self._mark_recovery(tree, request, "remote_status_unavailable")
                return self._inspect_tree(tree, request)
            self._apply_observation(
                tree,
                request,
                approval,
                observation,
                cancellation_pending=cancellation_pending,
            )
            return self._inspect_tree(tree, request)

    def inspect(
        self,
        *,
        project_id: str,
        run_id: str,
        slot_id: str | None = None,
        expected_slot_binding_digest: str | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._tree(
            project_id, run_id, create=False, slot_id=slot_id
        ) as tree:
            request = self._read_request(tree, project_id, run_id)
            self._verify_slot_access(tree, request, expected_slot_binding_digest)
            return self._inspect_tree(tree, request)

    def _publish_or_verify_slot_binding(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        authority: Mapping[str, Any] | None,
    ) -> AgentHarnessRemoteExecutionSlotBinding | None:
        if tree.slot_id is None:
            if authority is not None:
                raise ValueError("legacy remote execution must not receive slot authority")
            return None
        if authority is None or set(authority) != self._SLOT_AUTHORITY_FIELDS:
            raise ValueError("task-scoped remote execution requires exact slot authority")
        for field in ("planned_task_index", "attempt"):
            value = authority[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("remote execution slot indexes must be integers")
        binding = AgentHarnessRemoteExecutionSlotBinding(
            slot_id=tree.slot_id,
            project_id=request.project_id,
            run_id=request.run_id,
            controller_execution_id=authority["controller_execution_id"],
            controller_execution_digest=authority["controller_execution_digest"],
            planned_task_index=authority["planned_task_index"],
            task_id=request.task_id,
            attempt=authority["attempt"],
            task_authority_digest=authority["task_authority_digest"],
            dispatch_intent_digest=authority["dispatch_intent_digest"],
            compiled_options_digest=authority["compiled_options_digest"],
            input_artifacts_digest=authority["input_artifacts_digest"],
            output_contract_digest=authority["output_contract_digest"],
            remote_authority_id=authority["remote_authority_id"],
            remote_authority_digest=authority["remote_authority_digest"],
            remote_authority_set_id=authority["remote_authority_set_id"],
            remote_authority_set_digest=authority["remote_authority_set_digest"],
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            input_manifest_sha256=request.input_manifest.manifest_sha256,
            connection_id=request.connection_id,
            connection_profile_digest=request.connection_profile_digest,
            execution_profile_id=request.execution_profile_id,
            execution_profile_digest=request.execution_profile_digest,
            requested_resources_digest=_agent_digest(
                request.requested_resources.model_dump(mode="json")
            ),
            output_contract=request.output_contract,
            created_at=request.created_at,
        )
        if tree.exists("remote", "slot_binding.json"):
            existing = self._read_slot_binding(tree, request)
            if existing.model_dump(mode="json") != binding.model_dump(mode="json"):
                raise ValueError("remote execution slot is bound to different authority")
            return existing
        tree.publish_immutable_json(
            "remote",
            "slot_binding.json",
            binding.model_dump(mode="json"),
        )
        return self._read_slot_binding(tree, request)

    @staticmethod
    def _read_slot_binding(
        tree: PinnedExecutionTree,
        request: Any,
    ) -> AgentHarnessRemoteExecutionSlotBinding:
        if tree.slot_id is None:
            raise ValueError("legacy remote execution has no task slot binding")
        binding = AgentHarnessRemoteExecutionSlotBinding.model_validate(
            tree.read_json("remote", "slot_binding.json")
        )
        if (
            binding.slot_id != tree.slot_id
            or binding.project_id != request.project_id
            or binding.run_id != request.run_id
            or binding.task_id != request.task_id
            or binding.request_id != request.request_id
            or binding.request_sha256 != request.request_sha256
            or binding.input_manifest_sha256
            != request.input_manifest.manifest_sha256
            or binding.connection_id != request.connection_id
            or binding.connection_profile_digest
            != request.connection_profile_digest
            or binding.execution_profile_id != request.execution_profile_id
            or binding.execution_profile_digest
            != request.execution_profile_digest
            or binding.requested_resources_digest
            != _agent_digest(request.requested_resources.model_dump(mode="json"))
            or binding.output_contract != request.output_contract
        ):
            raise ValueError("remote execution slot binding does not match its request")
        return binding

    def _verify_slot_access(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        expected_slot_binding_digest: str | None,
    ) -> AgentHarnessRemoteExecutionSlotBinding | None:
        if tree.slot_id is None:
            if expected_slot_binding_digest is not None:
                raise ValueError("legacy remote execution has no slot binding digest")
            return None
        if not expected_slot_binding_digest:
            raise ValueError("task-scoped remote execution requires its slot binding digest")
        binding = self._read_slot_binding(tree, request)
        if binding.slot_binding_digest != expected_slot_binding_digest:
            raise ValueError("remote execution slot binding digest mismatch")
        return binding

    def _inspect_tree(self, tree: PinnedExecutionTree, request: Any) -> dict[str, Any]:
        slot_binding = (
            self._read_slot_binding(tree, request)
            if tree.slot_id is not None
            else None
        )
        stage = self._read_stage(tree)
        if stage is not None and stage.stage == request.task_id:
            expected_authority = {
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "connection_id": request.connection_id,
                "execution_profile_id": request.execution_profile_id,
            }
            if stage.details.get("remote_execution") != expected_authority:
                raise ValueError("StageState remote execution binding mismatch")
        approval = self._read_approval(tree, request) if tree.exists("remote", "approval.json") else None
        publication = None
        authority_succeeded = False
        if tree.exists("remote", "publication.json"):
            if approval is None:
                raise ValueError("publication is missing its approval")
            publication = self._read_publication(tree, request, approval)
            self._require_pending_success_anchor(tree, request, approval, publication)
            self._verify_local_publication(tree, request, approval, publication)
            if stage is not None and stage.status == RunStatus.SUCCEEDED:
                self._verify_completed_publication_anchor(tree, publication, stage)
                authority_succeeded = True
        elif stage is not None and stage.status == RunStatus.SUCCEEDED:
            raise ValueError("successful remote execution publication is unavailable")
        state = self._read_state(tree, request, required=False)
        if authority_succeeded:
            state = self._state_payload(request, status="SUCCEEDED", remote_job_id=str((state or {}).get("remote_job_id") or ""))
        elif publication is not None or stage is None:
            state = self._state_payload(request, status="RECOVERY_REQUIRED", error_code="authority_commit_incomplete")
        elif stage.status == RunStatus.FAILED:
            error = stage.error if isinstance(stage.error, dict) else {}
            state = self._state_payload(request, status="FAILED", error_code=str(error.get("code") or "remote_execution_failed"))
        elif stage.status == RunStatus.CANCELLED:
            state = self._state_payload(request, status="CANCELLED")
        elif approval is None:
            state = self._state_payload(request, status="WAITING_APPROVAL")
        elif state is None or state["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            state = self._state_payload(request, status="RECOVERY_REQUIRED", error_code="telemetry_unavailable")
        return {
            "request": request.model_dump(mode="json"),
            "state": state,
            "approval": approval.model_dump(mode="json") if approval else None,
            "publication": publication.model_dump(mode="json") if publication else None,
            "slot_binding": (
                slot_binding.model_dump(mode="json") if slot_binding else None
            ),
        }

    def _apply_observation(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        approval: Any,
        observation: Any,
        *,
        cancellation_pending: bool = False,
    ) -> None:
        lifecycle = self._types()
        stage = self._read_stage(tree)
        if stage is not None and stage.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return
        if observation.status == "SUCCEEDED":
            publication = observation.publication
            assert publication is not None
            self._verify_publication_binding(request, approval, publication)
            verify_remote_output_contract(request.output_contract, publication.artifacts)
            connection, _ = self._verify_current_profiles(request)
            pending_anchor = self._pending_success_anchor(request, approval, publication)
            self._anchor_pending_success(tree, request, pending_anchor)
            lifecycle._commit_boundary("success.pending_anchor")
            tree.assert_named_identity()
            committed = tree.output_is_committed(
                artifacts=publication.artifacts,
                request_sha256=request.request_sha256,
                publication_sha256=publication.publication_sha256,
                digest=lifecycle._digest,
            )
            if not committed:
                lifecycle._local_io_boundary("outputs.before_fetch")
                tree.assert_named_identity()
                try:
                    self.transport.fetch_outputs(
                        connection=connection,
                        request=request,
                        publication=publication,
                        tree=tree,
                    )
                except lifecycle.RemoteTransportError:
                    self._mark_recovery(tree, request, "output_transfer_unavailable")
                    return
                tree.assert_named_identity()
            self._verify_local_publication(tree, request, approval, publication)
            lifecycle._local_io_boundary("publication.before_record")
            tree.assert_named_identity()
            tree.publish_immutable_json("remote", "publication.json", publication.model_dump(mode="json"))
            lifecycle._commit_boundary("success.publication")
            tree.assert_named_identity()
            registry = self._publication_registry(tree, publication)
            tree.add_registry_group(registry)
            lifecycle._commit_boundary("success.registry")
            tree.assert_named_identity()
            self._write_success_stage(
                tree, request, publication, registry, pending_anchor=pending_anchor
            )
            lifecycle._commit_boundary("success.stage")
            tree.assert_named_identity()
            self._write_state(tree, request, status="SUCCEEDED", remote_job_id=observation.remote_job_id)
            lifecycle._commit_boundary("success.telemetry")
            tree.assert_named_identity()
            return
        cancellation_pending = cancellation_pending or tree.exists(
            "remote", "cancellation.json"
        )
        if cancellation_pending:
            self._read_cancellation(tree, request)
        if cancellation_pending and observation.status in {"ACCEPTED", "RUNNING", "CANCEL_REQUESTED"}:
            self._write_stage(tree, request, RunStatus.RUNNING, "remote_execution_monitor")
            self._write_state(tree, request, status="CANCEL_REQUESTED", remote_job_id=observation.remote_job_id)
            return
        if observation.status == "FAILED":
            self._write_stage(tree, request, RunStatus.FAILED, None, error_code=observation.error_code or "remote_execution_failed")
        elif observation.status == "CANCELLED":
            self._write_stage(tree, request, RunStatus.CANCELLED, None)
        else:
            self._write_stage(tree, request, RunStatus.RUNNING, "remote_execution_monitor")
        self._write_state(
            tree,
            request,
            status=observation.status,
            remote_job_id=observation.remote_job_id,
            error_code=observation.error_code,
        )

    def _recover_success(self, tree: PinnedExecutionTree, request: Any) -> None:
        lifecycle = self._types()
        tree.assert_named_identity()
        approval = self._read_approval(tree, request)
        publication = self._read_publication(tree, request, approval)
        pending_anchor = self._require_pending_success_anchor(
            tree, request, approval, publication
        )
        self._verify_local_publication(tree, request, approval, publication)
        registry = self._publication_registry(tree, publication)
        tree.add_registry_group(registry)
        lifecycle._commit_boundary("recovery.registry")
        self._write_success_stage(
            tree, request, publication, registry, pending_anchor=pending_anchor
        )
        lifecycle._commit_boundary("recovery.stage")
        prior = self._read_state(tree, request, required=False) or {}
        self._write_state(tree, request, status="SUCCEEDED", remote_job_id=str(prior.get("remote_job_id") or ""))
        lifecycle._commit_boundary("recovery.telemetry")
        tree.assert_named_identity()

    def _repair_waiting_authority(self, tree: PinnedExecutionTree, request: Any) -> None:
        self._verify_staged_inputs(tree, request)
        stage = self._read_stage(tree)
        if stage is not None:
            expected = {
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "connection_id": request.connection_id,
                "execution_profile_id": request.execution_profile_id,
            }
            if (
                stage.stage != request.task_id
                or stage.status != RunStatus.WAITING_USER
                or stage.details.get("remote_execution") != expected
            ):
                raise ValueError("prepared remote execution authority is inconsistent")
        else:
            self._write_stage(tree, request, RunStatus.WAITING_USER, "remote_execution_approval")
        self._write_state(tree, request, status="WAITING_APPROVAL")

    def _pending_success_anchor(
        self, request: Any, approval: Any, publication: Any
    ) -> dict[str, Any]:
        lifecycle = self._types()
        anchor: dict[str, Any] = {
            "schema_version": "molly_remote_pending_success.v1",
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "approval_sha256": approval.approval_sha256,
            "input_manifest_sha256": request.input_manifest.manifest_sha256,
            "publication_sha256": publication.publication_sha256,
            "output_contract": publication.output_contract,
            "artifacts": [
                item.model_dump(mode="json") for item in publication.artifacts
            ],
        }
        anchor["anchor_sha256"] = lifecycle._digest(
            lifecycle._canonical_bytes(anchor)
        )
        return anchor

    def _anchor_pending_success(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        anchor: Mapping[str, Any],
    ) -> None:
        stage = self._read_stage(tree)
        if stage is not None and stage.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise ValueError("terminal remote execution cannot be reopened")
        existing = stage.details.get("remote_execution_pending_success") if stage else None
        if existing is not None and existing != dict(anchor):
            raise ValueError("pending-success StageState anchor mismatch")
        self._write_stage(
            tree,
            request,
            RunStatus.RUNNING,
            "remote_execution_publication",
            pending_success_anchor=anchor,
        )

    def _require_pending_success_anchor(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        approval: Any,
        publication: Any,
    ) -> dict[str, Any]:
        stage = self._read_stage(tree)
        expected = self._pending_success_anchor(request, approval, publication)
        actual = stage.details.get("remote_execution_pending_success") if stage else None
        if actual != expected:
            raise ValueError("pending-success StageState anchor mismatch")
        return expected

    @staticmethod
    def _publication_registry(
        tree: PinnedExecutionTree,
        publication: Any,
    ) -> dict[str, str]:
        registry = {
            item.artifact_id: (
                f"{tree.remote_relative_root}/outputs/committed/payload/{item.relative_path}"
            )
            for item in publication.artifacts
        }
        registry[tree.publication_artifact_id] = (
            f"{tree.remote_relative_root}/publication.json"
        )
        return registry

    def _write_success_stage(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        publication: Any,
        registry: Mapping[str, str],
        *,
        pending_anchor: Mapping[str, Any],
    ) -> None:
        lifecycle = self._types()
        payload = tree.read_file("remote", "publication.json")
        self._write_stage(
            tree,
            request,
            RunStatus.SUCCEEDED,
            None,
            artifacts=[ArtifactRef(artifact_id=key, relative_path=value) for key, value in sorted(registry.items())],
            publication_anchor={
                "relative_path": f"{tree.remote_relative_root}/publication.json",
                "size_bytes": len(payload),
                "sha256": lifecycle._digest(payload),
            },
            pending_success_anchor=pending_anchor,
        )

    def _mark_recovery(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        error_code: str,
        remote_job_id: str = "",
    ) -> None:
        self._write_stage(tree, request, RunStatus.PAUSED_BY_USER, "remote_execution_recovery")
        self._write_state(tree, request, status="RECOVERY_REQUIRED", remote_job_id=remote_job_id, error_code=error_code)

    def _verify_current_profiles(self, request: Any) -> tuple[ConnectionProfile, ExecutionProfile]:
        connection = self.profiles.get_connection(request.connection_id)
        profile = self.profiles.resolve_execution_profile(request.execution_profile_id)
        verify_transfer_manifest_binding(request.input_manifest, connection=connection, execution_profile=profile)
        if (
            request.connection_profile_digest != connection.digest()
            or request.execution_profile_digest != profile.digest()
            or request.output_contract != profile.output_contract
        ):
            raise ValueError("execution profiles changed after request preparation")
        return connection, profile

    def _run_submission_preflight(self, request: Any, connection: ConnectionProfile) -> None:
        result = self.capability_probe.probe(connection.connection_id)
        profile = self.profiles.resolve_execution_profile(request.execution_profile_id)
        if (
            result.status != "available"
            or result.connection_profile_digest != request.connection_profile_digest
            or result.hostname != connection.expected_hostname
            or not set(profile.required_capabilities).issubset(set(result.verified_capabilities))
        ):
            raise ValueError("submission capability preflight did not satisfy the execution contract")

    @staticmethod
    def _validate_observation(request: Any, observation: Any) -> None:
        if observation.request_id != request.request_id or observation.request_sha256 != request.request_sha256:
            raise ValueError("remote observation does not bind the execution request")

    @staticmethod
    def _verify_publication_binding(request: Any, approval: Any, publication: Any) -> None:
        if (
            publication.request_id != request.request_id
            or publication.request_sha256 != request.request_sha256
            or publication.approval_sha256 != approval.approval_sha256
            or publication.input_manifest_sha256 != request.input_manifest.manifest_sha256
            or publication.output_contract != request.output_contract
        ):
            raise ValueError("remote publication binding mismatch")

    def _verify_local_publication(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        approval: Any,
        publication: Any,
    ) -> None:
        lifecycle = self._types()
        self._verify_publication_binding(request, approval, publication)
        if not tree.output_is_committed(
            artifacts=publication.artifacts,
            request_sha256=request.request_sha256,
            publication_sha256=publication.publication_sha256,
            digest=lifecycle._digest,
        ):
            raise ValueError("remote output publication is not committed")
        for item in publication.artifacts:
            payload = tree.read_output_file(item.relative_path)
            if len(payload) != item.size_bytes or lifecycle._digest(payload) != item.sha256:
                raise ValueError("remote output artifact digest mismatch")
        verify_remote_output_contents(
            publication.output_contract,
            publication.artifacts,
            tree.read_output_file,
        )

    def _verify_completed_publication_anchor(
        self, tree: PinnedExecutionTree, publication: Any, stage: StageState
    ) -> None:
        lifecycle = self._types()
        anchor = stage.details.get("remote_execution_publication")
        if not isinstance(anchor, dict) or set(anchor) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("remote publication StageState anchor is invalid")
        payload = tree.read_file("remote", "publication.json")
        if (
            anchor.get("relative_path")
            != f"{tree.remote_relative_root}/publication.json"
            or anchor.get("size_bytes") != len(payload)
            or anchor.get("sha256") != lifecycle._digest(payload)
        ):
            raise ValueError("remote publication StageState anchor mismatch")
        expected = self._publication_registry(tree, publication)
        registry = tree.read_registry()
        if any(registry.get(key) != value for key, value in expected.items()):
            raise ValueError("remote publication Artifact Registry binding mismatch")

    @staticmethod
    def _same_preparation(existing: Any, candidate: Any) -> bool:
        excluded = {"created_at", "request_sha256"}
        return existing.model_dump(mode="json", exclude=excluded) == candidate.model_dump(mode="json", exclude=excluded)

    def _stage_registered_inputs(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        input_artifacts: Mapping[str, str],
    ) -> None:
        lifecycle = self._types()
        bindings = {
            str(path): lifecycle._identifier(artifact_id, "artifact_id")
            for path, artifact_id in input_artifacts.items()
        }
        expected = {item.relative_path for item in request.input_manifest.artifacts}
        if set(bindings) != expected or len(set(bindings.values())) != len(bindings):
            raise ValueError("input artifact bindings must exactly cover the transfer roster")
        registry = tree.read_registry()
        for artifact in request.input_manifest.artifacts:
            registered = registry.get(bindings[artifact.relative_path])
            if not registered:
                raise ValueError("input artifact is not registered")
            lifecycle._local_io_boundary("inputs.before_copy")
            tree.assert_named_identity()
            tree.copy_run_artifact_to_inputs(
                source_relative_path=lifecycle._relative_path(registered, "registered artifact path"),
                destination_relative_path=artifact.relative_path,
                expected_size=artifact.size_bytes,
                expected_sha256=artifact.sha256,
                digest=lifecycle._digest,
            )
        self._verify_staged_inputs(tree, request)

    def _verify_staged_inputs(self, tree: PinnedExecutionTree, request: Any) -> None:
        lifecycle = self._types()
        expected = {item.relative_path: item for item in request.input_manifest.artifacts}
        if tree.scan_files("inputs") != set(expected):
            raise ValueError("staged remote input roster mismatch")
        for relative_path, artifact in expected.items():
            payload = tree.read_file("inputs", relative_path)
            if len(payload) != artifact.size_bytes or lifecycle._digest(payload) != artifact.sha256:
                raise ValueError("staged remote input digest mismatch")

    def _read_request(self, tree: PinnedExecutionTree, project_id: str, run_id: str) -> Any:
        lifecycle = self._types()
        request = lifecycle.RemoteExecutionRequest.model_validate(tree.read_json("remote", "execution_request.json"))
        if request.project_id != project_id or request.run_id != run_id:
            raise ValueError("execution request identity does not match its storage path")
        return request

    def _read_approval(self, tree: PinnedExecutionTree, request: Any) -> Any:
        lifecycle = self._types()
        approval = lifecycle.RemoteExecutionApproval.model_validate(tree.read_json("remote", "approval.json"))
        if approval.request_id != request.request_id or approval.request_sha256 != request.request_sha256:
            raise ValueError("approval identity mismatch")
        return approval

    def _read_cancellation(self, tree: PinnedExecutionTree, request: Any) -> Any:
        lifecycle = self._types()
        cancellation = lifecycle.RemoteExecutionCancellation.model_validate(
            tree.read_json("remote", "cancellation.json")
        )
        if (
            cancellation.request_id != request.request_id
            or cancellation.request_sha256 != request.request_sha256
        ):
            raise ValueError("cancellation identity mismatch")
        return cancellation

    def _read_publication(self, tree: PinnedExecutionTree, request: Any, approval: Any) -> Any:
        lifecycle = self._types()
        publication = lifecycle.RemotePublication.model_validate(tree.read_json("remote", "publication.json"))
        self._verify_publication_binding(request, approval, publication)
        return publication

    @staticmethod
    def _read_stage(tree: PinnedExecutionTree) -> StageState | None:
        payload = tree.read_stage()
        return StageState.model_validate(payload) if payload else None

    def _read_state(
        self, tree: PinnedExecutionTree, request: Any, *, required: bool
    ) -> dict[str, Any] | None:
        lifecycle = self._types()
        if not tree.exists("remote", "state.json"):
            if required:
                raise ValueError("remote lifecycle telemetry is unavailable")
            return None
        try:
            state = tree.read_json("remote", "state.json")
            expected = {
                "schema_version", "request_id", "request_sha256", "status",
                "remote_job_id", "error_code", "updated_at",
            }
            if set(state) != expected:
                raise ValueError
            if state["request_id"] != request.request_id or state["request_sha256"] != request.request_sha256:
                raise ValueError
            if state["status"] not in lifecycle._REMOTE_STATUSES | {
                "WAITING_APPROVAL",
                "APPROVED",
                "RECOVERY_REQUIRED",
            }:
                raise ValueError
            return state
        except (ValueError, TypeError):
            if required:
                raise ValueError("remote lifecycle telemetry is invalid") from None
            return None

    @staticmethod
    def _state_payload(
        request: Any,
        *,
        status: str,
        remote_job_id: str = "",
        error_code: str = "",
    ) -> dict[str, Any]:
        return {
            "schema_version": "molly_remote_execution_state.v1",
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "status": status,
            "remote_job_id": remote_job_id,
            "error_code": error_code,
            "updated_at": now_iso(),
        }

    def _write_state(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        *,
        status: str,
        remote_job_id: str = "",
        error_code: str = "",
    ) -> None:
        tree.write_json(
            "remote",
            "state.json",
            self._state_payload(request, status=status, remote_job_id=remote_job_id, error_code=error_code),
        )

    def _write_stage(
        self,
        tree: PinnedExecutionTree,
        request: Any,
        status: RunStatus,
        next_stage: str | None,
        *,
        error_code: str = "",
        artifacts: list[ArtifactRef] | None = None,
        publication_anchor: Mapping[str, Any] | None = None,
        pending_success_anchor: Mapping[str, Any] | None = None,
    ) -> None:
        previous = self._read_stage(tree)
        authority = {
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "connection_id": request.connection_id,
            "execution_profile_id": request.execution_profile_id,
        }
        if previous is not None:
            prior_authority = previous.details.get("remote_execution")
            if previous.stage == request.task_id and prior_authority not in (None, authority):
                raise ValueError("StageState belongs to a different remote execution")
            if pending_success_anchor is None:
                prior_pending = previous.details.get(
                    "remote_execution_pending_success"
                )
                if isinstance(prior_pending, dict):
                    pending_success_anchor = prior_pending
            if (
                previous.stage == request.task_id
                and previous.status == status
                and previous.next_stage == next_stage
                and prior_authority == authority
                and previous.details.get("remote_execution_publication")
                == (dict(publication_anchor) if publication_anchor is not None else None)
                and previous.details.get("remote_execution_pending_success")
                == (
                    dict(pending_success_anchor)
                    if pending_success_anchor is not None
                    else None
                )
            ):
                return
        timestamp = now_iso()
        history = list(previous.history) if previous else []
        history.append(StageHistoryItem(stage=request.task_id, status=status, updated_at=timestamp))
        state = StageState(
            stage=request.task_id,
            next_stage=next_stage,
            status=status,
            started_at=previous.started_at if previous is not None and previous.stage == request.task_id else timestamp,
            ended_at=timestamp if status in {RunStatus.WAITING_USER, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED} else None,
            updated_at=timestamp,
            error={"code": error_code} if error_code else None,
            details={
                "remote_execution": authority,
                **({"remote_execution_publication": dict(publication_anchor)} if publication_anchor is not None else {}),
                **(
                    {
                        "remote_execution_pending_success": dict(
                            pending_success_anchor
                        )
                    }
                    if pending_success_anchor is not None
                    else {}
                ),
                **(
                    {
                        "remote_execution_cancellation": self._read_cancellation(
                            tree, request
                        ).model_dump(mode="json")
                    }
                    if tree.exists("remote", "cancellation.json")
                    else {}
                ),
            },
            artifacts=artifacts or [],
            history=history,
        )
        tree.write_stage(state.model_dump(mode="json"))


__all__ = ["DescriptorRemoteExecutionLifecycleService"]
