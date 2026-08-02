from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ai4s_agent._utils import now_iso
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.schemas import (
    AgentHarnessControllerStatus,
    AgentPermissionPhase,
    AgentRunArtifactInspection,
    AgentRunControllerInspection,
    AgentRunInspection,
    AgentRunInspectionBinding,
    AgentRunInspectionSourceBinding,
    AgentRunInspectionStatus,
    AgentRunPlanInspection,
    AgentRunReplannerInspection,
    AgentRunTaskInspection,
    AgentRunToolCallInspection,
    _agent_digest,
)
from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerStore


_MAX_COLLECTION_ITEMS = 4096


class AgentRunInspectionReadError(ValueError):
    def __init__(
        self,
        reason_code: str,
        inspection_status: AgentRunInspectionStatus,
        http_status: int,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.inspection_status = inspection_status
        self.http_status = http_status

    def __str__(self) -> str:
        return self.reason_code


class AgentRunInspectionService:
    """Read-only composition of existing exact readers and current verifiers."""

    def __init__(
        self,
        *,
        storage: Any,
        proposal_store: Any,
        authorization_service: Any,
        control_store: Any,
        controller: Any,
        execution_agent_store: ExecutionAgentStore,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.control_store = control_store
        self.controller = controller
        self.execution_agent_store = execution_agent_store
        self.replanner_store = ScientificAgentReplannerStore(storage=storage)
        self.clock = clock

    def inspect(self, *, project_id: str, run_id: str) -> AgentRunInspection:
        try:
            return self._inspect(project_id=project_id, run_id=run_id)
        except AgentRunInspectionReadError:
            raise
        except FileNotFoundError as exc:
            raise AgentRunInspectionReadError(
                "RUN_INSPECTION_SOURCE_MISSING",
                AgentRunInspectionStatus.MISSING_SOURCE,
                404,
            ) from exc
        except Exception as exc:
            causes: list[BaseException] = []
            current: BaseException | None = exc
            while current is not None and current not in causes:
                causes.append(current)
                current = current.__cause__ or current.__context__
            if any(isinstance(item, FileNotFoundError) for item in causes):
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_SOURCE_MISSING",
                    AgentRunInspectionStatus.MISSING_SOURCE,
                    404,
                ) from exc
            if "missing" in str(exc).lower():
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_SOURCE_MISSING",
                    AgentRunInspectionStatus.MISSING_SOURCE,
                    404,
                ) from exc
            name = type(exc).__name__.lower()
            detail = str(exc).lower()
            if (
                "sourcechanged" in name
                or "stale" in name
                or any(token in detail for token in ("source changed", "no longer", "differ from the latest"))
            ):
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_SOURCE_STALE",
                    AgentRunInspectionStatus.STALE_SOURCE,
                    409,
                ) from exc
            if "recoveryrequired" in name:
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_RECOVERY_REQUIRED",
                    AgentRunInspectionStatus.RECOVERY_REQUIRED,
                    409,
                ) from exc
            if "conflict" in name:
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_SOURCE_REPLACED",
                    AgentRunInspectionStatus.REPLACED_SOURCE,
                    409,
                ) from exc
            if any(
                token in detail
                for token in (
                    "identity mismatch",
                    "digest mismatch",
                    "bytes do not match",
                    "different bytes",
                    "was replaced",
                    "already bound",
                )
            ):
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_SOURCE_REPLACED",
                    AgentRunInspectionStatus.REPLACED_SOURCE,
                    409,
                ) from exc
            if any(
                token in detail
                for token in ("binding mismatch", "binding is incomplete", "lacks exact")
            ):
                raise AgentRunInspectionReadError(
                    "RUN_INSPECTION_INCOMPLETE_AUTHORITY_BINDING",
                    AgentRunInspectionStatus.INCOMPLETE_AUTHORITY_CHAIN,
                    409,
                ) from exc
            raise AgentRunInspectionReadError(
                "RUN_INSPECTION_SOURCE_DAMAGED",
                AgentRunInspectionStatus.DAMAGED_SOURCE,
                422,
            ) from exc

    def _inspect(self, *, project_id: str, run_id: str) -> AgentRunInspection:
        project, run = self._existing_scope(project_id=project_id, run_id=run_id)
        publications = self._plan_publications(project_id=project, run_id=run)
        revisions, revision_applications = self._replanner_chain(
            project_id=project, run_id=run
        )
        head = self._select_plan_head(publications, revisions, revision_applications)

        authorizations = self._control_models(
            project,
            "authorization",
            self.control_store.read_authorization,
        )
        authorizations = [
            item for item in authorizations
            if item.run_id == run and item.proposal_id == head.proposal.proposal_id
        ]
        authorization = self._zero_or_one(authorizations, "AUTHORIZATION")

        start_intents = self._control_models(
            project,
            "start_intent",
            self.control_store.read_start_intent,
        )
        start_intents = [
            item for item in start_intents
            if item.run_id == run and item.proposal_id == head.proposal.proposal_id
        ]
        start_intent = self._zero_or_one(start_intents, "START_INTENT")

        permissions = self._control_models(
            project,
            "permission_decision",
            self.control_store.read_permission_decision,
        )
        permissions = [
            item for item in permissions
            if item.run_id == run and item.proposal_id == head.proposal.proposal_id
        ]
        permission = self._select_permission(
            permissions,
            authorization=authorization,
            start_intent=start_intent,
        )
        authority_sets = self._control_models(
            project,
            "remote_resource_authority_set",
            self.control_store.read_remote_resource_authority_set,
        )
        authority_sets = [
            item
            for item in authority_sets
            if item.run_id == run
            and item.proposal_id == head.proposal.proposal_id
        ]
        authority_set = self._zero_or_one(authority_sets, "AUTHORITY_SET")

        executions = [
            item for item in self.control_store.list_harness_controller_executions(project_id=project)
            if item.run_id == run
        ]
        execution = self._zero_or_one(executions, "CONTROLLER_EXECUTION")
        controller_result = None
        if execution is not None:
            controller_result = self.controller.get(
                project_id=project,
                controller_execution_id=execution.controller_execution_id,
            )
            execution = controller_result.execution
            if execution.proposal_id != head.proposal.proposal_id:
                # An applied successor deliberately requires fresh permission and
                # authorization.  The old execution remains historical authority.
                applied_successors = {
                    item.successor_proposal_id for item in revision_applications
                }
                if head.proposal.proposal_id not in applied_successors:
                    self._incomplete("CONTROLLER_PROPOSAL_BINDING")
                head = self.proposal_store.read(
                    project_id=project,
                    proposal_id=head.proposal.proposal_id,
                    verify_current=True,
                )
        else:
            # Before Controller start, currentness is established by the Planner's
            # current reader.  A running Controller uses its stronger post-start
            # receipt reconciliation instead.
            head = self.proposal_store.read(
                project_id=project,
                proposal_id=head.proposal.proposal_id,
                verify_current=True,
            )

        if authority_set is not None and (
            execution is None
            or execution.proposal_id != head.proposal.proposal_id
        ):
            verified_set = self.controller.resource_authority_service.verify_authority_set(
                project_id=project,
                authority_set_id=authority_set.authority_set_id,
                verify_current=True,
            )
            if verified_set.authority_set_digest != authority_set.authority_set_digest:
                self._incomplete("AUTHORITY_SET_BINDING")

        if authorization is not None:
            verified_authorization = self.authorization_service.verify_authorization(
                project_id=project,
                authorization_id=authorization.authorization_id,
                verify_current=execution is None,
            )
            if verified_authorization.authorization_digest != authorization.authorization_digest:
                self._incomplete("AUTHORIZATION_BINDING")
        if start_intent is not None:
            verified_intent = self.authorization_service.verify_start_intent(
                project_id=project,
                start_intent_id=start_intent.start_intent_id,
                verify_current=execution is None,
            )
            if verified_intent.start_intent_digest != start_intent.start_intent_digest:
                self._incomplete("START_INTENT_BINDING")

        decisions: list[Any] = []
        receipts: list[Any] = []
        if execution is not None:
            decisions = self.control_store.list_harness_controller_decisions(
                project_id=project,
                controller_execution_id=execution.controller_execution_id,
            )
            receipts = self.control_store.list_harness_controller_action_receipts(
                project_id=project,
                controller_execution_id=execution.controller_execution_id,
            )

        tool_proposals, tool_receipts = self._execution_agent_chain(
            project_id=project,
            run_id=run,
            controller_execution_id=(execution.controller_execution_id if execution else ""),
        )
        tool_receipts_by_proposal = {
            item.tool_call_proposal_id: item for item in tool_receipts
        }
        for tool_proposal in tool_proposals:
            application = tool_receipts_by_proposal.get(
                tool_proposal.tool_call_proposal_id
            )
            if application is None and controller_result is not None and (
                tool_proposal.controller_execution_digest
                != controller_result.execution.execution_digest
                or tool_proposal.inspection_digest
                != controller_result.inspection.inspection_digest
            ):
                # Retain the immutable proposal as a historical advisory fact;
                # it is never projected as a current executable choice.
                continue
            if application is not None:
                if application.controller_decision_id:
                    bound_decision = self.control_store.read_harness_controller_decision(
                        project_id=project,
                        decision_id=application.controller_decision_id,
                    )
                    if bound_decision.decision_digest != application.controller_decision_digest:
                        self._incomplete("TOOL_CONTROLLER_DECISION_BINDING")
                if application.controller_receipt_id:
                    bound_receipt = self.control_store.read_harness_controller_action_receipt(
                        project_id=project,
                        receipt_id=application.controller_receipt_id,
                    )
                    if bound_receipt.receipt_digest != application.controller_receipt_digest:
                        self._incomplete("TOOL_CONTROLLER_RECEIPT_BINDING")
        run_snapshot, stage = self.proposal_store.observation_builder._read_run_sources(
            project, run, head.catalog
        )
        registry: dict[str, str] = {}
        if run_snapshot.run_dir is not None:
            from ai4s_agent.scientific_agent_plan import _read_json_source

            registry_payload, registry_present, _ = _read_json_source(
                run_snapshot.run_dir / "artifact_registry.json",
                label="artifact registry",
            )
            if registry_present:
                raw_registry = registry_payload.get("artifacts")
                if not isinstance(raw_registry, dict):
                    raise ValueError("artifact registry roster is invalid")
                registry = {
                    str(artifact_id): str(relative_path)
                    for artifact_id, relative_path in raw_registry.items()
                }

        source_roster = self._source_roster(
            head=head,
            publications=publications,
            permissions=permissions,
            authority_set=authority_set,
            authorization=authorization,
            start_intent=start_intent,
            execution=execution,
            controller_result=controller_result,
            decisions=decisions,
            receipts=receipts,
            tool_proposals=tool_proposals,
            tool_receipts=tool_receipts,
            revisions=revisions,
            revision_applications=revision_applications,
        )
        plan = self._plan_projection(
            publication=head,
            permission=permission,
            authority_set=authority_set,
            authorization=authorization,
            start_intent=start_intent,
            execution=execution,
        )
        controller_projection = self._controller_projection(
            controller_result=controller_result,
            execution=execution,
            decisions=decisions,
            receipts=receipts,
        )
        task_projection, artifact_projection = self._tasks_and_artifacts(
            publication=head,
            authorization=authorization,
            execution=execution,
            controller_result=controller_result,
            receipts=receipts,
            stage=stage,
            registry=registry,
        )
        tool_projection = self._tool_projection(
            tool_proposals, tool_receipts, controller_result=controller_result
        )
        replan_projection = self._replan_projection(
            project_id=project,
            revisions=revisions,
            applications=revision_applications,
        )

        recovery_required = bool(
            controller_result
            and controller_result.inspection.status
            == AgentHarnessControllerStatus.RECOVERY_REQUIRED
        )
        status = (
            AgentRunInspectionStatus.RECOVERY_REQUIRED
            if recovery_required
            else AgentRunInspectionStatus.CURRENT
        )
        outcome = self._run_outcome(
            permission=permission,
            authorization=authorization,
            start_intent=start_intent,
            controller_result=controller_result,
        )
        return AgentRunInspection(
            project_id=project,
            run_id=run,
            created_at=self.clock(),
            inspection_status=status,
            reason_codes=[
                "RUN_INSPECTION_RECOVERY_REQUIRED"
                if recovery_required
                else "RUN_INSPECTION_CURRENT"
            ],
            authoritative_status_available=True,
            verifier_supported_run_outcome=outcome,
            plan=plan,
            controller=controller_projection,
            tool_calls=tool_projection,
            replanner=replan_projection,
            tasks=task_projection,
            artifacts=artifact_projection,
            source_roster=source_roster,
        )

    def _existing_scope(self, *, project_id: str, run_id: str) -> tuple[str, str]:
        from ai4s_agent.scientific_agent_plan import _safe_scope_id

        project = _safe_scope_id(project_id, field="project_id")
        run = _safe_scope_id(run_id, field="run_id")
        project_path = self.storage.projects_root / project
        if project_path.is_symlink() or not project_path.is_dir():
            raise FileNotFoundError("project not found")
        run_path = project_path / "runs" / run
        if run_path.is_symlink() or (run_path.exists() and not run_path.is_dir()):
            raise ValueError("run scope is unsafe")
        if run_path.exists() and not run_path.resolve().is_relative_to(project_path.resolve()):
            raise ValueError("run scope escapes project")
        return project, run

    @staticmethod
    def _children(root: Path | None) -> list[Path]:
        if root is None:
            return []
        if root.is_symlink() or not root.is_dir():
            raise ValueError("inspection collection is unsafe")
        children = sorted(root.iterdir(), key=lambda item: item.name)
        if len(children) > _MAX_COLLECTION_ITEMS:
            raise ValueError("inspection collection exceeds bounded roster")
        if any(item.is_symlink() or not item.is_dir() for item in children):
            raise ValueError("inspection collection contains an unsafe entry")
        return children

    def _plan_publications(self, *, project_id: str, run_id: str) -> list[Any]:
        root = self.proposal_store._planning_root(
            project_id=project_id,
            name="agent_plan_proposals",
            create=False,
        )
        result = []
        for child in self._children(root):
            publication = self.proposal_store.read_immutable_publication(
                project_id=project_id,
                proposal_id=child.name,
            )
            if publication.proposal.run_id == run_id:
                result.append(publication)
        if not result:
            raise FileNotFoundError("run proposal not found")
        return sorted(result, key=lambda item: item.proposal.proposal_id)

    def _control_models(
        self,
        project_id: str,
        kind: str,
        reader: Callable[..., Any],
    ) -> list[Any]:
        root = self.control_store._collection_root(
            project_id=project_id, kind=kind, create=False
        )
        id_field = {
            "permission_decision": "decision_id",
            "authorization": "authorization_id",
            "start_intent": "start_intent_id",
            "remote_resource_authority_set": "authority_set_id",
        }[kind]
        return [reader(project_id=project_id, **{id_field: child.name}) for child in self._children(root)]

    def _replanner_chain(self, *, project_id: str, run_id: str) -> tuple[list[Any], list[Any]]:
        try:
            revision_root = self.replanner_store._root(
                project_id, "agent_plan_revision_proposals", create=False
            )
        except FileNotFoundError:
            revision_root = None
        revisions = []
        for child in self._children(revision_root):
            revision = self.replanner_store.read_revision(
                project_id=project_id, revision_id=child.name
            )
            if revision.run_id == run_id:
                revisions.append(revision)
        try:
            application_root = self.replanner_store._root(
                project_id, "agent_plan_revision_applications", create=False
            )
        except FileNotFoundError:
            application_root = None
        applications = []
        for child in self._children(application_root):
            receipt = self.replanner_store.read_application(
                project_id=project_id, receipt_id=child.name
            )
            if any(item.revision_id == receipt.revision_id for item in revisions):
                applications.append(receipt)
        return (
            sorted(revisions, key=lambda item: item.revision_id),
            sorted(applications, key=lambda item: item.application_receipt_id),
        )

    def _select_plan_head(self, publications: list[Any], revisions: list[Any], applications: list[Any]):
        by_id = {item.proposal.proposal_id: item for item in publications}
        successors = {item.successor_proposal_id for item in applications}
        superseded = {item.supersedes_proposal_id for item in applications}
        if not successors.issubset(by_id) or not superseded.issubset(by_id):
            self._incomplete("REPLANNER_PROPOSAL_BINDING")
        heads = sorted(set(by_id).difference(superseded))
        if len(heads) != 1:
            raise AgentRunInspectionReadError(
                "RUN_INSPECTION_COMPETING_CURRENT_SOURCE",
                AgentRunInspectionStatus.REPLACED_SOURCE,
                409,
            )
        for receipt in applications:
            revision = next((item for item in revisions if item.revision_id == receipt.revision_id), None)
            if revision is None or revision.revision_digest != receipt.revision_digest:
                self._incomplete("REPLANNER_APPLICATION_BINDING")
        return by_id[heads[0]]

    @staticmethod
    def _zero_or_one(items: list[Any], label: str) -> Any | None:
        if len(items) > 1:
            raise AgentRunInspectionReadError(
                f"RUN_INSPECTION_COMPETING_{label}",
                AgentRunInspectionStatus.REPLACED_SOURCE,
                409,
            )
        return items[0] if items else None

    @staticmethod
    def _select_permission(
        items: list[Any], *, authorization: Any | None, start_intent: Any | None
    ) -> Any | None:
        if start_intent is not None:
            selected = [item for item in items if item.decision_id == start_intent.permission_decision_id]
            if len(selected) != 1:
                AgentRunInspectionService._incomplete("START_PERMISSION_BINDING")
            return selected[0]
        if authorization is not None:
            selected = [
                item
                for item in items
                if item.decision_id == authorization.permission_decision_id
            ]
            if len(selected) != 1:
                AgentRunInspectionService._incomplete(
                    "AUTHORIZATION_PERMISSION_BINDING"
                )
            return selected[0]
        priority = {
            AgentPermissionPhase.AUTHORIZED_START: 3,
            AgentPermissionPhase.AUTHORIZATION_CANDIDATE: 2,
            AgentPermissionPhase.PROPOSAL_REVIEW: 1,
            AgentPermissionPhase.SHADOW_COMPARISON: 0,
        }
        return max(items, key=lambda item: (priority[item.phase], item.decision_id)) if items else None

    def _execution_agent_chain(
        self, *, project_id: str, run_id: str, controller_execution_id: str
    ) -> tuple[list[Any], list[Any]]:
        root = self.execution_agent_store._root(
            project_id=project_id,
            name="agent_execution_agent_proposals",
            create=False,
        )
        proposals = []
        for child in self._children(root):
            publication = self.execution_agent_store.read_proposal(
                project_id=project_id, tool_call_proposal_id=child.name
            )
            proposal = publication.proposal
            if proposal.run_id == run_id and (
                not controller_execution_id
                or proposal.controller_execution_id == controller_execution_id
            ):
                proposals.append(proposal)
        receipts = []
        for proposal in proposals:
            receipt = self.execution_agent_store.read_committed_application_receipt(
                project_id=project_id,
                tool_call_proposal_id=proposal.tool_call_proposal_id,
            )
            if receipt is not None:
                if receipt.tool_call_proposal_digest != proposal.tool_call_proposal_digest:
                    self._incomplete("TOOL_APPLICATION_BINDING")
                receipts.append(receipt)
        return (
            sorted(proposals, key=lambda item: item.tool_call_proposal_id),
            sorted(receipts, key=lambda item: item.application_receipt_id),
        )

    @staticmethod
    def _binding(object_id: str, object_digest: str) -> AgentRunInspectionBinding:
        return AgentRunInspectionBinding(object_id=object_id, object_digest=object_digest)

    def _plan_projection(
        self,
        *,
        publication: Any,
        permission: Any,
        authority_set: Any,
        authorization: Any,
        start_intent: Any,
        execution: Any,
    ) -> AgentRunPlanInspection:
        proposal = publication.proposal
        if (
            execution
            and execution.proposal_id == proposal.proposal_id
            and execution.remote_authority_set_id
        ):
            if (
                authority_set is None
                or authority_set.authority_set_id
                != execution.remote_authority_set_id
                or authority_set.authority_set_digest
                != execution.remote_authority_set_digest
            ):
                self._incomplete("AUTHORITY_SET_BINDING")
        actor_digest = ""
        if authorization is not None:
            actor_digest = _agent_digest(
                {
                    "schema_version": "agent_run_trusted_actor_binding.v1",
                    "actor": authorization.actor,
                    "actor_source": authorization.actor_source,
                }
            )
        return AgentRunPlanInspection(
            proposal=self._binding(proposal.proposal_id, proposal.proposal_digest),
            semantic_plan=self._binding(proposal.semantic_plan_id, proposal.semantic_plan_digest),
            observation=self._binding(publication.observation.observation_id, publication.observation.observation_digest),
            tool_catalog_digest=publication.catalog.catalog_digest,
            permission_decision=(self._binding(permission.decision_id, permission.decision_digest) if permission else None),
            permission_result=(permission.outcome.value.lower() if permission else "not_evaluated"),
            authority_set=(
                self._binding(
                    authority_set.authority_set_id,
                    authority_set.authority_set_digest,
                )
                if authority_set
                else None
            ),
            authorization=(self._binding(authorization.authorization_id, authorization.authorization_digest) if authorization else None),
            authorization_mode=(authorization.authorization_mode.value if authorization else ""),
            trusted_actor_binding_digest=actor_digest,
            start_intent=(self._binding(start_intent.start_intent_id, start_intent.start_intent_digest) if start_intent else None),
            dispatch_state=("controller_created" if execution else start_intent.dispatch_state if start_intent else "not_requested"),
            required_gates=proposal.required_gates,
        )

    def _controller_projection(self, *, controller_result: Any, execution: Any, decisions: list[Any], receipts: list[Any]):
        if controller_result is None:
            return None
        latest_receipt = self.controller._latest_receipt(execution)
        latest_decision = None
        if latest_receipt is not None:
            latest_decision = self.control_store.read_harness_controller_decision(
                project_id=execution.project_id, decision_id=latest_receipt.decision_id
            )
        elif decisions:
            latest_decision = decisions[-1]
        inspection = controller_result.inspection
        route = ""
        if inspection.current_task_index is not None:
            route = execution.task_slots[inspection.current_task_index].execution_route
        durable = "committed" if latest_receipt and (
            latest_receipt.execution_started or latest_receipt.dispatch_occurred
        ) else "none"
        recovery = "required" if inspection.status == AgentHarnessControllerStatus.RECOVERY_REQUIRED else "not_required"
        return AgentRunControllerInspection(
            execution=self._binding(execution.controller_execution_id, execution.execution_digest),
            decision=(self._binding(latest_decision.decision_id, latest_decision.decision_digest) if latest_decision else None),
            receipt=(self._binding(latest_receipt.receipt_id, latest_receipt.receipt_digest) if latest_receipt else None),
            controller_revision=len(receipts),
            status=inspection.status.value,
            current_task_id=inspection.current_task_id,
            execution_route=route,
            durable_effect_state=durable,
            recovery_state=recovery,
            inspection_digest=inspection.inspection_digest,
        )

    def _tasks_and_artifacts(
        self,
        *,
        publication: Any,
        authorization: Any,
        execution: Any,
        controller_result: Any,
        receipts: list[Any],
        stage: Any,
        registry: dict[str, str],
    ) -> tuple[list[AgentRunTaskInspection], list[AgentRunArtifactInspection]]:
        proposal = publication.proposal
        intents = {
            item.task_id: item
            for item in (authorization.dispatch_intents if authorization else proposal.dispatch_intents)
        }
        slots = {item.task_id: item for item in execution.task_slots} if execution else {}
        gate_by_task: dict[str, list[str]] = {}
        if authorization:
            for binding in authorization.gate_bindings:
                gate_by_task.setdefault(binding.task_id, []).append(binding.gate_id)
        registry_binding = self._binding(
            f"registry-{proposal.run_id}", _agent_digest(registry)
        )
        stage_binding = (
            self._binding(
                f"stage-{proposal.run_id}",
                _agent_digest(stage.model_dump(mode="json")),
            )
            if stage is not None
            else None
        )
        verified_outputs: dict[str, AgentRunInspectionBinding] = {}
        output_digests: dict[str, str] = {}
        completed_tasks: set[str] = set()
        gate_snapshots: dict[str, AgentRunInspectionBinding] = {}
        gate_decisions: dict[str, AgentRunInspectionBinding] = {}
        for receipt in receipts:
            if set(receipt.reason_codes).intersection({"TASK_COMPLETED", "TASK_ADOPTED"}):
                completed_tasks.add(receipt.task_id)
            for output in receipt.verified_output_bindings:
                output_digests[output.artifact_id] = output.content_sha256
                if receipt.local_execution_publication_id:
                    verified_outputs[output.artifact_id] = self._binding(
                        receipt.local_execution_publication_id,
                        receipt.local_execution_publication_digest,
                    )
            if receipt.remote_publication_digest and receipt.task_id:
                for artifact_id in next(
                    item.output_artifacts for item in proposal.run_plan.tasks if item.task_id == receipt.task_id
                ):
                    verified_outputs[artifact_id] = self._binding(
                        f"remote-publication-{receipt.slot_id}", receipt.remote_publication_digest
                    )
            if receipt.task_id and receipt.gate_snapshot_id and receipt.gate_snapshot_hash:
                gate_snapshots[receipt.task_id] = self._binding(
                    receipt.gate_snapshot_id,
                    receipt.gate_snapshot_hash,
                )
            if receipt.task_id and receipt.gate_decision_digest:
                gate_decisions[receipt.task_id] = self._binding(
                    f"gate-decisions-{receipt.task_id}-a{receipt.attempt_ordinal}",
                    receipt.gate_decision_digest,
                )
        current_index = (
            controller_result.inspection.current_task_index if controller_result else None
        )
        task_views = []
        for index, task in enumerate(proposal.run_plan.tasks):
            intent = intents[task.task_id]
            resource_digest = (
                _agent_digest(intent.requested_resources.model_dump(mode="json"))
                if intent.requested_resources is not None
                else ""
            )
            task_stage = stage if stage is not None and stage.stage == task.task_id else None
            outcome = "not_available"
            if task.task_id in completed_tasks or (current_index is not None and index < current_index):
                outcome = "succeeded"
            elif task_stage is not None:
                outcome = task_stage.status.value.lower()
            recovery = bool(
                controller_result
                and controller_result.inspection.current_task_id == task.task_id
                and controller_result.inspection.status == AgentHarnessControllerStatus.RECOVERY_REQUIRED
            )
            publication_binding = next(
                (verified_outputs[item] for item in task.output_artifacts if item in verified_outputs),
                None,
            )
            task_views.append(
                AgentRunTaskInspection(
                    task_id=task.task_id,
                    dependency_roster=task.depends_on,
                    execution_route=(
                        intent.execution_route.value
                        if hasattr(intent.execution_route, "value")
                        else str(intent.execution_route)
                    ),
                    logical_profile_id=intent.logical_profile_id or "",
                    requested_resource_summary_digest=resource_digest,
                    stage_state=stage_binding if task_stage is not None else None,
                    stage_status=(task_stage.status.value.lower() if task_stage else "not_started"),
                    registry_binding=(registry_binding if set(task.required_artifacts + task.output_artifacts).intersection(registry) else None),
                    verified_publication=publication_binding,
                    input_artifact_refs=task.required_artifacts,
                    output_artifact_refs=task.output_artifacts,
                    verifier_supported_outcome=outcome,
                    gate_requirements=gate_by_task.get(task.task_id, []),
                    gate_snapshot=gate_snapshots.get(task.task_id),
                    gate_decision=gate_decisions.get(task.task_id),
                    recovery_required=recovery,
                )
            )

        observed = {item.artifact_id: item for item in publication.observation.available_artifacts}
        authorization_bindings = {
            item.artifact_id: item for item in (authorization.artifact_bindings if authorization else [])
        }
        producers = {
            artifact_id: task.task_id
            for task in proposal.run_plan.tasks
            for artifact_id in task.output_artifacts
        }
        consumers: dict[str, list[str]] = {}
        for task in proposal.run_plan.tasks:
            for artifact_id in task.required_artifacts:
                consumers.setdefault(artifact_id, []).append(task.task_id)
        artifact_ids = sorted(
            set(observed)
            | set(authorization_bindings)
            | set(producers)
            | set(consumers)
        )
        artifacts = []
        for artifact_id in artifact_ids:
            observation = observed.get(artifact_id)
            auth_binding = authorization_bindings.get(artifact_id)
            digest = output_digests.get(artifact_id) or (
                auth_binding.content_digest if auth_binding else observation.content_digest if observation else ""
            )
            producer = producers.get(artifact_id) or (
                observation.producer_task_id if observation and observation.producer_task_id else ""
            )
            is_input = artifact_id in consumers
            is_output = artifact_id in producers
            role = "intermediate" if is_input and is_output else "output" if is_output else "input"
            currentness = "current" if artifact_id in registry else "missing" if is_input else "stale"
            publication_binding = verified_outputs.get(artifact_id)
            artifacts.append(
                AgentRunArtifactInspection(
                    artifact_id=artifact_id,
                    artifact_digest=digest,
                    artifact_type=(observation.logical_kind if observation else "planned_artifact"),
                    artifact_role=role,
                    producer_task_id=producer,
                    consumer_task_roster=consumers.get(artifact_id, []),
                    registry_binding=(registry_binding if artifact_id in registry else None),
                    verified_publication_binding=publication_binding,
                    provenance_digest=_agent_digest(
                        {
                            "schema_version": "agent_run_artifact_provenance.v1",
                            "artifact_id": artifact_id,
                            "producer_task_id": producer,
                            "consumer_task_roster": sorted(consumers.get(artifact_id, [])),
                            "artifact_digest": digest,
                            "publication_digest": publication_binding.object_digest if publication_binding else "",
                        }
                    ),
                    currentness=currentness,
                )
            )
        return task_views, artifacts

    def _tool_projection(
        self,
        proposals: list[Any],
        receipts: list[Any],
        *,
        controller_result: Any,
    ) -> list[AgentRunToolCallInspection]:
        by_proposal = {item.tool_call_proposal_id: item for item in receipts}
        result = []
        for item in proposals:
            application = by_proposal.get(item.tool_call_proposal_id)
            is_current = bool(
                controller_result
                and item.controller_execution_digest
                == controller_result.execution.execution_digest
                and item.inspection_digest
                == controller_result.inspection.inspection_digest
            )
            result.append(AgentRunToolCallInspection(
                proposal=self._binding(item.tool_call_proposal_id, item.tool_call_proposal_digest),
                status=("applied" if application else item.status if is_current else "historical"),
                application_receipt=(
                    self._binding(
                        application.application_receipt_id,
                        application.application_receipt_digest,
                    )
                    if application else None
                ),
                durable_effect_state=(
                    "committed" if application and application.side_effect_attempted else "none"
                ),
            ))
        return result

    def _replan_projection(self, *, project_id: str, revisions: list[Any], applications: list[Any]) -> list[AgentRunReplannerInspection]:
        by_revision = {item.revision_id: item for item in applications}
        result = []
        for revision in revisions:
            application = by_revision.get(revision.revision_id)
            feedback = None
            if revision.observation.feedback_receipt_id:
                feedback = self.replanner_store.read_feedback_receipt(
                    project_id=project_id,
                    feedback_receipt_id=revision.observation.feedback_receipt_id,
                )
                if feedback.feedback_receipt_digest != revision.observation.feedback_receipt_digest:
                    self._incomplete("REPLANNER_FEEDBACK_BINDING")
            result.append(
                AgentRunReplannerInspection(
                    revision=self._binding(revision.revision_id, revision.revision_digest),
                    status=(application.status if application else revision.status),
                    feedback_receipt=(self._binding(feedback.feedback_receipt_id, feedback.feedback_receipt_digest) if feedback else None),
                    plan_diff=self._binding(revision.plan_diff.plan_diff_id, revision.plan_diff.plan_diff_digest),
                    successor_proposal=(self._binding(revision.successor_candidate.proposal_id, revision.successor_candidate.proposal_digest) if revision.successor_candidate else None),
                    application_receipt=(self._binding(application.application_receipt_id, application.application_receipt_digest) if application else None),
                    fresh_permission_required=bool(application and application.fresh_permission_required),
                    fresh_authorization_required=bool(application and application.fresh_authorization_required),
                )
            )
        return result

    def _source_roster(self, **values: Any) -> list[AgentRunInspectionSourceBinding]:
        head = values["head"]
        rows: list[AgentRunInspectionSourceBinding] = []

        def add(name: str, kind: str, object_id: str, digest: str, currentness: str = "current") -> None:
            rows.append(AgentRunInspectionSourceBinding(
                source_name=name,
                source_kind=kind,
                source_id=object_id,
                source_digest=digest,
                currentness=currentness,
            ))

        for publication in values["publications"]:
            current = publication.proposal.proposal_id == head.proposal.proposal_id
            add("plan_proposal", "planner_publication", publication.proposal.proposal_id, publication.proposal.proposal_digest, "current" if current else "historical")
            add(
                "planner_observation",
                f"planner_observation_{publication.proposal.proposal_id}",
                publication.observation.observation_id,
                publication.observation.observation_digest,
                "current" if current else "historical",
            )
            add(
                "tool_catalog",
                f"tool_catalog_{publication.proposal.proposal_id}",
                publication.catalog.catalog_id,
                publication.catalog.catalog_digest,
                "current" if current else "historical",
            )
            for binding in publication.observation.source_bindings:
                add(
                    "planner_source",
                    f"planner_source_{publication.proposal.proposal_id}",
                    binding.source_id,
                    binding.source_digest,
                    "current" if current else "historical",
                )
        for item in values["permissions"]:
            add("permission_decision", "permission_decision", item.decision_id, item.decision_digest)
        authorization = values["authorization"]
        if authorization:
            add("authorization", "authorization", authorization.authorization_id, authorization.authorization_digest)
        authority_set = values["authority_set"]
        if authority_set:
            add(
                "authority_set",
                "remote_resource_authority_set",
                authority_set.authority_set_id,
                authority_set.authority_set_digest,
            )
        start_intent = values["start_intent"]
        if start_intent:
            add("start_intent", "start_intent", start_intent.start_intent_id, start_intent.start_intent_digest)
        execution = values["execution"]
        if execution:
            add("controller_execution", "controller_execution", execution.controller_execution_id, execution.execution_digest)
            for binding in execution.source_bindings:
                add(
                    "controller_authority",
                    "controller_source_binding",
                    binding.source_id,
                    binding.source_digest,
                )
        controller_result = values["controller_result"]
        if controller_result:
            add("controller_inspection", "derived_inspection", controller_result.inspection.controller_execution_id, controller_result.inspection.inspection_digest)
            for fact in controller_result.inspection.facts:
                if fact.source_id:
                    add(fact.name, "controller_fact", fact.source_id, fact.source_digest)
        for item in values["decisions"]:
            add("controller_decision", "controller_decision", item.decision_id, item.decision_digest)
        for item in values["receipts"]:
            add("controller_receipt", "controller_receipt", item.receipt_id, item.receipt_digest)
            if item.gate_snapshot_id and item.gate_snapshot_hash:
                add(
                    "gate_snapshot",
                    "gate_snapshot",
                    item.gate_snapshot_id,
                    item.gate_snapshot_hash,
                )
            if item.gate_decision_digest and item.task_id:
                add(
                    "gate_decision",
                    "gate_decision_roster",
                    f"gate-decisions-{item.task_id}-a{item.attempt_ordinal}",
                    item.gate_decision_digest,
                )
        applied_tool_ids = {
            item.tool_call_proposal_id for item in values["tool_receipts"]
        }
        current_controller = values["controller_result"]
        for item in values["tool_proposals"]:
            is_current = bool(
                item.tool_call_proposal_id not in applied_tool_ids
                and current_controller
                and item.controller_execution_digest
                == current_controller.execution.execution_digest
                and item.inspection_digest
                == current_controller.inspection.inspection_digest
            )
            add(
                "tool_call_proposal",
                "execution_agent_proposal",
                item.tool_call_proposal_id,
                item.tool_call_proposal_digest,
                "current" if is_current else "historical",
            )
        for item in values["tool_receipts"]:
            add("tool_call_application", "execution_agent_receipt", item.application_receipt_id, item.application_receipt_digest, "historical")
        for item in values["revisions"]:
            add("replanner_revision", "replanner_revision", item.revision_id, item.revision_digest)
            add("plan_diff", "plan_diff", item.plan_diff.plan_diff_id, item.plan_diff.plan_diff_digest)
            if item.observation.feedback_receipt_id:
                add("replanner_feedback", "replanner_feedback_receipt", item.observation.feedback_receipt_id, item.observation.feedback_receipt_digest)
        for item in values["revision_applications"]:
            add("revision_application", "replanner_application_receipt", item.application_receipt_id, item.application_receipt_digest)
        unique = {
            (item.source_name, item.source_kind, item.source_id): item
            for item in rows
        }
        if len(unique) != len(rows):
            # Repeated exact facts are harmless only if their digest/currentness agree.
            for key, item in unique.items():
                matches = [row for row in rows if (row.source_name, row.source_kind, row.source_id) == key]
                if any(row != item for row in matches):
                    self._incomplete("SOURCE_ROSTER_CONFLICT")
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _run_outcome(*, permission: Any, authorization: Any, start_intent: Any, controller_result: Any) -> str:
        if controller_result is not None:
            mapping = {
                AgentHarnessControllerStatus.SUCCEEDED: "succeeded",
                AgentHarnessControllerStatus.FAILED: "failed",
                AgentHarnessControllerStatus.CANCELLED: "cancelled",
                AgentHarnessControllerStatus.RECOVERY_REQUIRED: "recovery_required",
                AgentHarnessControllerStatus.WAITING_GATE: "waiting_user",
                AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL: "waiting_user",
                AgentHarnessControllerStatus.RUNNING_REMOTE: "running",
                AgentHarnessControllerStatus.ACTIVE: "running",
            }
            return mapping[controller_result.inspection.status]
        if start_intent is not None:
            return "start_requested"
        if authorization is not None:
            return "authorized"
        if permission is not None:
            return "permission_decided"
        return "plan_proposed"

    @staticmethod
    def _incomplete(binding: str) -> None:
        raise AgentRunInspectionReadError(
            f"RUN_INSPECTION_INCOMPLETE_{binding}",
            AgentRunInspectionStatus.INCOMPLETE_AUTHORITY_CHAIN,
            409,
        )


__all__ = [
    "AgentRunInspectionReadError",
    "AgentRunInspectionService",
]
