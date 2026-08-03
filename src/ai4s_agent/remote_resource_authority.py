"""Server-owned configured remote resource authority control plane v1.

This module derives immutable, non-executable resource facts.  It never builds
a RemoteExecutionRequest and never imports or calls any transport, worker,
adapter, Gate, StageState, Executor, or remote execution service.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from platformdirs import user_config_path
from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.remote_execution_lifecycle import (
    validate_requested_resources_against_execution_profile,
)
from ai4s_agent.resource_profiles import ResourceProfileStore
from ai4s_agent.runtime_environments import (
    _absolute_config_path,
    _private_process_lock,
    _read_private_json,
    _write_private_json,
)
from ai4s_agent.schemas import (
    AgentConfiguredRemoteResources,
    AgentRemoteResourceAggregateBudget,
    AgentRemoteResourceAuthority,
    AgentRemoteResourceAuthorityDecision,
    AgentRemoteResourceAuthorityFinding,
    AgentRemoteResourceAuthorityOutcome,
    AgentRemoteResourceAuthorityRequest,
    AgentRemoteResourceAuthoritySet,
    AgentRemoteResourceSourceBinding,
    AgentRemoteResourceTaskBudgetBinding,
    AgentRemoteResourceTaskDecision,
    RemoteResourceAuthorityPolicy,
    RemoteResourceAuthorityPolicyEntry,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationConflict,
    ScientificAgentAuthorizationError,
)
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanPublication,
    _exclusive_process_lock,
    _fsync_directory,
    _pretty_json_bytes,
    _safe_scope_id,
)


RESOURCE_AUTHORITY_POLICY_FILENAME = "resource_authority_policies.json"
RESOURCE_AUTHORITY_POLICY_LOCK = ".resource_authority_policies.lock"
RESOURCE_AUTHORITY_REQUEST_VERSION = "agent_remote_resource_authority_request_binding.v1"
RESOURCE_AUTHORITY_DERIVATION_VERSION = "agent_remote_resource_authority_derivation.v1"


class RemoteResourceAuthorityError(ValueError):
    """Base fail-closed resource authority error."""


class RemoteResourceAuthorityConflict(RemoteResourceAuthorityError):
    """One immutable request or authority identity has conflicting bytes."""


class RemoteResourceAuthorityDenied(RemoteResourceAuthorityError):
    def __init__(self, decision: AgentRemoteResourceAuthorityDecision) -> None:
        self.decision = decision
        super().__init__("remote resource authority denied")


class RemoteResourceAuthorityStale(RemoteResourceAuthorityError):
    """A persisted resource authority no longer matches current sources."""


class RemoteResourceAuthorityUnavailable(RemoteResourceAuthorityError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class RemoteResourceAuthorityPolicyStore:
    """Private, process-locked owner policy store."""

    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = environ if environ is not None else os.environ
        root = config_dir or env.get("MOLLY_CONFIG_DIR") or user_config_path(
            "Molly", appauthor=False
        )
        self.config_dir = _absolute_config_path(root)
        self.path = self.config_dir / RESOURCE_AUTHORITY_POLICY_FILENAME
        self.lock_path = self.config_dir / RESOURCE_AUTHORITY_POLICY_LOCK

    def read(self) -> RemoteResourceAuthorityPolicy:
        with _private_process_lock(
            self.config_dir, self.lock_path.name
        ) as config_fd:
            payload = _read_private_json(config_fd, self.path.name)
        if payload is None:
            raise FileNotFoundError("remote resource authority policy is not configured")
        try:
            return RemoteResourceAuthorityPolicy.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise RemoteResourceAuthorityError(
                "remote resource authority policy failed strict validation"
            ) from exc

    def save(self, policy: RemoteResourceAuthorityPolicy) -> RemoteResourceAuthorityPolicy:
        """Owner-side configuration primitive; not exposed as a project API."""

        validated = RemoteResourceAuthorityPolicy.model_validate(
            policy.model_dump(mode="json")
        )
        with _private_process_lock(
            self.config_dir, self.lock_path.name
        ) as config_fd:
            _write_private_json(config_fd, self.path.name, validated.model_dump(mode="json"))
        return validated


@dataclass(frozen=True)
class RemoteResourceAuthorityEvaluation:
    decision: AgentRemoteResourceAuthorityDecision
    authorities: tuple[AgentRemoteResourceAuthority, ...]
    authority_set: AgentRemoteResourceAuthoritySet | None


@dataclass(frozen=True)
class RemoteResourceAuthorityPublication:
    decision: AgentRemoteResourceAuthorityDecision
    authorities: tuple[AgentRemoteResourceAuthority, ...]
    authority_set: AgentRemoteResourceAuthoritySet


@dataclass(frozen=True)
class CurrentRemoteResourceAuthorityBinding:
    """Permission input proving one task belongs to a complete current set."""

    authority: AgentRemoteResourceAuthority
    authority_set: AgentRemoteResourceAuthoritySet

    @property
    def authority_id(self) -> str:
        return self.authority.authority_id

    @property
    def authority_digest(self) -> str:
        return self.authority.authority_digest

    @property
    def authority_set_id(self) -> str:
        return self.authority_set.authority_set_id

    @property
    def authority_set_digest(self) -> str:
        return self.authority_set.authority_set_digest


@dataclass(frozen=True)
class _RequestSession:
    project_id: str
    proposal_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


class RemoteResourceAuthorityService:
    """Derive, publish, recover, and reverify exact per-task authorities."""

    def __init__(
        self,
        *,
        proposal_store: ScientificAgentPlanProposalStore,
        resource_profiles: ResourceProfileStore,
        policy_store: RemoteResourceAuthorityPolicyStore,
        control_store: AgentPlanControlStore,
        clock: Callable[[], str] = now_iso,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.proposal_store = proposal_store
        self.resource_profiles = resource_profiles
        self.policy_store = policy_store
        self.control_store = control_store
        self.clock = clock
        self.fault_injector = fault_injector

    def _fault(self, phase: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(phase)

    def evaluate(
        self,
        *,
        project_id: str,
        proposal_id: str,
        request: AgentRemoteResourceAuthorityRequest,
        publish_decision: bool = True,
    ) -> RemoteResourceAuthorityEvaluation:
        publication = self._verified_publication(
            project_id, proposal_id, request.expected_proposal_digest
        )
        result = self._derive(publication)
        if publish_decision:
            decision = self.control_store.publish_remote_resource_authority_decision(
                result.decision
            )
            return RemoteResourceAuthorityEvaluation(
                decision, result.authorities, result.authority_set
            )
        return result

    def publish(
        self,
        *,
        project_id: str,
        proposal_id: str,
        request: AgentRemoteResourceAuthorityRequest,
    ) -> RemoteResourceAuthorityPublication:
        request_digest = _agent_digest(
            {
                "schema_version": RESOURCE_AUTHORITY_REQUEST_VERSION,
                "project_id": project_id,
                "proposal_id": proposal_id,
                "request": request.model_dump(mode="json"),
            }
        )
        with self._request_session(
            project_id=project_id,
            proposal_id=proposal_id,
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            publication = self._verified_publication(
                project_id, proposal_id, request.expected_proposal_digest
            )
            self._fault("after_initial_proposal_read")
            candidate = self._derive(publication)
            self._fault("after_resource_decision")
            if candidate.decision.outcome == AgentRemoteResourceAuthorityOutcome.DENY:
                decision = self.control_store.publish_remote_resource_authority_decision(
                    candidate.decision
                )
                raise RemoteResourceAuthorityDenied(decision)
            if candidate.authority_set is None:
                raise RemoteResourceAuthorityError(
                    "configured remote resource decision lacks an authority set"
                )

            checkpoint = {
                "schema_version": "agent_remote_resource_authority_checkpoint.v1",
                "decision": candidate.decision.model_dump(mode="json"),
                "authorities": [item.model_dump(mode="json") for item in candidate.authorities],
                "authority_set": candidate.authority_set.model_dump(mode="json"),
            }
            self._write_or_verify(
                session.request_dir / "checkpoint.json",
                _pretty_json_bytes(checkpoint),
                conflict="resource authority checkpoint differs from the immutable request",
            )
            self._fault("after_authority_checkpoint")

            # Re-read every source through the same derivation immediately
            # before any immutable publication.
            current = self._derive(
                self._verified_publication(
                    project_id, proposal_id, request.expected_proposal_digest
                )
            )
            if not _same_evaluation(candidate, current):
                raise RemoteResourceAuthorityStale(
                    "resource authority sources changed before publication"
                )
            decision = self.control_store.publish_remote_resource_authority_decision(
                candidate.decision
            )
            self._write_marker(
                session,
                filename="decision_committed.json",
                status="DECISION_COMMITTED",
                values={
                    "decision_id": decision.decision_id,
                    "decision_digest": decision.decision_digest,
                },
            )
            published: list[AgentRemoteResourceAuthority] = []
            for index, authority in enumerate(candidate.authorities):
                published.append(
                    self.control_store.publish_remote_resource_authority(
                        authority, staging_parent=session.request_dir
                    )
                )
                self._fault(f"after_remote_authority_{index + 1}")

            verified = tuple(
                self.control_store.read_remote_resource_authority(
                    project_id=project_id,
                    authority_id=item.authority_id,
                )
                for item in published
            )
            if [item.model_dump(mode="json") for item in verified] != [
                item.model_dump(mode="json") for item in candidate.authorities
            ]:
                raise RemoteResourceAuthorityConflict(
                    "published remote authority roster differs from the checkpoint"
                )
            self._fault("after_authority_roster_verification")
            # Per-task files remain inert until the complete manifest is
            # published.  Re-read every source immediately before that
            # manifest-last activation boundary.
            before_set = self._derive(
                self._verified_publication(
                    project_id, proposal_id, request.expected_proposal_digest
                )
            )
            if not _same_evaluation(candidate, before_set):
                raise RemoteResourceAuthorityStale(
                    "resource authority sources changed before authority-set publication"
                )
            authority_set = self.control_store.publish_remote_resource_authority_set(
                candidate.authority_set,
                staging_parent=session.request_dir,
            )
            self._fault("after_authority_set_publication")
            self.verify_authority_set(
                project_id=project_id,
                authority_set_id=authority_set.authority_set_id,
                verify_current=True,
            )

            # The mutation/fault opportunity is deliberately before the last
            # source re-derive.  A changed policy/profile/probe cannot produce
            # the immutable request success marker or a success response.
            self._fault("before_authorities_committed_marker")
            final = self._derive(
                self._verified_publication(
                    project_id, proposal_id, request.expected_proposal_digest
                )
            )
            if not _same_evaluation(candidate, final):
                raise RemoteResourceAuthorityStale(
                    "resource authority sources changed before success marker"
                )
            self.verify_authority_set(
                project_id=project_id,
                authority_set_id=authority_set.authority_set_id,
                verify_current=True,
            )
            self._write_marker(
                session,
                filename="authorities_committed.json",
                status="AUTHORITIES_COMMITTED",
                values={
                    "decision_id": decision.decision_id,
                    "decision_digest": decision.decision_digest,
                    "authority_ids": [item.authority_id for item in verified],
                    "authority_digests": [item.authority_digest for item in verified],
                    "authority_set_id": authority_set.authority_set_id,
                    "authority_set_digest": authority_set.authority_set_digest,
                    "remote_task_ids": authority_set.remote_task_ids,
                    "complete_roster_digest": authority_set.complete_roster_digest,
                    "aggregate_budget_digest": authority_set.aggregate_budget_digest,
                },
            )
            verified_set = self.verify_authority_set(
                project_id=project_id,
                authority_set_id=authority_set.authority_set_id,
                verify_current=True,
            )
            return RemoteResourceAuthorityPublication(decision, verified, verified_set)

    def verify_authority(
        self,
        *,
        project_id: str,
        authority_id: str,
        verify_current: bool = True,
    ) -> AgentRemoteResourceAuthority:
        authority = self.control_store.read_remote_resource_authority(
            project_id=project_id, authority_id=authority_id
        )
        if not verify_current:
            return authority
        publication = self._verified_publication(
            project_id, authority.proposal_id, authority.proposal_digest
        )
        current = self._derive(publication)
        expected_set = current.authority_set
        if (
            current.decision.outcome != AgentRemoteResourceAuthorityOutcome.CONFIGURED
            or expected_set is None
        ):
            raise RemoteResourceAuthorityStale(
                "remote resource authority set is not currently configured"
            )
        published_set = self.verify_authority_set(
            project_id=project_id,
            authority_set_id=expected_set.authority_set_id,
            verify_current=True,
        )
        binding = next(
            (
                item
                for item in published_set.authority_bindings
                if item.authority_id == authority_id
            ),
            None,
        )
        match = next(
            (item for item in current.authorities if item.task_id == authority.task_id),
            None,
        )
        if (
            binding is None
            or binding.task_id != authority.task_id
            or binding.authority_digest != authority.authority_digest
            or match is None
            or match.model_dump(mode="json") != authority.model_dump(mode="json")
        ):
            raise RemoteResourceAuthorityStale(
                "remote resource authority no longer matches current sources"
            )
        return authority

    def verify_authority_set(
        self,
        *,
        project_id: str,
        authority_set_id: str,
        verify_current: bool = True,
    ) -> AgentRemoteResourceAuthoritySet:
        authority_set = self.control_store.read_remote_resource_authority_set(
            project_id=project_id,
            authority_set_id=authority_set_id,
        )
        if not verify_current:
            return authority_set
        publication = self._verified_publication(
            project_id,
            authority_set.proposal_id,
            authority_set.proposal_digest,
        )
        current = self._derive(publication)
        expected = current.authority_set
        if (
            current.decision.outcome != AgentRemoteResourceAuthorityOutcome.CONFIGURED
            or expected is None
            or expected.model_dump(mode="json")
            != authority_set.model_dump(mode="json")
        ):
            raise RemoteResourceAuthorityStale(
                "remote resource authority set no longer matches current sources"
            )
        expected_by_task = {item.task_id: item for item in current.authorities}
        for binding in authority_set.authority_bindings:
            persisted = self.control_store.read_remote_resource_authority(
                project_id=project_id,
                authority_id=binding.authority_id,
            )
            expected_authority = expected_by_task.get(binding.task_id)
            if (
                expected_authority is None
                or persisted.model_dump(mode="json")
                != expected_authority.model_dump(mode="json")
                or persisted.authority_digest != binding.authority_digest
            ):
                raise RemoteResourceAuthorityStale(
                    "remote resource authority set contains a stale task binding"
                )
        return authority_set

    def current_authority(
        self,
        *,
        publication: ScientificAgentPlanPublication,
        task_id: str,
    ) -> CurrentRemoteResourceAuthorityBinding:
        evaluation = self._derive(publication)
        task = next(
            (item for item in evaluation.decision.task_decisions if item.task_id == task_id),
            None,
        )
        if task is None or task.outcome != AgentRemoteResourceAuthorityOutcome.CONFIGURED:
            reason = (
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
                if task is None or not task.reason_codes
                else task.reason_codes[0]
            )
            raise RemoteResourceAuthorityUnavailable(reason)
        expected_set = evaluation.authority_set
        if expected_set is None:
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
            )
        try:
            published_set = self.control_store.read_remote_resource_authority_set(
                project_id=publication.proposal.project_id,
                authority_set_id=expected_set.authority_set_id,
            )
        except FileNotFoundError as exc:
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
            ) from exc
        if published_set.model_dump(mode="json") != expected_set.model_dump(mode="json"):
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_SOURCE_CHANGED"
            )
        binding = next(
            (
                item
                for item in published_set.authority_bindings
                if item.task_id == task_id
            ),
            None,
        )
        if (
            binding is None
            or binding.authority_id != task.authority_id
            or binding.authority_digest != task.authority_digest
        ):
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
            )
        try:
            authority = self.control_store.read_remote_resource_authority(
                project_id=publication.proposal.project_id,
                authority_id=binding.authority_id,
            )
        except FileNotFoundError as exc:
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
            ) from exc
        expected = next(
            item for item in evaluation.authorities if item.task_id == task_id
        )
        if authority.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise RemoteResourceAuthorityUnavailable(
                "REMOTE_RESOURCE_SOURCE_CHANGED"
            )
        return CurrentRemoteResourceAuthorityBinding(
            authority=authority,
            authority_set=published_set,
        )

    def _derive(
        self, publication: ScientificAgentPlanPublication
    ) -> RemoteResourceAuthorityEvaluation:
        proposal = publication.proposal
        ordered_task_ids = [item.task_id for item in proposal.run_plan.tasks]
        dispatch_by_task = {item.task_id: item for item in proposal.dispatch_intents}
        remote_task_ids = [
            task_id
            for task_id in ordered_task_ids
            if dispatch_by_task[task_id].execution_route
            == "remote_execution_service"
        ]
        remote_dispatches = [
            dispatch_by_task[task_id] for task_id in remote_task_ids
        ]
        try:
            policy = self.policy_store.read()
        except FileNotFoundError:
            policy = None
            policy_failure_reason = "REMOTE_RESOURCE_POLICY_MISSING"
        except (RemoteResourceAuthorityError, ValueError):
            policy = None
            policy_failure_reason = "REMOTE_RESOURCE_POLICY_STALE"
        else:
            policy_failure_reason = ""
        policy_version = (
            "remote-resource-authority-policy.v1"
            if policy is None
            else policy.policy_version
        )
        policy_digest = (
            _agent_digest(
                {
                    "schema_version": "missing_remote_resource_authority_policy.v1"
                }
            )
            if policy is None
            else policy.policy_digest
        )
        task_decisions: list[AgentRemoteResourceTaskDecision] = []
        authorities: list[AgentRemoteResourceAuthority] = []
        global_findings: list[AgentRemoteResourceAuthorityFinding] = []

        allowed_resource_questions = {
            f"remote_resources_{item.task_id}" for item in remote_dispatches
        }
        unsupported_questions = [
            item
            for item in proposal.questions
            if item.blocks_proposal and item.question_id not in allowed_resource_questions
        ]
        if unsupported_questions:
            global_findings.append(
                _finding(
                    "REMOTE_RESOURCE_SOURCE_CHANGED",
                    detail="Proposal contains a non-resource blocking question.",
                )
            )
        if proposal.missing_artifacts or proposal.run_plan.missing_artifacts:
            global_findings.append(
                _finding(
                    "REMOTE_RESOURCE_SOURCE_CHANGED",
                    detail="Proposal contains a missing artifact.",
                )
            )

        for planned in proposal.run_plan.tasks:
            dispatch = dispatch_by_task.get(planned.task_id)
            if dispatch is None or dispatch.execution_route != "remote_execution_service":
                continue
            findings: list[AgentRemoteResourceAuthorityFinding] = []
            authority: AgentRemoteResourceAuthority | None = None
            task_id = planned.task_id
            if dispatch.remote_task_type is None:
                findings.append(_finding("REMOTE_RESOURCE_TASK_TYPE_MISMATCH", task_id))
            if dispatch.logical_profile_id is None:
                findings.append(_finding("REMOTE_RESOURCE_PROFILE_MISMATCH", task_id))
            try:
                profile = (
                    None
                    if dispatch.logical_profile_id is None
                    else self.resource_profiles.resolve_execution_profile(
                        dispatch.logical_profile_id
                    )
                )
            except ValueError:
                profile = None
            if profile is None:
                findings.append(_finding("REMOTE_RESOURCE_EXECUTION_PROFILE_UNKNOWN", task_id))
            elif profile.task_type != dispatch.remote_task_type:
                findings.append(_finding("REMOTE_RESOURCE_TASK_TYPE_MISMATCH", task_id))

            entries = [] if policy is None else list(policy.entries)
            enabled_matches = [
                entry
                for entry in entries
                if entry.enabled
                and entry.execution_profile_id == dispatch.logical_profile_id
                and entry.remote_task_type == dispatch.remote_task_type
                and task_id in entry.allowed_task_ids
            ]
            disabled_matches = [
                entry
                for entry in entries
                if not entry.enabled
                and entry.execution_profile_id == dispatch.logical_profile_id
                and entry.remote_task_type == dispatch.remote_task_type
                and task_id in entry.allowed_task_ids
            ]
            broad_matches = [
                entry
                for entry in entries
                if entry.execution_profile_id == dispatch.logical_profile_id
                and entry.remote_task_type == dispatch.remote_task_type
            ]
            entry: RemoteResourceAuthorityPolicyEntry | None = None
            if policy is None:
                findings.append(_finding(policy_failure_reason, task_id))
            elif len(enabled_matches) > 1:
                findings.append(_finding("REMOTE_RESOURCE_POLICY_AMBIGUOUS", task_id))
            elif len(enabled_matches) == 1:
                entry = enabled_matches[0]
            elif disabled_matches:
                findings.append(_finding("REMOTE_RESOURCE_POLICY_DISABLED", task_id))
            elif broad_matches:
                findings.append(_finding("REMOTE_RESOURCE_TASK_NOT_ALLOWED", task_id))
            else:
                findings.append(_finding("REMOTE_RESOURCE_POLICY_MISSING", task_id))

            snapshot = None
            if entry is not None:
                if entry.execution_profile_id != dispatch.logical_profile_id:
                    findings.append(_finding("REMOTE_RESOURCE_PROFILE_MISMATCH", task_id))
                if entry.remote_task_type != dispatch.remote_task_type:
                    findings.append(_finding("REMOTE_RESOURCE_TASK_TYPE_MISMATCH", task_id))
                try:
                    snapshot = self.resource_profiles.authority_snapshot(
                        entry.connection_id,
                        execution_profile_id=entry.execution_profile_id,
                    )
                except ValueError:
                    findings.append(_finding("REMOTE_RESOURCE_CONNECTION_MISSING", task_id))
                if snapshot is not None:
                    if not snapshot.connection.enabled:
                        findings.append(_finding("REMOTE_RESOURCE_CONNECTION_DISABLED", task_id))
                    probe = snapshot.probe
                    if probe is None:
                        findings.append(_finding("REMOTE_RESOURCE_PROBE_MISSING", task_id))
                    elif probe.connection_profile_digest != snapshot.connection.digest():
                        findings.append(_finding("REMOTE_RESOURCE_PROBE_STALE", task_id))
                    elif probe.status != "available":
                        findings.append(_finding("REMOTE_RESOURCE_PROBE_UNAVAILABLE", task_id))
                    elif profile is not None:
                        observed_profile = next(
                            (
                                item
                                for item in publication.observation.logical_execution_profiles
                                if item.profile_id == profile.profile_id
                            ),
                            None,
                        )
                        if (
                            observed_profile is None
                            or observed_profile.availability_state != "available"
                            or observed_profile.capability_digest
                            != snapshot.profile_capability_digest
                        ):
                            findings.append(
                                _finding("REMOTE_RESOURCE_PROBE_STALE", task_id)
                            )
                        missing = sorted(
                            set(profile.required_capabilities).difference(
                                probe.verified_capabilities
                            )
                        )
                        if missing:
                            findings.append(
                                _finding("REMOTE_RESOURCE_CAPABILITY_MISSING", task_id)
                            )
                        if (
                            probe.details.cpu_threads is not None
                            and entry.configured_resources.cpu_threads
                            > probe.details.cpu_threads
                        ):
                            findings.append(
                                _finding("REMOTE_RESOURCE_LIMIT_EXCEEDED", task_id)
                            )
                        if entry.configured_resources.gpu_count > 0 and (
                            "gpu" not in probe.verified_capabilities
                            or probe.details.cuda is None
                            or probe.details.cuda.status != "available"
                        ):
                            findings.append(
                                _finding("REMOTE_RESOURCE_CAPABILITY_MISSING", task_id)
                            )

            if entry is not None and profile is not None:
                try:
                    validate_requested_resources_against_execution_profile(
                        entry.configured_resources.model_dump(mode="json"),
                        execution_profile=profile,
                    )
                except (ValidationError, ValueError) as exc:
                    code = (
                        "REMOTE_RESOURCE_DEVICE_POLICY_MISMATCH"
                        if "GPU" in str(exc) or "GPU" in str(exc).upper()
                        else "REMOTE_RESOURCE_LIMIT_EXCEEDED"
                    )
                    findings.append(_finding(code, task_id))

                resources = entry.configured_resources
                intent = dispatch.requested_resources
                if intent is None:
                    findings.append(_finding("REMOTE_RESOURCE_SOURCE_CHANGED", task_id))
                else:
                    if intent.walltime_sec is not None and resources.walltime_sec > intent.walltime_sec:
                        findings.append(_finding("REMOTE_RESOURCE_LIMIT_EXCEEDED", task_id))
                    if intent.gpu_count is not None or intent.cpu_threads is not None:
                        findings.append(_finding("REMOTE_RESOURCE_SOURCE_CHANGED", task_id))
                derived_gpu_hours = resources.gpu_count * resources.walltime_sec / 3600.0
                if resources.walltime_sec > entry.budget_limits.max_runtime_sec:
                    findings.append(_finding("REMOTE_RESOURCE_BUDGET_EXCEEDED", task_id))
                if derived_gpu_hours > entry.budget_limits.max_gpu_hours:
                    findings.append(_finding("REMOTE_RESOURCE_BUDGET_EXCEEDED", task_id))
                if entry.budget_limits.max_cost_usd is not None:
                    findings.append(
                        _finding("REMOTE_RESOURCE_COST_AUTHORITY_UNAVAILABLE", task_id)
                    )

            if not findings and entry is not None and profile is not None and snapshot is not None and snapshot.probe is not None:
                authority = self._build_authority(
                    publication=publication,
                    dispatch=dispatch,
                    entry=entry,
                    policy=policy,
                    profile=profile,
                    snapshot=snapshot,
                )
                authorities.append(authority)
                findings.append(
                    AgentRemoteResourceAuthorityFinding(
                        reason_code="REMOTE_RESOURCE_AUTHORITY_CONFIGURED",
                        outcome=AgentRemoteResourceAuthorityOutcome.CONFIGURED,
                        task_id=task_id,
                        detail="Exact server-owned resources are configured and current.",
                    )
                )
            outcome = (
                AgentRemoteResourceAuthorityOutcome.DENY
                if any(item.outcome == AgentRemoteResourceAuthorityOutcome.DENY for item in findings)
                else AgentRemoteResourceAuthorityOutcome.CONFIGURED
            )
            task_decisions.append(
                AgentRemoteResourceTaskDecision(
                    task_id=task_id,
                    outcome=outcome,
                    reason_codes=sorted({item.reason_code for item in findings}),
                    authority_id="" if authority is None else authority.authority_id,
                    authority_digest="" if authority is None else authority.authority_digest,
                    findings=findings,
                )
            )

        aggregate_budget = self._build_aggregate_budget(
            proposal=proposal,
            remote_task_ids=remote_task_ids,
            authorities=authorities,
        )
        if remote_task_ids:
            if (
                aggregate_budget.plan_max_runtime_sec is not None
                and aggregate_budget.total_walltime_upper_bound_sec
                > aggregate_budget.plan_max_runtime_sec
            ):
                global_findings.append(
                    _finding(
                        "REMOTE_RESOURCE_AGGREGATE_BUDGET_EXCEEDED",
                        detail="Sequential remote walltime exceeds the exact plan limit.",
                    )
                )
            if (
                aggregate_budget.plan_max_gpu_hours is not None
                and aggregate_budget.total_derived_gpu_hours
                > aggregate_budget.plan_max_gpu_hours
            ):
                global_findings.append(
                    _finding(
                        "REMOTE_RESOURCE_AGGREGATE_BUDGET_EXCEEDED",
                        detail="Aggregate remote GPU-hours exceed the exact plan limit.",
                    )
                )
            if aggregate_budget.plan_max_cost_usd is not None:
                global_findings.append(
                    _finding(
                        "REMOTE_RESOURCE_COST_AUTHORITY_UNAVAILABLE",
                        detail="No server-owned remote cost model is configured.",
                    )
                )

        if not remote_dispatches:
            global_findings.append(
                AgentRemoteResourceAuthorityFinding(
                    reason_code="REMOTE_RESOURCE_AUTHORITY_NOT_REQUIRED",
                    outcome=AgentRemoteResourceAuthorityOutcome.CONFIGURED,
                    detail="The exact proposal contains no remote task.",
                )
            )
        outcome = (
            AgentRemoteResourceAuthorityOutcome.DENY
            if any(item.outcome == AgentRemoteResourceAuthorityOutcome.DENY for item in task_decisions)
            or any(item.outcome == AgentRemoteResourceAuthorityOutcome.DENY for item in global_findings)
            else AgentRemoteResourceAuthorityOutcome.CONFIGURED
        )
        reason_codes = sorted(
            {item.reason_code for item in global_findings}
            | {code for item in task_decisions for code in item.reason_codes}
        )
        decision = AgentRemoteResourceAuthorityDecision(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            policy_version=policy_version,
            policy_digest=policy_digest,
            ordered_task_ids=ordered_task_ids,
            remote_task_ids=remote_task_ids,
            task_decisions=task_decisions,
            aggregate_budget=aggregate_budget,
            outcome=outcome,
            reason_codes=reason_codes,
            findings=global_findings,
            created_at=proposal.created_at,
            executable=False,
        )
        authority_set = (
            self._build_authority_set(
                decision=decision,
                aggregate_budget=aggregate_budget,
            )
            if remote_task_ids
            and outcome == AgentRemoteResourceAuthorityOutcome.CONFIGURED
            else None
        )
        return RemoteResourceAuthorityEvaluation(
            decision, tuple(authorities), authority_set
        )

    @staticmethod
    def _build_aggregate_budget(
        *,
        proposal: Any,
        remote_task_ids: Sequence[str],
        authorities: Sequence[AgentRemoteResourceAuthority],
    ) -> AgentRemoteResourceAggregateBudget:
        authorities_by_task = {item.task_id: item for item in authorities}
        bindings = [
            AgentRemoteResourceTaskBudgetBinding(
                task_id=task_id,
                authority_id=authorities_by_task[task_id].authority_id,
                authority_digest=authorities_by_task[task_id].authority_digest,
                configured_resources=authorities_by_task[task_id].configured_resources,
                budget_limits=authorities_by_task[task_id].budget_limits,
                budget_policy_digest=authorities_by_task[task_id].budget_policy_digest,
                derived_gpu_hours=authorities_by_task[task_id].derived_gpu_hours,
            )
            for task_id in remote_task_ids
            if task_id in authorities_by_task
        ]
        return AgentRemoteResourceAggregateBudget(
            remote_task_ids=list(remote_task_ids),
            per_task_budget_bindings=bindings,
            walltime_aggregation_policy="sequential_sum.v1",
            total_derived_gpu_hours=sum(item.derived_gpu_hours for item in bindings),
            total_configured_cpu_threads=sum(
                item.configured_resources.cpu_threads for item in bindings
            ),
            total_walltime_upper_bound_sec=sum(
                item.configured_resources.walltime_sec for item in bindings
            ),
            plan_max_runtime_sec=proposal.limits.get("max_runtime_sec"),
            plan_max_gpu_hours=proposal.limits.get("max_gpu_hours"),
            plan_max_cost_usd=proposal.limits.get("max_cost_usd"),
        )

    @staticmethod
    def _build_authority_set(
        *,
        decision: AgentRemoteResourceAuthorityDecision,
        aggregate_budget: AgentRemoteResourceAggregateBudget,
    ) -> AgentRemoteResourceAuthoritySet:
        bindings = list(aggregate_budget.per_task_budget_bindings)
        complete_roster_digest = _agent_digest(
            {
                "schema_version": "agent_remote_resource_authority_complete_roster.v1",
                "remote_task_ids": decision.remote_task_ids,
                "authority_bindings": [
                    item.model_dump(mode="json") for item in bindings
                ],
            }
        )
        return AgentRemoteResourceAuthoritySet(
            project_id=decision.project_id,
            run_id=decision.run_id,
            proposal_id=decision.proposal_id,
            proposal_digest=decision.proposal_digest,
            decision_id=decision.decision_id,
            decision_digest=decision.decision_digest,
            ordered_task_ids=decision.ordered_task_ids,
            remote_task_ids=decision.remote_task_ids,
            authority_bindings=bindings,
            complete_roster_digest=complete_roster_digest,
            aggregate_budget=aggregate_budget,
            aggregate_budget_digest=aggregate_budget.aggregate_budget_digest,
            created_at=decision.created_at,
            executable=False,
        )

    @staticmethod
    def _build_authority(
        *,
        publication: ScientificAgentPlanPublication,
        dispatch: Any,
        entry: RemoteResourceAuthorityPolicyEntry,
        policy: RemoteResourceAuthorityPolicy,
        profile: Any,
        snapshot: Any,
    ) -> AgentRemoteResourceAuthority:
        proposal = publication.proposal
        ordered = [item.task_id for item in proposal.run_plan.tasks]
        probe = snapshot.probe
        assert probe is not None
        dispatch_digest = _agent_digest(dispatch.model_dump(mode="json"))
        probe_digest = _agent_digest(probe.model_dump(mode="json"))
        run_plan_digest = _agent_digest(proposal.run_plan.model_dump(mode="json"))
        roster_digest = _agent_digest(
            {"schema_version": "agent_remote_task_roster.v1", "task_ids": ordered}
        )
        budget_digest = _agent_digest(entry.budget_limits.model_dump(mode="json"))
        profile_observation = next(
            item
            for item in publication.observation.logical_execution_profiles
            if item.profile_id == profile.profile_id
        )
        derived_gpu_hours = (
            entry.configured_resources.gpu_count
            * entry.configured_resources.walltime_sec
            / 3600.0
        )
        return AgentRemoteResourceAuthority(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            semantic_plan_id=proposal.semantic_plan_id,
            semantic_plan_digest=proposal.semantic_plan_digest,
            observation_id=proposal.observation_id,
            observation_digest=proposal.observation_digest,
            tool_catalog_digest=proposal.tool_catalog_digest,
            run_plan_digest=run_plan_digest,
            ordered_task_ids=ordered,
            task_roster_digest=roster_digest,
            task_id=dispatch.task_id,
            dispatch_intent_digest=dispatch_digest,
            remote_task_type=dispatch.remote_task_type,
            logical_profile_id=dispatch.logical_profile_id,
            connection_id=snapshot.connection.connection_id,
            connection_profile_digest=snapshot.connection.digest(),
            execution_profile_id=profile.profile_id,
            execution_profile_digest=profile.digest(),
            capability_probe_digest=probe_digest,
            capability_probe_status="available",
            verified_capabilities=probe.verified_capabilities,
            configured_resources=entry.configured_resources,
            budget_policy_digest=budget_digest,
            budget_limits=entry.budget_limits,
            derived_gpu_hours=derived_gpu_hours,
            resource_policy_id=entry.policy_id,
            resource_policy_digest=entry.digest(),
            authority_policy_version=policy.policy_version,
            authority_policy_digest=policy.policy_digest,
            source_bindings=[
                AgentRemoteResourceSourceBinding(
                    source_id="proposal", source_digest=proposal.proposal_digest
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="dispatch-intent", source_digest=dispatch_digest
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="task-roster", source_digest=roster_digest
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="connection-profile",
                    source_digest=snapshot.connection.digest(),
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="capability-probe", source_digest=probe_digest
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="execution-profile", source_digest=profile.digest()
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="profile-capability",
                    source_digest=profile_observation.capability_digest,
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="resource-policy", source_digest=entry.digest()
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="authority-policy", source_digest=policy.policy_digest
                ),
                AgentRemoteResourceSourceBinding(
                    source_id="budget-policy", source_digest=budget_digest
                ),
            ],
            created_at=proposal.created_at,
            executable=False,
        )

    def _verified_publication(
        self, project_id: str, proposal_id: str, expected_digest: str
    ) -> ScientificAgentPlanPublication:
        publication = self.proposal_store.read(
            project_id=project_id,
            proposal_id=proposal_id,
            verify_current=True,
        )
        if publication.proposal.proposal_digest != expected_digest:
            raise RemoteResourceAuthorityStale("proposal digest does not match request")
        return publication

    @contextmanager
    def _request_session(
        self,
        *,
        project_id: str,
        proposal_id: str,
        client_request_id: str,
        request_digest: str,
    ) -> Iterator[_RequestSession]:
        clean_project = _safe_scope_id(project_id, field="project_id")
        clean_proposal = _safe_scope_id(proposal_id, field="proposal_id")
        clean_request = _safe_scope_id(client_request_id, field="client_request_id")
        control = self.control_store._control_root(project_id=clean_project, create=True)
        if control is None:  # pragma: no cover
            raise RemoteResourceAuthorityError("control root is unavailable")
        root = control / "remote_resource_authority_requests"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise RemoteResourceAuthorityError("resource authority request root is unsafe")
        if not root.exists():
            try:
                root.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(control)
        request_dir = root / clean_request
        if request_dir.is_symlink() or (request_dir.exists() and not request_dir.is_dir()):
            raise RemoteResourceAuthorityConflict("resource authority request path is unsafe")
        if not request_dir.exists():
            try:
                request_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                pass
            _fsync_directory(root)
        if request_dir.is_symlink() or not request_dir.is_dir():
            raise RemoteResourceAuthorityConflict("resource authority request path is unsafe")
        with _exclusive_process_lock(request_dir / "request.lock"):
            session = _RequestSession(
                project_id=clean_project,
                proposal_id=clean_proposal,
                client_request_id=clean_request,
                request_digest=request_digest,
                request_dir=request_dir,
            )
            self._write_or_verify(
                request_dir / "reservation.json",
                _pretty_json_bytes(
                    {
                        "schema_version": RESOURCE_AUTHORITY_REQUEST_VERSION,
                        "status": "RESERVED",
                        "project_id": clean_project,
                        "proposal_id": clean_proposal,
                        "client_request_id": clean_request,
                        "request_digest": request_digest,
                    }
                ),
                conflict="client request ID is bound to different resource authority content",
            )
            yield session

    def _write_marker(
        self,
        session: _RequestSession,
        *,
        filename: str,
        status: str,
        values: Mapping[str, Any],
    ) -> None:
        self._write_or_verify(
            session.request_dir / filename,
            _pretty_json_bytes(
                {
                    "schema_version": RESOURCE_AUTHORITY_REQUEST_VERSION,
                    "status": status,
                    "project_id": session.project_id,
                    "proposal_id": session.proposal_id,
                    "client_request_id": session.client_request_id,
                    "request_digest": session.request_digest,
                    **dict(values),
                }
            ),
            conflict=f"{status} marker differs from immutable request",
        )

    @staticmethod
    def _write_or_verify(path: Path, payload: bytes, *, conflict: str) -> None:
        try:
            AgentPlanControlStore.write_or_verify_request_file(
                path, payload, conflict=conflict
            )
        except ScientificAgentAuthorizationConflict as exc:
            raise RemoteResourceAuthorityConflict(str(exc)) from exc
        except ScientificAgentAuthorizationError as exc:
            raise RemoteResourceAuthorityError(str(exc)) from exc


def _finding(
    reason_code: str, task_id: str = "", detail: str = ""
) -> AgentRemoteResourceAuthorityFinding:
    return AgentRemoteResourceAuthorityFinding(
        reason_code=reason_code,
        outcome=AgentRemoteResourceAuthorityOutcome.DENY,
        task_id=task_id,
        detail=detail,
    )


def _same_evaluation(
    left: RemoteResourceAuthorityEvaluation,
    right: RemoteResourceAuthorityEvaluation,
) -> bool:
    return (
        left.decision.model_dump(mode="json") == right.decision.model_dump(mode="json")
        and [item.model_dump(mode="json") for item in left.authorities]
        == [item.model_dump(mode="json") for item in right.authorities]
        and (
            None
            if left.authority_set is None
            else left.authority_set.model_dump(mode="json")
        )
        == (
            None
            if right.authority_set is None
            else right.authority_set.model_dump(mode="json")
        )
    )


__all__ = [
    "CurrentRemoteResourceAuthorityBinding",
    "RemoteResourceAuthorityConflict",
    "RemoteResourceAuthorityDenied",
    "RemoteResourceAuthorityError",
    "RemoteResourceAuthorityEvaluation",
    "RemoteResourceAuthorityPolicyStore",
    "RemoteResourceAuthorityPublication",
    "RemoteResourceAuthorityService",
    "RemoteResourceAuthorityStale",
    "RemoteResourceAuthorityUnavailable",
]
