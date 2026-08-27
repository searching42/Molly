"""Conversation-runtime wiring for the bounded failure-recovery contract.

The foundation in :mod:`scientific_agent_failure_recovery` deliberately has no
knowledge of Conversation state.  This module is the small server-owned
adapter between that contract and the authoritative Controller.  It never
executes an adapter or a worker itself: effectful recovery is delegated to the
foundation's trusted successor applicator or the existing Replanner.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.scientific_agent_failure_recovery import (
    FailureRecoveryConflict,
    FailureRecoveryDecisionInvalid,
    FailureRecoveryEffectUnknown,
    FailureRecoveryObservationInvalid,
    FailureRecoveryProviderOutcomeUnknown,
    FailureRecoveryResult,
    FailureRecoveryStale,
    FailureRecoveryStore,
    RecoverySuccessorApplicator,
    ScientificAgentFailureRecoveryService,
    failure_evidence_from_controller,
)
from ai4s_agent.scientific_agent_plan import _exclusive_process_lock
from ai4s_agent.llm_provider import LLMProvider
from ai4s_agent.schemas import (
    AgentEffectCertainty,
    AgentFailureClass,
    AgentFailureObservation,
    AgentTaskFailureEvidence,
    AgentRecoveryAction,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GRANT_FILE = "autonomy_grants.json"


class FailureRecoveryRuntimeEligibility(str, Enum):
    """Deterministic gate before the recovery foundation is entered."""

    ELIGIBLE = "ELIGIBLE"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True)
class ScientificAgentAutonomyGrantBinding:
    """A current typed grant plus its server-owned lineage anchors."""

    grant: AutonomyGrant
    authority_epoch: str
    run_id: str = ""
    session_id: str = ""
    # These are provenance facts, not new capability fields.  Lease issuance
    # copies them only into its server-owned creator binding; they never come
    # from a request, provider, or Conversation message.
    actor: str = ""
    actor_source: str = ""


@dataclass(frozen=True)
class FailureRecoveryRuntimeResult:
    """One bounded runtime continuation result.

    ``recovery`` is absent for a human/closed boundary reached before the
    foundation.  The Conversation projection may safely expose the fixed
    reason code and never receives raw exception/provider material.
    """

    eligibility: FailureRecoveryRuntimeEligibility
    observation: AgentFailureObservation | None = None
    recovery: FailureRecoveryResult | None = None
    controller_result: Any | None = None
    session_id: str = ""
    authority_epoch: str = ""
    reason_code: str = ""
    failure_class: AgentFailureClass | None = None
    effect_certainty: AgentEffectCertainty | None = None
    provider_calls_total: int = 0
    effect_count_total: int = 0
    question: str = ""

    @property
    def is_boundary(self) -> bool:
        return self.eligibility is not FailureRecoveryRuntimeEligibility.ELIGIBLE


class ScientificAgentAutonomyGrantStore:
    """Read current server-published :class:`AutonomyGrant` artifacts.

    This store is intentionally not registered as a public HTTP mutation
    endpoint.  A server bootstrap/authority service may call
    :meth:`publish_server_grant`; Conversation only calls
    :meth:`resolve_current`.  In particular, no request payload, user text, or
    recovery budget can create a grant.
    """

    def __init__(
        self,
        *,
        storage: Any,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.clock = clock

    @staticmethod
    def _clean_id(value: Any, *, field: str, allow_empty: bool = False) -> str:
        clean = str(value or "").strip()
        if allow_empty and not clean:
            return ""
        if _ID.fullmatch(clean) is None:
            raise ValueError(f"{field} is invalid")
        return clean

    def _path(self, project_id: str, *, create: bool) -> Path:
        project = self._clean_id(project_id, field="project_id")
        root = self.storage.project_dir(project) / "agent-autonomy"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FailureRecoveryConflict("autonomy grant store is unsafe")
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root / _GRANT_FILE

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.is_symlink() or not path.is_file():
            raise FailureRecoveryConflict("autonomy grant store is unsafe")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FailureRecoveryConflict("autonomy grant store is invalid") from exc
        records = payload.get("grants", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            raise FailureRecoveryConflict("autonomy grant store is invalid")
        if any(not isinstance(item, dict) for item in records):
            raise FailureRecoveryConflict("autonomy grant store contains an invalid record")
        return list(records)

    @staticmethod
    def _grant_digest(grant: AutonomyGrant) -> str:
        expected = _agent_digest(grant.scope_material())
        if grant.grant_digest != expected:
            raise FailureRecoveryStale("autonomy grant digest is stale")
        return expected

    def publish_server_grant(
        self,
        *,
        grant: AutonomyGrant,
        authority_epoch: str,
        actor: str,
        actor_source: str,
        run_id: str = "",
        session_id: str = "",
    ) -> ScientificAgentAutonomyGrantBinding:
        """Append one immutable grant issued by a trusted server authority.

        The method is for server bootstrap/authority integration and test
        fixtures.  It is deliberately absent from the Conversation routes.
        One epoch cannot be rebound to a different grant digest.
        """

        if not isinstance(grant, AutonomyGrant):
            raise TypeError("server autonomy grant must be typed")
        self._grant_digest(grant)
        epoch = self._clean_id(authority_epoch, field="authority_epoch")
        clean_actor = str(actor or "").strip()
        clean_source = str(actor_source or "").strip()
        if not clean_actor or not clean_source:
            raise ValueError("server grant actor and actor_source are required")
        if not clean_source.startswith(("config:", "server:", "wsgi.")):
            raise ValueError("server grant actor_source is not trusted")
        clean_run = self._clean_id(run_id, field="run_id", allow_empty=True)
        clean_session = self._clean_id(session_id, field="session_id", allow_empty=True)
        record = {
            "schema_version": "scientific_agent_autonomy_grant_binding.v1",
            "grant": grant.model_dump(mode="json"),
            "authority_epoch": epoch,
            "run_id": clean_run,
            "session_id": clean_session,
            "actor": clean_actor,
            "actor_source": clean_source,
            "created_at": self.clock(),
            "active": True,
        }
        path = self._path(grant.project_id, create=True)
        # Multiple independent authority approvals may issue grants for one
        # project concurrently.  Serialize the append so an atomic file
        # replacement cannot silently drop a sibling grant.
        with _exclusive_process_lock(path.with_name("autonomy_grants.lock")):
            records = self._read(path)
            for existing in records:
                existing_grant = existing.get("grant")
                if not isinstance(existing_grant, dict):
                    continue
                if existing.get("authority_epoch") == epoch:
                    if existing_grant != record["grant"]:
                        raise FailureRecoveryConflict(
                            "authority epoch is already bound to a different grant"
                        )
                    return ScientificAgentAutonomyGrantBinding(
                        grant=grant,
                        authority_epoch=epoch,
                        run_id=str(existing.get("run_id") or ""),
                        session_id=str(existing.get("session_id") or ""),
                        actor=str(existing.get("actor") or clean_actor),
                        actor_source=str(existing.get("actor_source") or clean_source),
                    )
            records.append(record)
            write_json(path, {"project_id": grant.project_id, "grants": records})
        return ScientificAgentAutonomyGrantBinding(
            grant=grant,
            authority_epoch=epoch,
            run_id=clean_run,
            session_id=clean_session,
            actor=clean_actor,
            actor_source=clean_source,
        )

    def resolve_current(
        self,
        *,
        project_id: str,
        run_id: str = "",
        session_id: str = "",
        include_expired: bool = False,
    ) -> ScientificAgentAutonomyGrantBinding | None:
        """Return the newest active grant for this lineage.

        ``include_expired`` is a server-internal restart/reconciliation seam.
        It lets the lease runtime reread an immutable grant after its validity
        window has ended so it can report ``EXPIRED`` or reconcile a known
        effect.  It does not make the grant eligible for a new effect.
        """

        path = self._path(project_id, create=False)
        records = self._read(path)
        clean_run = self._clean_id(run_id, field="run_id", allow_empty=True)
        clean_session = self._clean_id(session_id, field="session_id", allow_empty=True)
        try:
            now = datetime.fromisoformat(str(self.clock()).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise FailureRecoveryConflict("autonomy grant clock is invalid") from exc
        if now.tzinfo is None:
            raise FailureRecoveryConflict("autonomy grant clock lacks timezone")
        now = now.astimezone(timezone.utc)
        candidates: list[tuple[str, int, ScientificAgentAutonomyGrantBinding]] = []
        for index, record in enumerate(records):
            if record.get("active", True) is not True:
                continue
            if record.get("schema_version") != "scientific_agent_autonomy_grant_binding.v1":
                raise FailureRecoveryConflict("autonomy grant record version is invalid")
            actor = str(record.get("actor") or "").strip()
            actor_source = str(record.get("actor_source") or "").strip()
            # The actor fields were added to the server binding after the
            # original grant artifact format.  Keep old records readable,
            # but never accept a partially populated or untrusted new
            # provenance pair; lease issuance falls back to its own
            # server-owned creator only for the fully absent legacy pair.
            if (bool(actor) != bool(actor_source)) or (
                actor_source
                and not actor_source.startswith(("config:", "server:", "wsgi."))
            ):
                raise FailureRecoveryConflict("autonomy grant record provenance is invalid")
            raw_grant = record.get("grant")
            if not isinstance(raw_grant, dict):
                raise FailureRecoveryConflict("autonomy grant record is invalid")
            try:
                grant = AutonomyGrant.model_validate(raw_grant)
                if grant.project_id != project_id:
                    raise FailureRecoveryConflict(
                        "autonomy grant project binding is invalid"
                    )
                epoch = self._clean_id(record.get("authority_epoch"), field="authority_epoch")
                record_run = self._clean_id(record.get("run_id"), field="run_id", allow_empty=True)
                record_session = self._clean_id(record.get("session_id"), field="session_id", allow_empty=True)
                self._grant_digest(grant)
                valid_until = datetime.fromisoformat(grant.valid_until.replace("Z", "+00:00"))
                valid_from = (
                    datetime.fromisoformat(grant.valid_from.replace("Z", "+00:00"))
                    if grant.valid_from
                    else None
                )
            except (TypeError, ValueError, AttributeError) as exc:
                raise FailureRecoveryConflict("autonomy grant record failed validation") from exc
            if record_run and record_run != clean_run:
                continue
            if record_session and record_session != clean_session:
                continue
            if not include_expired and (
                (valid_from is not None and now < valid_from) or now >= valid_until
            ):
                continue
            candidates.append(
                (
                    str(record.get("created_at") or ""),
                    index,
                    ScientificAgentAutonomyGrantBinding(
                        grant=grant,
                        authority_epoch=epoch,
                        run_id=record_run,
                        session_id=record_session,
                        actor=actor,
                        actor_source=actor_source,
                    ),
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][2]


class ScientificAgentAutonomyGrantIssuer:
    """Server-owned issuance adapter for an approved authority chain.

    Recovery never creates authority at a failure boundary.  This issuer is
    called only by the existing ``approve_and_start`` authority lifecycle and
    derives the grant scope from the immutable authorization plus server
    configuration.  No request/recovery budget is accepted here.
    """

    def __init__(
        self,
        *,
        grant_store: ScientificAgentAutonomyGrantStore,
        registry: Any | None = None,
        max_retries: int = 1,
        max_replans: int = 1,
        grant_ttl_seconds: int = 86_400,
        max_active_execution_seconds: float = 900.0,
        max_remote_runtime_seconds: float = 900.0,
        enabled: bool = True,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.grant_store = grant_store
        self.registry = registry
        self.max_retries = self._count(max_retries, field="max_retries")
        self.max_replans = self._count(max_replans, field="max_replans")
        self.grant_ttl_seconds = self._count(
            grant_ttl_seconds,
            field="grant_ttl_seconds",
        )
        self.max_active_execution_seconds = self._budget(
            max_active_execution_seconds,
            field="max_active_execution_seconds",
        )
        self.max_remote_runtime_seconds = self._budget(
            max_remote_runtime_seconds,
            field="max_remote_runtime_seconds",
        )
        self.enabled = bool(enabled)
        self.clock = clock

    @staticmethod
    def _count(value: Any, *, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a non-negative integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a non-negative integer") from exc
        if parsed < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return parsed

    @staticmethod
    def _budget(value: Any, *, field: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a finite non-negative number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a finite non-negative number") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{field} must be a finite non-negative number")
        return parsed

    @staticmethod
    def _future_timestamp(now: str, seconds: int) -> str:
        try:
            parsed = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("grant issuer clock must return an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("grant issuer clock must return a timezone-aware timestamp")
        return (parsed + timedelta(seconds=seconds)).astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def _effect_classes(self, authorization: Any) -> list[str]:
        values: set[str] = set()
        if self.registry is None:
            return []
        for task_id in getattr(authorization, "task_ids", ()):
            try:
                spec = self.registry.get(task_id)
            except (TypeError, ValueError):
                continue
            effect = getattr(spec, "effect_class", None)
            if effect:
                values.add(str(getattr(effect, "value", effect)))
        return sorted(values)

    def _parameter_bounds(self, authorization: Any) -> dict[str, Any]:
        """Bind approved compiled options exactly for successor recovery.

        Automatic recovery may retry the approved values, but a parameter
        change is an authority expansion and therefore requires a new grant.
        """

        bounds: dict[str, Any] = {}
        options_by_task = getattr(authorization, "compiled_task_options", {})
        if not isinstance(options_by_task, Mapping):
            return bounds
        for task_id, options in options_by_task.items():
            if not isinstance(options, Mapping):
                continue
            for key, value in options.items():
                bounds[f"{task_id}.{key}"] = AutonomyParameterBound(
                    allowed_values=[value]
                )
        return bounds

    def issue_from_approved_chain(self, result: Any) -> ScientificAgentAutonomyGrantBinding | None:
        """Publish the durable grant for one verified server approval.

        The operation is idempotent: the authorization digest is the
        authority epoch, so a replay of ``approve_and_start`` verifies the
        same immutable grant bytes rather than minting a new authority.
        """

        if not self.enabled or (self.max_retries == 0 and self.max_replans == 0):
            return None
        authorization = getattr(result, "authorization", None)
        start_intent = getattr(result, "start_intent", None)
        if authorization is None or start_intent is None:
            raise ValueError("approved authority chain is incomplete")
        if (
            getattr(start_intent, "project_id", None)
            != getattr(authorization, "project_id", None)
            or getattr(start_intent, "run_id", None)
            != getattr(authorization, "run_id", None)
            or getattr(start_intent, "authorization_id", None)
            != getattr(authorization, "authorization_id", None)
            or getattr(start_intent, "authorization_digest", None)
            != getattr(authorization, "authorization_digest", None)
        ):
            raise ValueError("approved authority chain has inconsistent bindings")
        project_id = str(getattr(authorization, "project_id", "") or "")
        run_id = str(getattr(authorization, "run_id", "") or "")
        actor = str(getattr(authorization, "actor", "") or "").strip()
        if not project_id or not run_id or not actor:
            raise ValueError("approved authority chain has incomplete grant identity")
        # Bind timestamps to the immutable authorization publication so a
        # replay of the same approve-and-start request produces byte-identical
        # grant content for the epoch (and cannot accidentally rebind it).
        created_at = str(getattr(authorization, "created_at", "") or self.clock())
        valid_until = self._future_timestamp(created_at, self.grant_ttl_seconds)
        grant = AutonomyGrant(
            project_id=project_id,
            allowed_tasks=list(getattr(authorization, "task_ids", ()) or ()),
            allowed_effect_classes=self._effect_classes(authorization),
            parameter_bounds=self._parameter_bounds(authorization),
            resource_profiles=sorted(
                {
                    str(getattr(binding, "profile_id", ""))
                    for binding in (getattr(authorization, "profile_bindings", ()) or ())
                    if str(getattr(binding, "profile_id", "") or "")
                }
            ),
            aggregate_budget={
                "active_execution_seconds": self.max_active_execution_seconds,
                "remote_runtime_seconds": self.max_remote_runtime_seconds,
            },
            max_retries=self.max_retries,
            max_replans=self.max_replans,
            valid_from=created_at,
            valid_until=valid_until,
            created_at=created_at,
        )
        authorization_digest = str(getattr(authorization, "authorization_digest", "") or "")
        if _DIGEST.fullmatch(authorization_digest) is None:
            raise ValueError("approved authority chain has no authorization digest")
        epoch = "authorization-epoch-" + authorization_digest.split(":", 1)[1][:40]
        return self.grant_store.publish_server_grant(
            grant=grant,
            authority_epoch=epoch,
            actor=actor,
            actor_source="server:authorization-issuance",
            run_id=run_id,
        )


class ScientificAgentFailureRecoveryServiceFactory:
    """Build foundation services from fixed server dependencies."""

    def __init__(
        self,
        *,
        storage: Any,
        controller: Any,
        replanner: Any | None,
        successor_applicator: RecoverySuccessorApplicator | None,
        proposal_store: Any | None = None,
        authorization_service: Any | None = None,
        registry: Any | None = None,
        tool_roster_provider: Callable[..., Sequence[str]] | None = None,
        allowed_recovery_tools: Sequence[str] | None = None,
        tool_schemas: Mapping[str, Mapping[str, Any]] | None = None,
        tool_semantic_boundaries: Mapping[str, SemanticBoundary | str] | None = None,
        actor: str = "system",
        actor_source: str = "config:AI4S_AGENT_AUTHORIZATION_OWNER",
        baseline_authorization_id: str = "",
        baseline_authorization_digest: str = "",
        effect_reconciler: Callable[..., Any] | None = None,
        clock: Callable[[], str] = now_iso,
        store: FailureRecoveryStore | None = None,
    ) -> None:
        self.storage = storage
        self.controller = controller
        self.replanner = replanner
        self.successor_applicator = successor_applicator
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.registry = registry
        self.tool_roster_provider = tool_roster_provider
        self.allowed_recovery_tools = tuple(allowed_recovery_tools or ())
        self.tool_schemas = {str(key): dict(value) for key, value in (tool_schemas or {}).items()}
        self.tool_semantic_boundaries = dict(tool_semantic_boundaries or {})
        self.actor = str(actor or "system")
        self.actor_source = str(actor_source or "config:AI4S_AGENT_AUTHORIZATION_OWNER")
        self.baseline_authorization_id = baseline_authorization_id
        self.baseline_authorization_digest = baseline_authorization_digest
        self.effect_reconciler = effect_reconciler
        self.clock = clock
        self.store = store or FailureRecoveryStore(storage=storage)

    def build(
        self,
        *,
        provider: LLMProvider | None,
        grant: AutonomyGrant,
        session_id: str,
        authority_epoch: str,
    ) -> ScientificAgentFailureRecoveryService:
        effect_reconciler = self.effect_reconciler
        if effect_reconciler is None and isinstance(
            self.successor_applicator, RecoverySuccessorApplicator
        ):
            # The reviewed production applicator exposes a deterministic,
            # request-id-bound replay hook.  Select it here, inside the
            # server-owned factory, so a Conversation caller cannot inject an
            # arbitrary reconciliation callback while crash windows still
            # resolve to the same Controller effect.
            candidate = getattr(
                self.successor_applicator,
                "reconcile_recovery_successor",
                None,
            )
            if callable(candidate):
                # The concrete applicator reconciles the Permission ->
                # Authorization -> StartIntent -> Controller successor path.
                # A REPLAN effect, however, belongs to the existing Replanner
                # and must be replayed through its own request checkpoint.  A
                # single generic callback would otherwise turn a crashed
                # Replanner request into a second Controller successor.
                def reconcile_effect(*, observation: Any, decision: Any) -> Any:
                    action = getattr(
                        getattr(decision, "recovery_action", None),
                        "value",
                        getattr(decision, "recovery_action", None),
                    )
                    if action == AgentRecoveryAction.REPLAN.value:
                        method = getattr(
                            self.replanner,
                            "create_current_controller_failure_revision",
                            None,
                        )
                        if not callable(method) or provider is None:
                            raise FailureRecoveryEffectUnknown(
                                "existing Replanner outcome is unknown"
                            )
                        return method(
                            project_id=observation.project_id,
                            run_id=observation.run_id,
                            controller_execution_id=observation.controller_execution_id,
                            controller_execution_digest=observation.controller_execution_digest,
                            actor=self.actor,
                            actor_source=self.actor_source,
                            provider=provider,
                        )
                    return candidate(observation=observation, decision=decision)

                effect_reconciler = reconcile_effect
        return ScientificAgentFailureRecoveryService(
            storage=self.storage,
            controller=self.controller,
            provider=provider,
            grant=grant,
            replanner=self.replanner,
            successor_applicator=self.successor_applicator,
            effect_reconciler=effect_reconciler,
            tool_roster_provider=self.tool_roster_provider,
            allowed_recovery_tools=self.allowed_recovery_tools,
            tool_schemas=self.tool_schemas,
            tool_semantic_boundaries=self.tool_semantic_boundaries,
            registry=self.registry,
            actor=self.actor,
            actor_source=self.actor_source,
            baseline_authorization_id=self.baseline_authorization_id,
            baseline_authorization_digest=self.baseline_authorization_digest,
            session_id=session_id,
            authority_epoch=authority_epoch,
            clock=self.clock,
            store=self.store,
        )


class ScientificAgentFailureRecoveryRuntime:
    """Run at most one bounded recovery operation for one FAILED snapshot."""

    def __init__(
        self,
        *,
        storage: Any,
        controller: Any,
        grant_source: ScientificAgentAutonomyGrantStore | Any | None,
        service_factory: ScientificAgentFailureRecoveryServiceFactory | Any,
        proposal_store: Any | None = None,
        authorization_service: Any | None = None,
        registry: Any | None = None,
        store: FailureRecoveryStore | None = None,
        recovery_provider_resolver: Callable[
            [], AbstractContextManager[LLMProvider | None] | LLMProvider | None
        ]
        | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.controller = controller
        self.grant_source = grant_source
        self.service_factory = service_factory
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.registry = registry
        self.store = store or getattr(service_factory, "store", None) or FailureRecoveryStore(storage=storage)
        # Production wiring supplies a server-owned context factory.  The
        # legacy ``provider`` argument to ``continue_failed`` remains useful
        # for direct foundation fixtures, but Conversation never forwards its
        # ordinary request provider into this runtime.
        self.recovery_provider_resolver = recovery_provider_resolver
        self.clock = clock

    def _provider_context(
        self,
        *,
        fallback_provider: LLMProvider | None,
    ) -> AbstractContextManager[LLMProvider | None]:
        resolver = self.recovery_provider_resolver
        if resolver is None:
            return nullcontext(fallback_provider)
        try:
            resolved = resolver()
        except Exception:
            # Provider availability is a deterministic no-provider boundary;
            # the foundation will choose ASK_USER/STOP or fail closed without
            # crossing an arbitrary request-selected provider boundary.
            return nullcontext(None)
        if resolved is None:
            return nullcontext(None)
        if hasattr(resolved, "__enter__") and hasattr(resolved, "__exit__"):
            return resolved  # type: ignore[return-value]
        # Test doubles and structural LLMProvider implementations need not
        # inherit a concrete class; the runtime only requires the provider
        # protocol when the foundation actually makes a call.
        return nullcontext(resolved)  # type: ignore[arg-type]

    @staticmethod
    def session_id(*, conversation_id: str, run_id: str) -> str:
        # Conversation ID is the stable user-visible lineage.  Run ID is
        # included to prevent accidental cross-run aggregate sharing when a
        # coordinator is reused for two conversations.
        return "conversation-recovery-" + _agent_digest(
            {"conversation_id": str(conversation_id), "run_id": str(run_id)}
        ).split(":", 1)[1][:40]

    @staticmethod
    def authority_epoch_for_grant(grant: AutonomyGrant) -> str:
        # A direct injected grant (used by server integration fixtures) still
        # gets a non-constant epoch.  The production store supplies an
        # explicit generation/epoch and is preferred by ``_grant_binding``.
        return "grant-epoch-" + grant.grant_digest.split(":", 1)[1][:40]

    @staticmethod
    def _validate_grant_binding(
        binding: ScientificAgentAutonomyGrantBinding,
        *,
        project_id: str,
        run_id: str,
        session_id: str,
    ) -> ScientificAgentAutonomyGrantBinding:
        """Verify every grant returned by a server authority source.

        The production source is :class:`ScientificAgentAutonomyGrantStore`,
        but keeping this check at the runtime boundary prevents an alternate
        in-process source from smuggling an unbound, forged, or expired grant
        into the foundation.
        """

        if not isinstance(binding, ScientificAgentAutonomyGrantBinding):
            raise FailureRecoveryObservationInvalid(
                "server autonomy grant resolver returned an invalid binding"
            )
        grant = binding.grant
        if not isinstance(grant, AutonomyGrant):
            raise FailureRecoveryObservationInvalid(
                "server autonomy grant resolver returned an untyped grant"
            )
        if grant.project_id != project_id:
            raise FailureRecoveryStale("autonomy grant project binding is stale")
        expected_digest = _agent_digest(grant.scope_material())
        if grant.grant_digest != expected_digest:
            raise FailureRecoveryStale("autonomy grant digest is stale")
        epoch = str(binding.authority_epoch or "").strip()
        if _ID.fullmatch(epoch) is None:
            raise FailureRecoveryStale("autonomy grant authority epoch is invalid")
        if binding.run_id and binding.run_id != run_id:
            raise FailureRecoveryStale("autonomy grant run binding is stale")
        if binding.session_id and binding.session_id != session_id:
            raise FailureRecoveryStale("autonomy grant session binding is stale")
        try:
            valid_until = datetime.fromisoformat(grant.valid_until.replace("Z", "+00:00"))
            valid_from = (
                datetime.fromisoformat(grant.valid_from.replace("Z", "+00:00"))
                if grant.valid_from
                else None
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise FailureRecoveryStale("autonomy grant validity is invalid") from exc
        now = datetime.now(timezone.utc)
        if (valid_from is not None and now < valid_from) or now > valid_until:
            raise FailureRecoveryStale("autonomy grant is not currently valid")
        return ScientificAgentAutonomyGrantBinding(
            grant=grant,
            authority_epoch=epoch,
            run_id=binding.run_id,
            session_id=binding.session_id,
            actor=binding.actor,
            actor_source=binding.actor_source,
        )

    def _grant_binding(
        self,
        *,
        project_id: str,
        run_id: str,
        session_id: str,
    ) -> ScientificAgentAutonomyGrantBinding | None:
        source = self.grant_source
        if source is None:
            return None
        resolver = getattr(source, "resolve_current", None)
        if not callable(resolver):
            raise FailureRecoveryObservationInvalid("server autonomy grant resolver is unavailable")
        try:
            value = resolver(
                project_id=project_id,
                run_id=run_id,
                session_id=session_id,
            )
        except TypeError:
            value = resolver(project_id=project_id, run_id=run_id)
        if value is None:
            return None
        if isinstance(value, ScientificAgentAutonomyGrantBinding):
            binding = value
        elif isinstance(value, AutonomyGrant):
            binding = ScientificAgentAutonomyGrantBinding(
                grant=value,
                authority_epoch=self.authority_epoch_for_grant(value),
            )
        elif isinstance(value, Mapping):
            raw_grant = value.get("grant", value)
            grant = raw_grant if isinstance(raw_grant, AutonomyGrant) else AutonomyGrant.model_validate(raw_grant)
            epoch = str(value.get("authority_epoch") or self.authority_epoch_for_grant(grant))
            binding = ScientificAgentAutonomyGrantBinding(
                grant=grant,
                authority_epoch=epoch,
                run_id=str(value.get("run_id") or ""),
                session_id=str(value.get("session_id") or ""),
                actor=str(value.get("actor") or ""),
                actor_source=str(value.get("actor_source") or ""),
            )
        else:
            raise FailureRecoveryObservationInvalid("server autonomy grant resolver returned an invalid value")
        return self._validate_grant_binding(
            binding,
            project_id=project_id,
            run_id=run_id,
            session_id=session_id,
        )

    def _existing_observation(
        self,
        *,
        project_id: str,
        state: Mapping[str, Any] | None,
        execution: Any,
        inspection: Any,
    ) -> AgentFailureObservation | None:
        failure_id = str((state or {}).get("last_recovery_failure_id") or "")
        if failure_id:
            try:
                candidate = self.store.read_observation(project_id=project_id, failure_id=failure_id)
            except FileNotFoundError:
                candidate = None
            if candidate is not None:
                if (
                    candidate.controller_execution_id == execution.controller_execution_id
                    and candidate.controller_execution_digest == execution.execution_digest
                    and candidate.inspection_digest == inspection.inspection_digest
                ):
                    return candidate
        finder = getattr(self.store, "find_observation_for_controller", None)
        if callable(finder):
            return finder(
                project_id=project_id,
                controller_execution_id=execution.controller_execution_id,
                controller_execution_digest=execution.execution_digest,
                inspection_digest=inspection.inspection_digest,
            )
        return None

    def _current_snapshot(self, *, controller_result: Any) -> Any:
        execution = getattr(controller_result, "execution", None)
        inspection = getattr(controller_result, "inspection", None)
        if execution is None or inspection is None:
            raise FailureRecoveryStale("current Controller snapshot is incomplete")
        reader = getattr(self.controller, "read_execution_agent_snapshot", None)
        if not callable(reader):
            raise FailureRecoveryDecisionInvalid(
                "automatic recovery requires a current Controller snapshot verifier"
            )
        snapshot = reader(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
            expected_controller_execution_digest=execution.execution_digest,
        )
        if (
            getattr(snapshot, "execution", None) is None
            or getattr(snapshot, "inspection", None) is None
        ):
            raise FailureRecoveryStale("current Controller snapshot is incomplete")
        return snapshot

    def _typed_evidence(self, *, snapshot: Any) -> AgentTaskFailureEvidence:
        execution = snapshot.execution
        inspection = snapshot.inspection
        task_id = str(getattr(inspection, "current_task_id", "") or "")
        logical_tool_id = task_id
        registry = self.registry
        if registry is not None and task_id:
            try:
                logical_tool_id = str(getattr(registry.get(task_id), "scientific_tool_id", "") or task_id)
            except (ValueError, TypeError):
                logical_tool_id = task_id
        stage_reader = getattr(self.storage, "read_stage_state", None)
        stage = None
        if callable(stage_reader):
            stage = stage_reader(execution.project_id, execution.run_id)
        details = getattr(stage, "details", None) if stage is not None else None
        raw = details.get("typed_failure_evidence") if isinstance(details, Mapping) else None
        if isinstance(raw, Mapping):
            try:
                evidence = AgentTaskFailureEvidence.model_validate(raw)
            except (TypeError, ValueError) as exc:
                raise FailureRecoveryObservationInvalid("Controller failure evidence is invalid") from exc
            if evidence.task_id and task_id and evidence.task_id != task_id:
                raise FailureRecoveryStale("Controller failure evidence task binding is stale")
            if evidence.logical_tool_id and evidence.logical_tool_id not in {task_id, logical_tool_id}:
                raise FailureRecoveryStale("Controller failure evidence tool binding is stale")
            if evidence.task_id and evidence.logical_tool_id:
                return evidence
            payload = evidence.model_dump(mode="json")
            payload.update(
                {
                    "task_id": evidence.task_id or task_id,
                    "logical_tool_id": evidence.logical_tool_id or logical_tool_id,
                    # These identities are recomputed by the typed model;
                    # retaining the old digest after filling bindings would
                    # turn a valid server projection into a stale one.
                    "evidence_id": "",
                    "evidence_digest": "",
                }
            )
            return AgentTaskFailureEvidence.model_validate(payload)
        receipt = getattr(snapshot, "receipt", None)
        try:
            return failure_evidence_from_controller(
                receipt=receipt,
                inspection=inspection,
                task_id=task_id,
                logical_tool_id=logical_tool_id,
            )
        except (TypeError, ValueError) as exc:
            raise FailureRecoveryObservationInvalid(
                "Controller has no typed failure evidence for recovery"
            ) from exc

    def _plan_inputs(self, *, snapshot: Any) -> tuple[str, str, dict[str, Any], str]:
        execution = snapshot.execution
        inspection = snapshot.inspection
        task_id = str(inspection.current_task_id or "")
        if not task_id:
            raise FailureRecoveryObservationInvalid("FAILED Controller has no current task")
        slot = None
        if inspection.current_task_index is not None:
            try:
                slot = execution.task_slots[inspection.current_task_index]
            except (IndexError, TypeError):
                raise FailureRecoveryStale("FAILED Controller task slot is stale") from None
        if slot is None or slot.task_id != task_id:
            raise FailureRecoveryStale("FAILED Controller task slot binding is stale")
        logical_tool_id = task_id
        if self.registry is not None:
            try:
                logical_tool_id = str(getattr(self.registry.get(task_id), "scientific_tool_id", "") or task_id)
            except (ValueError, TypeError):
                logical_tool_id = task_id
        arguments: dict[str, Any] = {}
        proposal_store = self.proposal_store
        authorization_service = self.authorization_service
        if proposal_store is not None and authorization_service is not None:
            try:
                authorization = authorization_service.verify_authorization(
                    project_id=execution.project_id,
                    authorization_id=execution.authorization_id,
                    verify_current=False,
                )
                raw = authorization.compiled_task_options.get(task_id, {})
                if isinstance(raw, Mapping):
                    arguments = {str(key): value for key, value in raw.items()}
            except Exception as exc:
                raise FailureRecoveryStale("current authorization options are unavailable") from exc
        return task_id, logical_tool_id, arguments, str(slot.input_artifacts_digest or "")

    def _catalog_for_task(self, *, task_id: str, logical_tool_id: str, evidence: AgentTaskFailureEvidence) -> tuple[list[str], dict[str, Mapping[str, Any]], dict[str, SemanticBoundary]]:
        ids: set[str] = {logical_tool_id}
        ids.update(str(item) for item in evidence.safe_alternative_tool_ids if str(item))
        schemas: dict[str, Mapping[str, Any]] = {}
        boundaries: dict[str, SemanticBoundary] = {}
        if self.registry is not None:
            for candidate in self.registry.list_tasks():
                candidate_id = str(getattr(candidate, "scientific_tool_id", "") or "")
                if not candidate_id or candidate_id not in ids:
                    continue
                route = getattr(candidate, "execution_route", None)
                if route not in {None, "local_executor"}:
                    continue
                schemas[candidate_id] = dict(getattr(candidate, "option_schema", {}) or {})
                boundaries[candidate_id] = (
                    SemanticBoundary.SCIENTIFIC_CONFIRMATION
                    if getattr(candidate, "gates", ())
                    else SemanticBoundary.NONE
                )
        return sorted(ids), schemas, boundaries

    def _totals(self, *, project_id: str, session_id: str, grant: AutonomyGrant, authority_epoch: str) -> tuple[int, int]:
        provider_calls = 0
        effects = 0
        try:
            receipts = self.store.list_receipts(project_id=project_id)
        except FileNotFoundError:
            return 0, 0
        for receipt in receipts:
            if (
                receipt.session_id == session_id
                and receipt.autonomy_grant_id == grant.grant_id
                and receipt.autonomy_grant_digest == grant.grant_digest
                and receipt.authority_epoch == authority_epoch
            ):
                provider_calls += int(receipt.provider_call_count)
                effects += int(receipt.effect_started)
        return provider_calls, effects

    def continue_failed(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        state: Mapping[str, Any] | None,
        controller_result: Any,
        provider: LLMProvider | None = None,
    ) -> FailureRecoveryRuntimeResult:
        """Inspect, observe, and invoke recovery at most once."""

        try:
            snapshot = self._current_snapshot(controller_result=controller_result)
        except FailureRecoveryDecisionInvalid as exc:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                reason_code="RECOVERY_CONTROLLER_VERIFIER_REQUIRED",
                question="当前失败状态无法由 Controller 验证，已停止自动恢复。",
            )
        except (FailureRecoveryConflict, OSError, ValueError) as exc:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                reason_code="RECOVERY_CONTROLLER_STATE_STALE",
                question="当前失败状态已发生变化，需要重新检查后再继续。",
            )
        except Exception:
            # Controller implementations are server extensions.  An
            # untyped reader failure is still an unverifiable authority
            # boundary; do not leak its details or continue automatically.
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                reason_code="RECOVERY_CONTROLLER_STATE_STALE",
                question="当前失败状态无法完成安全验证，需要重新检查后再继续。",
            )
        supplied_execution = getattr(controller_result, "execution", None)
        supplied_inspection = getattr(controller_result, "inspection", None)
        if supplied_execution is None or supplied_inspection is None:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                reason_code="RECOVERY_CONTROLLER_STATE_STALE",
                question="当前失败状态已发生变化，需要重新检查后再继续。",
            )
        if (
            snapshot.execution.project_id != project_id
            or snapshot.execution.run_id != run_id
            or snapshot.execution.controller_execution_id
            != getattr(supplied_execution, "controller_execution_id", "")
            or snapshot.execution.execution_digest
            != getattr(supplied_execution, "execution_digest", "")
        ):
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                reason_code="RECOVERY_CONTROLLER_STATE_STALE",
                question="当前失败状态已发生变化，需要重新检查后再继续。",
            )
        inspection = snapshot.inspection
        expected_inspection_digest = str(
            getattr(supplied_inspection, "inspection_digest", "") or ""
        )
        current_inspection_digest = str(
            getattr(inspection, "inspection_digest", "") or ""
        )
        if (
            not expected_inspection_digest
            or not current_inspection_digest
            or current_inspection_digest != expected_inspection_digest
        ):
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                reason_code="RECOVERY_CONTROLLER_STATE_STALE",
                question="当前失败状态已发生变化，需要重新检查后再继续。",
            )
        status_value = getattr(inspection, "status", None)
        status = getattr(status_value, "value", status_value)
        if str(status).lower() not in {"failed", "recovery_required"}:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                reason_code="RECOVERY_NOT_CURRENT_FAILED",
            )
        next_action_value = getattr(inspection, "next_action", None)
        next_action = getattr(next_action_value, "value", next_action_value)
        if str(next_action).lower() in {
            "prepare_local_gate",
            "wait_for_gate",
            "stop_gate_rejected",
            "prepare_remote_request",
            "wait_for_remote_approval",
            "dispatch_remote_task",
            "stop_remote_rejected",
        }:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.REQUIRE_HUMAN,
                controller_result=snapshot,
                reason_code="RECOVERY_AUTHORITY_BOUNDARY",
                question="当前失败位于 Gate 或远程 authority 边界，需要显式处理。",
            )
        session_id = self.session_id(conversation_id=conversation_id, run_id=run_id)
        try:
            binding = self._grant_binding(project_id=project_id, run_id=run_id, session_id=session_id)
            existing = self._existing_observation(
                project_id=project_id,
                state=state,
                execution=snapshot.execution,
                inspection=inspection,
            )
        except FailureRecoveryStale:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                session_id=session_id,
                reason_code="RECOVERY_AUTONOMY_GRANT_STALE",
                question="当前 recovery authority 已过期，需要重新检查。",
            )
        except (FailureRecoveryConflict, FailureRecoveryObservationInvalid, ValueError, TypeError):
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.REQUIRE_HUMAN,
                controller_result=snapshot,
                session_id=session_id,
                reason_code="RECOVERY_AUTONOMY_GRANT_REQUIRED",
                question="当前失败没有可验证的 server-issued AutonomyGrant，已停止自动恢复。",
            )
        except Exception:
            # A malformed or unavailable server authority source is not a
            # caller-controlled recovery boundary.  Keep it fail-closed and
            # avoid exposing extension errors or crossing a provider/effect
            # boundary while the authority state is unverifiable.
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                controller_result=snapshot,
                session_id=session_id,
                reason_code="RECOVERY_AUTONOMY_GRANT_UNAVAILABLE",
                question="当前 recovery authority 无法完成安全验证，需要重新检查后再继续。",
            )
        if binding is None:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.REQUIRE_HUMAN,
                observation=existing,
                controller_result=snapshot,
                session_id=session_id,
                reason_code="RECOVERY_AUTONOMY_GRANT_REQUIRED",
                failure_class=existing.failure_class if existing else None,
                effect_certainty=existing.effect_certainty if existing else None,
                question="当前失败没有可验证的 server-issued AutonomyGrant，已停止自动恢复。",
            )
        grant = binding.grant
        authority_epoch = binding.authority_epoch or self.authority_epoch_for_grant(grant)
        if existing is not None and existing.authority_digest and existing.authority_digest != grant.grant_digest:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=existing,
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_GRANT_REPLACED",
                failure_class=existing.failure_class,
                effect_certainty=existing.effect_certainty,
                question="恢复所依据的 authority 已被替换，旧失败不能自动套用新授权。",
            )
        try:
            evidence = existing and AgentTaskFailureEvidence(
                failure_code="persisted_failure_observation",
                failure_class=existing.failure_class,
                effect_certainty=existing.effect_certainty,
                task_id=existing.task_id,
                logical_tool_id=existing.logical_tool_id,
                reason_codes=existing.reason_codes,
                source_receipt_ids=existing.source_receipt_ids,
                source_receipt_digests=existing.source_receipt_digests,
            )
            if evidence is None:
                evidence = self._typed_evidence(snapshot=snapshot)
            task_id, logical_tool_id, arguments, input_digest = self._plan_inputs(snapshot=snapshot)
            tool_ids, schemas, boundaries = self._catalog_for_task(
                task_id=task_id,
                logical_tool_id=logical_tool_id,
                evidence=evidence,
            )
            # Resolve the provider only after the locked, digest-verified
            # FAILED snapshot and current grant are known.  In production the
            # resolver is server-owned; the ordinary Conversation provider is
            # never used when that resolver is installed.
            with self._provider_context(fallback_provider=provider) as recovery_provider:
                service = self.service_factory.build(
                    provider=recovery_provider,
                    grant=grant,
                    session_id=session_id,
                    authority_epoch=authority_epoch,
                )
                # The factory normally carries the complete reviewed catalog.
                # A test/server factory may intentionally leave it empty; fill
                # it before the immutable observation is built so its catalog
                # digest remains stable on replay.
                if hasattr(service, "tool_schemas") and not getattr(service, "tool_schemas", {}):
                    service.tool_schemas = dict(schemas)
                if hasattr(service, "tool_semantic_boundaries") and not getattr(service, "tool_semantic_boundaries", {}):
                    service.tool_semantic_boundaries = dict(boundaries)
                observation = existing or service.observe_failure(
                    project_id=project_id,
                    run_id=run_id,
                    controller_execution_id=snapshot.execution.controller_execution_id,
                    controller_execution_digest=snapshot.execution.execution_digest,
                    inspection_digest=inspection.inspection_digest,
                    task_id=task_id,
                    logical_tool_id=logical_tool_id,
                    evidence=evidence,
                    arguments=arguments,
                    input_artifact_digest=input_digest,
                    authority_digest=grant.grant_digest,
                    available_recovery_tools=tool_ids,
                    grant=grant,
                    session_id=session_id,
                    authority_epoch=authority_epoch,
                )
                request_id = "conversation-recovery-request-" + _agent_digest(
                    {
                        "failure_digest": observation.failure_digest,
                        "session_id": session_id,
                        "authority_epoch": authority_epoch,
                    }
                ).split(":", 1)[1][:40]
                recovery = service.recover_failure(
                    observation=observation,
                    grant=grant,
                    session_id=session_id,
                    authority_epoch=authority_epoch,
                    client_request_id=request_id,
                )
        except FailureRecoveryProviderOutcomeUnknown:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=locals().get("observation"),
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_PROVIDER_OUTCOME_UNKNOWN",
                failure_class=evidence.failure_class if "evidence" in locals() else None,
                effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
                question="恢复 provider 的结果无法确认，已停止自动重试。",
            )
        except FailureRecoveryEffectUnknown:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=locals().get("observation"),
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_EFFECT_OUTCOME_UNKNOWN",
                failure_class=evidence.failure_class if "evidence" in locals() else None,
                effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
                question="恢复 effect 的结果无法确认，已停止自动重试。",
            )
        except FailureRecoveryStale:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=locals().get("observation") or existing,
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_AUTHORITY_STALE",
                failure_class=evidence.failure_class if "evidence" in locals() else None,
                effect_certainty=evidence.effect_certainty if "evidence" in locals() else None,
                question="恢复所依据的 Controller 或 authority 已过期，需要重新检查。",
            )
        except (FailureRecoveryObservationInvalid, FailureRecoveryDecisionInvalid, FailureRecoveryConflict, OSError, ValueError):
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=locals().get("observation") or existing,
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_EVIDENCE_UNAVAILABLE",
                failure_class=evidence.failure_class if "evidence" in locals() else None,
                effect_certainty=evidence.effect_certainty if "evidence" in locals() else None,
                question="当前失败缺少可验证的 typed evidence，已停止自动恢复。",
            )
        except Exception:
            # Production dependencies are server extensions, but a recovery
            # continuation must never turn an unexpected extension failure
            # into an unbounded retry or an HTTP 500 that invites a caller to
            # repeat an effect.  The durable foundation checkpoints remain
            # available for a later authoritative reconciliation.
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=locals().get("observation") or existing,
                controller_result=snapshot,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_EVIDENCE_UNAVAILABLE",
                failure_class=evidence.failure_class if "evidence" in locals() else None,
                effect_certainty=evidence.effect_certainty if "evidence" in locals() else None,
                question="当前失败缺少可验证的 recovery evidence，已停止自动恢复。",
            )
        successor_id = str(recovery.receipt.successor_controller_execution_id or "")
        next_controller = snapshot
        if successor_id:
            try:
                next_controller = self.controller.get(
                    project_id=project_id,
                    controller_execution_id=successor_id,
                )
            except Exception:
                return FailureRecoveryRuntimeResult(
                    FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                    observation=recovery.observation,
                    recovery=recovery,
                    controller_result=snapshot,
                    session_id=session_id,
                    authority_epoch=authority_epoch,
                    reason_code="RECOVERY_SUCCESSOR_STATE_UNAVAILABLE",
                    failure_class=recovery.observation.failure_class,
                    effect_certainty=recovery.observation.effect_certainty,
                    question="恢复 successor 已提交，但当前 Controller 状态无法验证。",
                )
        try:
            total_provider_calls, total_effects = self._totals(
                project_id=project_id,
                session_id=session_id,
                grant=grant,
                authority_epoch=authority_epoch,
            )
        except Exception:
            return FailureRecoveryRuntimeResult(
                FailureRecoveryRuntimeEligibility.FAIL_CLOSED,
                observation=recovery.observation,
                recovery=recovery,
                controller_result=next_controller,
                session_id=session_id,
                authority_epoch=authority_epoch,
                reason_code="RECOVERY_RECEIPT_EVIDENCE_UNAVAILABLE",
                failure_class=recovery.observation.failure_class,
                effect_certainty=recovery.observation.effect_certainty,
                question="恢复 receipt 的 aggregate evidence 无法验证，已停止自动继续。",
            )
        question = ""
        if recovery.decision.recovery_action.value == "ASK_USER":
            if recovery.observation.effect_certainty is AgentEffectCertainty.EFFECT_UNKNOWN:
                question = "effect 结果未知，需要显式的人类处理；Molly 不会自动重试。"
            elif recovery.budget.retries_remaining <= 0 and recovery.observation.failure_class in {
                AgentFailureClass.TRANSIENT,
                AgentFailureClass.PARAMETER_RECOVERABLE,
                AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE,
            }:
                question = "恢复 retry 预算已耗尽，未执行新的 provider 或 effect。"
            elif recovery.budget.replans_remaining <= 0 and recovery.observation.failure_class is AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT:
                question = "恢复 replan 预算已耗尽，未执行新的 provider 或 effect。"
            else:
                question = "当前失败需要显式的人类恢复决定；Molly 不会自动继续。"
        return FailureRecoveryRuntimeResult(
            FailureRecoveryRuntimeEligibility.ELIGIBLE,
            observation=recovery.observation,
            recovery=recovery,
            controller_result=next_controller,
            session_id=session_id,
            authority_epoch=authority_epoch,
            reason_code="RECOVERY_COMMITTED",
            failure_class=recovery.observation.failure_class,
            effect_certainty=recovery.observation.effect_certainty,
            provider_calls_total=total_provider_calls,
            effect_count_total=total_effects,
            question=question,
        )


__all__ = [
    "FailureRecoveryRuntimeEligibility",
    "FailureRecoveryRuntimeResult",
    "ScientificAgentAutonomyGrantBinding",
    "ScientificAgentAutonomyGrantIssuer",
    "ScientificAgentAutonomyGrantStore",
    "ScientificAgentFailureRecoveryRuntime",
    "ScientificAgentFailureRecoveryServiceFactory",
]
