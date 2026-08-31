"""Thin host-owned assembly service for Core v2 runtime operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from molly.core.agent_loop import RUN_STARTED, AgentLoop
from molly.core.approvals import ApprovalDecision, ApprovalRecord
from molly.core.artifacts import ArtifactRecord, ArtifactStore
from molly.core.inspection import RunInspector
from molly.core.ids import (
    new_server_id,
    normalize_timestamp,
    utc_timestamp,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
)
from molly.core.ledger import LedgerEvent, RunLedger
from molly.core.lineage import ArtifactLineage
from molly.core.reviews import ReviewDecision, ReviewRecord
from molly.core.runs import RunBudget, RunRequest, RunResult

from .errors import RuntimeBindingError, RuntimeProfileUnavailable, RuntimeStateError
from .profiles import RuntimeProfile, RuntimeProfileRegistry


class _UnavailableDecisionProvider:
    def next_action(self, context: Any, model_visible_tools: Any) -> Any:
        raise RuntimeProfileUnavailable(
            "RUNTIME_PROFILE_UNAVAILABLE: DecisionProvider is required for this turn"
        )


@dataclass(frozen=True, slots=True)
class ApprovalOutcome:
    approval: ApprovalRecord
    result: RunResult

    def to_dict(self) -> dict[str, Any]:
        return {"approval": self.approval.to_dict(), "result": self.result.to_dict()}


@dataclass(frozen=True, slots=True)
class ReviewPublication:
    review: ReviewRecord
    review_artifact: ArtifactRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_record_artifact_id": self.review_artifact.artifact_id,
            "review_digest": self.review.digest,
            "target_artifact_id": self.review.artifact_id,
            "decision": self.review.decision,
            "review": self.review.to_dict(),
        }


class RuntimeService:
    """Delegate runtime operations to existing AgentLoop and Core stores."""

    def __init__(
        self,
        root: Path | str,
        *,
        profiles: RuntimeProfileRegistry,
        clock: Callable[[], str] = utc_timestamp,
    ) -> None:
        if not isinstance(profiles, RuntimeProfileRegistry):
            raise RuntimeStateError("RuntimeService requires a RuntimeProfileRegistry")
        if not callable(clock):
            raise RuntimeStateError("RuntimeService clock must be callable")
        configured = Path(root)
        if configured.is_symlink():
            raise RuntimeStateError("runtime state root cannot be a symlink")
        self.root = configured.absolute()
        self.profiles = profiles
        self.clock = clock

    def _ensure_root(self, *, create: bool) -> None:
        if self.root.is_symlink():
            raise RuntimeStateError("runtime state root cannot be a symlink")
        if not self.root.exists():
            if not create:
                raise RuntimeStateError("runtime state is unavailable")
            self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise RuntimeStateError("runtime state root is not a regular directory")

    def _open_components(self, *, create: bool) -> tuple[ArtifactStore, RunLedger, ArtifactLineage]:
        self._ensure_root(create=create)
        artifact_root = self.root / "artifacts"
        if not create:
            required = (artifact_root, artifact_root / "objects", artifact_root / "metadata")
            if any(not item.exists() or item.is_symlink() or not item.is_dir() for item in required):
                raise RuntimeStateError("runtime artifact state is unavailable or unsafe")
        store = ArtifactStore(artifact_root)
        return store, RunLedger(self.root / "events.jsonl"), ArtifactLineage(self.root / "lineage.jsonl")

    @staticmethod
    def _request_from_events(events: tuple[LedgerEvent, ...]) -> RunRequest:
        if not events or events[0].event_type != RUN_STARTED:
            raise RuntimeBindingError("run has no immutable RUN_STARTED request")
        start = events[0]
        raw = start.metadata.get("request")
        if not isinstance(raw, Mapping):
            raise RuntimeBindingError("RUN_STARTED request binding is malformed")
        try:
            request = RunRequest.from_dict(raw)
            digest = validate_digest_reference(
                str(start.metadata.get("request_digest", "")), field="request_digest"
            )
        except Exception as exc:
            raise RuntimeBindingError("RUN_STARTED request binding is malformed") from exc
        if request.run_id != start.run_id or request.request_sha256 != digest:
            raise RuntimeBindingError("RUN_STARTED request binding is inconsistent")
        return request

    def _request_for_run(self, run_id: str, *, create: bool = False) -> tuple[RunRequest, ArtifactStore, RunLedger, ArtifactLineage]:
        store, ledger, lineage = self._open_components(create=create)
        events = ledger.for_run(run_id)
        request = self._request_from_events(events)
        return request, store, ledger, lineage

    def _profile_for_request(self, request: RunRequest) -> RuntimeProfile:
        profile_id = request.metadata.get("runtime_profile_ref")
        profile_digest = request.metadata.get("runtime_profile_digest")
        if not isinstance(profile_id, str) or not isinstance(profile_digest, str):
            raise RuntimeProfileUnavailable(
                "RUNTIME_PROFILE_UNAVAILABLE: run has no complete runtime profile binding"
            )
        return self.profiles.resolve(profile_id, expected_digest=profile_digest)

    def _loop(
        self,
        *,
        request: RunRequest,
        store: ArtifactStore,
        ledger: RunLedger,
        lineage: ArtifactLineage,
        profile: RuntimeProfile,
        require_provider: bool,
    ) -> AgentLoop:
        registry = profile.create_registry()
        policy = profile.create_policy()
        if policy.digest != request.tool_policy_digest:
            raise RuntimeBindingError("runtime profile ToolPolicy does not match the immutable request")
        if require_provider:
            provider = profile.create_decision_provider()
        else:
            provider = (
                profile.create_decision_provider()
                if profile.decision_provider_factory is not None
                else _UnavailableDecisionProvider()
            )
        return AgentLoop(
            store=store,
            ledger=ledger,
            lineage=lineage,
            registry=registry,
            policy=policy,
            decision_provider=provider,
            clock=self.clock,
        )

    def start_run(
        self,
        *,
        profile_id: str,
        goal: str,
        input_artifact_ids: tuple[str, ...] = (),
        budget: RunBudget | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunResult:
        profile = self.profiles.resolve(profile_id)
        store, ledger, lineage = self._open_components(create=True)
        policy = profile.create_policy()
        registry = profile.create_registry()
        provider = profile.create_decision_provider()
        supplied_metadata = {} if metadata is None else dict(metadata)
        protected = {"runtime_profile_ref", "runtime_profile_digest"}
        if protected & set(supplied_metadata):
            raise RuntimeBindingError("runtime profile binding is server-owned")
        supplied_metadata.update(
            {
                "runtime_profile_ref": profile.profile_id,
                "runtime_profile_digest": profile.digest,
            }
        )
        request = RunRequest.create(
            goal=goal,
            tool_policy_digest=policy.digest,
            budget=budget if budget is not None else RunBudget(),
            input_artifact_ids=input_artifact_ids,
            metadata=supplied_metadata,
            created_at=normalize_timestamp(self.clock(), field="request created_at"),
        )
        loop = AgentLoop(
            store=store,
            ledger=ledger,
            lineage=lineage,
            registry=registry,
            policy=policy,
            decision_provider=provider,
            clock=self.clock,
        )
        return loop.run(request)

    def resume_run(self, run_id: str) -> RunResult:
        request, store, ledger, lineage = self._request_for_run(run_id)
        profile = self._profile_for_request(request)
        loop = self._loop(
            request=request,
            store=store,
            ledger=ledger,
            lineage=lineage,
            profile=profile,
            require_provider=True,
        )
        return loop.run(request)

    def inspect_run(self, run_id: str):
        store, ledger, lineage = self._open_components(create=False)
        return RunInspector(store=store, ledger=ledger, lineage=lineage).inspect_run(run_id)

    def inspect_artifact(self, artifact_id: str):
        store, ledger, lineage = self._open_components(create=False)
        return RunInspector(store=store, ledger=ledger, lineage=lineage).inspect_artifact(artifact_id)

    def record_approval(
        self,
        run_id: str,
        *,
        decision: str | ApprovalDecision,
        reviewer_ref: str,
        call_id: str | None = None,
    ) -> ApprovalOutcome:
        request, store, ledger, lineage = self._request_for_run(run_id)
        inspection = RunInspector(store=store, ledger=ledger, lineage=lineage).inspect_run(run_id)
        if inspection.pending_call is None:
            raise RuntimeBindingError("run has no exact pending approval call")
        call = self._call_from_pending(inspection.pending_call)
        if call_id is not None:
            validate_identifier(call_id, field="call_id")
            if call.call_id != call_id:
                raise RuntimeBindingError("requested call_id is not the pending exact call")
        approval = ApprovalRecord.for_call(
            call,
            decision=decision,
            reviewer_ref=reviewer_ref,
            created_at=normalize_timestamp(self.clock(), field="approval created_at"),
        )
        profile = self._profile_for_request(request)
        loop = self._loop(
            request=request,
            store=store,
            ledger=ledger,
            lineage=lineage,
            profile=profile,
            require_provider=False,
        )
        result = loop.run(request, approval=approval)
        return ApprovalOutcome(approval=approval, result=result)

    @staticmethod
    def _call_from_pending(value: Mapping[str, Any]):
        from molly.core.tools import MaterializedToolCall

        try:
            return MaterializedToolCall.from_dict(value)
        except Exception as exc:
            raise RuntimeBindingError("pending materialized call is malformed") from exc

    def create_review(
        self,
        artifact_id: str,
        *,
        decision: str | ReviewDecision,
        reviewer_ref: str,
        reason: str = "",
    ) -> ReviewPublication:
        store, _, _ = self._open_components(create=False)
        artifact = store.verify(validate_artifact_id(artifact_id))
        review = ReviewRecord.for_artifact(
            artifact,
            review_id=new_server_id("review"),
            decision=decision,
            reviewer=reviewer_ref,
            reason=reason,
            created_at=normalize_timestamp(self.clock(), field="review created_at"),
        )
        review_artifact = store.put_json(
            review.to_dict(),
            schema_name="molly.core.review-record",
            schema_version="1",
        )
        return ReviewPublication(review=review, review_artifact=review_artifact)

    def observe_run(self, run_id: str, exporter: Any):
        from molly.observability import ObservationService

        store, ledger, lineage = self._open_components(create=False)
        inspector = RunInspector(store=store, ledger=ledger, lineage=lineage)
        return ObservationService(inspector).export_run(run_id, exporter)


__all__ = ["ApprovalOutcome", "ReviewPublication", "RuntimeService"]
