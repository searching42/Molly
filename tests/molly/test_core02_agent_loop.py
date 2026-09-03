"""Focused CORE-02 tests for one bounded AgentLoop authority."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from molly.core import (
    ActionError,
    AgentLoop,
    ApprovalDecision,
    ApprovalError,
    ApprovalRecord,
    ArtifactDraft,
    ArtifactLineage,
    ArtifactStore,
    ArtifactIntegrityError,
    MAX_TOOL_RESULT_DATA_BYTES,
    MaterializedToolCall,
    RelationType,
    RequestReviewAction,
    RunInspector,
    RunBindingError,
    RunLedger,
    RunRequest,
    RunStateError,
    RunStatus,
    SchemaValidationError,
    SideEffectClass,
    StopAction,
    ToolAccessError,
    ToolCallProposal,
    ToolContractError,
    ToolPolicy,
    ToolPolicyError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.core.agent_loop import (
    APPROVAL_RECORDED,
    APPROVAL_REQUIRED,
    BUDGET_EXHAUSTED,
    DECISION_RECORDED,
    TOOL_CALL_MATERIALIZED,
    TOOL_CALL_REJECTED,
    TOOL_EXECUTION_FAILED,
    TOOL_EXECUTION_STARTED,
    TOOL_EXECUTION_SUCCEEDED,
)
from molly.core.ids import canonical_json_bytes, sha256_bytes, utc_timestamp


pytestmark = pytest.mark.unit


class ScriptedProvider:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)
        self.calls = 0
        self.contexts = []
        self.tool_views = []

    def next_action(self, context, model_visible_tools):
        self.calls += 1
        self.contexts.append(context)
        self.tool_views.append(model_visible_tools)
        if not self.actions:
            raise StopIteration
        return self.actions.pop(0)


class EndlessProvider:
    def __init__(self) -> None:
        self.calls = 0

    def next_action(self, context, model_visible_tools):
        self.calls += 1
        return ToolCallProposal("emit", {"value": self.calls})


def _spec(*, name: str = "emit", version: str = "1", approval: bool = False) -> ToolSpec:
    return ToolSpec(
        name=name,
        version=version,
        description="deterministic local fixture tool",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        side_effect_class=SideEffectClass.PURE,
        requires_approval=approval,
    )


def _policy(spec: ToolSpec, *, approval_class: bool = False) -> ToolPolicy:
    return ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(spec.side_effect_class,),
        approval_required_side_effect_classes=(
            (spec.side_effect_class,) if approval_class else ()
        ),
    )


def _environment(
    tmp_path: Path,
    *,
    spec: ToolSpec | None = None,
    policy: ToolPolicy | None = None,
    provider: ScriptedProvider | None = None,
    executor=None,
):
    spec = spec or _spec()
    policy = policy or _policy(spec)
    provider = provider or ScriptedProvider()
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    registry = ToolRegistry()
    if executor is None:
        executor = lambda context: ToolResult(
            data={"value": 7},
            artifacts=(ArtifactDraft(b"deterministic-output", "text/plain"),),
        )
    registry.register(spec, executor)
    loop = AgentLoop(
        store=store,
        ledger=ledger,
        lineage=lineage,
        registry=registry,
        policy=policy,
        decision_provider=provider,
    )
    request = RunRequest.create(
        goal="run deterministic CORE-02 fixture",
        tool_policy_digest=policy.digest,
    )
    return loop, request, store, ledger, lineage, registry, spec, policy


def _events(ledger: RunLedger, event_type: str):
    return [event for event in ledger.events if event.event_type == event_type]


def test_run_request_is_server_owned_digest_bound_and_restartable(tmp_path: Path) -> None:
    provider = ScriptedProvider(StopAction("done"))
    loop, request, _, ledger, _, _, _, policy = _environment(tmp_path, provider=provider)
    first = loop.run(request)

    assert first.status == RunStatus.STOPPED.value
    assert request.run_id.startswith("run_")
    assert request.request_sha256 == RunRequest.from_dict(request.to_dict()).request_sha256
    assert _events(ledger, "RUN_STARTED")[0].metadata["request_digest"] == request.digest

    resumed_provider = ScriptedProvider(StopAction("must not run"))
    resumed = AgentLoop(
        store=loop.store,
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(loop.lineage.path),
        registry=loop.registry,
        policy=policy,
        decision_provider=resumed_provider,
    ).run(request)
    assert resumed.status == RunStatus.STOPPED.value
    assert resumed_provider.calls == 0

    changed = RunRequest(
        run_id=request.run_id,
        goal="changed goal",
        input_artifact_ids=request.input_artifact_ids,
        tool_policy_digest=request.tool_policy_digest,
        created_at=request.created_at,
        metadata=request.metadata,
    )
    with pytest.raises(RunBindingError):
        loop.run(changed)

    other_policy = ToolPolicy(allowed_tools=(), allowed_side_effect_classes=())
    mismatched = RunRequest(
        run_id=request.run_id,
        goal=request.goal,
        input_artifact_ids=request.input_artifact_ids,
        tool_policy_digest=other_policy.digest,
        created_at=request.created_at,
        metadata=request.metadata,
    )
    with pytest.raises(RunBindingError):
        loop.run(mismatched)


def test_registry_is_closed_and_model_view_is_sanitized() -> None:
    spec = _spec()
    registry = ToolRegistry()
    executor = lambda context: ToolResult({"value": 1})
    registry.register(spec, executor)

    assert registry.resolve("emit") == spec
    with pytest.raises(ToolContractError):
        registry.resolve("unknown")
    with pytest.raises(ToolContractError):
        registry.register(spec, executor)
    view = registry.model_visible_tools()[0]
    assert set(view) == {"name", "description", "input_schema"}
    assert "executor" not in view
    assert "module" not in view
    assert "filesystem_path" not in view


def test_tool_spec_schema_and_policy_contracts_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ToolContractError):
        ToolSpec(
            name="bad_schema",
            description="bad",
            input_schema={"$ref": "https://example.invalid/schema"},
            output_schema={},
        )

    spec = _spec()
    policy = _policy(spec)
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": "not-an-integer"}))
    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path, spec=spec, policy=policy, provider=provider
    )
    with pytest.raises(SchemaValidationError):
        loop.run(request)
    assert not _events(ledger, TOOL_CALL_MATERIALIZED)

    denied_policy = ToolPolicy(allowed_tools=(), allowed_side_effect_classes=())
    denied_provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    denied_loop, denied_request, _, denied_ledger, _, _, _, _ = _environment(
        tmp_path / "denied",
        spec=spec,
        policy=denied_policy,
        provider=denied_provider,
    )
    with pytest.raises(ToolPolicyError):
        denied_loop.run(denied_request)
    assert not _events(denied_ledger, TOOL_CALL_MATERIALIZED)


def test_invalid_tool_output_is_not_a_success_or_publication(tmp_path: Path) -> None:
    calls = 0

    def invalid_output(context):
        nonlocal calls
        calls += 1
        return ToolResult({"value": "invalid"})

    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, store, ledger, lineage, _, _, _ = _environment(
        tmp_path, provider=provider, executor=invalid_output
    )
    result = loop.run(request)
    assert result.status == RunStatus.ACTIVE.value
    assert calls == 1
    assert len(_events(ledger, TOOL_EXECUTION_STARTED)) == 1
    assert len(_events(ledger, TOOL_EXECUTION_FAILED)) == 1
    assert not _events(ledger, TOOL_EXECUTION_SUCCEEDED)
    assert lineage.relations == ()
    assert tuple((tmp_path / "artifacts" / "objects").rglob("*")) == ()


def test_server_hard_limit_stops_an_endless_provider(tmp_path: Path) -> None:
    provider = EndlessProvider()
    loop, request, _, ledger, _, _, _, _ = _environment(tmp_path, provider=provider)

    result = loop.run(request)

    assert result.status == RunStatus.BUDGET_EXHAUSTED.value
    assert provider.calls == 8
    assert len(_events(ledger, TOOL_CALL_MATERIALIZED)) == 8
    assert len(_events(ledger, BUDGET_EXHAUSTED)) == 1
    assert _events(ledger, BUDGET_EXHAUSTED)[0].metadata["server_limits"] == {
        "max_decisions": 12,
        "max_tool_calls": 8,
        "max_steps": 8,
    }


def test_legacy_v2_request_and_terminal_are_read_without_digest_drift(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    policy = ToolPolicy(allowed_tools=(), allowed_side_effect_classes=())
    raw_request = {
        "run_id": "run_legacy_v2",
        "goal": "legacy request",
        "input_artifact_ids": [],
        "tool_policy_digest": policy.digest,
        "budget": {"max_decisions": 12, "max_tool_calls": 8, "max_steps": 8},
        "created_at": utc_timestamp(),
        "metadata": {},
    }
    request_digest = sha256_bytes(canonical_json_bytes(raw_request))
    ledger.append(
        event_id="evt_legacy_start",
        run_id=raw_request["run_id"],
        event_type="RUN_STARTED",
        status="STARTED",
        timestamp=raw_request["created_at"],
        metadata={
            "request": raw_request,
            "request_digest": request_digest,
            "policy_digest": policy.digest,
            "initial_artifact_ids": [],
        },
    )
    ledger.append(
        event_id="evt_legacy_exhausted",
        run_id=raw_request["run_id"],
        event_type=BUDGET_EXHAUSTED,
        status="EXHAUSTED",
        timestamp=utc_timestamp(),
    )

    reconstructed = RunRequest.from_dict(raw_request)
    assert reconstructed.request_sha256 == request_digest
    inspector = RunInspector(store=store, ledger=ledger, lineage=lineage)
    assert inspector.inspect_run(raw_request["run_id"]).status == RunStatus.BUDGET_EXHAUSTED.value


def test_legacy_active_request_resume_uses_the_lower_persisted_limit(tmp_path: Path) -> None:
    provider = EndlessProvider()
    loop, _, _, ledger, _, _, _, policy = _environment(tmp_path, provider=provider)
    raw_request = {
        "run_id": "run_legacy_active",
        "goal": "legacy active request",
        "input_artifact_ids": [],
        "tool_policy_digest": policy.digest,
        "budget": {"max_decisions": 1, "max_tool_calls": 1, "max_steps": 1},
        "created_at": utc_timestamp(),
        "metadata": {},
    }
    request = RunRequest.from_dict(raw_request)
    ledger.append(
        event_id="evt_legacy_active_start",
        run_id=request.run_id,
        event_type="RUN_STARTED",
        status="STARTED",
        timestamp=request.created_at,
        metadata={
            "request": raw_request,
            "request_digest": request.request_sha256,
            "policy_digest": policy.digest,
            "initial_artifact_ids": [],
        },
    )
    ledger.append(
        event_id="evt_legacy_active_decision",
        run_id=request.run_id,
        event_type=DECISION_RECORDED,
        status="PROPOSED",
        timestamp=utc_timestamp(),
        metadata={"action": StopAction("historical decision").to_dict()},
    )

    result = loop.run(request)

    assert result.status == RunStatus.BUDGET_EXHAUSTED.value
    assert provider.calls == 0
    assert len(_events(ledger, TOOL_CALL_MATERIALIZED)) == 0
    assert _events(ledger, BUDGET_EXHAUSTED)[0].metadata["server_limits"] == {
        "max_decisions": 1,
        "max_tool_calls": 1,
        "max_steps": 1,
    }


def test_local_tool_execution_integrates_artifacts_ledger_and_lineage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    parent = store.put(b"parent", media_type="text/plain")
    provider = ScriptedProvider(
        ToolCallProposal("emit", {"value": 1}, input_artifact_ids=(parent.artifact_id,)),
        StopAction("done"),
    )

    def read_and_emit(context):
        assert not hasattr(context, "store")
        assert not hasattr(context, "root")
        assert context.read_artifact(parent.artifact_id) == b"parent"
        return ToolResult(
            {"value": 8},
            (ArtifactDraft(b"child", "text/plain"),),
        )

    spec = _spec()
    policy = _policy(spec)
    loop, request, _, ledger, lineage, _, _, _ = _environment(
        tmp_path,
        spec=spec,
        policy=policy,
        provider=provider,
        executor=read_and_emit,
    )
    request = RunRequest(
        run_id=request.run_id,
        goal=request.goal,
        input_artifact_ids=(parent.artifact_id,),
        tool_policy_digest=request.tool_policy_digest,
        created_at=request.created_at,
        metadata=request.metadata,
    )
    result = loop.run(request)
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    child_id = success.output_artifact_ids[0]
    assert result.status == RunStatus.STOPPED.value
    assert result.visible_artifact_ids == (parent.artifact_id, child_id)
    assert success.input_artifact_ids == (parent.artifact_id,)
    assert {item.relation_type for item in lineage.relations} == {
        RelationType.PRODUCED_BY.value,
        RelationType.DERIVED_FROM.value,
        RelationType.CONSUMED_BY.value,
    }
    assert lineage.producer_steps(child_id) == (success.step_id,)
    assert lineage.parents(child_id) == (parent.artifact_id,)


def test_materialized_arguments_reach_executor_and_bind_digest(tmp_path: Path) -> None:
    original_arguments = {"value": 6}
    proposal = ToolCallProposal("emit", original_arguments)
    original_arguments["value"] = 99
    observed_arguments = []

    def parameterized_executor(context):
        observed_arguments.append(context.arguments)
        assert context.arguments["value"] == 6
        with pytest.raises(TypeError):
            context.arguments["value"] = 7
        return ToolResult({"value": context.arguments["value"]})

    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path,
        provider=ScriptedProvider(proposal),
        executor=parameterized_executor,
    )
    assert loop.run(request).status == RunStatus.ACTIVE.value

    materialized = MaterializedToolCall.from_dict(
        _events(ledger, TOOL_CALL_MATERIALIZED)[0].metadata["materialized_call"]
    )
    assert materialized.arguments == {"value": 6}
    assert observed_arguments[0] == materialized.arguments
    changed = replace(materialized, arguments={"value": 7}, tool_call_digest=None)
    assert changed.digest != materialized.digest
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    assert success.metadata["result_data"] == {"value": 6}


def test_parameterized_output_changes_with_materialized_arguments(tmp_path: Path) -> None:
    spec = _spec()
    policy = _policy(spec)
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    registry = ToolRegistry()

    def multiply_executor(context):
        return ToolResult({"value": context.arguments["value"] * 2}, artifacts=())

    registry.register(spec, multiply_executor)
    for value in (6, 7):
        provider = ScriptedProvider(ToolCallProposal("emit", {"value": value}))
        loop = AgentLoop(
            store=store,
            ledger=ledger,
            lineage=lineage,
            registry=registry,
            policy=policy,
            decision_provider=provider,
        )
        request = RunRequest.create(
            goal=f"multiply {value}",
            tool_policy_digest=policy.digest,
        )
        assert loop.run(request).status == RunStatus.ACTIVE.value

    successes = _events(ledger, TOOL_EXECUTION_SUCCEEDED)
    assert [event.metadata["result_data"]["value"] for event in successes] == [12, 14]
    calls = [
        MaterializedToolCall.from_dict(event.metadata["materialized_call"])
        for event in _events(ledger, TOOL_CALL_MATERIALIZED)
    ]
    assert calls[0].digest != calls[1].digest


def test_tool_execution_context_rejects_undeclared_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.put(b"first", media_type="text/plain")
    second = store.put(b"second", media_type="text/plain")
    from molly.core import ToolExecutionContext

    context = ToolExecutionContext(
        run_id="run_context",
        step_id="step_context",
        call_id="call_context",
        idempotency_key="a" * 64,
        arguments={"value": 1},
        input_artifact_ids=(first.artifact_id,),
        reader=store.read,
    )
    assert context.arguments == {"value": 1}
    with pytest.raises(TypeError):
        context.arguments["value"] = 2
    assert context.read_artifact(first.artifact_id) == b"first"
    with pytest.raises(ToolAccessError):
        context.read_artifact(second.artifact_id)
    with pytest.raises(ToolAccessError):
        context.read_artifact("not-an-artifact-id")


def test_exact_approval_resume_reuses_persisted_call_without_provider_recreation(
    tmp_path: Path,
) -> None:
    spec = _spec(approval=True)
    policy = _policy(spec)
    first_provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, store, ledger, lineage, registry, _, _ = _environment(
        tmp_path, spec=spec, policy=policy, provider=first_provider
    )
    waiting = loop.run(request)
    assert waiting.status == RunStatus.WAITING_APPROVAL.value
    assert first_provider.calls == 1
    assert len(_events(ledger, APPROVAL_REQUIRED)) == 1
    call = MaterializedToolCall.from_dict(waiting.pending_call)
    approval = ApprovalRecord.for_call(
        call,
        decision=ApprovalDecision.APPROVED,
        reviewer_ref="reviewer-ref",
        created_at="2026-01-01T00:00:01Z",
    )

    resume_provider = ScriptedProvider()
    resumed_loop = AgentLoop(
        store=ArtifactStore(store.root),
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(lineage.path),
        registry=registry,
        policy=policy,
        decision_provider=resume_provider,
    )
    resumed = resumed_loop.run(request, approval=approval)
    assert resumed.status == RunStatus.ACTIVE.value
    assert resume_provider.calls == 0
    assert len(_events(ledger, APPROVAL_RECORDED)) == 1
    assert len(_events(ledger, TOOL_EXECUTION_SUCCEEDED)) == 1
    assert len(_events(ledger, DECISION_RECORDED)) == 1

    with pytest.raises(ApprovalError):
        resumed_loop.run(request, approval=ApprovalRecord(
            tool_call_digest="b" * 64,
            decision=ApprovalDecision.APPROVED,
            reviewer_ref="reviewer-ref",
        ))


def test_approval_restart_preserves_exact_parameterized_arguments(tmp_path: Path) -> None:
    spec = _spec(approval=True)
    policy = _policy(spec)
    original_arguments = {"value": 6}
    first_provider = ScriptedProvider(ToolCallProposal("emit", original_arguments))
    observed_arguments = []

    def parameterized_executor(context):
        observed_arguments.append(context.arguments)
        return ToolResult({"value": context.arguments["value"]})

    loop, request, store, ledger, lineage, registry, _, _ = _environment(
        tmp_path,
        spec=spec,
        policy=policy,
        provider=first_provider,
        executor=parameterized_executor,
    )
    waiting = loop.run(request)
    original_arguments["value"] = 99
    call = MaterializedToolCall.from_dict(waiting.pending_call)
    assert call.arguments == {"value": 6}
    approval = ApprovalRecord.for_call(
        call,
        decision=ApprovalDecision.APPROVED,
        reviewer_ref="reviewer-ref",
    )

    resume_provider = ScriptedProvider()
    resumed_loop = AgentLoop(
        store=ArtifactStore(store.root),
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(lineage.path),
        registry=registry,
        policy=policy,
        decision_provider=resume_provider,
    )
    resumed = resumed_loop.run(request, approval=approval)
    assert resumed.status == RunStatus.ACTIVE.value
    assert resume_provider.calls == 0
    assert observed_arguments == [{"value": 6}]
    assert _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0].metadata["result_data"] == {"value": 6}


def test_rejected_approval_is_durable_and_does_not_execute(tmp_path: Path) -> None:
    spec = _spec(approval=True)
    policy = _policy(spec)
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path, spec=spec, policy=policy, provider=provider
    )
    waiting = loop.run(request)
    call = MaterializedToolCall.from_dict(waiting.pending_call)
    rejected = ApprovalRecord.for_call(
        call,
        decision=ApprovalDecision.REJECTED,
        reviewer_ref="reviewer-ref",
    )
    result = loop.run(request, approval=rejected)
    assert result.status == RunStatus.ACTIVE.value
    assert _events(ledger, TOOL_EXECUTION_STARTED) == []
    assert len(_events(ledger, TOOL_CALL_REJECTED)) == 1

    next_provider = ScriptedProvider(StopAction("after rejection"))
    next_loop = AgentLoop(
        store=loop.store,
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(loop.lineage.path),
        registry=loop.registry,
        policy=policy,
        decision_provider=next_provider,
    )
    assert next_loop.run(request).status == RunStatus.STOPPED.value
    assert next_provider.calls == 1


def test_approval_digest_and_active_tool_semantics_are_exact(tmp_path: Path) -> None:
    spec = _spec(approval=True)
    policy = _policy(spec)
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, store, ledger, lineage, _, _, _ = _environment(
        tmp_path, spec=spec, policy=policy, provider=provider
    )
    waiting = loop.run(request)
    call = MaterializedToolCall.from_dict(waiting.pending_call)
    wrong = ApprovalRecord(
        tool_call_digest="c" * 64,
        decision=ApprovalDecision.APPROVED,
        reviewer_ref="reviewer-ref",
    )
    with pytest.raises(ApprovalError):
        loop.run(request, approval=wrong)
    assert not _events(ledger, APPROVAL_RECORDED)

    changed_registry = ToolRegistry()
    changed_registry.register(_spec(approval=True, version="2"), lambda context: ToolResult({"value": 7}))
    changed_loop = AgentLoop(
        store=store,
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(lineage.path),
        registry=changed_registry,
        policy=policy,
        decision_provider=ScriptedProvider(),
    )
    exact = ApprovalRecord.for_call(call, decision="APPROVED", reviewer_ref="reviewer-ref")
    with pytest.raises(ToolContractError):
        changed_loop.run(request, approval=exact)

    changed_args = MaterializedToolCall(
        run_id=call.run_id,
        step_id=call.step_id,
        call_id=call.call_id,
        tool_name=call.tool_name,
        tool_version=call.tool_version,
        tool_spec_digest=call.tool_spec_digest,
        policy_digest=call.policy_digest,
        arguments={"value": 2},
        input_artifact_ids=call.input_artifact_ids,
        created_at=call.created_at,
    )
    with pytest.raises(ApprovalError):
        exact.assert_binds_to(changed_args)


def test_artifact_visibility_is_run_scoped_and_explicit_cross_run_inputs_work(
    tmp_path: Path,
) -> None:
    external_store = ArtifactStore(tmp_path / "external" / "artifacts")
    external = external_store.put(b"external", media_type="text/plain")

    guessed_provider = ScriptedProvider(
        ToolCallProposal("emit", {"value": 1}, input_artifact_ids=(external.artifact_id,))
    )
    guessed_loop, guessed_request, _, _, _, _, _, _ = _environment(
        tmp_path / "guessed", provider=guessed_provider
    )
    # Copying the same object into a store does not make it run-visible.
    guessed_loop.store.put(b"external", media_type="text/plain")
    with pytest.raises(ToolContractError):
        guessed_loop.run(guessed_request)

    explicit_provider = ScriptedProvider(
        ToolCallProposal("emit", {"value": 1}, input_artifact_ids=(external.artifact_id,))
    )
    explicit_loop, explicit_request, store, ledger, _, _, spec, policy = _environment(
        tmp_path / "explicit", provider=explicit_provider
    )
    store.put(b"external", media_type="text/plain")
    explicit_request = RunRequest(
        run_id=explicit_request.run_id,
        goal=explicit_request.goal,
        input_artifact_ids=(external.artifact_id,),
        tool_policy_digest=policy.digest,
        created_at=explicit_request.created_at,
        metadata=explicit_request.metadata,
    )
    assert explicit_loop.run(explicit_request).status == RunStatus.ACTIVE.value
    assert _events(ledger, TOOL_EXECUTION_SUCCEEDED)


def test_identical_outputs_across_runs_keep_distinct_occurrences(tmp_path: Path) -> None:
    spec = _spec()
    policy = _policy(spec)
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    registry = ToolRegistry()
    registry.register(
        spec,
        lambda context: ToolResult(
            {"value": 7}, (ArtifactDraft(b"same-output", "text/plain"),)
        ),
    )

    requests = []
    for run_name in ("A", "B"):
        provider = ScriptedProvider(
            ToolCallProposal("emit", {"value": 1}),
            StopAction("done"),
        )
        loop = AgentLoop(
            store=store,
            ledger=ledger,
            lineage=lineage,
            registry=registry,
            policy=policy,
            decision_provider=provider,
        )
        request = RunRequest.create(
            goal=f"run {run_name}",
            tool_policy_digest=policy.digest,
        )
        requests.append(request)
        assert loop.run(request).status == RunStatus.STOPPED.value

    successes = _events(ledger, TOOL_EXECUTION_SUCCEEDED)
    assert len(successes) == 2
    assert successes[0].output_artifact_ids == successes[1].output_artifact_ids
    assert successes[0].step_id != successes[1].step_id
    assert successes[0].metadata["call_id"] != successes[1].metadata["call_id"]
    output_id = successes[0].output_artifact_ids[0]
    production = [
        relation
        for relation in lineage.for_subject(output_id)
        if relation.relation_type == RelationType.PRODUCED_BY.value
    ]
    assert [relation.metadata["run_id"] for relation in production] == [
        requests[0].run_id,
        requests[1].run_id,
    ]


def test_success_data_is_visible_to_the_next_decision_turn(tmp_path: Path) -> None:
    spec = _spec(name="calculate")
    policy = _policy(spec)

    class ResultAwareProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.contexts = []

        def next_action(self, context, model_visible_tools):
            self.calls += 1
            self.contexts.append(context)
            if self.calls == 1:
                return ToolCallProposal("calculate", {"value": 21})
            assert context.previous_tool_outcome["data"]["value"] == 42
            return StopAction("result consumed")

    provider = ResultAwareProvider()

    def calculate(context):
        return ToolResult({"value": context.arguments["value"] * 2}, artifacts=())

    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path,
        spec=spec,
        policy=policy,
        provider=provider,
        executor=calculate,
    )
    result = loop.run(request)
    assert result.status == RunStatus.STOPPED.value
    assert provider.calls == 2
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    expected_digest = sha256_bytes(canonical_json_bytes({"value": 42}))
    assert success.output_artifact_ids == ()
    assert success.metadata["result_data"] == {"value": 42}
    assert success.metadata["result_data_sha256"] == expected_digest
    assert provider.contexts[1].previous_tool_outcome["data_sha256"] == expected_digest


def test_data_only_result_and_observation_survive_restart(tmp_path: Path) -> None:
    spec = _spec()
    policy = _policy(spec)
    first_provider = ScriptedProvider(ToolCallProposal("emit", {"value": 21}))

    def calculate(context):
        return ToolResult({"value": context.arguments["value"] * 2}, artifacts=())

    loop, request, store, ledger, lineage, registry, _, _ = _environment(
        tmp_path,
        spec=spec,
        policy=policy,
        provider=first_provider,
        executor=calculate,
    )
    first = loop.run(request)
    assert first.status == RunStatus.ACTIVE.value
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    expected_digest = sha256_bytes(canonical_json_bytes({"value": 42}))
    assert success.output_artifact_ids == ()
    assert success.metadata["result_data"] == {"value": 42}
    assert success.metadata["result_data_sha256"] == expected_digest

    restart_provider = ScriptedProvider(StopAction("after restart"))
    restarted = AgentLoop(
        store=ArtifactStore(store.root),
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(lineage.path),
        registry=registry,
        policy=policy,
        decision_provider=restart_provider,
    )
    final = restarted.run(request)
    assert final.status == RunStatus.STOPPED.value
    assert restart_provider.calls == 1
    assert restart_provider.contexts[0].previous_tool_outcome["data"] == {"value": 42}
    assert restart_provider.contexts[0].previous_tool_outcome["data_sha256"] == expected_digest


def test_tampered_success_result_digest_fails_closed(tmp_path: Path) -> None:
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, _, ledger, _, _, _, _ = _environment(tmp_path, provider=provider)
    loop.run(request)
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    tampered_metadata = dict(success.metadata)
    tampered_metadata["result_data"] = {"value": 999}
    tampered = replace(success, metadata=tampered_metadata)

    with pytest.raises(RunStateError, match="digest mismatch"):
        loop._project((tampered,))


def test_oversized_result_data_fails_as_tool_execution_failure(tmp_path: Path) -> None:
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))

    def oversized(context):
        return ToolResult({"value": "x" * MAX_TOOL_RESULT_DATA_BYTES})

    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path,
        provider=provider,
        executor=oversized,
    )
    result = loop.run(request)
    assert result.status == RunStatus.ACTIVE.value
    assert len(_events(ledger, TOOL_EXECUTION_FAILED)) == 1
    assert not _events(ledger, TOOL_EXECUTION_SUCCEEDED)


def test_restart_reconciles_missing_lineage_idempotently(tmp_path: Path) -> None:
    provider = ScriptedProvider(ToolCallProposal("emit", {"value": 1}))
    loop, request, _, ledger, lineage, registry, spec, policy = _environment(
        tmp_path, provider=provider
    )
    first = loop.run(request)
    assert first.status == RunStatus.ACTIVE.value
    success = _events(ledger, TOOL_EXECUTION_SUCCEEDED)[0]
    original = len(lineage.relations)
    assert original == 1

    # Simulate a crash after the durable success event, before projection.
    lineage.path.write_bytes(b"")
    restart_provider = ScriptedProvider(StopAction("after reconciliation"))
    restarted = AgentLoop(
        store=ArtifactStore(loop.store.root),
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(lineage.path),
        registry=registry,
        policy=policy,
        decision_provider=restart_provider,
    )
    assert restarted.run(request).status == RunStatus.STOPPED.value
    assert len(restarted.lineage.relations) == original
    assert restarted.lineage.relations[0].metadata["success_event_id"] == success.event_id
    assert restart_provider.calls == 1
    before = restarted.lineage.path.read_bytes()
    assert restarted.run(request).status == RunStatus.STOPPED.value
    assert restarted.lineage.path.read_bytes() == before


def test_interrupted_started_call_stops_without_reexecution(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    loop, request, _, ledger, _, registry, spec, policy = _environment(
        tmp_path, provider=provider
    )
    loop._initialize_or_validate(request)
    call = MaterializedToolCall(
        run_id=request.run_id,
        step_id="step_interrupted",
        call_id="call_interrupted",
        tool_name=spec.name,
        tool_version=spec.version,
        tool_spec_digest=spec.digest,
        policy_digest=policy.digest,
        arguments={"value": 1},
        created_at="2026-01-01T00:00:00Z",
    )
    loop._append_materialized(request, call)
    loop._append(
        request,
        TOOL_EXECUTION_STARTED,
        step_id=call.step_id,
        status="STARTED",
        tool_name=call.tool_name,
        tool_version=call.tool_version,
        input_artifact_ids=call.input_artifact_ids,
        metadata={"call_id": call.call_id, "tool_call_digest": call.digest},
    )

    restarted_provider = ScriptedProvider(StopAction("must not run"))
    restarted = AgentLoop(
        store=loop.store,
        ledger=RunLedger(ledger.path),
        lineage=ArtifactLineage(loop.lineage.path),
        registry=registry,
        policy=policy,
        decision_provider=restarted_provider,
    )
    result = restarted.run(request)
    assert result.status == RunStatus.INTERRUPTED.value
    assert restarted_provider.calls == 0
    assert not _events(ledger, TOOL_EXECUTION_SUCCEEDED)


def test_run_continues_until_provider_stops(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        ToolCallProposal("emit", {"value": 1}),
        StopAction("done"),
    )
    loop, request, _, ledger, _, _, _, _ = _environment(
        tmp_path, provider=provider
    )

    result = loop.run(request)

    assert result.status == RunStatus.STOPPED.value
    assert set(request.to_dict()) == {
        "run_id",
        "goal",
        "input_artifact_ids",
        "tool_policy_digest",
        "created_at",
        "metadata",
    }
    assert set(result.to_dict()) == {
        "run_id",
        "status",
        "visible_artifact_ids",
        "last_event_id",
        "pending_call",
        "message",
    }


def test_stop_review_and_unknown_actions_are_closed(tmp_path: Path) -> None:
    stop_loop, stop_request, _, _, _, _, _, _ = _environment(
        tmp_path / "stop", provider=ScriptedProvider(StopAction("done"))
    )
    assert stop_loop.run(stop_request).status == RunStatus.STOPPED.value

    review_loop, review_request, _, review_ledger, _, _, _, _ = _environment(
        tmp_path / "review", provider=ScriptedProvider(RequestReviewAction("inspect"))
    )
    assert review_loop.run(review_request).status == RunStatus.WAITING_REVIEW.value
    assert review_ledger.events[-1].event_type == "REVIEW_REQUESTED"

    unknown_loop, unknown_request, _, _, _, _, _, _ = _environment(
        tmp_path / "unknown", provider=ScriptedProvider({"action_type": "UNKNOWN"})
    )
    with pytest.raises(ActionError):
        unknown_loop.run(unknown_request)


def test_model_actions_cannot_supply_authority_fields() -> None:
    with pytest.raises(ToolContractError):
        ToolCallProposal("emit", {"path": "arbitrary"})
    with pytest.raises(ToolContractError):
        ToolCallProposal.from_dict(
            {
                "action_type": "TOOL_CALL",
                "tool_name": "emit",
                "arguments": {"value": 1},
                "call_id": "model-chosen",
            }
        )


def test_deterministic_lineage_ids_fail_closed_on_conflict(tmp_path: Path) -> None:
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl", known_ids=("step_one",))
    artifact = "sha256:" + "a" * 64
    lineage.register_artifact(artifact)
    created = lineage.add_relation_idempotent(
        RelationType.PRODUCED_BY,
        artifact,
        "step_one",
        relation_id="rel_execution_1",
        created_at="2026-01-01T00:00:00Z",
        metadata={"run_id": "run_one"},
    )
    assert lineage.add_relation_idempotent(
        RelationType.PRODUCED_BY,
        artifact,
        "step_one",
        relation_id=created.relation_id,
        created_at=created.created_at,
        metadata={"run_id": "run_one"},
    ) == created
    with pytest.raises(Exception):
        lineage.add_relation_idempotent(
            RelationType.PRODUCED_BY,
            artifact,
            "step_one",
            relation_id=created.relation_id,
            created_at=created.created_at,
            metadata={"run_id": "run_two"},
        )


def test_production_namespace_has_no_legacy_or_spike_imports() -> None:
    source_root = Path(__file__).parents[2] / "src" / "molly"
    forbidden = {"ai4s_agent", "prototypes.core_v2_contract_spike"}
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module or ""}
            else:
                continue
            assert not any(
                name == item or name.startswith(item + ".") for name in names for item in forbidden
            ), path
