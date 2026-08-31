"""Focused CORE-07 deterministic observer and exporter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from molly.core import (
    ArtifactDraft,
    RunBudget,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.observability import (
    ExporterUnavailableError,
    JsonTraceExporter,
    LangSmithExporter,
    ObservationService,
    ObserverIntegrityError,
    OpenTelemetryExporter,
    RunTraceProjector,
)
from molly.runtime import RuntimeProfile, RuntimeProfileRegistry, RuntimeService


pytestmark = pytest.mark.unit


class ScriptedProvider:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)

    def next_action(self, context, model_visible_tools):
        if not self.actions:
            raise StopIteration
        return self.actions.pop(0)


def _service(tmp_path: Path):
    spec = ToolSpec(
        name="observe_emit",
        description="observer fixture tool",
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
    )
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(spec.side_effect_class,),
    )

    def registry_factory():
        registry = ToolRegistry()
        registry.register(
            spec,
            lambda context: ToolResult(
                {"value": context.arguments["value"]},
                (ArtifactDraft(b"observer-output", "text/plain"),),
            ),
        )
        return registry

    def provider_factory():
        return ScriptedProvider(
            ToolCallProposal("observe_emit", {"value": 5}, reason_summary="hidden reasoning must not export"),
            StopAction("done"),
        )

    profile = RuntimeProfile(
        profile_id="profile:observe",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=provider_factory,
        config={"logical_observer_ref": "test-profile"},
    )
    service = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    result = service.start_run(
        profile_id=profile.profile_id,
        goal="observe a deterministic run",
        budget=RunBudget(max_decisions=4, max_tool_calls=2, max_steps=2),
    )
    return service, profile, result


def test_trace_projection_is_deterministic_across_restart_and_has_approval_safe_shape(
    tmp_path: Path,
) -> None:
    service, profile, result = _service(tmp_path)

    store, ledger, lineage = service._open_components(create=False)  # noqa: SLF001
    from molly.core.inspection import RunInspector

    trace_a = RunTraceProjector(RunInspector(store=store, ledger=ledger, lineage=lineage)).project_run(result.run_id)
    restarted = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    store_b, ledger_b, lineage_b = restarted._open_components(create=False)  # noqa: SLF001
    trace_b = RunTraceProjector(RunInspector(store=store_b, ledger=ledger_b, lineage=lineage_b)).project_run(result.run_id)
    assert trace_a.canonical_bytes() == trace_b.canonical_bytes()
    assert trace_a.trace_id == trace_b.trace_id
    assert len(trace_a.spans) == 2
    assert trace_a.spans[0].parent_span_id is None
    assert trace_a.spans[1].parent_span_id == trace_a.spans[0].span_id
    trace_text = trace_a.canonical_bytes().decode("utf-8")
    assert "hidden reasoning" not in trace_text
    assert '"result_data":' not in trace_text
    assert service.inspect_run(result.run_id).status == "STOPPED"


def test_json_otel_and_langsmith_exports_are_observer_only(tmp_path: Path) -> None:
    service, _, result = _service(tmp_path)
    captured: list[dict] = []

    class FakeCollector:
        def export_trace(self, payload):
            captured.append(payload)

    json_outcome = service.observe_run(result.run_id, JsonTraceExporter())
    assert json_outcome.status == "EXPORTED"
    otel_outcome = service.observe_run(
        result.run_id,
        OpenTelemetryExporter(client=FakeCollector()),
    )
    lang_outcome = service.observe_run(
        result.run_id,
        LangSmithExporter(client=FakeCollector()),
    )
    assert otel_outcome.status == "EXPORTED"
    assert lang_outcome.status == "EXPORTED"
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert captured[0]["trace_id"] == json_outcome.trace.trace_id
    assert "hidden reasoning" not in str(captured)


def test_export_failure_is_reported_without_changing_authoritative_facts(tmp_path: Path) -> None:
    service, _, result = _service(tmp_path)
    events_path = tmp_path / "runtime" / "events.jsonl"
    lineage_path = tmp_path / "runtime" / "lineage.jsonl"
    before_events = events_path.read_bytes()
    before_lineage = lineage_path.read_bytes()

    class FailingExporter:
        name = "failing"

        def export(self, trace):
            raise RuntimeError("synthetic-secret-and-private-path")

    outcome = service.observe_run(result.run_id, FailingExporter())
    assert outcome.status == "EXPORT_FAILED"
    assert outcome.error_type == "RuntimeError"
    assert service.inspect_run(result.run_id).status == "STOPPED"
    assert events_path.read_bytes() == before_events
    assert lineage_path.read_bytes() == before_lineage
    assert "synthetic-secret" not in str(outcome.to_dict())
    assert "private-path" not in str(outcome.to_dict())


def test_unavailable_optional_exporter_is_bounded_and_not_a_run_failure(tmp_path: Path) -> None:
    service, _, result = _service(tmp_path)
    with pytest.raises(ExporterUnavailableError):
        service.observe_run(result.run_id, OpenTelemetryExporter())
    assert service.inspect_run(result.run_id).status == "STOPPED"


def test_observer_integrity_change_fails_closed(tmp_path: Path) -> None:
    service, _, result = _service(tmp_path)
    store, ledger, lineage = service._open_components(create=False)  # noqa: SLF001

    class MutatingExporter:
        name = "mutating"

        def export(self, trace):
            ledger.append(
                run_id=result.run_id,
                event_type="RUN_FAILED",
                status="FAILED",
            )

    with pytest.raises(ObserverIntegrityError):
        service.observe_run(result.run_id, MutatingExporter())
    assert store is not None and lineage is not None
