from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ai4s_agent.harness_tracing import (
    CompositeHarnessTracer,
    NoopHarnessTracer,
    OpenTelemetryHarnessTracer,
    build_harness_observability,
)
from ai4s_agent.langsmith_adapter import LangSmithHarnessTracer
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.observability_config import (
    HarnessObservabilityConfig,
    HarnessTelemetryHealth,
)
from ai4s_agent.observability_correlation import (
    TELEMETRY_PRIVACY_POLICY_VERSION,
    build_harness_telemetry_correlation,
    privacy_safe_telemetry_attributes,
)
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    HarnessTelemetryCorrelationContext,
    CORE_SCHEMA_MODELS,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage


NOW = "2026-08-02T00:00:00Z"


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def add_event(self, name, attributes=None) -> None:
        self.events.append((name, dict(attributes or {})))


class _FailingAttributeSpan(_RecordingSpan):
    def set_attribute(self, key, value) -> None:
        raise RuntimeError("/private/attribute token=secret")

    def add_event(self, name, attributes=None) -> None:
        raise RuntimeError("stderr event payload")


class _RecordingContext:
    def __init__(self, span: _RecordingSpan, *, fail_exit: bool = False) -> None:
        self.span = span
        self.fail_exit = fail_exit

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fail_exit:
            raise RuntimeError("/private/exporter-end token=secret")
        return False


class _RecordingOtelDelegate:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self.fail_start = fail_start
        self.fail_exit = fail_exit
        self.records: list[dict[str, object]] = []

    def start_as_current_span(self, name, **kwargs):
        if self.fail_start:
            raise RuntimeError("ssh://private-host Authorization: Bearer secret")
        span = _RecordingSpan()
        self.records.append(
            {
                "name": name,
                "attributes": dict(kwargs.get("attributes") or {}),
                "span": span,
                "record_exception": kwargs.get("record_exception"),
                "set_status_on_exception": kwargs.get("set_status_on_exception"),
            }
        )
        return _RecordingContext(span, fail_exit=self.fail_exit)


class _Provider:
    def __init__(self, *, fail_shutdown: bool = False) -> None:
        self.fail_shutdown = fail_shutdown

    def force_flush(self, **_):
        if self.fail_shutdown:
            raise RuntimeError("collector endpoint private")


class _BlockingProvider:
    def __init__(self) -> None:
        self.release = threading.Event()

    def force_flush(self, **_):
        self.release.wait(5)

    def shutdown(self):
        return None
        return True

    def shutdown(self):
        if self.fail_shutdown:
            raise RuntimeError("collector endpoint private")


class _LangSmithClient:
    def __init__(
        self,
        *,
        fail_create: bool = False,
        fail_update: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.fail_create = fail_create
        self.fail_update = fail_update
        self.fail_close = fail_close
        self.created: list[dict[str, object]] = []
        self.updated: list[tuple[object, dict[str, object]]] = []

    def create_run(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("token=secret at /private/langsmith")
        self.created.append(kwargs)
        return SimpleNamespace(id="vendor-run-private-id")

    def update_run(self, run_id, **kwargs):
        if self.fail_update:
            raise RuntimeError("10.0.0.1 stderr payload")
        self.updated.append((run_id, kwargs))

    def close(self):
        if self.fail_close:
            raise RuntimeError("LANGSMITH_API_KEY=secret")


class _ExplodingHarnessTracer:
    def start_span(self, *_, **__):
        raise RuntimeError("raw private tracer failure")

    def shutdown(self):
        raise RuntimeError("raw private shutdown failure")


class _CountingStubProvider(StubLLMProvider):
    def __init__(self, *, response: dict[str, object]) -> None:
        super().__init__(response=response)
        self.call_count = 0

    def complete_json(self, **kwargs):
        self.call_count += 1
        return super().complete_json(**kwargs)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock")
    }


def _planning_result(tmp_path: Path, *, tracer, provider=None):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: NOW)
    store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {"count": 4, "seed": 1}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce an OLED host–dopant candidate roster"],
        rationales=["Use a bounded host material workflow."],
        assumptions=[],
        questions=[],
    )
    active_provider = provider or StubLLMProvider(
        response=response.model_dump(mode="json")
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=store,
        tracer=tracer,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Generate OLED host material candidates",
        user_constraints=["Keep the host-dopant pair roster bounded"],
        provider=active_provider,
        client_request_id="observability-plan-1",
    )
    return proposal, _snapshot(storage.project_dir("project-1"))


def test_config_defaults_modes_and_structured_content_authorization() -> None:
    assert HarnessObservabilityConfig.from_environ() == HarnessObservabilityConfig()
    enabled = HarnessObservabilityConfig.from_environ(
        {
            "AI4S_HARNESS_OTEL_MODE": "otlp_grpc",
            "AI4S_HARNESS_LANGSMITH_MODE": "structured_content",
        }
    )
    assert enabled.otel_mode == "otlp_grpc"
    assert enabled.langsmith_mode == "metadata_only"
    allowed = HarnessObservabilityConfig.from_environ(
        {
            "AI4S_HARNESS_LANGSMITH_MODE": "structured_content",
            "AI4S_HARNESS_LANGSMITH_STRUCTURED_CONTENT_ALLOWED": "true",
        }
    )
    assert allowed.langsmith_mode == "structured_content"
    invalid = HarnessObservabilityConfig.from_environ(
        {
            "AI4S_HARNESS_OTEL_MODE": "private-endpoint-mode",
            "AI4S_HARNESS_LANGSMITH_MODE": "upload_everything",
        }
    )
    assert invalid.otel_mode == invalid.langsmith_mode == "disabled"
    assert invalid.otel_config_valid is invalid.langsmith_config_valid is False


def test_correlation_schema_is_canonical_namespaced_and_privacy_bounded() -> None:
    context = build_harness_telemetry_correlation(
        project_id="project-1",
        run_id="run-1",
        proposal_id="proposal-1",
        proposal_digest="sha256:" + "1" * 64,
        controller_execution_id="controller-1",
        controller_execution_digest="sha256:" + "2" * 64,
        task_id="inspect_dataset",
        task_index=0,
        slot_id="slot-1",
        operation="agent.execution.local",
        component="executor",
        phase="completed",
    )
    attributes = privacy_safe_telemetry_attributes(context)
    assert context.telemetry_authoritative is False
    assert attributes["molly.telemetry_authoritative"] is False
    assert attributes["molly.schema_version"] == "harness_telemetry_correlation.v1"
    assert attributes["molly.project_id"] == "project-1"
    assert TELEMETRY_PRIVACY_POLICY_VERSION == "harness_telemetry_privacy_policy.v1"
    serialized = json.dumps(attributes, sort_keys=True)
    for forbidden in (
        "/private/user/path",
        "ssh://private-host",
        "10.0.0.1",
        "user@example",
        "token=secret",
        "Authorization: Bearer secret",
        "bash -c",
        "stdout payload",
        "stderr payload",
    ):
        assert forbidden not in serialized
    with pytest.raises(ValueError):
        build_harness_telemetry_correlation(
            project_id="project-1",
            run_id="run-1",
            task_id="/private/user/path",
            operation="agent.execution.local",
            component="executor",
            phase="running",
        )


def test_otel_exports_stable_namespaced_allowlist_without_raw_errors() -> None:
    delegate = _RecordingOtelDelegate()
    tracer = OpenTelemetryHarnessTracer(
        tracer=delegate,
        provider=_Provider(),
    )
    with tracer.start_span(
        "controller.execution",
        attributes={
            "project_id": "project-1",
            "operation": "agent.controller.create",
            "component": "controller",
            "phase": "create",
        },
    ) as span:
        span.set_attribute("task_id", "inspect_dataset")
        span.set_attribute("task_id", "/private/user/path")
        span.add_event(
            "controller.failure",
            {
                "reason_code": "CONTROLLER_SAFE_FAILURE",
                "exception_type_code": "EXCEPTION_RUNTIMEERROR",
            },
        )
    record = delegate.records[0]
    assert record["name"] == "agent.controller.create"
    assert record["record_exception"] is False
    assert record["set_status_on_exception"] is False
    assert record["attributes"]["molly.telemetry_authoritative"] is False
    assert record["attributes"]["molly.project_id"] == "project-1"
    span = record["span"]
    assert span.attributes["molly.task_id"] == "inspect_dataset"
    assert "/private/user/path" not in repr(delegate.records)


def test_otel_start_end_and_shutdown_failures_are_fail_open() -> None:
    config = HarnessObservabilityConfig(otel_mode="otlp_http")
    health = HarnessTelemetryHealth(config=config)
    tracer = OpenTelemetryHarnessTracer(
        tracer=_RecordingOtelDelegate(fail_start=True, fail_exit=True),
        provider=_Provider(fail_shutdown=True),
        health=health,
    )
    completed = False
    with tracer.start_span("controller.advance"):
        completed = True
    tracer.shutdown()
    assert completed is True
    assert health.snapshot().telemetry_authoritative is False


def test_otel_attribute_event_and_export_timeout_are_fail_open() -> None:
    delegate = _RecordingOtelDelegate()
    delegate.start_as_current_span = lambda *_, **__: _RecordingContext(
        _FailingAttributeSpan()
    )
    provider = _BlockingProvider()
    health = HarnessTelemetryHealth(
        config=HarnessObservabilityConfig(otel_mode="otlp_http")
    )
    tracer = OpenTelemetryHarnessTracer(
        tracer=delegate,
        provider=provider,
        health=health,
        shutdown_timeout_seconds=0.01,
    )
    with tracer.start_span("controller.advance") as span:
        span.set_attribute("task_id", "inspect_dataset")
        span.add_event("controller.waiting", {"status": "active"})
    tracer.shutdown()
    assert health.snapshot().otel_last_result_code == "OTEL_SHUTDOWN_TIMEOUT"
    provider.release.set()


def test_langsmith_metadata_only_records_one_safe_llm_run() -> None:
    config = HarnessObservabilityConfig(langsmith_mode="metadata_only")
    health = HarnessTelemetryHealth(config=config)
    client = _LangSmithClient()
    tracer = LangSmithHarnessTracer(
        client=client,
        mode="metadata_only",
        health=health,
    )
    correlation = build_harness_telemetry_correlation(
        project_id="project-1",
        run_id="run-1",
        proposal_id="proposal-1",
        proposal_digest="sha256:" + "3" * 64,
        operation="agent.plan.llm_call",
        component="planner",
        phase="provider_call",
    )
    with tracer.start_span(
        "planner.llm_call",
        attributes=privacy_safe_telemetry_attributes(correlation),
    ) as span:
        span.set_attribute("request_digest", "sha256:" + "4" * 64)
        span.set_attribute("response_digest", "sha256:" + "5" * 64)
        span.set_attribute("status", "token=secret")
    assert len(client.created) == len(client.updated) == 1
    assert client.created[0]["inputs"] == {}
    serialized = repr((client.created, client.updated))
    assert "molly.telemetry_authoritative" in serialized
    for forbidden in (
        "prompt raw text",
        "response raw text",
        "token=secret",
        "/private/",
        "private feedback",
        "chain-of-thought",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("failure", ["create", "update", "close"])
def test_langsmith_vendor_failures_never_escape(failure: str) -> None:
    config = HarnessObservabilityConfig(langsmith_mode="metadata_only")
    health = HarnessTelemetryHealth(config=config)
    client = _LangSmithClient(
        fail_create=failure == "create",
        fail_update=failure == "update",
        fail_close=failure == "close",
    )
    tracer = LangSmithHarnessTracer(
        client=client,
        mode="metadata_only",
        health=health,
    )
    completed = False
    with tracer.start_span("planner.llm_call"):
        completed = True
    tracer.shutdown()
    assert completed is True
    assert health.snapshot().telemetry_authoritative is False


def test_composite_isolates_one_broken_adapter_from_another() -> None:
    client = _LangSmithClient()
    config = HarnessObservabilityConfig(langsmith_mode="metadata_only")
    health = HarnessTelemetryHealth(config=config)
    tracer = CompositeHarnessTracer(
        (
            _ExplodingHarnessTracer(),
            LangSmithHarnessTracer(
                client=client,
                mode="metadata_only",
                health=health,
            ),
        )
    )
    with tracer.start_span("planner.llm_call"):
        pass
    tracer.shutdown()
    assert len(client.created) == 1


def test_composite_reports_only_safe_exception_type_and_preserves_error() -> None:
    client = _LangSmithClient()
    health = HarnessTelemetryHealth(
        config=HarnessObservabilityConfig(langsmith_mode="metadata_only")
    )
    tracer = CompositeHarnessTracer(
        (
            LangSmithHarnessTracer(
                client=client,
                mode="metadata_only",
                health=health,
            ),
        )
    )
    with pytest.raises(RuntimeError, match="private provider message"):
        with tracer.start_span("planner.llm_call"):
            raise RuntimeError("private provider message /private/path token=secret")
    assert client.updated[0][1]["outputs"] == {
        "outcome": "failed",
        "exception_type_code": "EXCEPTION_RUNTIMEERROR",
    }
    assert "private provider message" not in repr(client.updated)


def test_concurrent_run_correlation_does_not_cross_runs() -> None:
    delegate = _RecordingOtelDelegate()
    tracer = OpenTelemetryHarnessTracer(tracer=delegate, provider=_Provider())

    def emit(index: int) -> None:
        with tracer.start_span(
            "controller.advance",
            attributes={
                "project_id": f"project-{index}",
                "run_id": f"run-{index}",
                "operation": "agent.controller.advance",
                "component": "controller",
                "phase": "advance",
            },
        ):
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(emit, range(32)))
    pairs = {
        (
            record["attributes"]["molly.project_id"],
            record["attributes"]["molly.run_id"],
        )
        for record in delegate.records
    }
    assert pairs == {(f"project-{index}", f"run-{index}") for index in range(32)}


def test_correlation_serialization_is_hash_seed_stable() -> None:
    code = """
import json
from ai4s_agent.observability_correlation import build_harness_telemetry_correlation, privacy_safe_telemetry_attributes
context = build_harness_telemetry_correlation(project_id='project-1', run_id='run-1', task_id='host_material', operation='agent.execution.local', component='executor', phase='completed')
print(json.dumps(privacy_safe_telemetry_attributes(context), separators=(',', ':')))
"""
    outputs = []
    for seed in ("1", "73"):
        environ = dict(os.environ)
        environ["PYTHONHASHSEED"] = seed
        environ["PYTHONPATH"] = "src:."
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                env=environ,
                text=True,
            )
        )
    assert outputs[0] == outputs[1]


def test_missing_optional_langsmith_dependency_degrades_to_noop() -> None:
    tracer, health = build_harness_observability(
        environ={"AI4S_HARNESS_LANGSMITH_MODE": "metadata_only"},
        langsmith_client_factory=lambda: (_ for _ in ()).throw(
            ImportError("langsmith unavailable")
        ),
    )
    assert isinstance(tracer, NoopHarnessTracer)
    assert health.snapshot().langsmith_available is False
    with tracer.start_span("planner.llm_call"):
        pass


def test_missing_optional_otel_dependency_degrades_to_noop() -> None:
    tracer, health = build_harness_observability(
        environ={"AI4S_HARNESS_OTEL_MODE": "otlp_http"},
        otel_tracer_factory=lambda **_: (_ for _ in ()).throw(
            ImportError("opentelemetry unavailable")
        ),
    )
    assert isinstance(tracer, NoopHarnessTracer)
    snapshot = health.snapshot()
    assert snapshot.otel_enabled is True
    assert snapshot.otel_available is False
    assert snapshot.otel_last_result_code == "OTEL_INITIALIZATION_FAILED"


def test_disabled_app_registers_inspection_route_without_vendor_sdks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ai4s_agent.app import create_app

    monkeypatch.delenv("AI4S_HARNESS_OTEL_MODE", raising=False)
    monkeypatch.delenv("AI4S_HARNESS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("AI4S_HARNESS_LANGSMITH_MODE", raising=False)
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
    )
    assert isinstance(app.extensions["harness_tracer"], NoopHarnessTracer)
    assert (
        app.extensions["harness_telemetry_health"]
        .snapshot()
        .telemetry_authoritative
        is False
    )
    assert any(
        rule.rule == "/api/projects/<project_id>/agent-runs/<run_id>/inspection"
        for rule in app.url_map.iter_rules()
    )


def test_langsmith_observer_reinitializes_after_process_restart_boundary() -> None:
    clients: list[_LangSmithClient] = []

    def factory() -> _LangSmithClient:
        client = _LangSmithClient()
        clients.append(client)
        return client

    for _ in range(2):
        tracer, health = build_harness_observability(
            environ={"AI4S_HARNESS_LANGSMITH_MODE": "metadata_only"},
            langsmith_client_factory=factory,
        )
        with tracer.start_span("planner.llm_call"):
            pass
        tracer.shutdown()
        assert health.snapshot().telemetry_authoritative is False
    assert len(clients) == 2
    assert [len(client.created) for client in clients] == [1, 1]


@pytest.mark.pr_fast
def test_failing_telemetry_preserves_planner_authority_bytes_and_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ai4s_agent.scientific_agent_plan.uuid.uuid4",
        lambda: SimpleNamespace(hex="a" * 32),
    )
    monotonic_values = iter((0.0, 0.1, 0.0, 0.1))
    monkeypatch.setattr(
        "ai4s_agent.scientific_agent_plan.time.monotonic",
        lambda: next(monotonic_values),
    )
    baseline, baseline_bytes = _planning_result(
        tmp_path / "baseline",
        tracer=NoopHarnessTracer(),
    )
    config = HarnessObservabilityConfig(langsmith_mode="metadata_only")
    health = HarnessTelemetryHealth(config=config)
    failing = CompositeHarnessTracer(
        (
            OpenTelemetryHarnessTracer(
                tracer=_RecordingOtelDelegate(fail_start=True),
                provider=_Provider(fail_shutdown=True),
            ),
            LangSmithHarnessTracer(
                client=_LangSmithClient(fail_create=True),
                mode="metadata_only",
                health=health,
            ),
        )
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {"count": 4, "seed": 1}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce an OLED host–dopant candidate roster"],
        rationales=["Use a bounded host material workflow."],
        assumptions=[],
        questions=[],
    )
    provider = _CountingStubProvider(response=response.model_dump(mode="json"))
    observed, observed_bytes = _planning_result(
        tmp_path / "observed",
        tracer=failing,
        provider=provider,
    )
    assert observed == baseline
    assert observed.proposal_digest == baseline.proposal_digest
    assert observed.semantic_plan_digest == baseline.semantic_plan_digest
    assert observed_bytes == baseline_bytes
    assert provider.call_count == 1


@pytest.mark.pr_fast
def test_inspection_digest_and_authority_bytes_ignore_telemetry(tmp_path: Path) -> None:
    from tests.test_agent_run_inspection import _chain

    storage, _, _, _, _, service, _ = _chain(tmp_path)
    root = storage.project_dir("project-1")
    before_bytes = _snapshot(root)
    baseline = service.inspect(project_id="project-1", run_id="run-1")
    service.tracer = OpenTelemetryHarnessTracer(
        tracer=_RecordingOtelDelegate(fail_start=True),
        provider=_Provider(fail_shutdown=True),
    )
    observed = service.inspect(project_id="project-1", run_id="run-1")
    assert observed == baseline
    assert observed.inspection_digest == baseline.inspection_digest
    assert _snapshot(root) == before_bytes
    with pytest.raises(ValueError, match="conflicts with the verified inspection"):
        build_harness_telemetry_correlation(
            inspection=baseline,
            project_id="wrong-project",
            operation="agent.run_inspection.read",
            component="run_inspection",
            phase="completed",
        )


def test_observability_schemas_are_generated_from_pydantic_source() -> None:
    root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    expected = {
        "harness_telemetry_correlation": HarnessTelemetryCorrelationContext,
        "harness_telemetry_health": CORE_SCHEMA_MODELS["harness_telemetry_health"],
    }
    for name, model in expected.items():
        assert json.loads((root / f"{name}.schema.json").read_text()) == (
            model.model_json_schema()
        )
