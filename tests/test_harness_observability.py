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
    _PrivacySafeBatchSpanProcessor,
    _PrivacySafeOTelExporter,
    _build_otel_tracer,
    build_harness_observability,
)
from ai4s_agent.langsmith_adapter import (
    LangSmithHarnessTracer,
    _langsmith_safe_metadata,
    _langsmith_safe_outputs,
)
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


class _OtelDelegateExporter:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.batches: list[list[object]] = []
        self.started = threading.Event()
        self.release: threading.Event | None = None

    def export(self, spans):
        self.batches.append(list(spans))
        self.started.set()
        if self.release is not None:
            self.release.wait(5)
        if self.error is not None:
            raise self.error
        return self.result

    def shutdown(self, **_):
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
        return None

    def update_run(self, run_id, **kwargs):
        if self.fail_update:
            raise RuntimeError("192.0.2.1 stderr payload")
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
        "192.0.2.1",
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


@pytest.mark.pr_fast
def test_otel_real_sdk_resource_ignores_malicious_environment_metadata(
    monkeypatch,
    caplog,
) -> None:
    import logging
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    delegate = _OtelDelegateExporter(result=SpanExportResult.SUCCESS)
    captured_kwargs: dict[str, object] = {}

    def exporter_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return delegate

    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "host.name=private-host,user.name=private-user,token=secret",
    )
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", exporter_factory)
    config = HarnessObservabilityConfig(otel_mode="otlp_http")
    health = HarnessTelemetryHealth(config=config)
    tracer = _build_otel_tracer(config=config, health=health)
    assert isinstance(tracer, OpenTelemetryHarnessTracer)
    with tracer.start_span(
        "controller.advance",
        attributes={
            "project_id": "project-1",
            "run_id": "run-1",
            "operation": "agent.controller.advance",
            "component": "controller",
            "phase": "advance",
        },
    ):
        pass
    assert tracer.provider.force_flush(timeout_millis=1000) is True
    assert captured_kwargs == {"timeout": 5.0}
    assert tracer.provider.resource.attributes == {
        "service.name": "molly-scientific-agent-harness"
    }
    assert delegate.batches
    assert delegate.batches[0][0].resource.attributes == {
        "service.name": "molly-scientific-agent-harness"
    }
    serialized = repr(delegate.batches)
    assert "private-host" not in serialized
    assert "private-user" not in serialized
    assert "token=secret" not in serialized
    logging.getLogger(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    ).error("endpoint=%s token=%s", "https://private-host", "secret")
    assert "MOLLY_OTEL_VENDOR_LOG_REDACTED" in caplog.text
    assert "private-host" not in caplog.text
    assert "token=" not in caplog.text
    tracer.shutdown()


@pytest.mark.pr_fast
def test_otel_real_grpc_sdk_redacts_init_and_rpc_error_logs(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    import logging

    import grpc
    from opentelemetry.sdk.trace.export import SpanExportResult

    base_logger_name = "opentelemetry.exporter.otlp.proto.grpc.exporter"
    trace_logger_name = (
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    )
    private_endpoint = "https://private-collector.invalid:4317"
    private_certificate = tmp_path / "private-client-certificate.pem"
    private_details = "Authorization: Bearer secret; token=secret"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", private_endpoint)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
        str(private_certificate),
    )
    caplog.set_level(logging.DEBUG, logger=base_logger_name)
    caplog.set_level(logging.DEBUG, logger=trace_logger_name)

    config = HarnessObservabilityConfig(otel_mode="otlp_grpc")
    health = HarnessTelemetryHealth(config=config)
    tracer = _build_otel_tracer(config=config, health=health)
    assert isinstance(tracer, OpenTelemetryHarnessTracer)

    class SensitiveRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNKNOWN

        def details(self):
            return private_details

        def trailing_metadata(self):
            return ()

    def fail_export(**_kwargs):
        raise SensitiveRpcError()

    processors = tracer.provider._active_span_processor._span_processors
    processor = next(
        item
        for item in processors
        if isinstance(item, _PrivacySafeBatchSpanProcessor)
    )
    delegate = processor.exporter.delegate
    delegate._client = SimpleNamespace(Export=fail_export)
    assert processor.exporter.export([]) == SpanExportResult.FAILURE

    vendor_records = [
        record
        for record in caplog.records
        if record.name in {base_logger_name, trace_logger_name}
    ]
    # One record covers credential-file initialization and another covers the
    # real OTLPExporterMixin RpcError path before it returns FAILURE.
    assert len(vendor_records) >= 2
    assert {
        record.getMessage() for record in vendor_records
    } == {"MOLLY_OTEL_VENDOR_LOG_REDACTED"}
    captured = caplog.text
    assert str(private_certificate) not in captured
    assert private_endpoint not in captured
    assert private_details not in captured
    assert "Bearer secret" not in captured
    assert "Traceback" not in captured
    assert health.snapshot().export_failure_count == 1
    tracer.shutdown()


@pytest.mark.pr_fast
def test_otel_export_exception_and_failure_are_safely_counted(caplog) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult

    config = HarnessObservabilityConfig(otel_mode="otlp_http")
    health = HarnessTelemetryHealth(config=config)
    exploding = _PrivacySafeOTelExporter(
        delegate=_OtelDelegateExporter(
            error=RuntimeError(
                "collector=https://private-host token=secret /private/path"
            )
        ),
        health=health,
        success_result=SpanExportResult.SUCCESS,
        failure_result=SpanExportResult.FAILURE,
    )
    assert exploding.export([object()]) == SpanExportResult.FAILURE
    failing = _PrivacySafeOTelExporter(
        delegate=_OtelDelegateExporter(result=SpanExportResult.FAILURE),
        health=health,
        success_result=SpanExportResult.SUCCESS,
        failure_result=SpanExportResult.FAILURE,
    )
    assert failing.export([object()]) == SpanExportResult.FAILURE
    snapshot = health.snapshot()
    assert snapshot.export_failure_count == 2
    assert snapshot.otel_last_result_code == "OTEL_EXPORT_FAILED"
    assert "private-host" not in caplog.text
    assert "token=secret" not in caplog.text
    assert "/private/path" not in caplog.text


@pytest.mark.pr_fast
def test_otel_queue_full_is_dropped_and_counted_without_sdk_warning(caplog) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.trace import TraceFlags

    config = HarnessObservabilityConfig(otel_mode="otlp_http")
    health = HarnessTelemetryHealth(config=config)
    delegate = _OtelDelegateExporter(result=SpanExportResult.SUCCESS)
    delegate.release = threading.Event()
    exporter = _PrivacySafeOTelExporter(
        delegate=delegate,
        health=health,
        success_result=SpanExportResult.SUCCESS,
        failure_result=SpanExportResult.FAILURE,
    )
    processor = _PrivacySafeBatchSpanProcessor(
        exporter=exporter,
        health=health,
        max_queue_size=1,
        schedule_delay_millis=5000,
        max_export_batch_size=1,
    )
    sampled_span = SimpleNamespace(
        context=SimpleNamespace(trace_flags=TraceFlags(TraceFlags.SAMPLED))
    )
    processor.on_end(sampled_span)
    assert delegate.started.wait(1)
    processor.on_end(sampled_span)
    processor.on_end(sampled_span)
    snapshot = health.snapshot()
    assert snapshot.dropped_event_count == 1
    assert snapshot.otel_last_result_code == "OTEL_QUEUE_FULL"
    assert "Queue full" not in caplog.text
    delegate.release.set()
    processor.shutdown(timeout_millis=1000)


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
    assert client.created[0]["id"] == client.updated[0][0]
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


@pytest.mark.pr_fast
def test_langsmith_real_sdk_send_boundary_strips_runtime_and_env_metadata(
    monkeypatch,
) -> None:
    from langsmith import Client

    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "private-revision")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://private-host")
    monkeypatch.setenv("LANGSMITH_PROJECT", "private-project")
    client = Client(
        api_url="https://unused.invalid",
        api_key="test-key",
        auto_batch_tracing=False,
        hide_inputs=True,
        hide_metadata=_langsmith_safe_metadata,
        hide_outputs=_langsmith_safe_outputs,
        omit_traced_runtime_info=True,
    )
    captured: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        Client,
        "_create_run",
        lambda self, run_create, **_: captured.append(run_create),
    )
    monkeypatch.setattr(
        Client,
        "_update_run",
        lambda self, run_update, **_: updates.append(run_update),
    )
    run_id = __import__("uuid").uuid4()
    client.create_run(
        id=run_id,
        name="agent.plan.llm_call",
        project_name="molly-scientific-agent-harness",
        run_type="llm",
        inputs={"prompt": "/private/paper token=secret"},
        outputs={"outcome": "completed", "raw": "private response"},
        extra={
            "metadata": {
                "molly.project_id": "project-1",
                "molly.telemetry_authoritative": False,
                "private_key": "secret",
            },
        },
    )
    assert len(captured) == 1
    payload = captured[0]
    assert payload["id"] == run_id
    assert payload["inputs"] == {}
    assert payload["outputs"] == {"outcome": "completed"}
    assert payload["extra"] == {
        "metadata": {
            "molly.project_id": "project-1",
            "molly.telemetry_authoritative": False,
        },
    }
    serialized = repr(payload)
    for forbidden in (
        "private-revision",
        "LANGCHAIN_ENDPOINT",
        "/private/paper",
        "token=secret",
        "private response",
        "private_key",
        "private-host",
        "private-project",
        "python_version",
        "platform",
    ):
        assert forbidden not in serialized

    health = HarnessTelemetryHealth(
        config=HarnessObservabilityConfig(langsmith_mode="metadata_only")
    )
    tracer = LangSmithHarnessTracer(
        client=client,
        mode="metadata_only",
        health=health,
    )
    with tracer.start_span(
        "planner.llm_call",
        attributes={
            "project_id": "project-1",
            "run_id": "run-1",
            "operation": "agent.plan.llm_call",
            "component": "planner",
            "phase": "provider_call",
        },
    ) as span:
        span.add_event(
            "planner.provider_completed", {"outcome": "completed"}
        )
    assert len(updates) == 1
    assert captured[-1]["id"] == updates[0]["id"]
    assert captured[-1]["session_name"] == "molly-scientific-agent-harness"
    assert updates[0]["outputs"] == {
        "outcome": "completed",
        "exception_type_code": "NO_EXCEPTION",
    }
    final_payload = repr((captured[-1], updates[0]))
    assert "runtime" not in final_payload
    assert "private-revision" not in final_payload
    assert "private-host" not in final_payload
    assert "private-project" not in final_payload


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
        langsmith_client_factory=lambda **_: (_ for _ in ()).throw(
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

    def factory(**kwargs) -> _LangSmithClient:
        assert kwargs["omit_traced_runtime_info"] is True
        assert kwargs["auto_batch_tracing"] is False
        assert kwargs["timeout_ms"] == 1000
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
