from __future__ import annotations

from ai4s_agent.harness_tracing import (
    HarnessSpanLink,
    NoopHarnessTracer,
    OpenTelemetryHarnessTracer,
    build_harness_tracer,
)


class _FailingContext:
    def __enter__(self):
        raise RuntimeError("raw private exporter failure")

    def __exit__(self, exc_type, exc_value, traceback):
        raise RuntimeError("raw private exporter failure")


class _FailingTracer:
    def start_as_current_span(self, *args, **kwargs):
        return _FailingContext()


class _FailingProvider:
    def shutdown(self):
        raise RuntimeError("raw private exporter failure")


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def add_event(self, name, attributes=None) -> None:
        self.events.append((name, dict(attributes or {})))


class _RecordingContext:
    def __init__(self, span: _RecordingSpan) -> None:
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _RecordingTracer:
    def __init__(self, span: _RecordingSpan) -> None:
        self.span = span
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def start_as_current_span(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _RecordingContext(self.span)


def test_noop_is_default_and_invalid_telemetry_never_changes_business_flow() -> None:
    tracer = build_harness_tracer({})
    assert isinstance(tracer, NoopHarnessTracer)

    completed = False
    with tracer.start_span(
        "not-allowlisted",
        attributes={"raw_path": "/private/data"},
    ) as span:
        span.set_attribute("raw_path", "/private/data")
        span.add_event("not-allowlisted", {"raw_path": "/private/data"})
        completed = True
    assert completed is True


def test_missing_optional_opentelemetry_dependencies_degrade_to_noop() -> None:
    tracer = build_harness_tracer({"AI4S_HARNESS_OTEL_ENABLED": "true"})
    # The test environment may or may not install the optional extra. Either
    # result must expose the narrow HarnessTracer surface and stay usable.
    with tracer.start_span(
        "controller.execution",
        attributes={"run_id": "run-a"},
    ):
        pass
    tracer.shutdown()


def test_exporter_and_context_failures_are_non_authoritative() -> None:
    tracer = OpenTelemetryHarnessTracer(
        tracer=_FailingTracer(),
        provider=_FailingProvider(),
    )
    completed = False
    with tracer.start_span(
        "controller.advance",
        attributes={"task_index": 0},
    ) as span:
        span.add_event("controller.failure", {"reason_code": "EXPORT_FAILED"})
        completed = True
    tracer.shutdown()
    assert completed is True


def test_only_allowlisted_bounded_attributes_and_events_reach_delegate() -> None:
    recorded = _RecordingSpan()
    delegate = _RecordingTracer(recorded)
    tracer = OpenTelemetryHarnessTracer(tracer=delegate, provider=_FailingProvider())

    with tracer.start_span(
        "controller.action",
        attributes={"task_id": "inspect_dataset", "task_index": 0},
    ) as span:
        span.set_attribute("outcome", "committed")
        span.set_attribute("receipt_digest", "sha256:" + "b" * 64)
        span.set_attribute("raw_path", "/private/data")
        span.add_event("controller.receipt", {"status": "active"})
        span.add_event("raw.event", {"status": "active"})

    assert recorded.attributes == {
        "outcome": "committed",
        "receipt_digest": "sha256:" + "b" * 64,
    }
    assert recorded.events == [("controller.receipt", {"status": "active"})]
    assert delegate.calls[0][1]["attributes"] == {
        "task_id": "inspect_dataset",
        "task_index": 0,
    }
    assert delegate.calls[0][1]["record_exception"] is False
    assert delegate.calls[0][1]["set_status_on_exception"] is False


def test_safe_links_and_event_bounds_do_not_accept_private_payloads() -> None:
    link = HarnessSpanLink(trace_id="a" * 32, span_id="b" * 16)
    with NoopHarnessTracer().start_span(
        "controller.advance",
        attributes={"controller_execution_id": "controller-a"},
        links=[link],
    ):
        pass

    recorded = _RecordingSpan()
    delegate = _RecordingTracer(recorded)
    tracer = OpenTelemetryHarnessTracer(tracer=delegate, provider=_FailingProvider())

    with tracer.start_span(
        "controller.advance",
        attributes={"controller_execution_id": "controller-a"},
    ) as span:
        for index in range(40):
            span.add_event("controller.waiting", {"retry_count": index})
        span.set_attribute("task_id", "/private/secret.csv")
        span.set_attribute("task_id", "user@example.com")

    assert len(recorded.events) == 32
    assert "task_id" not in recorded.attributes
    assert delegate.calls[0][1]["links"] == []
