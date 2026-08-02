"""Fail-open, privacy-bounded tracing seam for the Scientific Agent Harness.

Business code depends only on :class:`HarnessTracer`.  OpenTelemetry imports
remain lazy and optional, and no tracing value participates in authority,
idempotency, recovery, or result verification.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ai4s_agent.observability_config import (
    HarnessObservabilityConfig,
    HarnessTelemetryHealth,
)


_SPAN_NAMES = frozenset(
    {
        "controller.execution",
        "controller.advance",
        "controller.action",
        "executor.local_task",
        "remote.prepare",
        "remote.await_approval",
        "remote.dispatch",
        "remote.refresh",
        "remote.recover",
        "remote.adopt",
        "execution_agent.propose",
        "execution_agent.observe",
        "execution_agent.llm_call",
        "execution_agent.validate_response",
        "execution_agent.publish_proposal",
        "execution_agent.apply",
        "replanner.feedback",
        "replanner.observe",
        "replanner.llm_call",
        "replanner.compile",
        "replanner.publish",
        "replanner.apply",
        "planner.propose",
        "planner.llm_call",
        "permission.evaluate",
        "authorization.create",
        "start_intent.create",
        "run_inspection.read",
        "dataset.inspect",
        "dataset.clean",
        "dataset.confirm",
        "model.train",
        "candidate.generate",
        "candidate.predict",
        "candidate.rank",
        "candidate.validate",
    }
)
_EVENT_NAMES = frozenset(
    {
        "controller.decision",
        "controller.receipt",
        "controller.waiting",
        "controller.conflict",
        "controller.failure",
        "executor.completed",
        "remote.transition",
        "execution_agent.observation_frozen",
        "execution_agent.llm_response_validated",
        "execution_agent.proposal_committed",
        "execution_agent.proposal_stale",
        "execution_agent.controller_advance_applied",
        "execution_agent.user_action_required",
        "execution_agent.application_reconciled",
        "replanner.feedback_recorded",
        "replanner.observation_frozen",
        "replanner.provider_outcome_committed",
        "replanner.proposal_committed",
        "replanner.no_material_change",
        "replanner.successor_committed",
        "replanner.application_committed",
        "planner.provider_completed",
        "planner.provider_failed",
        "permission.decision",
        "authorization.committed",
        "start_intent.committed",
        "run_inspection.completed",
        "telemetry.error",
    }
)
_STRING_ATTRIBUTE_KEYS = frozenset(
    {
        "schema_version",
        "controller_policy_version",
        "controller_execution_id",
        "action_id",
        "decision_id",
        "receipt_id",
        "run_id",
        "task_id",
        "slot_id",
        "execution_route",
        "action",
        "outcome",
        "status",
        "gate_id",
        "remote_task_type",
        "execution_profile_id",
        "reason_code",
        "tool_call_proposal_id",
        "selected_tool_id",
        "controller_status",
        "next_controller_action",
        "current_task_id",
        "application_outcome",
        "inspection_digest",
        "observation_digest",
        "tool_catalog_digest",
        "application_receipt_digest",
        "authority_digest",
        "decision_digest",
        "receipt_digest",
        "feedback_receipt_id",
        "revision_id",
        "revision_digest",
        "plan_diff_digest",
        "successor_proposal_id",
        "project_id",
        "proposal_id",
        "proposal_digest",
        "semantic_plan_id",
        "semantic_plan_digest",
        "permission_decision_id",
        "authorization_id",
        "start_intent_id",
        "controller_execution_digest",
        "tool_call_application_receipt_id",
        "plan_diff_id",
        "revision_application_receipt_id",
        "gate_snapshot_id",
        "gate_decision_digest",
        "publication_id",
        "publication_digest",
        "operation",
        "component",
        "phase",
        "authority_class",
        "provider_kind",
        "provider_model_digest",
        "request_digest",
        "response_digest",
        "exception_type_code",
        "content_mode",
    }
)
_INTEGER_ATTRIBUTE_KEYS = frozenset(
    {
        "task_index",
        "attempt",
        "duration_ms",
        "retry_count",
        "output_count",
        "controller_revision",
    }
)
_BOOLEAN_ATTRIBUTE_KEYS = frozenset({"telemetry_authoritative"})
_ATTRIBUTE_KEYS = _STRING_ATTRIBUTE_KEYS | _INTEGER_ATTRIBUTE_KEYS | _BOOLEAN_ATTRIBUTE_KEYS
_NAMESPACED_ATTRIBUTE_KEYS = frozenset(
    f"molly.{key}"
    for key in _ATTRIBUTE_KEYS | {"schema_version"}
)
_SAFE_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ATTRIBUTES = 48
_MAX_EVENTS = 32
_MAX_LINKS = 16
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")

_EXPORTED_SPAN_NAMES = {
    "planner.propose": "agent.plan.propose",
    "planner.llm_call": "agent.plan.llm_call",
    "permission.evaluate": "agent.permission.evaluate",
    "authorization.create": "agent.authorization.create",
    "start_intent.create": "agent.start_intent.create",
    "controller.execution": "agent.controller.create",
    "controller.advance": "agent.controller.advance",
    "controller.action": "agent.controller.inspect",
    "executor.local_task": "agent.execution.local",
    "remote.prepare": "agent.execution.remote.prepare",
    "remote.await_approval": "agent.execution.remote.await_approval",
    "remote.dispatch": "agent.execution.remote.dispatch",
    "remote.refresh": "agent.execution.remote.refresh",
    "remote.recover": "agent.execution.remote.recover",
    "remote.adopt": "agent.execution.remote.adopt",
    "execution_agent.propose": "agent.execution_agent.propose",
    "execution_agent.llm_call": "agent.execution_agent.llm_call",
    "execution_agent.apply": "agent.execution_agent.apply",
    "replanner.feedback": "agent.replanner.feedback",
    "replanner.llm_call": "agent.replanner.llm_call",
    "replanner.publish": "agent.replanner.create_revision",
    "replanner.apply": "agent.replanner.apply_revision",
    "run_inspection.read": "agent.run_inspection.read",
    "dataset.inspect": "agent.dataset.inspect",
    "dataset.clean": "agent.dataset.clean",
    "dataset.confirm": "agent.dataset.confirm",
    "model.train": "agent.model.train",
    "candidate.generate": "agent.candidate.generate",
    "candidate.predict": "agent.candidate.predict",
    "candidate.rank": "agent.candidate.rank",
    "candidate.validate": "agent.candidate.validate",
}


class _PrivacySafeOTelLogFilter(logging.Filter):
    """Replace vendor-created log material before handler serialization."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "MOLLY_OTEL_VENDOR_LOG_REDACTED"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _install_privacy_safe_otel_log_filter(module_name: str) -> None:
    logger = logging.getLogger(module_name)
    if any(isinstance(item, _PrivacySafeOTelLogFilter) for item in logger.filters):
        return
    logger.addFilter(_PrivacySafeOTelLogFilter())


class HarnessTracingError(ValueError):
    """A caller attempted to put non-allowlisted data into tracing."""


class HarnessSpan(Protocol):
    def set_attribute(self, key: str, value: str | int | bool) -> None: ...

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int | bool] | None = None,
    ) -> None: ...

    def record_error(self, reason_code: str) -> None: ...


@dataclass(frozen=True)
class HarnessSpanLink:
    """Non-authoritative safe correlation to a completed Harness action."""

    trace_id: str
    span_id: str

    def __post_init__(self) -> None:
        if _TRACE_ID_PATTERN.fullmatch(self.trace_id) is None:
            raise HarnessTracingError("tracing link trace ID is invalid")
        if _SPAN_ID_PATTERN.fullmatch(self.span_id) is None:
            raise HarnessTracingError("tracing link span ID is invalid")


class HarnessTracer(Protocol):
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int | bool] | None = None,
        links: Sequence[HarnessSpanLink] = (),
    ) -> AbstractContextManager[HarnessSpan]: ...

    def shutdown(self) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: str | int | bool) -> None:
        try:
            _validate_attribute(key, value)
        except HarnessTracingError:
            return None

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int | bool] | None = None,
    ) -> None:
        try:
            _validate_event(name, attributes)
        except HarnessTracingError:
            return None

    def record_error(self, reason_code: str) -> None:
        self.add_event("telemetry.error", {"reason_code": reason_code})


class _NoopSpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(self, span: HarnessSpan | None = None) -> None:
        self._span = span or _NoopSpan()

    def __enter__(self) -> HarnessSpan:
        return self._span

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False


class NoopHarnessTracer:
    """Default tracer. It validates caller privacy without exporting anything."""

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int | bool] | None = None,
        links: Sequence[HarnessSpanLink] = (),
    ) -> AbstractContextManager[HarnessSpan]:
        try:
            _validate_span(name, attributes)
            _validate_links(links)
        except HarnessTracingError:
            return _NoopSpanContext()
        return _NoopSpanContext()

    def shutdown(self) -> None:
        return None


def _validate_attribute(
    key: str,
    value: str | int | bool,
) -> tuple[str, str | int | bool]:
    raw_key = key.removeprefix("molly.") if key.startswith("molly.") else key
    if key not in _NAMESPACED_ATTRIBUTE_KEYS and raw_key not in _ATTRIBUTE_KEYS:
        raise HarnessTracingError("tracing attribute key is not allowlisted")
    if raw_key in _BOOLEAN_ATTRIBUTE_KEYS:
        if not isinstance(value, bool):
            raise HarnessTracingError("tracing boolean attribute is invalid")
        if raw_key == "telemetry_authoritative" and value is not False:
            raise HarnessTracingError("telemetry cannot be authoritative")
        return key, value
    if raw_key in _INTEGER_ATTRIBUTE_KEYS:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
            raise HarnessTracingError("tracing numeric attribute is invalid")
        return key, value
    if not isinstance(value, str) or _SAFE_VALUE_PATTERN.fullmatch(value) is None:
        raise HarnessTracingError("tracing string attribute is not a bounded safe label")
    if raw_key.endswith("_digest") and _DIGEST_PATTERN.fullmatch(value) is None:
        raise HarnessTracingError("tracing digest attribute is invalid")
    return key, value


def _validate_attributes(
    attributes: Mapping[str, str | int | bool] | None,
) -> dict[str, str | int | bool]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping) or len(attributes) > _MAX_ATTRIBUTES:
        raise HarnessTracingError("tracing attributes must be a bounded mapping")
    return dict(_validate_attribute(key, value) for key, value in attributes.items())


def _validate_span(
    name: str,
    attributes: Mapping[str, str | int | bool] | None,
) -> dict[str, str | int | bool]:
    if name not in _SPAN_NAMES:
        raise HarnessTracingError("tracing span name is not allowlisted")
    return _validate_attributes(attributes)


def _validate_event(
    name: str,
    attributes: Mapping[str, str | int | bool] | None,
) -> dict[str, str | int | bool]:
    if name not in _EVENT_NAMES:
        raise HarnessTracingError("tracing event name is not allowlisted")
    return _validate_attributes(attributes)


def _validate_links(links: Sequence[HarnessSpanLink]) -> tuple[HarnessSpanLink, ...]:
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        raise HarnessTracingError("tracing links must be a bounded sequence")
    if len(links) > _MAX_LINKS or any(not isinstance(item, HarnessSpanLink) for item in links):
        raise HarnessTracingError("tracing links must be bounded safe Harness links")
    return tuple(links)


def _export_span_name(name: str) -> str:
    return _EXPORTED_SPAN_NAMES.get(name, f"agent.{name}")


def _export_attributes(
    attributes: Mapping[str, str | int | bool],
) -> dict[str, str | int | bool]:
    exported: dict[str, str | int | bool] = {
        "molly.schema_version": "harness_telemetry_correlation.v1",
        "molly.telemetry_authoritative": False,
    }
    for key, value in attributes.items():
        exported[key if key.startswith("molly.") else f"molly.{key}"] = value
    if len(exported) > _MAX_ATTRIBUTES:
        raise HarnessTracingError("exported tracing attributes exceed the bound")
    return exported


class _PrivacySafeOTelExporter:
    """Swallow delegate failures before the OTel SDK can log raw details."""

    def __init__(
        self,
        *,
        delegate: Any,
        health: HarnessTelemetryHealth,
        success_result: Any,
        failure_result: Any,
    ) -> None:
        self.delegate = delegate
        self.health = health
        self.success_result = success_result
        self.failure_result = failure_result
        self._state_lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence[Any]) -> Any:
        with self._state_lock:
            if self._shutdown:
                self.health.otel_result(
                    "OTEL_EXPORTER_SHUTDOWN", available=False
                )
                return self.failure_result
        try:
            result = self.delegate.export(spans)
        except Exception:
            self.health.otel_result("OTEL_EXPORT_FAILED", available=False)
            return self.failure_result
        if result != self.success_result:
            self.health.otel_result("OTEL_EXPORT_FAILED", available=False)
            return self.failure_result
        self.health.otel_result("OTEL_EXPORT_COMPLETED", available=True)
        return self.success_result

    def shutdown(self, timeout_millis: int | None = None) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        try:
            if timeout_millis is None:
                self.delegate.shutdown()
            else:
                try:
                    self.delegate.shutdown(timeout_millis=timeout_millis)
                except TypeError:
                    self.delegate.shutdown()
        except Exception:
            self.health.otel_result(
                "OTEL_EXPORTER_SHUTDOWN_FAILED", available=False
            )


class _PrivacySafeBatchSpanProcessor:
    """Bounded non-blocking processor with no raw SDK exception logging."""

    def __init__(
        self,
        *,
        exporter: _PrivacySafeOTelExporter,
        health: HarnessTelemetryHealth,
        max_queue_size: int,
        schedule_delay_millis: int,
        max_export_batch_size: int,
    ) -> None:
        self.exporter = exporter
        self.health = health
        self.max_queue_size = max_queue_size
        self.max_export_batch_size = max_export_batch_size
        self.schedule_delay_seconds = schedule_delay_millis / 1000.0
        self._queue: deque[Any] = deque()
        self._condition = threading.Condition()
        self._export_lock = threading.Lock()
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._run,
            name="molly-otel-batch-export",
            daemon=True,
        )
        self._worker.start()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del span, parent_context

    def _on_ending(self, span: Any) -> None:
        del span

    def on_end(self, span: Any) -> None:
        context = getattr(span, "context", None)
        trace_flags = getattr(context, "trace_flags", None)
        if not bool(getattr(trace_flags, "sampled", False)):
            return
        with self._condition:
            if self._shutdown:
                self.health.dropped(
                    reason_code="OTEL_PROCESSOR_SHUTDOWN",
                    vendor="otel",
                )
                return
            if len(self._queue) >= self.max_queue_size:
                self.health.dropped(
                    reason_code="OTEL_QUEUE_FULL",
                    vendor="otel",
                )
                return
            self._queue.append(span)
            self._condition.notify()

    def _take_batch(self) -> list[Any]:
        with self._condition:
            count = min(len(self._queue), self.max_export_batch_size)
            return [self._queue.popleft() for _ in range(count)]

    def _export(self, batch: Sequence[Any]) -> None:
        if not batch:
            return
        try:
            with self._export_lock:
                self.exporter.export(batch)
        except Exception:
            # Defensive boundary: the safe exporter itself must never affect
            # the worker or print a raw delegate exception.
            self.health.otel_result("OTEL_EXPORT_FAILED", available=False)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._shutdown:
                    self._condition.wait()
                if self._shutdown and not self._queue:
                    return
                if (
                    not self._shutdown
                    and len(self._queue) < self.max_export_batch_size
                ):
                    self._condition.wait(self.schedule_delay_seconds)
                batch = [
                    self._queue.popleft()
                    for _ in range(
                        min(len(self._queue), self.max_export_batch_size)
                    )
                ]
            self._export(batch)

    def force_flush(self, timeout_millis: int = 500) -> bool:
        completed = threading.Event()

        def flush() -> None:
            try:
                while True:
                    batch = self._take_batch()
                    if not batch:
                        break
                    self._export(batch)
                # Synchronize with a batch already removed by the background
                # worker before reporting a successful flush.
                with self._export_lock:
                    pass
            finally:
                completed.set()

        worker = threading.Thread(
            target=flush,
            name="molly-otel-force-flush",
            daemon=True,
        )
        worker.start()
        if completed.wait(max(0.0, timeout_millis / 1000.0)):
            return True
        self.health.otel_result("OTEL_FORCE_FLUSH_TIMEOUT", available=False)
        return False

    def shutdown(self, timeout_millis: int = 500) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        self._worker.join(max(0.0, timeout_millis / 1000.0))
        if self._worker.is_alive():
            self.health.otel_result(
                "OTEL_PROCESSOR_SHUTDOWN_TIMEOUT", available=False
            )
            return
        self.exporter.shutdown(timeout_millis=timeout_millis)


class _OpenTelemetrySpan:
    def __init__(self, span: Any) -> None:
        self._span = span
        self._events = 0

    def set_attribute(self, key: str, value: str | int | bool) -> None:
        try:
            validated_key, validated_value = _validate_attribute(key, value)
            exported = _export_attributes({validated_key: validated_value})
            for export_key, export_value in exported.items():
                self._span.set_attribute(export_key, export_value)
        except Exception:
            return None

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int | bool] | None = None,
    ) -> None:
        if self._events >= _MAX_EVENTS:
            return None
        try:
            validated = _validate_event(name, attributes)
            self._events += 1
            self._span.add_event(name, attributes=_export_attributes(validated))
        except Exception:
            return None

    def record_error(self, reason_code: str) -> None:
        self.add_event("telemetry.error", {"reason_code": reason_code})


class _OpenTelemetrySpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(
        self,
        tracer: Any,
        name: str,
        attributes: dict[str, str | int | bool],
        links: tuple[HarnessSpanLink, ...],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._links = links
        self._delegate: Any = None
        self._span: HarnessSpan = _NoopSpan()

    def __enter__(self) -> HarnessSpan:
        try:
            otel_links = []
            if self._links:
                from opentelemetry.trace import Link, NonRecordingSpan, SpanContext, TraceFlags

                for item in self._links:
                    context = SpanContext(
                        trace_id=int(item.trace_id, 16),
                        span_id=int(item.span_id, 16),
                        is_remote=False,
                        trace_flags=TraceFlags(0),
                    )
                    otel_links.append(Link(NonRecordingSpan(context).get_span_context()))
            self._delegate = self._tracer.start_as_current_span(
                _export_span_name(self._name),
                attributes=_export_attributes(self._attributes),
                links=otel_links,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._span = _OpenTelemetrySpan(self._delegate.__enter__())
        except Exception:
            self._delegate = None
            self._span = _NoopSpan()
        return self._span

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._delegate is not None:
            try:
                # Never give OpenTelemetry the exception object or message.
                self._delegate.__exit__(None, None, None)
            except Exception:
                pass
        return False


@dataclass
class OpenTelemetryHarnessTracer:
    """Small fail-open adapter around a private OpenTelemetry provider."""

    tracer: Any
    provider: Any
    health: HarnessTelemetryHealth | None = None
    shutdown_timeout_seconds: float = 1.0

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int | bool] | None = None,
        links: Sequence[HarnessSpanLink] = (),
    ) -> AbstractContextManager[HarnessSpan]:
        try:
            validated = _validate_span(name, attributes)
            validated_links = _validate_links(links)
        except HarnessTracingError:
            return _NoopSpanContext()
        return _OpenTelemetrySpanContext(
            self.tracer,
            name,
            validated,
            validated_links,
        )

    def shutdown(self) -> None:
        completed = threading.Event()

        def close_provider() -> None:
            try:
                force_flush = getattr(self.provider, "force_flush", None)
                if callable(force_flush):
                    force_flush(timeout_millis=500)
                self.provider.shutdown()
                if self.health is not None:
                    self.health.otel_result("OTEL_SHUTDOWN_COMPLETED", available=True)
            except Exception:
                if self.health is not None:
                    self.health.otel_result("OTEL_SHUTDOWN_FAILED", available=False)
            finally:
                completed.set()

        worker = threading.Thread(
            target=close_provider,
            name="molly-otel-shutdown",
            daemon=True,
        )
        worker.start()
        if not completed.wait(max(0.0, self.shutdown_timeout_seconds)):
            if self.health is not None:
                self.health.otel_result("OTEL_SHUTDOWN_TIMEOUT", available=False)


class _CompositeSpan:
    def __init__(self, spans: Sequence[HarnessSpan]) -> None:
        self._spans = tuple(spans)

    def set_attribute(self, key: str, value: str | int | bool) -> None:
        for span in self._spans:
            try:
                span.set_attribute(key, value)
            except Exception:
                continue

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int | bool] | None = None,
    ) -> None:
        for span in self._spans:
            try:
                span.add_event(name, attributes)
            except Exception:
                continue

    def record_error(self, reason_code: str) -> None:
        for span in self._spans:
            try:
                span.record_error(reason_code)
            except Exception:
                continue


class _CompositeSpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(
        self,
        contexts: Sequence[AbstractContextManager[HarnessSpan]],
    ) -> None:
        self._contexts = tuple(contexts)
        self._entered: list[AbstractContextManager[HarnessSpan]] = []

    def __enter__(self) -> HarnessSpan:
        spans: list[HarnessSpan] = []
        for context in self._contexts:
            try:
                spans.append(context.__enter__())
                self._entered.append(context)
            except Exception:
                continue
        return _CompositeSpan(spans or [_NoopSpan()])

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        for context in reversed(self._entered):
            try:
                # Adapters receive only the exception class. The exception object,
                # traceback, message, paths, and credentials remain private.
                context.__exit__(exc_type, None, None)
            except Exception:
                continue
        return False


@dataclass
class CompositeHarnessTracer:
    """Fan out to optional vendors without allowing cross-adapter failure."""

    tracers: Sequence[HarnessTracer]

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int | bool] | None = None,
        links: Sequence[HarnessSpanLink] = (),
    ) -> AbstractContextManager[HarnessSpan]:
        contexts: list[AbstractContextManager[HarnessSpan]] = []
        for tracer in self.tracers:
            try:
                contexts.append(
                    tracer.start_span(name, attributes=attributes, links=links)
                )
            except Exception:
                continue
        return _CompositeSpanContext(contexts)

    def shutdown(self) -> None:
        for tracer in self.tracers:
            try:
                tracer.shutdown()
            except Exception:
                continue


def build_harness_tracer(
    environ: Mapping[str, str] | None = None,
) -> HarnessTracer:
    """Compatibility builder for the complete optional observability fanout."""

    tracer, _ = build_harness_observability(environ=environ)
    return tracer


def _build_otel_tracer(
    *,
    config: HarnessObservabilityConfig,
    health: HarnessTelemetryHealth,
) -> HarnessTracer | None:
    if config.otel_mode == "disabled":
        return None
    try:
        if config.otel_mode == "otlp_grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter_log_modules = (
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
                # Credential reads, endpoint/retry failures, RpcError details,
                # and tracebacks are emitted by the shared exporter mixin.
                "opentelemetry.exporter.otlp.proto.grpc.exporter",
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter_log_modules = (
                "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            )
        # Install before exporter construction: the gRPC SDK can log private
        # certificate paths while resolving credentials in __init__.
        for exporter_log_module in exporter_log_modules:
            _install_privacy_safe_otel_log_filter(exporter_log_module)
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SpanExportResult

        provider = TracerProvider(
            # Direct construction intentionally bypasses default/environment
            # detectors such as OTEL_RESOURCE_ATTRIBUTES.
            resource=Resource(
                {"service.name": "molly-scientific-agent-harness"}
            )
        )
        exporter = _PrivacySafeOTelExporter(
            delegate=OTLPSpanExporter(timeout=5.0),
            health=health,
            success_result=SpanExportResult.SUCCESS,
            failure_result=SpanExportResult.FAILURE,
        )
        provider.add_span_processor(
            _PrivacySafeBatchSpanProcessor(
                exporter=exporter,
                health=health,
                max_queue_size=2048,
                schedule_delay_millis=5000,
                max_export_batch_size=512,
            )
        )
        tracer = provider.get_tracer("ai4s_agent.harness", "1")
        health.otel_result("OTEL_READY", available=True)
        return OpenTelemetryHarnessTracer(
            tracer=tracer,
            provider=provider,
            health=health,
        )
    except Exception:
        health.otel_result("OTEL_INITIALIZATION_FAILED", available=False)
        return None


def build_harness_observability(
    *,
    environ: Mapping[str, str] | None = None,
    langsmith_client_factory: Any | None = None,
    otel_tracer_factory: Any | None = None,
) -> tuple[HarnessTracer, HarnessTelemetryHealth]:
    """Build independent optional vendors; any failure degrades to safe no-op."""

    config = HarnessObservabilityConfig.from_environ(environ)
    health = HarnessTelemetryHealth(config=config)
    tracers: list[HarnessTracer] = []
    try:
        otel = (otel_tracer_factory or _build_otel_tracer)(
            config=config,
            health=health,
        )
    except Exception:
        health.otel_result("OTEL_INITIALIZATION_FAILED", available=False)
        otel = None
    if otel is not None:
        tracers.append(otel)
    if config.langsmith_mode != "disabled":
        try:
            from ai4s_agent.langsmith_adapter import build_langsmith_harness_tracer

            langsmith = build_langsmith_harness_tracer(
                config=config,
                health=health,
                client_factory=langsmith_client_factory,
            )
            if langsmith is not None:
                tracers.append(langsmith)
        except Exception:
            health.langsmith_result(
                "LANGSMITH_INITIALIZATION_FAILED", available=False
            )
    if not tracers:
        return NoopHarnessTracer(), health
    if len(tracers) == 1:
        return tracers[0], health
    return CompositeHarnessTracer(tuple(tracers)), health


__all__ = [
    "HarnessSpan",
    "HarnessSpanLink",
    "HarnessTracer",
    "HarnessTracingError",
    "CompositeHarnessTracer",
    "NoopHarnessTracer",
    "OpenTelemetryHarnessTracer",
    "build_harness_observability",
    "build_harness_tracer",
]
