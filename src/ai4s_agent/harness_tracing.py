"""Fail-open, privacy-bounded tracing seam for the Scientific Agent Harness.

Business code depends only on :class:`HarnessTracer`.  OpenTelemetry imports
remain lazy and optional, and no tracing value participates in authority,
idempotency, recovery, or result verification.
"""

from __future__ import annotations

import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


_SPAN_NAMES = frozenset(
    {
        "controller.execution",
        "controller.advance",
        "controller.action",
        "executor.local_task",
        "remote.prepare",
        "remote.dispatch",
        "remote.refresh",
        "remote.recover",
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
        "authority_digest",
        "decision_digest",
        "receipt_digest",
    }
)
_INTEGER_ATTRIBUTE_KEYS = frozenset(
    {
        "task_index",
        "attempt",
        "duration_ms",
        "retry_count",
        "output_count",
    }
)
_ATTRIBUTE_KEYS = _STRING_ATTRIBUTE_KEYS | _INTEGER_ATTRIBUTE_KEYS
_SAFE_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ATTRIBUTES = 32
_MAX_EVENTS = 32


class HarnessTracingError(ValueError):
    """A caller attempted to put non-allowlisted data into tracing."""


class HarnessSpan(Protocol):
    def set_attribute(self, key: str, value: str | int) -> None: ...

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> None: ...


class HarnessTracer(Protocol):
    def start_span(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> AbstractContextManager[HarnessSpan]: ...

    def shutdown(self) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: str | int) -> None:
        try:
            _validate_attribute(key, value)
        except HarnessTracingError:
            return None

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> None:
        try:
            _validate_event(name, attributes)
        except HarnessTracingError:
            return None


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
        attributes: Mapping[str, str | int] | None = None,
    ) -> AbstractContextManager[HarnessSpan]:
        try:
            _validate_span(name, attributes)
        except HarnessTracingError:
            return _NoopSpanContext()
        return _NoopSpanContext()

    def shutdown(self) -> None:
        return None


def _validate_attribute(key: str, value: str | int) -> tuple[str, str | int]:
    if key not in _ATTRIBUTE_KEYS:
        raise HarnessTracingError("tracing attribute key is not allowlisted")
    if key in _INTEGER_ATTRIBUTE_KEYS:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
            raise HarnessTracingError("tracing numeric attribute is invalid")
        return key, value
    if not isinstance(value, str) or _SAFE_VALUE_PATTERN.fullmatch(value) is None:
        raise HarnessTracingError("tracing string attribute is not a bounded safe label")
    if key.endswith("_digest") and _DIGEST_PATTERN.fullmatch(value) is None:
        raise HarnessTracingError("tracing digest attribute is invalid")
    return key, value


def _validate_attributes(
    attributes: Mapping[str, str | int] | None,
) -> dict[str, str | int]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping) or len(attributes) > _MAX_ATTRIBUTES:
        raise HarnessTracingError("tracing attributes must be a bounded mapping")
    return dict(_validate_attribute(key, value) for key, value in attributes.items())


def _validate_span(
    name: str,
    attributes: Mapping[str, str | int] | None,
) -> dict[str, str | int]:
    if name not in _SPAN_NAMES:
        raise HarnessTracingError("tracing span name is not allowlisted")
    return _validate_attributes(attributes)


def _validate_event(
    name: str,
    attributes: Mapping[str, str | int] | None,
) -> dict[str, str | int]:
    if name not in _EVENT_NAMES:
        raise HarnessTracingError("tracing event name is not allowlisted")
    return _validate_attributes(attributes)


class _OpenTelemetrySpan:
    def __init__(self, span: Any) -> None:
        self._span = span
        self._events = 0

    def set_attribute(self, key: str, value: str | int) -> None:
        try:
            validated_key, validated_value = _validate_attribute(key, value)
            self._span.set_attribute(validated_key, validated_value)
        except Exception:
            return None

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> None:
        if self._events >= _MAX_EVENTS:
            return None
        try:
            validated = _validate_event(name, attributes)
            self._events += 1
            self._span.add_event(name, attributes=validated)
        except Exception:
            return None


class _OpenTelemetrySpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(self, tracer: Any, name: str, attributes: dict[str, str | int]) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._delegate: Any = None
        self._span: HarnessSpan = _NoopSpan()

    def __enter__(self) -> HarnessSpan:
        try:
            self._delegate = self._tracer.start_as_current_span(
                self._name,
                attributes=self._attributes,
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

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
    ) -> AbstractContextManager[HarnessSpan]:
        try:
            validated = _validate_span(name, attributes)
        except HarnessTracingError:
            return _NoopSpanContext()
        return _OpenTelemetrySpanContext(self.tracer, name, validated)

    def shutdown(self) -> None:
        try:
            self.provider.shutdown()
        except Exception:
            return None


def build_harness_tracer(
    environ: Mapping[str, str] | None = None,
) -> HarnessTracer:
    """Build optional OTLP tracing, degrading safely to no-op on any failure."""

    values = os.environ if environ is None else environ
    enabled = str(values.get("AI4S_HARNESS_OTEL_ENABLED", "")).strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return NoopHarnessTracer()
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": "molly-scientific-agent-harness"})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        tracer = provider.get_tracer("ai4s_agent.harness", "1")
        return OpenTelemetryHarnessTracer(tracer=tracer, provider=provider)
    except Exception:
        return NoopHarnessTracer()


__all__ = [
    "HarnessSpan",
    "HarnessTracer",
    "HarnessTracingError",
    "NoopHarnessTracer",
    "OpenTelemetryHarnessTracer",
    "build_harness_tracer",
]
