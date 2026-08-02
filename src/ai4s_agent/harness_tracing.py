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
from typing import Any, Mapping, Protocol, Sequence


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
        "execution_agent.propose",
        "execution_agent.observe",
        "execution_agent.llm_call",
        "execution_agent.validate_response",
        "execution_agent.publish_proposal",
        "execution_agent.apply",
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
_MAX_LINKS = 16
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


class HarnessTracingError(ValueError):
    """A caller attempted to put non-allowlisted data into tracing."""


class HarnessSpan(Protocol):
    def set_attribute(self, key: str, value: str | int) -> None: ...

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int] | None = None,
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
        attributes: Mapping[str, str | int] | None = None,
        links: Sequence[HarnessSpanLink] = (),
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

    def record_error(self, reason_code: str) -> None:
        self.add_event("controller.failure", {"reason_code": reason_code})


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
        attributes: Mapping[str, str | int] | None = None,
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


def _validate_links(links: Sequence[HarnessSpanLink]) -> tuple[HarnessSpanLink, ...]:
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        raise HarnessTracingError("tracing links must be a bounded sequence")
    if len(links) > _MAX_LINKS or any(not isinstance(item, HarnessSpanLink) for item in links):
        raise HarnessTracingError("tracing links must be bounded safe Harness links")
    return tuple(links)


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

    def record_error(self, reason_code: str) -> None:
        self.add_event("controller.failure", {"reason_code": reason_code})


class _OpenTelemetrySpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(
        self,
        tracer: Any,
        name: str,
        attributes: dict[str, str | int],
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
                self._name,
                attributes=self._attributes,
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

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str | int] | None = None,
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
    "HarnessSpanLink",
    "HarnessTracer",
    "HarnessTracingError",
    "NoopHarnessTracer",
    "OpenTelemetryHarnessTracer",
    "build_harness_tracer",
]
