"""Optional metadata-only LangSmith adapter over the shared HarnessTracer seam."""

from __future__ import annotations

import threading
import uuid
import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai4s_agent.harness_tracing import (
    HarnessSpan,
    HarnessSpanLink,
    HarnessTracingError,
    _NoopSpanContext,
    _export_attributes,
    _export_span_name,
    _validate_attribute,
    _validate_event,
    _validate_links,
    _validate_span,
    _NAMESPACED_ATTRIBUTE_KEYS,
)
from ai4s_agent.observability_config import (
    HarnessObservabilityConfig,
    HarnessTelemetryHealth,
)
from ai4s_agent.observability_correlation import safe_exception_type_code


_LANGSMITH_LLM_SPANS = frozenset(
    {
        "planner.llm_call",
        "execution_agent.llm_call",
        "replanner.llm_call",
        "document.contextual_mapping.llm_call",
    }
)
_MAX_EVENTS = 32
_LANGSMITH_PROJECT_NAME = "molly-scientific-agent-harness"


def _configured_project_name() -> str:
    value = str(os.environ.get("LANGSMITH_PROJECT") or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        return value
    return _LANGSMITH_PROJECT_NAME


def _langsmith_safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Final SDK serialization guard for the frozen Molly namespace."""

    safe: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _NAMESPACED_ATTRIBUTE_KEYS:
            continue
        try:
            raw_key = key.removeprefix("molly.")
            _, validated = _validate_attribute(raw_key, item)
        except Exception:
            continue
        safe[key] = validated
    return safe


def _langsmith_safe_outputs(value: dict[str, Any]) -> dict[str, Any]:
    """Allow only fixed terminal classifications at the SDK send boundary."""

    safe: dict[str, Any] = {}
    outcome = value.get("outcome")
    if outcome in {"completed", "failed"}:
        safe["outcome"] = outcome
    exception_type = value.get("exception_type_code")
    try:
        _, validated = _validate_attribute(
            "exception_type_code", exception_type
        )
    except Exception:
        pass
    else:
        safe["exception_type_code"] = validated
    return safe


class _LangSmithSpan:
    def __init__(self, *, attributes: dict[str, str | int | bool]) -> None:
        self.attributes = attributes
        self.events: list[tuple[str, dict[str, str | int | bool]]] = []

    def set_attribute(self, key: str, value: str | int | bool) -> None:
        try:
            validated_key, validated_value = _validate_attribute(key, value)
            self.attributes[validated_key] = validated_value
        except Exception:
            return None

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, str | int | bool] | None = None,
    ) -> None:
        if len(self.events) >= _MAX_EVENTS:
            return None
        try:
            self.events.append((name, _validate_event(name, attributes)))
        except Exception:
            return None

    def record_error(self, reason_code: str) -> None:
        self.add_event("telemetry.error", {"reason_code": reason_code})


class _LangSmithSpanContext(AbstractContextManager[HarnessSpan]):
    def __init__(
        self,
        *,
        client: Any,
        name: str,
        attributes: dict[str, str | int | bool],
        mode: str,
        health: HarnessTelemetryHealth,
        project_name: str,
    ) -> None:
        self.client = client
        self.name = name
        self.mode = mode
        self.health = health
        self.project_name = project_name
        self.span = _LangSmithSpan(attributes=attributes)
        self.run_id: uuid.UUID | None = None

    def __enter__(self) -> HarnessSpan:
        run_id = uuid.uuid4()
        try:
            self.client.create_run(
                id=run_id,
                name=_export_span_name(self.name),
                project_name=self.project_name,
                run_type="llm",
                inputs={},
                extra={
                    "metadata": _export_attributes(
                        {**self.span.attributes, "content_mode": self.mode}
                    )
                },
            )
            self.run_id = run_id
            self.health.langsmith_result("LANGSMITH_RUN_STARTED", available=True)
        except Exception:
            self.run_id = None
            self.health.langsmith_result(
                "LANGSMITH_RUN_CREATE_FAILED", available=False
            )
        return self.span

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self.run_id is None:
            return False
        outcome = "failed" if exc_type is not None else "completed"
        safe_events = [
            {
                "name": name,
                "attributes": _export_attributes(attributes),
            }
            for name, attributes in self.span.events
        ]
        try:
            self.client.update_run(
                self.run_id,
                outputs={
                    "outcome": outcome,
                    "exception_type_code": safe_exception_type_code(exc_type),
                },
                extra={
                    "metadata": _export_attributes(self.span.attributes),
                    "events": safe_events,
                },
            )
            self.health.langsmith_result("LANGSMITH_RUN_COMPLETED", available=True)
        except Exception:
            self.health.langsmith_result(
                "LANGSMITH_RUN_END_FAILED", available=False
            )
        return False


@dataclass
class LangSmithHarnessTracer:
    """Fail-open LLM telemetry; prompt and response bodies are never accepted."""

    client: Any
    mode: str
    health: HarnessTelemetryHealth
    project_name: str = _LANGSMITH_PROJECT_NAME
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
            _validate_links(links)
        except HarnessTracingError:
            self.health.dropped(
                reason_code="LANGSMITH_PRIVACY_POLICY_REJECTED",
                vendor="langsmith",
            )
            return _NoopSpanContext()
        if name not in _LANGSMITH_LLM_SPANS:
            return _NoopSpanContext()
        return _LangSmithSpanContext(
            client=self.client,
            name=name,
            attributes=validated,
            mode=self.mode,
            health=self.health,
            project_name=self.project_name,
        )

    def shutdown(self) -> None:
        close = getattr(self.client, "close", None)
        if not callable(close):
            return None
        completed = threading.Event()

        def close_client() -> None:
            try:
                close()
                self.health.langsmith_result(
                    "LANGSMITH_SHUTDOWN_COMPLETED", available=True
                )
            except Exception:
                self.health.langsmith_result(
                    "LANGSMITH_SHUTDOWN_FAILED", available=False
                )
            finally:
                completed.set()

        worker = threading.Thread(
            target=close_client,
            name="molly-langsmith-shutdown",
            daemon=True,
        )
        worker.start()
        if not completed.wait(max(0.0, self.shutdown_timeout_seconds)):
            self.health.langsmith_result(
                "LANGSMITH_SHUTDOWN_TIMEOUT", available=False
            )


def build_langsmith_harness_tracer(
    *,
    config: HarnessObservabilityConfig,
    health: HarnessTelemetryHealth,
    client_factory: Any | None = None,
) -> LangSmithHarnessTracer | None:
    if config.langsmith_mode == "disabled":
        return None
    try:
        if client_factory is None:
            from langsmith import Client

            client_factory = Client
        client = client_factory(
            auto_batch_tracing=False,
            hide_inputs=True,
            hide_metadata=_langsmith_safe_metadata,
            hide_outputs=_langsmith_safe_outputs,
            omit_traced_runtime_info=True,
            timeout_ms=1000,
        )
    except Exception:
        health.langsmith_result("LANGSMITH_INITIALIZATION_FAILED", available=False)
        return None
    health.langsmith_result("LANGSMITH_READY", available=True)
    return LangSmithHarnessTracer(
        client=client,
        mode=config.langsmith_mode,
        health=health,
        project_name=_configured_project_name(),
    )


__all__ = [
    "LangSmithHarnessTracer",
    "build_langsmith_harness_tracer",
]
