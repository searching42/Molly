"""Observer-only JSON, OpenTelemetry, and LangSmith exporters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from molly.core.ids import validate_identifier

from .errors import ExporterFailedError, ExporterUnavailableError
from .model import RunTrace


class TraceExporter(Protocol):
    name: str

    def export(self, trace: RunTrace) -> Mapping[str, Any]:
        ...


def _epoch_nanoseconds(timestamp: str) -> int:
    raw = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    value = datetime.fromisoformat(raw).astimezone(timezone.utc)
    return int(value.timestamp() * 1_000_000_000)


class JsonTraceExporter:
    """Dependency-free baseline exporter used as the canonical test format."""

    name = "json"

    def export(self, trace: RunTrace) -> Mapping[str, Any]:
        if not isinstance(trace, RunTrace):
            raise ExporterFailedError("JSON exporter requires a RunTrace")
        return trace.to_dict()


class OpenTelemetryExporter:
    """Optional OTel exporter with lazy imports and safe trace attributes."""

    name = "otel"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        timeout_seconds: float = 10.0,
        client: Any = None,
    ) -> None:
        if endpoint is not None:
            if not isinstance(endpoint, str) or len(endpoint) > 2_048:
                raise ExporterUnavailableError("OTel endpoint is invalid")
            parsed = urlsplit(endpoint)
            if parsed.scheme.casefold() not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
                raise ExporterUnavailableError("OTel endpoint must be a configured host URL")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 60:
            raise ExporterUnavailableError("OTel export timeout is outside the bounded range")
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        self.client = client

    @staticmethod
    def _injected_export(client: Any, payload: Mapping[str, Any]) -> Any:
        if hasattr(client, "export_trace") and callable(client.export_trace):
            return client.export_trace(payload)
        if callable(client):
            return client(payload)
        raise ExporterUnavailableError("injected OTel collector has no export interface")

    def export(self, trace: RunTrace) -> Mapping[str, Any]:
        if not isinstance(trace, RunTrace):
            raise ExporterFailedError("OTel exporter requires a RunTrace")
        payload = trace.to_dict()
        if self.client is not None:
            self._injected_export(self.client, payload)
            return {"status": "EXPORTED", "exporter": self.name, "trace_id": trace.trace_id}
        if self.endpoint is None:
            raise ExporterUnavailableError("EXPORTER_UNAVAILABLE: OTel endpoint is not configured")
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError as exc:
            raise ExporterUnavailableError("EXPORTER_UNAVAILABLE: OTel dependencies are unavailable") from exc

        provider = TracerProvider()
        otel_exporter = OTLPSpanExporter(endpoint=self.endpoint, timeout=self.timeout_seconds)
        provider.add_span_processor(SimpleSpanProcessor(otel_exporter))
        tracer = provider.get_tracer("molly.observability")
        try:
            root = trace.spans[0]
            with tracer.start_as_current_span(
                root.name,
                start_time=_epoch_nanoseconds(root.start_time),
            ) as root_span:
                for key, value in root.attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        root_span.set_attribute(key, value)
                for event in root.events:
                    root_span.add_event(event.name, attributes=dict(event.attributes), timestamp=_epoch_nanoseconds(event.timestamp))
                for span in trace.spans[1:]:
                    with tracer.start_as_current_span(
                        span.name,
                        start_time=_epoch_nanoseconds(span.start_time),
                    ) as child_span:
                        for key, value in span.attributes.items():
                            if isinstance(value, (str, int, float, bool)):
                                child_span.set_attribute(key, value)
                        for event in span.events:
                            child_span.add_event(event.name, attributes=dict(event.attributes), timestamp=_epoch_nanoseconds(event.timestamp))
            flushed = provider.force_flush(int(self.timeout_seconds * 1_000))
            if flushed is False:
                raise ExporterFailedError("OTel exporter did not flush within its bounded timeout")
        except ExporterFailedError:
            raise
        except Exception as exc:
            raise ExporterFailedError("OTel export failed") from exc
        finally:
            provider.shutdown()
        return {"status": "EXPORTED", "exporter": self.name, "trace_id": trace.trace_id}


class LangSmithExporter:
    """Optional LangSmith observer with server-owned client configuration."""

    name = "langsmith"

    def __init__(
        self,
        *,
        profile_ref: str = "langsmith-default",
        api_url: str | None = None,
        client: Any = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        validate_identifier(profile_ref, field="LangSmith profile_ref")
        if api_url is not None:
            parsed = urlsplit(api_url)
            if parsed.scheme.casefold() not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
                raise ExporterUnavailableError("LangSmith API URL is invalid")
        if client_factory is not None and not callable(client_factory):
            raise ExporterUnavailableError("LangSmith client_factory must be callable")
        self.profile_ref = profile_ref
        self.api_url = api_url
        self.client = client
        self.client_factory = client_factory

    @staticmethod
    def _injected_export(client: Any, payload: Mapping[str, Any], trace: RunTrace) -> Any:
        if hasattr(client, "export_trace") and callable(client.export_trace):
            return client.export_trace(payload)
        if hasattr(client, "create_run") and callable(client.create_run):
            return client.create_run(
                name="molly.run",
                run_type="chain",
                inputs={"trace_id": trace.trace_id, "run_id": trace.run_id},
                outputs={"status": trace.status, "span_count": len(trace.spans)},
                extra={"molly_trace": payload},
            )
        if callable(client):
            return client(payload)
        raise ExporterUnavailableError("injected LangSmith client has no export interface")

    def export(self, trace: RunTrace) -> Mapping[str, Any]:
        if not isinstance(trace, RunTrace):
            raise ExporterFailedError("LangSmith exporter requires a RunTrace")
        payload = trace.to_dict()
        client = self.client
        if client is None and self.client_factory is not None:
            client = self.client_factory()
        if client is None:
            try:
                from langsmith import Client
            except ImportError as exc:
                raise ExporterUnavailableError("EXPORTER_UNAVAILABLE: LangSmith dependencies are unavailable") from exc
            try:
                client = Client(api_url=self.api_url) if self.api_url else Client()
            except Exception as exc:
                raise ExporterFailedError("LangSmith client construction failed") from exc
        self._injected_export(client, payload, trace)
        return {"status": "EXPORTED", "exporter": self.name, "trace_id": trace.trace_id, "profile_ref": self.profile_ref}


__all__ = ["JsonTraceExporter", "LangSmithExporter", "OpenTelemetryExporter", "TraceExporter"]
