"""Private, optional, fail-open Scientific Agent observability configuration."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Mapping

from ai4s_agent.schemas import HarnessTelemetryHealthSnapshot


_OTEL_MODES = frozenset({"disabled", "otlp_http", "otlp_grpc"})
_LANGSMITH_MODES = frozenset({"disabled", "metadata_only", "structured_content"})


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HarnessObservabilityConfig:
    """Validated modes only; endpoints and credentials never enter this object."""

    otel_mode: str = "disabled"
    langsmith_mode: str = "disabled"
    structured_content_allowed: bool = False
    otel_config_valid: bool = True
    langsmith_config_valid: bool = True
    service_name: str = "molly-scientific-agent-harness"

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "HarnessObservabilityConfig":
        values = os.environ if environ is None else environ
        requested_otel = str(
            values.get("AI4S_HARNESS_OTEL_MODE") or ""
        ).strip().lower()
        if not requested_otel and _truthy(values.get("AI4S_HARNESS_OTEL_ENABLED")):
            requested_otel = "otlp_http"
        requested_otel = requested_otel or "disabled"
        otel_valid = requested_otel in _OTEL_MODES
        otel_mode = requested_otel if otel_valid else "disabled"

        requested_langsmith = str(
            values.get("AI4S_HARNESS_LANGSMITH_MODE") or ""
        ).strip().lower() or "disabled"
        langsmith_valid = requested_langsmith in _LANGSMITH_MODES
        langsmith_mode = (
            requested_langsmith if langsmith_valid else "disabled"
        )
        structured_allowed = _truthy(
            values.get("AI4S_HARNESS_LANGSMITH_STRUCTURED_CONTENT_ALLOWED")
        )
        if langsmith_mode == "structured_content" and not structured_allowed:
            # Misconfigured content capture never blocks business logic and never
            # uploads content. It deterministically degrades to metadata-only.
            langsmith_mode = "metadata_only"
        service_name = str(
            values.get("AI4S_HARNESS_SERVICE_NAME")
            or "molly-scientific-agent-harness"
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", service_name):
            service_name = "molly-scientific-agent-harness"
        return cls(
            otel_mode=otel_mode,
            langsmith_mode=langsmith_mode,
            structured_content_allowed=structured_allowed,
            otel_config_valid=otel_valid,
            langsmith_config_valid=langsmith_valid,
            service_name=service_name,
        )


class HarnessTelemetryHealth:
    """Thread-safe, process-local counters with fixed privacy-safe codes."""

    def __init__(self, *, config: HarnessObservabilityConfig) -> None:
        self._lock = threading.Lock()
        self._otel_enabled = config.otel_mode != "disabled"
        self._otel_available = False
        self._otel_code = (
            "OTEL_CONFIG_INVALID"
            if not config.otel_config_valid
            else "TELEMETRY_DISABLED"
            if not self._otel_enabled
            else "OTEL_INITIALIZING"
        )
        self._langsmith_enabled = config.langsmith_mode != "disabled"
        self._langsmith_available = False
        self._langsmith_code = (
            "LANGSMITH_CONFIG_INVALID"
            if not config.langsmith_config_valid
            else "TELEMETRY_DISABLED"
            if not self._langsmith_enabled
            else "LANGSMITH_INITIALIZING"
        )
        self._dropped = 0
        self._failures = 0

    def otel_result(self, code: str, *, available: bool) -> None:
        with self._lock:
            self._otel_available = available
            self._otel_code = code
            if code.endswith("FAILED") or code.endswith("TIMEOUT"):
                self._failures += 1

    def langsmith_result(self, code: str, *, available: bool) -> None:
        with self._lock:
            self._langsmith_available = available
            self._langsmith_code = code
            if code.endswith("FAILED") or code.endswith("TIMEOUT"):
                self._failures += 1

    def dropped(self, *, reason_code: str, vendor: str) -> None:
        with self._lock:
            self._dropped += 1
            if vendor == "otel":
                self._otel_code = reason_code
            elif vendor == "langsmith":
                self._langsmith_code = reason_code

    def snapshot(self) -> HarnessTelemetryHealthSnapshot:
        with self._lock:
            return HarnessTelemetryHealthSnapshot(
                otel_enabled=self._otel_enabled,
                otel_available=self._otel_available,
                otel_last_result_code=self._otel_code,
                langsmith_enabled=self._langsmith_enabled,
                langsmith_available=self._langsmith_available,
                langsmith_last_result_code=self._langsmith_code,
                dropped_event_count=self._dropped,
                export_failure_count=self._failures,
            )


__all__ = [
    "HarnessObservabilityConfig",
    "HarnessTelemetryHealth",
]
