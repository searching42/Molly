"""Observer orchestration with explicit authoritative-state isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molly.core.inspection import RunInspector

from .errors import ExporterUnavailableError, ObserverIntegrityError
from .model import RunTrace
from .projection import RunTraceProjector


@dataclass(frozen=True, slots=True)
class ObservationOutcome:
    status: str
    exporter: str
    trace: RunTrace
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": self.status,
            "exporter": self.exporter,
            "trace": self.trace.to_dict(),
        }
        if self.error_type is not None:
            value["error_type"] = self.error_type
        return value


class ObservationService:
    """Project then export; exporters never receive Core store handles."""

    def __init__(self, inspector: RunInspector) -> None:
        if not isinstance(inspector, RunInspector):
            raise TypeError("ObservationService requires a RunInspector")
        self.inspector = inspector

    @staticmethod
    def _snapshot(inspection: Any) -> tuple[Any, ...]:
        return (
            inspection.ledger_sha256,
            inspection.lineage_sha256,
            tuple(inspection.referenced_artifact_ids),
            inspection.status,
        )

    def _assert_unchanged(self, run_id: str, snapshot: tuple[Any, ...], *, cause: BaseException | None = None) -> None:
        try:
            after = self.inspector.inspect_run(run_id)
        except Exception as exc:
            raise ObserverIntegrityError("observer changed or corrupted authoritative Core facts") from (cause or exc)
        if self._snapshot(after) != snapshot:
            raise ObserverIntegrityError("observer changed authoritative Core facts") from cause

    def export_run(self, run_id: str, exporter: Any) -> ObservationOutcome:
        if not hasattr(exporter, "export") or not callable(exporter.export):
            raise ExporterUnavailableError("EXPORTER_UNAVAILABLE: exporter has no export method")
        name = getattr(exporter, "name", type(exporter).__name__.casefold())
        before = self.inspector.inspect_run(run_id)
        trace = RunTraceProjector(self.inspector).project_run(run_id)
        snapshot = self._snapshot(before)
        try:
            exporter.export(trace)
        except ExporterUnavailableError:
            # An unavailable optional dependency/configuration is a bounded
            # operator error, not an export attempt and not a run failure.
            self._assert_unchanged(run_id, snapshot)
            raise
        except Exception as exc:
            self._assert_unchanged(run_id, snapshot, cause=exc)
            error_type = type(exc).__name__
            if not error_type.isidentifier():
                error_type = "ExporterError"
            return ObservationOutcome(
                status="EXPORT_FAILED",
                exporter=str(name),
                trace=trace,
                error_type=error_type,
            )
        self._assert_unchanged(run_id, snapshot)
        return ObservationOutcome(status="EXPORTED", exporter=str(name), trace=trace)


__all__ = ["ObservationOutcome", "ObservationService"]
