"""Deterministic, source-neutral trace records for observer-only exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    validate_digest_reference,
    validate_identifier,
)


_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    timestamp: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.name, field="trace event name")
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp, field="trace timestamp"))
        object.__setattr__(self, "attributes", freeze_json_mapping(self.attributes, field="trace event attributes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": thaw_json(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class TraceSpan:
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: str
    end_time: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[TraceEvent, ...] = ()

    def __post_init__(self) -> None:
        if not _SPAN_ID_RE.fullmatch(self.span_id):
            raise ValueError("span_id must be a 16-character lowercase hexadecimal value")
        if self.parent_span_id is not None and not _SPAN_ID_RE.fullmatch(self.parent_span_id):
            raise ValueError("parent_span_id must be a 16-character lowercase hexadecimal value")
        validate_identifier(self.name, field="trace span name")
        object.__setattr__(self, "start_time", normalize_timestamp(self.start_time, field="span start_time"))
        if self.end_time is not None:
            object.__setattr__(self, "end_time", normalize_timestamp(self.end_time, field="span end_time"))
        object.__setattr__(self, "attributes", freeze_json_mapping(self.attributes, field="trace span attributes"))
        object.__setattr__(
            self,
            "events",
            tuple(item if isinstance(item, TraceEvent) else TraceEvent(**item) for item in self.events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attributes": thaw_json(self.attributes),
            "events": [item.to_dict() for item in self.events],
        }


@dataclass(frozen=True, slots=True)
class RunTrace:
    """Canonical observer data derived from a validated RunInspection."""

    trace_id: str
    run_id: str
    status: str
    source_ledger_sha256: str
    source_lineage_sha256: str
    spans: tuple[TraceSpan, ...]
    schema_name: str = "molly.observability.run-trace"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not _TRACE_ID_RE.fullmatch(self.trace_id):
            raise ValueError("trace_id must be a 32-character lowercase hexadecimal value")
        validate_identifier(self.run_id, field="trace run_id")
        validate_identifier(self.status, field="trace status")
        validate_digest_reference(self.source_ledger_sha256, field="source_ledger_sha256")
        validate_digest_reference(self.source_lineage_sha256, field="source_lineage_sha256")
        validate_identifier(self.schema_name, field="trace schema_name")
        validate_identifier(self.schema_version, field="trace schema_version")
        spans = tuple(item if isinstance(item, TraceSpan) else TraceSpan(**item) for item in self.spans)
        if not spans or len({item.span_id for item in spans}) != len(spans):
            raise ValueError("trace spans must be non-empty and uniquely identified")
        if spans[0].parent_span_id is not None:
            raise ValueError("trace root span cannot have a parent")
        object.__setattr__(self, "spans", spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "status": self.status,
            "source_ledger_sha256": self.source_ledger_sha256,
            "source_lineage_sha256": self.source_lineage_sha256,
            "spans": [item.to_dict() for item in self.spans],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return sha256_bytes(self.canonical_bytes())


__all__ = ["RunTrace", "TraceEvent", "TraceSpan"]
