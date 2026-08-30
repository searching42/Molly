"""Append-only factual run records for Molly Core."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import uuid
from typing import Any, Mapping

from ._persistence import append_all, locked_append
from .errors import CoreContractError, LedgerCorruptionError, LedgerError
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_ids,
    validate_digest_reference,
    validate_identifier,
)


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One immutable observation in the append-only run ledger."""

    event_id: str
    run_id: str
    event_type: str
    step_id: str | None = None
    status: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    model_profile: Mapping[str, Any] = field(default_factory=dict)
    provider_profile: Mapping[str, Any] = field(default_factory=dict)
    prompt_digest: str | None = None
    config_digest: str | None = None
    seed_metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    previous_event_sha256: str | None = None
    event_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.event_id, field="event_id")
        validate_identifier(self.run_id, field="run_id")
        validate_identifier(self.event_type, field="event_type")
        for value, field_name in (
            (self.step_id, "step_id"),
            (self.tool_name, "tool_name"),
            (self.tool_version, "tool_version"),
        ):
            if value is not None:
                validate_identifier(value, field=field_name)
        if self.status is not None:
            validate_identifier(self.status, field="status")
        object.__setattr__(
            self,
            "input_artifact_ids",
            validate_artifact_ids(self.input_artifact_ids, field="input_artifact_ids"),
        )
        object.__setattr__(
            self,
            "output_artifact_ids",
            validate_artifact_ids(self.output_artifact_ids, field="output_artifact_ids"),
        )
        object.__setattr__(
            self,
            "model_profile",
            freeze_json_mapping(self.model_profile, field="model_profile"),
        )
        object.__setattr__(
            self,
            "provider_profile",
            freeze_json_mapping(self.provider_profile, field="provider_profile"),
        )
        object.__setattr__(
            self,
            "seed_metadata",
            freeze_json_mapping(self.seed_metadata, field="seed_metadata"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="metadata"),
        )
        for value, field_name in (
            (self.prompt_digest, "prompt_digest"),
            (self.config_digest, "config_digest"),
            (self.previous_event_sha256, "previous_event_sha256"),
            (self.event_sha256, "event_sha256"),
        ):
            if value is not None:
                validate_digest_reference(value, field=field_name)
        object.__setattr__(
            self,
            "timestamp",
            normalize_timestamp(self.timestamp, field="timestamp"),
        )

    @property
    def created_at(self) -> str:
        """Alias for the event timestamp."""

        return self.timestamp

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "event_type": self.event_type,
            "status": self.status,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "model_profile": thaw_json(self.model_profile),
            "provider_profile": thaw_json(self.provider_profile),
            "prompt_digest": self.prompt_digest,
            "config_digest": self.config_digest,
            "seed_metadata": thaw_json(self.seed_metadata),
            "timestamp": self.timestamp,
            "metadata": thaw_json(self.metadata),
            "previous_event_sha256": self.previous_event_sha256,
        }
        if include_digest:
            payload["event_sha256"] = self.event_sha256
        return payload

    @property
    def computed_sha256(self) -> str:
        """Digest of the canonical event body, excluding its stored digest."""

        return sha256_bytes(canonical_json_bytes(self.to_dict(include_digest=False)))

    @property
    def digest(self) -> str:
        """The verified or prospective event digest."""

        return self.event_sha256 or self.computed_sha256

    def canonical_bytes(self) -> bytes:
        """Return the canonical persisted JSON object bytes."""

        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerEvent":
        if not isinstance(value, Mapping):
            raise LedgerCorruptionError("ledger event must be a JSON object")
        try:
            event = cls(
                event_id=str(value["event_id"]),
                run_id=str(value["run_id"]),
                event_type=str(value["event_type"]),
                step_id=None if value.get("step_id") is None else str(value["step_id"]),
                status=None if value.get("status") is None else str(value["status"]),
                tool_name=(None if value.get("tool_name") is None else str(value["tool_name"])),
                tool_version=(
                    None if value.get("tool_version") is None else str(value["tool_version"])
                ),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                output_artifact_ids=tuple(value.get("output_artifact_ids", ())),
                model_profile=dict(value.get("model_profile", {})),
                provider_profile=dict(value.get("provider_profile", {})),
                prompt_digest=(
                    None
                    if value.get("prompt_digest") is None
                    else str(value["prompt_digest"])
                ),
                config_digest=(
                    None if value.get("config_digest") is None else str(value["config_digest"])
                ),
                seed_metadata=dict(value.get("seed_metadata", {})),
                timestamp=str(value["timestamp"]),
                metadata=dict(value.get("metadata", {})),
                previous_event_sha256=(
                    None
                    if value.get("previous_event_sha256") is None
                    else str(value["previous_event_sha256"])
                ),
                event_sha256=(
                    None if value.get("event_sha256") is None else str(value["event_sha256"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError("ledger event is malformed") from exc
        if event.event_sha256 is None:
            raise LedgerCorruptionError("persisted ledger event has no digest")
        if event.computed_sha256 != event.event_sha256:
            raise LedgerCorruptionError("ledger event digest mismatch")
        return event


class RunLedger:
    """A local JSONL ledger whose prior events are never rewritten."""

    def __init__(self, path: Path | str) -> None:
        configured = Path(path)
        if configured.exists() and configured.is_dir():
            configured = configured / "events.jsonl"
        self.path = configured.absolute()

    def _read_events(self) -> list[LedgerEvent]:
        if self.path.is_symlink():
            raise LedgerCorruptionError("ledger path cannot be a symlink")
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise LedgerCorruptionError("ledger path is not a regular file")
        events: list[LedgerEvent] = []
        previous_digest: str | None = None
        seen_ids: set[str] = set()
        try:
            with self.path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.endswith(b"\n"):
                        raise LedgerCorruptionError(
                            f"ledger line {line_number} is truncated"
                        )
                    payload = raw_line[:-1]
                    if not payload:
                        raise LedgerCorruptionError(f"ledger line {line_number} is empty")
                    try:
                        value = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise LedgerCorruptionError(
                            f"ledger line {line_number} is not valid JSON"
                        ) from exc
                    event = LedgerEvent.from_dict(value)
                    if event.event_id in seen_ids:
                        raise LedgerCorruptionError("ledger contains a duplicate event ID")
                    if event.previous_event_sha256 != previous_digest:
                        raise LedgerCorruptionError("ledger event chain is discontinuous")
                    events.append(event)
                    seen_ids.add(event.event_id)
                    previous_digest = event.event_sha256
        except OSError as exc:
            raise LedgerCorruptionError("ledger could not be read") from exc
        return events

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """Return a read-only snapshot without changing authoritative state."""

        return tuple(self._read_events())

    def read_all(self) -> tuple[LedgerEvent, ...]:
        """Explicit inspection alias."""

        return self.events

    def inspect(self) -> tuple[LedgerEvent, ...]:
        """Return a read-only event snapshot."""

        return self.events

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def last_event(self) -> LedgerEvent | None:
        events = self._read_events()
        return events[-1] if events else None

    def append(
        self,
        event: LedgerEvent | Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> LedgerEvent:
        """Validate and append one event, serializing concurrent appends."""

        if event is not None and fields:
            raise CoreContractError("append accepts an event or event fields, not both")
        if event is None:
            if "event_id" not in fields:
                fields["event_id"] = f"evt_{uuid.uuid4().hex}"
            event = LedgerEvent(**fields)
        elif isinstance(event, Mapping):
            event = LedgerEvent.from_dict(event)
        elif not isinstance(event, LedgerEvent):
            raise CoreContractError("append requires a LedgerEvent")

        with locked_append(self.path) as descriptor:
            existing = self._read_events()
            if any(item.event_id == event.event_id for item in existing):
                raise LedgerError(f"duplicate ledger event ID: {event.event_id}")
            previous_digest = existing[-1].event_sha256 if existing else None
            prepared = replace(
                event,
                previous_event_sha256=previous_digest,
                event_sha256=None,
            )
            persisted = replace(prepared, event_sha256=prepared.computed_sha256)
            append_all(descriptor, persisted.canonical_bytes() + b"\n")
            return persisted

    def append_event(self, event: LedgerEvent | Mapping[str, Any] | None = None, **fields: Any):
        """Descriptive alias for :meth:`append`."""

        return self.append(event, **fields)
