"""Bounded artifact dependency lineage, not causal or scheduling semantics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
import uuid
from typing import Any, Iterable, Mapping

from ._persistence import append_all, locked_append
from .artifacts import ArtifactRecord
from .errors import CoreContractError, LineageError
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_id,
    validate_artifact_ids,
    validate_identifier,
    validate_reference,
    validate_sha256,
)


class RelationType(str, Enum):
    """The only dependency/provenance relations in Core-01."""

    CONSUMED_BY = "CONSUMED_BY"
    PRODUCED_BY = "PRODUCED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTED_BY = "SUPPORTED_BY"


RELATION_TYPES = frozenset(item.value for item in RelationType)


def _relation_value(value: str | RelationType) -> str:
    if isinstance(value, RelationType):
        return value.value
    if not isinstance(value, str):
        raise LineageError("relation type must be a string")
    normalized = value.strip().upper()
    if normalized not in RELATION_TYPES:
        raise LineageError(f"unknown lineage relation: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class LineageRelation:
    """One immutable subject/object provenance relation."""

    relation_type: str | RelationType
    subject_id: str
    object_id: str
    relation_id: str = field(default_factory=lambda: f"rel_{uuid.uuid4().hex}")
    created_at: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    previous_relation_sha256: str | None = None
    relation_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_type", _relation_value(self.relation_type))
        validate_identifier(self.relation_id, field="relation_id")
        validate_reference(self.subject_id, field="subject_id")
        validate_reference(self.object_id, field="object_id")
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="lineage metadata"),
        )
        for value, field_name in (
            (self.previous_relation_sha256, "previous_relation_sha256"),
            (self.relation_sha256, "relation_sha256"),
        ):
            if value is not None:
                try:
                    validate_sha256(value, field=field_name)
                except CoreContractError as exc:
                    raise LineageError(str(exc)) from exc

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "relation_type": self.relation_type,
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation_id": self.relation_id,
            "created_at": self.created_at,
            "metadata": thaw_json(self.metadata),
            "previous_relation_sha256": self.previous_relation_sha256,
        }
        if include_digest:
            value["relation_sha256"] = self.relation_sha256
        return value

    @property
    def computed_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict(include_digest=False)))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LineageRelation":
        if not isinstance(value, Mapping):
            raise LineageError("lineage record must be a JSON object")
        try:
            relation = cls(
                relation_type=value["relation_type"],
                subject_id=str(value["subject_id"]),
                object_id=str(value["object_id"]),
                relation_id=str(value["relation_id"]),
                created_at=str(value["created_at"]),
                metadata=dict(value.get("metadata", {})),
                previous_relation_sha256=(
                    None
                    if value.get("previous_relation_sha256") is None
                    else str(value["previous_relation_sha256"])
                ),
                relation_sha256=(
                    None if value.get("relation_sha256") is None else str(value["relation_sha256"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError("lineage record is malformed") from exc
        if relation.relation_sha256 is None:
            raise LineageError("persisted lineage record has no digest")
        if relation.computed_sha256 != relation.relation_sha256:
            raise LineageError("lineage record digest mismatch")
        return relation


class ArtifactLineage:
    """A small in-memory or JSONL-backed relation ledger."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        known_ids: Iterable[str] = (),
        strict: bool = False,
    ) -> None:
        configured = None if path is None else Path(path)
        if configured is not None and configured.exists() and configured.is_dir():
            configured = configured / "lineage.jsonl"
        self.path = None if configured is None else configured.absolute()
        self._relations: list[LineageRelation] = []
        self._known_ids = set()
        self._strict = False
        for value in known_ids:
            self._known_ids.add(validate_reference(value, field="lineage identity"))
        self._strict = strict or bool(self._known_ids)

    def _hydrate_persisted_identities(self) -> None:
        """Remember identities already persisted before enabling strict mode."""

        if self.path is None or self._strict or not self.path.exists():
            return
        for relation in self._read_relations():
            self._known_ids.update((relation.subject_id, relation.object_id))

    def _read_relations(self) -> list[LineageRelation]:
        if self.path is None:
            return []
        if self.path.is_symlink():
            raise LineageError("lineage path cannot be a symlink")
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise LineageError("lineage path is not a regular file")
        relations: list[LineageRelation] = []
        previous_digest: str | None = None
        seen_ids: set[str] = set()
        try:
            with self.path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.endswith(b"\n"):
                        raise LineageError(f"lineage line {line_number} is truncated")
                    try:
                        value = json.loads(raw_line[:-1].decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise LineageError(f"lineage line {line_number} is invalid JSON") from exc
                    relation = LineageRelation.from_dict(value)
                    if relation.relation_id in seen_ids:
                        raise LineageError("lineage contains a duplicate relation ID")
                    if relation.previous_relation_sha256 != previous_digest:
                        raise LineageError("lineage relation chain is discontinuous")
                    self._check_known(relation.subject_id, relation.object_id)
                    relations.append(relation)
                    seen_ids.add(relation.relation_id)
                    previous_digest = relation.relation_sha256
        except OSError as exc:
            raise LineageError("lineage could not be read") from exc
        return relations

    @property
    def relations(self) -> tuple[LineageRelation, ...]:
        if self.path is None:
            return tuple(self._relations)
        return tuple(self._read_relations())

    def __iter__(self):
        return iter(self.relations)

    def register_identity(self, value: str) -> str:
        self._hydrate_persisted_identities()
        validated = validate_reference(value, field="lineage identity")
        self._known_ids.add(validated)
        self._strict = True
        return validated

    def register_artifact(self, artifact_id: str) -> str:
        self._hydrate_persisted_identities()
        validated = validate_artifact_id(artifact_id)
        self._known_ids.add(validated)
        self._strict = True
        return validated

    def register_step(self, step_id: str) -> str:
        self._hydrate_persisted_identities()
        validated = validate_identifier(step_id, field="step_id")
        self._known_ids.add(validated)
        self._strict = True
        return validated

    def add_artifact(self, record: ArtifactRecord) -> tuple[LineageRelation, ...]:
        """Register an artifact identity without inventing an occurrence.

        Artifact content metadata is not authoritative production provenance.
        Use :meth:`record_production` when a run/step occurrence is known.
        """

        if not isinstance(record, ArtifactRecord):
            raise LineageError("add_artifact requires an ArtifactRecord")
        self.register_artifact(record.artifact_id)
        return ()

    def record_production(
        self,
        *,
        artifact_id: str,
        producer_step_id: str,
        input_artifact_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> tuple[LineageRelation, ...]:
        """Record one production occurrence for an immutable artifact.

        The same ``artifact_id`` may be passed repeatedly for identical bytes
        produced by different steps or runs.  Each call appends its own
        ``PRODUCED_BY`` relation and its direct ``DERIVED_FROM`` relations;
        no first-writer artifact metadata is consulted or copied.
        """

        artifact_id = validate_artifact_id(artifact_id)
        producer_step_id = validate_identifier(producer_step_id, field="producer_step_id")
        input_ids = validate_artifact_ids(tuple(input_artifact_ids), field="input_artifact_ids")
        self.register_artifact(artifact_id)
        self.register_step(producer_step_id)
        for input_id in input_ids:
            self.register_artifact(input_id)

        occurrence_metadata = {} if metadata is None else metadata
        occurrence_timestamp = created_at or utc_timestamp()
        relations = [
            self.add_relation(
                RelationType.PRODUCED_BY,
                artifact_id,
                producer_step_id,
                created_at=occurrence_timestamp,
                metadata=occurrence_metadata,
            )
        ]
        relations.extend(
            self.add_relation(
                RelationType.DERIVED_FROM,
                artifact_id,
                input_id,
                created_at=occurrence_timestamp,
                metadata=occurrence_metadata,
            )
            for input_id in input_ids
        )
        return tuple(relations)

    def _check_known(self, subject_id: str, object_id: str) -> None:
        if self._strict and (
            subject_id not in self._known_ids or object_id not in self._known_ids
        ):
            raise LineageError("lineage relation references an unknown identity")

    def add_relation(
        self,
        relation_type: str | RelationType,
        subject_id: str,
        object_id: str,
        *,
        relation_id: str | None = None,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LineageRelation:
        relation_type = _relation_value(relation_type)
        validate_reference(subject_id, field="subject_id")
        validate_reference(object_id, field="object_id")
        self._check_known(subject_id, object_id)
        candidate = LineageRelation(
            relation_type=relation_type,
            subject_id=subject_id,
            object_id=object_id,
            relation_id=relation_id or f"rel_{uuid.uuid4().hex}",
            created_at=created_at or utc_timestamp(),
            metadata={} if metadata is None else metadata,
        )

        if self.path is None:
            existing = getattr(self, "_relations", [])
            if any(item.relation_id == candidate.relation_id for item in existing):
                raise LineageError("duplicate lineage relation ID")
            previous_digest = existing[-1].relation_sha256 if existing else None
            prepared = replace(candidate, previous_relation_sha256=previous_digest)
            persisted = replace(prepared, relation_sha256=prepared.computed_sha256)
            self._relations = [*existing, persisted]
            return persisted

        with locked_append(self.path) as descriptor:
            existing = self._read_relations()
            if any(item.relation_id == candidate.relation_id for item in existing):
                raise LineageError("duplicate lineage relation ID")
            previous_digest = existing[-1].relation_sha256 if existing else None
            prepared = replace(candidate, previous_relation_sha256=previous_digest)
            persisted = replace(prepared, relation_sha256=prepared.computed_sha256)
            append_all(descriptor, persisted.canonical_bytes() + b"\n")
            return persisted

    def add_relation_idempotent(
        self,
        relation_type: str | RelationType,
        subject_id: str,
        object_id: str,
        *,
        relation_id: str,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LineageRelation:
        """Append a deterministic relation, or verify an existing one.

        Execution projections derive relation IDs from a durable success event.
        Reconciliation may therefore encounter a relation already projected by
        a previous process.  Identical semantic facts are returned unchanged;
        a reused ID with different semantics fails closed.
        """

        relation_type = _relation_value(relation_type)
        validate_reference(subject_id, field="subject_id")
        validate_reference(object_id, field="object_id")
        validate_identifier(relation_id, field="relation_id")
        requested_timestamp = normalize_timestamp(
            created_at or utc_timestamp(), field="created_at"
        )
        requested_metadata = freeze_json_mapping(
            {} if metadata is None else metadata, field="lineage metadata"
        )
        existing_relations = self.relations
        for existing in existing_relations:
            if existing.relation_id != relation_id:
                continue
            if (
                existing.relation_type != relation_type
                or existing.subject_id != subject_id
                or existing.object_id != object_id
                or existing.created_at != requested_timestamp
                or existing.metadata != requested_metadata
            ):
                raise LineageError(
                    "deterministic lineage relation ID has conflicting semantics"
                )
            return existing
        return self.add_relation(
            relation_type,
            subject_id,
            object_id,
            relation_id=relation_id,
            created_at=requested_timestamp,
            metadata=requested_metadata,
        )

    def register(self, relation: LineageRelation) -> LineageRelation:
        """Append a prebuilt relation using the same integrity checks."""

        if not isinstance(relation, LineageRelation):
            raise LineageError("register requires a LineageRelation")
        return self.add_relation(
            relation.relation_type,
            relation.subject_id,
            relation.object_id,
            relation_id=relation.relation_id,
            created_at=relation.created_at,
            metadata=relation.metadata,
        )

    def for_subject(self, subject_id: str) -> tuple[LineageRelation, ...]:
        validate_reference(subject_id, field="subject_id")
        return tuple(item for item in self.relations if item.subject_id == subject_id)

    def for_object(self, object_id: str) -> tuple[LineageRelation, ...]:
        validate_reference(object_id, field="object_id")
        return tuple(item for item in self.relations if item.object_id == object_id)

    def parents(self, artifact_id: str) -> tuple[str, ...]:
        validate_artifact_id(artifact_id)
        return tuple(
            item.object_id
            for item in self.for_subject(artifact_id)
            if item.relation_type == RelationType.DERIVED_FROM.value
        )

    def producer_steps(self, artifact_id: str) -> tuple[str, ...]:
        validate_artifact_id(artifact_id)
        return tuple(
            item.object_id
            for item in self.for_subject(artifact_id)
            if item.relation_type == RelationType.PRODUCED_BY.value
        )

    def supported_by(self, artifact_id: str) -> tuple[str, ...]:
        validate_artifact_id(artifact_id)
        return tuple(
            item.object_id
            for item in self.for_subject(artifact_id)
            if item.relation_type == RelationType.SUPPORTED_BY.value
        )

    def inputs(self, artifact_id: str) -> tuple[str, ...]:
        """Alias for the direct ``DERIVED_FROM`` parent identities."""

        return self.parents(artifact_id)

    def inspect(self) -> tuple[LineageRelation, ...]:
        """Return a read-only relation snapshot."""

        return self.relations

    def validate_known_identities(self) -> None:
        for item in self.relations:
            self._check_known(item.subject_id, item.object_id)
