"""Closed-scope validation results for immutable Core records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .errors import ValidationContractError
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_ids,
    validate_identifier,
    validate_reference,
)


class ValidationScope(str, Enum):
    ARTIFACT = "ARTIFACT"
    RELATION = "RELATION"
    BUNDLE = "BUNDLE"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


VALIDATION_SCOPES = frozenset(item.value for item in ValidationScope)
VALIDATION_STATUSES = frozenset(item.value for item in ValidationStatus)


def _enum_value(value: str | Enum, *, field: str, allowed: frozenset[str]) -> str:
    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise ValidationContractError(f"{field} must be a string")
    normalized = candidate.strip().upper()
    if normalized not in allowed:
        raise ValidationContractError(f"unknown {field}: {candidate!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """An immutable, deterministic result for one bounded validation scope."""

    validator_id: str
    validator_version: str
    scope: str | ValidationScope
    subject_ids: tuple[str, ...]
    status: str | ValidationStatus
    reason: str = ""
    evidence_artifact_ids: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    timestamp: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.validator_id, field="validator_id")
        validate_identifier(self.validator_version, field="validator_version")
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, field="validation scope", allowed=VALIDATION_SCOPES),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, field="validation status", allowed=VALIDATION_STATUSES),
        )
        subjects = tuple(validate_reference(value, field="subject_id") for value in self.subject_ids)
        if not subjects:
            raise ValidationContractError("subject_ids must contain at least one identity")
        if len(subjects) != len(set(subjects)):
            raise ValidationContractError("subject_ids must not contain duplicates")
        object.__setattr__(self, "subject_ids", subjects)
        object.__setattr__(
            self,
            "evidence_artifact_ids",
            validate_artifact_ids(
                self.evidence_artifact_ids, field="evidence_artifact_ids"
            ),
        )
        references = tuple(
            validate_reference(value, field="source_reference")
            for value in self.source_references
        )
        if len(references) != len(set(references)):
            raise ValidationContractError("source_references must not contain duplicates")
        object.__setattr__(self, "source_references", references)
        if not isinstance(self.reason, str):
            raise ValidationContractError("reason must be text")
        object.__setattr__(
            self,
            "timestamp",
            normalize_timestamp(self.timestamp, field="timestamp"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="validation metadata"),
        )

    @property
    def message(self) -> str:
        """Descriptive alias for the contract's reason field."""

        return self.reason

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "scope": self.scope,
            "subject_ids": list(self.subject_ids),
            "status": self.status,
            "reason": self.reason,
            "message": self.reason,
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "source_references": list(self.source_references),
            "timestamp": self.timestamp,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationResult":
        if not isinstance(value, Mapping):
            raise ValidationContractError("validation result must be a JSON object")
        try:
            reason = value.get("reason", value.get("message", ""))
            return cls(
                validator_id=str(value["validator_id"]),
                validator_version=str(value["validator_version"]),
                scope=value["scope"],
                subject_ids=tuple(value["subject_ids"]),
                status=value["status"],
                reason=str(reason),
                evidence_artifact_ids=tuple(value.get("evidence_artifact_ids", ())),
                source_references=tuple(value.get("source_references", ())),
                timestamp=str(value["timestamp"]),
                metadata=dict(value.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationContractError("validation result is malformed") from exc
