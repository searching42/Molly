"""Bounded OLED records built from evidence and structured mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)
from molly.documents.locators import SourceLocator

from .identity import MoleculeIdentity
from .units import NormalizedProperty


class ClaimLevel(str, Enum):
    SOURCE_REPORTED = "SOURCE_REPORTED"
    DERIVED_NORMALIZATION = "DERIVED_NORMALIZATION"
    SYNTHETIC_CONTRACT_ONLY = "SYNTHETIC_CONTRACT_ONLY"


class OledValidationStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


def _enum(value: str | Enum, enum_type: type[Enum], field: str) -> str:
    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise CoreContractError(f"{field} must be text")
    try:
        return enum_type(candidate.strip().upper()).value
    except ValueError as exc:
        raise CoreContractError(f"unknown {field}: {candidate!r}") from exc


def _optional_text(value: Any, field: str, maximum: int = 512) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise CoreContractError(f"{field} must be bounded text")
    return value.strip() or None


@dataclass(frozen=True, slots=True)
class MeasurementCondition:
    """Explicit condition key; different conditions never form exact duplicates."""

    condition_status: str = "EXPLICIT"
    medium: str | None = None
    host: str | None = None
    dopant_concentration: str | None = None
    temperature: str | None = None
    measurement_environment: str | None = None

    def __post_init__(self) -> None:
        status = _optional_text(self.condition_status, "condition_status", 64)
        if status is None:
            raise CoreContractError("condition_status is required")
        object.__setattr__(self, "condition_status", status.upper())
        for name in ("medium", "host", "dopant_concentration", "temperature", "measurement_environment"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))

    @property
    def condition_key(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def status(self) -> str:
        return self.condition_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_status": self.condition_status,
            "medium": self.medium,
            "host": self.host,
            "dopant_concentration": self.dopant_concentration,
            "temperature": self.temperature,
            "measurement_environment": self.measurement_environment,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MeasurementCondition":
        if value is None:
            return cls(condition_status="UNSPECIFIED")
        if not isinstance(value, Mapping):
            raise CoreContractError("measurement_condition must be an object")
        allowed = {"condition_status", "status", "medium", "host", "dopant_concentration", "temperature", "measurement_environment"}
        if set(value) - allowed:
            raise CoreContractError("measurement_condition has unknown fields")
        return cls(
            condition_status=value.get("condition_status", value.get("status", "EXPLICIT")),
            medium=value.get("medium"),
            host=value.get("host"),
            dopant_concentration=value.get("dopant_concentration"),
            temperature=value.get("temperature"),
            measurement_environment=value.get("measurement_environment"),
        )


@dataclass(frozen=True, slots=True)
class OledEvidenceRef:
    field_name: str
    candidate_id: str
    source_artifact_id: str
    source_locator: SourceLocator

    def __post_init__(self) -> None:
        validate_identifier(self.field_name, field="evidence field_name")
        validate_identifier(self.candidate_id, field="evidence candidate_id")
        validate_artifact_id(self.source_artifact_id)
        locator = self.source_locator if isinstance(self.source_locator, SourceLocator) else SourceLocator.from_dict(self.source_locator)
        if locator.source_artifact_id != self.source_artifact_id:
            raise CoreContractError("evidence locator source does not match source_artifact_id")
        object.__setattr__(self, "source_locator", locator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "candidate_id": self.candidate_id,
            "source_artifact_id": self.source_artifact_id,
            "source_locator": self.source_locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OledEvidenceRef":
        if not isinstance(value, Mapping):
            raise CoreContractError("evidence reference must be an object")
        allowed = {"field_name", "field", "candidate_id", "source_artifact_id", "source_locator", "locator"}
        if set(value) - allowed:
            raise CoreContractError("evidence reference has unknown fields")
        return cls(
            field_name=str(value.get("field_name", value.get("field"))),
            candidate_id=str(value["candidate_id"]),
            source_artifact_id=str(value["source_artifact_id"]),
            source_locator=SourceLocator.from_dict(value.get("source_locator", value.get("locator"))),
        )


@dataclass(frozen=True, slots=True)
class OledRecord:
    record_id: str
    canonical_document_artifact_id: str
    source_artifact_id: str
    molecule_identity: MoleculeIdentity
    property: NormalizedProperty
    measurement_condition: MeasurementCondition
    evidence: tuple[OledEvidenceRef, ...]
    claim_level: str | ClaimLevel
    validation_status: str | OledValidationStatus = OledValidationStatus.UNVALIDATED
    candidate_bundle_artifact_id: str | None = None
    mapping_artifact_id: str | None = None
    mapping_request_digest: str | None = None
    duplicate_group: str | None = None
    relation: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.record_id, field="record_id")
        validate_artifact_id(self.canonical_document_artifact_id)
        validate_artifact_id(self.source_artifact_id)
        identity = self.molecule_identity if isinstance(self.molecule_identity, MoleculeIdentity) else MoleculeIdentity.from_mapping(self.molecule_identity)
        prop = self.property if isinstance(self.property, NormalizedProperty) else NormalizedProperty.from_mapping(self.property)
        condition = self.measurement_condition if isinstance(self.measurement_condition, MeasurementCondition) else MeasurementCondition.from_mapping(self.measurement_condition)
        object.__setattr__(self, "molecule_identity", identity)
        object.__setattr__(self, "property", prop)
        object.__setattr__(self, "measurement_condition", condition)
        object.__setattr__(self, "claim_level", _enum(self.claim_level, ClaimLevel, "claim_level"))
        object.__setattr__(self, "validation_status", _enum(self.validation_status, OledValidationStatus, "validation_status"))
        evidence = tuple(item if isinstance(item, OledEvidenceRef) else OledEvidenceRef.from_dict(item) for item in self.evidence)
        if not evidence:
            raise CoreContractError("OLED record requires explicit evidence")
        for item in evidence:
            if item.source_artifact_id != self.source_artifact_id:
                raise CoreContractError("OLED evidence source does not match record source")
        object.__setattr__(self, "evidence", evidence)
        for name in ("candidate_bundle_artifact_id", "mapping_artifact_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_artifact_id(value))
        if self.mapping_request_digest is not None:
            object.__setattr__(self, "mapping_request_digest", validate_digest_reference(self.mapping_request_digest, field="mapping_request_digest"))
        for name in ("duplicate_group", "relation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _optional_text(value, name, 256))

    @property
    def comparison_key(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "identity": self.molecule_identity.identity_key,
            "property_id": self.property.property_id,
            "unit": self.property.unit,
            "condition": self.measurement_condition.condition_key,
        }))

    @property
    def property_id(self) -> str:
        return self.property.property_id

    @property
    def value(self) -> float | int | None:
        return self.property.value

    @property
    def unit(self) -> str | None:
        return self.property.unit

    @property
    def condition(self) -> MeasurementCondition:
        return self.measurement_condition

    def with_validation_status(self, status: str | OledValidationStatus) -> "OledRecord":
        from dataclasses import replace
        return replace(self, validation_status=status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "canonical_document_artifact_id": self.canonical_document_artifact_id,
            "source_artifact_id": self.source_artifact_id,
            "molecule_identity": self.molecule_identity.to_dict(),
            "property": self.property.to_dict(),
            "measurement_condition": self.measurement_condition.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "claim_level": self.claim_level,
            "validation_status": self.validation_status,
            "candidate_bundle_artifact_id": self.candidate_bundle_artifact_id,
            "mapping_artifact_id": self.mapping_artifact_id,
            "mapping_request_digest": self.mapping_request_digest,
            "duplicate_group": self.duplicate_group,
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OledRecord":
        if not isinstance(value, Mapping):
            raise CoreContractError("OLED record must be an object")
        allowed = {"record_id", "canonical_document_artifact_id", "source_artifact_id", "molecule_identity", "property", "measurement_condition", "evidence", "claim_level", "validation_status", "candidate_bundle_artifact_id", "mapping_artifact_id", "mapping_request_digest", "duplicate_group", "relation"}
        if set(value) - allowed:
            raise CoreContractError("OLED record has unknown fields")
        try:
            return cls(
                record_id=str(value["record_id"]),
                canonical_document_artifact_id=str(value["canonical_document_artifact_id"]),
                source_artifact_id=str(value["source_artifact_id"]),
                molecule_identity=MoleculeIdentity.from_mapping(value["molecule_identity"]),
                property=NormalizedProperty.from_mapping(value["property"]),
                measurement_condition=MeasurementCondition.from_mapping(value["measurement_condition"]),
                evidence=tuple(OledEvidenceRef.from_dict(item) for item in value["evidence"]),
                claim_level=value["claim_level"],
                validation_status=value.get("validation_status", OledValidationStatus.UNVALIDATED.value),
                candidate_bundle_artifact_id=value.get("candidate_bundle_artifact_id"),
                mapping_artifact_id=value.get("mapping_artifact_id"),
                mapping_request_digest=value.get("mapping_request_digest"),
                duplicate_group=value.get("duplicate_group"),
                relation=value.get("relation"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoreContractError("OLED record is malformed") from exc


__all__ = [
    "ClaimLevel",
    "MeasurementCondition",
    "OledEvidenceRef",
    "OledRecord",
    "OledValidationStatus",
]
