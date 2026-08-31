"""Digest-bound structured OLED mapping contracts.

The provider boundary accepts data only.  The host constructs the immutable
request, validates the returned schema and evidence bindings, and publishes
the resulting mapping as an ArtifactDraft.  No provider, prompt, or model
output owns execution or review authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from molly.core.tools import ArtifactDraft
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
)
from molly.domains.oled import (
    ClaimLevel,
    MeasurementCondition,
    MoleculeIdentity,
    NormalizedProperty,
    OledEvidenceRef,
)
from molly.documents.locators import SourceLocator

from .candidates import EvidenceCandidate, EvidenceCandidateBundle
from .errors import EvidenceContractError, EvidenceIntegrityError
from .packets import EvidencePacket, EvidencePacketBuilder


MAPPING_SCHEMA_NAME = "molly.evidence.oled-mapping"
MAPPING_SCHEMA_VERSION = "1"
MAPPING_SCHEMA_DIGEST = sha256_bytes(canonical_json_bytes({
    "schema_name": MAPPING_SCHEMA_NAME,
    "schema_version": MAPPING_SCHEMA_VERSION,
    "fields": ["molecule_identity", "property_id", "property_value", "unit", "measurement_condition"],
    "claim_levels": [item.value for item in ClaimLevel],
    "evidence_required_for_non_null_fields": True,
}))
PROMPT_TEMPLATE_DIGEST = sha256_bytes(canonical_json_bytes({
    "template": "map bounded OLED fields from supplied evidence packet only",
    "version": "1",
}))
MAX_MAPPING_RECORDS = 10_000
MAX_MAPPING_WARNINGS = 256


def _text(value: Any, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise EvidenceContractError(f"{field} must be bounded non-empty text")
    return value.strip()


def _optional_text(value: Any, field: str, maximum: int = 512) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise EvidenceContractError(f"{field} must be bounded text")
    return value.strip() or None


@dataclass(frozen=True, slots=True)
class MappingConfig:
    """Small server-owned mapping configuration included in request identity."""

    max_records: int = MAX_MAPPING_RECORDS
    max_packet_text_chars: int = 12_000
    temperature: float = 0.0
    schema_version: str = MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.max_records, bool) or not isinstance(self.max_records, int) or not 1 <= self.max_records <= MAX_MAPPING_RECORDS:
            raise EvidenceContractError("max_records is outside the bounded range")
        if isinstance(self.max_packet_text_chars, bool) or not isinstance(self.max_packet_text_chars, int) or not 64 <= self.max_packet_text_chars <= 64_000:
            raise EvidenceContractError("max_packet_text_chars is outside the bounded range")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(float(self.temperature)) or not 0 <= float(self.temperature) <= 2:
            raise EvidenceContractError("mapping temperature is outside the bounded range")
        validate_identifier(self.schema_version, field="mapping config schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_records": self.max_records,
            "max_packet_text_chars": self.max_packet_text_chars,
            "temperature": float(self.temperature),
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class FrozenOledMappingRequest:
    """The exact immutable input identity sent to a structured provider."""

    request_schema_version: str
    candidate_bundle_artifact_id: str
    packet_ids: tuple[str, ...]
    packet_digests: tuple[str, ...]
    oled_mapping_schema_digest: str
    prompt_template_digest: str
    provider_profile_ref: str
    model_identifier: str
    model_version: str
    mapping_config_digest: str
    request_digest: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.request_schema_version, field="request_schema_version")
        validate_artifact_id(self.candidate_bundle_artifact_id)
        ids = tuple(validate_identifier(value, field="packet_id") for value in self.packet_ids)
        digests = tuple(validate_digest_reference(value, field="packet_digest") for value in self.packet_digests)
        if not ids or len(ids) != len(digests) or len(ids) != len(set(ids)):
            raise EvidenceContractError("mapping packet IDs/digests must be unique and aligned")
        object.__setattr__(self, "packet_ids", ids)
        object.__setattr__(self, "packet_digests", digests)
        for name in ("oled_mapping_schema_digest", "prompt_template_digest", "mapping_config_digest"):
            object.__setattr__(self, name, validate_digest_reference(getattr(self, name), field=name))
        for name in ("provider_profile_ref", "model_identifier", "model_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        if self.request_digest is not None:
            object.__setattr__(self, "request_digest", validate_digest_reference(self.request_digest, field="request_digest"))
            if self.request_digest != self.computed_digest:
                raise EvidenceIntegrityError("mapping request digest does not match canonical request")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_schema_version": self.request_schema_version,
            "candidate_bundle_artifact_id": self.candidate_bundle_artifact_id,
            "packet_ids": list(self.packet_ids),
            "packet_digests": list(self.packet_digests),
            "oled_mapping_schema_digest": self.oled_mapping_schema_digest,
            "prompt_template_digest": self.prompt_template_digest,
            "provider_profile_ref": self.provider_profile_ref,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
            "mapping_config_digest": self.mapping_config_digest,
        }
        if include_digest:
            payload["request_digest"] = self.request_digest or self.computed_digest
        return payload

    @property
    def computed_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict(include_digest=False)))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def oled_schema_digest(self) -> str:
        return self.oled_mapping_schema_digest

    @classmethod
    def create(
        cls,
        *,
        candidate_bundle_artifact_id: str,
        packets: Sequence[EvidencePacket],
        provider_profile_ref: str,
        model_identifier: str,
        model_version: str,
        mapping_config_digest: str,
        request_schema_version: str = MAPPING_SCHEMA_VERSION,
        oled_mapping_schema_digest: str = MAPPING_SCHEMA_DIGEST,
        prompt_template_digest: str = PROMPT_TEMPLATE_DIGEST,
    ) -> "FrozenOledMappingRequest":
        packet_values = tuple(packets)
        initial = cls(
            request_schema_version=request_schema_version,
            candidate_bundle_artifact_id=candidate_bundle_artifact_id,
            packet_ids=tuple(item.packet_id for item in packet_values),
            packet_digests=tuple(item.digest for item in packet_values),
            oled_mapping_schema_digest=oled_mapping_schema_digest,
            prompt_template_digest=prompt_template_digest,
            provider_profile_ref=provider_profile_ref,
            model_identifier=model_identifier,
            model_version=model_version,
            mapping_config_digest=mapping_config_digest,
        )
        return cls(
            request_schema_version=initial.request_schema_version,
            candidate_bundle_artifact_id=initial.candidate_bundle_artifact_id,
            packet_ids=initial.packet_ids,
            packet_digests=initial.packet_digests,
            oled_mapping_schema_digest=initial.oled_mapping_schema_digest,
            prompt_template_digest=initial.prompt_template_digest,
            provider_profile_ref=initial.provider_profile_ref,
            model_identifier=initial.model_identifier,
            model_version=initial.model_version,
            mapping_config_digest=initial.mapping_config_digest,
            request_digest=initial.computed_digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenOledMappingRequest":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("mapping request must be an object")
        allowed = {"request_schema_version", "candidate_bundle_artifact_id", "packet_ids", "packet_digests", "oled_mapping_schema_digest", "prompt_template_digest", "provider_profile_ref", "model_identifier", "model_version", "mapping_config_digest", "request_digest"}
        if set(value) - allowed:
            raise EvidenceContractError("mapping request has unknown fields")
        try:
            return cls(
                request_schema_version=str(value["request_schema_version"]),
                candidate_bundle_artifact_id=str(value["candidate_bundle_artifact_id"]),
                packet_ids=tuple(str(item) for item in value["packet_ids"]),
                packet_digests=tuple(str(item) for item in value["packet_digests"]),
                oled_mapping_schema_digest=str(value["oled_mapping_schema_digest"]),
                prompt_template_digest=str(value["prompt_template_digest"]),
                provider_profile_ref=str(value["provider_profile_ref"]),
                model_identifier=str(value["model_identifier"]),
                model_version=str(value["model_version"]),
                mapping_config_digest=str(value["mapping_config_digest"]),
                request_digest=None if value.get("request_digest") is None else str(value["request_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("mapping request is malformed") from exc


def _evidence(value: Mapping[str, Any]) -> OledEvidenceRef:
    return OledEvidenceRef.from_dict(value)


@dataclass(frozen=True, slots=True)
class MappedField:
    """Optional flat representation used by provider adapters."""

    field_name: str
    value: Any
    candidate_id: str | None = None
    source_locator: SourceLocator | None = None
    status: str = "MAPPED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "candidate_id": self.candidate_id,
            "source_locator": None if self.source_locator is None else self.source_locator.to_dict(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class OledMappingRecord:
    """One schema-validated provider mapping with field-level evidence."""

    mapped_record_id: str
    molecule_identity: MoleculeIdentity
    property: NormalizedProperty
    measurement_condition: MeasurementCondition
    evidence: tuple[OledEvidenceRef, ...]
    confidence: float | None = None
    mapping_status: str = "MAPPED"
    claim_level: str | ClaimLevel = ClaimLevel.SYNTHETIC_CONTRACT_ONLY

    def __post_init__(self) -> None:
        validate_identifier(self.mapped_record_id, field="mapped_record_id")
        object.__setattr__(self, "molecule_identity", self.molecule_identity if isinstance(self.molecule_identity, MoleculeIdentity) else MoleculeIdentity.from_mapping(self.molecule_identity))
        object.__setattr__(self, "property", self.property if isinstance(self.property, NormalizedProperty) else NormalizedProperty.from_mapping(self.property))
        object.__setattr__(self, "measurement_condition", self.measurement_condition if isinstance(self.measurement_condition, MeasurementCondition) else MeasurementCondition.from_mapping(self.measurement_condition))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
                raise EvidenceContractError("mapping confidence must be a number from 0 to 1")
            object.__setattr__(self, "confidence", float(self.confidence))
        if self.mapping_status not in {"MAPPED", "UNRESOLVED", "REVIEW"}:
            raise EvidenceContractError("unknown mapping_status")
        claim = self.claim_level.value if isinstance(self.claim_level, ClaimLevel) else self.claim_level
        if not isinstance(claim, str) or claim not in {item.value for item in ClaimLevel}:
            raise EvidenceContractError("unknown mapping claim_level")
        object.__setattr__(self, "claim_level", claim)
        refs = tuple(item if isinstance(item, OledEvidenceRef) else _evidence(item) for item in self.evidence)
        if not refs:
            raise EvidenceContractError("mapping record requires field-level evidence")
        object.__setattr__(self, "evidence", refs)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.candidate_id for item in self.evidence))

    @property
    def fields(self) -> tuple[MappedField, ...]:
        return tuple(
            MappedField(
                field_name=field_name,
                value=value,
                candidate_id=next((item.candidate_id for item in self.evidence if item.field_name == field_name), None),
                source_locator=next((item.source_locator for item in self.evidence if item.field_name == field_name), None),
            )
            for field_name, value in (
                ("molecule_identity", self.molecule_identity.to_dict()),
                ("property_id", self.property.property_id),
                ("property_value", self.property.value),
                ("unit", self.property.unit),
                ("measurement_condition", self.measurement_condition.to_dict()),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped_record_id": self.mapped_record_id,
            "molecule_identity": self.molecule_identity.to_dict(),
            "property": self.property.to_dict(),
            "measurement_condition": self.measurement_condition.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "candidate_ids": list(self.candidate_ids),
            "confidence": self.confidence,
            "mapping_status": self.mapping_status,
            "claim_level": self.claim_level,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OledMappingRecord":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("mapping record must be an object")
        allowed = {"mapped_record_id", "molecule_identity", "property", "property_id", "value", "property_value", "unit", "original_value", "original_unit", "measurement_condition", "condition", "evidence", "candidate_ids", "confidence", "mapping_status", "claim_level"}
        if set(value) - allowed:
            raise EvidenceContractError("mapping record has unknown fields")
        property_value: Any = value.get("property")
        if property_value is None:
            property_value = {
                "property_id": value.get("property_id"),
                "value": value.get("value", value.get("property_value")),
                "unit": value.get("unit"),
            }
        condition = value.get("measurement_condition", value.get("condition", {"condition_status": "UNSPECIFIED"}))
        try:
            return cls(
                mapped_record_id=str(value["mapped_record_id"]),
                molecule_identity=MoleculeIdentity.from_mapping(value.get("molecule_identity")),
                property=NormalizedProperty.from_mapping(property_value),
                measurement_condition=MeasurementCondition.from_mapping(condition),
                evidence=tuple(OledEvidenceRef.from_dict(item) for item in value.get("evidence", ())),
                confidence=value.get("confidence"),
                mapping_status=str(value.get("mapping_status", "MAPPED")),
                claim_level=value.get("claim_level", ClaimLevel.SYNTHETIC_CONTRACT_ONLY.value),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("mapping record is malformed") from exc

    def to_oled_record(
        self,
        *,
        canonical_document_artifact_id: str,
        source_artifact_id: str,
        candidate_bundle_artifact_id: str,
        mapping_artifact_id: str,
        mapping_request_digest: str,
    ):
        from molly.domains.oled import OledRecord
        return OledRecord(
            record_id=self.mapped_record_id,
            canonical_document_artifact_id=canonical_document_artifact_id,
            source_artifact_id=source_artifact_id,
            molecule_identity=self.molecule_identity,
            property=self.property,
            measurement_condition=self.measurement_condition,
            evidence=self.evidence,
            claim_level=self.claim_level,
            validation_status="UNVALIDATED",
            candidate_bundle_artifact_id=candidate_bundle_artifact_id,
            mapping_artifact_id=mapping_artifact_id,
            mapping_request_digest=mapping_request_digest,
        )


def _check_evidence_bindings(record: OledMappingRecord, bundle: EvidenceCandidateBundle) -> None:
    known: dict[str, EvidenceCandidate] = {item.candidate_id: item for item in bundle.candidates}
    for evidence in record.evidence:
        candidate = known.get(evidence.candidate_id)
        if candidate is None:
            raise EvidenceIntegrityError("mapping evidence references an unknown candidate")
        if evidence.source_artifact_id != candidate.source_artifact_id:
            raise EvidenceIntegrityError("mapping evidence source does not match candidate")
        if evidence.source_locator.canonical_bytes() not in {item.canonical_bytes() for item in candidate.source_locators}:
            raise EvidenceIntegrityError("mapping evidence locator is not contained in candidate")
    bound = {item.field_name for item in record.evidence}
    required = {"molecule_identity", "property_id", "property_value", "unit", "measurement_condition"}
    if record.molecule_identity.smiles or record.molecule_identity.inchikey or record.molecule_identity.name:
        if "molecule_identity" not in bound:
            raise EvidenceIntegrityError("mapped molecule identity has no explicit evidence binding")
    if record.property.property_id is not None and "property_id" not in bound:
        raise EvidenceIntegrityError("mapped property has no explicit evidence binding")
    if record.property.value is not None and "property_value" not in bound:
        raise EvidenceIntegrityError("mapped property value has no explicit evidence binding")
    if record.property.unit is not None and "unit" not in bound:
        raise EvidenceIntegrityError("mapped property unit has no explicit evidence binding")
    if record.measurement_condition.condition_status != "UNSPECIFIED" and "measurement_condition" not in bound:
        raise EvidenceIntegrityError("mapped measurement condition has no explicit evidence binding")


@dataclass(frozen=True, slots=True)
class OledMappingResult:
    request_digest: str
    provider_profile_ref: str
    model_identifier: str
    model_version: str
    mapping_config_digest: str
    prompt_template_digest: str
    mapping_schema_digest: str
    records: tuple[OledMappingRecord, ...]
    warnings: tuple[str, ...] = ()
    response_digest: str | None = None
    schema_name: str = MAPPING_SCHEMA_NAME
    schema_version: str = MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_digest", validate_digest_reference(self.request_digest, field="request_digest"))
        for name in ("mapping_config_digest", "prompt_template_digest", "mapping_schema_digest"):
            object.__setattr__(self, name, validate_digest_reference(getattr(self, name), field=name))
        if self.schema_name != MAPPING_SCHEMA_NAME or self.schema_version != MAPPING_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported mapping result schema")
        records = tuple(sorted(self.records, key=lambda item: item.mapped_record_id))
        if len(records) > MAX_MAPPING_RECORDS:
            raise EvidenceContractError("mapping result exceeds the bounded record count")
        ids = tuple(item.mapped_record_id for item in records)
        if len(ids) != len(set(ids)):
            raise EvidenceContractError("mapping record IDs must be unique")
        object.__setattr__(self, "records", records)
        warnings = tuple(_text(item, "mapping warning", 1_024) for item in self.warnings)
        if len(warnings) > MAX_MAPPING_WARNINGS:
            raise EvidenceContractError("mapping warnings exceed the bounded limit")
        object.__setattr__(self, "warnings", warnings)
        if self.response_digest is not None:
            object.__setattr__(self, "response_digest", validate_digest_reference(self.response_digest, field="response_digest"))
            if self.response_digest != self.computed_response_digest:
                raise EvidenceIntegrityError("mapping response digest does not match result content")

    def to_dict(self, *, include_response_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "request_digest": self.request_digest,
            "provider_profile_ref": self.provider_profile_ref,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
            "mapping_config_digest": self.mapping_config_digest,
            "prompt_template_digest": self.prompt_template_digest,
            "mapping_schema_digest": self.mapping_schema_digest,
            "records": [item.to_dict() for item in self.records],
            "warnings": list(self.warnings),
        }
        if include_response_digest:
            payload["response_digest"] = self.response_digest or self.computed_response_digest
        return payload

    @property
    def computed_response_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict(include_response_digest=False)))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.canonical_bytes()))

    def to_artifact_draft(self) -> ArtifactDraft:
        return ArtifactDraft(
            content=self.canonical_bytes(),
            media_type="application/json",
            schema_name=self.schema_name,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OledMappingResult":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("mapping result must be an object")
        allowed = {"schema_name", "schema_version", "request_digest", "provider_profile_ref", "model_identifier", "model_version", "mapping_config_digest", "prompt_template_digest", "mapping_schema_digest", "records", "warnings", "response_digest"}
        if set(value) - allowed:
            raise EvidenceContractError("mapping result has unknown fields")
        try:
            return cls(
                schema_name=str(value.get("schema_name", MAPPING_SCHEMA_NAME)),
                schema_version=str(value["schema_version"]),
                request_digest=str(value["request_digest"]),
                provider_profile_ref=str(value["provider_profile_ref"]),
                model_identifier=str(value["model_identifier"]),
                model_version=str(value["model_version"]),
                mapping_config_digest=str(value["mapping_config_digest"]),
                prompt_template_digest=str(value["prompt_template_digest"]),
                mapping_schema_digest=str(value["mapping_schema_digest"]),
                records=tuple(OledMappingRecord.from_dict(item) for item in value.get("records", ())),
                warnings=tuple(str(item) for item in value.get("warnings", ())),
                response_digest=None if value.get("response_digest") is None else str(value["response_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("mapping result is malformed") from exc

    @classmethod
    def from_provider_payload(
        cls,
        payload: Mapping[str, Any],
        request: FrozenOledMappingRequest,
        candidate_bundle: EvidenceCandidateBundle,
    ) -> "OledMappingResult":
        if not isinstance(payload, Mapping):
            raise EvidenceContractError("structured mapping response must be an object")
        allowed = {"schema_name", "schema_version", "request_digest", "provider_profile_ref", "model_identifier", "model_version", "mapping_config_digest", "prompt_template_digest", "mapping_schema_digest", "records", "warnings", "response_digest"}
        if set(payload) - allowed:
            raise EvidenceContractError("structured mapping response has unknown fields")
        supplied_request_digest = payload.get("request_digest")
        if supplied_request_digest != request.request_digest and supplied_request_digest != request.computed_digest:
            raise EvidenceIntegrityError("mapping response is bound to a different request digest")
        records = tuple(OledMappingRecord.from_dict(item) for item in payload.get("records", ()))
        if len(records) > MAX_MAPPING_RECORDS:
            raise EvidenceContractError("mapping response has too many records")
        for record in records:
            _check_evidence_bindings(record, candidate_bundle)
        result = cls(
            schema_name=str(payload.get("schema_name", MAPPING_SCHEMA_NAME)),
            schema_version=str(payload.get("schema_version", MAPPING_SCHEMA_VERSION)),
            request_digest=request.computed_digest,
            provider_profile_ref=str(payload.get("provider_profile_ref", request.provider_profile_ref)),
            model_identifier=str(payload.get("model_identifier", request.model_identifier)),
            model_version=str(payload.get("model_version", request.model_version)),
            mapping_config_digest=str(payload.get("mapping_config_digest", request.mapping_config_digest)),
            prompt_template_digest=str(payload.get("prompt_template_digest", request.prompt_template_digest)),
            mapping_schema_digest=str(payload.get("mapping_schema_digest", request.oled_mapping_schema_digest)),
            records=records,
            warnings=tuple(payload.get("warnings", ())),
            response_digest=None if payload.get("response_digest") is None else str(payload["response_digest"]),
        )
        if result.provider_profile_ref != request.provider_profile_ref or result.model_identifier != request.model_identifier or result.model_version != request.model_version or result.mapping_config_digest != request.mapping_config_digest or result.prompt_template_digest != request.prompt_template_digest or result.mapping_schema_digest != request.oled_mapping_schema_digest:
            raise EvidenceIntegrityError("mapping response metadata does not match exact request")
        return result


class StructuredMappingProvider(Protocol):
    """Data-only provider interface; it has no Core authority."""

    provider_profile_ref: str
    model_identifier: str
    model_version: str

    def map(self, request: FrozenOledMappingRequest, packets: Sequence[EvidencePacket]) -> Mapping[str, Any]:
        ...


class ScriptedMappingProvider:
    """Deterministic offline provider keyed by the exact request digest."""

    def __init__(
        self,
        responses: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        profile_ref: str = "scripted-contract",
        model_identifier: str = "scripted-mapper",
        model_version: str = "1",
    ) -> None:
        self.responses = dict(responses or {})
        self.provider_profile_ref = _text(profile_ref, "provider_profile_ref")
        self.model_identifier = _text(model_identifier, "model_identifier")
        self.model_version = _text(model_version, "model_version")
        self.calls = 0

    def add_response(self, request_digest: str, payload: Mapping[str, Any]) -> None:
        self.responses[validate_digest_reference(request_digest, field="request_digest")] = dict(payload)

    def map(self, request: FrozenOledMappingRequest, packets: Sequence[EvidencePacket]) -> Mapping[str, Any]:
        self.calls += 1
        key = request.request_digest or request.computed_digest
        try:
            payload = dict(self.responses[key])
        except KeyError as exc:
            raise EvidenceContractError("scripted mapping provider has no exact request response") from exc
        if "request_digest" not in payload:
            payload["request_digest"] = key
        return payload


@dataclass(frozen=True, slots=True)
class MappingOutcome:
    request: FrozenOledMappingRequest
    result: OledMappingResult
    artifact_draft: ArtifactDraft


class MappingService:
    """Host-owned request construction and provider result verification."""

    def __init__(
        self,
        provider: StructuredMappingProvider,
        *,
        mapping_config: MappingConfig | None = None,
        provider_profile_ref: str | None = None,
        model_identifier: str | None = None,
        model_version: str | None = None,
    ) -> None:
        if not callable(getattr(provider, "map", None)):
            raise EvidenceContractError("mapping provider must expose map")
        self.provider = provider
        self.mapping_config = mapping_config or MappingConfig()
        self.provider_profile_ref = provider_profile_ref or provider.provider_profile_ref
        self.model_identifier = model_identifier or provider.model_identifier
        self.model_version = model_version or provider.model_version
        for value, field_name in ((self.provider_profile_ref, "provider_profile_ref"), (self.model_identifier, "model_identifier"), (self.model_version, "model_version")):
            _text(value, field_name, 256)

    def build_request(self, bundle: EvidenceCandidateBundle) -> tuple[FrozenOledMappingRequest, tuple[EvidencePacket, ...]]:
        packets = EvidencePacketBuilder().build(bundle)
        request = FrozenOledMappingRequest.create(
            candidate_bundle_artifact_id=bundle.artifact_id,
            packets=packets,
            provider_profile_ref=self.provider_profile_ref,
            model_identifier=self.model_identifier,
            model_version=self.model_version,
            mapping_config_digest=self.mapping_config.digest,
        )
        return request, packets

    def map(self, bundle: EvidenceCandidateBundle) -> MappingOutcome:
        if not isinstance(bundle, EvidenceCandidateBundle):
            raise EvidenceContractError("mapping requires an evidence candidate bundle")
        request, packets = self.build_request(bundle)
        payload = self.provider.map(request, packets)
        if isinstance(payload, OledMappingResult):
            result = payload
            if result.request_digest != request.computed_digest:
                raise EvidenceIntegrityError("provider result request digest mismatch")
            for record in result.records:
                _check_evidence_bindings(record, bundle)
        else:
            result = OledMappingResult.from_provider_payload(payload, request, bundle)
        return MappingOutcome(request=request, result=result, artifact_draft=result.to_artifact_draft())


__all__ = [
    "FrozenOledMappingRequest",
    "MAPPING_SCHEMA_DIGEST",
    "MAPPING_SCHEMA_NAME",
    "MAPPING_SCHEMA_VERSION",
    "MappedField",
    "MappingConfig",
    "MappingOutcome",
    "MappingService",
    "OledMappingRecord",
    "OledMappingResult",
    "PROMPT_TEMPLATE_DIGEST",
    "ScriptedMappingProvider",
    "StructuredMappingProvider",
]
