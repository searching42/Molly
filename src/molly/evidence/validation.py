"""Deterministic OLED validation, duplicate/conflict, and leakage checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from molly.core.tools import ArtifactDraft
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    sha256_bytes,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
)
from molly.core.validation import ValidationResult, ValidationScope, ValidationStatus
from molly.domains.oled import OledRecord, OledValidationStatus

from .errors import EvidenceContractError, EvidenceIntegrityError


VALIDATION_SCHEMA_NAME = "molly.evidence.oled-validation"
VALIDATION_SCHEMA_VERSION = "1"
VALIDATION_TIMESTAMP = "1970-01-01T00:00:00Z"
MAX_BLOCKING_ISSUES = 512


@dataclass(frozen=True, slots=True)
class OledValidationConfig:
    validator_id: str = "oled_deterministic_validator"
    validator_version: str = "1"
    duplicate_tolerance: float = 0.02
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_identifier(self.validator_id, field="validator_id")
        validate_identifier(self.validator_version, field="validator_version")
        validate_identifier(self.schema_version, field="validation schema_version")
        if isinstance(self.duplicate_tolerance, bool) or not isinstance(self.duplicate_tolerance, (int, float)) or not math.isfinite(float(self.duplicate_tolerance)) or not 0 <= float(self.duplicate_tolerance) <= 1:
            raise EvidenceContractError("duplicate_tolerance is outside the bounded range")
        object.__setattr__(self, "duplicate_tolerance", float(self.duplicate_tolerance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "duplicate_tolerance": self.duplicate_tolerance,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


class DuplicateClassification:
    PRIMARY = "PRIMARY"
    CONSISTENT_DUPLICATE_CANDIDATE = "CONSISTENT_DUPLICATE_CANDIDATE"
    CONFLICT_CANDIDATE = "CONFLICT_CANDIDATE"
    CONFLICTING_DUPLICATE_CANDIDATE = "CONFLICTING_DUPLICATE_CANDIDATE"


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    group_id: str
    comparison_key: str
    record_ids: tuple[str, ...]
    classifications: Mapping[str, str]
    status: str

    def __post_init__(self) -> None:
        validate_identifier(self.group_id, field="duplicate group_id")
        validate_digest_reference(self.comparison_key, field="duplicate comparison_key")
        ids = tuple(validate_identifier(value, field="duplicate record_id") for value in self.record_ids)
        if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise EvidenceContractError("duplicate group record IDs must be sorted and unique")
        object.__setattr__(self, "record_ids", ids)
        classifications = dict(self.classifications)
        if set(classifications) != set(ids):
            raise EvidenceContractError("duplicate classifications must cover exactly the record IDs")
        allowed = {
            DuplicateClassification.PRIMARY,
            DuplicateClassification.CONSISTENT_DUPLICATE_CANDIDATE,
            DuplicateClassification.CONFLICT_CANDIDATE,
            DuplicateClassification.CONFLICTING_DUPLICATE_CANDIDATE,
        }
        if any(value not in allowed for value in classifications.values()):
            raise EvidenceContractError("unknown duplicate classification")
        object.__setattr__(self, "classifications", dict(sorted(classifications.items())))
        if self.status not in {ValidationStatus.PASS.value, ValidationStatus.REVIEW.value, ValidationStatus.FAIL.value}:
            raise EvidenceContractError("unknown duplicate group status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "comparison_key": self.comparison_key,
            "record_ids": list(self.record_ids),
            "classifications": dict(self.classifications),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DuplicateGroup":
        try:
            return cls(
                group_id=str(value["group_id"]),
                comparison_key=str(value["comparison_key"]),
                record_ids=tuple(str(item) for item in value["record_ids"]),
                classifications=dict(value["classifications"]),
                status=str(value["status"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("duplicate group is malformed") from exc


def _validation(
    config: OledValidationConfig,
    *,
    subject_ids: Sequence[str],
    status: str,
    reason: str,
    evidence_artifact_ids: Sequence[str] = (),
    source_references: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ValidationResult:
    return ValidationResult(
        validator_id=config.validator_id,
        validator_version=config.validator_version,
        scope=ValidationScope.BUNDLE if len(subject_ids) > 1 else ValidationScope.ARTIFACT,
        subject_ids=tuple(subject_ids),
        status=status,
        reason=reason,
        evidence_artifact_ids=tuple(dict.fromkeys(evidence_artifact_ids)),
        source_references=tuple(dict.fromkeys(source_references)),
        timestamp=VALIDATION_TIMESTAMP,
        metadata=metadata or {},
    )


@dataclass(frozen=True, slots=True)
class OledValidationReport:
    candidate_bundle_artifact_id: str
    mapping_artifact_id: str
    records: tuple[OledRecord, ...]
    validation_results: tuple[ValidationResult, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    blocking_issues: tuple[str, ...]
    validation_config_digest: str
    schema_name: str = VALIDATION_SCHEMA_NAME
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_artifact_id(self.candidate_bundle_artifact_id)
        validate_artifact_id(self.mapping_artifact_id)
        if self.schema_name != VALIDATION_SCHEMA_NAME or self.schema_version != VALIDATION_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported OLED validation schema")
        object.__setattr__(self, "validation_config_digest", validate_digest_reference(self.validation_config_digest, field="validation_config_digest"))
        records = tuple(self.records)
        ids = tuple(item.record_id for item in records)
        if len(ids) != len(set(ids)):
            raise EvidenceContractError("validation record IDs must be unique")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "validation_results", tuple(self.validation_results))
        groups = tuple(self.duplicate_groups)
        group_ids = tuple(item.group_id for item in groups)
        if len(group_ids) != len(set(group_ids)):
            raise EvidenceContractError("duplicate group IDs must be unique")
        object.__setattr__(self, "duplicate_groups", groups)
        issues = tuple(sorted(set(str(item) for item in self.blocking_issues)))
        if len(issues) > MAX_BLOCKING_ISSUES or any(not item or "\x00" in item for item in issues):
            raise EvidenceContractError("blocking issues are outside the bounded contract")
        object.__setattr__(self, "blocking_issues", issues)

    @property
    def status(self) -> str:
        if self.blocking_issues:
            return ValidationStatus.REVIEW.value
        return ValidationStatus.PASS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "candidate_bundle_artifact_id": self.candidate_bundle_artifact_id,
            "mapping_artifact_id": self.mapping_artifact_id,
            "records": [item.to_dict() for item in self.records],
            "validation_results": [item.to_dict() for item in self.validation_results],
            "duplicate_groups": [item.to_dict() for item in self.duplicate_groups],
            "blocking_issues": list(self.blocking_issues),
            "validation_config_digest": self.validation_config_digest,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.canonical_bytes()))

    def to_artifact_draft(self) -> ArtifactDraft:
        return ArtifactDraft(content=self.canonical_bytes(), media_type="application/json", schema_name=self.schema_name, schema_version=self.schema_version)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OledValidationReport":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("validation report must be an object")
        allowed = {"schema_name", "schema_version", "candidate_bundle_artifact_id", "mapping_artifact_id", "records", "validation_results", "duplicate_groups", "blocking_issues", "validation_config_digest"}
        if set(value) - allowed:
            raise EvidenceContractError("validation report has unknown fields")
        try:
            return cls(
                schema_name=str(value.get("schema_name", VALIDATION_SCHEMA_NAME)),
                schema_version=str(value["schema_version"]),
                candidate_bundle_artifact_id=str(value["candidate_bundle_artifact_id"]),
                mapping_artifact_id=str(value["mapping_artifact_id"]),
                records=tuple(OledRecord.from_dict(item) for item in value.get("records", ())),
                validation_results=tuple(ValidationResult.from_dict(item) for item in value.get("validation_results", ())),
                duplicate_groups=tuple(DuplicateGroup.from_dict(item) for item in value.get("duplicate_groups", ())),
                blocking_issues=tuple(str(item) for item in value.get("blocking_issues", ())),
                validation_config_digest=str(value["validation_config_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("validation report is malformed") from exc


def _record_issue(record: OledRecord) -> tuple[str, str] | None:
    if not record.molecule_identity.resolved:
        return "identity_unresolved", f"{record.record_id}: molecule identity is unresolved"
    if not record.property.normalized:
        return "property_unit_unresolved", f"{record.record_id}: property value/unit is unresolved"
    if record.measurement_condition.condition_status in {"UNSPECIFIED", "UNKNOWN"}:
        return "condition_unresolved", f"{record.record_id}: measurement condition is unresolved"
    if not record.evidence:
        return "evidence_missing", f"{record.record_id}: explicit evidence is missing"
    return None


def validate_records(
    records: Sequence[OledRecord],
    candidate_bundle_artifact_id: str,
    mapping_artifact_id: str,
    config: OledValidationConfig | None = None,
    *,
    dataset_groups: Mapping[str, str] | None = None,
    paper_groups: Mapping[str, str] | None = None,
    scaffold_groups: Mapping[str, str] | None = None,
) -> OledValidationReport:
    """Validate records without splitting, merging, or making scientific claims."""

    config = config or OledValidationConfig()
    validate_artifact_id(candidate_bundle_artifact_id)
    validate_artifact_id(mapping_artifact_id)
    values = tuple(sorted(records, key=lambda item: item.record_id))
    if any(not isinstance(item, OledRecord) for item in values):
        raise EvidenceContractError("validate_records accepts only OledRecord values")
    by_key: dict[str, list[OledRecord]] = defaultdict(list)
    for record in values:
        by_key[record.comparison_key].append(record)
    groups: list[DuplicateGroup] = []
    issues: list[str] = []
    results: list[ValidationResult] = []
    revised: dict[str, OledRecord] = {}

    for record in values:
        issue = _record_issue(record)
        if issue is None:
            results.append(_validation(config, subject_ids=(record.record_id,), status=ValidationStatus.PASS.value, reason="record satisfies bounded OLED validation", evidence_artifact_ids=(candidate_bundle_artifact_id, mapping_artifact_id), source_references=tuple(item.candidate_id for item in record.evidence)))
            revised[record.record_id] = record.with_validation_status(OledValidationStatus.PASS.value)
        else:
            code, message = issue
            issues.append(code)
            results.append(_validation(config, subject_ids=(record.record_id,), status=ValidationStatus.REVIEW.value, reason=message, evidence_artifact_ids=(candidate_bundle_artifact_id, mapping_artifact_id), source_references=tuple(item.candidate_id for item in record.evidence), metadata={"issue_code": code}))
            revised[record.record_id] = record.with_validation_status(OledValidationStatus.REVIEW.value)

    for comparison_key in sorted(by_key):
        group_records = sorted(by_key[comparison_key], key=lambda item: item.record_id)
        if len(group_records) == 1:
            classifications = {group_records[0].record_id: DuplicateClassification.PRIMARY}
            group_status = ValidationStatus.PASS.value
        else:
            numbers = [record.property.value for record in group_records]
            consistent = all(a is not None and b is not None and abs(float(a) - float(b)) <= config.duplicate_tolerance for index, a in enumerate(numbers) for b in numbers[index + 1:])
            if consistent:
                classifications = {record.record_id: (DuplicateClassification.PRIMARY if index == 0 else DuplicateClassification.CONSISTENT_DUPLICATE_CANDIDATE) for index, record in enumerate(group_records)}
                group_status = ValidationStatus.PASS.value
            else:
                classifications = {record.record_id: (DuplicateClassification.CONFLICT_CANDIDATE if index == 0 else DuplicateClassification.CONFLICTING_DUPLICATE_CANDIDATE) for index, record in enumerate(group_records)}
                group_status = ValidationStatus.REVIEW.value
                issues.append(f"duplicate_conflict:{comparison_key}")
                results.append(_validation(config, subject_ids=tuple(record.record_id for record in group_records), status=ValidationStatus.REVIEW.value, reason="same identity/property/condition has conflicting values", evidence_artifact_ids=(candidate_bundle_artifact_id, mapping_artifact_id), source_references=tuple(item.candidate_id for record in group_records for item in record.evidence), metadata={"comparison_key": comparison_key, "duplicate_conflict": True}))
                for record in group_records:
                    revised[record.record_id] = record.with_validation_status(OledValidationStatus.REVIEW.value)
        group_digest = sha256_bytes(canonical_json_bytes({
            "comparison_key": comparison_key,
            "record_ids": [item.record_id for item in group_records],
        }))
        group_id = f"duplicate_{group_digest}"
        group = DuplicateGroup(group_id=group_id, comparison_key=comparison_key, record_ids=tuple(item.record_id for item in group_records), classifications=classifications, status=group_status)
        groups.append(group)
        for record in group_records:
            revised[record.record_id] = replace(revised[record.record_id], duplicate_group=group_id)

    leakage_results = detect_leakage(values, dataset_groups=dataset_groups, paper_groups=paper_groups, scaffold_groups=scaffold_groups, config=config, evidence_artifact_ids=(candidate_bundle_artifact_id, mapping_artifact_id))
    results.extend(leakage_results)
    if leakage_results:
        issues.extend(item.metadata.get("issue_code", "leakage") for item in leakage_results)
    ordered_records = tuple(revised[item.record_id] for item in sorted(values, key=lambda item: item.record_id))
    return OledValidationReport(
        candidate_bundle_artifact_id=candidate_bundle_artifact_id,
        mapping_artifact_id=mapping_artifact_id,
        records=ordered_records,
        validation_results=tuple(results),
        duplicate_groups=tuple(sorted(groups, key=lambda item: item.group_id)),
        blocking_issues=tuple(sorted(set(issues))),
        validation_config_digest=config.digest,
    )


def detect_leakage(
    records: Sequence[OledRecord],
    *,
    dataset_groups: Mapping[str, str] | None = None,
    paper_groups: Mapping[str, str] | None = None,
    scaffold_groups: Mapping[str, str] | None = None,
    config: OledValidationConfig | None = None,
    evidence_artifact_ids: Sequence[str] = (),
) -> tuple[ValidationResult, ...]:
    """Report explicit identity/source/group collisions; never perform a split."""

    config = config or OledValidationConfig()
    output: list[ValidationResult] = []
    group_specs = (("exact_identity", dataset_groups, lambda record: record.molecule_identity.identity_key), ("paper_source", paper_groups, lambda record: record.source_artifact_id), ("scaffold", scaffold_groups, lambda record: record.molecule_identity.identity_key))
    for code, groups, key_fn in group_specs:
        if not groups:
            continue
        buckets: dict[str, list[OledRecord]] = defaultdict(list)
        for record in records:
            group = groups.get(record.record_id)
            if group is not None:
                buckets[key_fn(record)].append(record)
        for key, bucket in sorted(buckets.items()):
            explicit_groups = {groups.get(record.record_id) for record in bucket}
            if len(bucket) > 1 and len(explicit_groups) > 1:
                output.append(_validation(config, subject_ids=tuple(sorted(record.record_id for record in bucket)), status=ValidationStatus.REVIEW.value, reason=f"{code} leakage across explicit groups", evidence_artifact_ids=tuple(evidence_artifact_ids), source_references=tuple(item.candidate_id for record in bucket for item in record.evidence), metadata={"issue_code": f"leakage:{code}", "identity_key": key}))
    return tuple(output)


class OledValidator:
    """Object-oriented wrapper for callers that prefer a service boundary."""

    def __init__(self, config: OledValidationConfig | None = None) -> None:
        self.config = config or OledValidationConfig()

    def validate(self, records: Sequence[OledRecord], candidate_bundle_artifact_id: str, mapping_artifact_id: str, **kwargs: Any) -> OledValidationReport:
        return validate_records(records, candidate_bundle_artifact_id, mapping_artifact_id, self.config, **kwargs)


__all__ = [
    "DuplicateClassification",
    "DuplicateGroup",
    "OledValidationConfig",
    "OledValidationReport",
    "OledValidator",
    "VALIDATION_SCHEMA_NAME",
    "VALIDATION_SCHEMA_VERSION",
    "VALIDATION_TIMESTAMP",
    "detect_leakage",
    "validate_records",
]
