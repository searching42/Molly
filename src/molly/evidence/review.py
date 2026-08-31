"""Immutable OLED review bundles and host-created review binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any

from molly.core.tools import ArtifactDraft
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_artifact_id,
    validate_identifier,
)
from molly.core.reviews import ReviewRecord
from molly.core.validation import ValidationResult

from molly.domains.oled import OledRecord

from .candidates import EvidenceCandidateBundle
from .errors import EvidenceContractError, EvidenceIntegrityError
from .mapping import OledMappingResult, _check_evidence_bindings
from .validation import DuplicateGroup, OledValidationReport


REVIEW_BUNDLE_SCHEMA_NAME = "molly.evidence.oled-review-bundle"
REVIEW_BUNDLE_SCHEMA_VERSION = "1"
MAX_REVIEW_ISSUES = 512


def _ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    result = tuple(validate_artifact_id(value) for value in values)
    if len(result) != len(set(result)):
        raise EvidenceContractError(f"{field} must not contain duplicate artifact IDs")
    return result


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    """The exact immutable material prepared for human scientific review."""

    canonical_document_artifact_ids: tuple[str, ...]
    candidate_bundle_artifact_ids: tuple[str, ...]
    mapping_artifact_ids: tuple[str, ...]
    validation_report_artifact_id: str
    records: tuple[OledRecord, ...]
    validation_results: tuple[ValidationResult, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    blocking_issues: tuple[str, ...]
    review_summary: Mapping[str, Any]
    schema_name: str = REVIEW_BUNDLE_SCHEMA_NAME
    schema_version: str = REVIEW_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != REVIEW_BUNDLE_SCHEMA_NAME or self.schema_version != REVIEW_BUNDLE_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported review bundle schema")
        object.__setattr__(self, "canonical_document_artifact_ids", _ids(self.canonical_document_artifact_ids, "canonical_document_artifact_ids"))
        object.__setattr__(self, "candidate_bundle_artifact_ids", _ids(self.candidate_bundle_artifact_ids, "candidate_bundle_artifact_ids"))
        object.__setattr__(self, "mapping_artifact_ids", _ids(self.mapping_artifact_ids, "mapping_artifact_ids"))
        validate_artifact_id(self.validation_report_artifact_id)
        records = tuple(item if isinstance(item, OledRecord) else OledRecord.from_dict(item) for item in self.records)
        if len({item.record_id for item in records}) != len(records):
            raise EvidenceContractError("review records must have unique IDs")
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda item: item.record_id)))
        object.__setattr__(self, "validation_results", tuple(self.validation_results))
        object.__setattr__(self, "duplicate_groups", tuple(sorted(self.duplicate_groups, key=lambda item: item.group_id)))
        issues = tuple(sorted(set(str(item) for item in self.blocking_issues)))
        if len(issues) > MAX_REVIEW_ISSUES or any(not item or "\x00" in item for item in issues):
            raise EvidenceContractError("review blocking issues are outside the bounded contract")
        object.__setattr__(self, "blocking_issues", issues)
        object.__setattr__(self, "review_summary", freeze_json_mapping(self.review_summary, field="review_summary"))

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.canonical_bytes()))

    @property
    def digest(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    @property
    def canonical_document_artifact_id(self) -> str | None:
        return self.canonical_document_artifact_ids[0] if len(self.canonical_document_artifact_ids) == 1 else None

    @property
    def candidate_bundle_artifact_id(self) -> str | None:
        return self.candidate_bundle_artifact_ids[0] if len(self.candidate_bundle_artifact_ids) == 1 else None

    @property
    def mapping_artifact_id(self) -> str | None:
        return self.mapping_artifact_ids[0] if len(self.mapping_artifact_ids) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "canonical_document_artifact_ids": list(self.canonical_document_artifact_ids),
            "candidate_bundle_artifact_ids": list(self.candidate_bundle_artifact_ids),
            "mapping_artifact_ids": list(self.mapping_artifact_ids),
            "validation_report_artifact_id": self.validation_report_artifact_id,
            "records": [item.to_dict() for item in self.records],
            "validation_results": [item.to_dict() for item in self.validation_results],
            "duplicate_groups": [item.to_dict() for item in self.duplicate_groups],
            "blocking_issues": list(self.blocking_issues),
            "review_summary": thaw_json(self.review_summary),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_artifact_draft(self) -> ArtifactDraft:
        return ArtifactDraft(content=self.canonical_bytes(), media_type="application/json", schema_name=REVIEW_BUNDLE_SCHEMA_NAME, schema_version=REVIEW_BUNDLE_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewBundle":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("review bundle must be an object")
        allowed = {"schema_name", "schema_version", "canonical_document_artifact_ids", "candidate_bundle_artifact_ids", "mapping_artifact_ids", "validation_report_artifact_id", "records", "validation_results", "duplicate_groups", "blocking_issues", "review_summary"}
        if set(value) - allowed:
            raise EvidenceContractError("review bundle has unknown fields")
        try:
            return cls(
                schema_name=str(value.get("schema_name", REVIEW_BUNDLE_SCHEMA_NAME)),
                schema_version=str(value["schema_version"]),
                canonical_document_artifact_ids=tuple(str(item) for item in value["canonical_document_artifact_ids"]),
                candidate_bundle_artifact_ids=tuple(str(item) for item in value["candidate_bundle_artifact_ids"]),
                mapping_artifact_ids=tuple(str(item) for item in value["mapping_artifact_ids"]),
                validation_report_artifact_id=str(value["validation_report_artifact_id"]),
                records=tuple(OledRecord.from_dict(item) for item in value.get("records", ())),
                validation_results=tuple(ValidationResult.from_dict(item) for item in value.get("validation_results", ())),
                duplicate_groups=tuple(DuplicateGroup.from_dict(item) for item in value.get("duplicate_groups", ())),
                blocking_issues=tuple(str(item) for item in value.get("blocking_issues", ())),
                review_summary=dict(value.get("review_summary", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("review bundle is malformed") from exc


class ReviewBundleBuilder:
    """Construct review material only from exact preceding artifacts."""

    @staticmethod
    def build(
        *,
        canonical_document_artifact_ids: Sequence[str],
        candidate_bundle: EvidenceCandidateBundle,
        mapping_result: OledMappingResult,
        validation_report: OledValidationReport,
    ) -> ReviewBundle:
        if not isinstance(candidate_bundle, EvidenceCandidateBundle) or not isinstance(mapping_result, OledMappingResult) or not isinstance(validation_report, OledValidationReport):
            raise EvidenceContractError("review bundle inputs have incorrect types")
        if validation_report.candidate_bundle_artifact_id != candidate_bundle.artifact_id or validation_report.mapping_artifact_id != mapping_result.artifact_id:
            raise EvidenceIntegrityError("review inputs are not bound to exact candidate/mapping artifacts")
        if candidate_bundle.canonical_document_artifact_id not in tuple(canonical_document_artifact_ids):
            raise EvidenceIntegrityError("review bundle omits the canonical document used by candidates")
        return ReviewBundle(
            canonical_document_artifact_ids=tuple(canonical_document_artifact_ids),
            candidate_bundle_artifact_ids=(candidate_bundle.artifact_id,),
            mapping_artifact_ids=(mapping_result.artifact_id,),
            validation_report_artifact_id=validation_report.artifact_id,
            records=validation_report.records,
            validation_results=validation_report.validation_results,
            duplicate_groups=validation_report.duplicate_groups,
            blocking_issues=validation_report.blocking_issues,
            review_summary={
                "record_count": len(validation_report.records),
                "validation_status": validation_report.status,
                "duplicate_group_count": len(validation_report.duplicate_groups),
                "blocking_issue_count": len(validation_report.blocking_issues),
            },
        )

    @staticmethod
    def _read_exact(artifact_id: str, reader: Callable[[str], bytes]) -> bytes:
        validate_artifact_id(artifact_id)
        try:
            payload = reader(artifact_id)
        except Exception as exc:
            raise EvidenceIntegrityError("review input artifact could not be read") from exc
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise EvidenceIntegrityError("review input reader returned non-bytes")
        return bytes(payload)

    @classmethod
    def from_artifacts(
        cls,
        *,
        canonical_document_artifact_ids: Sequence[str],
        candidate_bundle_artifact_id: str,
        mapping_artifact_id: str,
        validation_report_artifact_id: str,
        reader: Callable[[str], bytes],
    ) -> ReviewBundle:
        from molly.documents.canonical import CanonicalDocument

        canonical_ids = tuple(validate_artifact_id(item) for item in canonical_document_artifact_ids)
        for canonical_id in canonical_ids:
            raw = cls._read_exact(canonical_id, reader)
            try:
                document = CanonicalDocument.from_dict(json.loads(raw.decode("utf-8")))
            except Exception as exc:
                raise EvidenceIntegrityError("canonical document review input is malformed") from exc
            if document.artifact_id != canonical_id:
                raise EvidenceIntegrityError("canonical document review input digest mismatch")
        candidate_raw = cls._read_exact(candidate_bundle_artifact_id, reader)
        mapping_raw = cls._read_exact(mapping_artifact_id, reader)
        validation_raw = cls._read_exact(validation_report_artifact_id, reader)
        try:
            candidate = EvidenceCandidateBundle.from_dict(json.loads(candidate_raw.decode("utf-8")))
            mapping = OledMappingResult.from_dict(json.loads(mapping_raw.decode("utf-8")))
            validation = OledValidationReport.from_dict(json.loads(validation_raw.decode("utf-8")))
        except AttributeError:
            raise
        except Exception as exc:
            raise EvidenceIntegrityError("review input artifact is malformed") from exc
        if candidate.artifact_id != candidate_bundle_artifact_id or mapping.artifact_id != mapping_artifact_id or validation.artifact_id != validation_report_artifact_id:
            raise EvidenceIntegrityError("review input artifact bytes do not match declared identities")
        for record in mapping.records:
            _check_evidence_bindings(record, candidate)
        return cls.build(canonical_document_artifact_ids=canonical_ids, candidate_bundle=candidate, mapping_result=mapping, validation_report=validation)


__all__ = ["REVIEW_BUNDLE_SCHEMA_NAME", "REVIEW_BUNDLE_SCHEMA_VERSION", "ReviewBundle", "ReviewBundleBuilder"]
