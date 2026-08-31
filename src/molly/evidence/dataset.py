"""Deterministic reviewed OLED dataset export with an exact review gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import io
import json
import math
from typing import Any

from molly.core.tools import ArtifactDraft
from molly.core.errors import CoreContractError
from molly.core.ids import artifact_id_for_sha256, canonical_json_bytes, sha256_bytes, validate_artifact_id
from molly.core.reviews import ReviewDecision, ReviewRecord
from molly.domains.oled import OledRecord, OledValidationStatus

from .errors import EvidenceContractError, EvidenceIntegrityError
from .review import ReviewBundle, REVIEW_BUNDLE_SCHEMA_NAME


DATASET_SCHEMA_NAME = "molly.evidence.reviewed-dataset"
DATASET_CSV_SCHEMA_NAME = "molly.evidence.reviewed-dataset-csv"
DATASET_SCHEMA_VERSION = "1"
DEFAULT_DATASET_COLUMNS = (
    "row_id",
    "molecule_identity",
    "property_id",
    "value",
    "unit",
    "original_value",
    "original_unit",
    "measurement_condition",
    "canonical_document_artifact_id",
    "source_artifact_id",
    "evidence_candidate_ids",
    "evidence_locators",
    "mapping_artifact_id",
    "mapping_request_digest",
    "claim_level",
    "review_bundle_artifact_id",
)


@dataclass(frozen=True, slots=True)
class DatasetExportConfig:
    schema_version: str = DATASET_SCHEMA_VERSION
    columns: tuple[str, ...] = DEFAULT_DATASET_COLUMNS
    numeric_precision: int = 15
    numeric_format_version: str = "g15"

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported dataset schema version")
        columns = tuple(str(item) for item in self.columns)
        if columns != DEFAULT_DATASET_COLUMNS or len(columns) != len(set(columns)):
            raise EvidenceContractError("dataset columns are not the frozen fixed order")
        object.__setattr__(self, "columns", columns)
        if isinstance(self.numeric_precision, bool) or not isinstance(self.numeric_precision, int) or not 1 <= self.numeric_precision <= 17:
            raise EvidenceContractError("numeric_precision is outside the bounded range")
        if self.numeric_format_version != "g15":
            raise EvidenceContractError("unsupported numeric format version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "columns": list(self.columns),
            "numeric_precision": self.numeric_precision,
            "numeric_format_version": self.numeric_format_version,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DatasetExport:
    rows: tuple[Mapping[str, Any], ...]
    json_draft: ArtifactDraft
    csv_draft: ArtifactDraft
    review_bundle_artifact_id: str
    review_digest: str
    export_config_digest: str

    @property
    def json_artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.json_draft.content))

    @property
    def csv_artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.csv_draft.content))

    @property
    def json_artifact_draft(self) -> ArtifactDraft:
        return self.json_draft

    @property
    def csv_artifact_draft(self) -> ArtifactDraft:
        return self.csv_draft


def _format_number(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceContractError("dataset cannot export non-finite numbers")
        return format(value, f".{precision}g")
    return str(value)


def _json_cell(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _row(record: OledRecord, review_bundle_artifact_id: str) -> dict[str, Any]:
    return {
        "row_id": record.record_id,
        "molecule_identity": record.molecule_identity.to_dict(),
        "property_id": record.property.property_id,
        "value": record.property.value,
        "unit": record.property.unit,
        "original_value": record.property.original_value,
        "original_unit": record.property.original_unit,
        "measurement_condition": record.measurement_condition.to_dict(),
        "canonical_document_artifact_id": record.canonical_document_artifact_id,
        "source_artifact_id": record.source_artifact_id,
        "evidence_candidate_ids": [item.candidate_id for item in record.evidence],
        "evidence_locators": [item.source_locator.to_dict() for item in record.evidence],
        "mapping_artifact_id": record.mapping_artifact_id,
        "mapping_request_digest": record.mapping_request_digest,
        "claim_level": record.claim_level,
        "review_bundle_artifact_id": review_bundle_artifact_id,
    }


class DatasetExporter:
    """Pure deterministic exporter; publication and review creation stay host-owned."""

    def __init__(self, config: DatasetExportConfig | None = None) -> None:
        self.config = config or DatasetExportConfig()

    def export(
        self,
        review_bundle: ReviewBundle,
        review_record: ReviewRecord,
        *,
        review_bundle_artifact_id: str | None = None,
        review_bundle_sha256: str | None = None,
    ) -> DatasetExport:
        if not isinstance(review_bundle, ReviewBundle) or not isinstance(review_record, ReviewRecord):
            raise EvidenceContractError("dataset export requires ReviewBundle and ReviewRecord")
        bundle_id = review_bundle_artifact_id or review_bundle.artifact_id
        validate_artifact_id(bundle_id)
        if bundle_id != review_bundle.artifact_id:
            raise EvidenceIntegrityError("review bundle identity does not match its exact bytes")
        bundle_sha = sha256_bytes(review_bundle.canonical_bytes())
        if review_bundle_sha256 is not None and review_bundle_sha256 != bundle_sha:
            raise EvidenceIntegrityError("review bundle SHA-256 does not match exact bytes")
        review_record.assert_matches(bundle_id, bundle_sha)
        if review_record.decision != ReviewDecision.APPROVED.value:
            raise EvidenceContractError("reviewed dataset export requires APPROVED review")
        if review_bundle.blocking_issues:
            raise EvidenceContractError("structural review blockers prevent dataset export")
        if any(item.validation_status != OledValidationStatus.PASS.value for item in review_bundle.records):
            raise EvidenceContractError("all reviewed OLED records must have PASS validation status")
        rows = tuple(_row(record, bundle_id) for record in sorted(review_bundle.records, key=lambda item: item.record_id))
        json_body = {
            "schema_name": DATASET_SCHEMA_NAME,
            "schema_version": self.config.schema_version,
            "review_bundle_artifact_id": bundle_id,
            "review_digest": review_record.digest,
            "export_config_digest": self.config.digest,
            "rows": list(rows),
        }
        json_bytes = canonical_json_bytes(json_body)
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(self.config.columns)
        for row in rows:
            writer.writerow((
                row["row_id"],
                _json_cell(row["molecule_identity"]),
                row["property_id"],
                _format_number(row["value"], self.config.numeric_precision),
                row["unit"] or "",
                _format_number(row["original_value"], self.config.numeric_precision),
                row["original_unit"] or "",
                _json_cell(row["measurement_condition"]),
                row["canonical_document_artifact_id"],
                row["source_artifact_id"],
                ";".join(row["evidence_candidate_ids"]),
                _json_cell(row["evidence_locators"]),
                row["mapping_artifact_id"] or "",
                row["mapping_request_digest"] or "",
                row["claim_level"],
                row["review_bundle_artifact_id"],
            ))
        csv_bytes = stream.getvalue().encode("utf-8")
        return DatasetExport(
            rows=rows,
            json_draft=ArtifactDraft(content=json_bytes, media_type="application/json", schema_name=DATASET_SCHEMA_NAME, schema_version=self.config.schema_version),
            csv_draft=ArtifactDraft(content=csv_bytes, media_type="text/csv", schema_name=DATASET_CSV_SCHEMA_NAME, schema_version=self.config.schema_version),
            review_bundle_artifact_id=bundle_id,
            review_digest=review_record.digest,
            export_config_digest=self.config.digest,
        )

    def export_from_bytes(
        self,
        review_bundle_bytes: bytes,
        review_record_bytes: bytes,
        review_bundle_artifact_id: str,
        *,
        review_bundle_sha256: str | None = None,
    ) -> DatasetExport:
        validate_artifact_id(review_bundle_artifact_id)
        try:
            bundle = ReviewBundle.from_dict(json.loads(bytes(review_bundle_bytes).decode("utf-8")))
            review = ReviewRecord.from_dict(json.loads(bytes(review_record_bytes).decode("utf-8")))
        except Exception as exc:
            raise EvidenceIntegrityError("review export input is not valid canonical JSON") from exc
        if bundle.artifact_id != review_bundle_artifact_id:
            raise EvidenceIntegrityError("review bundle bytes do not match declared artifact ID")
        return self.export(bundle, review, review_bundle_artifact_id=review_bundle_artifact_id, review_bundle_sha256=review_bundle_sha256)


__all__ = [
    "DATASET_CSV_SCHEMA_NAME",
    "DATASET_SCHEMA_NAME",
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_DATASET_COLUMNS",
    "DatasetExport",
    "DatasetExportConfig",
    "DatasetExporter",
    "ReviewedDataset",
]


ReviewedDataset = DatasetExport
