from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import write_json


PropertyValueKind = Literal["numeric_scalar", "numeric_range", "numeric_tuple", "unknown"]
PropertyFamily = Literal[
    "photophysical",
    "electronic",
    "energetic",
    "device",
    "structural",
    "synthetic",
    "spectroscopic",
    "computational",
    "other",
    "unknown",
]
UnitStatus = Literal["explicit", "inferred", "not_applicable", "missing", "unknown"]
TrainabilityDecision = Literal["candidate", "reject", "needs_review"]
ExtractionSourceKind = Literal["deterministic", "llm_agent", "hybrid", "human_seeded", "unknown"]

_SCHEMA_VERSION = "custom_corpus_property_candidate.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")
_PRIVATE_PATH_MARKERS = ("/Users/", "/home/", "C:\\")
_LABEL_LIMIT = 200
_SUMMARY_LIMIT = 500
_VALUE_RAW_LIMIT = 200


class CustomCorpusPropertyCandidateError(ValueError):
    pass


class PropertyCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_candidate_id: str
    corpus_id: str
    dry_run_id: str
    document_id: str
    source_record_id: str = ""
    source_artifact_sha256: str
    parsed_document_sha256: str = ""
    page: int | None = None
    table_id: str = ""
    row_id: str = ""
    column_name: str = ""
    raw_property_label: str
    canonical_property_guess: str = ""
    property_family: PropertyFamily = "unknown"
    field_name: str
    value_kind: PropertyValueKind
    value_raw_summary: str = ""
    value_normalized: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    value_tuple: list[float] = Field(default_factory=list)
    unit_raw: str = ""
    unit_normalized: str = ""
    unit_status: UnitStatus = "unknown"
    entity_id: str
    entity_type: str
    entity_label_summary: str = ""
    method_summary: str = ""
    condition_summary: str = ""
    provenance_summary: str
    extraction_source: ExtractionSourceKind = "unknown"
    extractor_label: str = ""
    confidence: float
    trainability_decision: TrainabilityDecision
    decision_reason: str
    review_required: bool = True
    rejection_reason: str = ""
    notes: str = ""

    @field_validator(
        "property_candidate_id",
        "corpus_id",
        "dry_run_id",
        "document_id",
        "field_name",
        "entity_id",
        mode="before",
    )
    @classmethod
    def _clean_required_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("source_record_id", "table_id", "row_id", "column_name", mode="before")
    @classmethod
    def _clean_optional_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if clean and not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator(
        "raw_property_label",
        "canonical_property_guess",
        "unit_raw",
        "unit_normalized",
        "entity_type",
        "entity_label_summary",
        "method_summary",
        "condition_summary",
        "provenance_summary",
        "extractor_label",
        "decision_reason",
        "rejection_reason",
        "notes",
        mode="before",
    )
    @classmethod
    def _clean_summary_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_LIMIT)

    @field_validator("value_raw_summary", mode="before")
    @classmethod
    def _clean_raw_value_summary(cls, value: Any) -> str:
        return _clean_text_value(value, field_name="value_raw_summary", max_length=_VALUE_RAW_LIMIT)

    @field_validator("canonical_property_guess", "entity_type", "extractor_label")
    @classmethod
    def _validate_safe_optional_label(cls, value: str, info: Any) -> str:
        if value and not _SAFE_ID_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a safe label")
        return value

    @field_validator("source_artifact_sha256", mode="before")
    @classmethod
    def _clean_source_sha(cls, value: Any) -> str:
        return _normalize_required_sha256(value, field_name="source_artifact_sha256")

    @field_validator("parsed_document_sha256", mode="before")
    @classmethod
    def _clean_optional_sha(cls, value: Any) -> str:
        return _normalize_optional_sha256(value, field_name="parsed_document_sha256")

    @field_validator("value_normalized", "value_min", "value_max", mode="before")
    @classmethod
    def _clean_optional_float(cls, value: Any, info: Any) -> float | None:
        return _optional_finite_float(value, field_name=str(info.field_name))

    @field_validator("value_tuple", mode="before")
    @classmethod
    def _clean_value_tuple(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("value_tuple must be a list")
        cleaned = [_required_finite_float(item, field_name="value_tuple") for item in value]
        return cleaned

    @field_validator("confidence", mode="before")
    @classmethod
    def _clean_confidence(cls, value: Any) -> float:
        confidence = _required_finite_float(value, field_name="confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return confidence

    @model_validator(mode="after")
    def _validate_record(self) -> "PropertyCandidateRecord":
        self._validate_value_shape()
        if self.trainability_decision == "candidate":
            self._validate_candidate()
        if self.trainability_decision == "reject" and not self.rejection_reason:
            raise ValueError("trainability_decision=reject requires rejection_reason")
        if self.trainability_decision == "needs_review" and not (self.notes or self.decision_reason):
            raise ValueError("trainability_decision=needs_review requires notes or decision_reason")
        return self

    def _validate_value_shape(self) -> None:
        if self.value_kind == "numeric_scalar":
            if self.value_min is not None or self.value_max is not None or self.value_tuple:
                raise ValueError("numeric_scalar records must not include range or tuple values")
        elif self.value_kind == "numeric_range":
            if self.value_normalized is not None or self.value_tuple:
                raise ValueError("numeric_range records must not include scalar or tuple values")
            if self.value_min is not None and self.value_max is not None and self.value_min > self.value_max:
                raise ValueError("value_min must be less than or equal to value_max")
        elif self.value_kind == "numeric_tuple":
            if self.value_normalized is not None or self.value_min is not None or self.value_max is not None:
                raise ValueError("numeric_tuple records must not include scalar or range values")
            if self.value_tuple and not (2 <= len(self.value_tuple) <= 8):
                raise ValueError("value_tuple must contain 2 to 8 values")
        elif self.value_kind == "unknown":
            if self.value_normalized is not None or self.value_min is not None or self.value_max is not None or self.value_tuple:
                raise ValueError("unknown value_kind must not include machine-readable numeric values")

    def _validate_candidate(self) -> None:
        if self.value_kind == "unknown":
            raise ValueError("candidate records require value_kind to be known")
        if self.value_kind == "numeric_scalar" and self.value_normalized is None:
            raise ValueError("candidate numeric_scalar records require value_normalized")
        if self.value_kind == "numeric_range" and (self.value_min is None or self.value_max is None):
            raise ValueError("candidate numeric_range records require value_min and value_max")
        if self.value_kind == "numeric_tuple" and not self.value_tuple:
            raise ValueError("candidate numeric_tuple records require value_tuple")
        for field_name in ("raw_property_label", "field_name", "entity_id", "entity_type", "provenance_summary", "decision_reason"):
            if not getattr(self, field_name):
                raise ValueError(f"candidate records require {field_name}")
        if self.unit_status in {"explicit", "inferred"} and not self.unit_normalized:
            raise ValueError("candidate records with explicit or inferred units require unit_normalized")
        if self.unit_status == "missing" and self.unit_normalized != "not_applicable":
            raise ValueError("candidate records cannot use unit_status=missing without unit_normalized=not_applicable")
        if self.review_required is not True:
            raise ValueError("candidate records require review_required=true")

    def candidate_target(self) -> tuple[str, str, str, str, str, str]:
        return (self.document_id, self.entity_id, self.field_name, self.table_id, self.row_id, self.column_name)


class PropertyCandidateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    property_candidate_manifest_id: str
    corpus_id: str
    dry_run_id: str
    created_at: str
    created_by: str
    source_manifest_sha256: str = ""
    source_dry_run_report_sha256: str
    candidate_policy: str
    extraction_scope: str
    records: list[PropertyCandidateRecord]

    @field_validator("schema_version", "created_at", "created_by", "candidate_policy", "extraction_scope", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_LIMIT)

    @field_validator("property_candidate_manifest_id", "corpus_id", "dry_run_id", mode="before")
    @classmethod
    def _clean_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("created_by")
    @classmethod
    def _validate_created_by(cls, value: str) -> str:
        if not value:
            raise ValueError("created_by is required")
        if "@" in value and "redacted" not in value.lower():
            raise ValueError("created_by must be redacted and must not look like a private email address")
        return value

    @field_validator("source_manifest_sha256", mode="before")
    @classmethod
    def _clean_optional_manifest_sha(cls, value: Any) -> str:
        return _normalize_optional_sha256(value, field_name="source_manifest_sha256")

    @field_validator("source_dry_run_report_sha256", mode="before")
    @classmethod
    def _clean_required_dry_run_sha(cls, value: Any) -> str:
        return _normalize_required_sha256(value, field_name="source_dry_run_report_sha256")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError("schema_version must be custom_corpus_property_candidate.v1")
        return value

    @field_validator("records")
    @classmethod
    def _validate_records_present(cls, value: list[PropertyCandidateRecord]) -> list[PropertyCandidateRecord]:
        if not value:
            raise ValueError("records must be non-empty")
        duplicate_id = _first_duplicate([record.property_candidate_id for record in value])
        if duplicate_id:
            raise ValueError("duplicate property_candidate_id")
        duplicate_target = _first_duplicate([record.candidate_target() for record in value])
        if duplicate_target:
            raise ValueError("duplicate property candidate target")
        return value

    @model_validator(mode="after")
    def _validate_record_binding(self) -> "PropertyCandidateManifest":
        for record in self.records:
            if record.corpus_id != self.corpus_id:
                raise ValueError("record corpus_id must match manifest corpus_id")
            if record.dry_run_id != self.dry_run_id:
                raise ValueError("record dry_run_id must match manifest dry_run_id")
        return self


def load_property_candidate_manifest(path: str | Path) -> PropertyCandidateManifest:
    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyCandidateError(f"could not read property candidate manifest: {exc.__class__.__name__}") from exc
    return validate_property_candidate_manifest(payload)


def validate_property_candidate_manifest(value: Any) -> PropertyCandidateManifest:
    try:
        return PropertyCandidateManifest.model_validate(value)
    except CustomCorpusPropertyCandidateError:
        raise
    except Exception as exc:
        raise CustomCorpusPropertyCandidateError(_safe_error_message(str(exc))) from exc


def property_candidate_manifest_summary(
    manifest: PropertyCandidateManifest,
    path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = ""
    manifest_sha = ""
    if path is not None:
        manifest_path = Path(path).name or "property_candidates.json"
        try:
            manifest_sha = sha256_file(path)
        except Exception:
            manifest_sha = ""
    decisions = [record.trainability_decision for record in manifest.records]
    return {
        "schema_version": manifest.schema_version,
        "property_candidate_manifest_path": manifest_path,
        "property_candidate_manifest_sha256": manifest_sha,
        "property_candidate_manifest_id": manifest.property_candidate_manifest_id,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": manifest.dry_run_id,
        "candidate_policy": manifest.candidate_policy,
        "extraction_scope": manifest.extraction_scope,
        "record_count": len(manifest.records),
        "candidate_count": decisions.count("candidate"),
        "needs_review_count": decisions.count("needs_review"),
        "rejected_count": decisions.count("reject"),
        "review_required_count": sum(1 for record in manifest.records if record.review_required),
        "unique_document_count": len({record.document_id for record in manifest.records}),
        "unique_entity_count": len({record.entity_id for record in manifest.records if record.entity_id}),
        "unique_field_count": len({record.field_name for record in manifest.records}),
        "value_kind_counts": dict(sorted(Counter(record.value_kind for record in manifest.records).items())),
        "property_family_counts": dict(sorted(Counter(record.property_family for record in manifest.records).items())),
        "unit_status_counts": dict(sorted(Counter(record.unit_status for record in manifest.records).items())),
        "extraction_source_counts": dict(sorted(Counter(record.extraction_source for record in manifest.records).items())),
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_dry_run_report_sha256": manifest.source_dry_run_report_sha256,
        "warnings": [],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        manifest = load_property_candidate_manifest(args.property_candidates)
        summary = property_candidate_manifest_summary(manifest, path=args.property_candidates)
        if args.output_summary:
            write_json(Path(args.output_summary).expanduser(), summary)
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 0
    except CustomCorpusPropertyCandidateError as exc:
        err.write(f"property candidate manifest invalid: {exc}\n")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_candidate",
        description="Validate a custom corpus property candidate manifest offline.",
    )
    parser.add_argument("--property-candidates", required=True)
    parser.add_argument("--output-summary", default="")
    return parser


def _clean_text_value(value: Any, *, field_name: str, max_length: int) -> str:
    clean = str(value or "").strip()
    lowered = clean.lower()
    if len(clean) > max_length:
        raise ValueError(f"{field_name} is too long")
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise ValueError(f"{field_name} contains forbidden credential-like value")
    if any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        raise ValueError(f"{field_name} contains forbidden private path-like value")
    if _contains_url_query_or_signature(clean):
        raise ValueError(f"{field_name} contains forbidden URL value")
    return clean


def _normalize_required_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(2).lower()}"


def _normalize_optional_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    return _normalize_required_sha256(clean, field_name=field_name)


def _optional_finite_float(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    return _required_finite_float(value, field_name=field_name)


def _required_finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _contains_url_query_or_signature(value: str) -> bool:
    lowered = value.lower()
    if ("http://" in lowered or "https://" in lowered) and "?" in value:
        return True
    return any(marker in lowered for marker in ("x-amz-signature", "signature=", "sig=", "signedurl", "signed-url"))


def _first_duplicate(values: list[Any]) -> Any:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _safe_error_message(message: str) -> str:
    lowered = _relevant_error_text(str(message or "")).lower()
    if "duplicate property_candidate_id" in lowered:
        return "duplicate property_candidate_id"
    if "duplicate property candidate target" in lowered:
        return "duplicate property candidate target"
    if "require value_normalized" in lowered:
        return "value_normalized is invalid"
    if "require value_tuple" in lowered:
        return "value_tuple is invalid"
    if "numeric_scalar" in lowered:
        return "numeric_scalar value shape is invalid"
    if "numeric_range" in lowered:
        return "numeric_range value shape is invalid"
    if "numeric_tuple" in lowered:
        return "numeric_tuple value shape is invalid"
    if "unit_status=missing" in lowered:
        return "unit_status is invalid"
    if "needs_review" in lowered:
        return "needs_review decision is invalid"
    for field in (
        "schema_version",
        "property_candidate_manifest_id",
        "property_candidate_id",
        "corpus_id",
        "dry_run_id",
        "document_id",
        "source_record_id",
        "source_artifact_sha256",
        "parsed_document_sha256",
        "table_id",
        "row_id",
        "column_name",
        "raw_property_label",
        "canonical_property_guess",
        "field_name",
        "value_kind",
        "value_raw_summary",
        "value_normalized",
        "value_min",
        "value_max",
        "value_tuple",
        "unit_status",
        "unit_normalized",
        "entity_id",
        "entity_type",
        "provenance_summary",
        "confidence",
        "decision_reason",
        "rejection_reason",
        "created_by",
        "source_manifest_sha256",
        "source_dry_run_report_sha256",
    ):
        if field.lower() in lowered:
            if "credential-like" in lowered:
                return f"{field} contains forbidden credential-like value"
            if "private path-like" in lowered:
                return f"{field} contains forbidden private path-like value"
            if "url" in lowered:
                return f"{field} contains forbidden URL value"
            if "finite" in lowered:
                return f"{field} must be finite"
            return f"{field} is invalid"
    if "credential-like" in lowered or any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return "property candidate manifest contains forbidden credential-like value"
    if "private path-like" in lowered or any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        return "property candidate manifest contains forbidden private path-like value"
    if "url" in lowered:
        return "property candidate manifest contains forbidden URL value"
    if "too long" in lowered:
        return "property candidate manifest contains overlong text"
    return "property candidate manifest is invalid"


def _relevant_error_text(message: str) -> str:
    lowered = message.lower()
    marker = "value error,"
    if marker in lowered:
        start = lowered.index(marker) + len(marker)
        end = lowered.find("[type=", start)
        if end == -1:
            end = len(message)
        return message[start:end].strip()
    return message


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
