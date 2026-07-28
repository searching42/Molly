from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import OledLayeredRecord, OledLayeredSchemaFinding


class OledGoldDatasetRecord(BaseModel):
    record_id: str
    layered_record: OledLayeredRecord
    evidence_refs: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("record_id is required")
        return clean

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        return [clean for item in value if (clean := str(item or "").strip())]


class OledGoldValidationFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str
    record_id: str
    layer: OledCausalLayer | None = None
    property_id: str | None = None
    property_label: str | None = None


class OledGoldValidationReport(BaseModel):
    records: list[OledGoldDatasetRecord] = Field(default_factory=list)
    findings: list[OledGoldValidationFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]

    @property
    def valid_record_ids(self) -> list[str]:
        invalid_record_ids = {
            finding.record_id for finding in self.findings if finding.severity == "error"
        }
        return [record.record_id for record in self.records if record.record_id not in invalid_record_ids]


def validate_oled_gold_dataset(records: Iterable[OledGoldDatasetRecord]) -> OledGoldValidationReport:
    gold_records = list(records)
    record_id_counts = Counter(record.record_id for record in gold_records)
    findings: list[OledGoldValidationFinding] = []

    for record in gold_records:
        if record_id_counts[record.record_id] > 1:
            findings.append(
                OledGoldValidationFinding(
                    code="duplicate_gold_record_id",
                    message=f"gold record id `{record.record_id}` appears more than once",
                    record_id=record.record_id,
                )
            )
        if not record.evidence_refs:
            findings.append(
                OledGoldValidationFinding(
                    code="gold_missing_evidence_refs",
                    message=f"gold record `{record.record_id}` has no top-level evidence refs",
                    record_id=record.record_id,
                )
            )

        schema_report = record.layered_record.validate_schema()
        for schema_finding in schema_report.findings:
            findings.append(_gold_finding_from_schema(record.record_id, schema_finding))

    return OledGoldValidationReport(records=gold_records, findings=findings)


def _gold_finding_from_schema(
    record_id: str,
    schema_finding: OledLayeredSchemaFinding,
) -> OledGoldValidationFinding:
    if schema_finding.code in _GOLD_HARD_GATE_SCHEMA_WARNINGS:
        return OledGoldValidationFinding(
            code=_GOLD_HARD_GATE_SCHEMA_WARNINGS[schema_finding.code],
            severity="error",
            message=f"gold record `{record_id}` fails hard gate: {schema_finding.message}",
            record_id=record_id,
            layer=schema_finding.layer,
            property_id=schema_finding.property_id,
            property_label=schema_finding.property_label,
        )
    return OledGoldValidationFinding(
        code=schema_finding.code,
        severity=_schema_severity(schema_finding),
        message=schema_finding.message,
        record_id=record_id,
        layer=schema_finding.layer,
        property_id=schema_finding.property_id,
        property_label=schema_finding.property_label,
    )


def _schema_severity(schema_finding: OledLayeredSchemaFinding) -> Literal["error", "warning"]:
    return "warning" if schema_finding.severity == "warning" else "error"


_GOLD_HARD_GATE_SCHEMA_WARNINGS = {
    "missing_provenance": "gold_missing_provenance",
    "missing_confidence": "gold_missing_confidence",
    "missing_confounder_tags": "gold_missing_confounder_tags",
}


__all__ = [
    "OledGoldDatasetRecord",
    "OledGoldValidationFinding",
    "OledGoldValidationReport",
    "validate_oled_gold_dataset",
]
