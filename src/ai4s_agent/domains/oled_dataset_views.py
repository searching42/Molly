from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_condition_dedup import (
    OledConditionDedupObservation,
    build_oled_condition_dedup_observations,
    detect_oled_condition_dedup_conflicts,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_feature_materialization import materialize_oled_baseline_feature_table
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset
from ai4s_agent.domains.oled_layered_schema import OledConfounderType, OledLayeredCanonicalObservation


class OledDatasetViewKind(str, Enum):
    RAW_ALL_MEASUREMENTS = "raw_all_measurements"
    CURATED_DEVICE_BASELINE = "curated_device_baseline"
    BEST_REPORTED = "best_reported"
    CURATED_INTRINSIC = "curated_intrinsic"


class OledDatasetViewRow(BaseModel):
    view_kind: OledDatasetViewKind
    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)
    target_property_id: str
    target_value: float | int | str | None = None
    target_unit: str | None = None
    target_layer: OledCausalLayer
    condition_hash: str | None = None
    dedup_key_hash: str | None = None
    feature_view: OledBaselineFeatureView | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids")
    @classmethod
    def validate_source_record_ids(cls, value: list[str]) -> list[str]:
        return sorted({clean for item in value if (clean := str(item or "").strip())})

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        return sorted({clean for item in value if (clean := str(item or "").strip())})


class OledDatasetViewFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str
    view_kind: OledDatasetViewKind
    record_ids: list[str] = Field(default_factory=list)
    dedup_key_hash: str | None = None
    target_property_id: str | None = None


class OledDatasetViewReport(BaseModel):
    view_kind: OledDatasetViewKind
    target_property_id: str
    rows: list[OledDatasetViewRow] = Field(default_factory=list)
    findings: list[OledDatasetViewFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def record_ids(self) -> list[str]:
        return sorted({row.record_id for row in self.rows})


def build_oled_dataset_view(
    records: Iterable[OledGoldDatasetRecord],
    *,
    view_kind: OledDatasetViewKind | str,
    target_property_id: str = "eqe_percent",
) -> OledDatasetViewReport:
    gold_records = list(records)
    kind = _coerce_view_kind(view_kind)
    clean_target = str(target_property_id or "").strip()
    if not clean_target:
        raise ValueError("target_property_id is required")

    gold_report = validate_oled_gold_dataset(gold_records)
    if not gold_report.is_valid:
        raise ValueError(f"invalid_gold_records:{','.join(gold_report.error_codes)}")

    if kind == OledDatasetViewKind.RAW_ALL_MEASUREMENTS:
        return OledDatasetViewReport(
            view_kind=kind,
            target_property_id=clean_target,
            rows=_device_level_rows(gold_records, kind, clean_target),
            findings=_dedup_conflict_findings_as_view_findings(
                gold_records,
                kind,
                clean_target,
                severity="warning",
            ),
            metadata={"policy": "no_filtering"},
        )
    if kind == OledDatasetViewKind.CURATED_DEVICE_BASELINE:
        return _build_curated_device_baseline_view(gold_records, kind, clean_target)
    if kind == OledDatasetViewKind.BEST_REPORTED:
        return _build_best_reported_view(gold_records, kind, clean_target)
    return _build_curated_intrinsic_view(gold_records, kind, clean_target)


def _build_curated_device_baseline_view(
    records: list[OledGoldDatasetRecord],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> OledDatasetViewReport:
    kept_records: list[OledGoldDatasetRecord] = []
    findings: list[OledDatasetViewFinding] = []
    for record in records:
        flags = record.layered_record.confounder_flags
        if flags.is_outcoupling_modified or _record_has_confounder_type(record, OledConfounderType.OUTCOUPLING_STRUCTURE):
            findings.append(
                _finding(
                    "excluded_outcoupling_modified",
                    "warning",
                    "outcoupling-modified records are excluded from curated device baseline views",
                    view_kind,
                    [record.record_id],
                    target_property_id=target_property_id,
                )
            )
            continue
        if flags.is_best_reported or _record_has_confounder_type(record, OledConfounderType.BEST_REPORTED):
            findings.append(
                _finding(
                    "excluded_best_reported",
                    "warning",
                    "best-reported records are excluded from curated device baseline views",
                    view_kind,
                    [record.record_id],
                    target_property_id=target_property_id,
                )
            )
            continue
        kept_records.append(record)

    if not kept_records:
        return OledDatasetViewReport(
            view_kind=view_kind,
            target_property_id=target_property_id,
            findings=findings,
            metadata={"policy": "exclude_outcoupling_and_best_reported"},
        )

    findings.extend(
        _dedup_conflict_findings_as_view_findings(
            kept_records,
            view_kind,
            target_property_id,
            severity="error",
        )
    )
    conflict_key_hashes = {finding.dedup_key_hash for finding in findings if finding.code == "dedup_value_conflict"}

    rows = _collapse_device_rows_by_dedup_key(
        _device_level_rows(kept_records, view_kind, target_property_id),
        conflict_key_hashes=conflict_key_hashes,
        duplicate_policy="collapsed_consistent_duplicate",
        single_policy="single_observation",
    )
    return OledDatasetViewReport(
        view_kind=view_kind,
        target_property_id=target_property_id,
        rows=rows,
        findings=findings,
        metadata={"policy": "exclude_outcoupling_best_reported_and_conflicting_duplicates"},
    )


def _build_best_reported_view(
    records: list[OledGoldDatasetRecord],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> OledDatasetViewReport:
    best_records = [
        record
        for record in records
        if record.layered_record.confounder_flags.is_best_reported
        or _record_has_confounder_type(record, OledConfounderType.BEST_REPORTED)
    ]
    if not best_records:
        return OledDatasetViewReport(
            view_kind=view_kind,
            target_property_id=target_property_id,
            findings=[
                _finding(
                    "no_best_reported_rows",
                    "warning",
                    "best-reported view requires explicit best-reported flags or tags; none were found",
                    view_kind,
                    [record.record_id for record in records],
                    target_property_id=target_property_id,
                )
            ],
            metadata={"biased_dataset": True, "policy": "explicit_best_reported_only"},
        )

    findings = _dedup_conflict_findings_as_view_findings(
        best_records,
        view_kind,
        target_property_id,
        severity="error",
    )
    findings.append(
        _finding(
            "best_reported_view_is_biased",
            "warning",
            "best-reported views intentionally expose explicitly flagged best-reported observations as biased datasets",
            view_kind,
            [record.record_id for record in best_records],
            target_property_id=target_property_id,
        )
    )
    conflict_key_hashes = {finding.dedup_key_hash for finding in findings if finding.code == "dedup_value_conflict"}
    selected_rows = _collapse_device_rows_by_dedup_key(
        _device_level_rows(best_records, view_kind, target_property_id),
        conflict_key_hashes=conflict_key_hashes,
        duplicate_policy="collapsed_consistent_best_reported_duplicate",
        single_policy="best_reported_observation",
    )
    selected_rows = [
        row.model_copy(update={"metadata": {**row.metadata, "biased_dataset": True}})
        for row in selected_rows
    ]
    return OledDatasetViewReport(
        view_kind=view_kind,
        target_property_id=target_property_id,
        rows=selected_rows,
        findings=findings,
        metadata={"biased_dataset": True, "policy": "maximum_numeric_target"},
    )


def _build_curated_intrinsic_view(
    records: list[OledGoldDatasetRecord],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> OledDatasetViewReport:
    rows: list[OledDatasetViewRow] = []
    for record in records:
        schema_report = record.layered_record.validate_schema()
        for observation in schema_report.observations:
            if observation.layer != OledCausalLayer.MOLECULE or observation.property_id != target_property_id:
                continue
            rows.append(_intrinsic_row(record, observation, view_kind, target_property_id))
    findings: list[OledDatasetViewFinding] = []
    if not rows:
        findings.append(
            _finding(
                "no_intrinsic_target_rows",
                "error",
                f"no molecular-layer rows found for target `{target_property_id}`",
                view_kind,
                [record.record_id for record in records],
                target_property_id=target_property_id,
            )
        )
    rows, intrinsic_findings = _dedup_intrinsic_rows(rows, view_kind, target_property_id)
    findings.extend(intrinsic_findings)
    return OledDatasetViewReport(
        view_kind=view_kind,
        target_property_id=target_property_id,
        rows=rows,
        findings=findings,
        metadata={"policy": "molecular_layer_only"},
    )


def _device_level_rows(
    records: list[OledGoldDatasetRecord],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> list[OledDatasetViewRow]:
    table = materialize_oled_baseline_feature_table(
        records,
        feature_view=OledBaselineFeatureView.FULL_CONTEXT,
        target_property_id=target_property_id,
    )
    dedup_by_record = _dedup_observations_by_record(records, target_property_id)
    rows: list[OledDatasetViewRow] = []
    for feature_row in table.rows:
        dedup_observation = dedup_by_record[feature_row.record_id].pop(0)
        rows.append(
            OledDatasetViewRow(
                view_kind=view_kind,
                record_id=feature_row.record_id,
                source_record_ids=[feature_row.record_id],
                target_property_id=target_property_id,
                target_value=feature_row.target_value,
                target_unit=feature_row.target_unit,
                target_layer=OledCausalLayer.MEASUREMENT,
                condition_hash=feature_row.condition_hash,
                dedup_key_hash=dedup_observation.dedup_key.key_hash,
                feature_view=feature_row.feature_view,
                features=feature_row.features,
                evidence_refs=feature_row.evidence_refs,
                confidence_score=feature_row.confidence_score,
                metadata={},
            )
        )
    return rows


def _dedup_conflict_findings_as_view_findings(
    records: list[OledGoldDatasetRecord],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
    *,
    severity: Literal["error", "warning"],
) -> list[OledDatasetViewFinding]:
    if not records:
        return []
    dedup_report = detect_oled_condition_dedup_conflicts(records, target_property_id=target_property_id)
    return [
        _finding(
            code=finding.code,
            severity=severity,
            message=finding.message,
            view_kind=view_kind,
            record_ids=finding.record_ids,
            dedup_key_hash=finding.dedup_key_hash,
            target_property_id=finding.target_property_id,
        )
        for finding in dedup_report.findings
    ]


def _collapse_device_rows_by_dedup_key(
    rows: list[OledDatasetViewRow],
    *,
    conflict_key_hashes: set[str | None],
    duplicate_policy: str,
    single_policy: str,
) -> list[OledDatasetViewRow]:
    grouped_rows: dict[str, list[OledDatasetViewRow]] = defaultdict(list)
    for row in rows:
        if row.dedup_key_hash in conflict_key_hashes:
            continue
        grouped_rows[row.dedup_key_hash or row.record_id].append(row)

    collapsed_rows: list[OledDatasetViewRow] = []
    for dedup_key_hash, group in sorted(grouped_rows.items()):
        if len(group) == 1:
            row = group[0]
            collapsed_rows.append(row.model_copy(update={"metadata": {**row.metadata, "dedup_policy": single_policy}}))
            continue
        collapsed_rows.append(
            _collapse_consistent_duplicate_group(
                dedup_key_hash,
                group,
                duplicate_policy=duplicate_policy,
            )
        )
    return collapsed_rows


def _dedup_observations_by_record(
    records: list[OledGoldDatasetRecord],
    target_property_id: str,
) -> dict[str, list[OledConditionDedupObservation]]:
    grouped: dict[str, list[OledConditionDedupObservation]] = defaultdict(list)
    for observation in build_oled_condition_dedup_observations(records, target_property_id=target_property_id):
        grouped[observation.record_id].append(observation)
    return grouped


def _collapse_consistent_duplicate_group(
    dedup_key_hash: str,
    rows: list[OledDatasetViewRow],
    *,
    duplicate_policy: str,
) -> OledDatasetViewRow:
    sorted_rows = sorted(rows, key=lambda row: row.record_id)
    selected = sorted_rows[0]
    source_record_ids = sorted({record_id for row in sorted_rows for record_id in row.source_record_ids})
    evidence_refs = sorted({evidence_ref for row in sorted_rows for evidence_ref in row.evidence_refs})
    return selected.model_copy(
        update={
            "source_record_ids": source_record_ids,
            "evidence_refs": evidence_refs,
            "metadata": {
                **selected.metadata,
                "dedup_key_hash": dedup_key_hash,
                "dedup_policy": duplicate_policy,
            },
        }
    )


def _dedup_intrinsic_rows(
    rows: list[OledDatasetViewRow],
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> tuple[list[OledDatasetViewRow], list[OledDatasetViewFinding]]:
    grouped_rows: dict[str, list[OledDatasetViewRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[_intrinsic_dedup_key(row)].append(row)

    collapsed_rows: list[OledDatasetViewRow] = []
    findings: list[OledDatasetViewFinding] = []
    for key, group in sorted(grouped_rows.items()):
        if len(group) == 1:
            row = group[0]
            collapsed_rows.append(row.model_copy(update={"metadata": {**row.metadata, "dedup_policy": "single_observation"}}))
            continue
        if not _row_values_are_consistent(group):
            findings.append(
                _finding(
                    "intrinsic_value_conflict",
                    "error",
                    f"intrinsic dedup key `{key}` has conflicting values for `{target_property_id}`",
                    view_kind,
                    sorted({record_id for row in group for record_id in row.source_record_ids}),
                    target_property_id=target_property_id,
                )
            )
            continue
        collapsed_rows.append(_collapse_consistent_intrinsic_group(key, group))
    return collapsed_rows, findings


def _collapse_consistent_intrinsic_group(
    intrinsic_key: str,
    rows: list[OledDatasetViewRow],
) -> OledDatasetViewRow:
    sorted_rows = sorted(rows, key=lambda row: row.record_id)
    selected = sorted_rows[0]
    source_record_ids = sorted({record_id for row in sorted_rows for record_id in row.source_record_ids})
    evidence_refs = sorted({evidence_ref for row in sorted_rows for evidence_ref in row.evidence_refs})
    return selected.model_copy(
        update={
            "source_record_ids": source_record_ids,
            "evidence_refs": evidence_refs,
            "metadata": {
                **selected.metadata,
                "dedup_key": intrinsic_key,
                "dedup_policy": "collapsed_consistent_intrinsic_duplicate",
            },
        }
    )


def _intrinsic_row(
    record: OledGoldDatasetRecord,
    observation: OledLayeredCanonicalObservation,
    view_kind: OledDatasetViewKind,
    target_property_id: str,
) -> OledDatasetViewRow:
    molecule = record.layered_record.molecule
    evidence_refs = {source.source_id for source in observation.evidence_sources}
    evidence_refs.update(record.evidence_refs)
    return OledDatasetViewRow(
        view_kind=view_kind,
        record_id=record.record_id,
        source_record_ids=[record.record_id],
        target_property_id=target_property_id,
        target_value=observation.normalized_value,
        target_unit=observation.normalized_unit,
        target_layer=OledCausalLayer.MOLECULE,
        condition_hash=None,
        dedup_key_hash=None,
        feature_view=OledBaselineFeatureView.MOLECULE_ONLY,
        features={
            "molecule.canonical_smiles": molecule.canonical_smiles if molecule else None,
            "molecule.inchikey": molecule.inchikey if molecule else None,
        },
        evidence_refs=sorted(evidence_refs),
        confidence_score=observation.confidence.score if observation.confidence else None,
        metadata={"policy": "molecular_layer_only"},
    )


def _intrinsic_dedup_key(row: OledDatasetViewRow) -> str:
    inchikey = row.features.get("molecule.inchikey")
    canonical_smiles = row.features.get("molecule.canonical_smiles")
    if isinstance(inchikey, str) and inchikey.strip():
        identity = f"inchikey:{inchikey.strip().lower()}"
    elif isinstance(canonical_smiles, str) and canonical_smiles.strip():
        identity = f"canonical_smiles:{canonical_smiles.strip()}"
    else:
        identity = f"record:{row.record_id}"
    return f"{identity}|target:{row.target_property_id}"


def _row_values_are_consistent(rows: list[OledDatasetViewRow]) -> bool:
    if len(rows) <= 1:
        return True
    numeric_values = [_numeric_value(row.target_value) for row in rows]
    if all(value is not None for value in numeric_values):
        clean_values = [float(value) for value in numeric_values if value is not None]
        return max(clean_values) - min(clean_values) <= 0.0
    serialized_values = {repr(row.target_value) for row in rows}
    return len(serialized_values) <= 1


def _numeric_value(value: float | int | str | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_has_confounder_type(
    record: OledGoldDatasetRecord,
    confounder_type: OledConfounderType,
) -> bool:
    return any(tag.confounder_type == confounder_type for tag in record.layered_record.confounder_tags)


def _coerce_view_kind(view_kind: OledDatasetViewKind | str) -> OledDatasetViewKind:
    if isinstance(view_kind, OledDatasetViewKind):
        return view_kind
    return OledDatasetViewKind(str(view_kind or "").strip())


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    view_kind: OledDatasetViewKind,
    record_ids: list[str],
    *,
    dedup_key_hash: str | None = None,
    target_property_id: str | None = None,
) -> OledDatasetViewFinding:
    return OledDatasetViewFinding(
        code=code,
        severity=severity,
        message=message,
        view_kind=view_kind,
        record_ids=sorted({record_id for record_id in record_ids if record_id}),
        dedup_key_hash=dedup_key_hash,
        target_property_id=target_property_id,
    )


def _is_numeric(value: float | int | str | None) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "OledDatasetViewFinding",
    "OledDatasetViewKind",
    "OledDatasetViewReport",
    "OledDatasetViewRow",
    "build_oled_dataset_view",
]
