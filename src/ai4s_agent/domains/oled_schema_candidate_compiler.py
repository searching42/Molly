from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledDeviceLayer,
    OledEvidenceSource,
    OledEvidenceType,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    validate_oled_schema_candidates,
)


_MISSING_CONDITION_VALUE_MARKERS = frozenset(
    {
        "",
        "-",
        "--",
        "–",
        "—",
        "na",
        "n/a",
        "n.a.",
        "none",
        "null",
        "not available",
        "not reported",
    }
)


class OledSchemaCompilationStatus(str, Enum):
    COMPILED = "compiled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class OledSchemaCompilationGroupKey(BaseModel):
    source_paper_id: str
    source_candidate_hashes: list[str] = Field(default_factory=list)
    row_index: int | None = None
    device_label: str | None = None
    system_label: str | None = None
    target_property_ids: list[str] = Field(default_factory=list)

    @field_validator("source_candidate_hashes", "target_property_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})

    @property
    def key_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class OledCompiledLayeredRecordCandidate(BaseModel):
    record_id: str
    status: OledSchemaCompilationStatus
    group_key: OledSchemaCompilationGroupKey

    layered_record: OledLayeredRecord | None = None

    source_schema_candidate_ids: list[str] = Field(default_factory=list)
    source_candidate_hashes: list[str] = Field(default_factory=list)
    source_evidence_anchors: list[str] = Field(default_factory=list)

    schema_error_codes: list[str] = Field(default_factory=list)
    schema_warning_codes: list[str] = Field(default_factory=list)

    confidence_score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_schema_candidate_ids",
        "source_candidate_hashes",
        "source_evidence_anchors",
        "schema_error_codes",
        "schema_warning_codes",
        "reason_codes",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledSchemaCompilationFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    candidate_id: str | None = None
    record_id: str | None = None
    source_candidate_hash: str | None = None
    source_evidence_anchor: str | None = None


class OledSchemaCompilationReport(BaseModel):
    source_schema_candidate_count: int
    compiled_records: list[OledCompiledLayeredRecordCandidate] = Field(default_factory=list)
    findings: list[OledSchemaCompilationFinding] = Field(default_factory=list)
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
    def compiled_count(self) -> int:
        return sum(1 for record in self.compiled_records if record.status == OledSchemaCompilationStatus.COMPILED)

    @property
    def partial_count(self) -> int:
        return sum(1 for record in self.compiled_records if record.status == OledSchemaCompilationStatus.PARTIAL)

    @property
    def rejected_count(self) -> int:
        return sum(1 for record in self.compiled_records if record.status == OledSchemaCompilationStatus.REJECTED)


def compile_oled_schema_candidates_to_layered_records(
    schema_candidates: Iterable[OledSchemaCandidate],
    *,
    require_measurement_condition: bool = True,
    require_device_context_for_measurement: bool = True,
) -> OledSchemaCompilationReport:
    candidates = list(schema_candidates)
    validation_report = validate_oled_schema_candidates(candidates)
    findings = [_semantic_finding_to_compilation_finding(finding) for finding in validation_report.findings]
    validation_error_codes_by_id: dict[str, list[str]] = defaultdict(list)
    for finding in validation_report.findings:
        if finding.severity == "error" and finding.candidate_id:
            validation_error_codes_by_id[finding.candidate_id].append(finding.code)

    compiled_records: list[OledCompiledLayeredRecordCandidate] = []
    for group_key, group_candidates in group_oled_schema_candidates_for_compilation(candidates):
        group_error_codes = sorted(
            {
                error_code
                for candidate in group_candidates
                for error_code in validation_error_codes_by_id.get(candidate.candidate_id, [])
            }
        )
        if group_error_codes:
            compiled_records.append(_rejected_record(group_key, group_candidates, group_error_codes))
            continue
        compiled_record, group_findings = _compile_group(
            group_key,
            group_candidates,
            require_measurement_condition=require_measurement_condition,
            require_device_context_for_measurement=require_device_context_for_measurement,
        )
        compiled_records.append(compiled_record)
        findings.extend(group_findings)

    return OledSchemaCompilationReport(
        source_schema_candidate_count=len(candidates),
        compiled_records=compiled_records,
        findings=findings,
        metadata={
            "compiler_scope": "proposed_layered_record_candidates_only",
            "gold_records_created": False,
            "curated_dataset_written": False,
            "llm_called": False,
            "mineru_called": False,
        },
    )


def group_oled_schema_candidates_for_compilation(
    schema_candidates: Iterable[OledSchemaCandidate],
) -> list[tuple[OledSchemaCompilationGroupKey, list[OledSchemaCandidate]]]:
    grouped: dict[tuple[str, str, int | None], list[OledSchemaCandidate]] = {}
    for candidate in schema_candidates:
        row_index = _candidate_row_index(candidate)
        key = (
            candidate.source_paper_id,
            candidate.source_candidate_hash,
            row_index,
        )
        grouped.setdefault(key, []).append(candidate)

    output: list[tuple[OledSchemaCompilationGroupKey, list[OledSchemaCandidate]]] = []
    for key, candidates in grouped.items():
        source_paper_id, source_candidate_hash, row_index = key
        target_property_ids = [candidate.property_id for candidate in candidates if candidate.property_id]
        group_key = OledSchemaCompilationGroupKey(
            source_paper_id=source_paper_id,
            source_candidate_hashes=[source_candidate_hash],
            row_index=row_index,
            device_label=_first_device_label(candidates),
            system_label=_first_system_label(candidates),
            target_property_ids=target_property_ids,
        )
        output.append((group_key, sorted(candidates, key=lambda candidate: candidate.candidate_id)))
    return sorted(
        output,
        key=lambda item: (
            item[0].source_paper_id,
            item[0].source_candidate_hashes,
            -1 if item[0].row_index is None else item[0].row_index,
            item[0].device_label or "",
            item[0].system_label or "",
        ),
    )


def validate_compiled_oled_layered_record_candidates(
    compiled_records: Iterable[OledCompiledLayeredRecordCandidate],
) -> OledSchemaCompilationReport:
    records = list(compiled_records)
    findings: list[OledSchemaCompilationFinding] = []
    refreshed_records: list[OledCompiledLayeredRecordCandidate] = []
    for record_candidate in records:
        if record_candidate.layered_record is None:
            findings.append(
                OledSchemaCompilationFinding(
                    code="missing_layered_record",
                    severity="error",
                    message="compiled record candidate has no layered_record",
                    record_id=record_candidate.record_id,
                )
            )
            refreshed_records.append(record_candidate)
            continue
        schema_report = record_candidate.layered_record.validate_schema()
        schema_error_codes = schema_report.error_codes
        schema_warning_codes = schema_report.warning_codes
        for schema_finding in schema_report.findings:
            findings.append(
                OledSchemaCompilationFinding(
                    code=schema_finding.code,
                    severity="error" if schema_finding.severity == "error" else "warning",
                    message=schema_finding.message,
                    record_id=record_candidate.record_id,
                )
            )
        refreshed_records.append(
            record_candidate.model_copy(
                update={
                    "schema_error_codes": schema_error_codes,
                    "schema_warning_codes": schema_warning_codes,
                    "metadata": {
                        **record_candidate.metadata,
                        "schema_validation_ran": True,
                    },
                }
            )
        )
    return OledSchemaCompilationReport(
        source_schema_candidate_count=sum(len(record.source_schema_candidate_ids) for record in records),
        compiled_records=refreshed_records,
        findings=findings,
        metadata={"validation_scope": "compiled_layered_record_candidates"},
    )


def _compile_group(
    group_key: OledSchemaCompilationGroupKey,
    candidates: list[OledSchemaCandidate],
    *,
    require_measurement_condition: bool,
    require_device_context_for_measurement: bool,
) -> tuple[OledCompiledLayeredRecordCandidate, list[OledSchemaCompilationFinding]]:
    molecule: OledMolecularLayer | None = None
    interaction: OledInteractionLayer | None = None
    device: OledDeviceLayer | None = None
    measurement: OledMeasurementLayer | None = None
    findings: list[OledSchemaCompilationFinding] = []
    reason_codes: set[str] = {"compiled_from_schema_candidates"}

    row_condition = _measurement_condition_from_candidates(candidates)
    group_metadata = _group_metadata(group_key, candidates)

    for candidate in candidates:
        if candidate.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE:
            interaction = interaction or OledInteractionLayer()
            _apply_material_role(interaction, candidate)
            reason_codes.add("material_role_compiled")
            continue
        if candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION:
            observation = _property_observation_from_candidate(candidate, row_condition)
            if candidate.target_layer == OledCausalLayer.MOLECULE:
                molecule = molecule or OledMolecularLayer()
                molecule.properties.append(observation)
            elif candidate.target_layer == OledCausalLayer.INTERACTION:
                interaction = interaction or OledInteractionLayer()
                interaction.properties.append(observation)
            elif candidate.target_layer == OledCausalLayer.DEVICE:
                device = device or OledDeviceLayer()
                device.properties.append(observation)
            elif candidate.target_layer == OledCausalLayer.MEASUREMENT:
                measurement = measurement or OledMeasurementLayer()
                measurement.measurements.append(observation)
            else:
                findings.append(_candidate_finding(candidate, "missing_target_layer", "property target layer is missing"))
            reason_codes.add("property_observation_compiled")
            continue
        if candidate.candidate_type == OledSchemaCandidateType.MEASUREMENT_CONDITION:
            measurement = measurement or OledMeasurementLayer()
            measurement.metadata.setdefault("condition_candidates", []).append(_condition_metadata(candidate))
            reason_codes.add("measurement_condition_compiled")
            continue
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE:
            device = device or OledDeviceLayer()
            device.device_stack = list(candidate.device_stack)
            device.metadata.update(_device_structure_metadata(candidate))
            reason_codes.add("device_structure_compiled")
            continue
        findings.append(_candidate_finding(candidate, "unsupported_candidate_type", "schema candidate type needs review"))
        reason_codes.add("unsupported_candidate_type")

    device_label = _first_metadata_value(candidates, "device_label")
    if device_label is not None:
        device = device or OledDeviceLayer()
        device.metadata["device_label"] = device_label
    system_label = _first_metadata_value(candidates, "system_label")
    if system_label is not None:
        for layer in (molecule, interaction, device, measurement):
            if layer is not None:
                layer.metadata["system_label"] = system_label

    if molecule is not None:
        molecule.metadata.update(group_metadata)
    if interaction is not None:
        interaction.metadata.update(group_metadata)
    if device is not None:
        device.metadata.update(group_metadata)
    if measurement is not None:
        measurement.metadata.update(group_metadata)

    layered_record = OledLayeredRecord(
        molecule=molecule,
        interaction=interaction,
        device=device,
        measurement=measurement,
    )
    if not _has_usable_layered_content(layered_record):
        return _rejected_record(group_key, candidates, ["no_usable_layered_content"]), findings

    if require_measurement_condition and measurement is not None:
        for observation in measurement.measurements:
            if observation.condition is None:
                findings.append(
                    OledSchemaCompilationFinding(
                        code="measurement_condition_missing",
                        severity="warning",
                        message="measurement observation has no compiled measurement condition",
                        record_id=_record_id(group_key),
                    )
                )
                reason_codes.add("measurement_condition_missing")
    if require_device_context_for_measurement and measurement is not None and device is None:
        findings.append(
            OledSchemaCompilationFinding(
                code="measurement_without_device_context",
                severity="warning",
                message="measurement record candidate has no compiled device context",
                record_id=_record_id(group_key),
            )
        )
        reason_codes.add("measurement_without_device_context")

    schema_report = layered_record.validate_schema()
    status = _status_for_compiled_record(
        layered_record,
        schema_report.error_codes,
        schema_report.warning_codes,
        reason_codes,
    )
    record_candidate = OledCompiledLayeredRecordCandidate(
        record_id=_record_id(group_key),
        status=status,
        group_key=group_key,
        layered_record=layered_record,
        source_schema_candidate_ids=[candidate.candidate_id for candidate in candidates],
        source_candidate_hashes=[candidate.source_candidate_hash for candidate in candidates],
        source_evidence_anchors=[candidate.source_evidence_anchor for candidate in candidates],
        schema_error_codes=schema_report.error_codes,
        schema_warning_codes=schema_report.warning_codes,
        confidence_score=_mean_confidence(candidates),
        reason_codes=list(reason_codes),
        metadata={
            **group_metadata,
            "schema_validation_ran": True,
            "schema_observation_count": len(schema_report.observations),
            "compiled_record_candidate_only": True,
        },
    )
    return record_candidate, findings


def _property_observation_from_candidate(
    candidate: OledSchemaCandidate,
    row_condition: OledMeasurementCondition | None,
) -> OledPropertyObservation:
    target_layer = candidate.target_layer or OledCausalLayer.MEASUREMENT
    condition = row_condition if target_layer == OledCausalLayer.MEASUREMENT else None
    metadata = {
        **candidate.metadata,
        "source_schema_candidate_id": candidate.candidate_id,
        "source_property_id": candidate.property_id,
        "schema_candidate_reason_codes": list(candidate.reason_codes),
        "compiled_from_schema_candidate": True,
        "evidence_refs": [evidence_ref.model_dump(mode="json") for evidence_ref in candidate.evidence_refs],
    }
    if target_layer == OledCausalLayer.MEASUREMENT and condition is None:
        condition = _measurement_condition_from_property_metadata(candidate)
    return OledPropertyObservation(
        property_label=candidate.property_label or candidate.property_id or "unknown",
        value=candidate.value,
        unit=candidate.unit,
        condition=condition,
        evidence_sources=[_evidence_source(evidence_ref, target_layer, candidate) for evidence_ref in candidate.evidence_refs],
        confidence=_confidence(candidate),
        metadata=metadata,
    )


def _measurement_condition_from_candidates(candidates: list[OledSchemaCandidate]) -> OledMeasurementCondition | None:
    updates: dict[str, Any] = {}
    metadata: dict[str, Any] = {"raw_conditions": [], "units": {}}
    for candidate in candidates:
        if candidate.candidate_type != OledSchemaCandidateType.MEASUREMENT_CONDITION:
            continue
        field_name = _condition_model_field(candidate.condition_field)
        raw_condition = _condition_metadata(candidate)
        metadata["raw_conditions"].append(raw_condition)
        if field_name is None:
            metadata.setdefault("unsupported_conditions", []).append(raw_condition)
            continue
        if _is_explicit_missing_condition_value(candidate.condition_value):
            metadata.setdefault("missing_conditions", []).append(raw_condition)
            continue
        if not _is_numeric_condition_value(candidate.condition_value):
            metadata.setdefault("unparsed_conditions", []).append(raw_condition)
            continue
        updates[field_name] = candidate.condition_value
        if candidate.condition_unit:
            metadata["units"][field_name] = candidate.condition_unit
    if not updates and not metadata["raw_conditions"]:
        return None
    updates["metadata"] = {key: value for key, value in metadata.items() if value}
    return OledMeasurementCondition(**updates)


def _measurement_condition_from_property_metadata(candidate: OledSchemaCandidate) -> OledMeasurementCondition | None:
    field = _condition_model_field(str(candidate.metadata.get("condition_field") or ""))
    value = candidate.metadata.get("condition_value")
    unit = candidate.metadata.get("condition_unit")
    if field is None or value is None:
        return None
    raw_condition = {
        "condition_field": candidate.metadata.get("condition_field"),
        "condition_value": value,
        "condition_unit": unit,
        "source_schema_candidate_id": candidate.candidate_id,
    }
    metadata = {
        "raw_conditions": [raw_condition],
        "units": {field: unit} if unit else {},
    }
    updates: dict[str, Any] = {"metadata": metadata}
    if _is_explicit_missing_condition_value(value):
        metadata["missing_conditions"] = [raw_condition]
    elif not _is_numeric_condition_value(value):
        metadata["unparsed_conditions"] = [raw_condition]
    else:
        updates[field] = value
    return OledMeasurementCondition(**updates)


def _apply_material_role(interaction: OledInteractionLayer, candidate: OledSchemaCandidate) -> None:
    role = str(candidate.material_role or "").strip()
    name = str(candidate.material_name or "").strip()
    if not role or not name:
        return
    role_key = role.lower()
    interaction.metadata.setdefault("material_roles", {})[role_key] = name
    interaction.metadata.setdefault("material_role_candidates", []).append(
        {
            "role": role_key,
            "material_name": name,
            "source_schema_candidate_id": candidate.candidate_id,
            "evidence_refs": [evidence_ref.model_dump(mode="json") for evidence_ref in candidate.evidence_refs],
        }
    )
    if role_key == "host":
        interaction.metadata["host_name"] = name
    elif role_key in {"emitter", "emitter_dopant", "fluorescent_emitter"}:
        interaction.metadata["emitter_name"] = name
    elif role_key in {"assistant_dopant", "dopant", "tadf_assistant"}:
        interaction.metadata["assistant_dopant_name"] = name
        interaction.metadata[f"{role_key}_name"] = name
    else:
        interaction.metadata[f"{role_key}_name"] = name


def _evidence_source(
    evidence_ref: OledSchemaEvidenceRef,
    layer: OledCausalLayer,
    candidate: OledSchemaCandidate,
) -> OledEvidenceSource:
    source_type = _evidence_type(evidence_ref.source_candidate_type)
    return OledEvidenceSource(
        source_id=f"{evidence_ref.source_candidate_hash}:{evidence_ref.source_evidence_anchor}",
        source_type=source_type,
        layer=layer,
        locator=evidence_ref.source_evidence_anchor,
        metadata={
            "source_schema_candidate_id": candidate.candidate_id,
            "row_index": evidence_ref.row_index,
            "column_name": evidence_ref.column_name,
            "cell_value": evidence_ref.cell_value,
            "field_name": evidence_ref.field_name,
        },
    )


def _evidence_type(value: str) -> OledEvidenceType:
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return OledEvidenceType.TABLE
    if normalized == "figure":
        return OledEvidenceType.FIGURE
    if normalized == "text":
        return OledEvidenceType.TEXT
    return OledEvidenceType.TEXT


def _confidence(candidate: OledSchemaCandidate) -> OledConfidenceAssessment:
    score = candidate.confidence_score if candidate.confidence_score is not None else 0.5
    return OledConfidenceAssessment(
        score=score,
        rationale=["compiled from intermediate OledSchemaCandidate", *candidate.reason_codes],
    )


def _condition_model_field(condition_field: str | None) -> str | None:
    normalized = str(condition_field or "").strip().lower()
    return {
        "luminance": "luminance_cd_m2",
        "current_density": "current_density_ma_cm2",
        "voltage": "voltage_v",
        "turn_on_voltage": "voltage_v",
        "temperature": "temperature_k",
    }.get(normalized)


def _is_explicit_missing_condition_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _MISSING_CONDITION_VALUE_MARKERS


def _is_numeric_condition_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _condition_metadata(candidate: OledSchemaCandidate) -> dict[str, Any]:
    return {
        "condition_field": candidate.condition_field,
        "condition_value": candidate.condition_value,
        "condition_unit": candidate.condition_unit,
        "source_schema_candidate_id": candidate.candidate_id,
        "evidence_refs": [evidence_ref.model_dump(mode="json") for evidence_ref in candidate.evidence_refs],
    }


def _device_structure_metadata(candidate: OledSchemaCandidate) -> dict[str, Any]:
    metadata = {
        "source_schema_candidate_id": candidate.candidate_id,
        "raw_stack_preserved": True,
        "normalization_policy": "no_device_material_normalization",
        "evidence_refs": [evidence_ref.model_dump(mode="json") for evidence_ref in candidate.evidence_refs],
    }
    if "source_text" in candidate.metadata:
        metadata["source_text"] = candidate.metadata["source_text"]
    return metadata


def _status_for_compiled_record(
    layered_record: OledLayeredRecord,
    schema_error_codes: list[str],
    schema_warning_codes: list[str],
    reason_codes: set[str],
) -> OledSchemaCompilationStatus:
    if not _has_usable_layered_content(layered_record):
        return OledSchemaCompilationStatus.REJECTED
    if "measurement_without_device_context" in reason_codes or "measurement_condition_missing" in reason_codes:
        return OledSchemaCompilationStatus.NEEDS_REVIEW
    if schema_error_codes:
        return OledSchemaCompilationStatus.PARTIAL
    if schema_warning_codes or _lacks_material_identifiers(layered_record):
        return OledSchemaCompilationStatus.PARTIAL
    return OledSchemaCompilationStatus.COMPILED


def _lacks_material_identifiers(layered_record: OledLayeredRecord) -> bool:
    if layered_record.molecule is not None and layered_record.molecule.canonical_smiles is None:
        return True
    if layered_record.interaction is not None:
        return layered_record.interaction.host_smiles is None or layered_record.interaction.emitter_smiles is None
    return False


def _has_usable_layered_content(layered_record: OledLayeredRecord) -> bool:
    return any(
        [
            layered_record.molecule is not None
            and (layered_record.molecule.properties or layered_record.molecule.canonical_smiles),
            layered_record.interaction is not None
            and (
                layered_record.interaction.properties
                or layered_record.interaction.metadata
                or layered_record.interaction.host_smiles
                or layered_record.interaction.emitter_smiles
            ),
            layered_record.device is not None and (layered_record.device.device_stack or layered_record.device.properties or layered_record.device.metadata),
            layered_record.measurement is not None and (layered_record.measurement.measurements or layered_record.measurement.metadata),
        ]
    )


def _group_metadata(
    group_key: OledSchemaCompilationGroupKey,
    candidates: list[OledSchemaCandidate],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_paper_id": group_key.source_paper_id,
        "source_candidate_hashes": list(group_key.source_candidate_hashes),
        "source_evidence_anchors": _source_evidence_anchors(candidates),
        "source_schema_candidate_ids": sorted(candidate.candidate_id for candidate in candidates),
    }
    if group_key.row_index is not None:
        metadata["row_index"] = group_key.row_index
    if group_key.device_label:
        metadata["device_label"] = group_key.device_label
    if group_key.system_label:
        metadata["system_label"] = group_key.system_label
    caption = _first_metadata_value(candidates, "source_caption")
    if caption is not None:
        metadata["source_caption"] = caption
    return metadata


def _rejected_record(
    group_key: OledSchemaCompilationGroupKey,
    candidates: list[OledSchemaCandidate],
    error_codes: list[str],
) -> OledCompiledLayeredRecordCandidate:
    return OledCompiledLayeredRecordCandidate(
        record_id=_record_id(group_key),
        status=OledSchemaCompilationStatus.REJECTED,
        group_key=group_key,
        layered_record=None,
        source_schema_candidate_ids=[candidate.candidate_id for candidate in candidates],
        source_candidate_hashes=[candidate.source_candidate_hash for candidate in candidates],
        source_evidence_anchors=[candidate.source_evidence_anchor for candidate in candidates],
        schema_error_codes=error_codes,
        reason_codes=["rejected_schema_candidate_group"],
        metadata={"schema_validation_ran": False},
    )


def _semantic_finding_to_compilation_finding(finding: Any) -> OledSchemaCompilationFinding:
    return OledSchemaCompilationFinding(
        code=finding.code,
        severity="error" if finding.severity == "error" else "warning",
        message=finding.message,
        candidate_id=finding.candidate_id,
        source_candidate_hash=finding.source_candidate_hash,
        source_evidence_anchor=finding.source_evidence_anchor,
    )


def _candidate_finding(
    candidate: OledSchemaCandidate,
    code: str,
    message: str,
    *,
    severity: Literal["error", "warning"] = "warning",
) -> OledSchemaCompilationFinding:
    return OledSchemaCompilationFinding(
        code=code,
        severity=severity,
        message=message,
        candidate_id=candidate.candidate_id,
        source_candidate_hash=candidate.source_candidate_hash,
        source_evidence_anchor=candidate.source_evidence_anchor,
    )


def _candidate_row_index(candidate: OledSchemaCandidate) -> int | None:
    row_indexes = [evidence_ref.row_index for evidence_ref in candidate.evidence_refs if evidence_ref.row_index is not None]
    return min(row_indexes) if row_indexes else None


def _first_device_label(candidates: list[OledSchemaCandidate]) -> str | None:
    for candidate in candidates:
        value = candidate.metadata.get("device_label")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_system_label(candidates: list[OledSchemaCandidate]) -> str | None:
    for candidate in candidates:
        value = candidate.metadata.get("system_label") or candidate.metadata.get("el_colour") or candidate.metadata.get("color")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_metadata_value(candidates: list[OledSchemaCandidate], key: str) -> Any:
    for candidate in candidates:
        value = candidate.metadata.get(key)
        if value is not None:
            return value
    return None


def _source_evidence_anchors(candidates: list[OledSchemaCandidate]) -> list[str]:
    anchors = {candidate.source_evidence_anchor for candidate in candidates if candidate.source_evidence_anchor}
    for candidate in candidates:
        anchors.update(ref.source_evidence_anchor for ref in candidate.evidence_refs if ref.source_evidence_anchor)
    return sorted(anchors)


def _mean_confidence(candidates: list[OledSchemaCandidate]) -> float | None:
    scores = [candidate.confidence_score for candidate in candidates if candidate.confidence_score is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


def _record_id(group_key: OledSchemaCompilationGroupKey) -> str:
    return f"compiled-oled:{group_key.key_hash[:16]}"


__all__ = [
    "OledSchemaCompilationStatus",
    "OledSchemaCompilationGroupKey",
    "OledCompiledLayeredRecordCandidate",
    "OledSchemaCompilationFinding",
    "OledSchemaCompilationReport",
    "compile_oled_schema_candidates_to_layered_records",
    "group_oled_schema_candidates_for_compilation",
    "validate_compiled_oled_layered_record_candidates",
]
