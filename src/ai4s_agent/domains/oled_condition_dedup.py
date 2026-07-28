from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset
from ai4s_agent.domains.oled_layered_schema import (
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredCanonicalObservation,
    OledMeasurementCondition,
    OledMolecularLayer,
)


class OledConditionAwareDedupKey(BaseModel):
    target_property_id: str
    components: dict[str, Any]

    @property
    def key_hash(self) -> str:
        return _stable_hash(
            {
                "components": self.components,
                "target_property_id": self.target_property_id,
            }
        )


class OledConditionDedupObservation(BaseModel):
    record_id: str
    target_property_id: str
    dedup_key: OledConditionAwareDedupKey
    normalized_value: float | int | str | None = None
    normalized_unit: str | None = None


class OledConditionDedupFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str
    target_property_id: str
    dedup_key_hash: str
    record_ids: list[str]
    values: list[float | int | str | None] = Field(default_factory=list)
    units: list[str | None] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None


class OledConditionDedupReport(BaseModel):
    observations: list[OledConditionDedupObservation] = Field(default_factory=list)
    findings: list[OledConditionDedupFinding] = Field(default_factory=list)
    consistent_duplicate_key_hashes: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def conflict_count(self) -> int:
        return sum(1 for finding in self.findings if finding.code == "dedup_value_conflict")

    @property
    def consistent_duplicate_count(self) -> int:
        return len(self.consistent_duplicate_key_hashes)


def build_oled_condition_dedup_observations(
    records: Iterable[OledGoldDatasetRecord],
    *,
    target_property_id: str = "eqe_percent",
) -> list[OledConditionDedupObservation]:
    gold_records = list(records)
    gold_report = validate_oled_gold_dataset(gold_records)
    if not gold_report.is_valid:
        raise ValueError(f"invalid_gold_records:{','.join(gold_report.error_codes)}")

    clean_target = str(target_property_id or "").strip()
    if not clean_target:
        raise ValueError("target_property_id is required")

    observations: list[OledConditionDedupObservation] = []
    for record in gold_records:
        schema_report = record.layered_record.validate_schema()
        target_observations = [
            observation
            for observation in schema_report.observations
            if observation.layer == OledCausalLayer.MEASUREMENT
            and observation.property_id == clean_target
        ]
        for observation in target_observations:
            observations.append(
                OledConditionDedupObservation(
                    record_id=record.record_id,
                    target_property_id=clean_target,
                    dedup_key=_dedup_key_for_observation(record, observation, clean_target),
                    normalized_value=observation.normalized_value,
                    normalized_unit=observation.normalized_unit,
                )
            )
    return observations


def detect_oled_condition_dedup_conflicts(
    records: Iterable[OledGoldDatasetRecord],
    *,
    target_property_id: str = "eqe_percent",
    value_tolerance: float = 0.0,
) -> OledConditionDedupReport:
    observations = build_oled_condition_dedup_observations(records, target_property_id=target_property_id)
    groups: dict[str, list[OledConditionDedupObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.dedup_key.key_hash].append(observation)

    findings: list[OledConditionDedupFinding] = []
    consistent_duplicate_key_hashes: list[str] = []
    tolerance = max(float(value_tolerance), 0.0)
    for key_hash, grouped_observations in sorted(groups.items()):
        if len(grouped_observations) <= 1:
            continue
        conflict = _conflict_finding_for_group(key_hash, grouped_observations, tolerance)
        if conflict is None:
            consistent_duplicate_key_hashes.append(key_hash)
            continue
        findings.append(conflict)

    return OledConditionDedupReport(
        observations=observations,
        findings=findings,
        consistent_duplicate_key_hashes=consistent_duplicate_key_hashes,
    )


def _dedup_key_for_observation(
    record: OledGoldDatasetRecord,
    observation: OledLayeredCanonicalObservation,
    target_property_id: str,
) -> OledConditionAwareDedupKey:
    layered_record = record.layered_record
    return OledConditionAwareDedupKey(
        target_property_id=target_property_id,
        components={
            "device": _device_component(layered_record.device),
            "interaction": _interaction_component(layered_record.interaction),
            "measurement_condition": _condition_component(observation.normalized_condition or observation.condition),
            "molecule": _molecule_component(layered_record.molecule, layered_record.interaction),
        },
    )


def _molecule_component(
    molecule: OledMolecularLayer | None,
    interaction: OledInteractionLayer | None,
) -> dict[str, Any]:
    if molecule is not None and _clean_text(molecule.inchikey):
        return {"identity_type": "inchikey", "identity": _clean_text(molecule.inchikey)}
    if molecule is not None and _clean_text(molecule.canonical_smiles):
        return {"identity_type": "canonical_smiles", "identity": _clean_text(molecule.canonical_smiles)}
    if interaction is not None and _clean_text(interaction.emitter_smiles):
        return {"identity_type": "emitter_smiles", "identity": _clean_text(interaction.emitter_smiles)}
    return {"identity_type": "missing", "identity": None}


def _interaction_component(interaction: OledInteractionLayer | None) -> dict[str, Any]:
    if interaction is None:
        return {}
    return {
        "aggregation_state": _normalize_label(interaction.aggregation_state),
        "doping_ratio": interaction.doping_ratio,
        "emitter_smiles": _clean_text(interaction.emitter_smiles),
        "film_type": _normalize_label(interaction.film_type),
        "host_smiles": _clean_text(interaction.host_smiles),
        "matrix_type": _normalize_label(interaction.matrix_type),
    }


def _device_component(device: OledDeviceLayer | None) -> dict[str, Any]:
    if device is None:
        return {}
    return {
        "device_stack": [_normalize_label(layer) for layer in device.device_stack if _normalize_label(layer)],
        "etl_material": _normalize_label(device.etl_material),
        "fabrication_method": _normalize_label(device.fabrication_method),
        "htl_material": _normalize_label(device.htl_material),
        "layer_thickness_nm": {
            _normalize_label(key): value
            for key, value in sorted(device.layer_thickness_nm.items())
        },
        "outcoupling_structure": _normalize_label(device.outcoupling_structure),
    }


def _condition_component(condition: OledMeasurementCondition | None) -> dict[str, Any]:
    if condition is None:
        return {}
    return {
        "atmosphere": _normalize_label(condition.atmosphere),
        "condition_label": _normalize_label(condition.condition_label),
        "current_density_ma_cm2": condition.current_density_ma_cm2,
        "luminance_cd_m2": condition.luminance_cd_m2,
        "temperature_k": condition.temperature_k,
        "voltage_v": condition.voltage_v,
    }


def _conflict_finding_for_group(
    key_hash: str,
    observations: list[OledConditionDedupObservation],
    tolerance: float,
) -> OledConditionDedupFinding | None:
    numeric_values = [_numeric_value(observation.normalized_value) for observation in observations]
    record_ids = sorted(observation.record_id for observation in observations)
    values = [observation.normalized_value for observation in observations]
    units = [observation.normalized_unit for observation in observations]
    target_property_id = observations[0].target_property_id
    if all(value is not None for value in numeric_values):
        clean_values = [float(value) for value in numeric_values if value is not None]
        minimum = min(clean_values)
        maximum = max(clean_values)
        if maximum - minimum <= tolerance:
            return None
        return OledConditionDedupFinding(
            code="dedup_value_conflict",
            message=(
                f"dedup key `{key_hash}` has conflicting values for `{target_property_id}` "
                f"outside tolerance {tolerance}"
            ),
            target_property_id=target_property_id,
            dedup_key_hash=key_hash,
            record_ids=record_ids,
            values=values,
            units=units,
            min_value=minimum,
            max_value=maximum,
        )

    distinct_values = {_stable_json(value) for value in values}
    if len(distinct_values) <= 1:
        return None
    return OledConditionDedupFinding(
        code="dedup_value_conflict",
        message=f"dedup key `{key_hash}` has conflicting non-numeric values for `{target_property_id}`",
        target_property_id=target_property_id,
        dedup_key_hash=key_hash,
        record_ids=record_ids,
        values=values,
        units=units,
    )


def _numeric_value(value: float | int | str | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: str | None) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _normalize_label(value: str | None) -> str | None:
    clean = " ".join(str(value or "").strip().lower().split())
    return clean or None


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "OledConditionAwareDedupKey",
    "OledConditionDedupFinding",
    "OledConditionDedupObservation",
    "OledConditionDedupReport",
    "build_oled_condition_dedup_observations",
    "detect_oled_condition_dedup_conflicts",
]
