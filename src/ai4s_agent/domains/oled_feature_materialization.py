from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset
from ai4s_agent.domains.oled_layered_schema import OledLayeredCanonicalObservation


class OledFeatureMaterializationRow(BaseModel):
    record_id: str
    feature_view: OledBaselineFeatureView
    target_property_id: str
    target_value: float | int | str | None
    target_unit: str | None = None
    condition_hash: str | None = None
    confidence_score: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "condition_hash": self.condition_hash,
            "confidence_score": self.confidence_score,
            "evidence_refs": list(self.evidence_refs),
            "feature_view": self.feature_view.value,
            "record_id": self.record_id,
            "target_property_id": self.target_property_id,
            "target_unit": self.target_unit,
            "target_value": self.target_value,
        }
        for key in sorted(self.features):
            record[f"feature.{key}"] = self.features[key]
        return record


class OledFeatureMaterializationTable(BaseModel):
    feature_view: OledBaselineFeatureView
    target_property_id: str
    rows: list[OledFeatureMaterializationRow] = Field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        columns = [
            "record_id",
            "feature_view",
            "target_property_id",
            "target_value",
            "target_unit",
            "condition_hash",
            "confidence_score",
            "evidence_refs",
        ]
        feature_columns = sorted(
            {
                f"feature.{feature_name}"
                for row in self.rows
                for feature_name in row.features
            }
        )
        return columns + feature_columns

    def to_records(self) -> list[dict[str, Any]]:
        return [_order_record(row.to_record(), self.columns) for row in self.rows]

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for record in self.to_records()
        )


def materialize_oled_baseline_feature_tables(
    records: Iterable[OledGoldDatasetRecord],
    *,
    target_property_id: str = "eqe_percent",
) -> list[OledFeatureMaterializationTable]:
    gold_records = list(records)
    return [
        materialize_oled_baseline_feature_table(
            gold_records,
            feature_view=feature_view,
            target_property_id=target_property_id,
        )
        for feature_view in (
            OledBaselineFeatureView.MOLECULE_ONLY,
            OledBaselineFeatureView.MOLECULE_INTERACTION,
            OledBaselineFeatureView.FULL_CONTEXT,
        )
    ]


def materialize_oled_baseline_feature_table(
    records: Iterable[OledGoldDatasetRecord],
    *,
    feature_view: OledBaselineFeatureView,
    target_property_id: str = "eqe_percent",
) -> OledFeatureMaterializationTable:
    gold_records = list(records)
    gold_report = validate_oled_gold_dataset(gold_records)
    if not gold_report.is_valid:
        raise ValueError(f"invalid_gold_records:{','.join(gold_report.error_codes)}")

    clean_target = str(target_property_id or "").strip()
    if not clean_target:
        raise ValueError("target_property_id is required")

    rows: list[OledFeatureMaterializationRow] = []
    for gold_record in gold_records:
        schema_report = gold_record.layered_record.validate_schema()
        target_observations = [
            observation
            for observation in schema_report.observations
            if observation.layer == OledCausalLayer.MEASUREMENT
            and observation.property_id == clean_target
        ]
        for observation in target_observations:
            rows.append(
                OledFeatureMaterializationRow(
                    record_id=gold_record.record_id,
                    feature_view=feature_view,
                    target_property_id=clean_target,
                    target_value=observation.normalized_value,
                    target_unit=observation.normalized_unit,
                    condition_hash=observation.normalized_condition_hash,
                    confidence_score=observation.confidence.score if observation.confidence else None,
                    evidence_refs=_evidence_refs(gold_record, observation),
                    features=_features_for_view(gold_record, feature_view, observation),
                )
            )
    if not rows:
        raise ValueError(f"no_gold_records_for_target:{clean_target}")
    return OledFeatureMaterializationTable(
        feature_view=feature_view,
        target_property_id=clean_target,
        rows=rows,
    )


def write_oled_feature_table_jsonl(
    table: OledFeatureMaterializationTable,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = table.to_jsonl()
    output_path.write_text(f"{content}\n" if content else "", encoding="utf-8")
    return output_path


def _features_for_view(
    gold_record: OledGoldDatasetRecord,
    feature_view: OledBaselineFeatureView,
    observation: OledLayeredCanonicalObservation,
) -> dict[str, Any]:
    record = gold_record.layered_record
    features = _molecule_features(gold_record)
    if feature_view in {
        OledBaselineFeatureView.MOLECULE_INTERACTION,
        OledBaselineFeatureView.FULL_CONTEXT,
    }:
        features.update(_interaction_features(gold_record))
    if feature_view == OledBaselineFeatureView.FULL_CONTEXT:
        features.update(_device_features(gold_record))
        features.update(_condition_features(observation))
        features.update(
            {
                "confounder_flags.is_best_reported": record.confounder_flags.is_best_reported,
                "confounder_flags.is_device_optimized": record.confounder_flags.is_device_optimized,
                "confounder_flags.is_outcoupling_modified": record.confounder_flags.is_outcoupling_modified,
            }
        )
    return features


def _molecule_features(gold_record: OledGoldDatasetRecord) -> dict[str, Any]:
    molecule = gold_record.layered_record.molecule
    return {
        "molecule.canonical_smiles": molecule.canonical_smiles if molecule else None,
        "molecule.inchikey": molecule.inchikey if molecule else None,
    }


def _interaction_features(gold_record: OledGoldDatasetRecord) -> dict[str, Any]:
    interaction = gold_record.layered_record.interaction
    return {
        "interaction.aggregation_state": interaction.aggregation_state if interaction else None,
        "interaction.doping_ratio": interaction.doping_ratio if interaction else None,
        "interaction.emitter_smiles": interaction.emitter_smiles if interaction else None,
        "interaction.film_type": interaction.film_type if interaction else None,
        "interaction.host_smiles": interaction.host_smiles if interaction else None,
        "interaction.matrix_type": interaction.matrix_type if interaction else None,
    }


def _device_features(gold_record: OledGoldDatasetRecord) -> dict[str, Any]:
    device = gold_record.layered_record.device
    return {
        "device.device_stack": list(device.device_stack) if device else [],
        "device.etl_material": device.etl_material if device else None,
        "device.fabrication_method": device.fabrication_method if device else None,
        "device.htl_material": device.htl_material if device else None,
        "device.layer_thickness_nm": dict(sorted(device.layer_thickness_nm.items())) if device else {},
        "device.outcoupling_structure": device.outcoupling_structure if device else None,
    }


def _condition_features(observation: OledLayeredCanonicalObservation) -> dict[str, Any]:
    condition = observation.normalized_condition or observation.condition
    return {
        "condition.atmosphere": condition.atmosphere if condition else None,
        "condition.condition_hash": condition.condition_hash if condition else None,
        "condition.condition_label": condition.condition_label if condition else None,
        "condition.current_density_ma_cm2": condition.current_density_ma_cm2 if condition else None,
        "condition.luminance_cd_m2": condition.luminance_cd_m2 if condition else None,
        "condition.temperature_k": condition.temperature_k if condition else None,
        "condition.voltage_v": condition.voltage_v if condition else None,
    }


def _evidence_refs(
    gold_record: OledGoldDatasetRecord,
    observation: OledLayeredCanonicalObservation,
) -> list[str]:
    refs = {source.source_id for source in observation.evidence_sources}
    refs.update(gold_record.evidence_refs)
    return sorted(refs)


def _order_record(record: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for column in columns:
        if column in record:
            ordered[column] = record[column]
    return ordered


__all__ = [
    "OledFeatureMaterializationRow",
    "OledFeatureMaterializationTable",
    "materialize_oled_baseline_feature_table",
    "materialize_oled_baseline_feature_tables",
    "write_oled_feature_table_jsonl",
]
