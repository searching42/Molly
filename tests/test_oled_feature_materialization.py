from __future__ import annotations

import json

import pytest

from ai4s_agent.domains import materialize_oled_baseline_feature_tables as PackageMaterializeFeatureTables
from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_feature_materialization import (
    materialize_oled_baseline_feature_table,
    materialize_oled_baseline_feature_tables,
    write_oled_feature_table_jsonl,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledConfounderFlags,
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


def test_materializes_three_baseline_feature_views_with_stable_columns() -> None:
    tables = materialize_oled_baseline_feature_tables([_gold_record("gold-oled-001")])

    assert [table.feature_view for table in tables] == [
        OledBaselineFeatureView.MOLECULE_ONLY,
        OledBaselineFeatureView.MOLECULE_INTERACTION,
        OledBaselineFeatureView.FULL_CONTEXT,
    ]
    molecule_columns = tables[0].columns
    assert "feature.molecule.canonical_smiles" in molecule_columns
    assert "feature.interaction.host_smiles" not in molecule_columns
    interaction_columns = tables[1].columns
    assert "feature.interaction.host_smiles" in interaction_columns
    assert "feature.device.device_stack" not in interaction_columns
    full_columns = tables[2].columns
    assert "feature.device.device_stack" in full_columns
    assert "feature.condition.luminance_cd_m2" in full_columns
    assert "feature.confounder_flags.is_device_optimized" in full_columns


def test_feature_table_records_and_jsonl_are_flat_and_deterministic() -> None:
    table = materialize_oled_baseline_feature_table(
        [
            _gold_record(
                "gold-oled-002",
                measurement_observation=_measurement_observation(
                    "gold-oled-002",
                    100,
                    reported_value_text="19.50",
                    reported_decimal_places=2,
                ),
            )
        ],
        feature_view=OledBaselineFeatureView.FULL_CONTEXT,
    )

    records = table.to_records()
    jsonl = table.to_jsonl()

    assert records == [json.loads(jsonl)]
    assert records[0]["record_id"] == "gold-oled-002"
    assert records[0]["target_property_id"] == "eqe_percent"
    assert records[0]["target_value"] == 19.5
    assert records[0]["target_reported_value_text"] == "19.50"
    assert records[0]["target_reported_decimal_places"] == 2
    assert records[0]["target_reported_unit"] == "%"
    assert records[0]["feature.interaction.host_smiles"] == "c1ccccc1"
    assert records[0]["feature.device.device_stack"] == ["ITO", "HTL", "EML", "ETL", "Al"]
    assert records[0]["feature.condition.luminance_cd_m2"] == 100.0
    assert table.to_jsonl() == jsonl


def test_full_context_condition_features_are_bound_to_each_target_row() -> None:
    table = materialize_oled_baseline_feature_table(
        [
            _gold_record(
                "gold-oled-multi-condition",
                measurement_observations=[
                    _measurement_observation("gold-oled-multi-condition", 100),
                    _measurement_observation("gold-oled-multi-condition", 1000),
                ],
            )
        ],
        feature_view=OledBaselineFeatureView.FULL_CONTEXT,
    )

    records = table.to_records()

    assert [record["target_value"] for record in records] == [19.5, 19.5]
    assert [record["feature.condition.luminance_cd_m2"] for record in records] == [100.0, 1000.0]
    assert records[0]["condition_hash"] != records[1]["condition_hash"]
    assert records[0]["feature.condition.condition_hash"] == records[0]["condition_hash"]
    assert records[1]["feature.condition.condition_hash"] == records[1]["condition_hash"]


def test_feature_materialization_uses_normalized_targets_and_condition_features() -> None:
    table = materialize_oled_baseline_feature_table(
        [
            _gold_record(
                "gold-oled-normalized-units",
                measurement_observation=_measurement_observation(
                    "gold-oled-normalized-units",
                    100,
                    value=0.195,
                    unit="fraction",
                    current_density_ma_cm2=42,
                    temperature_k=25,
                    condition_metadata={
                        "current_density_unit": "A/m²",
                        "temperature_unit": "°C",
                    },
                ),
            )
        ],
        feature_view=OledBaselineFeatureView.FULL_CONTEXT,
    )

    record = table.to_records()[0]

    assert record["target_value"] == pytest.approx(19.5)
    assert record["target_unit"] == "%"
    assert record["feature.condition.current_density_ma_cm2"] == pytest.approx(4.2)
    assert record["feature.condition.temperature_k"] == pytest.approx(298.15)


def test_feature_table_jsonl_writer_uses_one_line_per_record(tmp_path) -> None:
    table = materialize_oled_baseline_feature_table(
        [_gold_record("gold-oled-003"), _gold_record("gold-oled-004")],
        feature_view=OledBaselineFeatureView.MOLECULE_ONLY,
    )
    output_path = tmp_path / "molecule_only.jsonl"

    written_path = write_oled_feature_table_jsonl(table, output_path)

    assert written_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["record_id"] == "gold-oled-003"
    assert json.loads(lines[1])["record_id"] == "gold-oled-004"


def test_feature_materialization_rejects_invalid_gold_records() -> None:
    invalid_record = _gold_record(
        "gold-oled-invalid",
        measurement_observation=OledPropertyObservation(
            property_label="EQE (%)",
            value=19.5,
            unit="%",
            condition=OledMeasurementCondition(luminance_cd_m2=100),
        ),
    )

    with pytest.raises(ValueError, match="gold_missing_provenance"):
        materialize_oled_baseline_feature_table(
            [invalid_record],
            feature_view=OledBaselineFeatureView.MOLECULE_ONLY,
        )


def test_feature_materializer_is_exported_from_domain_package() -> None:
    tables = PackageMaterializeFeatureTables([_gold_record("gold-oled-005")])

    assert tables[0].target_property_id == "eqe_percent"


def _gold_record(
    record_id: str,
    *,
    measurement_observation: OledPropertyObservation | None = None,
    measurement_observations: list[OledPropertyObservation] | None = None,
) -> OledGoldDatasetRecord:
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey=f"{record_id}-INCHIKEY",
            ),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=0.08,
                film_type="doped",
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
                etl_material="TPBi",
                htl_material="TAPC",
                outcoupling_structure="none",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    *(measurement_observations or [measurement_observation or _measurement_observation(record_id, 100)])
                ]
            ),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[f"{record_id}:table-2:row-4"],
    )


def _measurement_observation(
    record_id: str,
    luminance_cd_m2: float,
    *,
    value: float = 19.5,
    unit: str = "%",
    current_density_ma_cm2: float = 4.2,
    temperature_k: float = 298.15,
    condition_metadata: dict[str, str] | None = None,
    reported_value_text: str | None = None,
    reported_decimal_places: int | None = None,
) -> OledPropertyObservation:
    return OledPropertyObservation(
        property_label="EQE (%)",
        value=value,
        unit=unit,
        reported_value_text=reported_value_text,
        reported_decimal_places=reported_decimal_places,
        condition=OledMeasurementCondition(
            luminance_cd_m2=luminance_cd_m2,
            current_density_ma_cm2=current_density_ma_cm2,
            temperature_k=temperature_k,
            metadata=condition_metadata or {},
        ),
        evidence_sources=[
            OledEvidenceSource(
                source_id=f"{record_id}:table-2:row-4",
                source_type=OledEvidenceType.TABLE,
                layer=OledCausalLayer.MEASUREMENT,
            )
        ],
        confidence=OledConfidenceAssessment(score=0.92),
    )
