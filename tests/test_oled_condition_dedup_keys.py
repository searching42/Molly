from __future__ import annotations

import pytest

from ai4s_agent.domains import build_oled_condition_dedup_observations as PackageBuildDedupObservations
from ai4s_agent.domains.oled_condition_dedup import (
    build_oled_condition_dedup_observations,
    detect_oled_condition_dedup_conflicts,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
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


def test_condition_dedup_key_uses_normalized_condition_and_target_property() -> None:
    records = [
        _gold_record(
            "gold-normalized-a",
            value=0.195,
            unit="fraction",
            current_density_ma_cm2=42,
            temperature_k=25,
            condition_metadata={
                "current_density_unit": "A/m²",
                "temperature_unit": "°C",
            },
        ),
        _gold_record(
            "gold-normalized-b",
            value=19.5,
            unit="%",
            current_density_ma_cm2=4.2,
            temperature_k=298.15,
        ),
    ]

    observations = build_oled_condition_dedup_observations(records, target_property_id="eqe_percent")

    assert [observation.record_id for observation in observations] == ["gold-normalized-a", "gold-normalized-b"]
    assert observations[0].dedup_key.key_hash == observations[1].dedup_key.key_hash
    assert observations[0].dedup_key.target_property_id == "eqe_percent"
    assert observations[0].normalized_value == pytest.approx(19.5)
    assert observations[0].normalized_unit == "%"
    assert observations[0].dedup_key.components["measurement_condition"]["current_density_ma_cm2"] == pytest.approx(4.2)
    assert observations[0].dedup_key.components["measurement_condition"]["temperature_k"] == pytest.approx(298.15)


def test_condition_dedup_key_keeps_device_and_measurement_condition_context_separate() -> None:
    records = [
        _gold_record("gold-100-cd", luminance_cd_m2=100),
        _gold_record("gold-1000-cd", luminance_cd_m2=1000),
        _gold_record("gold-alt-device", etl_material="BPhen"),
    ]

    observations = build_oled_condition_dedup_observations(records)

    key_hashes = {observation.dedup_key.key_hash for observation in observations}
    assert len(key_hashes) == 3


def test_condition_dedup_conflict_report_uses_condition_aware_key() -> None:
    records = [
        _gold_record("gold-conflict-a", value=19.5),
        _gold_record("gold-conflict-b", value=21.0),
        _gold_record("gold-distinct-condition", value=21.0, luminance_cd_m2=1000),
    ]

    report = detect_oled_condition_dedup_conflicts(records, value_tolerance=0.1)

    assert report.is_valid is False
    assert report.error_codes == ["dedup_value_conflict"]
    assert report.conflict_count == 1
    assert report.consistent_duplicate_count == 0
    finding = report.findings[0]
    assert finding.record_ids == ["gold-conflict-a", "gold-conflict-b"]
    assert finding.target_property_id == "eqe_percent"
    assert finding.min_value == pytest.approx(19.5)
    assert finding.max_value == pytest.approx(21.0)


def test_condition_dedup_conflict_report_counts_consistent_duplicates() -> None:
    report = detect_oled_condition_dedup_conflicts(
        [_gold_record("gold-dup-a", value=19.5), _gold_record("gold-dup-b", value=19.5)]
    )

    assert report.is_valid is True
    assert report.error_codes == []
    assert report.conflict_count == 0
    assert report.consistent_duplicate_count == 1


def test_condition_dedup_builder_is_exported_from_domain_package() -> None:
    observations = PackageBuildDedupObservations([_gold_record("gold-export")])

    assert observations[0].record_id == "gold-export"
    assert observations[0].dedup_key.target_property_id == "eqe_percent"


def _gold_record(
    record_id: str,
    *,
    value: float = 19.5,
    unit: str = "%",
    luminance_cd_m2: float = 100,
    current_density_ma_cm2: float = 4.2,
    temperature_k: float = 298.15,
    condition_metadata: dict[str, str] | None = None,
    etl_material: str = "TPBi",
) -> OledGoldDatasetRecord:
    evidence_ref = f"paper:{record_id}:table-1:row-1"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey="SYNTHETIC-INCHIKEY",
            ),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=0.08,
                film_type="doped",
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
                etl_material=etl_material,
                htl_material="TAPC",
                outcoupling_structure="none",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    OledPropertyObservation(
                        property_label="EQE (%)",
                        value=value,
                        unit=unit,
                        condition=OledMeasurementCondition(
                            luminance_cd_m2=luminance_cd_m2,
                            current_density_ma_cm2=current_density_ma_cm2,
                            temperature_k=temperature_k,
                            metadata=condition_metadata or {},
                        ),
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=evidence_ref,
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MEASUREMENT,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.92),
                    )
                ]
            ),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[evidence_ref],
    )
