from __future__ import annotations

from ai4s_agent.domains import OledGoldDatasetRecord as PackageOledGoldDatasetRecord
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_gold_validation import (
    OledGoldDatasetRecord,
    validate_oled_gold_dataset,
)
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


def test_gold_validation_accepts_complete_layered_records() -> None:
    gold_record = OledGoldDatasetRecord(
        record_id="gold-oled-001",
        layered_record=_complete_layered_record(),
        evidence_refs=["paper-1:table-2:row-4"],
    )

    report = validate_oled_gold_dataset([gold_record])

    assert report.is_valid is True
    assert report.error_codes == []
    assert report.valid_record_ids == ["gold-oled-001"]


def test_gold_validation_promotes_missing_provenance_and_confidence_to_errors() -> None:
    gold_record = OledGoldDatasetRecord(
        record_id="gold-oled-002",
        layered_record=_complete_layered_record(
            measurement_observation=OledPropertyObservation(
                property_label="EQE (%)",
                value=19.5,
                unit="%",
                condition=OledMeasurementCondition(luminance_cd_m2=100),
            )
        ),
        evidence_refs=["paper-1:table-2:row-4"],
    )

    report = validate_oled_gold_dataset([gold_record])

    assert report.is_valid is False
    assert "gold_missing_provenance" in report.error_codes
    assert "gold_missing_confidence" in report.error_codes
    assert report.valid_record_ids == []


def test_gold_validation_rejects_schema_invalid_records() -> None:
    gold_record = OledGoldDatasetRecord(
        record_id="gold-oled-003",
        layered_record=_complete_layered_record(
            measurement_observation=OledPropertyObservation(
                property_label="EQE (%)",
                value=19.5,
                unit="%",
                evidence_sources=[_measurement_evidence()],
                confidence=OledConfidenceAssessment(score=0.92),
            )
        ),
        evidence_refs=["paper-1:table-2:row-4"],
    )

    report = validate_oled_gold_dataset([gold_record])

    assert report.is_valid is False
    assert "missing_measurement_condition" in report.error_codes
    assert report.valid_record_ids == []


def test_gold_validation_rejects_duplicate_ids_and_missing_evidence_refs() -> None:
    records = [
        OledGoldDatasetRecord(
            record_id="gold-oled-004",
            layered_record=_complete_layered_record(),
            evidence_refs=[],
        ),
        OledGoldDatasetRecord(
            record_id="gold-oled-004",
            layered_record=_complete_layered_record(),
            evidence_refs=["paper-1:table-2:row-4"],
        ),
    ]

    report = validate_oled_gold_dataset(records)

    assert report.is_valid is False
    assert report.error_codes.count("duplicate_gold_record_id") == 2
    assert "gold_missing_evidence_refs" in report.error_codes
    assert report.valid_record_ids == []


def test_gold_dataset_record_is_exported_from_domain_package() -> None:
    gold_record = PackageOledGoldDatasetRecord(
        record_id="gold-oled-005",
        layered_record=_complete_layered_record(),
        evidence_refs=["paper-1:table-2:row-4"],
    )

    assert gold_record.record_id == "gold-oled-005"


def _complete_layered_record(
    *,
    measurement_observation: OledPropertyObservation | None = None,
) -> OledLayeredRecord:
    return OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            doping_ratio=0.08,
        ),
        device=OledDeviceLayer(
            device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
            etl_material="TPBi",
            htl_material="TAPC",
        ),
        measurement=OledMeasurementLayer(
            measurements=[
                measurement_observation
                or OledPropertyObservation(
                    property_label="EQE (%)",
                    value=19.5,
                    unit="%",
                    condition=OledMeasurementCondition(luminance_cd_m2=100),
                    evidence_sources=[_measurement_evidence()],
                    confidence=OledConfidenceAssessment(
                        score=0.92,
                        factors={"manual_review": 0.95, "table_alignment": 0.9},
                    ),
                )
            ]
        ),
        confounder_flags=OledConfounderFlags(is_device_optimized=True),
    )


def _measurement_evidence() -> OledEvidenceSource:
    return OledEvidenceSource(
        source_id="paper-1:table-2:row-4",
        source_type=OledEvidenceType.TABLE,
        layer=OledCausalLayer.MEASUREMENT,
        locator="Table 2, row 4",
    )
