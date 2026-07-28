from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.domains import OledConfidenceAssessment as PackageOledConfidenceAssessment
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledEvidenceSource,
    OledEvidenceType,
    OledLayeredRecord,
    OledMeasurementLayer,
    OledPropertyObservation,
)


def test_observation_provenance_and_confidence_are_preserved_in_schema_report() -> None:
    record = OledLayeredRecord(
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE (%)",
                    value=21.4,
                    unit="%",
                    evidence_sources=[
                        OledEvidenceSource(
                            source_id="paper-1:table-2:row-4",
                            source_type=OledEvidenceType.TABLE,
                            layer=OledCausalLayer.MEASUREMENT,
                            locator="Table 2, row 4",
                        )
                    ],
                    confidence=OledConfidenceAssessment(
                        score=0.91,
                        factors={"table_extraction": 0.96, "unit_normalized": 0.86},
                        rationale=["structured_table_hit", "unit_suffix_present"],
                    ),
                )
            ]
        )
    )

    report = record.validate_schema()

    assert report.observations[0].property_id == "eqe_percent"
    assert report.observations[0].evidence_sources[0].source_id == "paper-1:table-2:row-4"
    assert report.observations[0].confidence is not None
    assert report.observations[0].confidence.score == 0.91
    assert report.confidence_by_layer == {OledCausalLayer.MEASUREMENT: 0.91}


def test_schema_report_warns_when_observation_lacks_provenance() -> None:
    record = OledLayeredRecord(
        measurement=OledMeasurementLayer(
            measurements=[OledPropertyObservation(property_label="EQE (%)", value=21.4, unit="%")]
        )
    )

    report = record.validate_schema()

    assert report.is_valid is False
    assert "missing_provenance" in report.warning_codes
    assert "missing_confidence" in report.warning_codes


def test_schema_report_detects_evidence_layer_mismatch() -> None:
    record = OledLayeredRecord(
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE (%)",
                    value=21.4,
                    unit="%",
                    evidence_sources=[
                        OledEvidenceSource(
                            source_id="paper-1:table-2",
                            source_type=OledEvidenceType.TABLE,
                            layer=OledCausalLayer.MOLECULE,
                            locator="Table 2",
                        )
                    ],
                    confidence=OledConfidenceAssessment(score=0.8),
                )
            ]
        )
    )

    report = record.validate_schema()

    assert "evidence_layer_mismatch" in report.error_codes
    assert report.findings[0].property_id == "eqe_percent"


def test_confidence_type_is_exported_from_domain_package() -> None:
    confidence = PackageOledConfidenceAssessment(score=0.73)

    assert confidence.score == 0.73


def test_confidence_score_must_be_probability() -> None:
    with pytest.raises(ValidationError, match="confidence score must be between 0.0 and 1.0"):
        OledConfidenceAssessment(score=1.2)

    with pytest.raises(ValidationError, match="confidence score must be a number, got bool"):
        OledConfidenceAssessment(score=True)
