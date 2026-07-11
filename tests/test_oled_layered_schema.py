from __future__ import annotations

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)


def test_valid_layered_record_passes_with_measurement_context() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(
            canonical_smiles="N1C=CC=C1",
            inchikey="SYNTHETIC-INCHIKEY",
            properties=[OledPropertyObservation(property_label="HOMO level", value=-5.4, unit="eV")],
        ),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            doping_ratio=0.08,
            film_type="doped",
            properties=[OledPropertyObservation(property_label="PLQY", value=0.72, unit="fraction")],
        ),
        device=OledDeviceLayer(
            device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
            etl_material="TPBi",
            htl_material="TAPC",
        ),
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="max EQE (%)",
                    value=22.4,
                    unit="%",
                    condition=OledMeasurementCondition(luminance_cd_m2=100),
                )
            ]
        ),
    )

    report = record.validate_schema()

    assert report.is_valid is True
    assert report.error_codes == []
    assert report.canonical_property_ids == ["homo_ev", "plqy", "eqe_percent"]


def test_layered_schema_preserves_reported_numeric_lexeme() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            properties=[
                OledPropertyObservation(
                    property_label="ΔE ST",
                    value=0.03,
                    unit="eV",
                    reported_value_text="0.030",
                    reported_decimal_places=3,
                )
            ],
        )
    )

    observation = record.validate_schema().observations[0]

    assert observation.value == 0.03
    assert observation.reported_value_text == "0.030"
    assert observation.reported_decimal_places == 3


def test_layered_record_rejects_measurement_property_on_molecular_layer() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(
            canonical_smiles="N1C=CC=C1",
            properties=[OledPropertyObservation(property_label="external quantum efficiency", value=18.0, unit="%")],
        )
    )

    report = record.validate_schema()

    assert report.is_valid is False
    assert "property_layer_not_allowed" in report.error_codes
    assert report.findings[0].layer == OledCausalLayer.MOLECULE
    assert report.findings[0].property_id == "eqe_percent"


def test_layered_record_requires_device_context_for_measurement() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(emitter_smiles="N1C=CC=C1", host_smiles="c1ccccc1"),
        measurement=OledMeasurementLayer(
            measurements=[OledPropertyObservation(property_label="EQE", value=16.8, unit="%")]
        ),
    )

    report = record.validate_schema()

    assert report.is_valid is False
    assert "required_bound_layer_missing:device" in report.error_codes
    assert "required_dependency_layer_missing:device" in report.error_codes


def test_layered_record_canonicalizes_table_property_labels() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(emitter_smiles="N1C=CC=C1"),
        device=OledDeviceLayer(device_stack=["ITO", "EML", "Al"]),
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE (%)",
                    value=11.2,
                    unit="%",
                    condition=OledMeasurementCondition(luminance_cd_m2=100),
                )
            ]
        ),
    )

    report = record.validate_schema()

    assert report.canonical_property_ids == ["eqe_percent"]
    assert report.error_codes == []
    assert set(report.finding_codes_for_property("eqe_percent")) == {
        "missing_confounder_tags",
        "missing_confidence",
        "missing_provenance",
    }
