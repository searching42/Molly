from __future__ import annotations

import pytest

from ai4s_agent.domains.oled_layered_schema import (
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_units import (
    OledUnitNormalizationStatus,
    normalize_oled_condition_field,
    normalize_oled_property_unit,
)


def test_property_unit_normalization_handles_priority_oled_units() -> None:
    assert normalize_oled_property_unit("homo_ev", -5400, "meV").normalized_value == pytest.approx(-5.4)
    assert normalize_oled_property_unit("lumo_ev", -2.3, "eV").normalized_unit == "eV"
    assert normalize_oled_property_unit("s1_ev", 2700, "meV").normalized_value == pytest.approx(2.7)
    assert normalize_oled_property_unit("t1_ev", 2.1, "eV").normalized_unit == "eV"
    assert normalize_oled_property_unit("delta_e_st_ev", 180, "meV").normalized_value == pytest.approx(0.18)

    assert normalize_oled_property_unit("eqe_percent", 0.215, "fraction").normalized_value == pytest.approx(21.5)
    assert normalize_oled_property_unit("eqe_percent", 21.5, "%").normalized_unit == "%"
    assert normalize_oled_property_unit("plqy", 72, "%").normalized_value == pytest.approx(0.72)
    assert normalize_oled_property_unit("plqy", 0.72, "fraction").normalized_unit == "fraction"

    assert normalize_oled_property_unit("doping_ratio_percent", 8, "wt%").normalized_unit == "wt%"
    assert normalize_oled_property_unit("doping_ratio_percent", 4.5, "mol%").normalized_unit == "mol%"
    assert normalize_oled_property_unit("doping_ratio_percent", 0.08, "fraction").normalized_value == pytest.approx(8)


def test_property_unit_normalization_accepts_unicode_minus_lexemes() -> None:
    homo = normalize_oled_property_unit("homo_ev", "−1.59", "eV")
    exponent = normalize_oled_property_unit("delta_e_st_ev", "1e−3", "eV")

    assert homo.status == OledUnitNormalizationStatus.UNCHANGED
    assert homo.normalized_value == pytest.approx(-1.59)
    assert exponent.status == OledUnitNormalizationStatus.UNCHANGED
    assert exponent.normalized_value == pytest.approx(0.001)


def test_condition_field_normalization_handles_priority_oled_units() -> None:
    assert normalize_oled_condition_field("luminance_cd_m2", 100, "cd/m²").normalized_unit == "cd/m^2"
    assert normalize_oled_condition_field("current_density_ma_cm2", 42, "A/m²").normalized_value == pytest.approx(4.2)
    assert normalize_oled_condition_field("temperature_k", 25, "°C").normalized_value == pytest.approx(298.15)
    assert normalize_oled_condition_field("temperature_k", 298.15, "K").normalized_unit == "K"


def test_photophysical_property_and_context_units_are_normalized() -> None:
    assert normalize_oled_property_unit(
        "photoluminescence_peak_nm", 0.49, "µm"
    ).normalized_value == pytest.approx(490)
    assert normalize_oled_property_unit(
        "prompt_lifetime_ns", 0.0132, "µs"
    ).normalized_value == pytest.approx(13.2)
    assert normalize_oled_property_unit(
        "delayed_lifetime_us", 0.0139, "ms"
    ).normalized_value == pytest.approx(13.9)
    assert normalize_oled_condition_field(
        "measurement_temperature", 25, "°C"
    ).normalized_value == pytest.approx(298.15)
    assert normalize_oled_condition_field(
        "excitation_wavelength", 0.3, "µm"
    ).normalized_value == pytest.approx(300)
    assert normalize_oled_condition_field(
        "dopant_concentration", 0.08, "fraction"
    ).normalized_value == pytest.approx(8)


def test_layered_schema_reports_normalized_observation_values_and_conditions() -> None:
    condition = OledMeasurementCondition(
        luminance_cd_m2=100,
        current_density_ma_cm2=42,
        temperature_k=25,
        metadata={
            "luminance_unit": "cd/m²",
            "current_density_unit": "A/m²",
            "temperature_unit": "°C",
        },
    )
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(
            canonical_smiles="N1C=CC=C1",
            inchikey="SYNTHETIC-INCHIKEY",
            properties=[OledPropertyObservation(property_label="HOMO level", value=-5400, unit="meV")],
        ),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            properties=[OledPropertyObservation(property_label="PLQY", value=72, unit="%")],
        ),
        device=OledDeviceLayer(device_stack=["ITO", "EML", "Al"]),
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE",
                    value=0.215,
                    unit="fraction",
                    condition=condition,
                )
            ]
        ),
    )

    report = record.validate_schema()

    assert report.error_codes == []
    observations = {observation.property_id: observation for observation in report.observations}
    assert observations["homo_ev"].normalized_value == pytest.approx(-5.4)
    assert observations["homo_ev"].normalized_unit == "eV"
    assert observations["plqy"].normalized_value == pytest.approx(0.72)
    assert observations["plqy"].normalized_unit == "fraction"
    assert observations["eqe_percent"].normalized_value == pytest.approx(21.5)
    assert observations["eqe_percent"].normalized_unit == "%"

    normalized_condition = observations["eqe_percent"].normalized_condition
    assert normalized_condition is not None
    assert normalized_condition.luminance_cd_m2 == pytest.approx(100)
    assert normalized_condition.current_density_ma_cm2 == pytest.approx(4.2)
    assert normalized_condition.temperature_k == pytest.approx(298.15)
    assert observations["eqe_percent"].normalized_condition_hash == normalized_condition.condition_hash


def test_unknown_units_are_reported_as_warnings_not_schema_errors() -> None:
    record = OledLayeredRecord(
        molecule=OledMolecularLayer(
            canonical_smiles="N1C=CC=C1",
            properties=[OledPropertyObservation(property_label="HOMO level", value=-5.4, unit="hartree-ish")],
        )
    )

    report = record.validate_schema()

    assert report.is_valid is True
    assert "unknown_unit" in report.warning_codes
    assert report.observations[0].unit_normalization_status == OledUnitNormalizationStatus.UNKNOWN_UNIT
    assert report.observations[0].normalized_value == -5.4
