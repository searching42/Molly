from __future__ import annotations

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledComparisonContextStatus,
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
    oled_observations_are_directly_comparable,
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


def test_photophysical_observation_preserves_missing_context_without_comparability() -> None:
    record = _prompt_lifetime_record(
        OledMeasurementCondition(sample_form="doped film")
    )

    report = record.validate_schema()
    observation = report.observations[0]

    assert report.is_valid is True
    assert observation.property_id == "prompt_lifetime_ns"
    assert observation.comparison_context_status == OledComparisonContextStatus.INCOMPLETE
    assert observation.comparison_context_hash is None
    assert observation.comparison_context_missing_fields == [
        "measurement_temperature",
        "host_material",
        "dopant_concentration",
        "excitation_wavelength",
        "lifetime_fit_method",
    ]
    assert "incomplete_photophysical_comparison_context" in report.warning_codes


def test_photophysical_comparison_requires_complete_matching_normalized_context() -> None:
    condition = OledMeasurementCondition(
        measurement_temperature=25,
        measurement_temperature_unit="°C",
        host_material="mCBP",
        dopant_concentration=0.08,
        dopant_concentration_unit="fraction",
        sample_form="doped film",
        excitation_wavelength=0.3,
        excitation_wavelength_unit="µm",
        lifetime_fit_method="biexponential fit",
    )
    first = _prompt_lifetime_record(condition).validate_schema().observations[0]
    second = _prompt_lifetime_record(condition.model_copy()).validate_schema().observations[0]
    different_host = _prompt_lifetime_record(
        condition.model_copy(update={"host_material": "DPEPO"})
    ).validate_schema().observations[0]
    incomplete = _prompt_lifetime_record(
        condition.model_copy(update={"lifetime_fit_method": None})
    ).validate_schema().observations[0]

    assert first.comparison_context_status == OledComparisonContextStatus.COMPLETE
    assert first.comparison_context_missing_fields == []
    assert first.comparison_context_hash
    assert first.normalized_condition is not None
    assert first.normalized_condition.measurement_temperature == 298.15
    assert first.normalized_condition.measurement_temperature_unit == "K"
    assert first.normalized_condition.dopant_concentration == 8
    assert first.normalized_condition.dopant_concentration_unit == "%"
    assert first.normalized_condition.excitation_wavelength == 300
    assert first.normalized_condition.excitation_wavelength_unit == "nm"
    assert oled_observations_are_directly_comparable(first, second) is True
    assert oled_observations_are_directly_comparable(first, different_host) is False
    assert oled_observations_are_directly_comparable(first, incomplete) is False


def _prompt_lifetime_record(condition: OledMeasurementCondition) -> OledLayeredRecord:
    return OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            properties=[
                OledPropertyObservation(
                    property_label="prompt PL lifetime",
                    value=13.2,
                    unit="ns",
                    condition=condition,
                )
            ],
        ),
    )
