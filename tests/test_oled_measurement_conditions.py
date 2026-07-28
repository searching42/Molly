from __future__ import annotations

from ai4s_agent.domains import OledMeasurementCondition as PackageOledMeasurementCondition
from ai4s_agent.domains.oled_contracts import OledCausalLayer, RepresentationContract
from ai4s_agent.domains.oled_layered_schema import (
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)


def test_measurement_condition_is_preserved_with_stable_hash() -> None:
    condition = OledMeasurementCondition(
        luminance_cd_m2=100,
        current_density_ma_cm2=4.2,
        temperature_k=298.15,
        atmosphere="nitrogen",
    )
    record = _record_with_measurement_condition(condition)

    report = record.validate_schema()

    assert report.error_codes == []
    assert report.observations[0].condition == condition
    assert report.observations[0].condition_hash == condition.condition_hash
    assert condition.condition_hash == OledMeasurementCondition(
        atmosphere="nitrogen",
        current_density_ma_cm2=4.2,
        luminance_cd_m2=100.0,
        temperature_k=298.15,
    ).condition_hash


def test_measurement_observation_requires_condition_context() -> None:
    record = _record_with_measurement_condition(None)

    report = record.validate_schema()

    assert "missing_measurement_condition" in report.error_codes
    assert report.findings[0].property_id == "eqe_percent"


def test_custom_contract_layer_order_is_used_for_dependency_calculation() -> None:
    custom_contract = RepresentationContract(
        layer_order={
            OledCausalLayer.MOLECULE: 0,
            OledCausalLayer.MEASUREMENT: 1,
            OledCausalLayer.INTERACTION: 2,
            OledCausalLayer.DEVICE: 3,
        },
        required_bound_layers={OledCausalLayer.MEASUREMENT: {OledCausalLayer.MEASUREMENT}},
        required_dependency_layers={},
    )
    record = _record_with_measurement_condition(OledMeasurementCondition(luminance_cd_m2=100))

    report = record.validate_schema(contract=custom_contract)

    assert "downstream_dependency_forbidden" not in report.error_codes


def test_measurement_condition_is_exported_from_domain_package() -> None:
    condition = PackageOledMeasurementCondition(luminance_cd_m2=100)

    assert condition.condition_hash == OledMeasurementCondition(luminance_cd_m2=100).condition_hash


def _record_with_measurement_condition(condition: OledMeasurementCondition | None) -> OledLayeredRecord:
    return OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(emitter_smiles="N1C=CC=C1", host_smiles="c1ccccc1"),
        device=OledDeviceLayer(device_stack=["ITO", "EML", "Al"]),
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE (%)",
                    value=18.2,
                    unit="%",
                    condition=condition,
                )
            ]
        ),
    )
