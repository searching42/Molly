from __future__ import annotations

from ai4s_agent.domains.oled_contracts import (
    DEFAULT_OLED_REPRESENTATION_CONTRACT,
    OledCausalLayer,
    RepresentationClaim,
)


def test_molecular_claim_rejects_device_dependency() -> None:
    claim = RepresentationClaim(
        property_id="homo_ev",
        target_layer=OledCausalLayer.MOLECULE,
        bound_layers={OledCausalLayer.MOLECULE},
        dependency_layers={OledCausalLayer.MOLECULE, OledCausalLayer.DEVICE},
    )

    report = DEFAULT_OLED_REPRESENTATION_CONTRACT.validate_claim(claim)

    assert report.is_valid is False
    assert "downstream_dependency_forbidden" in report.error_codes
    assert "device_to_intrinsic_property_forbidden" in report.error_codes


def test_measurement_claim_requires_interaction_and_device_bindings() -> None:
    claim = RepresentationClaim(
        property_id="eqe_percent",
        target_layer=OledCausalLayer.MEASUREMENT,
        bound_layers={OledCausalLayer.MOLECULE, OledCausalLayer.MEASUREMENT},
        dependency_layers={OledCausalLayer.MOLECULE, OledCausalLayer.MEASUREMENT},
    )

    report = DEFAULT_OLED_REPRESENTATION_CONTRACT.validate_claim(claim)

    assert report.is_valid is False
    assert "required_bound_layer_missing:interaction" in report.error_codes
    assert "required_bound_layer_missing:device" in report.error_codes
    assert "required_dependency_layer_missing:interaction" in report.error_codes
    assert "required_dependency_layer_missing:device" in report.error_codes


def test_valid_measurement_claim_passes_with_full_causal_context() -> None:
    claim = RepresentationClaim(
        property_id="eqe_percent",
        target_layer=OledCausalLayer.MEASUREMENT,
        bound_layers={
            OledCausalLayer.MOLECULE,
            OledCausalLayer.INTERACTION,
            OledCausalLayer.DEVICE,
            OledCausalLayer.MEASUREMENT,
        },
        dependency_layers={
            OledCausalLayer.MOLECULE,
            OledCausalLayer.INTERACTION,
            OledCausalLayer.DEVICE,
            OledCausalLayer.MEASUREMENT,
        },
    )

    report = DEFAULT_OLED_REPRESENTATION_CONTRACT.validate_claim(claim)

    assert report.is_valid is True
    assert report.error_codes == []


def test_device_claim_rejects_measurement_dependency() -> None:
    claim = RepresentationClaim(
        property_id="device_stack_quality",
        target_layer=OledCausalLayer.DEVICE,
        bound_layers={OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION, OledCausalLayer.DEVICE},
        dependency_layers={
            OledCausalLayer.MOLECULE,
            OledCausalLayer.INTERACTION,
            OledCausalLayer.DEVICE,
            OledCausalLayer.MEASUREMENT,
        },
    )

    report = DEFAULT_OLED_REPRESENTATION_CONTRACT.validate_claim(claim)

    assert report.is_valid is False
    assert "downstream_dependency_forbidden" in report.error_codes
