from __future__ import annotations

import pytest

from ai4s_agent.domains.oled_contracts import (
    DEFAULT_OLED_REPRESENTATION_CONTRACT,
    OledCausalLayer,
)
from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OledPropertyDefinition,
    OledPropertyOntology,
    OledPropertyValueConstraint,
)


def test_default_ontology_resolves_aliases_to_canonical_property() -> None:
    definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.resolve("HOMO level")

    assert definition.property_id == "homo_ev"
    assert definition.name == "HOMO energy level"
    assert definition.canonical_unit == "eV"
    assert definition.allowed_layers == {OledCausalLayer.MOLECULE}
    assert "frontier orbital" in definition.physical_interpretation


def test_default_ontology_rejects_property_layer_mismatch() -> None:
    report = DEFAULT_OLED_PROPERTY_ONTOLOGY.validate_layer("external quantum efficiency", OledCausalLayer.MOLECULE)

    assert report.is_valid is False
    assert report.error_codes == ["property_layer_not_allowed"]
    assert report.definition.property_id == "eqe_percent"
    assert report.findings[0].allowed_layers == {OledCausalLayer.MEASUREMENT}


def test_default_ontology_enforces_value_constraints() -> None:
    valid_report = DEFAULT_OLED_PROPERTY_ONTOLOGY.validate_value("PLQY", 0.82)
    invalid_report = DEFAULT_OLED_PROPERTY_ONTOLOGY.validate_value("PLQY", 1.2)

    assert valid_report.is_valid is True
    assert valid_report.error_codes == []
    assert invalid_report.is_valid is False
    assert invalid_report.error_codes == ["value_above_maximum"]
    assert invalid_report.definition.value_constraint is not None
    assert invalid_report.definition.value_constraint.maximum == 1


def test_excited_state_energies_allow_neat_film_interaction_context() -> None:
    for property_id in ("s1_ev", "t1_ev", "delta_e_st_ev"):
        definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.get(property_id)
        assert definition.allowed_layers == {
            OledCausalLayer.MOLECULE,
            OledCausalLayer.INTERACTION,
        }
        assert DEFAULT_OLED_PROPERTY_ONTOLOGY.validate_layer(
            property_id,
            OledCausalLayer.INTERACTION,
        ).is_valid


def test_ontology_builds_canonical_contract_claim() -> None:
    claim = DEFAULT_OLED_PROPERTY_ONTOLOGY.representation_claim(
        "EQE",
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

    assert claim.property_id == "eqe_percent"
    assert DEFAULT_OLED_REPRESENTATION_CONTRACT.validate_claim(claim).is_valid is True


def test_ontology_rejects_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="duplicate OLED property ontology alias"):
        OledPropertyOntology(
            [
                OledPropertyDefinition(
                    property_id="plqy",
                    name="Photoluminescence quantum yield",
                    aliases={"PLQY"},
                    allowed_layers={OledCausalLayer.INTERACTION},
                    canonical_unit="fraction",
                    value_constraint=OledPropertyValueConstraint(minimum=0, maximum=1),
                    physical_interpretation="radiative photon yield under a specified photophysical environment",
                ),
                OledPropertyDefinition(
                    property_id="phi_pl",
                    name="Photoluminescence yield",
                    aliases={"PLQY"},
                    allowed_layers={OledCausalLayer.INTERACTION},
                    canonical_unit="fraction",
                    value_constraint=OledPropertyValueConstraint(minimum=0, maximum=1),
                    physical_interpretation="alternate naming that should not share aliases silently",
                ),
            ]
        )
