from __future__ import annotations

import pytest

from ai4s_agent.domains import DEFAULT_OLED_PROPERTY_TAXONOMY as PACKAGE_OLED_PROPERTY_TAXONOMY
from ai4s_agent.domains.oled_property_taxonomy import DEFAULT_OLED_PROPERTY_TAXONOMY


def test_taxonomy_canonicalizes_alias_with_unit_suffix() -> None:
    match = DEFAULT_OLED_PROPERTY_TAXONOMY.canonicalize("max EQE (%)")

    assert match.raw_term == "max EQE (%)"
    assert match.canonical_property_id == "eqe_percent"
    assert match.canonical_name == "External quantum efficiency"
    assert match.unit_hint == "%"
    assert match.normalized_term == "max_eqe"


def test_taxonomy_normalizes_greek_symbols_and_spacing() -> None:
    match = DEFAULT_OLED_PROPERTY_TAXONOMY.canonicalize(" ΔE ST ")

    assert match.canonical_property_id == "delta_e_st_ev"
    assert match.canonical_name == "Singlet-triplet energy gap"
    assert match.unit_hint == "eV"


def test_taxonomy_batch_canonicalization_preserves_order() -> None:
    matches = DEFAULT_OLED_PROPERTY_TAXONOMY.canonicalize_many(
        ["HOMO level", "external quantum efficiency", "PLQY"]
    )

    assert [match.canonical_property_id for match in matches] == ["homo_ev", "eqe_percent", "plqy"]
    assert [match.unit_hint for match in matches] == ["eV", "%", "fraction"]


def test_taxonomy_is_exported_from_domain_package() -> None:
    match = PACKAGE_OLED_PROPERTY_TAXONOMY.canonicalize("HOMO")

    assert match.canonical_property_id == "homo_ev"


def test_taxonomy_try_canonicalize_does_not_silently_map_unknown_terms() -> None:
    assert DEFAULT_OLED_PROPERTY_TAXONOMY.try_canonicalize("paper device score") is None

    with pytest.raises(KeyError, match="unknown OLED property taxonomy term"):
        DEFAULT_OLED_PROPERTY_TAXONOMY.canonicalize("paper device score")
