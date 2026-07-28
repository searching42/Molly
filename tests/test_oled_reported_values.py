from __future__ import annotations

import pytest

from ai4s_agent.domains.oled_reported_values import (
    reported_decimal_places,
    reported_value_fields,
    validate_reported_value_contract,
)


def test_reported_value_fields_preserve_trailing_zero() -> None:
    assert reported_value_fields("The reported value was 0.030 eV.", 0.03) == (
        "0.030",
        3,
    )
    assert reported_value_fields("HOMO = −5.50 eV", -5.5) == ("−5.50", 2)


def test_reported_decimal_places_counts_source_mantissa() -> None:
    assert reported_decimal_places("0.030") == 3
    assert reported_decimal_places("-5.50") == 2
    assert reported_decimal_places("1.20e−3") == 2
    assert reported_decimal_places("4") == 0


def test_reported_value_contract_rejects_numeric_or_precision_mismatch() -> None:
    with pytest.raises(ValueError, match="numeric value does not match"):
        validate_reported_value_contract(
            value=0.03,
            reported_value_text="0.031",
            reported_decimal_places_value=3,
        )
    with pytest.raises(ValueError, match="decimal_places does not match"):
        validate_reported_value_contract(
            value=0.03,
            reported_value_text="0.030",
            reported_decimal_places_value=2,
        )
