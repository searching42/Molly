from __future__ import annotations

import re
from enum import Enum
from typing import Callable

from pydantic import BaseModel


class OledUnitNormalizationStatus(str, Enum):
    NORMALIZED = "normalized"
    UNCHANGED = "unchanged"
    MISSING_UNIT = "missing_unit"
    UNKNOWN_UNIT = "unknown_unit"
    UNKNOWN_PROPERTY = "unknown_property"
    NOT_NUMERIC = "not_numeric"


class OledUnitNormalizationResult(BaseModel):
    original_value: float | int | str | None
    original_unit: str | None = None
    normalized_value: float | int | str | None
    normalized_unit: str | None = None
    status: OledUnitNormalizationStatus
    property_id: str | None = None
    field_name: str | None = None
    message: str | None = None


def normalize_oled_property_unit(
    property_id: str,
    value: float | int | str | None,
    unit: str | None,
) -> OledUnitNormalizationResult:
    """Normalize a property observation using the property's semantic unit contract."""

    clean_property_id = str(property_id or "").strip()
    rule = _PROPERTY_UNIT_RULES.get(clean_property_id)
    if rule is None:
        return _result(
            value,
            unit,
            value,
            unit,
            OledUnitNormalizationStatus.UNKNOWN_PROPERTY,
            property_id=clean_property_id,
            message=f"no OLED unit normalization rule for property `{clean_property_id}`",
        )
    return _apply_rule(rule, value, unit, property_id=clean_property_id)


def normalize_oled_condition_field(
    field_name: str,
    value: float | int | str | None,
    unit: str | None,
) -> OledUnitNormalizationResult:
    """Normalize a measurement-condition field into the schema field's canonical unit."""

    clean_field_name = str(field_name or "").strip()
    rule = _CONDITION_UNIT_RULES.get(clean_field_name)
    if rule is None:
        return _result(
            value,
            unit,
            value,
            unit,
            OledUnitNormalizationStatus.UNKNOWN_PROPERTY,
            field_name=clean_field_name,
            message=f"no OLED unit normalization rule for condition field `{clean_field_name}`",
        )
    return _apply_rule(rule, value, unit, field_name=clean_field_name)


class _UnitRule(BaseModel):
    canonical_unit: str
    conversions: dict[str, float | str]
    canonical_status_units: set[str]


def _apply_rule(
    rule: _UnitRule,
    value: float | int | str | None,
    unit: str | None,
    *,
    property_id: str | None = None,
    field_name: str | None = None,
) -> OledUnitNormalizationResult:
    if unit is None or not str(unit).strip():
        return _result(
            value,
            unit,
            value,
            rule.canonical_unit,
            OledUnitNormalizationStatus.MISSING_UNIT,
            property_id=property_id,
            field_name=field_name,
            message="unit is missing; value is left unchanged",
        )

    unit_key = _normalize_unit_token(unit)
    conversion = rule.conversions.get(unit_key)
    if conversion is None:
        return _result(
            value,
            unit,
            value,
            unit,
            OledUnitNormalizationStatus.UNKNOWN_UNIT,
            property_id=property_id,
            field_name=field_name,
            message=f"unit `{unit}` is not supported for normalization",
        )

    if isinstance(conversion, str) and conversion in _CONVERTERS:
        normalized_unit = rule.canonical_unit
        converter = _CONVERTERS[conversion]
    elif isinstance(conversion, str):
        normalized_unit = conversion
        converter = _scale(1.0)
    else:
        normalized_unit = rule.canonical_unit
        converter = _scale(conversion)
    numeric_value = _numeric_value(value)
    if numeric_value is None:
        return _result(
            value,
            unit,
            value,
            normalized_unit,
            OledUnitNormalizationStatus.NOT_NUMERIC,
            property_id=property_id,
            field_name=field_name,
            message="value is not numeric; unit is recognized but value is left unchanged",
        )

    normalized_value = converter(numeric_value)
    status = (
        OledUnitNormalizationStatus.UNCHANGED
        if unit_key in rule.canonical_status_units and _numbers_equal(normalized_value, numeric_value)
        else OledUnitNormalizationStatus.NORMALIZED
    )
    return _result(
        value,
        unit,
        normalized_value,
        normalized_unit,
        status,
        property_id=property_id,
        field_name=field_name,
    )


def _result(
    original_value: float | int | str | None,
    original_unit: str | None,
    normalized_value: float | int | str | None,
    normalized_unit: str | None,
    status: OledUnitNormalizationStatus,
    *,
    property_id: str | None = None,
    field_name: str | None = None,
    message: str | None = None,
) -> OledUnitNormalizationResult:
    return OledUnitNormalizationResult(
        original_value=original_value,
        original_unit=original_unit,
        normalized_value=normalized_value,
        normalized_unit=normalized_unit,
        status=status,
        property_id=property_id,
        field_name=field_name,
        message=message,
    )


def _numeric_value(value: float | int | str | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scale(factor: float) -> Callable[[float], float]:
    return lambda value: value * factor


def _celsius_to_kelvin(value: float) -> float:
    return value + 273.15


def _numbers_equal(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12


def _normalize_unit_token(unit: str | None) -> str:
    text = str(unit or "").strip().lower()
    replacements = {
        "μ": "u",
        "µ": "u",
        "²": "2",
        "−": "-",
        "–": "-",
        "—": "-",
        "℃": "°c",
        "％": "%",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("per", "/")
    text = re.sub(r"\s+", "", text)
    text = text.replace("cm^-2", "cm-2").replace("m^-2", "m-2")
    text = text.replace("cm^2", "cm2").replace("m^2", "m2")
    text = text.replace("degc", "°c").replace("degreec", "°c")
    return text


_CONVERTERS: dict[str, Callable[[float], float]] = {
    "fraction_to_percent": lambda value: value * 100.0,
    "percent_to_fraction": lambda value: value / 100.0,
    "mev_to_ev": lambda value: value / 1000.0,
    "celsius_to_kelvin": _celsius_to_kelvin,
    "a_m2_to_ma_cm2": lambda value: value * 0.1,
    "a_cm2_to_ma_cm2": lambda value: value * 1000.0,
}


_EV_RULE = _UnitRule(
    canonical_unit="eV",
    conversions={
        "ev": 1.0,
        "electronvolt": 1.0,
        "electronvolts": 1.0,
        "mev": "mev_to_ev",
        "millielectronvolt": "mev_to_ev",
        "millielectronvolts": "mev_to_ev",
    },
    canonical_status_units={"ev", "electronvolt", "electronvolts"},
)

_PROPERTY_UNIT_RULES: dict[str, _UnitRule] = {
    "homo_ev": _EV_RULE,
    "lumo_ev": _EV_RULE,
    "s1_ev": _EV_RULE,
    "t1_ev": _EV_RULE,
    "delta_e_st_ev": _EV_RULE,
    "eqe_percent": _UnitRule(
        canonical_unit="%",
        conversions={
            "%": 1.0,
            "percent": 1.0,
            "percentage": 1.0,
            "fraction": "fraction_to_percent",
            "frac": "fraction_to_percent",
            "unitless": "fraction_to_percent",
            "1": "fraction_to_percent",
        },
        canonical_status_units={"%", "percent", "percentage"},
    ),
    "plqy": _UnitRule(
        canonical_unit="fraction",
        conversions={
            "fraction": 1.0,
            "frac": 1.0,
            "unitless": 1.0,
            "1": 1.0,
            "%": "percent_to_fraction",
            "percent": "percent_to_fraction",
            "percentage": "percent_to_fraction",
        },
        canonical_status_units={"fraction", "frac", "unitless", "1"},
    ),
    "luminance_cd_m2": _UnitRule(
        canonical_unit="cd/m^2",
        conversions={
            "cd/m2": 1.0,
            "cdm-2": 1.0,
            "cd/m-2": 1.0,
        },
        canonical_status_units={"cd/m2", "cdm-2", "cd/m-2"},
    ),
    "current_density_ma_cm2": _UnitRule(
        canonical_unit="mA/cm^2",
        conversions={
            "ma/cm2": 1.0,
            "macm-2": 1.0,
            "ma/cm-2": 1.0,
            "a/m2": "a_m2_to_ma_cm2",
            "am-2": "a_m2_to_ma_cm2",
            "a/m-2": "a_m2_to_ma_cm2",
            "a/cm2": "a_cm2_to_ma_cm2",
            "acm-2": "a_cm2_to_ma_cm2",
            "a/cm-2": "a_cm2_to_ma_cm2",
        },
        canonical_status_units={"ma/cm2", "macm-2", "ma/cm-2"},
    ),
    "doping_ratio_percent": _UnitRule(
        canonical_unit="%",
        conversions={
            "%": 1.0,
            "percent": 1.0,
            "percentage": 1.0,
            "wt%": "wt%",
            "wtpercent": "wt%",
            "weight%": "wt%",
            "weightpercent": "wt%",
            "mol%": "mol%",
            "molpercent": "mol%",
            "mole%": "mol%",
            "molepercent": "mol%",
            "fraction": "fraction_to_percent",
            "frac": "fraction_to_percent",
            "unitless": "fraction_to_percent",
            "1": "fraction_to_percent",
        },
        canonical_status_units={
            "%",
            "percent",
            "percentage",
            "wt%",
            "wtpercent",
            "mol%",
            "molpercent",
        },
    ),
}

_CONDITION_UNIT_RULES: dict[str, _UnitRule] = {
    "luminance_cd_m2": _PROPERTY_UNIT_RULES["luminance_cd_m2"],
    "current_density_ma_cm2": _PROPERTY_UNIT_RULES["current_density_ma_cm2"],
    "temperature_k": _UnitRule(
        canonical_unit="K",
        conversions={
            "k": 1.0,
            "kelvin": 1.0,
            "°c": "celsius_to_kelvin",
            "c": "celsius_to_kelvin",
            "celsius": "celsius_to_kelvin",
        },
        canonical_status_units={"k", "kelvin"},
    ),
}


__all__ = [
    "OledUnitNormalizationResult",
    "OledUnitNormalizationStatus",
    "normalize_oled_condition_field",
    "normalize_oled_property_unit",
]
