"""Closed OLED property vocabulary and explicit unit normalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes


SUPPORTED_PROPERTIES = frozenset({"PLQY"})


class PropertyUnitStatus(str, Enum):
    NORMALIZED = "NORMALIZED"
    UNRESOLVED = "UNRESOLVED"


def _property(value: Any) -> str:
    if not isinstance(value, str):
        raise CoreContractError("property_id must be text")
    normalized = value.strip().upper().replace(" ", "_")
    aliases = {"PHOTOLUMINESCENCE_QUANTUM_YIELD": "PLQY", "QUANTUM_YIELD": "PLQY"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROPERTIES:
        raise CoreContractError(f"unsupported OLED property: {value!r}")
    return normalized


def _number(value: Any, *, field: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoreContractError(f"{field} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise CoreContractError(f"{field} must be finite")
    return value


def _unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 64 or "\x00" in value:
        raise CoreContractError("property unit must be bounded text")
    return value.strip().casefold() or None


@dataclass(frozen=True, slots=True)
class NormalizedProperty:
    property_id: str
    value: float | int | None
    unit: str | None
    original_value: Any = None
    original_unit: str | None = None
    status: str | PropertyUnitStatus = PropertyUnitStatus.UNRESOLVED

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_id", _property(self.property_id))
        value = _number(self.value, field="property value")
        original_value = self.original_value if self.original_value is not None else value
        if isinstance(original_value, bool) or not isinstance(original_value, (int, float, str, type(None))):
            raise CoreContractError("original property value must be scalar")
        if isinstance(original_value, float) and not math.isfinite(original_value):
            raise CoreContractError("original property value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", _unit(self.unit))
        object.__setattr__(self, "original_value", original_value)
        object.__setattr__(self, "original_unit", _unit(self.original_unit))
        status = self.status.value if isinstance(self.status, PropertyUnitStatus) else self.status
        if not isinstance(status, str):
            raise CoreContractError("property unit status must be text")
        try:
            object.__setattr__(self, "status", PropertyUnitStatus(status.strip().upper()).value)
        except ValueError as exc:
            raise CoreContractError(f"unknown property unit status: {status!r}") from exc
        if self.status == PropertyUnitStatus.NORMALIZED.value:
            if value is None or self.unit is None:
                raise CoreContractError("normalized property requires value and unit")
            if self.property_id == "PLQY" and not (0.0 <= float(value) <= 1.0):
                raise CoreContractError("normalized PLQY fraction must be between 0 and 1")

    @property
    def normalized(self) -> bool:
        return self.status == PropertyUnitStatus.NORMALIZED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "value": self.value,
            "unit": self.unit,
            "original_value": self.original_value,
            "original_unit": self.original_unit,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalizedProperty":
        if not isinstance(value, Mapping):
            raise CoreContractError("property must be an object")
        allowed = {"property_id", "value", "unit", "original_value", "original_unit", "status", "property_value"}
        if set(value) - allowed:
            raise CoreContractError("property has unknown fields")
        property_id = value["property_id"]
        raw_value = value.get("value", value.get("property_value"))
        raw_unit = value.get("unit")
        if "status" not in value:
            return cls.normalize(property_id, raw_value, raw_unit)
        return cls(
            property_id=property_id,
            value=raw_value,
            unit=raw_unit,
            original_value=value.get("original_value", raw_value),
            original_unit=value.get("original_unit", raw_unit),
            status=value["status"],
        )

    @classmethod
    def normalize(
        cls,
        property_id: str,
        value: Any,
        unit: str | None,
    ) -> "NormalizedProperty":
        property_id = _property(property_id)
        original_value = value
        original_unit = unit
        if value is None or unit is None:
            return cls(property_id, None, None, original_value, original_unit, PropertyUnitStatus.UNRESOLVED)
        try:
            numeric = _number(value, field="property value")
        except CoreContractError:
            return cls(property_id, None, _unit(unit), original_value, original_unit, PropertyUnitStatus.UNRESOLVED)
        normalized_unit = _unit(unit)
        if normalized_unit in {"%", "percent", "percentage"}:
            numeric = float(numeric) / 100.0
            normalized_unit = "fraction"
        elif normalized_unit not in {"fraction", "unitless"}:
            return cls(property_id, None, normalized_unit, original_value, original_unit, PropertyUnitStatus.UNRESOLVED)
        return cls(property_id, numeric, "fraction", original_value, original_unit, PropertyUnitStatus.NORMALIZED)


__all__ = ["NormalizedProperty", "PropertyUnitStatus", "SUPPORTED_PROPERTIES"]
