from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_NUMERIC_LEXEME_PATTERN = re.compile(
    r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
)


def reported_decimal_places(reported_value_text: str) -> int:
    clean = _clean_reported_value_text(reported_value_text)
    match = _NUMERIC_LEXEME_PATTERN.fullmatch(clean)
    if match is None:
        raise ValueError("reported_value_text must be a numeric lexeme without a unit")
    mantissa = clean.lower().split("e", 1)[0]
    return len(mantissa.split(".", 1)[1]) if "." in mantissa else 0


def reported_value_fields(
    raw_source_value: str | None,
    numeric_value: float | int | str | None,
) -> tuple[str | None, int | None]:
    if raw_source_value is None or not _is_numeric_value(numeric_value):
        return None, None
    target = _as_decimal(numeric_value)
    for match in _NUMERIC_LEXEME_PATTERN.finditer(str(raw_source_value)):
        lexeme = match.group(0)
        if _as_decimal(lexeme) == target:
            return lexeme, reported_decimal_places(lexeme)
    return None, None


def is_numeric_reported_value(value: Any) -> bool:
    return _is_numeric_value(value)


def validate_reported_value_contract(
    *,
    value: float | int | str | None,
    reported_value_text: str | None,
    reported_decimal_places_value: int | None,
    label: str = "reported value",
) -> None:
    if reported_value_text is None and reported_decimal_places_value is None:
        return
    if reported_value_text is None or reported_decimal_places_value is None:
        raise ValueError(
            f"{label} requires both reported_value_text and reported_decimal_places"
        )
    clean = _clean_reported_value_text(reported_value_text)
    actual_places = reported_decimal_places(clean)
    if reported_decimal_places_value < 0:
        raise ValueError(f"{label} reported_decimal_places must be non-negative")
    if reported_decimal_places_value != actual_places:
        raise ValueError(
            f"{label} reported_decimal_places does not match reported_value_text"
        )
    if _is_numeric_value(value) and _as_decimal(value) != _as_decimal(clean):
        raise ValueError(f"{label} numeric value does not match reported_value_text")


def _clean_reported_value_text(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("reported_value_text must be non-empty")
    return clean


def _is_numeric_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        _as_decimal(value)
    except ValueError:
        return False
    return True


def _as_decimal(value: Any) -> Decimal:
    clean = str(value).strip().replace("−", "-")
    try:
        return Decimal(clean)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a numeric value: {value}") from exc


__all__ = [
    "is_numeric_reported_value",
    "reported_decimal_places",
    "reported_value_fields",
    "validate_reported_value_contract",
]
