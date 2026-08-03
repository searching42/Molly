"""Canonical input identity helpers for the BR1 applicability preflight.

This module is intentionally an operator-side contract.  It does not publish
runtime artifacts, register a dataset, or create an acceptance run.  The
helpers make the distinction between the source CSV bytes and the canonical
provider input bytes explicit so that a source artifact cannot be confused
with the mapped Uni-Mol input.
"""

from __future__ import annotations

import csv
import io
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from ai4s_agent.structured_dataset_confirmation import REQUIRED_COLUMNS


CANONICALIZATION_CONTRACT_VERSION = "br1_provider_input_canonicalization.v1"
MAPPING_BINDING_VERSION = "br1_preflight_mapping_binding.v1"
SOURCE_PUBLICATION_REGISTRY_SCHEMA = "br1_source_publication_registry.v1"
SOURCE_AUTHORITY_SCHEMA = "br1_preflight_source_authority.v1"
PROVIDER_NAME = "unimol-tools"
EXECUTION_PROFILE_ID = "unimol-train-br1-v2"

SOURCE_COLUMN_ORDER = tuple(REQUIRED_COLUMNS)
PROVIDER_INPUT_COLUMN_ORDER = ("smiles", "target_value")
CONDITION_CONTEXT_FIELDS = (
    "material_role",
    "emission_mechanism",
    "medium",
    "host",
    "doping_ratio",
    "temperature",
    "measurement_condition",
    "comparable",
    "paper_id",
    "paper_evidence",
)


def _canonical_decimal(value: str) -> str:
    """Return a stable decimal spelling without changing invalid values."""

    raw = str(value or "").strip()
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    if not number.is_finite():
        return raw
    if number == 0:
        return "0"
    rendered = format(number.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def canonical_field_value(field: str, value: Any) -> str:
    """Normalize one checked-in Raw Dataset field for canonical CSV output."""

    rendered = str(value or "").strip()
    if field == "target_value":
        return _canonical_decimal(rendered)
    if field == "measurement_condition" and rendered.startswith("{"):
        try:
            parsed = json.loads(rendered)
        except json.JSONDecodeError:
            return rendered
        if isinstance(parsed, dict):
            return json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    return rendered


def canonical_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return the source contract in fixed columns and row-id order."""

    prepared = [
        {
            field: canonical_field_value(field, row.get(field, ""))
            for field in SOURCE_COLUMN_ORDER
        }
        for row in rows
    ]
    return sorted(prepared, key=lambda row: row["row_id"])


def canonical_provider_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return the normalized, uniquely identified provider roster.

    Provider bytes and provider invocation positions must be derived from the
    same ordered row objects.  Validate the row identity here so duplicate or
    empty IDs cannot be hidden by sorting or accidentally rebound by position.
    """

    provider_rows = canonical_source_rows(rows)
    row_ids = [row["row_id"] for row in provider_rows]
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("canonical provider rows require unique non-empty row_id")
    return provider_rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(columns),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return stream.getvalue().encode("utf-8")


def canonical_source_dataset_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical bytes for the complete structured Raw Dataset contract."""

    return _csv_bytes(canonical_source_rows(rows), SOURCE_COLUMN_ORDER)


def canonical_provider_input_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical bytes for the exact ``smiles,target_value`` provider input."""

    return canonical_provider_input_bytes_from_rows(canonical_provider_rows(rows))


def canonical_provider_input_bytes_from_rows(
    provider_rows: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize an already canonical provider row roster without reordering."""

    row_ids = [str(row.get("row_id") or "") for row in provider_rows]
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("canonical provider rows require unique non-empty row_id")
    if row_ids != sorted(row_ids):
        raise ValueError("provider rows are not in canonical row_id order")
    return _csv_bytes(provider_rows, PROVIDER_INPUT_COLUMN_ORDER)


def mapping_binding(expected_provider_version: str) -> dict[str, Any]:
    """Return the server-owned mapping execution binding for BR1 preflight."""

    return {
        "schema_version": MAPPING_BINDING_VERSION,
        "mapping_policy_schema_version": "br1_raw_dataset_mapping_policy.v1",
        "source_schema_version": "structured_raw_dataset.v1",
        "provider_name": PROVIDER_NAME,
        "expected_provider_version": str(expected_provider_version),
        "execution_profile_id": EXECUTION_PROFILE_ID,
        "molecule_representation_field": "smiles",
        "target_field": "target_value",
        "row_identity_field": "row_id",
        "condition_context_fields": list(CONDITION_CONTEXT_FIELDS),
        "required_source_columns": list(SOURCE_COLUMN_ORDER),
        "missing_value_policy": "reject",
        "filter_policy": "no_implicit_filter",
        "duplicate_row_policy": "reject_duplicate_standard_inchikey",
        "canonical_row_order": "row_id_ascending",
        "output_columns": list(PROVIDER_INPUT_COLUMN_ORDER),
    }


def mapping_binding_semantic_material(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: binding[key]
        for key in (
            "schema_version",
            "mapping_policy_schema_version",
            "source_schema_version",
            "provider_name",
            "expected_provider_version",
            "execution_profile_id",
            "molecule_representation_field",
            "target_field",
            "row_identity_field",
            "condition_context_fields",
            "required_source_columns",
            "missing_value_policy",
            "filter_policy",
            "duplicate_row_policy",
            "canonical_row_order",
            "output_columns",
        )
    }


def canonical_mapping_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return dict(mapping_binding_semantic_material(binding))


__all__ = [
    "CANONICALIZATION_CONTRACT_VERSION",
    "CONDITION_CONTEXT_FIELDS",
    "EXECUTION_PROFILE_ID",
    "MAPPING_BINDING_VERSION",
    "PROVIDER_INPUT_COLUMN_ORDER",
    "PROVIDER_NAME",
    "SOURCE_AUTHORITY_SCHEMA",
    "SOURCE_COLUMN_ORDER",
    "SOURCE_PUBLICATION_REGISTRY_SCHEMA",
    "canonical_field_value",
    "canonical_mapping_binding",
    "canonical_provider_input_bytes",
    "canonical_source_dataset_bytes",
    "canonical_source_rows",
    "mapping_binding",
    "mapping_binding_semantic_material",
]
