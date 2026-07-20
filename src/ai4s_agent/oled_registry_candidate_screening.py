from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
)
from ai4s_agent.oled_real_phase1_execution import (
    _EXECUTION_VERSION,
    _MODEL_KIND,
    _feature_vector_for_model,
    _json_bytes,
    _predict_feature_vector,
    _safe_token,
    _validated_split_by_row,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.trainability import generate_baseline_features


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class _PreparedScreeningInputs:
    execution: dict[str, Any]
    execution_sha256: str
    dataset: OledCategoricalDatasetExecutionArtifact
    dataset_sha256: str
    registry: OledMaterialRegistrySnapshot
    registry_sha256: str
    models: dict[str, dict[str, Any]]
    model_sha256: dict[str, str]
    property_ids: tuple[str, ...]
    directions: dict[str, str]
    training_material_ids: frozenset[str]
    training_registry_digests: frozenset[str]
    training_smiles: frozenset[str]


def _load_screening_inputs(
    *,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
) -> _PreparedScreeningInputs:
    execution_dir = _absolute_local_path(phase1_execution_dir)
    dataset_path = _absolute_local_path(dataset_snapshot_json)
    registry_path = _absolute_local_path(registry_snapshot_json)
    execution, execution_sha = _read_bound_json(
        execution_dir / "execution.json",
        "PR-AO execution receipt",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(execution)) != execution_sha:
        raise ValueError("PR-AO execution receipt is not in canonical form")
    if execution.get("execution_version") != _EXECUTION_VERSION:
        raise ValueError("unsupported PR-AO execution version")
    execution_id = _required_string(execution, "execution_id")
    config = _required_dict(execution, "config")
    property_ids_raw = config.get("property_ids")
    if (
        not isinstance(property_ids_raw, list)
        or not property_ids_raw
        or any(not isinstance(item, str) or not item for item in property_ids_raw)
        or property_ids_raw != sorted(set(property_ids_raw))
    ):
        raise ValueError("PR-AO property roster is invalid")
    property_ids = tuple(property_ids_raw)
    directions_raw = config.get("directions")
    if (
        not isinstance(directions_raw, dict)
        or set(directions_raw) != set(property_ids)
        or any(value not in {"minimize", "maximize"} for value in directions_raw.values())
    ):
        raise ValueError("PR-AO objective directions are invalid")
    directions = {key: str(directions_raw[key]) for key in property_ids}

    dataset_payload, dataset_sha = _read_bound_json(
        dataset_path,
        "PR-AI dataset snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    dataset = OledCategoricalDatasetExecutionArtifact.model_validate(dataset_payload)
    source = _required_dict(execution, "source")
    if (
        source.get("dataset_snapshot_id") != dataset.dataset_snapshot_id
        or source.get("dataset_snapshot_digest") != dataset.execution_artifact_digest
        or source.get("dataset_snapshot_sha256") != dataset_sha
    ):
        raise ValueError("PR-AO source dataset binding mismatch")
    split_by_row = _validated_split_by_row(dataset)

    artifacts = _required_dict(execution, "artifacts")
    models: dict[str, dict[str, Any]] = {}
    model_sha: dict[str, str] = {}
    for property_id in property_ids:
        filename = f"model__{_safe_token(property_id)}.json"
        expected_sha = artifacts.get(filename)
        if not isinstance(expected_sha, str):
            raise ValueError("PR-AO model artifact binding is missing")
        model_payload, actual_sha = _read_bound_json(
            execution_dir / filename,
            f"PR-AO model {property_id}",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if actual_sha != expected_sha:
            raise ValueError("PR-AO model SHA-256 mismatch")
        _validate_model_binding(
            model_payload,
            property_id=property_id,
            execution_id=execution_id,
            dataset=dataset,
            dataset_sha=dataset_sha,
            split_by_row=split_by_row,
        )
        models[property_id] = model_payload
        model_sha[property_id] = actual_sha

    registry_payload, registry_sha = _read_bound_json(
        registry_path,
        "OLED Material Registry snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    registry = OledMaterialRegistrySnapshot.model_validate(registry_payload)
    if _sha256_bytes(_registry_publication_bytes(registry)) != registry_sha:
        raise ValueError("Registry snapshot is not in canonical form")

    selected_rows = [row for row in dataset.rows if row.property_id in property_ids]
    train_rows = [row for row in selected_rows if split_by_row[row.row_id] == "train"]
    return _PreparedScreeningInputs(
        execution=execution,
        execution_sha256=execution_sha,
        dataset=dataset,
        dataset_sha256=dataset_sha,
        registry=registry,
        registry_sha256=registry_sha,
        models=models,
        model_sha256=model_sha,
        property_ids=property_ids,
        directions=directions,
        training_material_ids=frozenset(row.selected_material_id for row in train_rows),
        training_registry_digests=frozenset(row.registry_entry_digest for row in train_rows),
        training_smiles=frozenset(row.canonical_isomeric_smiles for row in train_rows),
    )


def _validate_model_binding(
    model: dict[str, Any],
    *,
    property_id: str,
    execution_id: str,
    dataset: OledCategoricalDatasetExecutionArtifact,
    dataset_sha: str,
    split_by_row: dict[str, str],
) -> None:
    if (
        model.get("model_kind") != _MODEL_KIND
        or model.get("property_id") != property_id
        or model.get("execution_id") != execution_id
        or model.get("source_dataset_snapshot_id") != dataset.dataset_snapshot_id
        or model.get("source_dataset_snapshot_digest")
        != dataset.execution_artifact_digest
        or model.get("source_dataset_snapshot_sha256") != dataset_sha
    ):
        raise ValueError("PR-AO model source binding mismatch")
    train_rows = sorted(
        (
            row
            for row in dataset.rows
            if row.property_id == property_id
            and split_by_row[row.row_id] == "train"
        ),
        key=lambda row: row.row_id,
    )
    if model.get("training_row_ids") != [row.row_id for row in train_rows]:
        raise ValueError("PR-AO model training row roster mismatch")
    if model.get("training_material_ids") != [
        row.selected_material_id for row in train_rows
    ]:
        raise ValueError("PR-AO model training material roster mismatch")
    feature_names = model.get("feature_names")
    if (
        not train_rows
        or not isinstance(feature_names, list)
        or feature_names != sorted(train_rows[0].features)
        or any(sorted(row.features) != feature_names for row in train_rows)
    ):
        raise ValueError("PR-AO model feature contract mismatch")


def _screen_registry_candidates(
    prepared: _PreparedScreeningInputs,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_names = list(prepared.models[prepared.property_ids[0]]["feature_names"])
    if any(
        model.get("feature_names") != feature_names
        for model in prepared.models.values()
    ):
        raise ValueError("screening models use inconsistent feature contracts")
    split_by_row = _validated_split_by_row(prepared.dataset)
    train_feature_types = {
        row.feature_type
        for row in prepared.dataset.rows
        if row.property_id in prepared.property_ids
        and split_by_row[row.row_id] == "train"
    }
    if len(train_feature_types) != 1:
        raise ValueError("training rows use inconsistent feature backends")
    expected_feature_type = next(iter(train_feature_types))

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for entry in prepared.registry.entries:
        reasons: list[str] = []
        if entry.material_id in prepared.training_material_ids:
            reasons.append("training_material_id_overlap")
        if entry.entry_digest in prepared.training_registry_digests:
            reasons.append("training_registry_digest_overlap")
        if entry.canonical_isomeric_smiles in prepared.training_smiles:
            reasons.append("training_smiles_overlap")
        if reasons:
            excluded.append(_excluded_entry(entry, reasons))
            continue
        try:
            generated = generate_baseline_features(
                [entry.canonical_isomeric_smiles],
                n_bits=len(feature_names),
            )
            if generated.feature_type != expected_feature_type or generated.fallback_reason:
                raise ValueError("candidate feature backend mismatch")
            features = dict(zip(feature_names, generated.matrix[0], strict=True))
            property_predictions = {
                property_id: _predict_feature_vector(
                    _feature_vector_for_model(features, prepared.models[property_id]),
                    prepared.models[property_id],
                )
                for property_id in prepared.property_ids
            }
        except (KeyError, TypeError, ValueError, ArithmeticError):
            excluded.append(_excluded_entry(entry, ["feature_or_prediction_failed"]))
            continue
        identity = {
            "material_id": entry.material_id,
            "registry_entry_digest": entry.entry_digest,
            "canonical_name": entry.canonical_name,
            "canonical_isomeric_smiles": entry.canonical_isomeric_smiles,
        }
        eligible.append(identity)
        predictions.append(identity | {"predictions": property_predictions})
    return eligible, excluded, predictions


def _excluded_entry(entry: Any, reason_codes: list[str]) -> dict[str, Any]:
    return {
        "material_id": entry.material_id,
        "registry_entry_digest": entry.entry_digest,
        "canonical_name": entry.canonical_name,
        "canonical_isomeric_smiles": entry.canonical_isomeric_smiles,
        "reason_codes": sorted(set(reason_codes)),
    }


def _parse_constraints(
    *,
    minimums: list[str],
    maximums: list[str],
    property_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    constraints: dict[str, dict[str, float]] = {}
    allowed = set(property_ids)
    for kind, values in (("min", minimums), ("max", maximums)):
        for raw in values:
            property_id, separator, value_raw = raw.partition("=")
            property_id = property_id.strip()
            if separator != "=" or not property_id or not value_raw.strip():
                raise ValueError(f"invalid {kind} constraint")
            if property_id not in allowed:
                raise ValueError("constraint references an unknown property")
            bound = constraints.setdefault(property_id, {})
            if kind in bound:
                label = "minimum" if kind == "min" else "maximum"
                raise ValueError(f"duplicate {label} constraint")
            try:
                numeric = float(value_raw)
            except ValueError as exc:
                raise ValueError("constraint value is not numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError("constraint value must be finite")
            bound[kind] = numeric
    for bound in constraints.values():
        if "min" in bound and "max" in bound and bound["min"] > bound["max"]:
            raise ValueError("constraint defines an empty feasible range")
    return {key: constraints[key] for key in sorted(constraints)}


def _rank_candidates(
    predictions: list[dict[str, Any]],
    *,
    property_ids: tuple[str, ...],
    directions: dict[str, str],
    constraints: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not predictions:
        return [], []
    if set(directions) != set(property_ids) or any(
        value not in {"minimize", "maximize"} for value in directions.values()
    ):
        raise ValueError("candidate ranking directions are invalid")
    scored = [dict(item) for item in predictions]
    for item in scored:
        values = item.get("predictions")
        if not isinstance(values, dict) or set(values) != set(property_ids):
            raise ValueError("candidate prediction roster is incomplete")
        if any(not math.isfinite(float(values[name])) for name in property_ids):
            raise ValueError("candidate prediction is non-finite")
        constraint_results: dict[str, dict[str, bool]] = {}
        reasons: list[str] = []
        for property_id, bounds in constraints.items():
            value = float(values[property_id])
            results: dict[str, bool] = {}
            if "min" in bounds:
                results["min"] = value >= bounds["min"]
                if not results["min"]:
                    reasons.append(f"hard_constraint_failed:{property_id}:min")
            if "max" in bounds:
                results["max"] = value <= bounds["max"]
                if not results["max"]:
                    reasons.append(f"hard_constraint_failed:{property_id}:max")
            constraint_results[property_id] = results
        item["constraint_results"] = constraint_results
        item["hard_constraints_passed"] = not reasons
        item["decision_reason_codes"] = sorted(reasons)

    percentiles = _property_percentiles(scored, property_ids, directions)
    for item in scored:
        material_id = str(item["material_id"])
        item["property_percentiles"] = percentiles[material_id]
        item["aggregate_percentile"] = sum(
            percentiles[material_id].values()
        ) / len(property_ids)

    feasible = [item for item in scored if item["hard_constraints_passed"]]
    for item in scored:
        dominated = item in feasible and any(
            other is not item
            and _dominates(other["predictions"], item["predictions"], directions)
            for other in feasible
        )
        item["pareto_dominated"] = dominated
        if dominated:
            item["decision_reason_codes"] = sorted(
                [*item["decision_reason_codes"], "pareto_dominated"]
            )
    scored.sort(key=lambda item: str(item["material_id"]))
    shortlist = [
        dict(item)
        for item in scored
        if item["hard_constraints_passed"] and not item["pareto_dominated"]
    ]
    shortlist.sort(
        key=lambda item: (
            -float(item["aggregate_percentile"]),
            str(item["material_id"]),
        )
    )
    for index, item in enumerate(shortlist, 1):
        item["rank"] = index
    return scored, shortlist


def _property_percentiles(
    rows: list[dict[str, Any]],
    property_ids: tuple[str, ...],
    directions: dict[str, str],
) -> dict[str, dict[str, float]]:
    output = {str(row["material_id"]): {} for row in rows}
    count = len(rows)
    for property_id in property_ids:
        ordered = sorted(
            rows,
            key=lambda row: (
                (
                    -float(row["predictions"][property_id])
                    if directions[property_id] == "maximize"
                    else float(row["predictions"][property_id])
                ),
                str(row["material_id"]),
            ),
        )
        cursor = 0
        while cursor < count:
            value = float(ordered[cursor]["predictions"][property_id])
            end = cursor + 1
            while (
                end < count
                and float(ordered[end]["predictions"][property_id]) == value
            ):
                end += 1
            average_position = (cursor + end - 1) / 2.0
            percentile = 0.5 if count == 1 else 1.0 - average_position / (count - 1)
            for row in ordered[cursor:end]:
                output[str(row["material_id"])][property_id] = percentile
            cursor = end
    return output


def _dominates(
    left: dict[str, float],
    right: dict[str, float],
    directions: dict[str, str],
) -> bool:
    no_worse = True
    strictly_better = False
    for property_id, direction in directions.items():
        left_value = float(left[property_id])
        right_value = float(right[property_id])
        if direction == "maximize":
            no_worse = no_worse and left_value >= right_value
            strictly_better = strictly_better or left_value > right_value
        else:
            no_worse = no_worse and left_value <= right_value
            strictly_better = strictly_better or left_value < right_value
    return no_worse and strictly_better


def _registry_publication_bytes(snapshot: OledMaterialRegistrySnapshot) -> bytes:
    return (
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"required object is missing: {key}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required string is missing: {key}")
    return value
