from __future__ import annotations

import hashlib
import json
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
    _json_bytes,
    _safe_token,
    _validated_split_by_row,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)


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
