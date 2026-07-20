from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from ai4s_agent._utils import now_iso

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
    _fit_property_model,
    _json_bytes,
    _predict_feature_vector,
    _safe_token,
    _stable_hash,
    _validated_split_by_row,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.trainability import generate_baseline_features


_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_SCREENING_VERSION = "oled_registry_candidate_screening.v1"


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


@dataclass(frozen=True)
class OledRegistryCandidateScreeningResult:
    screening_id: str
    output_dir: Path
    eligible_candidate_count: int
    excluded_candidate_count: int
    prediction_count: int
    shortlist_count: int


def run_oled_registry_candidate_screening_from_files(
    *,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    output_root: str | Path,
    minimums: list[str] | None = None,
    maximums: list[str] | None = None,
    generated_at: str | None = None,
) -> OledRegistryCandidateScreeningResult:
    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        prepared = _load_screening_inputs(
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
        )
        constraints = _parse_constraints(
            minimums=minimums or [],
            maximums=maximums or [],
            property_ids=prepared.property_ids,
        )
        eligible, excluded, raw_predictions = _screen_registry_candidates(prepared)
        if not eligible or not raw_predictions:
            raise ValueError("Registry screening has no eligible candidates")
        predictions, shortlist = _rank_candidates(
            raw_predictions,
            property_ids=prepared.property_ids,
            directions=prepared.directions,
            constraints=constraints,
        )
        config = {
            "property_ids": list(prepared.property_ids),
            "directions": prepared.directions,
            "constraints": constraints,
            "feature_policy": "exact_pr_ao_model_feature_contract",
            "scoring_policy": "pareto_then_mean_rank_percentile.v1",
        }
        screening_id = "oled-registry-screening:" + _stable_hash(
            {
                "phase1_execution_id": prepared.execution["execution_id"],
                "phase1_execution_sha256": prepared.execution_sha256,
                "dataset_snapshot_digest": prepared.dataset.execution_artifact_digest,
                "dataset_snapshot_sha256": prepared.dataset_sha256,
                "registry_snapshot_digest": prepared.registry.snapshot_digest,
                "registry_snapshot_sha256": prepared.registry_sha256,
                "config": config,
            }
        )
        payloads = _screening_payloads(
            prepared=prepared,
            screening_id=screening_id,
            config=config,
            eligible=eligible,
            excluded=excluded,
            predictions=predictions,
            shortlist=shortlist,
            generated_at=generated_at or now_iso(),
        )
        output_dir = root / screening_id
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=payloads,
            artifact_label="Registry candidate screening",
        )
    return OledRegistryCandidateScreeningResult(
        screening_id=screening_id,
        output_dir=output_dir,
        eligible_candidate_count=len(eligible),
        excluded_candidate_count=len(excluded),
        prediction_count=len(predictions),
        shortlist_count=len(shortlist),
    )


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
    try:
        alpha = float(config.get("alpha"))
    except (TypeError, ValueError) as exc:
        raise ValueError("PR-AO ridge alpha is invalid") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("PR-AO ridge alpha is invalid")

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
    expected_execution_id = "oled-real-phase1-execution:" + _stable_hash(
        {
            "source_snapshot_digest": dataset.execution_artifact_digest,
            "source_snapshot_sha256": dataset_sha,
            "config": config,
        }
    )
    if execution_id != expected_execution_id:
        raise ValueError("PR-AO execution ID mismatch")
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
            alpha=alpha,
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
    _validate_registry_identity_uniqueness(registry)

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
    alpha: float,
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
    replayed = _fit_property_model(
        [row for row in dataset.rows if row.property_id == property_id],
        split_by_row=split_by_row,
        property_id=property_id,
        alpha=alpha,
    )
    replayed.update(
        {
            "execution_id": execution_id,
            "source_dataset_snapshot_id": dataset.dataset_snapshot_id,
            "source_dataset_snapshot_digest": dataset.execution_artifact_digest,
            "source_dataset_snapshot_sha256": dataset_sha,
        }
    )
    if model != replayed:
        raise ValueError("PR-AO model deterministic replay mismatch")


def _validate_registry_identity_uniqueness(
    registry: OledMaterialRegistrySnapshot,
) -> None:
    identity_fields = (
        "canonical_isomeric_smiles",
        "standard_inchi",
        "inchikey",
    )
    for field_name in identity_fields:
        values = [getattr(entry, field_name) for entry in registry.entries]
        if len(values) != len(set(values)):
            raise ValueError("Registry chemical identity is duplicated")


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


def _screening_payloads(
    *,
    prepared: _PreparedScreeningInputs,
    screening_id: str,
    config: dict[str, Any],
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, bytes]:
    payloads = {
        "eligible_candidates.csv": _csv_bytes(
            eligible,
            [
                "material_id",
                "registry_entry_digest",
                "canonical_name",
                "canonical_isomeric_smiles",
            ],
        ),
        "excluded_candidates.jsonl": _jsonl_bytes(excluded),
        "predictions.jsonl": _jsonl_bytes(predictions),
        "ranked_shortlist.csv": _shortlist_csv_bytes(
            shortlist, prepared.property_ids
        ),
    }
    artifact_hashes = {
        name: _sha256_bytes(content) for name, content in sorted(payloads.items())
    }
    reason_counts = Counter(
        reason
        for item in excluded
        for reason in item.get("reason_codes", [])
    )
    reason_counts.update(
        reason
        for item in predictions
        for reason in item.get("decision_reason_codes", [])
    )
    receipt = {
        "screening_version": _SCREENING_VERSION,
        "screening_id": screening_id,
        "generated_at": generated_at,
        "status": "completed",
        "sources": {
            "phase1_execution_id": prepared.execution["execution_id"],
            "phase1_execution_sha256": prepared.execution_sha256,
            "dataset_snapshot_id": prepared.dataset.dataset_snapshot_id,
            "dataset_snapshot_digest": prepared.dataset.execution_artifact_digest,
            "dataset_snapshot_sha256": prepared.dataset_sha256,
            "registry_id": prepared.registry.registry_id,
            "registry_version": prepared.registry.registry_version,
            "registry_snapshot_digest": prepared.registry.snapshot_digest,
            "registry_snapshot_sha256": prepared.registry_sha256,
            "model_sha256": prepared.model_sha256,
        },
        "config": config,
        "counts": {
            "registry_candidate_count": len(prepared.registry.entries),
            "eligible_candidate_count": len(eligible),
            "excluded_candidate_count": len(excluded),
            "prediction_count": len(predictions),
            "shortlist_count": len(shortlist),
        },
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "artifacts": artifact_hashes,
        "claims": {
            "independent_registry_candidate_pool": True,
            "training_identity_exclusion_applied": True,
            "experimental_validation_claimed": False,
            "benchmark_validated": False,
            "production_ready": False,
            "model_registered": False,
            "registry_mutated": False,
        },
    }
    payloads["screening.json"] = _json_bytes(receipt)
    payloads["report.md"] = _report_bytes(receipt, shortlist)
    return payloads


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _shortlist_csv_bytes(
    rows: list[dict[str, Any]],
    property_ids: tuple[str, ...],
) -> bytes:
    fieldnames = [
        "rank",
        "material_id",
        "registry_entry_digest",
        "canonical_name",
        "canonical_isomeric_smiles",
        "aggregate_percentile",
        *[f"predicted_{property_id}" for property_id in property_ids],
    ]
    flattened = [
        {
            **{name: row.get(name, "") for name in fieldnames[:6]},
            **{
                f"predicted_{property_id}": row["predictions"][property_id]
                for property_id in property_ids
            },
        }
        for row in rows
    ]
    return _csv_bytes(flattened, fieldnames)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode("utf-8")


def _report_bytes(receipt: dict[str, Any], shortlist: list[dict[str, Any]]) -> bytes:
    counts = receipt["counts"]
    lines = [
        "# OLED Registry candidate screening",
        "",
        f"- Screening: `{receipt['screening_id']}`",
        f"- Registry candidates: `{counts['registry_candidate_count']}`",
        f"- Training-overlap/invalid exclusions: `{counts['excluded_candidate_count']}`",
        f"- Complete predictions: `{counts['prediction_count']}`",
        f"- Shortlist: `{counts['shortlist_count']}`",
        "- Experimental validation claimed: `false`",
        "- Production ready: `false`",
        "",
        "## Shortlist",
        "",
    ]
    lines.extend(
        f"- {item['rank']}. `{item['material_id']}` "
        f"(aggregate percentile={item['aggregate_percentile']:.6f})"
        for item in shortlist
    )
    lines.extend(
        [
            "",
            "This is a model-based review shortlist, not an experimental "
            "validation, benchmark, model registration, or promotion claim.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen an immutable OLED Registry with exact PR-AO models."
    )
    parser.add_argument("--phase1-execution-dir", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--registry-snapshot", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min", dest="minimums", action="append", default=[])
    parser.add_argument("--max", dest="maximums", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        result = run_oled_registry_candidate_screening_from_files(
            phase1_execution_dir=args.phase1_execution_dir,
            dataset_snapshot_json=args.dataset_snapshot,
            registry_snapshot_json=args.registry_snapshot,
            output_root=args.output_root,
            minimums=args.minimums,
            maximums=args.maximums,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "registry_candidate_screening_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "screening_id": result.screening_id,
                "eligible_candidate_count": result.eligible_candidate_count,
                "excluded_candidate_count": result.excluded_candidate_count,
                "prediction_count": result.prediction_count,
                "shortlist_count": result.shortlist_count,
                "output_directory": result.output_dir.name,
                "experimental_validation_claimed": False,
                "production_ready": False,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledRegistryCandidateScreeningResult",
    "run_oled_registry_candidate_screening_from_files",
    "main",
]
