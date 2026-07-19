from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
    OledCategoricalDatasetViewRow,
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


_MAX_SNAPSHOT_BYTES = 1024 * 1024 * 1024
_MODEL_KIND = "linear_kernel_ridge.v1"
_EXECUTION_VERSION = "oled_real_phase1_execution.v1"


@dataclass(frozen=True)
class OledRealPhase1ExecutionResult:
    execution_id: str
    output_dir: Path
    property_ids: tuple[str, ...]
    trained_model_count: int
    ranked_candidate_count: int
    source_snapshot_sha256: str


def run_oled_real_phase1_execution_from_files(
    *,
    dataset_snapshot_json: str | Path,
    output_root: str | Path,
    property_ids: list[str] | None = None,
    minimize_property_ids: set[str] | None = None,
    alpha: float = 1.0,
    generated_at: str | None = None,
) -> OledRealPhase1ExecutionResult:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    snapshot_path = _absolute_local_path(dataset_snapshot_json)
    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        payload, source_sha = _read_bound_json(
            snapshot_path,
            "categorical dataset snapshot",
            max_bytes=_MAX_SNAPSHOT_BYTES,
            reject_symlink_components=True,
        )
        snapshot = OledCategoricalDatasetExecutionArtifact.model_validate(payload)
        selected = _select_numeric_properties(snapshot, property_ids)
        directions = {
            property_id: (
                "minimize"
                if property_id in (minimize_property_ids or set())
                or _default_minimize(property_id)
                else "maximize"
            )
            for property_id in selected
        }
        config = {
            "model_kind": _MODEL_KIND,
            "alpha": float(alpha),
            "property_ids": selected,
            "directions": directions,
            "candidate_split_policy": "validation_and_test_only",
        }
        execution_id = "oled-real-phase1-execution:" + _stable_hash(
            {
                "source_snapshot_digest": snapshot.execution_artifact_digest,
                "source_snapshot_sha256": source_sha,
                "config": config,
            }
        )
        output_dir = root / execution_id
        files, candidate_count = _build_execution_payloads(
            snapshot=snapshot,
            source_sha=source_sha,
            execution_id=execution_id,
            config=config,
            generated_at=generated_at or now_iso(),
        )
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=files,
            artifact_label="real Phase 1 execution",
        )
    return OledRealPhase1ExecutionResult(
        execution_id=execution_id,
        output_dir=output_dir,
        property_ids=tuple(selected),
        trained_model_count=len(selected),
        ranked_candidate_count=candidate_count,
        source_snapshot_sha256=source_sha,
    )


def _select_numeric_properties(
    snapshot: OledCategoricalDatasetExecutionArtifact,
    requested: list[str] | None,
) -> list[str]:
    available = sorted(
        {
            row.property_id
            for row in snapshot.rows
            if _numeric(row.target_value) is not None
        }
    )
    selected = sorted(set(requested or available))
    if not selected:
        raise ValueError("real Phase 1 execution requires numeric properties")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError("requested properties are absent or non-numeric")
    return selected


def _build_execution_payloads(
    *,
    snapshot: OledCategoricalDatasetExecutionArtifact,
    source_sha: str,
    execution_id: str,
    config: dict[str, Any],
    generated_at: str,
) -> tuple[dict[str, bytes], int]:
    split_by_row = {item.row_id: item.split for item in snapshot.split_assignments}
    rows_by_property: dict[str, list[OledCategoricalDatasetViewRow]] = defaultdict(list)
    for row in snapshot.rows:
        if row.property_id in config["property_ids"]:
            rows_by_property[row.property_id].append(row)

    models: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for property_id in config["property_ids"]:
        model = _fit_property_model(
            rows_by_property[property_id],
            split_by_row=split_by_row,
            property_id=property_id,
            alpha=float(config["alpha"]),
        )
        model.update(
            {
                "execution_id": execution_id,
                "source_dataset_snapshot_id": snapshot.dataset_snapshot_id,
                "source_dataset_snapshot_digest": snapshot.execution_artifact_digest,
                "source_dataset_snapshot_sha256": source_sha,
            }
        )
        models[property_id] = model
        property_predictions = _predict_property_rows(
            rows_by_property[property_id],
            split_by_row=split_by_row,
            model=model,
        )
        predictions.extend(property_predictions)
        metrics.extend(_metrics_by_split(property_id, property_predictions))

    ranked = _rank_holdout_materials(
        predictions,
        property_ids=config["property_ids"],
        directions=config["directions"],
    )
    if not ranked:
        raise ValueError("real Phase 1 execution requires complete holdout candidates")

    files: dict[str, bytes] = {}
    for property_id, model in sorted(models.items()):
        files[f"model__{_safe_token(property_id)}.json"] = _json_bytes(model)
    files["predictions.jsonl"] = _jsonl_bytes(predictions)
    files["metrics.json"] = _json_bytes(metrics)
    files["ranked_candidates.csv"] = _ranked_csv_bytes(ranked, config["property_ids"])
    file_hashes = {
        name: f"sha256:{hashlib.sha256(content).hexdigest()}"
        for name, content in sorted(files.items())
    }
    summary = {
        "execution_version": _EXECUTION_VERSION,
        "execution_id": execution_id,
        "generated_at": generated_at,
        "status": "completed",
        "source": {
            "dataset_snapshot_id": snapshot.dataset_snapshot_id,
            "dataset_snapshot_sha256": source_sha,
            "dataset_snapshot_digest": snapshot.execution_artifact_digest,
            "paper_id": snapshot.paper_id,
            "material_group_count": snapshot.material_group_count,
        },
        "config": config,
        "results": {
            "trained_model_count": len(models),
            "prediction_count": len(predictions),
            "ranked_candidate_count": len(ranked),
            "metrics": metrics,
            "top_candidate": ranked[0],
        },
        "artifacts": file_hashes,
        "claims": {
            "real_model_fit": True,
            "holdout_only_ranking": True,
            "benchmark_validated": False,
            "production_ready": False,
            "model_registered": False,
        },
    }
    files["execution.json"] = _json_bytes(summary)
    files["report.md"] = _report_bytes(summary, ranked)
    return files, len(ranked)


def _fit_property_model(
    rows: list[OledCategoricalDatasetViewRow],
    *,
    split_by_row: dict[str, str],
    property_id: str,
    alpha: float,
) -> dict[str, Any]:
    train = sorted(
        [
            row
            for row in rows
            if split_by_row[row.row_id] == "train"
            and _numeric(row.target_value) is not None
        ],
        key=lambda row: row.row_id,
    )
    if len(train) < 2:
        raise ValueError(f"property {property_id} requires at least two train rows")
    feature_names = sorted(train[0].features)
    if not feature_names or any(sorted(row.features) != feature_names for row in rows):
        raise ValueError(f"property {property_id} feature columns are inconsistent")
    matrix = [[float(row.features[name]) for name in feature_names] for row in train]
    targets = [float(row.target_value) for row in train]  # type: ignore[arg-type]
    feature_mean = [sum(column) / len(matrix) for column in zip(*matrix, strict=True)]
    target_mean = sum(targets) / len(targets)
    centered = [
        [value - mean for value, mean in zip(vector, feature_mean, strict=True)]
        for vector in matrix
    ]
    kernel = [
        [
            sum(a * b for a, b in zip(left, right, strict=True))
            + (alpha if i == j else 0.0)
            for j, right in enumerate(centered)
        ]
        for i, left in enumerate(centered)
    ]
    dual = _solve_linear_system(kernel, [value - target_mean for value in targets])
    return {
        "model_kind": _MODEL_KIND,
        "property_id": property_id,
        "alpha": alpha,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "target_mean": target_mean,
        "training_row_ids": [row.row_id for row in train],
        "training_material_ids": [row.selected_material_id for row in train],
        "centered_training_features": centered,
        "dual_coefficients": dual,
    }


def _predict_property_rows(
    rows: list[OledCategoricalDatasetViewRow],
    *,
    split_by_row: dict[str, str],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.row_id):
        vector = [float(row.features[name]) for name in model["feature_names"]]
        centered = [
            value - mean
            for value, mean in zip(vector, model["feature_mean"], strict=True)
        ]
        kernels = [
            sum(a * b for a, b in zip(centered, train, strict=True))
            for train in model["centered_training_features"]
        ]
        predicted = float(model["target_mean"]) + sum(
            coefficient * kernel
            for coefficient, kernel in zip(
                model["dual_coefficients"], kernels, strict=True
            )
        )
        truth = _numeric(row.target_value)
        predictions.append(
            {
                "row_id": row.row_id,
                "selected_material_id": row.selected_material_id,
                "canonical_isomeric_smiles": row.canonical_isomeric_smiles,
                "property_id": row.property_id,
                "split": split_by_row[row.row_id],
                "y_true": truth,
                "y_pred": predicted,
                "residual": None if truth is None else predicted - truth,
            }
        )
    return predictions


def _metrics_by_split(
    property_id: str,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        group = [row for row in predictions if row["split"] == split]
        if not group:
            continue
        truths = [float(row["y_true"]) for row in group]
        predicted = [float(row["y_pred"]) for row in group]
        errors = [guess - truth for guess, truth in zip(predicted, truths, strict=True)]
        mean_truth = sum(truths) / len(truths)
        denominator = sum((value - mean_truth) ** 2 for value in truths)
        output.append(
            {
                "property_id": property_id,
                "split": split,
                "row_count": len(group),
                "mae": sum(abs(value) for value in errors) / len(errors),
                "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
                "r2": (
                    None
                    if len(group) < 2 or denominator == 0
                    else 1.0 - sum(value * value for value in errors) / denominator
                ),
            }
        )
    return output


def _rank_holdout_materials(
    predictions: list[dict[str, Any]],
    *,
    property_ids: list[str],
    directions: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in predictions:
        if item["split"] == "train":
            continue
        material = grouped.setdefault(
            item["selected_material_id"],
            {
                "selected_material_id": item["selected_material_id"],
                "canonical_isomeric_smiles": item["canonical_isomeric_smiles"],
                "split": item["split"],
                "predictions": {},
            },
        )
        if material["split"] != item["split"]:
            raise ValueError("material group crosses execution splits")
        material["predictions"][item["property_id"]] = item["y_pred"]
    candidates = [
        item
        for item in grouped.values()
        if set(item["predictions"]) == set(property_ids)
    ]
    for property_id in property_ids:
        values = [float(item["predictions"][property_id]) for item in candidates]
        low, high = min(values), max(values)
        for item, value in zip(candidates, values, strict=True):
            utility = 0.5 if high == low else (value - low) / (high - low)
            if directions[property_id] == "minimize":
                utility = 1.0 - utility
            item.setdefault("utilities", {})[property_id] = utility
    for item in candidates:
        item["score"] = sum(item["utilities"].values()) / len(property_ids)
    candidates.sort(key=lambda item: (-item["score"], item["selected_material_id"]))
    for rank, item in enumerate(candidates, 1):
        item["rank"] = rank
    return candidates


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [list(row) + [values[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _ranked_csv_bytes(rows: list[dict[str, Any]], properties: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = [
        "rank",
        "selected_material_id",
        "canonical_isomeric_smiles",
        "split",
        "score",
        *[f"predicted_{name}" for name in properties],
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "rank": row["rank"],
                "selected_material_id": row["selected_material_id"],
                "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                "split": row["split"],
                "score": row["score"],
                **{
                    f"predicted_{name}": row["predictions"][name]
                    for name in properties
                },
            }
        )
    return stream.getvalue().encode("utf-8")


def _report_bytes(summary: dict[str, Any], ranked: list[dict[str, Any]]) -> bytes:
    lines = [
        "# Real Phase 1 execution canary",
        "",
        f"- Execution: `{summary['execution_id']}`",
        f"- Source snapshot: `{summary['source']['dataset_snapshot_id']}`",
        f"- Models trained: `{summary['results']['trained_model_count']}`",
        f"- Holdout candidates ranked: `{len(ranked)}`",
        "- Benchmark validated: `false`",
        "- Production ready: `false`",
        "",
        "## Ranking",
        "",
    ]
    lines.extend(
        f"- {item['rank']}. `{item['selected_material_id']}` "
        f"({item['split']}, score={item['score']:.6f})"
        for item in ranked
    )
    lines.extend(
        [
            "",
            "This run fits deterministic kernel-ridge models on train-only "
            "rows and ranks only validation/test materials. The corpus is a "
            "small execution canary, not a benchmark or promotion claim.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in values
        )
        + ("\n" if values else "")
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _default_minimize(property_id: str) -> bool:
    clean = property_id.lower()
    return "delta_e_st" in clean or clean.endswith("_gap") or "energy_gap" in clean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit deterministic real models on a published OLED dataset snapshot "
            "and rank validation/test materials."
        )
    )
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--property-id", action="append", default=[])
    parser.add_argument("--minimize-property-id", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        result = run_oled_real_phase1_execution_from_files(
            dataset_snapshot_json=args.dataset_snapshot,
            output_root=args.output_root,
            property_ids=args.property_id or None,
            minimize_property_ids=set(args.minimize_property_id),
            alpha=args.alpha,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "real_phase1_execution_failed",
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
                "execution_id": result.execution_id,
                "trained_model_count": result.trained_model_count,
                "ranked_candidate_count": result.ranked_candidate_count,
                "output_directory": result.output_dir.name,
                "benchmark_validated": False,
                "production_ready": False,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OledRealPhase1ExecutionResult", "run_oled_real_phase1_execution_from_files", "main"]
