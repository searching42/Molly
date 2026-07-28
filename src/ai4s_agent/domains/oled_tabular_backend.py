from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum
from typing import Any

from ai4s_agent.domains.oled_baseline_loop import OledAblationReport, OledAblationReportEntry
from ai4s_agent.domains.oled_dataset_views import (
    OledDatasetViewKind,
    OledDatasetViewReport,
    OledDatasetViewRow,
    build_oled_dataset_view,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_split_leakage import OledLeakageGuardSplitPlan, validate_oled_split_leakage


class OledTabularBaselineBackendKind(str, Enum):
    RIDGE = "tabular_ridge_sklearn"
    RANDOM_FOREST = "tabular_random_forest_sklearn"


def run_oled_tabular_baseline_backend(
    records: Iterable[OledGoldDatasetRecord],
    *,
    backend: OledTabularBaselineBackendKind | str = OledTabularBaselineBackendKind.RIDGE,
    view_kind: OledDatasetViewKind | str = OledDatasetViewKind.CURATED_DEVICE_BASELINE,
    split_plan: OledLeakageGuardSplitPlan,
    target_property_id: str = "eqe_percent",
) -> OledAblationReport:
    backend_kind = OledTabularBaselineBackendKind(backend)
    clean_target = str(target_property_id or "").strip()
    if not clean_target:
        raise ValueError("target_property_id is required")

    leakage_report = validate_oled_split_leakage(split_plan.assignments)
    if not leakage_report.is_valid:
        raise ValueError(f"invalid_split_plan:{','.join(leakage_report.error_codes)}")

    view_report = build_oled_dataset_view(
        list(records),
        view_kind=view_kind,
        target_property_id=clean_target,
    )
    if not view_report.is_valid:
        raise ValueError(f"invalid_dataset_view:{','.join(view_report.error_codes)}")

    rows_by_split = _rows_by_split(view_report.rows, split_plan)
    model_cls = _load_sklearn_model(backend_kind)
    if model_cls is None:
        return _skipped_report(
            view_report,
            backend_kind,
            rows_by_split,
            skip_reason="optional_dependency_unavailable:sklearn",
        )

    train_rows = rows_by_split.get("train", [])
    train_targets = _target_values(train_rows)
    if not train_rows or not train_targets:
        raise ValueError("split_plan_missing_numeric_train_targets")

    feature_columns = _fit_feature_columns(train_rows)
    if not feature_columns:
        raise ValueError("no_tabular_features_for_backend")

    model = _new_model(model_cls, backend_kind)
    model.fit(_feature_matrix(train_rows, feature_columns), train_targets)
    predictions_by_split = {
        split: [float(value) for value in model.predict(_feature_matrix(rows, feature_columns))]
        for split, rows in rows_by_split.items()
        if rows
    }
    split_metrics = _split_metrics(rows_by_split, predictions_by_split)
    entry = OledAblationReportEntry(
        arm_id=f"{clean_target}:{view_report.view_kind.value}:{backend_kind.value}",
        status="completed",
        record_count=len(view_report.rows),
        metrics=_primary_metrics(split_metrics),
        split_metrics=split_metrics,
        train_record_count=len(rows_by_split.get("train", [])),
        validation_record_count=len(rows_by_split.get("validation", [])),
        test_record_count=len(rows_by_split.get("test", [])),
        leakage_checked=True,
        skip_reason=None,
        notes=[
            f"backend:{backend_kind.value}",
            f"dataset_view:{view_report.view_kind.value}",
            "split_aware_tabular_evaluation",
        ],
    )
    return OledAblationReport(
        spec_id=f"oled_tabular_baseline:{view_report.view_kind.value}:{clean_target}",
        target_property_id=clean_target,
        model_backend=backend_kind.value,
        status="completed",
        entries=[entry],
        leakage_checked=True,
        metadata={
            "backend_policy": "optional_sklearn_tabular",
            "dataset_view_kind": view_report.view_kind.value,
            "dataset_view_row_count": len(view_report.rows),
            "feature_column_count": len(feature_columns),
            "feature_columns": feature_columns,
        },
    )


def _skipped_report(
    view_report: OledDatasetViewReport,
    backend: OledTabularBaselineBackendKind,
    rows_by_split: dict[str, list[OledDatasetViewRow]],
    *,
    skip_reason: str,
) -> OledAblationReport:
    return OledAblationReport(
        spec_id=f"oled_tabular_baseline:{view_report.view_kind.value}:{view_report.target_property_id}",
        target_property_id=view_report.target_property_id,
        model_backend=backend.value,
        status="backend_skipped",
        entries=[
            OledAblationReportEntry(
                arm_id=f"{view_report.target_property_id}:{view_report.view_kind.value}:{backend.value}",
                status="skipped",
                record_count=len(view_report.rows),
                train_record_count=len(rows_by_split.get("train", [])),
                validation_record_count=len(rows_by_split.get("validation", [])),
                test_record_count=len(rows_by_split.get("test", [])),
                leakage_checked=True,
                skip_reason=skip_reason,
                notes=[f"dataset_view:{view_report.view_kind.value}"],
            )
        ],
        leakage_checked=True,
        metadata={
            "backend_policy": "optional_dependency",
            "dataset_view_kind": view_report.view_kind.value,
            "dataset_view_row_count": len(view_report.rows),
        },
    )


def _rows_by_split(
    rows: list[OledDatasetViewRow],
    split_plan: OledLeakageGuardSplitPlan,
) -> dict[str, list[OledDatasetViewRow]]:
    split_by_record_id = {assignment.record_id: assignment.split for assignment in split_plan.assignments}
    rows_by_split: dict[str, list[OledDatasetViewRow]] = {}
    for row in rows:
        source_record_ids = row.source_record_ids or [row.record_id]
        row_splits = {split_by_record_id.get(record_id) for record_id in source_record_ids}
        if None in row_splits:
            missing = sorted(record_id for record_id in source_record_ids if record_id not in split_by_record_id)
            raise ValueError(f"split_plan_missing_record:{','.join(missing)}")
        clean_splits = {split for split in row_splits if split is not None}
        if len(clean_splits) != 1:
            raise ValueError(f"dataset_view_row_cross_split_sources:{row.record_id}")
        split = next(iter(clean_splits))
        rows_by_split.setdefault(split, []).append(row)
    return {
        split: sorted(split_rows, key=lambda row: row.record_id)
        for split, split_rows in rows_by_split.items()
    }


def _fit_feature_columns(rows: list[OledDatasetViewRow]) -> list[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(_flatten_features(row.features).keys())
    return sorted(columns)


def _feature_matrix(
    rows: list[OledDatasetViewRow],
    feature_columns: list[str],
) -> list[list[float]]:
    flattened_rows = [_flatten_features(row.features) for row in rows]
    return [
        [float(flattened.get(column, 0.0)) for column in feature_columns]
        for flattened in flattened_rows
    ]


def _flatten_features(features: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in sorted(features.items()):
        _flatten_value(str(key), value, flattened)
    return flattened


def _flatten_value(prefix: str, value: Any, output: dict[str, float]) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)):
        output[prefix] = float(value)
        return
    if isinstance(value, str):
        clean = " ".join(value.strip().lower().split())
        if clean:
            output[f"{prefix}={clean}"] = 1.0
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_value(f"{prefix}.{index}", item, output)
        return
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            _flatten_value(f"{prefix}.{key}", item, output)


def _target_values(rows: list[OledDatasetViewRow]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if isinstance(row.target_value, bool):
            continue
        if isinstance(row.target_value, (int, float)):
            values.append(float(row.target_value))
    return values


def _split_metrics(
    rows_by_split: dict[str, list[OledDatasetViewRow]],
    predictions_by_split: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        rows = rows_by_split.get(split, [])
        if not rows:
            continue
        y_true = _target_values(rows)
        y_pred = predictions_by_split.get(split, [])
        if y_true and y_pred:
            metrics[split] = _regression_metrics(y_true, y_pred)
    return metrics


def _primary_metrics(split_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return split_metrics.get("test") or split_metrics.get("validation") or split_metrics.get("train") or {}


def _regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    errors = [predicted - actual for actual, predicted in zip(y_true, y_pred, strict=True)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    target_mean = sum(y_true) / n
    sse = sum(squared_errors)
    sst = sum((value - target_mean) ** 2 for value in y_true)
    r2 = 0.0 if sst == 0 else 1.0 - (sse / sst)
    return {
        "bias": round(sum(errors) / n, 6),
        "mae": round(sum(absolute_errors) / n, 6),
        "prediction_mean": round(sum(y_pred) / n, 6),
        "r2": round(r2, 6),
        "rmse": round(math.sqrt(sse / n), 6),
        "target_mean": round(target_mean, 6),
    }


def _new_model(model_cls: Any, backend: OledTabularBaselineBackendKind) -> Any:
    if backend == OledTabularBaselineBackendKind.RANDOM_FOREST:
        return model_cls(n_estimators=64, random_state=0)
    return model_cls(alpha=1.0)


def _load_sklearn_model(backend: OledTabularBaselineBackendKind) -> Any | None:
    try:
        if backend == OledTabularBaselineBackendKind.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor
        from sklearn.linear_model import Ridge

        return Ridge
    except ImportError:
        return None


__all__ = [
    "OledTabularBaselineBackendKind",
    "run_oled_tabular_baseline_backend",
]
