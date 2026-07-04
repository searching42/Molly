from __future__ import annotations

import math
from collections.abc import Iterable
from enum import Enum
from typing import Any

from ai4s_agent.domains.oled_baseline_loop import (
    OledAblationReport,
    OledAblationReportEntry,
    OledBaselineExperimentArm,
    OledBaselineExperimentSpec,
    build_oled_baseline_experiment_spec,
)
from ai4s_agent.domains.oled_feature_materialization import (
    OledFeatureMaterializationTable,
    materialize_oled_baseline_feature_table,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord


class OledBaselineBackendKind(str, Enum):
    DUMMY_MEAN = "dummy_mean"
    RIDGE_LIKE_SKLEARN = "ridge_like_sklearn"


def run_oled_baseline_backend(
    records: Iterable[OledGoldDatasetRecord],
    *,
    spec: OledBaselineExperimentSpec | None = None,
    backend: OledBaselineBackendKind | str = OledBaselineBackendKind.DUMMY_MEAN,
) -> OledAblationReport:
    backend_kind = OledBaselineBackendKind(backend)
    gold_records = list(records)
    experiment_spec = spec or build_oled_baseline_experiment_spec(gold_records)
    if backend_kind == OledBaselineBackendKind.RIDGE_LIKE_SKLEARN and _load_sklearn_ridge() is None:
        return _skipped_report(
            experiment_spec,
            backend=backend_kind,
            skip_reason="optional_dependency_unavailable:sklearn",
        )

    entries = [
        _run_arm(gold_records, experiment_spec.target_property_id, arm, backend_kind)
        for arm in experiment_spec.arms
    ]
    _attach_delta_metrics(entries)
    return OledAblationReport(
        spec_id=experiment_spec.spec_id,
        target_property_id=experiment_spec.target_property_id,
        model_backend=backend_kind.value,
        status="completed",
        entries=entries,
        metadata={"backend_policy": "lightweight_local_baseline"},
    )


def _run_arm(
    records: list[OledGoldDatasetRecord],
    target_property_id: str,
    arm: OledBaselineExperimentArm,
    backend: OledBaselineBackendKind,
) -> OledAblationReportEntry:
    table = materialize_oled_baseline_feature_table(
        records,
        feature_view=arm.feature_view,
        target_property_id=target_property_id,
    )
    arm_table = _filter_table_for_arm(table, arm)
    y_true = _target_values(arm_table)
    if backend == OledBaselineBackendKind.RIDGE_LIKE_SKLEARN:
        y_pred = _ridge_like_predictions(arm_table, arm)
    else:
        y_pred = _dummy_mean_predictions(y_true)
    return OledAblationReportEntry(
        arm_id=arm.arm_id,
        status="completed",
        record_count=len(y_true),
        metrics=_regression_metrics(y_true, y_pred),
        skip_reason=None,
        notes=[f"backend:{backend.value}"],
    )


def _filter_table_for_arm(
    table: OledFeatureMaterializationTable,
    arm: OledBaselineExperimentArm,
) -> OledFeatureMaterializationTable:
    allowed_ids = set(arm.record_ids)
    return OledFeatureMaterializationTable(
        feature_view=table.feature_view,
        target_property_id=table.target_property_id,
        rows=[row for row in table.rows if row.record_id in allowed_ids],
    )


def _target_values(table: OledFeatureMaterializationTable) -> list[float]:
    values: list[float] = []
    for row in table.rows:
        if isinstance(row.target_value, bool):
            continue
        if isinstance(row.target_value, (int, float)):
            values.append(float(row.target_value))
    if not values:
        raise ValueError(f"no_numeric_targets_for_backend:{table.target_property_id}")
    return values


def _dummy_mean_predictions(y_true: list[float]) -> list[float]:
    mean_value = sum(y_true) / len(y_true)
    return [mean_value for _ in y_true]


def _ridge_like_predictions(
    table: OledFeatureMaterializationTable,
    arm: OledBaselineExperimentArm,
) -> list[float]:
    ridge_cls = _load_sklearn_ridge()
    if ridge_cls is None:
        raise ValueError("optional_dependency_unavailable:sklearn")
    records = table.to_records()
    blocked_columns = {f"feature.{feature_name}" for feature_name in arm.blocked_features}
    feature_columns = [
        column
        for column in table.columns
        if column.startswith("feature.") and column not in blocked_columns
    ]
    numeric_columns = [
        column
        for column in feature_columns
        if all(_is_numeric_feature(record.get(column)) for record in records)
    ]
    if not numeric_columns:
        return _dummy_mean_predictions(_target_values(table))
    x_values = [
        [float(record.get(column) or 0.0) for column in numeric_columns]
        for record in records
    ]
    y_true = _target_values(table)
    model = ridge_cls(alpha=1.0)
    model.fit(x_values, y_true)
    return [float(value) for value in model.predict(x_values)]


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


def _attach_delta_metrics(entries: list[OledAblationReportEntry]) -> None:
    full_context = next((entry for entry in entries if entry.arm_id.endswith(":full_context")), None)
    if full_context is None:
        return
    full_mae = full_context.metrics.get("mae", 0.0)
    full_rmse = full_context.metrics.get("rmse", 0.0)
    for entry in entries:
        entry.delta_metrics = {
            "mae_delta_vs_full_context": round(entry.metrics.get("mae", 0.0) - full_mae, 6),
            "rmse_delta_vs_full_context": round(entry.metrics.get("rmse", 0.0) - full_rmse, 6),
        }


def _skipped_report(
    spec: OledBaselineExperimentSpec,
    *,
    backend: OledBaselineBackendKind,
    skip_reason: str,
) -> OledAblationReport:
    return OledAblationReport(
        spec_id=spec.spec_id,
        target_property_id=spec.target_property_id,
        model_backend=backend.value,
        status="backend_skipped",
        entries=[
            OledAblationReportEntry(
                arm_id=arm.arm_id,
                status="skipped",
                record_count=len(arm.record_ids),
                skip_reason=skip_reason,
            )
            for arm in spec.arms
        ],
        metadata={"backend_policy": "optional_dependency"},
    )


def _is_numeric_feature(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_sklearn_ridge() -> Any | None:
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        return None
    return Ridge


__all__ = [
    "OledBaselineBackendKind",
    "run_oled_baseline_backend",
]
