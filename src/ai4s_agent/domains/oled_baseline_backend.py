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
    OledFeatureMaterializationRow,
    OledFeatureMaterializationTable,
    materialize_oled_baseline_feature_table,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_split_leakage import OledLeakageGuardSplitPlan, validate_oled_split_leakage


class OledBaselineBackendKind(str, Enum):
    DUMMY_MEAN = "dummy_mean"
    RIDGE_LIKE_SKLEARN = "ridge_like_sklearn"


def run_oled_baseline_backend(
    records: Iterable[OledGoldDatasetRecord],
    *,
    spec: OledBaselineExperimentSpec | None = None,
    backend: OledBaselineBackendKind | str = OledBaselineBackendKind.DUMMY_MEAN,
    split_plan: OledLeakageGuardSplitPlan | None = None,
) -> OledAblationReport:
    backend_kind = OledBaselineBackendKind(backend)
    gold_records = list(records)
    experiment_spec = spec or build_oled_baseline_experiment_spec(gold_records)
    leakage_checked = split_plan is not None
    if split_plan is not None:
        leakage_report = validate_oled_split_leakage(split_plan.assignments)
        if not leakage_report.is_valid:
            raise ValueError(f"invalid_split_plan:{','.join(leakage_report.error_codes)}")
    if backend_kind == OledBaselineBackendKind.RIDGE_LIKE_SKLEARN and _load_sklearn_ridge() is None:
        return _skipped_report(
            experiment_spec,
            backend=backend_kind,
            skip_reason="optional_dependency_unavailable:sklearn",
            split_plan=split_plan,
            leakage_checked=leakage_checked,
        )

    entries = [
        _run_arm(gold_records, experiment_spec.target_property_id, arm, backend_kind, split_plan)
        for arm in experiment_spec.arms
    ]
    _attach_delta_metrics(entries)
    return OledAblationReport(
        spec_id=experiment_spec.spec_id,
        target_property_id=experiment_spec.target_property_id,
        model_backend=backend_kind.value,
        status="completed",
        entries=entries,
        leakage_checked=leakage_checked,
        metadata={"backend_policy": "lightweight_local_baseline"},
    )


def _run_arm(
    records: list[OledGoldDatasetRecord],
    target_property_id: str,
    arm: OledBaselineExperimentArm,
    backend: OledBaselineBackendKind,
    split_plan: OledLeakageGuardSplitPlan | None,
) -> OledAblationReportEntry:
    table = materialize_oled_baseline_feature_table(
        records,
        feature_view=arm.feature_view,
        target_property_id=target_property_id,
    )
    arm_table = _filter_table_for_arm(table, arm)
    if split_plan is not None:
        return _run_split_aware_arm(arm_table, arm, backend, split_plan)

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


def _run_split_aware_arm(
    table: OledFeatureMaterializationTable,
    arm: OledBaselineExperimentArm,
    backend: OledBaselineBackendKind,
    split_plan: OledLeakageGuardSplitPlan,
) -> OledAblationReportEntry:
    rows_by_split = _rows_by_split(table, split_plan)
    train_rows = rows_by_split.get("train", [])
    if not train_rows:
        raise ValueError("split_plan_missing_train_records")

    split_predictions = (
        _split_ridge_like_predictions(table, arm, rows_by_split)
        if backend == OledBaselineBackendKind.RIDGE_LIKE_SKLEARN
        else _split_dummy_mean_predictions(rows_by_split)
    )
    split_metrics = _split_metrics(rows_by_split, split_predictions)
    primary_metrics = (
        split_metrics.get("test")
        or split_metrics.get("validation")
        or split_metrics.get("train")
        or {}
    )
    return OledAblationReportEntry(
        arm_id=arm.arm_id,
        status="completed",
        record_count=sum(len(rows) for rows in rows_by_split.values()),
        metrics=primary_metrics,
        split_metrics=split_metrics,
        train_record_count=len(train_rows),
        validation_record_count=len(rows_by_split.get("validation", [])),
        test_record_count=len(rows_by_split.get("test", [])),
        leakage_checked=True,
        skip_reason=None,
        notes=[f"backend:{backend.value}", "split_aware_evaluation"],
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


def _target_values_for_rows(rows: list[OledFeatureMaterializationRow]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if isinstance(row.target_value, bool):
            continue
        if isinstance(row.target_value, (int, float)):
            values.append(float(row.target_value))
    return values


def _dummy_mean_predictions(y_true: list[float]) -> list[float]:
    mean_value = sum(y_true) / len(y_true)
    return [mean_value for _ in y_true]


def _split_dummy_mean_predictions(
    rows_by_split: dict[str, list[OledFeatureMaterializationRow]],
) -> dict[str, list[float]]:
    train_values = _target_values_for_rows(rows_by_split.get("train", []))
    if not train_values:
        raise ValueError("split_plan_missing_numeric_train_targets")
    train_mean = sum(train_values) / len(train_values)
    return {
        split: [train_mean for _ in _target_values_for_rows(rows)]
        for split, rows in rows_by_split.items()
        if rows
    }


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


def _split_ridge_like_predictions(
    table: OledFeatureMaterializationTable,
    arm: OledBaselineExperimentArm,
    rows_by_split: dict[str, list[OledFeatureMaterializationRow]],
) -> dict[str, list[float]]:
    ridge_cls = _load_sklearn_ridge()
    if ridge_cls is None:
        raise ValueError("optional_dependency_unavailable:sklearn")
    train_rows = rows_by_split.get("train", [])
    train_values = _target_values_for_rows(train_rows)
    if not train_values:
        raise ValueError("split_plan_missing_numeric_train_targets")
    numeric_columns = _numeric_feature_columns(table, arm)
    if not numeric_columns:
        return _split_dummy_mean_predictions(rows_by_split)
    model = ridge_cls(alpha=1.0)
    model.fit(_feature_matrix(train_rows, numeric_columns), train_values)
    return {
        split: [float(value) for value in model.predict(_feature_matrix(rows, numeric_columns))]
        for split, rows in rows_by_split.items()
        if rows
    }


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


def _split_metrics(
    rows_by_split: dict[str, list[OledFeatureMaterializationRow]],
    predictions_by_split: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        rows = rows_by_split.get(split, [])
        if not rows:
            continue
        y_true = _target_values_for_rows(rows)
        y_pred = predictions_by_split.get(split, [])
        if not y_true or not y_pred:
            continue
        metrics[split] = _regression_metrics(y_true, y_pred)
    return metrics


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
    split_plan: OledLeakageGuardSplitPlan | None = None,
    leakage_checked: bool = False,
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
                train_record_count=_split_record_count(arm.record_ids, split_plan, "train"),
                validation_record_count=_split_record_count(arm.record_ids, split_plan, "validation"),
                test_record_count=_split_record_count(arm.record_ids, split_plan, "test"),
                leakage_checked=leakage_checked,
                skip_reason=skip_reason,
            )
            for arm in spec.arms
        ],
        leakage_checked=leakage_checked,
        metadata={"backend_policy": "optional_dependency"},
    )


def _is_numeric_feature(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rows_by_split(
    table: OledFeatureMaterializationTable,
    split_plan: OledLeakageGuardSplitPlan,
) -> dict[str, list[OledFeatureMaterializationRow]]:
    split_by_record_id = {assignment.record_id: assignment.split for assignment in split_plan.assignments}
    rows_by_split: dict[str, list[OledFeatureMaterializationRow]] = {}
    for row in table.rows:
        try:
            split = split_by_record_id[row.record_id]
        except KeyError as exc:
            raise ValueError(f"split_plan_missing_record:{row.record_id}") from exc
        rows_by_split.setdefault(split, []).append(row)
    return rows_by_split


def _split_record_count(
    record_ids: list[str],
    split_plan: OledLeakageGuardSplitPlan | None,
    split: str,
) -> int:
    if split_plan is None:
        return 0
    split_by_record_id = {assignment.record_id: assignment.split for assignment in split_plan.assignments}
    return sum(1 for record_id in record_ids if split_by_record_id.get(record_id) == split)


def _numeric_feature_columns(
    table: OledFeatureMaterializationTable,
    arm: OledBaselineExperimentArm,
) -> list[str]:
    records = table.to_records()
    blocked_columns = {f"feature.{feature_name}" for feature_name in arm.blocked_features}
    feature_columns = [
        column
        for column in table.columns
        if column.startswith("feature.") and column not in blocked_columns
    ]
    return [
        column
        for column in feature_columns
        if all(_is_numeric_feature(record.get(column)) for record in records)
    ]


def _feature_matrix(
    rows: list[OledFeatureMaterializationRow],
    numeric_columns: list[str],
) -> list[list[float]]:
    records = [row.to_record() for row in rows]
    return [
        [float(record.get(column) or 0.0) for column in numeric_columns]
        for record in records
    ]


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
