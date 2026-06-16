from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import ai4s_agent.trainability as trainability_module
from ai4s_agent.trainability import (
    BackendOverride,
    ReadinessAssessment,
    SavedBackendOverride,
    assess_3d_relevance,
    assess_trainability,
    detect_task_type,
    generate_baseline_features,
    load_model,
    predict_from_model,
    recommend_backend,
    run_baseline,
    save_backend_override,
    save_model,
    train_property_model,
)


def _write_training_csv(path: Path) -> None:
    rows = []
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "COC", "CNC", "CCCC", "CCCO"]
    for i in range(120):
        smi = smiles[i % len(smiles)]
        rows.append(
            {
                "dataset_id": f"m{i:03d}",
                "SMILES": smi,
                "split_group": "1" if i % 5 == 0 else "2",
                "plqy": f"{0.35 + (i % 20) * 0.01:.3f}",
                "lambda_em": f"{480 + (i % 30)}",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_trainability_thresholds_and_numeric_task_detection() -> None:
    report = assess_trainability(
        [
            {"property_id": "ready_prop", "label_count": 100, "numeric_ratio": 1.0},
            {"property_id": "warning_prop", "label_count": 30, "numeric_ratio": 1.0},
            {"property_id": "blocked_prop", "label_count": 29, "numeric_ratio": 1.0},
            {"property_id": "bad_labels", "label_count": 120, "numeric_ratio": 0.3},
        ]
    )

    statuses = {item.property_id: item.status for item in report.properties}
    assert statuses["ready_prop"] == "TRAIN_READY"
    assert statuses["warning_prop"] == "TRAIN_WITH_WARNING"
    assert statuses["blocked_prop"] == "INSUFFICIENT_LABELS"
    assert statuses["bad_labels"] == "INVALID_LABELS"
    assert report.overall_status == "BLOCKED"

    assert detect_task_type(["1.0", "2", "3.5"]) == "numeric_regression"
    assert detect_task_type(["active", "inactive", "active"]) == "unsupported_task_type"


def test_generate_baseline_features_records_ecfp_fallback_reason() -> None:
    features = generate_baseline_features(["CCO", "CCN"], n_bits=64)
    assert features.feature_type in {"morgan_ecfp", "hashed_ecfp_like"}
    assert features.matrix
    assert len(features.matrix) == 2
    assert len(features.matrix[0]) == 64
    if features.feature_type == "hashed_ecfp_like":
        assert features.fallback_reason == "rdkit_unavailable"


def test_run_baseline_records_split_and_metrics(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_training_csv(train_csv)

    report = run_baseline(
        train_csv,
        properties=["plqy", "lambda_em"],
        output_dir=tmp_path / "baseline",
        run_id="r1",
    )

    assert report.backend in {"xgboost", "random_forest", "random_forest_fallback"}
    assert report.split_strategy in {"scaffold_split", "random_hash_split"}
    if report.split_strategy == "random_hash_split":
        assert report.split_fallback_reason
    assert report.properties
    assert report.output_paths["baseline_report_json"].endswith("r1_baseline_report.json")
    assert Path(report.output_paths["baseline_report_json"]).exists()
    assert Path(report.output_paths["model_metrics_json"]).exists()
    assert Path(report.output_paths["predictions_val_csv"]).exists()
    for item in report.properties:
        assert item.valid_size > 0
        assert "mae" in item.metrics
        assert "rmse" in item.metrics
        assert "r2" in item.metrics


def test_run_baseline_handles_whitespace_headers(tmp_path: Path) -> None:
    train_csv = tmp_path / "train_spaced.csv"
    train_csv.write_text(
        " SMILES , split_group , plqy , lambda_em \n"
        "CCO,1,0.35,480\n"
        "CCN,2,0.40,481\n"
        "CCC,2,0.45,482\n"
        "CCCl,2,0.50,483\n"
        "CCBr,2,0.55,484\n"
        "CCF,1,0.60,485\n",
        encoding="utf-8",
    )

    report = run_baseline(
        train_csv,
        properties=["plqy", "lambda_em"],
        output_dir=tmp_path / "baseline",
        run_id="r-spaced",
    )

    assert report.properties[0].effective_labels > 0
    assert Path(report.output_paths["baseline_report_json"]).exists()


def test_run_baseline_writes_report_json_once_after_output_paths(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    _write_training_csv(train_csv)
    calls: list[tuple[Path, dict]] = []
    original_write_json = trainability_module._write_json

    def counting_write_json(path: Path, payload: dict) -> None:
        calls.append((path, payload))
        original_write_json(path, payload)

    monkeypatch.setattr(trainability_module, "_write_json", counting_write_json)

    run_baseline(
        train_csv,
        properties=["plqy"],
        output_dir=tmp_path / "baseline",
        run_id="r1",
    )

    report_calls = [payload for path, payload in calls if path.name == "r1_baseline_report.json"]
    assert len(report_calls) == 1
    assert "baseline_report_json" in report_calls[0]["output_paths"]


def test_random_forest_backend_uses_feature_dependent_model() -> None:
    model = trainability_module._fit_model(
        features=[[0.0], [0.0], [1.0], [1.0]],
        labels=[0.0, 0.0, 10.0, 10.0],
        backend="random_forest",
    )

    assert model.predict_one([0.0]) != model.predict_one([1.0])


def test_readiness_3d_relevance_override_and_backend_recommendation(tmp_path: Path) -> None:
    trainability = assess_trainability(
        [
            {"property_id": "plqy", "label_count": 80, "numeric_ratio": 1.0},
            {"property_id": "homo", "label_count": 120, "numeric_ratio": 1.0},
        ]
    )
    baseline_summary = {
        "properties": [
            {"property_id": "plqy", "metrics": {"r2": 0.25, "mae": 0.1}},
            {"property_id": "homo", "metrics": {"r2": 0.1, "mae": 0.2}},
        ]
    }

    relevance = assess_3d_relevance("homo")
    assert relevance.relevance == "high"
    assert relevance.confidence >= 0.8

    override = save_backend_override(
        workspace_dir=tmp_path,
        project_id="proj-a",
        override=BackendOverride(
            property_id="plqy",
            backend="baseline",
            reason="smoke test",
            actor="user",
        ),
    )
    assert override.path.exists()
    saved = json.loads(override.path.read_text(encoding="utf-8"))
    assert saved["overrides"][0]["backend"] == "baseline"

    recommendation = recommend_backend(
        trainability_report=trainability,
        baseline_summary=baseline_summary,
        user_intent="formal 3D run",
        overrides=[override.override],
    )

    assert recommendation.selected_backend in {"baseline", "unimol"}
    assert len(recommendation.per_property) == 2
    assert recommendation.mixed_backend_warning is True
    assert recommendation.per_property[0].property_id == "plqy"
    assert recommendation.per_property[0].recommended_backend == "baseline"

    assessment = ReadinessAssessment.from_reports(trainability, recommendation)
    assert assessment.data_readiness == "WARNING"
    assert assessment.recommendation in {"train_baseline", "train_unimol", "review_data"}


def test_save_backend_override_rejects_project_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_backend_override(
            workspace_dir=tmp_path,
            project_id="../escape",
            override=BackendOverride(
                property_id="plqy",
                backend="baseline",
                reason="bad path",
                actor="user",
            ),
        )
    assert not (tmp_path / "escape" / "backend_overrides.json").exists()


def test_saved_backend_override_requires_existing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path must exist"):
        SavedBackendOverride(
            path=tmp_path / "missing.json",
            override=BackendOverride(
                property_id="plqy",
                backend="baseline",
                reason="test",
                actor="user",
            ),
        )


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    model = trainability_module._fit_model(
        features=[[0.0], [1.0], [2.0]],
        labels=[0.0, 5.0, 10.0],
        backend="random_forest_fallback",
    )
    meta = {"run_id": "r1", "property_id": "plqy", "backend": "fallback", "train_size": 3}
    save_model(model, tmp_path / "model", metadata=meta)

    loaded_model, loaded_meta = load_model(tmp_path / "model")
    assert loaded_meta["property_id"] == "plqy"
    assert abs(loaded_model.predict_one([0.0]) - model.predict_one([0.0])) < 1e-6


def test_train_property_model_saves_and_returns_metadata(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_training_csv(train_csv)

    meta = train_property_model(
        train_csv,
        property_id="plqy",
        model_dir=tmp_path / "model_plqy",
        run_id="r1",
    )

    assert meta["run_id"] == "r1"
    assert meta["property_id"] == "plqy"
    assert meta["backend"] in {"xgboost", "random_forest", "random_forest_fallback"}
    assert meta["train_size"] > 0
    assert "mae" in meta["metrics"]
    assert (tmp_path / "model_plqy" / "model.pkl").exists()
    assert (tmp_path / "model_plqy" / "model_metadata.json").exists()


def test_predict_from_model_uses_persisted_model(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    candidate_csv = tmp_path / "candidates.csv"
    _write_training_csv(train_csv)
    _write_training_csv(candidate_csv)

    model_dir = tmp_path / "model_lambda_em"
    meta = train_property_model(
        train_csv,
        property_id="lambda_em",
        model_dir=model_dir,
        run_id="r1",
    )

    output_csv = tmp_path / "pred.csv"
    result = predict_from_model(
        model_dir,
        candidate_csv,
        output_csv=output_csv,
    )

    assert result["property_id"] == "lambda_em"
    assert result["prediction_method"].endswith("_model")
    assert output_csv.exists()
    assert result["row_count"] > 0

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))
    assert all("lambda_em_pred" in row for row in rows)
    assert all(float(row["lambda_em_pred"]) > 0 for row in rows)


def test_export_json_schemas_includes_model_metadata(tmp_path: Path) -> None:
    from ai4s_agent.schemas import export_json_schemas

    exported = export_json_schemas(tmp_path)
    names = {path.name for path in exported}
    assert "model_metadata.schema.json" in names
