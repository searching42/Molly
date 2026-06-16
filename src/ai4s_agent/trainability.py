from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ai4s_agent._utils import hash01, now_iso, read_csv_dict_rows, safe_float, write_json
from ai4s_agent.storage import ProjectStorage

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
except Exception:
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    MurckoScaffold = None  # type: ignore


class TrainabilityProperty(BaseModel):
    property_id: str
    effective_labels: int
    numeric_ratio: float
    task_type: str
    status: str
    reason: str


class TrainabilityReport(BaseModel):
    overall_status: str
    properties: list[TrainabilityProperty]


class BaselineFeatures(BaseModel):
    feature_type: str
    matrix: list[list[float]]
    n_bits: int
    fallback_reason: str = ""


class BaselinePropertyReport(BaseModel):
    property_id: str
    status: str
    effective_labels: int
    train_size: int
    valid_size: int
    metrics: dict[str, float]


class BaselineReport(BaseModel):
    run_id: str
    backend: str
    feature_type: str
    split_strategy: str
    split_fallback_reason: str
    properties: list[BaselinePropertyReport]
    output_paths: dict[str, str] = Field(default_factory=dict)


class ThreeDRelevance(BaseModel):
    property_id: str
    relevance: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class BackendOverride(BaseModel):
    property_id: str
    backend: str
    reason: str
    actor: str


class SavedBackendOverride(BaseModel):
    path: Path
    override: BackendOverride

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError("path must exist")
        return value


class PropertyBackendRecommendation(BaseModel):
    property_id: str
    recommended_backend: str
    recommendation: str
    reason: str
    trainability_status: str
    three_d_relevance: str
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    override_applied: bool = False


class BackendRecommendation(BaseModel):
    selected_backend: str
    per_property: list[PropertyBackendRecommendation]
    mixed_backend_warning: bool
    warnings: list[str] = Field(default_factory=list)


class ReadinessAssessment(BaseModel):
    data_readiness: str
    model_readiness: str
    recommendation: str

    @classmethod
    def from_reports(
        cls,
        trainability_report: TrainabilityReport,
        backend_recommendation: BackendRecommendation,
    ) -> "ReadinessAssessment":
        if trainability_report.overall_status == "BLOCKED":
            data_readiness = "BLOCKED"
            recommendation = "review_data"
        elif trainability_report.overall_status == "WARNING":
            data_readiness = "WARNING"
            recommendation = (
                "train_unimol"
                if backend_recommendation.selected_backend == "unimol"
                else "train_baseline"
            )
        else:
            data_readiness = "READY"
            recommendation = (
                "train_unimol"
                if backend_recommendation.selected_backend == "unimol"
                else "train_baseline"
            )

        weak_count = 0
        evaluated_count = 0
        for item in backend_recommendation.per_property:
            if "r2" in item.baseline_metrics:
                evaluated_count += 1
                if float(item.baseline_metrics.get("r2", 0.0)) < 0.2:
                    weak_count += 1
        if evaluated_count == 0:
            model_readiness = "NOT_EVALUATED"
        elif weak_count == evaluated_count:
            model_readiness = "WEAK"
        elif weak_count > 0:
            model_readiness = "UNCERTAIN"
        else:
            model_readiness = "PROMISING"

        return cls(
            data_readiness=data_readiness,
            model_readiness=model_readiness,
            recommendation=recommendation,
        )


def detect_task_type(values: list[Any]) -> str:
    nonempty = [value for value in values if str(value or "").strip()]
    if not nonempty:
        return "unsupported_task_type"
    numeric = [safe_float(value) for value in nonempty]
    numeric_count = sum(1 for value in numeric if value is not None)
    return "numeric_regression" if numeric_count / len(nonempty) >= 0.8 else "unsupported_task_type"


def assess_trainability(property_stats: list[dict[str, Any]]) -> TrainabilityReport:
    properties: list[TrainabilityProperty] = []
    for item in property_stats:
        prop = str(item.get("property_id") or "").strip()
        labels = int(item.get("label_count", item.get("effective_labels", 0)) or 0)
        numeric_ratio = float(item.get("numeric_ratio", 1.0) or 0.0)
        task_type = str(item.get("task_type") or "numeric_regression")

        if task_type != "numeric_regression":
            status = "UNSUPPORTED_TASK_TYPE"
            reason = "UNSUPPORTED_TASK_TYPE"
        elif numeric_ratio < 0.6:
            status = "INVALID_LABELS"
            reason = "INVALID_LABELS"
        elif labels >= 100:
            status = "TRAIN_READY"
            reason = "TRAIN_READY"
        elif labels >= 30:
            status = "TRAIN_WITH_WARNING"
            reason = "LOW_LABEL_COUNT"
        else:
            status = "INSUFFICIENT_LABELS"
            reason = "INSUFFICIENT_LABELS"
        properties.append(
            TrainabilityProperty(
                property_id=prop,
                effective_labels=labels,
                numeric_ratio=numeric_ratio,
                task_type=task_type,
                status=status,
                reason=reason,
            )
        )

    if not properties:
        overall = "BLOCKED"
    elif any(item.status in {"INSUFFICIENT_LABELS", "INVALID_LABELS", "UNSUPPORTED_TASK_TYPE"} for item in properties):
        overall = "BLOCKED"
    elif any(item.status == "TRAIN_WITH_WARNING" for item in properties):
        overall = "WARNING"
    else:
        overall = "READY"
    return TrainabilityReport(overall_status=overall, properties=properties)


def generate_baseline_features(smiles_list: list[str], *, n_bits: int = 256, radius: int = 2) -> BaselineFeatures:
    if Chem is not None and AllChem is not None:
        matrix: list[list[float]] = []
        invalid = 0
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(str(smiles or ""))  # type: ignore[union-attr]
            if mol is None:
                invalid += 1
                matrix.append([0.0] * n_bits)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # type: ignore[union-attr]
            matrix.append([float(fp.GetBit(i)) for i in range(n_bits)])
        return BaselineFeatures(
            feature_type="morgan_ecfp",
            matrix=matrix,
            n_bits=n_bits,
            fallback_reason="invalid_smiles_zero_vectors" if invalid else "",
        )

    matrix = [_hashed_ecfp_like(str(smiles or ""), n_bits=n_bits, radius=radius) for smiles in smiles_list]
    return BaselineFeatures(
        feature_type="hashed_ecfp_like",
        matrix=matrix,
        n_bits=n_bits,
        fallback_reason="rdkit_unavailable",
    )


def _hashed_ecfp_like(smiles: str, *, n_bits: int, radius: int) -> list[float]:
    bits = [0.0] * n_bits
    token = smiles.strip()
    if not token:
        return bits
    for width in range(1, max(1, radius) + 3):
        for i in range(0, max(1, len(token) - width + 1)):
            chunk = token[i : i + width]
            digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            bits[int(digest[:8], 16) % n_bits] = 1.0
    return bits


def run_baseline(
    csv_path: Path,
    *,
    properties: list[str],
    output_dir: Path,
    run_id: str,
    smiles_col: str = "SMILES",
    split_col: str = "split_group",
    n_bits: int = 256,
) -> BaselineReport:
    rows = _read_rows(csv_path)
    smiles = [str(row.get(smiles_col, "") or "") for row in rows]
    features = generate_baseline_features(smiles, n_bits=n_bits)
    split = _make_split(rows, smiles_col=smiles_col, split_col=split_col)
    backend = _select_backend()

    property_reports: list[BaselinePropertyReport] = []
    prediction_rows: list[dict[str, Any]] = []
    for prop in properties:
        labels: list[tuple[int, float]] = []
        for idx, row in enumerate(rows):
            value = safe_float(row.get(prop))
            if value is not None:
                labels.append((idx, value))
        if len(labels) < 5:
            property_reports.append(
                BaselinePropertyReport(
                    property_id=prop,
                    status="UNAVAILABLE",
                    effective_labels=len(labels),
                    train_size=0,
                    valid_size=0,
                    metrics={},
                )
            )
            continue

        train_indices = [idx for idx, _ in labels if idx in split["train"]]
        valid_indices = [idx for idx, _ in labels if idx in split["valid"]]
        if not train_indices or not valid_indices:
            fallback = _random_hash_split(rows, smiles_col=smiles_col)
            split = {
                **fallback,
                "strategy": "random_hash_split",
                "fallback_reason": "provided_split_unusable_for_property",
            }
            train_indices = [idx for idx, _ in labels if idx in split["train"]]
            valid_indices = [idx for idx, _ in labels if idx in split["valid"]]

        label_map = {idx: value for idx, value in labels}
        model = _fit_model(
            [features.matrix[idx] for idx in train_indices],
            [label_map[idx] for idx in train_indices],
            backend=backend,
        )
        truth = [label_map[idx] for idx in valid_indices]
        preds = [float(model.predict_one(features.matrix[idx])) for idx in valid_indices]
        metrics = _metrics(truth, preds)
        property_reports.append(
            BaselinePropertyReport(
                property_id=prop,
                status="OK",
                effective_labels=len(labels),
                train_size=len(train_indices),
                valid_size=len(valid_indices),
                metrics=metrics,
            )
        )
        for idx, pred, true_value in zip(valid_indices, preds, truth):
            prediction_rows.append(
                {
                    "property_id": prop,
                    "dataset_id": rows[idx].get("dataset_id", str(idx)),
                    "truth": true_value,
                    "pred": pred,
                    "error": true_value - pred,
                }
            )

    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = BaselineReport(
        run_id=run_id,
        backend=backend,
        feature_type=features.feature_type,
        split_strategy=str(split["strategy"]),
        split_fallback_reason=str(split.get("fallback_reason", "")) or features.fallback_reason,
        properties=property_reports,
    )
    report_json = out_dir / f"{run_id}_baseline_report.json"
    metrics_json = out_dir / f"{run_id}_model_metrics.json"
    pred_csv = out_dir / f"{run_id}_predictions_val.csv"
    _write_json(metrics_json, {"properties": [item.model_dump(mode="json") for item in property_reports]})
    _write_predictions(pred_csv, prediction_rows)
    report.output_paths.update(
        {
            "baseline_report_json": str(report_json),
            "model_metrics_json": str(metrics_json),
            "predictions_val_csv": str(pred_csv),
        }
    )
    _write_json(report_json, report.model_dump(mode="json") | {"generated_at": now_iso()})
    return report


def _read_rows(path: Path) -> list[dict[str, str]]:
    rows, _ = read_csv_dict_rows(path.expanduser().resolve())
    return rows


def _make_split(rows: list[dict[str, str]], *, smiles_col: str, split_col: str) -> dict[str, Any]:
    if Chem is not None and MurckoScaffold is not None:
        scaffolds: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            mol = Chem.MolFromSmiles(str(row.get(smiles_col, "") or ""))  # type: ignore[union-attr]
            if mol is None:
                return _random_hash_split(rows, smiles_col=smiles_col) | {
                    "strategy": "random_hash_split",
                    "fallback_reason": "scaffold_generation_failed",
                }
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)  # type: ignore[union-attr]
            scaffolds.setdefault(scaffold or str(row.get(smiles_col, "")), []).append(idx)
        train: set[int] = set()
        valid: set[int] = set()
        for scaffold, indices in sorted(scaffolds.items()):
            target = valid if hash01(scaffold) < 0.2 else train
            target.update(indices)
        if train and valid:
            return {"strategy": "scaffold_split", "fallback_reason": "", "train": train, "valid": valid}
    return _random_hash_split(rows, smiles_col=smiles_col) | {
        "strategy": "random_hash_split",
        "fallback_reason": "rdkit_unavailable_for_scaffold_split",
    }


def _random_hash_split(rows: list[dict[str, str]], *, smiles_col: str) -> dict[str, Any]:
    train: set[int] = set()
    valid: set[int] = set()
    for idx, row in enumerate(rows):
        key = str(row.get("dataset_id") or row.get(smiles_col) or idx)
        if hash01(key) < 0.2:
            valid.add(idx)
        else:
            train.add(idx)
    if not valid and train:
        first = min(train)
        train.remove(first)
        valid.add(first)
    return {"train": train, "valid": valid}


def _select_backend() -> str:
    if importlib.util.find_spec("xgboost") is not None:
        return "xgboost"
    if importlib.util.find_spec("sklearn") is not None:
        return "random_forest"
    return "random_forest_fallback"


class _FallbackForest:
    def __init__(self, trees: list[tuple[int, float, float, float]], default: float) -> None:
        self.trees = trees
        self.default = default

    def predict_one(self, features: list[float]) -> float:
        if not self.trees:
            return self.default
        total = 0.0
        for feature_idx, threshold, left_mean, right_mean in self.trees:
            value = features[feature_idx] if feature_idx < len(features) else 0.0
            total += left_mean if value <= threshold else right_mean
        return total / len(self.trees)


class _ExternalRegressor:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_one(self, features: list[float]) -> float:
        return float(self.model.predict([features])[0])


def _fit_model(features: list[list[float]], labels: list[float], *, backend: str) -> _FallbackForest | _ExternalRegressor:
    if not labels:
        return _FallbackForest([], 0.0)
    default = sum(labels) / len(labels)
    external = _fit_external_model(features, labels, backend=backend)
    if external is not None:
        return external

    n_features = len(features[0]) if features else 0
    trees: list[tuple[int, float, float, float]] = []
    for seed in range(min(32, max(1, n_features))):
        feature_idx = int(hashlib.sha1(str(seed).encode("utf-8")).hexdigest()[:8], 16) % n_features
        values = [row[feature_idx] for row in features]
        threshold = sum(values) / len(values)
        left = [label for value, label in zip(values, labels) if value <= threshold]
        right = [label for value, label in zip(values, labels) if value > threshold]
        if not left or not right:
            continue
        trees.append((feature_idx, threshold, sum(left) / len(left), sum(right) / len(right)))
    return _FallbackForest(trees, default)


def _fit_external_model(
    features: list[list[float]],
    labels: list[float],
    *,
    backend: str,
) -> _ExternalRegressor | None:
    if backend == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except Exception:
            return None
        model = RandomForestRegressor(n_estimators=64, random_state=0, min_samples_leaf=1)
        model.fit(features, labels)
        return _ExternalRegressor(model)
    if backend == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception:
            return None
        model = XGBRegressor(n_estimators=64, random_state=0, max_depth=4, objective="reg:squarederror")
        model.fit(features, labels)
        return _ExternalRegressor(model)
    return None


def _metrics(truth: list[float], preds: list[float]) -> dict[str, float]:
    if not truth:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    errors = [t - p for t, p in zip(truth, preds)]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    mean_truth = sum(truth) / len(truth)
    ss_tot = sum((value - mean_truth) ** 2 for value in truth)
    ss_res = sum((t - p) ** 2 for t, p in zip(truth, preds))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["property_id", "dataset_id", "truth", "pred", "error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_model(model: _FallbackForest | _ExternalRegressor, model_dir: Path, *, metadata: dict[str, Any]) -> Path:
    import pickle

    model_dir = model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.pkl"
    with model_file.open("wb") as f:
        pickle.dump(model, f)
    write_json(model_dir / "model_metadata.json", metadata)
    return model_file


def load_model(model_dir: Path) -> tuple[_FallbackForest | _ExternalRegressor, dict[str, Any]]:
    import pickle

    model_dir = model_dir.expanduser().resolve()
    model_file = model_dir / "model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"model file not found: {model_file}")
    with model_file.open("rb") as f:
        model = pickle.load(f)
    metadata_path = model_dir / "model_metadata.json"
    meta: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            pass
    return model, meta


def train_property_model(
    csv_path: Path,
    *,
    property_id: str,
    model_dir: Path,
    run_id: str,
    smiles_col: str = "SMILES",
    split_col: str = "split_group",
    n_bits: int = 256,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    rows = _read_rows(csv_path)
    smiles = [str(row.get(smiles_col, "") or "") for row in rows]
    features = generate_baseline_features(smiles, n_bits=n_bits)
    split = _make_split(rows, smiles_col=smiles_col, split_col=split_col)
    backend = _select_backend()

    labels: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = safe_float(row.get(property_id))
        if value is not None:
            labels.append((idx, value))
    if len(labels) < 5:
        raise ValueError(f"not enough labels for {property_id}: {len(labels)}")

    train_indices = [idx for idx, _ in labels if idx in split["train"]]
    valid_indices = [idx for idx, _ in labels if idx in split["valid"]]
    if not train_indices or not valid_indices:
        fallback = _random_hash_split(rows, smiles_col=smiles_col)
        split = {
            **fallback,
            "strategy": "random_hash_split",
            "fallback_reason": "provided_split_unusable",
        }
        train_indices = [idx for idx, _ in labels if idx in split["train"]]
        valid_indices = [idx for idx, _ in labels if idx in split["valid"]]

    label_map = {idx: value for idx, value in labels}
    model = _fit_model(
        [features.matrix[idx] for idx in train_indices],
        [label_map[idx] for idx in train_indices],
        backend=backend,
    )
    truth = [label_map[idx] for idx in valid_indices]
    preds = [float(model.predict_one(features.matrix[idx])) for idx in valid_indices]
    metrics = _metrics(truth, preds)

    metadata = {
        "run_id": run_id,
        "property_id": property_id,
        "backend": backend,
        "feature_type": features.feature_type,
        "model_type": "sklearn" if backend in {"random_forest", "xgboost"} else "fallback",
        "model_dir": str(model_dir),
        "model_file": str(model_dir / "model.pkl"),
        "model_path": str(model_dir / "model.pkl"),
        "version": model_dir.name,
        "train_size": len(train_indices),
        "valid_size": len(valid_indices),
        "created_at": now_iso(),
        "metrics": metrics,
        "split_strategy": str(split["strategy"]),
        "split_fallback_reason": str(split.get("fallback_reason", "")) or features.fallback_reason,
        "n_bits": n_bits,
    }
    save_model(model, model_dir, metadata=metadata)
    return metadata


def predict_from_model(
    model_dir: Path,
    candidate_csv: Path,
    *,
    output_csv: Path,
    property_id: str | None = None,
    smiles_col: str = "SMILES",
) -> dict[str, Any]:
    model, meta = load_model(model_dir)
    prop = property_id or str(meta.get("property_id") or "")
    rows, headers, _ = _read_rows_for_prediction(candidate_csv)
    smiles_candidates = detect_smiles_column_for_prediction(headers)
    actual_smiles_col = smiles_col if smiles_col in headers else (smiles_candidates or smiles_col)
    n_bits = int(meta.get("n_bits", 256))
    smiles_list = [str(row.get(actual_smiles_col, "") or "") for row in rows]
    features = generate_baseline_features(smiles_list, n_bits=n_bits)

    pred_col = f"{prop}_pred"
    out_headers = list(headers)
    if pred_col not in out_headers:
        out_headers.append(pred_col)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = dict(row)
        pred = float(model.predict_one(features.matrix[idx]))
        out[pred_col] = round(pred, 8)
        out_rows.append(out)

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "property_id": prop,
        "prediction_method": f"{meta.get('backend', 'unknown')}_model",
        "prediction_column": pred_col,
        "output_csv": str(output_csv),
        "row_count": len(out_rows),
        "model_metadata": meta,
    }


def _read_rows_for_prediction(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    rows, headers = read_csv_dict_rows(path.expanduser().resolve())
    return rows, headers, ","


def detect_smiles_column_for_prediction(headers: list[str]) -> str:
    normalized = {"".join(ch.lower() for ch in str(h) if ch.isalnum()): h for h in headers}
    for key in ["smiles", "canonicalsmiles", "molsmiles", "structure", "chromophore"]:
        if key in normalized:
            return normalized[key]
    return ""


def assess_3d_relevance(property_id: str, *, description: str = "", unit: str = "") -> ThreeDRelevance:
    text = f"{property_id} {description} {unit}".lower()
    high_terms = ["homo", "lumo", "dipole", "binding", "steric", "electrostatic", "conformer", "energy"]
    medium_terms = ["lambda", "emission", "plqy", "gap", "bandgap"]
    if any(term in text for term in high_terms):
        return ThreeDRelevance(
            property_id=property_id,
            relevance="high",
            confidence=0.9,
            evidence=["property_name_matches_3d_sensitive_ontology"],
        )
    if any(term in text for term in medium_terms):
        return ThreeDRelevance(
            property_id=property_id,
            relevance="medium",
            confidence=0.7,
            evidence=["property_name_matches_photophysical_or_electronic_ontology"],
        )
    if any(term in text for term in ["mw", "logp", "count", "topology"]):
        return ThreeDRelevance(
            property_id=property_id,
            relevance="low",
            confidence=0.65,
            evidence=["property_name_matches_topology_dominated_ontology"],
        )
    return ThreeDRelevance(property_id=property_id, relevance="unknown", confidence=0.4, evidence=[])


def save_backend_override(
    *,
    workspace_dir: Path,
    project_id: str,
    override: BackendOverride,
) -> SavedBackendOverride:
    project_dir = ProjectStorage(workspace_dir=workspace_dir).project_dir(project_id)
    path = project_dir / "backend_overrides.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            payload = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            payload = {}
    overrides = payload.get("overrides", [])
    if not isinstance(overrides, list):
        overrides = []
    overrides.append(override.model_dump(mode="json") | {"saved_at": now_iso()})
    write_json(path, {"overrides": overrides})
    return SavedBackendOverride(path=path, override=override)


def recommend_backend(
    *,
    trainability_report: TrainabilityReport,
    baseline_summary: dict[str, Any],
    user_intent: str = "",
    overrides: list[BackendOverride] | None = None,
) -> BackendRecommendation:
    override_map = {override.property_id: override for override in (overrides or [])}
    metrics_by_prop: dict[str, dict[str, float]] = {}
    for item in baseline_summary.get("properties", []):
        if isinstance(item, dict) and item.get("property_id"):
            metrics = item.get("metrics", {})
            metrics_by_prop[str(item["property_id"])] = metrics if isinstance(metrics, dict) else {}

    intent = user_intent.lower()
    prefer_unimol = any(token in intent for token in ["formal", "3d", "unimol", "reliable", "production"])
    prefer_baseline = any(token in intent for token in ["smoke", "quick", "test"])

    per_property: list[PropertyBackendRecommendation] = []
    for prop in trainability_report.properties:
        relevance = assess_3d_relevance(prop.property_id)
        override = override_map.get(prop.property_id)
        if override is not None:
            backend = override.backend
            reason = f"user_override:{override.reason}"
            override_applied = True
        elif prop.status in {"INSUFFICIENT_LABELS", "INVALID_LABELS", "UNSUPPORTED_TASK_TYPE"}:
            backend = "none"
            reason = prop.reason
            override_applied = False
        elif prefer_baseline:
            backend = "baseline"
            reason = "user_intent_smoke_or_test"
            override_applied = False
        elif prefer_unimol or relevance.relevance == "high":
            backend = "unimol"
            reason = "3d_relevance_or_user_intent"
            override_applied = False
        elif relevance.relevance == "medium" and prop.effective_labels < 100:
            backend = "unimol"
            reason = "medium_3d_relevance_with_limited_labels"
            override_applied = False
        else:
            backend = "baseline"
            reason = "baseline_sufficient_initial_route"
            override_applied = False
        recommendation = {
            "baseline": "train_baseline",
            "unimol": "train_unimol",
            "none": "review_data",
        }.get(backend, "review_data")
        per_property.append(
            PropertyBackendRecommendation(
                property_id=prop.property_id,
                recommended_backend=backend,
                recommendation=recommendation,
                reason=reason,
                trainability_status=prop.status,
                three_d_relevance=relevance.relevance,
                baseline_metrics=metrics_by_prop.get(prop.property_id, {}),
                override_applied=override_applied,
            )
        )

    backends = [item.recommended_backend for item in per_property if item.recommended_backend in {"baseline", "unimol"}]
    if backends:
        selected_backend = "unimol" if backends.count("unimol") >= backends.count("baseline") else "baseline"
    else:
        selected_backend = "none"
    mixed = len(set(backends)) > 1
    warnings = ["single_backend_required_per_run"] if mixed else []
    return BackendRecommendation(
        selected_backend=selected_backend,
        per_property=per_property,
        mixed_backend_warning=mixed,
        warnings=warnings,
    )
