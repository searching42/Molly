from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.adapters.phase1 import (
    check_trainability_service,
    draft_cleaning_rules_adapter,
    execute_cleaning_adapter,
    inspect_dataset_service,
    run_baseline_service,
    train_model_baseline_adapter,
)
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent import trainability


class DatasetNotConfirmedError(RuntimeError):
    """Raised when Phase 1 receives a dataset without explicit confirmation."""


class Phase1TrainingError(RuntimeError):
    """Raised when the Phase 1 training orchestration cannot complete."""


@dataclass(frozen=True)
class Phase1TrainingResult:
    status: str
    training_metadata_json: str
    feature_config_json: str
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def run_phase1_training(
    *,
    confirmed_training_dataset_csv: str | Path,
    dataset_manifest_json: str | Path,
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation,
    property_ids: list[str] | None = None,
    n_bits: int = 256,
    random_seed: int = 0,
    generated_at: str | None = None,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    strict_smiles_cleaning: bool = True,
) -> Phase1TrainingResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    training_csv = Path(confirmed_training_dataset_csv).expanduser().resolve()
    manifest_path = Path(dataset_manifest_json).expanduser().resolve()
    props = property_ids or ["plqy"]

    manifest = _load_json(manifest_path)
    _require_confirmed(
        confirmation=confirmation,
        dataset_manifest=manifest,
        dataset_manifest_path=manifest_path,
        training_csv=training_csv,
    )
    _require_rdkit_morgan()

    dataset_hash = _sha256_file(training_csv)
    feature_config = {
        "feature_type": "rdkit_morgan_fingerprint",
        "radius": 2,
        "n_bits": int(n_bits),
        "random_seed": int(random_seed),
        "property_ids": props,
        "smiles_col": "SMILES",
        "split_col": "split_group",
        "dynamic_feature_selection": False,
    }
    config_hash = _sha256_json(feature_config)
    feature_config_json = output_path / "feature_config.json"
    write_json(feature_config_json, feature_config | {"config_hash": config_hash})

    inspect = inspect_dataset_service(
        {
            "input_csv": str(training_csv),
            "min_numeric_ratio": min_numeric_ratio,
            "min_nonempty": min_nonempty,
        }
    )
    _ensure_success("inspect_dataset", inspect)

    draft = draft_cleaning_rules_adapter(
        {
            "inspect_result": inspect,
            "strict_smiles_cleaning": strict_smiles_cleaning,
        }
    )
    _ensure_success("draft_cleaning_rules", draft)
    mapping = dict(draft["cleaning_rules_draft"])
    mapping["properties"] = [{"property_id": prop, "source_column": prop} for prop in props]

    cleaned = execute_cleaning_adapter(
        {
            "run_id": run_id,
            "input_csv": str(training_csv),
            "output_dir": str(output_path / "clean"),
            "mapping": mapping,
            "properties": props,
            "min_numeric_ratio": min_numeric_ratio,
            "min_nonempty": min_nonempty,
            "strict_smiles_cleaning": strict_smiles_cleaning,
        }
    )
    _ensure_success("clean_dataset", cleaned)
    cleaned_master_csv = str(cleaned["outputs"]["cleaned_master_csv"])
    property_catalog_json = str(cleaned["outputs"]["property_catalog_json"])

    trainability_result = check_trainability_service(
        {
            "run_id": run_id,
            "property_catalog_json": property_catalog_json,
            "output_dir": str(output_path),
        }
    )
    _ensure_success("check_trainability", trainability_result)

    baseline = run_baseline_service(
        {
            "run_id": run_id,
            "cleaned_master_csv": cleaned_master_csv,
            "output_dir": str(output_path / "baseline_evaluation"),
            "properties": props,
        }
    )
    _ensure_success("run_baseline", baseline)

    models: dict[str, dict[str, Any]] = {}
    model_hashes: dict[str, str] = {}
    for prop in props:
        trained = train_model_baseline_adapter(
            {
                "run_id": run_id,
                "cleaned_master_csv": cleaned_master_csv,
                "property_id": prop,
                "model_root": str(output_path / "models"),
                "n_bits": n_bits,
                "domain": "photophysical_scientific_dataset",
            }
        )
        _ensure_success(f"train_model:{prop}", trained)
        metadata = dict(trained["model_metadata"])
        model_path = str(metadata["model_path"])
        model_hash = _sha256_file(Path(model_path))
        model_hashes[prop] = model_hash
        models[prop] = {
            "property_id": prop,
            "model_path": model_path,
            "model_dir": metadata.get("model_dir", ""),
            "model_hash": model_hash,
            "model_metadata": metadata,
            "outputs": trained.get("outputs", {}),
        }

    hashes = {
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        **{f"{prop}_model_hash": value for prop, value in model_hashes.items()},
    }
    metadata_payload = {
        "run_id": run_id,
        "generated_at": generated,
        "status": "success",
        "confirmation": confirmation.to_dict(),
        "dataset_manifest": manifest,
        "dataset": {
            "confirmed_training_dataset_csv": str(training_csv),
            "cleaned_master_csv": cleaned_master_csv,
            "property_catalog_json": property_catalog_json,
        },
        "feature_config": feature_config,
        "hashes": hashes,
        "adapters": {
            "inspect_dataset": inspect,
            "clean_dataset": cleaned,
            "check_trainability": trainability_result,
            "baseline_evaluation": baseline,
        },
        "models": models,
        "baseline_evaluation": {
            "baseline_report": baseline.get("baseline_report", {}),
            "outputs": baseline.get("outputs", {}),
        },
    }
    training_metadata_json = output_path / "training_metadata.json"
    write_json(training_metadata_json, metadata_payload)
    outputs = {
        "training_metadata_json": str(training_metadata_json),
        "feature_config_json": str(feature_config_json),
        "cleaned_master_csv": cleaned_master_csv,
        "property_catalog_json": property_catalog_json,
        "baseline_report_json": str(baseline.get("outputs", {}).get("baseline_report_json", "")),
    }
    return Phase1TrainingResult(
        status="success",
        training_metadata_json=str(training_metadata_json),
        feature_config_json=str(feature_config_json),
        models=models,
        hashes=hashes,
        outputs=outputs,
    )


def _require_confirmed(
    *,
    confirmation: DatasetConfirmation,
    dataset_manifest: dict[str, Any],
    dataset_manifest_path: Path,
    training_csv: Path,
) -> None:
    if not confirmation.confirmed:
        raise DatasetNotConfirmedError("Phase 1 training requires DatasetConfirmation.confirmed=True")
    if not isinstance(dataset_manifest, dict):
        raise DatasetNotConfirmedError("dataset_manifest must be an object")

    status = str(dataset_manifest.get("status") or "").strip()
    if status != "confirmed":
        raise DatasetNotConfirmedError(f"dataset_manifest status is not confirmed: {status}")

    manifest_confirmation = dataset_manifest.get("confirmation")
    if not isinstance(manifest_confirmation, dict):
        raise DatasetNotConfirmedError("dataset_manifest confirmation is missing")
    if manifest_confirmation.get("confirmed") is not True:
        raise DatasetNotConfirmedError("dataset_manifest confirmation is not confirmed")

    artifacts = dataset_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DatasetNotConfirmedError("dataset_manifest artifacts are missing")
    manifest_training_csv = str(artifacts.get("training_dataset_csv") or "").strip()
    if not manifest_training_csv:
        raise DatasetNotConfirmedError("dataset_manifest artifacts.training_dataset_csv is missing")
    expected_training_csv = _resolve_manifest_path(manifest_training_csv, base=dataset_manifest_path.parent)
    actual_training_csv = training_csv.expanduser().resolve()
    if expected_training_csv != actual_training_csv:
        raise DatasetNotConfirmedError(
            "dataset_manifest training_dataset_csv does not match the requested training dataset"
        )


def _resolve_manifest_path(path_raw: str, *, base: Path) -> Path:
    path = Path(path_raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_rdkit_morgan() -> None:
    if trainability.Chem is None or trainability.AllChem is None:
        raise Phase1TrainingError("Phase 1 stabilized training requires RDKit Morgan fingerprints")


def _ensure_success(name: str, result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        raise Phase1TrainingError(f"{name} failed: {result.get('error') or result}")


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"
