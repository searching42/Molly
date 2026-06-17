from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ai4s_agent.adapters.phase1 import predict_candidates_domain_model_adapter


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_domain_model_scorer_merges_precomputed_predictions(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "predictions.csv"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_csv(
        candidate_csv,
        [
            {"candidate_id": "c1", "SMILES": "CCO", "solvent": "toluene"},
            {"candidate_id": "c2", "SMILES": "CCN", "solvent": "ethanol"},
        ],
    )
    _write_csv(
        model_dir / "predictions.csv",
        [
            {"candidate_id": "c1", "plqy_pred": "0.72"},
            {"candidate_id": "c2", "plqy_pred": "0.41"},
        ],
    )
    (model_dir / "domain_model_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "demo_plqy",
                "model_backend": "csv_lookup",
                "prediction_mode": "precomputed_csv",
                "prediction_csv": "predictions.csv",
                "join_key": "candidate_id",
                "prediction_column": "plqy_pred",
            }
        ),
        encoding="utf-8",
    )

    scorer = Path(__file__).resolve().parents[1] / "scripts" / "score_domain_model_candidates.py"
    result = subprocess.run(
        [
            sys.executable,
            str(scorer),
            str(candidate_csv),
            str(output_csv),
            "--property-name",
            "plqy",
            "--model-id",
            "demo_plqy",
            "--model-backend",
            "csv_lookup",
            "--model-dir",
            str(model_dir),
            "--input-columns-json",
            json.dumps({"canonical_smiles": "SMILES", "solvent": "solvent"}),
            "--required-inputs-json",
            json.dumps(["canonical_smiles", "solvent"]),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "success"
    assert _read_csv(output_csv) == [
        {"candidate_id": "c1", "SMILES": "CCO", "solvent": "toluene", "plqy_pred": "0.72"},
        {"candidate_id": "c2", "SMILES": "CCN", "solvent": "ethanol", "plqy_pred": "0.41"},
    ]


def test_domain_model_adapter_executes_scorer_package(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "predictions.csv"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_csv(candidate_csv, [{"candidate_id": "c1", "SMILES": "CCO", "solvent": "toluene"}])
    _write_csv(model_dir / "predictions.csv", [{"candidate_id": "c1", "plqy_pred": "0.72"}])
    (model_dir / "domain_model_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "demo_plqy",
                "model_backend": "csv_lookup",
                "prediction_mode": "precomputed_csv",
                "prediction_csv": "predictions.csv",
                "join_key": "candidate_id",
                "prediction_column": "plqy_pred",
            }
        ),
        encoding="utf-8",
    )

    result = predict_candidates_domain_model_adapter(
        {
            "run_id": "run-domain-exec",
            "candidate_csv": str(candidate_csv),
            "output_csv": str(output_csv),
            "property_id": "plqy",
            "model_id": "demo_plqy",
            "model_backend": "csv_lookup",
            "model_dir": str(model_dir),
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
            "required_inputs": ["canonical_smiles", "solvent"],
            "scorer_path": str(Path(__file__).resolve().parents[1] / "scripts" / "score_domain_model_candidates.py"),
            "execute": True,
        }
    )

    assert result["status"] == "success"
    assert result["output_csv"] == str(output_csv)
    assert result["row_count"] == 1
    assert _read_csv(output_csv)[0]["plqy_pred"] == "0.72"
