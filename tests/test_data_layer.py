from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ai4s_agent.data_layer import (
    DatasetRole,
    check_smiles_leakage,
    generate_property_catalog,
    inspect_dataset,
    register_dataset,
)
from ai4s_agent._utils import read_csv_dict_rows


def _write_csv(path: Path, rows: list[dict[str, str]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_register_dataset_copies_upload_and_records_role_manifest(tmp_path: Path) -> None:
    source = tmp_path / "upload.csv"
    _write_csv(source, [{"SMILES": "CCO", "target": "1.2"}])

    registered = register_dataset(
        workspace_dir=tmp_path,
        project_id="proj-a",
        source_path=source,
        role=DatasetRole.TRAIN,
        dataset_id="train-main",
    )

    assert registered.role == DatasetRole.TRAIN
    assert registered.version == "v001"
    assert registered.registered_path.exists()
    assert "assets/datasets/raw/train-main/v001/upload.csv" in registered.registered_path.as_posix()

    manifest = json.loads(registered.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "train-main"
    assert manifest["role"] == "train_dataset"
    assert manifest["content_hash"].startswith("sha256:")

    candidate = register_dataset(
        workspace_dir=tmp_path,
        project_id="proj-a",
        source_path=source,
        role="candidate_dataset",
        dataset_id="cand-main",
    )
    assert "assets/datasets/candidates/cand-main/v001/upload.csv" in candidate.registered_path.as_posix()


def test_inspect_dataset_detects_data_layer_features(tmp_path: Path) -> None:
    data = tmp_path / "train_semicolon.csv"
    _write_csv(
        data,
        [
            {"mol_id": "m1", "smiles": "CCO", "PLQY (%)": "80", "lambda_em_nm": "520", "split": "train"},
            {"mol_id": "m2", "smiles": "CCO", "PLQY (%)": "20", "lambda_em_nm": "530", "split": "train"},
            {"mol_id": "m3", "smiles": "CCN", "PLQY (%)": "75", "lambda_em_nm": "510", "split": "valid"},
            {"mol_id": "m4", "smiles": "CCC", "PLQY (%)": "70", "lambda_em_nm": "5000", "split": "test"},
            {"mol_id": "m5", "smiles": "CCCl", "PLQY (%)": "90", "lambda_em_nm": "500", "split": ""},
        ],
        delimiter=";",
    )

    inspection = inspect_dataset(data, min_numeric_ratio=0.6, min_nonempty=2)

    assert inspection.structure.delimiter == ";"
    assert inspection.structure.row_count == 5
    assert inspection.smiles_column == "smiles"

    property_by_id = {candidate.property_id: candidate for candidate in inspection.property_candidates}
    assert property_by_id["plqy"].source_column == "PLQY (%)"
    assert property_by_id["plqy"].unit == "percent"
    assert property_by_id["plqy"].scale == 0.01
    assert property_by_id["lambda_em"].source_column == "lambda_em_nm"

    assert inspection.duplicate_conflicts
    assert inspection.duplicate_conflicts[0].canonical_smiles == "CCO"
    assert "plqy" in inspection.duplicate_conflicts[0].conflicting_properties

    outlier_props = {warning.property_id for warning in inspection.outlier_warnings}
    assert "lambda_em" in outlier_props

    assert inspection.split_assessment.split_column == "split"
    assert inspection.split_assessment.status == "PARTIAL_SPLIT"
    assert inspection.split_assessment.fallback_strategy == "deterministic_hash"

    catalog = generate_property_catalog(inspection)
    assert catalog["properties"][0]["label_count"] >= 4
    assert any(item["property_id"] == "plqy" for item in catalog["properties"])


def test_inspect_dataset_normalizes_whitespace_headers_in_rows(tmp_path: Path) -> None:
    data = tmp_path / "train_spaced_headers.csv"
    data.write_text(
        " SMILES , PLQY (%) , split \n"
        "CCO,80,train\n"
        "CCN,75,valid\n"
        "CCC,70,test\n",
        encoding="utf-8",
    )

    inspection = inspect_dataset(data, min_numeric_ratio=0.6, min_nonempty=2)

    assert inspection.smiles_column == "SMILES"
    property_by_id = {candidate.property_id: candidate for candidate in inspection.property_candidates}
    assert property_by_id["plqy"].source_column == "PLQY (%)"
    assert property_by_id["plqy"].numeric_count == 3
    assert inspection.split_assessment.split_column == "split"
    assert inspection.split_assessment.status == "PROVIDED_SPLIT"


def test_read_csv_dict_rows_preserves_alignment_when_headers_are_blank(tmp_path: Path) -> None:
    data = tmp_path / "blank_header.csv"
    data.write_text("smiles,,plqy\nCCO,SHOULD_NOT_OVERWRITE,0.8\n", encoding="utf-8")

    rows, headers = read_csv_dict_rows(data)

    assert headers == ["smiles", "plqy"]
    assert rows == [{"smiles": "CCO", "plqy": "0.8"}]


def test_data_leakage_check_uses_canonical_smiles_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    candidates = tmp_path / "candidates.csv"
    _write_csv(train, [{"SMILES": "CCO"}, {"SMILES": "CCN"}])
    _write_csv(candidates, [{"smiles": "NCC"}, {"smiles": "CCC"}])

    report = check_smiles_leakage(train, candidates)

    assert report.train_smiles_column == "SMILES"
    assert report.other_smiles_column == "smiles"
    assert report.overlap_count == 1
    assert report.overlap_smiles == ["CCN"]


def test_register_dataset_rejects_unknown_role(tmp_path: Path) -> None:
    source = tmp_path / "upload.csv"
    _write_csv(source, [{"SMILES": "CCO"}])

    with pytest.raises(ValueError, match="unknown dataset role"):
        register_dataset(
            workspace_dir=tmp_path,
            project_id="proj-a",
            source_path=source,
            role="bad_role",
        )
