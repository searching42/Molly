from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase3_scientific_extractor import StructuredScientificRecord

try:
    from rdkit import Chem  # type: ignore
except Exception:
    Chem = None  # type: ignore


@dataclass(frozen=True)
class DatasetConfirmation:
    confirmed: bool
    confirmed_by: str
    confirmation_source: str
    confirmation_timestamp: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confirmed, bool):
            raise TypeError("confirmed must be a bool")
        if self.confirmed and not self.confirmed_by.strip():
            raise ValueError("confirmed datasets require confirmed_by")
        if self.confirmed and not self.confirmation_source.strip():
            raise ValueError("confirmed datasets require confirmation_source")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificDatasetBuildResult:
    status: str
    candidate_dataset_csv: str
    training_dataset_csv: str
    rejected_records_json: str
    dataset_manifest_json: str
    candidate_record_count: int
    training_record_count: int
    rejected_record_count: int
    confirmation: DatasetConfirmation


DATASET_FIELDNAMES = [
    "dataset_id",
    "SMILES",
    "plqy",
    "lambda_em_nm",
    "split_group",
    "paper_id",
    "page",
    "table_id",
    "row_id",
    "evidence_ref",
    "confidence",
    "n_records_aggregated",
]

CANDIDATE_FIELDNAMES = [
    "candidate_id",
    "SMILES",
    "paper_id",
    "page",
    "table_id",
    "row_id",
    "source_dataset_id",
]


def build_scientific_dataset(
    records: Iterable[StructuredScientificRecord],
    *,
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation | None = None,
    generated_at: str | None = None,
    plqy_range: tuple[float, float] = (0.0, 1.0),
    lambda_em_nm_range: tuple[float, float] = (350.0, 900.0),
    plqy_conflict_tolerance: float = 0.03,
    lambda_conflict_tolerance_nm: float = 5.0,
) -> ScientificDatasetBuildResult:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    generated = generated_at or now_iso()
    confirmation_context = confirmation or DatasetConfirmation(
        confirmed=False,
        confirmed_by="",
        confirmation_source="unspecified",
        confirmation_timestamp=None,
    )

    rejected: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    for record in records:
        row = _validated_record(
            record,
            plqy_range=plqy_range,
            lambda_em_nm_range=lambda_em_nm_range,
        )
        if row["status"] == "rejected":
            rejected.append(row["rejection"])
        else:
            valid_records.append(row["record"])

    candidate_rows: list[dict[str, Any]] = []
    for group in _group_by_smiles(valid_records).values():
        if _has_group_conflict(
            group,
            plqy_tolerance=plqy_conflict_tolerance,
            lambda_tolerance=lambda_conflict_tolerance_nm,
        ):
            for row in group:
                rejected.append(_rejection_from_row(row, reason="duplicate_conflict"))
            continue
        candidate_rows.append(_merge_group(group, sequence=len(candidate_rows) + 1))

    training_rows = candidate_rows if confirmation_context.confirmed else []
    candidate_csv = output_path / "candidate_dataset.csv"
    training_csv = output_path / "training_dataset.csv"
    rejected_json = output_path / "rejected_records.json"
    manifest_json = output_path / "dataset_manifest.json"

    _write_csv(candidate_csv, _candidate_rows(candidate_rows), CANDIDATE_FIELDNAMES)
    _write_csv(training_csv, training_rows, DATASET_FIELDNAMES)
    write_json(
        rejected_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "records": rejected,
        },
    )
    manifest = {
        "run_id": run_id,
        "generated_at": generated,
        "status": "confirmed" if confirmation_context.confirmed else "awaiting_confirmation",
        "candidate_record_count": len(candidate_rows),
        "training_record_count": len(training_rows),
        "rejected_record_count": len(rejected),
        "confirmation": confirmation_context.to_dict(),
        "provenance_fields": ["paper_id", "page", "table_id", "row_id"],
        "validation_rules": {
            "smiles": "rdkit_mol_from_smiles_required",
            "plqy_range": list(plqy_range),
            "lambda_em_nm_range": list(lambda_em_nm_range),
            "duplicate_policy": "merge_consistent_reject_conflicting",
            "plqy_conflict_tolerance": plqy_conflict_tolerance,
            "lambda_conflict_tolerance_nm": lambda_conflict_tolerance_nm,
        },
        "artifacts": {
            "candidate_dataset_csv": str(candidate_csv),
            "training_dataset_csv": str(training_csv),
            "rejected_records_json": str(rejected_json),
        },
    }
    write_json(manifest_json, manifest)
    return ScientificDatasetBuildResult(
        status=str(manifest["status"]),
        candidate_dataset_csv=str(candidate_csv),
        training_dataset_csv=str(training_csv),
        rejected_records_json=str(rejected_json),
        dataset_manifest_json=str(manifest_json),
        candidate_record_count=len(candidate_rows),
        training_record_count=len(training_rows),
        rejected_record_count=len(rejected),
        confirmation=confirmation_context,
    )


def _validated_record(
    record: StructuredScientificRecord,
    *,
    plqy_range: tuple[float, float],
    lambda_em_nm_range: tuple[float, float],
) -> dict[str, Any]:
    canonical = _canonical_smiles(record.smiles)
    if canonical is None:
        return {"status": "rejected", "rejection": _rejection_from_record(record, reason="invalid_smiles")}
    if record.plqy is None:
        return {"status": "rejected", "rejection": _rejection_from_record(record, reason="missing_plqy")}
    if not _in_range(record.plqy, plqy_range):
        return {"status": "rejected", "rejection": _rejection_from_record(record, reason="invalid_plqy_range")}
    if record.lambda_em_nm is None:
        return {"status": "rejected", "rejection": _rejection_from_record(record, reason="missing_lambda_em_nm")}
    if not _in_range(record.lambda_em_nm, lambda_em_nm_range):
        return {"status": "rejected", "rejection": _rejection_from_record(record, reason="invalid_lambda_em_nm_range")}
    return {
        "status": "valid",
        "record": {
            "record": record,
            "canonical_smiles": canonical,
            "plqy": float(record.plqy),
            "lambda_em_nm": float(record.lambda_em_nm),
        },
    }


def _canonical_smiles(smiles: str) -> str | None:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles or "").strip())  # type: ignore[union-attr]
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True))  # type: ignore[union-attr]


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return math.isfinite(value) and bounds[0] <= value <= bounds[1]


def _group_by_smiles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_smiles"])].append(row)
    return dict(grouped)


def _has_group_conflict(
    group: list[dict[str, Any]],
    *,
    plqy_tolerance: float,
    lambda_tolerance: float,
) -> bool:
    if len(group) <= 1:
        return False
    plqy_values = [float(row["plqy"]) for row in group]
    lambda_values = [float(row["lambda_em_nm"]) for row in group]
    return (max(plqy_values) - min(plqy_values) > plqy_tolerance) or (
        max(lambda_values) - min(lambda_values) > lambda_tolerance
    )


def _merge_group(group: list[dict[str, Any]], *, sequence: int) -> dict[str, Any]:
    representative = sorted(group, key=lambda row: (-row["record"].confidence, row["record"].record_id))[0]["record"]
    plqy = round(sum(float(row["plqy"]) for row in group) / len(group), 12)
    lambda_em_nm = round(sum(float(row["lambda_em_nm"]) for row in group) / len(group), 12)
    return {
        "dataset_id": representative.record_id,
        "SMILES": str(group[0]["canonical_smiles"]),
        "plqy": plqy,
        "lambda_em_nm": lambda_em_nm,
        "split_group": "valid" if sequence % 5 == 0 else "train",
        "paper_id": representative.paper_id,
        "page": representative.page,
        "table_id": representative.table_id,
        "row_id": representative.row_id,
        "evidence_ref": representative.evidence_ref,
        "confidence": representative.confidence,
        "n_records_aggregated": len(group),
    }


def _candidate_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"cand_{index:06d}",
            "SMILES": row["SMILES"],
            "paper_id": row["paper_id"],
            "page": row["page"],
            "table_id": row["table_id"],
            "row_id": row["row_id"],
            "source_dataset_id": row["dataset_id"],
        }
        for index, row in enumerate(dataset_rows, start=1)
    ]


def _rejection_from_record(record: StructuredScientificRecord, *, reason: str) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "smiles": record.smiles,
        "reason": reason,
        "paper_id": record.paper_id,
        "page": record.page,
        "table_id": record.table_id,
        "row_id": record.row_id,
        "raw_values": record.raw_values,
    }


def _rejection_from_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    record = row["record"]
    rejection = _rejection_from_record(record, reason=reason)
    rejection["canonical_smiles"] = row["canonical_smiles"]
    rejection["plqy"] = row["plqy"]
    rejection["lambda_em_nm"] = row["lambda_em_nm"]
    return rejection


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
