from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase3_scientific_extractor import StructuredScientificRecord

try:
    from rdkit import Chem  # type: ignore
except Exception:
    Chem = None  # type: ignore


@dataclass(frozen=True)
class CorpusConflictAuditResult:
    accepted_records: list[StructuredScientificRecord]
    rejected_records: list[dict[str, Any]]
    summary: dict[str, Any]
    conflict_report_json: str
    conflict_summary_json: str
    conflict_table_csv: str
    consistent_duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    conflicting_groups: list[dict[str, Any]] = field(default_factory=list)


def audit_corpus_conflicts(
    *,
    records: Iterable[StructuredScientificRecord],
    output_dir: str | Path,
    run_id: str,
    extraction_rejections: Iterable[dict[str, Any]] | None = None,
    generated_at: str | None = None,
    plqy_conflict_tolerance: float = 0.03,
    lambda_conflict_tolerance_nm: float = 5.0,
) -> CorpusConflictAuditResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    record_list = list(records)

    rejected_records: list[dict[str, Any]] = []
    valid_by_smiles: dict[str, list[StructuredScientificRecord]] = defaultdict(list)
    canonical_by_record_id: dict[str, str] = {}
    conflict_table_rows: list[dict[str, Any]] = []

    for record in record_list:
        canonical = _canonical_smiles(record.smiles)
        if canonical is None:
            rejected_records.append(_rejection_from_record(record, reason="invalid_smiles"))
            conflict_table_rows.append(_conflict_table_row(record, "", status="rejected", reason="invalid_smiles"))
            continue
        canonical_by_record_id[record.record_id] = canonical
        valid_by_smiles[canonical].append(record)

    for rejected in extraction_rejections or []:
        rejection = _rejection_from_extraction_attempt(rejected)
        rejected_records.append(rejection)
        conflict_table_rows.append(_attempt_table_row(rejection))

    accepted_records: list[StructuredScientificRecord] = []
    consistent_duplicate_groups: list[dict[str, Any]] = []
    conflicting_groups: list[dict[str, Any]] = []

    for canonical_smiles in sorted(valid_by_smiles):
        group = sorted(
            valid_by_smiles[canonical_smiles],
            key=lambda item: (item.paper_id, item.page, item.table_id, item.row_index, item.record_id),
        )
        conflict_properties = _conflicting_properties(
            group,
            plqy_tolerance=plqy_conflict_tolerance,
            lambda_tolerance=lambda_conflict_tolerance_nm,
        )
        if conflict_properties:
            conflict = {
                "canonical_smiles": canonical_smiles,
                "record_count": len(group),
                "properties": conflict_properties,
                "records": [_record_ref(record, canonical_smiles) for record in group],
                "status": "rejected_unresolved_conflict",
            }
            conflicting_groups.append(conflict)
            for record in group:
                rejected_records.append(
                    _rejection_from_record(
                        record,
                        reason="duplicate_conflict",
                        canonical_smiles=canonical_smiles,
                        conflict_properties=conflict_properties,
                    )
                )
                conflict_table_rows.append(
                    _conflict_table_row(record, canonical_smiles, status="rejected", reason="duplicate_conflict")
                )
            continue

        if len(group) > 1:
            consistent_duplicate_groups.append(
                {
                    "canonical_smiles": canonical_smiles,
                    "record_count": len(group),
                    "records": [_record_ref(record, canonical_smiles) for record in group],
                    "status": "merge_allowed",
                }
            )
        accepted_records.extend(group)
        for record in group:
            status = "consistent_duplicate" if len(group) > 1 else "accepted"
            conflict_table_rows.append(_conflict_table_row(record, canonical_smiles, status=status))

    reason_counts = Counter(str(record.get("reason") or "unknown") for record in rejected_records)
    conflicted_smiles = sorted(item["canonical_smiles"] for item in conflicting_groups)
    consistent_smiles = sorted(item["canonical_smiles"] for item in consistent_duplicate_groups)
    summary = {
        "run_id": run_id,
        "generated_at": generated,
        "document_count": len({record.paper_id for record in record_list}),
        "input_record_count": len(record_list),
        "valid_record_count": sum(len(group) for group in valid_by_smiles.values()),
        "accepted_record_count": len({canonical_by_record_id[item.record_id] for item in accepted_records}),
        "rejected_record_count": len(rejected_records),
        "consistent_duplicate_count": len(consistent_duplicate_groups),
        "conflict_count": len(conflicting_groups),
        "unresolved_conflict_count": len(conflicting_groups),
        "reason_counts": dict(sorted(reason_counts.items())),
        "conflicted_smiles": conflicted_smiles,
        "consistent_duplicate_smiles": consistent_smiles,
    }

    conflict_report_json = output_path / "corpus_conflict_report.json"
    conflict_summary_json = output_path / "conflict_summary.json"
    conflict_table_csv = output_path / "conflict_table.csv"
    write_json(
        conflict_report_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "summary": summary,
            "consistent_duplicate_groups": consistent_duplicate_groups,
            "conflicting_groups": conflicting_groups,
            "rejected_records": rejected_records,
            "notes": [
                "consistent_duplicates_may_be_merged",
                "unresolved_conflicts_are_rejected",
                "no_silent_averaging_outside_tolerance",
            ],
        },
    )
    write_json(conflict_summary_json, summary)
    _write_conflict_table(conflict_table_csv, conflict_table_rows)
    return CorpusConflictAuditResult(
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        summary=summary,
        conflict_report_json=str(conflict_report_json),
        conflict_summary_json=str(conflict_summary_json),
        conflict_table_csv=str(conflict_table_csv),
        consistent_duplicate_groups=consistent_duplicate_groups,
        conflicting_groups=conflicting_groups,
    )


def _canonical_smiles(smiles: str) -> str | None:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles or "").strip())  # type: ignore[union-attr]
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True))  # type: ignore[union-attr]


def _conflicting_properties(
    records: list[StructuredScientificRecord],
    *,
    plqy_tolerance: float,
    lambda_tolerance: float,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for property_id, tolerance in [("plqy", plqy_tolerance), ("lambda_em_nm", lambda_tolerance)]:
        values = [float(getattr(record, property_id)) for record in records if getattr(record, property_id) is not None]
        if len(values) <= 1:
            continue
        min_value = min(values)
        max_value = max(values)
        if max_value - min_value > tolerance:
            conflicts.append(
                {
                    "property_id": property_id,
                    "min_value": round(min_value, 12),
                    "max_value": round(max_value, 12),
                    "tolerance": tolerance,
                }
            )
    return conflicts


def _record_ref(record: StructuredScientificRecord, canonical_smiles: str) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "smiles": record.smiles,
        "canonical_smiles": canonical_smiles,
        "plqy": record.plqy,
        "lambda_em_nm": record.lambda_em_nm,
        "provenance": _provenance(record),
    }


def _provenance(record: StructuredScientificRecord) -> dict[str, Any]:
    return {
        **record.provenance,
        "paper_id": record.paper_id,
        "page": record.page,
        "table_id": record.table_id,
        "row_id": record.row_id,
    }


def _rejection_from_record(
    record: StructuredScientificRecord,
    *,
    reason: str,
    canonical_smiles: str | None = None,
    conflict_properties: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rejection = {
        "record_id": record.record_id,
        "smiles": record.smiles,
        "canonical_smiles": canonical_smiles,
        "reason": reason,
        "paper_id": record.paper_id,
        "page": record.page,
        "table_id": record.table_id,
        "row_id": record.row_id,
        "raw_values": record.raw_values,
        "provenance": _provenance(record),
    }
    if conflict_properties:
        rejection["conflict_properties"] = conflict_properties
    return rejection


def _rejection_from_extraction_attempt(rejected: dict[str, Any]) -> dict[str, Any]:
    raw_values = rejected.get("raw_values") if isinstance(rejected.get("raw_values"), dict) else {}
    reason = _classify_extraction_rejection(str(rejected.get("reason") or ""), raw_values)
    provenance = rejected.get("provenance") if isinstance(rejected.get("provenance"), dict) else {}
    return {
        "record_id": str(rejected.get("record_id") or ""),
        "smiles": str(rejected.get("smiles") or raw_values.get("SMILES") or ""),
        "reason": reason,
        "paper_id": str(rejected.get("paper_id") or provenance.get("paper_id") or ""),
        "page": rejected.get("page") or provenance.get("page"),
        "table_id": str(rejected.get("table_id") or provenance.get("table_id") or ""),
        "row_id": str(rejected.get("row_id") or provenance.get("row_id") or ""),
        "raw_values": raw_values,
        "provenance": {
            **provenance,
            "paper_id": str(rejected.get("paper_id") or provenance.get("paper_id") or ""),
        },
    }


def _classify_extraction_rejection(reason: str, raw_values: dict[str, Any]) -> str:
    if reason != "missing_required_properties":
        return reason or "extraction_rejected"
    plqy_raw = str(raw_values.get("PLQY") or raw_values.get("plqy") or "").strip()
    lambda_raw = str(raw_values.get("lambda_em_nm") or raw_values.get("lambda_em") or "").strip()
    if not plqy_raw:
        return "missing_plqy"
    if not lambda_raw:
        return "missing_lambda_em_nm"
    return "missing_required_properties"


def _conflict_table_row(
    record: StructuredScientificRecord,
    canonical_smiles: str,
    *,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "smiles": record.smiles,
        "canonical_smiles": canonical_smiles,
        "status": status,
        "reason": reason,
        "paper_id": record.paper_id,
        "page": record.page,
        "table_id": record.table_id,
        "row_id": record.row_id,
        "plqy": "" if record.plqy is None else record.plqy,
        "lambda_em_nm": "" if record.lambda_em_nm is None else record.lambda_em_nm,
    }


def _attempt_table_row(rejection: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": rejection.get("record_id", ""),
        "smiles": rejection.get("smiles", ""),
        "canonical_smiles": rejection.get("canonical_smiles", ""),
        "status": "rejected",
        "reason": rejection.get("reason", ""),
        "paper_id": rejection.get("paper_id", ""),
        "page": rejection.get("page", ""),
        "table_id": rejection.get("table_id", ""),
        "row_id": rejection.get("row_id", ""),
        "plqy": "",
        "lambda_em_nm": "",
    }


def _write_conflict_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "record_id",
        "smiles",
        "canonical_smiles",
        "status",
        "reason",
        "paper_id",
        "page",
        "table_id",
        "row_id",
        "plqy",
        "lambda_em_nm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
