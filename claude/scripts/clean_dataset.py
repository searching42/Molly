from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return None
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    try:
        parsed = float(raw.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed / 100.0 if percent else parsed


def _property_id(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return token or "property"


def _read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original = list(reader.fieldnames or [])
        pairs = [(header, str(header).strip()) for header in original if str(header or "").strip()]
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({clean: str(raw.get(source) or "").strip() for source, clean in pairs})
    return rows, [clean for _, clean in pairs]


def _find_header(headers: list[str], candidates: set[str]) -> str:
    return next((header for header in headers if header.strip().lower() in candidates), "")


def _infer_properties(
    rows: list[dict[str, str]],
    headers: list[str],
    excluded: set[str],
    min_numeric_ratio: float,
    min_nonempty: int,
) -> list[dict[str, Any]]:
    properties: list[dict[str, Any]] = []
    for header in headers:
        if header.lower() in excluded:
            continue
        nonempty = [row.get(header, "") for row in rows if str(row.get(header, "")).strip()]
        numeric = [value for value in nonempty if _safe_float(value) is not None]
        if len(nonempty) < min_nonempty:
            continue
        if not nonempty or len(numeric) / len(nonempty) < min_numeric_ratio:
            continue
        properties.append(
            {
                "property_id": _property_id(header),
                "source_column": header,
                "scale": 1.0,
                "offset": 0.0,
                "unit": "",
                "canonical_unit": "",
            }
        )
    return properties


def _normalize_split(raw: str, index: int, total: int) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "1": "train",
        "valid": "valid",
        "validation": "valid",
        "val": "valid",
        "2": "valid",
        "test": "test",
        "3": "test",
    }
    if value in aliases:
        return aliases[value]
    ratio = index / max(total, 1)
    return "train" if ratio < 0.8 else "valid" if ratio < 0.9 else "test"


def _canonicalizer(non_strict_rdkit: bool):
    if non_strict_rdkit:
        return lambda smiles: str(smiles or "").strip()
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover - adapter checks this first
        raise RuntimeError("RDKit is required for strict SMILES cleaning") from exc

    def canonicalize(smiles: str) -> str:
        molecule = Chem.MolFromSmiles(str(smiles or "").strip())
        return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""

    return canonicalize


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-numeric-ratio", type=float, default=0.6)
    parser.add_argument("--min-nonempty", type=int, default=30)
    parser.add_argument("--mapping-json", default="")
    parser.add_argument("--smiles-col", default="")
    parser.add_argument("--split-col", default="")
    parser.add_argument("--id-col", default="")
    parser.add_argument("--properties", default="")
    parser.add_argument("--drop-empty-target-rows", action="store_true")
    parser.add_argument("--non-strict-rdkit", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, headers = _read_rows(input_csv)
    mapping = _read_json(args.mapping_json)
    smiles_col = str(args.smiles_col or mapping.get("smiles_col") or "").strip()
    if not smiles_col:
        smiles_col = _find_header(headers, {"smiles", "canonical_smiles", "isomeric_smiles", "structure"})
    split_col = str(args.split_col or mapping.get("split_col") or "").strip()
    if not split_col:
        split_col = _find_header(headers, {"split", "split_group", "set", "partition"})
    id_col = str(args.id_col or mapping.get("id_col") or "").strip()
    if not id_col:
        id_col = _find_header(headers, {"dataset_id", "candidate_id", "compound_id", "id"})
    if not smiles_col:
        raise ValueError("SMILES column could not be inferred")

    property_specs = [item for item in mapping.get("properties", []) if isinstance(item, dict)]
    requested = [item.strip() for item in str(args.properties or "").split(",") if item.strip()]
    if requested:
        by_id = {
            str(item.get("property_id") or item.get("source_column") or "").strip(): item
            for item in property_specs
        }
        property_specs = [
            by_id.get(name)
            or {
                "property_id": _property_id(name),
                "source_column": name,
                "scale": 1.0,
                "offset": 0.0,
                "unit": "",
                "canonical_unit": "",
            }
            for name in requested
        ]
    if not property_specs:
        excluded = {value.lower() for value in (smiles_col, split_col, id_col) if value}
        excluded.update({"source_row", "n_records_aggregated"})
        property_specs = _infer_properties(
            rows,
            headers,
            excluded,
            float(args.min_numeric_ratio),
            int(args.min_nonempty),
        )

    canonicalize = _canonicalizer(bool(args.non_strict_rdkit))
    cleaned_by_smiles: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    valid_counts = {str(item.get("property_id") or ""): 0 for item in property_specs}

    for index, row in enumerate(rows, start=1):
        raw_smiles = str(row.get(smiles_col, "")).strip()
        canonical_smiles = canonicalize(raw_smiles)
        if not canonical_smiles:
            rejected.append({"source_row": index, "reason": "invalid_or_missing_smiles", **row})
            continue

        cleaned: dict[str, Any] = {
            "dataset_id": str(row.get(id_col, "")).strip() if id_col else f"row_{index}",
            "SMILES": canonical_smiles,
            "split_group": _normalize_split(row.get(split_col, "") if split_col else "", index - 1, len(rows)),
            "n_records_aggregated": 1,
            "source_row": index,
        }
        has_target = False
        for item in property_specs:
            property_id = str(item.get("property_id") or _property_id(item.get("source_column") or ""))
            source_column = str(item.get("source_column") or property_id)
            value = _safe_float(row.get(source_column, ""))
            if value is None:
                cleaned[property_id] = ""
                continue
            value = value * float(item.get("scale", 1.0) or 1.0) + float(item.get("offset", 0.0) or 0.0)
            cleaned[property_id] = value
            valid_counts[property_id] = valid_counts.get(property_id, 0) + 1
            has_target = True
        if args.drop_empty_target_rows and not has_target:
            rejected.append({"source_row": index, "reason": "missing_all_targets", **row})
            continue

        existing = cleaned_by_smiles.get(canonical_smiles)
        if existing is None:
            cleaned_by_smiles[canonical_smiles] = cleaned
        else:
            existing["n_records_aggregated"] = int(existing.get("n_records_aggregated", 1)) + 1

    cleaned_rows = list(cleaned_by_smiles.values())
    property_ids = [str(item.get("property_id") or _property_id(item.get("source_column") or "")) for item in property_specs]
    fieldnames = ["dataset_id", "SMILES", *property_ids, "split_group", "n_records_aggregated", "source_row"]

    cleaned_path = output_dir / f"{args.run_id}_cleaned_master.csv"
    catalog_path = output_dir / f"{args.run_id}_property_catalog.json"
    rejected_path = output_dir / f"{args.run_id}_rejected_rows.csv"
    report_path = output_dir / f"{args.run_id}_cleaning_report.json"
    _write_csv(cleaned_path, cleaned_rows, fieldnames)
    _write_csv(rejected_path, rejected, list(rejected[0].keys()) if rejected else ["source_row", "reason"])

    catalog_properties = []
    for item, property_id in zip(property_specs, property_ids, strict=False):
        catalog_properties.append(
            {
                "property_id": property_id,
                "source_column": str(item.get("source_column") or property_id),
                "valid_count_deduped": sum(1 for row in cleaned_rows if row.get(property_id) not in {None, ""}),
                "unit": str(item.get("unit") or ""),
                "canonical_unit": str(item.get("canonical_unit") or ""),
            }
        )
    catalog = {
        "run_id": args.run_id,
        "row_count_raw": len(rows),
        "row_count_deduped": len(cleaned_rows),
        "smiles_col": "SMILES",
        "split_col": "split_group",
        "properties": catalog_properties,
    }
    _write_json(catalog_path, catalog)

    report = {
        "run_id": args.run_id,
        "status": "success",
        "counts": {
            "input_rows": len(rows),
            "cleaned_rows": len(cleaned_rows),
            "rejected_rows": len(rejected),
        },
        "outputs": {
            "cleaned_master_csv": str(cleaned_path),
            "property_catalog_json": str(catalog_path),
            "rejected_rows_csv": str(rejected_path),
        },
    }
    _write_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
