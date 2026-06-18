from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem  # type: ignore
except Exception:
    Chem = None  # type: ignore


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original = list(reader.fieldnames or [])
        pairs = [(name, str(name or "").strip()) for name in original if str(name or "").strip()]
        rows = [
            {normalized: str(raw.get(source) or "") for source, normalized in pairs}
            for raw in reader
        ]
    return rows, [normalized for _, normalized in pairs]


def _write_rows(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    try:
        parsed = float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed / 100.0 if percent else parsed


def _normalize_property_id(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _load_mapping(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _infer_smiles_col(headers: list[str], explicit: str, mapping: dict[str, Any]) -> str:
    requested = explicit or str(mapping.get("smiles_col") or "")
    if requested in headers:
        return requested
    aliases = {"smiles", "canonical_smiles", "isomeric_smiles", "structure", "chromophore"}
    return next((header for header in headers if header.lower() in aliases), "")


def _property_specs(
    rows: list[dict[str, str]],
    headers: list[str],
    *,
    requested: str,
    mapping: dict[str, Any],
    smiles_col: str,
    split_col: str,
    id_col: str,
    min_numeric_ratio: float,
    min_nonempty: int,
) -> list[dict[str, Any]]:
    mapped = mapping.get("properties") if isinstance(mapping.get("properties"), list) else []
    specs = [dict(item) for item in mapped if isinstance(item, dict)]
    if requested:
        requested_names = [item.strip() for item in requested.split(",") if item.strip()]
        by_id = {str(item.get("property_id") or ""): item for item in specs}
        specs = [
            by_id.get(name)
            or {"property_id": _normalize_property_id(name), "source_column": name}
            for name in requested_names
        ]
    if specs:
        return specs

    excluded = {
        smiles_col.lower(),
        split_col.lower(),
        id_col.lower(),
        "dataset_id",
        "candidate_id",
        "id",
        "source_row",
        "n_records_aggregated",
    }
    inferred: list[dict[str, Any]] = []
    for header in headers:
        if header.lower() in excluded:
            continue
        nonempty = [row.get(header, "") for row in rows if str(row.get(header, "")).strip()]
        numeric = [value for value in nonempty if _safe_float(value) is not None]
        ratio = len(numeric) / len(nonempty) if nonempty else 0.0
        if len(nonempty) >= min_nonempty and ratio >= min_numeric_ratio:
            inferred.append({"property_id": _normalize_property_id(header), "source_column": header})
    return inferred


def _canonicalize(smiles: str, *, strict: bool) -> str | None:
    clean = str(smiles or "").strip()
    if not clean:
        return None
    if Chem is None:
        if strict:
            raise RuntimeError("Strict SMILES cleaning requires RDKit")
        return clean
    mol = Chem.MolFromSmiles(clean)  # type: ignore[union-attr]
    if mol is None:
        return None if strict else clean
    return str(Chem.MolToSmiles(mol, canonical=True))  # type: ignore[union-attr]


def main() -> int:
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
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, headers = _read_rows(input_path)
    mapping = _load_mapping(args.mapping_json)
    smiles_col = _infer_smiles_col(headers, args.smiles_col, mapping)
    if not smiles_col:
        raise RuntimeError("No SMILES column could be inferred")
    split_col = args.split_col or str(mapping.get("split_col") or "")
    id_col = args.id_col or str(mapping.get("id_col") or "")
    specs = _property_specs(
        rows,
        headers,
        requested=args.properties,
        mapping=mapping,
        smiles_col=smiles_col,
        split_col=split_col,
        id_col=id_col,
        min_numeric_ratio=args.min_numeric_ratio,
        min_nonempty=args.min_nonempty,
    )

    strict = not args.non_strict_rdkit
    cleaned_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid_smiles = 0
    duplicate_smiles = 0
    for raw in rows:
        canonical = _canonicalize(raw.get(smiles_col, ""), strict=strict)
        if not canonical:
            invalid_smiles += 1
            continue
        if canonical in seen:
            duplicate_smiles += 1
            continue
        row: dict[str, Any] = dict(raw)
        row[smiles_col] = canonical
        for spec in specs:
            source = str(spec.get("source_column") or spec.get("property_id") or "")
            property_id = str(spec.get("property_id") or _normalize_property_id(source))
            parsed = _safe_float(row.get(source, ""))
            if source != property_id and property_id not in row:
                row[property_id] = "" if parsed is None else parsed
            elif parsed is not None:
                row[source] = parsed
        if args.drop_empty_target_rows and specs:
            if not any(_safe_float(row.get(str(spec.get("property_id") or spec.get("source_column") or ""), "")) is not None for spec in specs):
                continue
        seen.add(canonical)
        cleaned_rows.append(row)

    output_headers = list(headers)
    for spec in specs:
        property_id = str(spec.get("property_id") or "").strip()
        if property_id and property_id not in output_headers:
            output_headers.append(property_id)

    cleaned_csv = output_dir / f"{args.run_id}_cleaned_master.csv"
    catalog_json = output_dir / f"{args.run_id}_property_catalog.json"
    report_json = output_dir / f"{args.run_id}_cleaning_report.json"
    _write_rows(cleaned_csv, cleaned_rows, output_headers)

    properties: list[dict[str, Any]] = []
    for spec in specs:
        property_id = str(spec.get("property_id") or "").strip()
        source = str(spec.get("source_column") or property_id).strip()
        values = [
            _safe_float(row.get(property_id, row.get(source, "")))
            for row in cleaned_rows
        ]
        valid = [value for value in values if value is not None]
        properties.append(
            {
                "property_id": property_id,
                "source_column": source,
                "valid_count_deduped": len(valid),
                "numeric_ratio": round(len(valid) / len(cleaned_rows), 6) if cleaned_rows else 0.0,
                "task_type": "numeric_regression",
            }
        )
    catalog = {
        "run_id": args.run_id,
        "smiles_col": smiles_col,
        "row_count": len(cleaned_rows),
        "properties": properties,
    }
    catalog_json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "run_id": args.run_id,
        "input_rows": len(rows),
        "output_rows": len(cleaned_rows),
        "invalid_smiles": invalid_smiles,
        "duplicate_smiles": duplicate_smiles,
        "strict_smiles_cleaning": strict,
        "outputs": {
            "cleaned_master_csv": str(cleaned_csv),
            "property_catalog_json": str(catalog_json),
        },
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
