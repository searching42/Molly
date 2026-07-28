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


def number(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    percent = raw.endswith("%")
    try:
        parsed = float(raw.rstrip("%").replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed / 100.0 if percent else parsed


def property_id(name: str) -> str:
    return "_".join(filter(None, "".join(ch.lower() if ch.isalnum() else " " for ch in name).split()))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        pairs = [(name, str(name or "").strip()) for name in reader.fieldnames or [] if str(name or "").strip()]
        return [
            {clean: str(row.get(source) or "") for source, clean in pairs}
            for row in reader
        ], [clean for _, clean in pairs]


def load_mapping(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def canonicalize(value: str, strict: bool) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    if Chem is None:
        if strict:
            raise RuntimeError("Strict SMILES cleaning requires RDKit")
        return value
    molecule = Chem.MolFromSmiles(value)  # type: ignore[union-attr]
    if molecule is None:
        return None if strict else value
    return str(Chem.MolToSmiles(molecule, canonical=True))  # type: ignore[union-attr]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping-json", default="")
    parser.add_argument("--smiles-col", default="")
    parser.add_argument("--split-col", default="")
    parser.add_argument("--id-col", default="")
    parser.add_argument("--properties", default="")
    parser.add_argument("--min-numeric-ratio", type=float, default=0.6)
    parser.add_argument("--min-nonempty", type=int, default=30)
    parser.add_argument("--drop-empty-target-rows", action="store_true")
    parser.add_argument("--non-strict-rdkit", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, headers = read_csv(input_path)
    mapping = load_mapping(args.mapping_json)
    smiles_col = args.smiles_col or str(mapping.get("smiles_col") or "")
    if smiles_col not in headers:
        aliases = {"smiles", "canonical_smiles", "isomeric_smiles", "structure", "chromophore"}
        smiles_col = next((item for item in headers if item.lower() in aliases), "")
    if not smiles_col:
        raise RuntimeError("No SMILES column could be inferred")

    raw_specs = mapping.get("properties") if isinstance(mapping.get("properties"), list) else []
    specs = [dict(item) for item in raw_specs if isinstance(item, dict)]
    requested = [item.strip() for item in args.properties.split(",") if item.strip()]
    if requested:
        known = {str(item.get("property_id") or ""): item for item in specs}
        specs = [known.get(item) or {"property_id": property_id(item), "source_column": item} for item in requested]
    if not specs:
        excluded = {smiles_col.lower(), args.split_col.lower(), args.id_col.lower(), "dataset_id", "candidate_id", "id"}
        for header in headers:
            values = [row.get(header, "") for row in rows if str(row.get(header, "")).strip()]
            valid = [value for value in values if number(value) is not None]
            ratio = len(valid) / len(values) if values else 0.0
            if header.lower() not in excluded and len(values) >= args.min_nonempty and ratio >= args.min_numeric_ratio:
                specs.append({"property_id": property_id(header), "source_column": header})

    strict = not args.non_strict_rdkit
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid = duplicates = 0
    for source_row in rows:
        smiles = canonicalize(source_row.get(smiles_col, ""), strict)
        if not smiles:
            invalid += 1
            continue
        if smiles in seen:
            duplicates += 1
            continue
        row: dict[str, Any] = dict(source_row)
        row[smiles_col] = smiles
        for spec in specs:
            source = str(spec.get("source_column") or spec.get("property_id") or "")
            target = str(spec.get("property_id") or property_id(source))
            parsed = number(row.get(source, ""))
            if source != target:
                row[target] = "" if parsed is None else parsed
            elif parsed is not None:
                row[source] = parsed
        if args.drop_empty_target_rows and not any(number(row.get(str(spec.get("property_id") or ""), "")) is not None for spec in specs):
            continue
        seen.add(smiles)
        cleaned.append(row)

    output_headers = list(headers)
    for spec in specs:
        target = str(spec.get("property_id") or "")
        if target and target not in output_headers:
            output_headers.append(target)
    cleaned_csv = output_dir / f"{args.run_id}_cleaned_master.csv"
    with cleaned_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(cleaned)

    properties = []
    for spec in specs:
        target = str(spec.get("property_id") or "")
        source = str(spec.get("source_column") or target)
        valid = [row for row in cleaned if number(row.get(target, row.get(source, ""))) is not None]
        properties.append({"property_id": target, "source_column": source, "valid_count_deduped": len(valid), "numeric_ratio": len(valid) / len(cleaned) if cleaned else 0.0, "task_type": "numeric_regression"})
    catalog_path = output_dir / f"{args.run_id}_property_catalog.json"
    catalog_path.write_text(json.dumps({"run_id": args.run_id, "smiles_col": smiles_col, "row_count": len(cleaned), "properties": properties}, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / f"{args.run_id}_cleaning_report.json"
    report = {"run_id": args.run_id, "input_rows": len(rows), "output_rows": len(cleaned), "invalid_smiles": invalid, "duplicate_smiles": duplicates, "strict_smiles_cleaning": strict, "outputs": {"cleaned_master_csv": str(cleaned_csv), "property_catalog_json": str(catalog_path)}}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
