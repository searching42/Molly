from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai4s_agent._utils import now_iso, read_csv_dict_rows, safe_float, write_json
from ai4s_agent.storage import ProjectStorage

try:
    from rdkit import Chem  # type: ignore
except Exception:
    Chem = None  # type: ignore


class DatasetRole(str, Enum):
    TRAIN = "train_dataset"
    EVALUATION = "evaluation_dataset"
    CANDIDATE = "candidate_dataset"


class DatasetRegistration(BaseModel):
    project_id: str
    dataset_id: str
    role: DatasetRole
    version: str
    source_path: Path
    registered_path: Path
    manifest_path: Path
    content_hash: str
    registered_at: str


class CsvStructure(BaseModel):
    path: str
    delimiter: str
    headers: list[str]
    row_count: int
    column_count: int


class PropertyCandidate(BaseModel):
    property_id: str
    source_column: str
    unit: str = ""
    canonical_unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    numeric_count: int
    nonempty_count: int
    numeric_ratio: float
    missing_rate: float
    min_value: float | None = None
    max_value: float | None = None
    median_value: float | None = None
    conversion_note: str = ""


class DuplicateConflict(BaseModel):
    canonical_smiles: str
    row_count: int
    conflicting_properties: list[str]
    row_indices: list[int]


class OutlierWarning(BaseModel):
    property_id: str
    method: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    outlier_count: int
    row_indices: list[int] = Field(default_factory=list)


class SplitAssessment(BaseModel):
    split_column: str = ""
    status: str
    split_counts: dict[str, int] = Field(default_factory=dict)
    fallback_strategy: str = ""
    fallback_counts: dict[str, int] = Field(default_factory=dict)
    reason: str = ""


class DatasetInspection(BaseModel):
    structure: CsvStructure
    smiles_column: str
    id_column: str = ""
    property_candidates: list[PropertyCandidate] = Field(default_factory=list)
    duplicate_conflicts: list[DuplicateConflict] = Field(default_factory=list)
    outlier_warnings: list[OutlierWarning] = Field(default_factory=list)
    split_assessment: SplitAssessment
    warnings: list[str] = Field(default_factory=list)


class LeakageReport(BaseModel):
    train_smiles_column: str
    other_smiles_column: str
    train_count: int
    other_count: int
    overlap_count: int
    overlap_smiles: list[str]


PROPERTY_ALIASES: dict[str, list[str]] = {
    "lambda_em": [
        "lambda_em",
        "lambda_em_nm",
        "emission_max_nm",
        "emission max",
        "emission wavelength",
    ],
    "plqy": [
        "plqy",
        "plqy_percent",
        "plqy_pct",
        "plqy (%)",
        "quantum_yield",
        "quantum yield",
    ],
    "mw": ["mw", "molecular_weight", "molecular weight", "mol_weight"],
    "homo": ["homo"],
    "lumo": ["lumo"],
    "gap": ["gap", "bandgap"],
}

KNOWN_RANGES: dict[str, tuple[float | None, float | None]] = {
    "lambda_em": (200.0, 1000.0),
    "plqy": (0.0, 1.0),
    "mw": (0.0, 3000.0),
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_dataset_id(source_path: Path) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in source_path.stem)
    token = "_".join(part for part in token.split("_") if part)
    return token or "dataset"


def _role_from_raw(role: DatasetRole | str) -> DatasetRole:
    if isinstance(role, DatasetRole):
        return role
    try:
        return DatasetRole(str(role))
    except ValueError as exc:
        raise ValueError(f"unknown dataset role: {role}") from exc


def _scope_for_role(role: DatasetRole, dataset_id: str) -> list[str]:
    if role == DatasetRole.CANDIDATE:
        return ["datasets", "candidates", dataset_id]
    return ["datasets", "raw", dataset_id]


def register_dataset(
    *,
    workspace_dir: Path,
    project_id: str,
    source_path: Path,
    role: DatasetRole | str,
    dataset_id: str | None = None,
) -> DatasetRegistration:
    source = source_path.expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"source dataset not found: {source}")

    parsed_role = _role_from_raw(role)
    clean_dataset_id = dataset_id or _safe_dataset_id(source)
    storage = ProjectStorage(workspace_dir=workspace_dir)
    version_dir = storage.create_asset_version_dir(project_id, _scope_for_role(parsed_role, clean_dataset_id))
    version = version_dir.name
    registered_path = version_dir / source.name
    shutil.copy2(source, registered_path)

    content_hash = _hash_file(registered_path)
    manifest_path = version_dir / "dataset_manifest.json"
    manifest = {
        "project_id": project_id,
        "dataset_id": clean_dataset_id,
        "role": parsed_role.value,
        "version": version,
        "source_path": str(source),
        "registered_path": str(registered_path),
        "content_hash": content_hash,
        "registered_at": now_iso(),
    }
    write_json(manifest_path, manifest)

    return DatasetRegistration(
        project_id=project_id,
        dataset_id=clean_dataset_id,
        role=parsed_role,
        version=version,
        source_path=source,
        registered_path=registered_path,
        manifest_path=manifest_path,
        content_hash=content_hash,
        registered_at=str(manifest["registered_at"]),
    )


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    if not sample.strip():
        return ","
    candidates = [",", "\t", ";", "|"]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        delimiter = str(getattr(dialect, "delimiter", ",") or ",")
        if delimiter in candidates:
            return delimiter
    except Exception:
        pass
    counts = {delimiter: sample.count(delimiter) for delimiter in candidates}
    delimiter, count = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
    return delimiter if count > 0 else ","


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    delimiter = detect_delimiter(path)
    rows, headers = read_csv_dict_rows(path, delimiter=delimiter)
    return rows, headers, delimiter


def detect_smiles_column(headers: list[str]) -> str:
    normalized = {_normalize_header(header): header for header in headers}
    for key in ["smiles", "canonicalsmiles", "molsmiles", "structure", "chromophore"]:
        if key in normalized:
            return normalized[key]
    return ""


def detect_split_column(headers: list[str]) -> str:
    normalized = {_normalize_header(header): header for header in headers}
    for key in ["splitgroup", "splithint", "split"]:
        if key in normalized:
            return normalized[key]
    return ""


def detect_id_column(headers: list[str]) -> str:
    normalized = {_normalize_header(header): header for header in headers}
    for key in ["datasetid", "candidateid", "molid", "id"]:
        if key in normalized:
            return normalized[key]
    return ""


def _normalize_header(text: str) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _canonical_property_id(header: str) -> str:
    key = _normalize_header(header)
    for prop, aliases in PROPERTY_ALIASES.items():
        for alias in aliases:
            alias_key = _normalize_header(alias)
            if key == alias_key or alias_key in key:
                return prop
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in header.strip())
    token = "_".join(part for part in token.split("_") if part)
    return token


def _unit_scale_from_header(header: str, values: list[float]) -> tuple[str, str, float, str]:
    key = header.lower()
    prop = _canonical_property_id(header)
    if "percent" in key or "%" in key or "pct" in key:
        return "percent", "fraction", 0.01, "suggest percent_to_fraction"
    if prop == "plqy" and values:
        median = sorted(values)[len(values) // 2]
        if median > 1.0:
            return "percent", "fraction", 0.01, "suggest percent_to_fraction"
        return "fraction", "fraction", 1.0, ""
    if "nm" in key or prop == "lambda_em":
        return "nm", "nm", 1.0, ""
    return "", "", 1.0, ""


def detect_property_candidates(
    rows: list[dict[str, str]],
    headers: list[str],
    *,
    smiles_column: str,
    split_column: str,
    id_column: str,
    min_numeric_ratio: float,
    min_nonempty: int,
) -> list[PropertyCandidate]:
    skipped = {
        smiles_column.lower(),
        split_column.lower(),
        id_column.lower(),
        "smiles",
        "canonical_smiles",
        "split",
        "split_group",
        "dataset_id",
        "candidate_id",
        "mol_id",
        "id",
    }
    candidates: list[PropertyCandidate] = []
    for header in headers:
        if header.lower() in skipped:
            continue
        nonempty = 0
        numeric_values: list[float] = []
        for row in rows:
            raw = row.get(header, "")
            if str(raw or "").strip():
                nonempty += 1
                parsed = safe_float(raw)
                if parsed is not None:
                    numeric_values.append(parsed)
        if nonempty < min_nonempty:
            continue
        ratio = len(numeric_values) / nonempty if nonempty else 0.0
        if ratio < min_numeric_ratio:
            continue
        unit, canonical_unit, scale, note = _unit_scale_from_header(header, numeric_values)
        scaled = [value * scale for value in numeric_values]
        candidates.append(
            PropertyCandidate(
                property_id=_canonical_property_id(header),
                source_column=header,
                unit=unit,
                canonical_unit=canonical_unit,
                scale=scale,
                numeric_count=len(numeric_values),
                nonempty_count=nonempty,
                numeric_ratio=round(ratio, 4),
                missing_rate=round((len(rows) - nonempty) / len(rows), 4) if rows else 0.0,
                min_value=min(scaled) if scaled else None,
                max_value=max(scaled) if scaled else None,
                median_value=_median(scaled),
                conversion_note=note,
            )
        )
    return candidates


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def canonicalize_smiles(smiles: str) -> str:
    raw = str(smiles or "").strip()
    if not raw:
        return ""
    if Chem is not None:
        mol = Chem.MolFromSmiles(raw)  # type: ignore[union-attr]
        if mol is not None:
            return str(Chem.MolToSmiles(mol, canonical=True))  # type: ignore[union-attr]
    return _fallback_canonical_smiles(raw)


def _fallback_canonical_smiles(smiles: str) -> str:
    # Fallback mode only handles simple linear atom-token strings. Branches,
    # rings, charges, bond orders, and stereochemistry require RDKit.
    if any(ch in smiles for ch in "()[]=#@+-\\/0123456789"):
        return smiles
    atoms: list[str] = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if not ch.isalpha() or not ch.isupper():
            return smiles
        atom = ch
        if i + 1 < len(smiles) and smiles[i + 1].islower():
            atom += smiles[i + 1]
            i += 1
        atoms.append(atom)
        i += 1
    if not atoms:
        return smiles
    reversed_smiles = "".join(reversed(atoms))
    return min(smiles, reversed_smiles)


def detect_duplicate_conflicts(
    rows: list[dict[str, str]],
    *,
    smiles_column: str,
    properties: list[PropertyCandidate],
    conflict_threshold: float = 0.2,
) -> list[DuplicateConflict]:
    by_smiles: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(rows, start=1):
        canonical = canonicalize_smiles(row.get(smiles_column, ""))
        if canonical:
            by_smiles[canonical].append((idx, row))

    conflicts: list[DuplicateConflict] = []
    for canonical, grouped in by_smiles.items():
        if len(grouped) < 2:
            continue
        conflict_props: list[str] = []
        for prop in properties:
            values = []
            for _, row in grouped:
                parsed = safe_float(row.get(prop.source_column))
                if parsed is not None:
                    values.append(parsed * prop.scale)
            if len(values) >= 2 and max(values) - min(values) > conflict_threshold:
                conflict_props.append(prop.property_id)
        if conflict_props:
            conflicts.append(
                DuplicateConflict(
                    canonical_smiles=canonical,
                    row_count=len(grouped),
                    conflicting_properties=conflict_props,
                    row_indices=[idx for idx, _ in grouped],
                )
            )
    return conflicts


def detect_outliers(
    rows: list[dict[str, str]],
    *,
    properties: list[PropertyCandidate],
) -> list[OutlierWarning]:
    warnings: list[OutlierWarning] = []
    for prop in properties:
        values_by_row: list[tuple[int, float]] = []
        for idx, row in enumerate(rows, start=1):
            parsed = safe_float(row.get(prop.source_column))
            if parsed is not None:
                values_by_row.append((idx, parsed * prop.scale))
        if not values_by_row:
            continue

        lower, upper = KNOWN_RANGES.get(prop.property_id, (None, None))
        method = "known_range"
        if lower is None and upper is None and len(values_by_row) >= 4:
            sorted_values = sorted(value for _, value in values_by_row)
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(len(sorted_values) * 3) // 4]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            method = "iqr"

        row_indices: list[int] = []
        if lower is not None or upper is not None:
            for idx, value in values_by_row:
                if lower is not None and value < lower:
                    row_indices.append(idx)
                elif upper is not None and value > upper:
                    row_indices.append(idx)
        if row_indices:
            warnings.append(
                OutlierWarning(
                    property_id=prop.property_id,
                    method=method,
                    lower_bound=lower,
                    upper_bound=upper,
                    outlier_count=len(row_indices),
                    row_indices=row_indices,
                )
            )
    return warnings


def assess_split(rows: list[dict[str, str]], split_column: str, smiles_column: str) -> SplitAssessment:
    if not split_column:
        return SplitAssessment(
            status="FALLBACK_PROPOSED",
            fallback_strategy="deterministic_hash",
            fallback_counts=_fallback_split_counts(rows, smiles_column),
            reason="missing_split_column",
        )

    raw_values = [str(row.get(split_column, "") or "").strip() for row in rows]
    nonempty = [value for value in raw_values if value]
    counts = dict(Counter(nonempty))
    normalized = {_normalize_split_value(value) for value in nonempty}
    valid = normalized.issubset({"train", "valid", "test", "0", "1", "2"})
    if not nonempty or not valid:
        status = "INVALID_SPLIT"
        reason = "invalid_split_values"
    elif len(nonempty) < len(rows):
        status = "PARTIAL_SPLIT"
        reason = "missing_split_values"
    else:
        status = "PROVIDED_SPLIT"
        reason = ""

    fallback_strategy = "deterministic_hash" if status != "PROVIDED_SPLIT" else ""
    fallback_counts = _fallback_split_counts(rows, smiles_column) if fallback_strategy else {}
    return SplitAssessment(
        split_column=split_column,
        status=status,
        split_counts=counts,
        fallback_strategy=fallback_strategy,
        fallback_counts=fallback_counts,
        reason=reason,
    )


def _normalize_split_value(value: str) -> str:
    token = value.strip().lower()
    aliases = {
        "training": "train",
        "tr": "train",
        "validation": "valid",
        "val": "valid",
        "testing": "test",
    }
    return aliases.get(token, token)


def _fallback_split_counts(rows: list[dict[str, str]], smiles_column: str) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    for row in rows:
        canonical = canonicalize_smiles(row.get(smiles_column, "")) if smiles_column else json.dumps(row, sort_keys=True)
        bucket = int(hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8], 16) % 10
        if bucket < 2:
            counts["test"] += 1
        elif bucket < 4:
            counts["valid"] += 1
        else:
            counts["train"] += 1
    return counts


def inspect_dataset(
    path: Path,
    *,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
) -> DatasetInspection:
    dataset_path = path.expanduser().resolve()
    rows, headers, delimiter = read_csv_rows(dataset_path)
    smiles_column = detect_smiles_column(headers)
    split_column = detect_split_column(headers)
    id_column = detect_id_column(headers)
    properties = detect_property_candidates(
        rows,
        headers,
        smiles_column=smiles_column,
        split_column=split_column,
        id_column=id_column,
        min_numeric_ratio=min_numeric_ratio,
        min_nonempty=min_nonempty,
    )
    warnings: list[str] = []
    if not smiles_column:
        warnings.append("missing_smiles_column")
    if not properties:
        warnings.append("no_property_candidates")

    return DatasetInspection(
        structure=CsvStructure(
            path=str(dataset_path),
            delimiter=delimiter,
            headers=headers,
            row_count=len(rows),
            column_count=len(headers),
        ),
        smiles_column=smiles_column,
        id_column=id_column,
        property_candidates=properties,
        duplicate_conflicts=detect_duplicate_conflicts(
            rows,
            smiles_column=smiles_column,
            properties=properties,
        )
        if smiles_column
        else [],
        outlier_warnings=detect_outliers(rows, properties=properties),
        split_assessment=assess_split(rows, split_column, smiles_column),
        warnings=warnings,
    )


def generate_property_catalog(inspection: DatasetInspection) -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "dataset_path": inspection.structure.path,
        "row_count": inspection.structure.row_count,
        "smiles_column": inspection.smiles_column,
        "split": inspection.split_assessment.model_dump(mode="json"),
        "properties": [
            {
                "property_id": prop.property_id,
                "source_column": prop.source_column,
                "label_count": prop.numeric_count,
                "missing_rate": prop.missing_rate,
                "unit": prop.unit,
                "canonical_unit": prop.canonical_unit,
                "scale": prop.scale,
                "offset": prop.offset,
                "min": prop.min_value,
                "max": prop.max_value,
                "median": prop.median_value,
                "conversion_note": prop.conversion_note,
            }
            for prop in inspection.property_candidates
        ],
        "duplicate_conflicts": [item.model_dump(mode="json") for item in inspection.duplicate_conflicts],
        "outlier_warnings": [item.model_dump(mode="json") for item in inspection.outlier_warnings],
    }


def check_smiles_leakage(train_csv: Path, other_csv: Path) -> LeakageReport:
    train_rows, train_headers, _ = read_csv_rows(train_csv.expanduser().resolve())
    other_rows, other_headers, _ = read_csv_rows(other_csv.expanduser().resolve())
    train_col = detect_smiles_column(train_headers)
    other_col = detect_smiles_column(other_headers)

    train_set = {
        canonicalize_smiles(row.get(train_col, ""))
        for row in train_rows
        if train_col and canonicalize_smiles(row.get(train_col, ""))
    }
    other_set = {
        canonicalize_smiles(row.get(other_col, ""))
        for row in other_rows
        if other_col and canonicalize_smiles(row.get(other_col, ""))
    }
    overlap = sorted(train_set & other_set)
    return LeakageReport(
        train_smiles_column=train_col,
        other_smiles_column=other_col,
        train_count=len(train_set),
        other_count=len(other_set),
        overlap_count=len(overlap),
        overlap_smiles=overlap,
    )
