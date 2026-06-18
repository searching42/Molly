from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return None
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    try:
        parsed = float(raw.replace(",", ""))
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed / 100.0 if percent else parsed


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}


PROTECTED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "input_csv",
        "train_csv",
        "cleaned_master_csv",
        "candidate_csv",
        "model_path",
        "model_dir",
        "property_catalog_json",
    }
)


def strict_smiles_cleaning_enabled(payload: dict[str, Any]) -> bool:
    if "strict_smiles_cleaning" in payload:
        return truthy(payload.get("strict_smiles_cleaning"))
    if "non_strict_rdkit" in payload:
        return not truthy(payload.get("non_strict_rdkit"))
    return True


def hash01(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) / float(16**12 - 1)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return path


def normalize_csv_fieldnames(fieldnames: list[str | None] | None) -> list[str]:
    return [str(header).strip() for header in (fieldnames or []) if str(header or "").strip()]


def read_csv_dict_rows(path: Path, *, delimiter: str = ",") -> tuple[list[dict[str, str]], list[str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        original_headers = list(reader.fieldnames or [])
        header_pairs = [
            (original, str(original).strip())
            for original in original_headers
            if str(original or "").strip()
        ]
        headers = [normalized for _, normalized in header_pairs]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row: dict[str, str] = {}
            for original, normalized in header_pairs:
                row[normalized] = str(raw_row.get(original) or "")
            rows.append(row)
    return rows, headers
