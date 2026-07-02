from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "C:\\",
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    ".pdf",
    "Authorization",
    "Bearer",
    "token",
    "secret",
    "password",
    "cookie",
    "x-api-key",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
    "SMILES",
    "InChI",
    "InChIKey",
    "C1=CC",
    "0.72",
)


def is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SAFE_ID_RE.fullmatch(value))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def is_valid_transition_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def redaction_errors(value: Any) -> tuple[str, ...]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    lowered = text.lower()
    return tuple(
        f"forbidden_marker:{marker}"
        for marker in FORBIDDEN_MARKERS
        if marker.lower() in lowered
    )
