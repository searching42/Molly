#!/usr/bin/env python3
"""Verify committed M3 representative inspection evidence without source paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _m3_inspection_validation import (  # noqa: E402
    CASE_ROSTER,
    CASE_FILENAME_BY_ID,
    EVIDENCE_VERSION,
    INSPECTION_VERSION,
    SOURCE_CLASSES,
    canonical_json_bytes,
    parse_canonical_json,
    public_privacy_violations,
    sha256_bytes,
)


def verify_evidence(root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    expected_top = {
        "README.md",
        "evidence_manifest.json",
        "evidence_summary.md",
        "review_checklist.md",
        "cases",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_top:
        raise ValueError("evidence package roster is invalid")
    case_dir = root / "cases"
    if {path.name for path in case_dir.iterdir()} != set(CASE_FILENAME_BY_ID.values()):
        raise ValueError("evidence case roster is invalid")
    package = b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())
    if public_privacy_violations(package):
        raise ValueError("evidence package privacy scan failed")
    manifest = parse_canonical_json((root / "evidence_manifest.json").read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("evidence manifest is invalid")
    if manifest.get("evidence_version") != EVIDENCE_VERSION or manifest.get("inspection_version") != INSPECTION_VERSION:
        raise ValueError("evidence contract version is invalid")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or [item.get("case_id") for item in cases if isinstance(item, dict)] != list(CASE_ROSTER):
        raise ValueError("evidence manifest case order is invalid")
    records = []
    for item in cases:
        if not isinstance(item, dict) or item.get("source_class") not in SOURCE_CLASSES:
            raise ValueError("evidence case metadata is invalid")
        case_path = root / str(item.get("case_file"))
        payload = case_path.read_bytes()
        if sha256_bytes(payload) != item.get("case_file_sha256"):
            raise ValueError("evidence case digest mismatch")
        record = parse_canonical_json(payload)
        if not isinstance(record, dict) or record.get("case_id") != item.get("case_id"):
            raise ValueError("evidence case identity mismatch")
        response = record.get("inspection_response")
        if response is not None and sha256_bytes(canonical_json_bytes(response)) != record.get("inspection_response_sha256"):
            raise ValueError("inspection response digest mismatch")
        if record.get("human_review_status") != "pending":
            raise ValueError("owner approval record is required before human review can change")
        records.append(record)
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("evidence summary is invalid")
    complete = all(record.get("machine_validation_status") == "passed" for record in records)
    if summary.get("machine_evidence_complete") is not complete:
        raise ValueError("machine evidence completeness claim is inconsistent")
    if summary.get("human_review_status") != "pending" or summary.get("m3_v_status") != "not_yet_claimed":
        raise ValueError("unreviewed evidence cannot claim M3 V")
    if require_complete and not complete:
        raise ValueError("machine evidence is incomplete")
    return {"case_count": len(records), "machine_evidence_complete": complete, "human_review_status": "pending"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_evidence(args.evidence_dir, require_complete=args.require_complete)
    except ValueError as exc:
        print(f"evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"case_count={result['case_count']}; machine_evidence_complete={str(result['machine_evidence_complete']).lower()}; human_review=pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
