from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ai4s_agent.adapters.phase1 import check_trainability_service, inspect_dataset_service
from ai4s_agent.schemas import (
    ConflictReport,
    ExtractionBenchmarkReport,
    ExtractedRecord,
    MergedRecord,
    ParsedDocument,
    UnitNormalizationReport,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_literature_dataset_demo"
RUN_ID = "phase3-literature-dataset-demo"
GENERATED_AT = "2026-06-24T00:00:00Z"


def test_phase3_literature_fixture_extracts_records(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    parsed = ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))
    extracted = [ExtractedRecord.model_validate(item) for item in _read_json(result["extracted_records_json"])["records"]]
    rejected = _read_json(result["rejected_records_json"])["records"]
    reasons = Counter(str(item["reason"]) for item in rejected)

    assert parsed.tables[0].table_id == "table_plqy_001"
    assert len(extracted) == 7
    assert all(record.source_id == "phase3-oled-fixture-paper" for record in extracted)
    assert all(record.evidence_ref == "phase3-oled-fixture-paper:table_plqy_001" for record in extracted)
    assert reasons["missing_smiles"] == 1
    assert reasons["low_confidence"] == 1
    assert result["summary"]["extracted_record_count"] == 7
    assert result["summary"]["rejected_record_count"] == 2


def test_phase3_literature_fixture_normalizes_units(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    report = UnitNormalizationReport.model_validate(_read_json(result["unit_normalization_report_json"]))
    normalized = [ExtractedRecord.model_validate(item) for item in _read_json(result["normalized_records_json"])["records"]]
    emitter_a = next(record for record in normalized if record.record_id == "rec_000001")

    assert emitter_a.properties["plqy"] == 0.65
    assert report.conversion_count == 4
    assert report.normalized_record_count == 7
    assert report.conversions[0]["rule"] == "percent_to_fraction"
    assert any(item["source_value"] == "65%" and item["canonical_value"] == 0.65 for item in report.conversions)


def test_phase3_literature_fixture_writes_confirmed_dataset(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    confirmed = Path(result["confirmed_dataset_csv"])
    rows = _csv_rows(confirmed)

    assert confirmed.exists()
    assert len(rows) == 6
    assert {"smiles", "plqy", "source_id", "paper_id", "evidence_ref", "confidence"} <= set(rows[0])
    assert rows[0]["smiles"]
    assert rows[0]["plqy"]
    assert rows[0]["evidence_ref"]


def test_phase3_literature_fixture_conflict_report(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    conflict_report = ConflictReport.model_validate(_read_json(result["conflict_report_json"]))
    merged = [MergedRecord.model_validate(item) for item in _read_json(result["merged_records_json"])["records"]]
    cco = next(record for record in merged if record.smiles == "CCO")

    assert conflict_report.input_record_count == 7
    assert conflict_report.merged_record_count == 6
    assert conflict_report.conflict_count == 0
    assert conflict_report.non_conflicting_record_count == 6
    assert cco.properties["plqy"] == 0.655
    assert cco.source_record_ids == ["rec_000001", "rec_000007"]


def test_phase3_literature_dataset_feeds_phase1_trainability(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    inspect = inspect_dataset_service(
        {
            "input_csv": result["confirmed_dataset_csv"],
            "min_numeric_ratio": 0.8,
            "min_nonempty": 2,
        }
    )
    assert inspect["status"] == "success"
    property_ids = {item["property_id"] for item in inspect["property_candidates"]}
    assert "plqy" in property_ids

    properties = [
        {
            "property_id": item["property_id"],
            "effective_labels": item["nonempty_count"],
            "numeric_ratio": item["numeric_ratio"],
        }
        for item in inspect["property_candidates"]
    ]
    trainability = check_trainability_service({"properties": properties})
    report = trainability["trainability_report"]

    assert trainability["status"] == "success"
    assert any(item["property_id"] == "plqy" for item in report["properties"])


def test_phase3_literature_fixture_report(tmp_path: Path) -> None:
    result = _run_phase3_fixture(tmp_path)

    report_json = _read_json(result["report_json"])
    report_md = Path(result["report_md"]).read_text(encoding="utf-8")
    benchmark = ExtractionBenchmarkReport.model_validate(_read_json(result["extraction_benchmark_report_json"]))

    assert benchmark.trainable_labels_gained == 6
    assert benchmark.counts["confirmed_records"] == 6
    assert report_json["summary"]["extracted_record_count"] == 7
    assert report_json["summary"]["normalization_conversion_count"] == 4
    assert report_json["summary"]["conflict_count"] == 0
    assert report_json["artifacts"]["confirmed_dataset_csv"] == result["confirmed_dataset_csv"]
    assert "Phase 3 literature-to-dataset fixture completed" in report_md
    assert "confirmed_dataset.csv" in report_md


def _run_phase3_fixture(tmp_path: Path) -> dict[str, Any]:
    output_dir = tmp_path / RUN_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))
    config = _read_json(FIXTURE_DIR / "extraction_config.json")
    target = _read_json(FIXTURE_DIR / "target_property.json")
    confidence_threshold = float(config["confidence_threshold"])

    extracted, rejected = _extract_fixture_records(parsed, confidence_threshold=confidence_threshold)
    normalized, normalization_report = _normalize_records(extracted)
    merged, conflict_report = _merge_records(normalized, tolerance=float(config["conflict_tolerance"]))

    extracted_json = _write_json(output_dir / "extracted_records.json", {"records": [record.model_dump(mode="json") for record in extracted]})
    extracted_jsonl = _write_jsonl(output_dir / "extracted_records.jsonl", [record.model_dump(mode="json") for record in extracted])
    rejected_json = _write_json(output_dir / "rejected_records.json", {"records": rejected})
    normalized_json = _write_json(output_dir / "normalized_records.json", {"records": [record.model_dump(mode="json") for record in normalized]})
    unit_report_json = _write_json(output_dir / "unit_normalization_report.json", normalization_report.model_dump(mode="json"))
    merged_json = _write_json(output_dir / "merged_records.json", {"records": [record.model_dump(mode="json") for record in merged]})
    conflict_report_json = _write_json(output_dir / "conflict_report.json", conflict_report.model_dump(mode="json"))
    confirmed_csv = _write_confirmed_dataset(output_dir / "confirmed_dataset.csv", merged, normalized)
    benchmark_report = _build_benchmark_report(
        extracted_count=len(extracted),
        rejected_count=len(rejected),
        confirmed_count=len(_csv_rows(confirmed_csv)),
        conflict_count=conflict_report.conflict_count,
    )
    benchmark_report_json = _write_json(output_dir / "extraction_benchmark_report.json", benchmark_report.model_dump(mode="json"))
    report_json, report_md = _write_fixture_report(
        output_dir=output_dir,
        target_property=target,
        extracted_count=len(extracted),
        rejected_count=len(rejected),
        normalization_report=normalization_report,
        conflict_report=conflict_report,
        confirmed_dataset=confirmed_csv,
        benchmark_report=benchmark_report_json,
    )

    return {
        "extracted_records_json": str(extracted_json),
        "extracted_records_jsonl": str(extracted_jsonl),
        "rejected_records_json": str(rejected_json),
        "normalized_records_json": str(normalized_json),
        "unit_normalization_report_json": str(unit_report_json),
        "merged_records_json": str(merged_json),
        "conflict_report_json": str(conflict_report_json),
        "confirmed_dataset_csv": str(confirmed_csv),
        "extraction_benchmark_report_json": str(benchmark_report_json),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "summary": {
            "extracted_record_count": len(extracted),
            "rejected_record_count": len(rejected),
            "confirmed_record_count": len(_csv_rows(confirmed_csv)),
        },
    }


def _extract_fixture_records(
    parsed: ParsedDocument,
    *,
    confidence_threshold: float,
) -> tuple[list[ExtractedRecord], list[dict[str, Any]]]:
    table = parsed.tables[0]
    records: list[ExtractedRecord] = []
    rejected: list[dict[str, Any]] = []
    for row_index, row in enumerate(table.rows):
        confidence = float(row["confidence"])
        smiles = str(row.get("smiles") or "").strip()
        attempt = {
            "paper_id": parsed.paper_id,
            "source_id": parsed.paper_id,
            "table_id": table.table_id,
            "row_index": row_index,
            "raw_values": dict(row),
        }
        if not smiles:
            rejected.append({**attempt, "reason": "missing_smiles"})
            continue
        if confidence < confidence_threshold:
            rejected.append({**attempt, "smiles": smiles, "reason": "low_confidence", "confidence": confidence})
            continue
        property_id, value = _parse_plqy_value(row["PLQY"])
        record = ExtractedRecord(
            record_id=f"rec_{len(records) + 1:06d}",
            smiles=smiles,
            properties={property_id: value},
            source_id=parsed.paper_id,
            paper_id=parsed.paper_id,
            page=table.page,
            table_id=table.table_id,
            row_index=row_index,
            evidence_ref=f"{parsed.paper_id}:{table.table_id}",
            citation_context=f"{parsed.paper_id} p.{table.page} {table.table_id} row {row_index + 1}",
            confidence=confidence,
            confidence_factors={
                "has_smiles": True,
                "has_plqy": True,
                "table_fixture": True,
                "source": "phase3_fixture_table_extractor",
            },
            raw_values={str(key): str(value) for key, value in row.items()},
        )
        records.append(record)
    return records, rejected


def _normalize_records(records: list[ExtractedRecord]) -> tuple[list[ExtractedRecord], UnitNormalizationReport]:
    normalized: list[ExtractedRecord] = []
    conversions: list[dict[str, Any]] = []
    for record in records:
        properties: dict[str, float] = {}
        converted = False
        for property_id, value in record.properties.items():
            if property_id == "plqy_percent":
                canonical = round(value / 100.0, 12)
                properties["plqy"] = canonical
                converted = True
                conversions.append(
                    {
                        "record_id": record.record_id,
                        "property_id": "plqy",
                        "source_unit": "%",
                        "canonical_unit": "fraction",
                        "source_value": record.raw_values["PLQY"],
                        "canonical_value": canonical,
                        "rule": "percent_to_fraction",
                    }
                )
            else:
                properties[property_id] = value
        data = record.model_dump(mode="json")
        data["properties"] = properties
        data["confidence_factors"] = {**record.confidence_factors, "unit_normalized": converted}
        normalized.append(ExtractedRecord.model_validate(data))
    report = UnitNormalizationReport(
        run_id=RUN_ID,
        input_record_count=len(records),
        normalized_record_count=len(normalized),
        conversion_count=len(conversions),
        warning_count=0,
        conversions=conversions,
        warnings=[],
        generated_at=GENERATED_AT,
        notes=["Fixture normalizes PLQY percent values to canonical fraction units."],
    )
    return normalized, report


def _merge_records(records: list[ExtractedRecord], *, tolerance: float) -> tuple[list[MergedRecord], ConflictReport]:
    groups: dict[str, list[ExtractedRecord]] = defaultdict(list)
    for record in records:
        groups[record.smiles].append(record)

    merged: list[MergedRecord] = []
    for smiles in sorted(groups):
        group = groups[smiles]
        values = [record.properties["plqy"] for record in group]
        if max(values) - min(values) > tolerance:
            status = "conflict"
            properties: dict[str, float] = {}
            property_status = {"plqy": "conflict"}
            conflict_ids = [f"conflict_{len(merged) + 1:06d}"]
        else:
            status = "merged"
            properties = {"plqy": round(sum(values) / len(values), 6)}
            property_status = {"plqy": "merged" if len(group) > 1 else "single_source"}
            conflict_ids = []
        merged.append(
            MergedRecord(
                merge_id=f"merge_{len(merged) + 1:06d}",
                smiles=smiles,
                properties=properties,
                property_status=property_status,
                source_record_ids=[record.record_id for record in group],
                source_ids=sorted({record.source_id for record in group}),
                citations=[record.citation_context for record in group],
                confidence=round(sum(record.confidence for record in group) / len(group), 6),
                conflict_ids=conflict_ids,
                status=status,
            )
        )
    conflict_count = sum(1 for record in merged if record.status == "conflict")
    report = ConflictReport(
        run_id=RUN_ID,
        input_record_count=len(records),
        merged_record_count=len(merged),
        conflict_count=conflict_count,
        non_conflicting_record_count=sum(1 for record in merged if record.status == "merged"),
        conflicts=[],
        generated_at=GENERATED_AT,
        notes=[f"Fixture groups by exact SMILES with plqy tolerance {tolerance}."],
    )
    return merged, report


def _write_confirmed_dataset(path: Path, records: list[MergedRecord], normalized: list[ExtractedRecord]) -> Path:
    by_id = {record.record_id: record for record in normalized}
    fieldnames = ["smiles", "plqy", "source_id", "paper_id", "evidence_ref", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record.status != "merged" or "plqy" not in record.properties:
                continue
            source_records = [by_id[record_id] for record_id in record.source_record_ids]
            writer.writerow(
                {
                    "smiles": record.smiles,
                    "plqy": record.properties["plqy"],
                    "source_id": ";".join(record.source_ids),
                    "paper_id": ";".join(sorted({item.paper_id for item in source_records})),
                    "evidence_ref": ";".join(item.evidence_ref for item in source_records),
                    "confidence": record.confidence,
                }
            )
    return path


def _build_benchmark_report(
    *,
    extracted_count: int,
    rejected_count: int,
    confirmed_count: int,
    conflict_count: int,
) -> ExtractionBenchmarkReport:
    attempted = extracted_count + rejected_count
    return ExtractionBenchmarkReport(
        run_id=RUN_ID,
        retrieval_recall=1.0,
        extraction_precision=round(extracted_count / attempted, 6),
        conflict_rate=round(conflict_count / max(extracted_count, 1), 6),
        confirmation_workload_count=confirmed_count,
        trainable_labels_gained=confirmed_count,
        downstream_model_performance_delta={},
        metric_statuses={
            "retrieval_recall": "fixture_known",
            "extraction_precision": "fixture_known",
            "trainability": "checked_by_phase1_helper",
        },
        counts={
            "attempted_rows": attempted,
            "extracted_records": extracted_count,
            "rejected_records": rejected_count,
            "confirmed_records": confirmed_count,
        },
        generated_at=GENERATED_AT,
        notes=["Fixture benchmark measures deterministic table extraction only."],
    )


def _write_fixture_report(
    *,
    output_dir: Path,
    target_property: dict[str, Any],
    extracted_count: int,
    rejected_count: int,
    normalization_report: UnitNormalizationReport,
    conflict_report: ConflictReport,
    confirmed_dataset: Path,
    benchmark_report: Path,
) -> tuple[Path, Path]:
    summary = {
        "target_property": target_property["property_id"],
        "extracted_record_count": extracted_count,
        "rejected_record_count": rejected_count,
        "normalization_conversion_count": normalization_report.conversion_count,
        "conflict_count": conflict_report.conflict_count,
        "confirmed_dataset_csv": str(confirmed_dataset),
    }
    artifacts = {
        "unit_normalization_report_json": str(output_dir / "unit_normalization_report.json"),
        "conflict_report_json": str(output_dir / "conflict_report.json"),
        "confirmed_dataset_csv": str(confirmed_dataset),
        "extraction_benchmark_report_json": str(benchmark_report),
    }
    report_json = _write_json(output_dir / "report.json", {"summary": summary, "artifacts": artifacts})
    lines = [
        "# Phase 3 Literature Dataset Fixture",
        "",
        "Phase 3 literature-to-dataset fixture completed.",
        "",
        f"- Extracted records: {extracted_count}",
        f"- Rejected records: {rejected_count}",
        f"- Unit conversions: {normalization_report.conversion_count}",
        f"- Conflicts: {conflict_report.conflict_count}",
        f"- Confirmed dataset: `{confirmed_dataset.name}`",
        f"- Benchmark report: `{benchmark_report.name}`",
        "",
        "This fixture validates structured extraction, normalization, provenance, and Phase 1 trainability intake only.",
    ]
    report_md = output_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_md


def _parse_plqy_value(raw: str) -> tuple[str, float]:
    clean = raw.strip().replace(" ", "")
    if clean.endswith("%"):
        return "plqy_percent", float(clean[:-1])
    return "plqy", float(clean)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
