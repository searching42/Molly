from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.oled_review_packet_generator import generate_oled_review_packet


GENERATED_AT = "2026-07-09T00:00:00Z"


def test_generates_review_packet_from_all_candidate_artifacts(tmp_path: Path) -> None:
    artifacts = _write_candidate_artifacts(tmp_path)

    result = generate_oled_review_packet(
        run_id="review-run",
        output_dir=tmp_path / "review",
        oled_candidates_json=artifacts["oled_candidates_json"],
        oled_text_evidence_candidates_json=artifacts["oled_text_evidence_candidates_json"],
        oled_schema_candidates_json=artifacts["oled_schema_candidates_json"],
        oled_compiled_records_json=artifacts["oled_compiled_records_json"],
        corpus_extraction_manifest_json=artifacts["corpus_extraction_manifest_json"],
        generated_at=GENERATED_AT,
    )

    assert result.review_item_count == 4
    assert result.high_priority_count == 3
    assert result.low_priority_count == 1
    assert result.medium_priority_count == 0
    assert Path(result.review_packet_json).exists()
    assert Path(result.review_packet_md).exists()
    assert Path(result.reviewer_decision_template_json).exists()
    assert Path(result.review_summary_json).exists()
    assert not (tmp_path / "review" / "candidate_dataset.csv").exists()
    assert not (tmp_path / "review" / "training_dataset.csv").exists()

    packet = _read_json(Path(result.review_packet_json))
    item_ids = [item["review_item_id"] for item in packet["review_items"]]

    assert packet["schema_version"] == "oled_review_packet.v1"
    assert packet["summary"]["review_item_count"] == 4
    assert [item["candidate_type"] for item in packet["review_items"]] == [
        "oled_compiled_record",
        "oled_schema_candidate",
        "oled_text_evidence",
        "oled_raw_candidate",
    ]
    assert all(item_id.startswith("review:review-run:") for item_id in item_ids)

    text_item = next(item for item in packet["review_items"] if item["candidate_type"] == "oled_text_evidence")
    assert text_item["priority"] == "high"
    assert text_item["paper_id"] == "paper001"
    assert text_item["property_id"] == "plqy"
    assert text_item["raw_value"] == "94 ± 2%"
    assert text_item["numeric_value"] == 94.0
    assert text_item["unit"] == "%"
    assert text_item["compound_mentions"] == ["4CzIPN"]
    assert text_item["condition_text"] == "in toluene"
    assert text_item["evidence_page"] == 1
    assert text_item["evidence_location"] == "el_p1_0001"
    assert text_item["provenance"]["source_document_id"] == "paper001-source"

    schema_item = next(item for item in packet["review_items"] if item["candidate_type"] == "oled_schema_candidate")
    assert schema_item["priority"] == "high"
    assert schema_item["source_candidate_id"] == "schema-001"
    assert schema_item["property_id"] == "eqe_percent"
    assert schema_item["evidence_page"] == 3
    assert schema_item["provenance"]["source_candidate_hash"] == "raw-hash-001"

    compiled_item = next(item for item in packet["review_items"] if item["candidate_type"] == "oled_compiled_record")
    assert compiled_item["priority"] == "high"
    assert compiled_item["source_candidate_id"] == "compiled-001"
    assert compiled_item["property_id"] == "eqe_percent"
    assert compiled_item["evidence_page"] == 3
    assert compiled_item["material_roles"] == [{"role": "emitter", "material": "4CzIPN"}]
    assert "schema_warning" in compiled_item["warnings"]

    raw_item = next(item for item in packet["review_items"] if item["candidate_type"] == "oled_raw_candidate")
    assert raw_item["priority"] == "low"
    assert raw_item["evidence_text"] == "OLED device performance was measured."

    decision_template = _read_json(Path(result.reviewer_decision_template_json))
    assert [decision["review_item_id"] for decision in decision_template["decisions"]] == item_ids
    assert {decision["decision"] for decision in decision_template["decisions"]} == {""}
    assert {decision["review_status"] for decision in decision_template["decisions"]} == {"pending"}

    summary = _read_json(Path(result.review_summary_json))
    assert summary["counts_by_candidate_type"] == {
        "oled_compiled_record": 1,
        "oled_raw_candidate": 1,
        "oled_schema_candidate": 1,
        "oled_text_evidence": 1,
    }
    assert summary["counts_by_priority"] == {"high": 3, "low": 1}
    assert summary["counts_by_paper"] == {"paper001": 4}
    assert summary["counts_by_property_id"] == {"eqe_percent": 2, "plqy": 1}
    assert "candidate_only_review_packet" in summary["governance_notes"]

    markdown = Path(result.review_packet_md).read_text(encoding="utf-8")
    assert "# OLED Evidence Review Packet" in markdown
    assert "this packet is candidate-only" in markdown
    assert "oled_reviewer_decision_template.json" in markdown
    assert "4CzIPN showed a photoluminescence quantum yield" in markdown


def test_review_packet_ids_and_order_are_stable(tmp_path: Path) -> None:
    artifacts = _write_candidate_artifacts(tmp_path)

    first = generate_oled_review_packet(
        run_id="review-run",
        output_dir=tmp_path / "first",
        generated_at=GENERATED_AT,
        **artifacts,
    )
    second = generate_oled_review_packet(
        run_id="review-run",
        output_dir=tmp_path / "second",
        generated_at=GENERATED_AT,
        **artifacts,
    )

    first_packet = _read_json(Path(first.review_packet_json))
    second_packet = _read_json(Path(second.review_packet_json))

    assert first_packet["review_items"] == second_packet["review_items"]
    assert first_packet["summary"] == second_packet["summary"]


def test_handles_missing_candidate_artifacts_as_empty_packet_with_warnings(tmp_path: Path) -> None:
    result = generate_oled_review_packet(
        run_id="empty-run",
        output_dir=tmp_path / "review",
        oled_candidates_json=tmp_path / "missing_oled_candidates.json",
        oled_text_evidence_candidates_json=tmp_path / "missing_text.json",
        oled_schema_candidates_json=tmp_path / "missing_schema.json",
        oled_compiled_records_json=tmp_path / "missing_compiled.json",
        corpus_extraction_manifest_json=tmp_path / "missing_manifest.json",
        generated_at=GENERATED_AT,
    )

    packet = _read_json(Path(result.review_packet_json))
    summary = _read_json(Path(result.review_summary_json))
    decisions = _read_json(Path(result.reviewer_decision_template_json))

    assert result.review_item_count == 0
    assert packet["review_items"] == []
    assert packet["summary"]["warnings"]
    assert summary["review_item_count"] == 0
    assert summary["warnings"]
    assert decisions["decisions"] == []
    assert "No review items generated" in Path(result.review_packet_md).read_text(encoding="utf-8")


def _write_candidate_artifacts(tmp_path: Path) -> dict[str, Path]:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    artifacts = {
        "oled_candidates_json": extraction / "oled_candidates.json",
        "oled_text_evidence_candidates_json": extraction / "oled_text_evidence_candidates.json",
        "oled_schema_candidates_json": extraction / "oled_schema_candidates.json",
        "oled_compiled_records_json": extraction / "oled_compiled_records.json",
        "corpus_extraction_manifest_json": extraction / "corpus_extraction_manifest.json",
    }
    _write_json(
        artifacts["oled_candidates_json"],
        {
            "run_id": "review-run",
            "candidates": [
                {
                    "paper_id": "paper001",
                    "candidate_type": "text",
                    "page_index": 1,
                    "block_index": 4,
                    "block_id": "raw-block-001",
                    "raw_text": "OLED device performance was measured.",
                    "evidence_anchor": "paper001:p1:block4",
                    "candidate_hash": "raw-hash-001",
                    "matched_terms": ["oled"],
                    "relevance_signals": ["oled_keyword"],
                    "source_format": "mineru_like",
                    "table_parse_status": "not_table",
                }
            ],
        },
    )
    _write_json(
        artifacts["oled_text_evidence_candidates_json"],
        {
            "run_id": "review-run",
            "text_evidence_candidates": [
                {
                    "candidate_id": "text-001",
                    "paper_id": "paper001",
                    "source_document_id": "paper001-source",
                    "source_path": "/tmp/paper001.pdf",
                    "page": 1,
                    "element_id": "el_p1_0001",
                    "evidence_text": "4CzIPN showed a photoluminescence quantum yield of 94 ± 2% in toluene.",
                    "evidence_span": {"start": 14, "end": 65},
                    "compound_mentions": ["4CzIPN"],
                    "property_id": "plqy",
                    "property_label": "photoluminescence quantum yield",
                    "raw_value": "94 ± 2%",
                    "numeric_value": 94.0,
                    "unit": "%",
                    "condition_text": "in toluene",
                    "confidence": 0.86,
                    "extraction_method": "deterministic_oled_text_evidence_v1",
                    "provenance": {"source_document_id": "paper001-source", "paper_id": "paper001"},
                }
            ],
        },
    )
    _write_json(
        artifacts["oled_schema_candidates_json"],
        {
            "run_id": "review-run",
            "schema_candidates": [
                {
                    "candidate_id": "schema-001",
                    "candidate_type": "property_observation",
                    "status": "proposed",
                    "source_paper_id": "paper001",
                    "source_candidate_hash": "raw-hash-001",
                    "source_evidence_anchor": "paper001:p3:table1:row1",
                    "target_layer": "device",
                    "property_id": "eqe_percent",
                    "property_label": "external quantum efficiency",
                    "value": 21.3,
                    "unit": "%",
                    "evidence_refs": [
                        {
                            "source_candidate_hash": "raw-hash-001",
                            "source_evidence_anchor": "paper001:p3:table1:row1",
                            "source_candidate_type": "table",
                            "row_index": 1,
                            "column_name": "EQE",
                            "cell_value": "21.3%",
                        }
                    ],
                    "confidence_score": 0.91,
                    "metadata": {"page": 3},
                }
            ],
        },
    )
    _write_json(
        artifacts["oled_compiled_records_json"],
        {
            "run_id": "review-run",
            "compiled_records": [
                {
                    "record_id": "compiled-001",
                    "status": "compiled",
                    "group_key": {
                        "source_paper_id": "paper001",
                        "source_candidate_hashes": ["raw-hash-001"],
                        "row_index": 1,
                        "target_property_ids": ["eqe_percent"],
                    },
                    "layered_record": {
                        "interaction": {
                            "metadata": {
                                "material_roles": [{"role": "emitter", "material": "4CzIPN"}],
                            }
                        },
                        "device": {
                            "device_stack": ["ITO", "TAPC", "emissive layer", "LiF/Al"],
                            "properties": [
                                {
                                    "property_label": "external quantum efficiency",
                                    "value": 21.3,
                                    "unit": "%",
                                    "metadata": {"property_id": "eqe_percent"},
                                }
                            ],
                        },
                    },
                    "source_schema_candidate_ids": ["schema-001"],
                    "source_candidate_hashes": ["raw-hash-001"],
                    "source_evidence_anchors": ["paper001:p3:table1:row1"],
                    "schema_warning_codes": ["schema_warning"],
                    "confidence_score": 0.88,
                }
            ],
        },
    )
    _write_json(
        artifacts["corpus_extraction_manifest_json"],
        {
            "run_id": "review-run",
            "report": {"oled_candidate_count": 1},
            "artifacts": {key: str(path) for key, path in artifacts.items()},
        },
    )
    return artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
