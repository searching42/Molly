from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledCausalLayer,
    OledMineruAcceptanceManifest,
    OledMineruParsedBundle,
    OledMineruReviewPacket,
    OledMineruReviewPacketReport,
    OledReviewDecision,
    OledReviewPacketMaterialRole,
    OledReviewPacketProperty,
    OledReviewPacketSourceRef,
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    build_oled_mineru_review_packets_from_compiled_records as package_build_oled_mineru_review_packets_from_compiled_records,
    compile_oled_schema_candidates_to_layered_records,
    run_oled_mineru_review_packet_builder as package_run_oled_mineru_review_packet_builder,
    write_oled_mineru_review_packets_jsonl as package_write_oled_mineru_review_packets_jsonl,
    write_oled_mineru_review_packets_markdown as package_write_oled_mineru_review_packets_markdown,
)
from ai4s_agent.domains.oled_mineru_review_packets import (
    build_oled_mineru_review_packets_from_compiled_records,
    main,
    run_oled_mineru_review_packet_builder,
    write_oled_mineru_review_packets_jsonl,
    write_oled_mineru_review_packets_markdown,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _content_list_payload() -> list[dict[str, object]]:
    return [
        {
            "type": "table",
            "table_caption": "Table 1. OLED photophysical properties.",
            "table_body": (
                "| Host | Emitter dopant | PLQY (%) |\n"
                "| --- | --- | --- |\n"
                "| mCBP | D1 | 82 |"
            ),
            "page_idx": 2,
        }
    ]


def _evidence_ref(
    *,
    row_index: int = 0,
    column_name: str = "PLQY (%)",
    cell_value: str = "82",
) -> OledSchemaEvidenceRef:
    return OledSchemaEvidenceRef(
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p2:b0:table",
        source_candidate_type="table",
        row_index=row_index,
        column_name=column_name,
        cell_value=cell_value,
    )


def _role_candidate(role: str, name: str) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=f"schema:hash-table:row-0:material-{role}",
        candidate_type=OledSchemaCandidateType.MATERIAL_ROLE,
        source_paper_id="paper-review",
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p2:b0:table",
        material_role=role,
        material_name=name,
        evidence_refs=[_evidence_ref(column_name=role, cell_value=name)],
        confidence_score=0.72,
    )


def _property_candidate(
    property_id: str = "plqy",
    label: str = "PLQY",
    value: float | int | str = 82,
    unit: str = "%",
    *,
    row_index: int = 0,
    target_layer: OledCausalLayer = OledCausalLayer.INTERACTION,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=f"schema:hash-table:row-{row_index}:{property_id}",
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-review",
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p2:b0:table",
        target_layer=target_layer,
        property_id=property_id,
        property_label=label,
        value=value,
        unit=unit,
        evidence_refs=[
            _evidence_ref(
                row_index=row_index,
                column_name="PLQY (%)" if property_id == "plqy" else "Max EQE (%)",
                cell_value=str(value),
            )
        ],
        confidence_score=0.74,
        metadata={"source_caption": "Table 1. OLED photophysical properties."},
    )


def _compiled_records():
    report = compile_oled_schema_candidates_to_layered_records(
        [
            _role_candidate("host", "mCBP"),
            _role_candidate("emitter_dopant", "D1"),
            _property_candidate(),
        ]
    )
    return report.compiled_records


def test_build_packets_from_compiled_records_extracts_review_fields() -> None:
    packets = build_oled_mineru_review_packets_from_compiled_records(
        _compiled_records(),
        paper_id="paper-review",
        source_label="synthetic-source",
    )

    assert len(packets) == 1
    packet = packets[0]
    assert packet.review_decision == OledReviewDecision.UNREVIEWED
    assert packet.paper_id == "paper-review"
    assert packet.source_label == "synthetic-source"
    assert packet.material_roles
    assert {role.role for role in packet.material_roles} >= {"host", "emitter_dopant"}
    assert any(role.material_name == "mCBP" for role in packet.material_roles)
    assert packet.properties
    assert packet.properties[0].property_id == "plqy"
    assert packet.properties[0].property_label == "PLQY"
    assert packet.properties[0].evidence_refs[0].row_index == 0
    assert packet.properties[0].evidence_refs[0].column_name == "PLQY (%)"
    assert "layered_record" not in packet.model_dump(mode="json")


def test_end_to_end_manifest_runner_builds_review_packets(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-001_content_list.json", _content_list_payload())
    md_path = tmp_path / "paper-001.md"
    md_path.write_text("Nearby context for OLED Table 1 only.", encoding="utf-8")
    manifest = OledMineruAcceptanceManifest(
        manifest_id="review-smoke",
        bundles=[
            OledMineruParsedBundle(
                paper_id="paper-001",
                content_list_path=str(content_path),
                md_path=str(md_path),
                source_label="synthetic-paper",
            )
        ],
    )

    report = run_oled_mineru_review_packet_builder(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert report.is_valid is True
    assert report.paper_count == 1
    assert report.packet_count > 0
    assert report.metadata["gold_records_created"] is False
    assert report.metadata["curated_dataset_written"] is False
    assert report.packets[0].review_decision == OledReviewDecision.UNREVIEWED


def test_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    packets = build_oled_mineru_review_packets_from_compiled_records(
        _compiled_records(),
        paper_id="paper-review",
    )
    output_path = tmp_path / "review_packets.jsonl"

    write_oled_mineru_review_packets_jsonl(packets, output_path)
    text = output_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert len(lines) == len(packets)
    payload = json.loads(lines[0])
    assert lines[0] == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert str(tmp_path) not in text
    assert "mCBP | D1 | 82" not in text
    assert "layered_record" not in text


def test_markdown_writer_outputs_checklist_without_absolute_paths(tmp_path: Path) -> None:
    report = OledMineruReviewPacketReport(
        manifest_id="md-report",
        status="completed_with_warnings",
        paper_count=1,
        packet_count=1,
        packets_by_status={"partial": 1},
        finding_code_counts={"missing_smiles": 1},
        packets=build_oled_mineru_review_packets_from_compiled_records(
            _compiled_records(),
            paper_id="paper-review",
        ),
    )
    output_path = tmp_path / "review_packets.md"

    write_oled_mineru_review_packets_markdown(report, output_path)
    text = output_path.read_text(encoding="utf-8")

    assert "## Packet" in text
    assert "paper-review" in text
    assert "Review decision: unreviewed" in text
    assert "PLQY" in text
    assert "Source anchors" in text
    assert str(tmp_path) not in text


def test_runner_requires_explicit_confirmation(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-001_content_list.json", _content_list_payload())
    manifest = OledMineruAcceptanceManifest(
        manifest_id="confirm-gate",
        bundles=[OledMineruParsedBundle(paper_id="paper-001", content_list_path=str(content_path))],
    )

    with pytest.raises(ValueError, match="confirmation_required:read_only_parsed_outputs"):
        run_oled_mineru_review_packet_builder(manifest)


def test_max_packets_per_paper_is_deterministic(tmp_path: Path) -> None:
    content_path = _write_json(
        tmp_path / "paper-rows_content_list.json",
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED photophysical properties.",
                "table_body": (
                    "| Host | Emitter dopant | PLQY (%) |\n"
                    "| --- | --- | --- |\n"
                    "| mCBP | D1 | 82 |\n"
                    "| DPEPO | D2 | 76 |"
                ),
            }
        ],
    )
    manifest = OledMineruAcceptanceManifest(
        manifest_id="max-packets",
        bundles=[OledMineruParsedBundle(paper_id="paper-rows", content_list_path=str(content_path))],
    )

    report = run_oled_mineru_review_packet_builder(
        manifest,
        confirm_read_only_parsed_outputs=True,
        max_packets_per_paper=1,
    )

    assert report.packet_count == 1
    assert [packet.packet_id for packet in report.packets] == sorted(packet.packet_id for packet in report.packets)


def test_cli_smoke_writes_jsonl_and_markdown(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-cli_content_list.json", _content_list_payload())
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {
            "manifest_id": "cli-review",
            "bundles": [
                {
                    "paper_id": "paper-cli",
                    "content_list_path": content_path.name,
                    "source_label": "synthetic-cli",
                }
            ],
        },
    )
    jsonl_path = tmp_path / "review_packets.jsonl"
    md_path = tmp_path / "review_packets.md"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--output-jsonl",
            str(jsonl_path),
            "--output-md",
            str(md_path),
            "--confirm-read-only-parsed-outputs",
        ]
    )

    assert exit_code == 0
    assert jsonl_path.exists()
    assert md_path.exists()
    assert jsonl_path.read_text(encoding="utf-8").strip()
    assert "paper-cli" in md_path.read_text(encoding="utf-8")


def test_public_review_packet_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    packets = package_build_oled_mineru_review_packets_from_compiled_records(
        _compiled_records(),
        paper_id="paper-review",
    )
    output_jsonl = tmp_path / "package-review.jsonl"
    output_md = tmp_path / "package-review.md"
    package_write_oled_mineru_review_packets_jsonl(packets, output_jsonl)
    package_write_oled_mineru_review_packets_markdown(
        OledMineruReviewPacketReport(
            manifest_id="package",
            status="completed",
            paper_count=1,
            packet_count=len(packets),
            packets=packets,
        ),
        output_md,
    )
    manifest = OledMineruAcceptanceManifest(
        manifest_id="package-runner",
        bundles=[],
    )
    report = package_run_oled_mineru_review_packet_builder(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert isinstance(packets[0], OledMineruReviewPacket)
    assert isinstance(packets[0].material_roles[0], OledReviewPacketMaterialRole)
    assert isinstance(packets[0].properties[0], OledReviewPacketProperty)
    assert isinstance(packets[0].properties[0].evidence_refs[0], OledReviewPacketSourceRef)
    assert isinstance(report, OledMineruReviewPacketReport)
    assert OledReviewDecision.ACCEPT.value == "accept"
    assert output_jsonl.exists()
    assert output_md.exists()
