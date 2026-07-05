from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledMineruAcceptanceManifest,
    OledMineruAcceptancePaperResult,
    OledMineruAcceptanceReport,
    OledMineruParsedBundle,
    load_oled_mineru_acceptance_manifest as package_load_oled_mineru_acceptance_manifest,
    redact_oled_mineru_acceptance_path as package_redact_oled_mineru_acceptance_path,
    run_oled_mineru_acceptance_harness as package_run_oled_mineru_acceptance_harness,
    write_oled_mineru_acceptance_report_json as package_write_oled_mineru_acceptance_report_json,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import (
    load_oled_mineru_acceptance_manifest,
    main,
    redact_oled_mineru_acceptance_path,
    run_oled_mineru_acceptance_harness,
    write_oled_mineru_acceptance_report_json,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _content_list_payload() -> list[dict[str, object]]:
    return [
        {
            "type": "table",
            "table_caption": "Table 1. OLED photophysical properties.",
            "table_body": "| Host | Emitter dopant | PLQY (%) |\n| --- | --- | --- |\n| mCBP | D1 | 82 |",
            "page_idx": 2,
        }
    ]


def _content_list_v2_payload() -> list[list[dict[str, object]]]:
    return [
        [
            {
                "type": "table",
                "content": {
                    "html": (
                        "<table><tr><th>Host</th><th>Emitter dopant</th><th>PLQY (%)</th></tr>"
                        "<tr><td>DPEPO</td><td>D2</td><td>76</td></tr></table>"
                    ),
                    "table_caption": ["Table S1. OLED photophysical properties."],
                },
            }
        ]
    ]


def _manifest_file(tmp_path: Path, bundles: list[dict[str, object]]) -> Path:
    return _write_json(
        tmp_path / "manifest.json",
        {
            "manifest_id": "oled-mineru-smoke-001",
            "bundles": bundles,
        },
    )


def test_manifest_loader_resolves_relative_paths_and_rejects_missing_or_forbidden_inputs(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-001_content_list.json", _content_list_payload())
    md_path = tmp_path / "paper-001.md"
    md_path.write_text("Before context. Table 1. OLED photophysical properties. After context.", encoding="utf-8")
    manifest_path = _manifest_file(
        tmp_path,
        [
            {
                "paper_id": "paper-001",
                "content_list_path": content_path.name,
                "md_path": md_path.name,
                "source_label": "synthetic",
            }
        ],
    )

    manifest = load_oled_mineru_acceptance_manifest(manifest_path)

    assert manifest.manifest_id == "oled-mineru-smoke-001"
    assert manifest.bundles[0].content_list_path == str(content_path.resolve())
    assert manifest.bundles[0].md_path == str(md_path.resolve())

    with pytest.raises(ValueError, match="missing_manifest_file:"):
        load_oled_mineru_acceptance_manifest(tmp_path / "missing_manifest.json")

    missing_bundle_manifest = _manifest_file(
        tmp_path,
        [{"paper_id": "missing-paper", "content_list_path": "missing_content_list.json"}],
    )
    with pytest.raises(ValueError, match="missing_bundle_file:"):
        load_oled_mineru_acceptance_manifest(missing_bundle_manifest)

    pdf_path = _write_json(tmp_path / "paper.pdf", {})
    pdf_manifest = _manifest_file(tmp_path, [{"paper_id": "paper-pdf", "content_list_path": pdf_path.name}])
    with pytest.raises(ValueError, match="forbidden_pdf_input:"):
        load_oled_mineru_acceptance_manifest(pdf_manifest)

    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not-read")
    image_manifest = _manifest_file(tmp_path, [{"paper_id": "paper-image", "content_list_path": image_path.name}])
    with pytest.raises(ValueError, match="forbidden_image_input:"):
        load_oled_mineru_acceptance_manifest(image_manifest)


def test_runner_requires_explicit_read_only_confirmation(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-001_content_list.json", _content_list_payload())
    manifest = OledMineruAcceptanceManifest(
        manifest_id="confirm-gate",
        bundles=[OledMineruParsedBundle(paper_id="paper-001", content_list_path=str(content_path))],
    )

    with pytest.raises(ValueError, match="confirmation_required:read_only_parsed_outputs"):
        run_oled_mineru_acceptance_harness(manifest)


def test_end_to_end_synthetic_content_list_run(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-001_content_list.json", _content_list_payload())
    md_path = tmp_path / "paper-001.md"
    md_path.write_text("Before context. Table 1. OLED photophysical properties. After context.", encoding="utf-8")
    manifest = OledMineruAcceptanceManifest(
        manifest_id="synthetic-content-list",
        bundles=[
            OledMineruParsedBundle(
                paper_id="paper-001",
                content_list_path=str(content_path),
                md_path=str(md_path),
                source_label="synthetic-paper",
            )
        ],
    )

    report = run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert report.is_valid is True
    assert report.completed_paper_count == 1
    assert report.total_mineru_candidate_count > 0
    assert report.total_semantic_candidate_count > 0
    assert report.total_compiled_record_count > 0
    assert report.paper_results[0].source_format_counts["content_list"] > 0
    assert report.paper_results[0].representative_evidence_anchors
    assert report.metadata["metadata_key_counts"]["paper_result:input_paths"] == 1
    assert report.metadata["gold_records_created"] is False
    assert report.metadata["curated_dataset_written"] is False


def test_content_list_v2_run(tmp_path: Path) -> None:
    v2_path = _write_json(tmp_path / "paper-002_content_list_v2.json", _content_list_v2_payload())
    manifest = OledMineruAcceptanceManifest(
        manifest_id="synthetic-v2",
        bundles=[OledMineruParsedBundle(paper_id="paper-002", content_list_v2_path=str(v2_path))],
    )

    report = run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert report.completed_paper_count == 1
    assert report.paper_results[0].source_format_counts["content_list_v2"] > 0
    assert report.total_compiled_record_count > 0


def test_bundle_with_content_list_and_v2_aggregates_source_formats(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-003_content_list.json", _content_list_payload())
    v2_path = _write_json(tmp_path / "paper-003_content_list_v2.json", _content_list_v2_payload())
    manifest = OledMineruAcceptanceManifest(
        manifest_id="synthetic-both",
        bundles=[
            OledMineruParsedBundle(
                paper_id="paper-003",
                content_list_path=str(content_path),
                content_list_v2_path=str(v2_path),
            )
        ],
    )

    report = run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert report.paper_results[0].source_format_counts["content_list"] > 0
    assert report.paper_results[0].source_format_counts["content_list_v2"] > 0
    assert report.total_mineru_candidate_count >= 2


def test_report_writer_redacts_paths_and_raw_text(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-004_content_list.json", _content_list_payload())
    manifest = OledMineruAcceptanceManifest(
        manifest_id="writer-redaction",
        bundles=[OledMineruParsedBundle(paper_id="paper-004", content_list_path=str(content_path))],
    )
    report = run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )
    output_path = tmp_path / "report.json"

    write_oled_mineru_acceptance_report_json(report, output_path)
    payload_text = output_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)

    assert payload["total_mineru_candidate_count"] > 0
    assert payload["paper_results"][0]["representative_evidence_anchors"]
    assert "finding_code_counts" in payload
    assert str(tmp_path) not in payload_text
    assert "mCBP | D1 | 82" not in payload_text
    assert payload_text == json.dumps(payload, sort_keys=True, indent=2) + "\n"


def test_cli_smoke_writes_report(tmp_path: Path) -> None:
    content_path = _write_json(tmp_path / "paper-005_content_list.json", _content_list_payload())
    manifest_path = _manifest_file(
        tmp_path,
        [{"paper_id": "paper-005", "content_list_path": content_path.name}],
    )
    output_path = tmp_path / "cli-report.json"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--output-report",
            str(output_path),
            "--confirm-read-only-parsed-outputs",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["manifest_id"] == "oled-mineru-smoke-001"


def test_runner_aggregates_bad_json_bundle_as_failed_paper(tmp_path: Path) -> None:
    valid_path = _write_json(tmp_path / "paper-valid_content_list.json", _content_list_payload())
    bad_path = tmp_path / "paper-bad_content_list.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    manifest = OledMineruAcceptanceManifest(
        manifest_id="bad-json-aggregation",
        bundles=[
            OledMineruParsedBundle(paper_id="paper-valid", content_list_path=str(valid_path)),
            OledMineruParsedBundle(paper_id="paper-bad", content_list_path=str(bad_path)),
        ],
    )

    report = run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )

    assert report.status == "failed"
    assert report.completed_paper_count == 1
    assert report.failed_paper_count == 1
    bad_result = next(result for result in report.paper_results if result.paper_id == "paper-bad")
    assert bad_result.status == "failed"
    assert "parsed_json_load_failed" in bad_result.reason_codes


def test_path_redaction_and_package_exports(tmp_path: Path) -> None:
    assert redact_oled_mineru_acceptance_path(tmp_path / "nested" / "paper.json") == "paper.json"
    assert package_redact_oled_mineru_acceptance_path(tmp_path / "nested" / "paper.json") == "paper.json"

    content_path = _write_json(tmp_path / "paper-006_content_list.json", _content_list_payload())
    manifest_path = _manifest_file(
        tmp_path,
        [{"paper_id": "paper-006", "content_list_path": content_path.name}],
    )
    manifest = package_load_oled_mineru_acceptance_manifest(manifest_path)
    report = package_run_oled_mineru_acceptance_harness(
        manifest,
        confirm_read_only_parsed_outputs=True,
    )
    output_path = tmp_path / "package-report.json"
    package_write_oled_mineru_acceptance_report_json(report, output_path)

    assert isinstance(manifest, OledMineruAcceptanceManifest)
    assert isinstance(manifest.bundles[0], OledMineruParsedBundle)
    assert isinstance(report, OledMineruAcceptanceReport)
    assert isinstance(report.paper_results[0], OledMineruAcceptancePaperResult)
    assert output_path.exists()
