from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorReviewArtifact,
    OledSupplementaryLocatorReviewStatus,
)
from ai4s_agent.domains.oled_supplementary_mineru_execution import (
    OledSupplementaryMineruExecutionManifest,
    OledSupplementaryMineruOutputHash,
    OledSupplementaryMineruSourceExecutionResult,
    build_oled_supplementary_mineru_execution_artifact,
)
from ai4s_agent.oled_supplementary_locator_review import (
    generate_oled_supplementary_locator_review_from_files,
    main,
)
from ai4s_agent.schemas import ParsedDocument


_GENERATED_AT = "2026-07-13T12:00:00Z"
_PARSER_BACKEND = "mineru_api:hybrid-engine"


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _default_tables() -> list[dict[str, Any]]:
    return [
        {
            "table_id": "table_p2_0006",
            "page": 2,
            "caption": "Table of Contents",
            "headers": ["Section", "Page"],
            "rows": [{"Section": "Supplementary Table S1", "Page": "38"}],
            "footnotes": [],
        },
        {
            "table_id": "table_p38_0178",
            "page": 2,
            "caption": "**Supplementary Table S1.** Photophysical properties",
            "headers": ["Emitter", "ΔE_ST (eV)", "PLQY (%)"],
            "rows": [
                {"Emitter": "Compound 1", "ΔE_ST (eV)": "0.030", "PLQY (%)": "87.0"}
            ],
            "footnotes": ["Measured in degassed toluene."],
            "source_bbox": {"x0": 10.0, "y0": 20.0, "x1": 500.0, "y1": 700.0},
        },
    ]


def _write_fixture(
    tmp_path: Path,
    *,
    tables: list[dict[str, Any]] | None = None,
    targets: list[dict[str, str]] | None = None,
    parsed_backend: str = _PARSER_BACKEND,
    parsed_page_count: int = 2,
) -> tuple[Path, Path, Path, Path, Path]:
    parsed = ParsedDocument(
        paper_id="paper016",
        source_path="/operator/private/paper016-si.pdf",
        parser_backend=parsed_backend,
        pages=[{"page": index} for index in range(1, parsed_page_count + 1)],
        tables=tables if tables is not None else _default_tables(),
    )
    parsed_path = tmp_path / "parsed-document.json"
    write_json(parsed_path, parsed.model_dump(mode="json"))
    parsed_sha256 = _sha256_file(parsed_path)
    output_hashes = [
        OledSupplementaryMineruOutputHash(
            output_kind="parsed_document_json",
            sha256=parsed_sha256,
            byte_size=parsed_path.stat().st_size,
        ),
        OledSupplementaryMineruOutputHash(
            output_kind="parsed_document_markdown",
            sha256="sha256:" + "2" * 64,
            byte_size=12,
        ),
        OledSupplementaryMineruOutputHash(
            output_kind="parser_audit_json",
            sha256="sha256:" + "3" * 64,
            byte_size=13,
        ),
    ]
    output_hashes.sort(key=lambda item: item.output_kind.value)
    target_payloads = targets or [
        {
            "recovery_item_id": "supplementary-recovery:item-001",
            "target_kind": "table",
            "target_locator": "S1",
        }
    ]
    source_result = OledSupplementaryMineruSourceExecutionResult(
        source_id="supp-source-001",
        source_pdf_sha256="sha256:" + "1" * 64,
        byte_size=1234,
        page_count=2,
        targets=target_payloads,
        status="success",
        mineru_called=True,
        provider="mineru_api",
        parser_backend=_PARSER_BACKEND,
        mineru_version="mineru-3.4.0",
        protocol_version="2",
        output_hashes=output_hashes,
    )
    execution_manifest = OledSupplementaryMineruExecutionManifest(
        run_id="supp-mineru-run-001",
        paper_id="paper016",
        preflight_plan_digest="sha256:" + "4" * 64,
        execution_confirmed=True,
        reviewed_by="reviewer-01",
        reviewed_at=_GENERATED_AT,
        endpoint_profile_name="node45-loopback",
        endpoint_preflight_sha256="sha256:" + "5" * 64,
        sources=[
            {
                "source_id": "supp-source-001",
                "local_pdf_path": "/operator/private/paper016-si.pdf",
            }
        ],
    )
    execution = build_oled_supplementary_mineru_execution_artifact(
        manifest=execution_manifest,
        generated_at=_GENERATED_AT,
        redacted_api_origin="http://127.0.0.1:18000",
        backend="hybrid-engine",
        effort="medium",
        parse_method="auto",
        source_results=[source_result],
    )
    execution_path = tmp_path / "execution.json"
    write_json(execution_path, execution.model_dump(mode="json"))
    manifest_path = tmp_path / "locator-manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "oled_supplementary_locator_manifest.v1",
            "run_id": execution.run_id,
            "paper_id": execution.paper_id,
            "execution_artifact_sha256": _sha256_file(execution_path),
            "execution_artifact_digest": execution.execution_artifact_digest,
            "sources": [
                {
                    "source_id": "supp-source-001",
                    "parsed_document_json": str(parsed_path),
                }
            ],
        },
    )
    return (
        execution_path,
        manifest_path,
        parsed_path,
        tmp_path / "locator-review.json",
        tmp_path / "locator-review.md",
    )


def _generate(paths: tuple[Path, Path, Path, Path, Path]) -> OledSupplementaryLocatorReviewArtifact:
    execution_path, manifest_path, _, output_json, output_markdown = paths
    return generate_oled_supplementary_locator_review_from_files(
        execution_artifact_json=execution_path,
        locator_manifest_json=manifest_path,
        output_json=output_json,
        output_markdown=output_markdown,
        generated_at=_GENERATED_AT,
    )


def test_exact_caption_locator_ignores_toc_decoy_and_preserves_reported_text(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    artifact = _generate(paths)

    assert artifact.status == OledSupplementaryLocatorReviewStatus.READY_FOR_HUMAN_REVIEW
    assert artifact.locator_resolved is True
    item = artifact.review_items[0]
    assert item.match_status.value == "exact_match"
    assert item.candidate_table_ids == ["table_p38_0178"]
    assert item.matched_table is not None
    assert item.matched_table.rows[0]["ΔE_ST (eV)"] == "0.030"
    assert item.matched_table.rows[0]["PLQY (%)"] == "87.0"
    assert item.matched_table.footnotes == ["Measured in degassed toluene."]
    packet_text = paths[3].read_text(encoding="utf-8") + paths[4].read_text(encoding="utf-8")
    assert "/operator/private" not in packet_text
    assert str(paths[2]) not in packet_text
    assert "candidate_regenerated" in packet_text
    assert "0.030" in packet_text


def test_s1_locator_does_not_match_s10(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        tables=[
            {
                "table_id": "table-s10",
                "page": 2,
                "caption": "Supplementary Table S10. Different evidence",
                "headers": ["Value"],
                "rows": [{"Value": "10"}],
            }
        ],
    )

    artifact = _generate(paths)

    assert artifact.status == OledSupplementaryLocatorReviewStatus.MANUAL_LOCATOR_REVIEW_REQUIRED
    assert artifact.review_items[0].match_status.value == "not_found"
    assert artifact.review_items[0].matched_table is None


@pytest.mark.parametrize(
    "caption",
    [
        "Supplementary Table S1-S3",
        "Supplementary Table S1/S2",
        "Supplementary Table S1, S2 and S3",
        "Supplementary Table S1, S2, and S3",
        "Table S1 and S2",
        "Table S1 to S3",
        "Supplementary Table S1–S3",
        "Supplementary Table S1 & S2",
        "Supplementary Table S1 or S2",
    ],
)
def test_range_or_list_caption_never_resolves_as_single_table(
    tmp_path: Path,
    caption: str,
) -> None:
    paths = _write_fixture(
        tmp_path,
        tables=[
            {
                "table_id": "table-series",
                "page": 2,
                "caption": caption,
                "headers": ["Value"],
                "rows": [{"Value": "0.030"}],
            }
        ],
    )

    artifact = _generate(paths)

    item = artifact.review_items[0]
    assert item.match_status.value == "not_found"
    assert item.matched_table is None
    assert item.candidate_table_ids == []
    assert artifact.locator_resolved is False
    assert artifact.status == OledSupplementaryLocatorReviewStatus.MANUAL_LOCATOR_REVIEW_REQUIRED


@pytest.mark.parametrize(
    "caption",
    [
        "Supplementary Table S1. Photophysical properties",
        "Table S1 (continued)",
    ],
)
def test_single_table_caption_remains_an_exact_match(tmp_path: Path, caption: str) -> None:
    paths = _write_fixture(
        tmp_path,
        tables=[
            {
                "table_id": "table-single",
                "page": 2,
                "caption": caption,
                "headers": ["Value"],
                "rows": [{"Value": "0.030"}],
            }
        ],
    )

    artifact = _generate(paths)

    item = artifact.review_items[0]
    assert item.match_status.value == "exact_match"
    assert item.matched_table is not None
    assert item.matched_table.table_id == "table-single"
    assert artifact.locator_resolved is True


def test_duplicate_exact_captions_fail_closed_as_ambiguous(tmp_path: Path) -> None:
    tables = _default_tables()
    tables.append(
        {
            "table_id": "table_p39_0179",
            "page": 2,
            "caption": "Table S1 (continued)",
            "headers": ["Emitter"],
            "rows": [{"Emitter": "Compound 2"}],
        }
    )
    paths = _write_fixture(tmp_path, tables=tables)

    artifact = _generate(paths)

    item = artifact.review_items[0]
    assert item.match_status.value == "ambiguous"
    assert item.candidate_table_ids == ["table_p38_0178", "table_p39_0179"]
    assert item.matched_table is None
    assert artifact.locator_resolved is False


def test_unsupported_figure_target_requires_manual_review(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        targets=[
            {
                "recovery_item_id": "supplementary-recovery:item-figure-001",
                "target_kind": "figure",
                "target_locator": "S2",
            }
        ],
    )

    artifact = _generate(paths)

    assert artifact.review_items[0].match_status.value == "unsupported_target_kind"
    assert artifact.review_items[0].candidate_table_ids == []
    assert artifact.status == OledSupplementaryLocatorReviewStatus.MANUAL_LOCATOR_REVIEW_REQUIRED


def test_changed_parsed_document_is_rejected_before_output(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths[2].write_text(paths[2].read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="bytes do not match"):
        _generate(paths)

    assert not paths[3].exists()
    assert not paths[4].exists()


def test_symlinked_parsed_document_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    link_path = tmp_path / "parsed-link.json"
    link_path.symlink_to(paths[2])
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    manifest["sources"][0]["parsed_document_json"] = str(link_path)
    write_json(paths[1], manifest)

    with pytest.raises(ValueError, match="input is unavailable"):
        _generate(paths)

    assert not paths[3].exists()


def test_manifest_must_bind_exact_execution_digest(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    manifest["execution_artifact_digest"] = "sha256:" + "9" * 64
    write_json(paths[1], manifest)

    with pytest.raises(ValueError, match="canonical content"):
        _generate(paths)


def test_manifest_must_exactly_cover_execution_sources(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    manifest["sources"].append(
        {
            "source_id": "unapproved-source-002",
            "parsed_document_json": str(paths[2]),
        }
    )
    write_json(paths[1], manifest)

    with pytest.raises(ValueError, match="exactly cover"):
        _generate(paths)


def test_parsed_document_backend_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, parsed_backend="mineru_api:pipeline")

    with pytest.raises(ValueError, match="backend"):
        _generate(paths)


def test_parsed_document_page_count_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, parsed_page_count=1)

    with pytest.raises(ValueError, match="page count"):
        _generate(paths)


def test_output_collision_cannot_replace_parsed_document(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    original = paths[2].read_bytes()

    with pytest.raises(ValueError, match="overwrite an input"):
        generate_oled_supplementary_locator_review_from_files(
            execution_artifact_json=paths[0],
            locator_manifest_json=paths[1],
            output_json=paths[2],
            output_markdown=paths[4],
            generated_at=_GENERATED_AT,
        )

    assert paths[2].read_bytes() == original
    assert _sha256_file(paths[2]) == f"sha256:{hashlib.sha256(original).hexdigest()}"


def test_review_artifact_rejects_downstream_flag_tampering(tmp_path: Path) -> None:
    artifact = _generate(_write_fixture(tmp_path))
    payload = artifact.model_dump(mode="json")
    payload["dataset_written"] = True

    with pytest.raises(ValueError, match="downstream boundary"):
        OledSupplementaryLocatorReviewArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["source_count"] = 2
    with pytest.raises(ValueError, match="source_count mismatch"):
        OledSupplementaryLocatorReviewArtifact.model_validate(payload)


def test_cli_stdout_is_redacted_and_contains_no_local_paths(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    stdout = StringIO()

    result = main(
        [
            "--execution-artifact",
            str(paths[0]),
            "--locator-manifest",
            str(paths[1]),
            "--output-json",
            str(paths[3]),
            "--output-markdown",
            str(paths[4]),
        ],
        stdout=stdout,
    )

    assert result == 0
    output = stdout.getvalue()
    assert "ready_for_human_review" in output
    assert str(tmp_path) not in output
    assert "/operator/private" not in output
