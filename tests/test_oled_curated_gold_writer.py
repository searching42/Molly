from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledCuratedGoldManifest,
    OledCuratedGoldWriteResult,
    OledCuratedGoldWriteStatus,
    OledCuratedGoldWriterFinding,
    OledCuratedGoldWriterPolicy,
    OledCuratedGoldWriterReport,
    OledEvidenceSource,
    OledEvidenceType,
    OledGoldDatasetRecord,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMolecularLayer,
    OledPropertyObservation,
    OledReviewedGoldCandidate,
    OledReviewedGoldCandidateStatus,
    load_oled_reviewed_gold_candidates_jsonl as package_load_oled_reviewed_gold_candidates_jsonl,
    run_oled_curated_gold_writer as package_run_oled_curated_gold_writer,
    select_oled_curated_gold_records as package_select_oled_curated_gold_records,
    write_oled_curated_gold_manifest_json as package_write_oled_curated_gold_manifest_json,
    write_oled_curated_gold_records_jsonl as package_write_oled_curated_gold_records_jsonl,
)
from ai4s_agent.domains.oled_curated_gold_writer import (
    load_oled_reviewed_gold_candidates_jsonl,
    main,
    run_oled_curated_gold_writer,
    select_oled_curated_gold_records,
    write_oled_curated_gold_manifest_json,
    write_oled_curated_gold_records_jsonl,
)


def _gold_record(
    record_id: str = "gold-record-candidate:valid",
    *,
    property_label: str = "PLQY",
    evidence_refs: list[str] | None = None,
    reviewer: str | None = "reviewer-1",
    metadata: dict[str, object] | None = None,
) -> OledGoldDatasetRecord:
    refs = evidence_refs if evidence_refs is not None else ["paper:p2:b0:table"]
    observation = OledPropertyObservation(
        property_label=property_label,
        value=82,
        unit="%",
        evidence_sources=[
            OledEvidenceSource(
                source_id="hash-table:paper:p2:b0:table",
                source_type=OledEvidenceType.TABLE,
                layer=OledCausalLayer.INTERACTION,
                locator=refs[0] if refs else None,
                metadata={"candidate_only": True},
            )
        ],
        confidence=OledConfidenceAssessment(score=0.8),
    )
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(metadata={"context_only": True}),
            interaction=OledInteractionLayer(properties=[observation]),
        ),
        evidence_refs=refs,
        reviewer=reviewer,
        notes="Checked.",
        metadata={
            "candidate_only": True,
            "curated_dataset_written": False,
            "training_data_written": False,
            "final_gold_dataset": False,
            **(metadata or {}),
        },
    )


def _candidate(
    candidate_id: str = "gold-candidate:valid",
    *,
    status: OledReviewedGoldCandidateStatus = OledReviewedGoldCandidateStatus.CONVERTED,
    gold_record: OledGoldDatasetRecord | None = None,
    anchors: list[str] | None = None,
    validation_errors: list[str] | None = None,
    validation_warnings: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> OledReviewedGoldCandidate:
    refs = anchors if anchors is not None else ["paper:p2:b0:table"]
    return OledReviewedGoldCandidate(
        candidate_id=candidate_id,
        status=status,
        source_reviewed_candidate_id=candidate_id.replace("gold-candidate:", "reviewed-extraction:"),
        source_packet_id=candidate_id.replace("gold-candidate:", "review:"),
        source_compiled_record_id=candidate_id.replace("gold-candidate:", "compiled:"),
        paper_id="paper-gold",
        source_label="synthetic",
        gold_record=gold_record if gold_record is not None else _gold_record(record_id=f"gold-record-candidate:{candidate_id}"),
        source_candidate_hashes=["hash-table"],
        source_evidence_anchors=refs,
        validation_error_codes=validation_errors or [],
        validation_warning_codes=validation_warnings or [],
        metadata={
            "candidate_only": True,
            "curated_dataset_written": False,
            "training_data_written": False,
            "final_gold_dataset": False,
            **(metadata or {}),
        },
    )


def _write_candidates(path: Path, candidates: list[OledReviewedGoldCandidate]) -> Path:
    path.write_text(
        "\n".join(json.dumps(candidate.model_dump(mode="json"), sort_keys=True) for candidate in candidates) + "\n",
        encoding="utf-8",
    )
    return path


def test_confirmation_gate_requires_explicit_curated_write() -> None:
    with pytest.raises(ValueError, match="confirmation_required:curated_gold_write"):
        select_oled_curated_gold_records([_candidate()], confirm_curated_gold_write=False)


def test_select_valid_converted_candidate() -> None:
    report = select_oled_curated_gold_records([_candidate()], confirm_curated_gold_write=True)

    assert report.is_valid
    assert len(report.records) == 1
    assert report.manifest.output_record_count == 1
    assert report.manifest.status_counts == {"written": 1}
    assert report.manifest.write_results[0].reason_codes == ["selected_for_write"]
    assert report.manifest.metadata["training_data_written"] is False
    assert report.manifest.metadata["dataset_views_run"] is False


def test_reject_missing_gold_record() -> None:
    candidate = _candidate("gold-candidate:missing-record", gold_record=_gold_record()).model_copy(update={"gold_record": None})

    report = select_oled_curated_gold_records([candidate], confirm_curated_gold_write=True)

    assert not report.is_valid
    assert report.records == []
    assert report.manifest.write_results[0].status == OledCuratedGoldWriteStatus.REJECTED
    assert "missing_gold_record" in report.manifest.reason_code_counts


def test_reject_invalid_and_warning_status_by_default() -> None:
    candidates = [
        _candidate("gold-candidate:invalid", status=OledReviewedGoldCandidateStatus.INVALID),
        _candidate(
            "gold-candidate:warnings",
            status=OledReviewedGoldCandidateStatus.CONVERTED_WITH_WARNINGS,
            validation_warnings=["unit_not_normalized"],
        ),
    ]

    report = select_oled_curated_gold_records(candidates, confirm_curated_gold_write=True)

    assert report.records == []
    assert report.manifest.status_counts == {"rejected": 2}
    assert report.manifest.reason_code_counts["status_not_writable"] == 2


def test_policy_can_allow_converted_with_warnings() -> None:
    candidate = _candidate(
        "gold-candidate:warnings-allowed",
        status=OledReviewedGoldCandidateStatus.CONVERTED_WITH_WARNINGS,
        validation_warnings=["unit_not_normalized"],
    )
    policy = OledCuratedGoldWriterPolicy(
        allow_converted_with_warnings=True,
        allow_validation_warnings=True,
    )

    report = select_oled_curated_gold_records([candidate], policy=policy, confirm_curated_gold_write=True)

    assert report.is_valid
    assert len(report.records) == 1
    assert report.manifest.write_results[0].status == OledCuratedGoldWriteStatus.WRITTEN


def test_post_selection_gold_validation_rejects_invalid_record() -> None:
    candidate = _candidate(
        "gold-candidate:post-validation",
        gold_record=_gold_record("gold-record-candidate:post-validation", property_label="Not a known OLED property"),
    )

    report = select_oled_curated_gold_records([candidate], confirm_curated_gold_write=True)

    assert not report.is_valid
    assert report.records == []
    assert "post_selection_validation_error" in report.manifest.reason_code_counts
    assert "unknown_property_label" in report.error_codes


def test_records_jsonl_writer_is_deterministic_and_returns_sha256(tmp_path: Path) -> None:
    record = _gold_record(metadata={"source_path": str(tmp_path / "paper.json"), "raw_text": "mCBP | D1 | 82"})
    output_path = tmp_path / "curated_gold.jsonl"

    first_hash = write_oled_curated_gold_records_jsonl([record], output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_hash = write_oled_curated_gold_records_jsonl([record], output_path)
    second_text = output_path.read_text(encoding="utf-8")

    assert first_hash == second_hash == hashlib.sha256(first_text.encode("utf-8")).hexdigest()
    assert first_text == second_text
    assert first_text.splitlines()[0] == json.dumps(json.loads(first_text.splitlines()[0]), sort_keys=True, separators=(",", ":"))
    assert str(tmp_path) not in first_text
    assert "mCBP | D1 | 82" not in first_text


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    report = select_oled_curated_gold_records([_candidate()], confirm_curated_gold_write=True)
    manifest = report.manifest.model_copy(update={"output_sha256": "abc123"})
    path = tmp_path / "manifest.json"

    write_oled_curated_gold_manifest_json(manifest, path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["output_sha256"] == "abc123"
    assert payload["policy"]["require_confirmation"] is True
    assert payload["reason_code_counts"]["selected_for_write"] == 1


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    records_path = tmp_path / "curated_gold.jsonl"
    manifest_path = tmp_path / "manifest.json"
    policy = OledCuratedGoldWriterPolicy(require_confirmation=False)

    report = run_oled_curated_gold_writer(
        [_candidate()],
        output_jsonl_path=None,
        output_manifest_path=manifest_path,
        policy=policy,
        confirm_curated_gold_write=False,
    )

    assert report.is_valid
    assert manifest_path.exists()
    assert not records_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["output_record_count"] == 1


def test_cli_smoke_writes_curated_jsonl_and_manifest(tmp_path: Path) -> None:
    input_path = _write_candidates(tmp_path / "gold_candidates.jsonl", [_candidate("gold-candidate:cli")])
    output_path = tmp_path / "curated_gold_records.jsonl"
    manifest_path = tmp_path / "curated_gold_manifest.json"

    exit_code = main(
        [
            "--gold-candidates",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--output-manifest",
            str(manifest_path),
            "--confirm-curated-gold-write",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["output_sha256"]


def test_loader_handles_valid_empty_invalid_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "gold_candidates.jsonl"
    path.write_text("\n" + json.dumps(_candidate("gold-candidate:load").model_dump(mode="json"), sort_keys=True) + "\n\n", encoding="utf-8")

    loaded = load_oled_reviewed_gold_candidates_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].candidate_id == "gold-candidate:load"

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(_candidate().model_dump(mode="json"), sort_keys=True) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_gold_candidate_jsonl:line-2"):
        load_oled_reviewed_gold_candidates_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_gold_candidate_jsonl:"):
        load_oled_reviewed_gold_candidates_jsonl(tmp_path / "missing.jsonl")


def test_public_curated_gold_writer_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    candidate = _candidate("gold-candidate:package")
    input_path = _write_candidates(tmp_path / "package-gold.jsonl", [candidate])
    output_path = tmp_path / "package-curated.jsonl"
    manifest_path = tmp_path / "package-manifest.json"

    loaded = package_load_oled_reviewed_gold_candidates_jsonl(input_path)
    report = package_select_oled_curated_gold_records(loaded, confirm_curated_gold_write=True)
    sha = package_write_oled_curated_gold_records_jsonl(report.records, output_path)
    manifest = report.manifest.model_copy(update={"output_jsonl_path": output_path.name, "output_sha256": sha})
    package_write_oled_curated_gold_manifest_json(manifest, manifest_path)
    runner_report = package_run_oled_curated_gold_writer(
        loaded,
        output_manifest_path=tmp_path / "package-runner-manifest.json",
        confirm_curated_gold_write=True,
    )

    assert isinstance(OledCuratedGoldWriterPolicy(), OledCuratedGoldWriterPolicy)
    assert isinstance(report, OledCuratedGoldWriterReport)
    assert isinstance(report.manifest, OledCuratedGoldManifest)
    assert isinstance(report.manifest.write_results[0], OledCuratedGoldWriteResult)
    assert isinstance(OledCuratedGoldWriterFinding(code="x", message="y"), OledCuratedGoldWriterFinding)
    assert report.manifest.write_results[0].status == OledCuratedGoldWriteStatus.WRITTEN
    assert runner_report.is_valid
    assert output_path.exists()
    assert manifest_path.exists()
