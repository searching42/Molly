from __future__ import annotations

import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.pdfgen.canvas import Canvas

import ai4s_agent.domains.oled_supplementary_parser_preflight as parser_preflight_domain
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPacketMappingProposal,
    OledLLMPaperMappingRequest,
    OledPaperContextElement,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidateType
from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSemanticMappingPacket
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY
from ai4s_agent.domains.oled_supplementary_parser_preflight import (
    OledSupplementaryParserPreflightManifest,
    build_oled_supplementary_parser_preflight_plan,
)
from ai4s_agent.domains.oled_supplementary_source_intake import (
    OledSupplementarySourceIntakeManifest,
)
from ai4s_agent.oled_llm_context_request import OledLLMContextRequestArtifact
from ai4s_agent.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryArtifact,
    prepare_oled_supplementary_evidence_recovery_artifact,
)
from ai4s_agent.oled_supplementary_parser_preflight import (
    OledSupplementaryParserPreflightArtifact,
    main,
    prepare_oled_supplementary_parser_preflight_artifact,
    prepare_oled_supplementary_parser_preflight_from_files,
)
from ai4s_agent.oled_supplementary_source_intake import (
    OledSupplementarySourceIntakeArtifact,
    prepare_oled_supplementary_source_intake_artifact,
)


_REVIEWED_AT = "2026-07-13T09:00:00Z"


def _write_pdf(path: Path, *, page_count: int = 2, text: str = "supplementary") -> Path:
    canvas = Canvas(str(path))
    for page_number in range(1, page_count + 1):
        canvas.drawString(72, 720, f"{text} page {page_number}")
        canvas.showPage()
    canvas.save()
    return path


def _recovery_artifact(*, include_manual_item: bool = False) -> OledSupplementaryEvidenceRecoveryArtifact:
    explicit_text = "The calculated values are reported in Supplementary Table S1."
    packets = [
        OledSemanticMappingPacket(
            packet_id="packet:explicit",
            source_candidate_hash="source-explicit",
            source_evidence_anchor="paper016:content_list:p3:packet:explicit:text",
            source_candidate_type=OledMineruCandidateType.TEXT,
            paper_id="paper016",
            raw_text=explicit_text,
            allowed_property_ids=["delta_e_st_ev"],
            allowed_layers=[OledCausalLayer.MOLECULE.value],
        )
    ]
    context = [
        OledPaperContextElement(
            element_id="ctx-explicit",
            page=3,
            element_type="paragraph",
            text=explicit_text,
            source_hash="sha256:main-paper",
        )
    ]
    if include_manual_item:
        manual_text = "Additional data are available in the Supplementary Information."
        packets.append(
            OledSemanticMappingPacket(
                packet_id="packet:manual",
                source_candidate_hash="source-manual",
                source_evidence_anchor="paper016:content_list:p4:packet:manual:text",
                source_candidate_type=OledMineruCandidateType.TEXT,
                paper_id="paper016",
                raw_text=manual_text,
                allowed_property_ids=["delta_e_st_ev"],
                allowed_layers=[OledCausalLayer.MOLECULE.value],
            )
        )
        context.append(
            OledPaperContextElement(
                element_id="ctx-manual",
                page=4,
                element_type="paragraph",
                text=manual_text,
                source_hash="sha256:main-paper",
            )
        )
    request = OledLLMPaperMappingRequest(
        paper_id="paper016",
        packets=packets,
        document_context=context,
        ontology=DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties(),
        deterministic_schema_candidates=[],
    )
    result = OledLLMContextMappingResult(
        paper_id="paper016",
        status="ready_for_human_review",
        request_digest=request.request_digest,
        packet_results=[
            OledLLMPacketMappingProposal(
                packet_id=packet.packet_id,
                action="needs_source_check",
                scope_classification="property_bearing",
                source_check_questions=["Inspect the unavailable supplementary material."],
                source_check_missing_evidence=["supplementary_information"],
                rationale_summary="The main document cites unavailable supplementary evidence.",
            )
            for packet in packets
        ],
    )
    request_artifact = OledLLMContextRequestArtifact(
        run_id="run-paper016",
        paper_id="paper016",
        generated_at=_REVIEWED_AT,
        request_digest=request.request_digest,
        request=request,
    )
    return prepare_oled_supplementary_evidence_recovery_artifact(
        request_artifact=request_artifact,
        mapping_result=result,
        run_id="run-paper016",
        generated_at=_REVIEWED_AT,
    )


def _source_intake_artifact(
    recovery_artifact: OledSupplementaryEvidenceRecoveryArtifact,
    pdf_path: Path,
) -> OledSupplementarySourceIntakeArtifact:
    plan = recovery_artifact.plan
    manifest = OledSupplementarySourceIntakeManifest(
        paper_id=plan.paper_id,
        source_request_digest=plan.source_request_digest,
        source_mapping_result_digest=plan.source_mapping_result_digest,
        source_context_digest=plan.source_context_digest,
        recovery_plan_digest=plan.plan_digest,
        intake_confirmed=True,
        sources=[
            {
                "source_id": "supp-source-001",
                "local_pdf_path": str(pdf_path),
                "provenance_category": "publisher-supplied",
                "access_policy": "reviewer-approved-local-copy",
                "provenance_note": "Approved local supplementary source.",
            }
        ],
        decisions=[
            {
                "recovery_item_id": item.item_id,
                "decision": "approved",
                "source_id": "supp-source-001",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
            }
            for item in plan.items
        ],
    )
    return prepare_oled_supplementary_source_intake_artifact(
        recovery_artifact=recovery_artifact,
        intake_manifest=manifest,
        generated_at=_REVIEWED_AT,
    )


def _explicit_item_id(source_intake_artifact: OledSupplementarySourceIntakeArtifact) -> str:
    return next(
        item.recovery_item_id
        for item in source_intake_artifact.intake_plan.items
        if item.target_locator == "S1"
    )


def _parse_manifest(
    source_intake_artifact: OledSupplementarySourceIntakeArtifact,
    pdf_path: Path,
    *,
    selected_recovery_item_ids: list[str] | None = None,
    parse_confirmed: bool = True,
) -> OledSupplementaryParserPreflightManifest:
    plan = source_intake_artifact.intake_plan
    selected = selected_recovery_item_ids or [_explicit_item_id(source_intake_artifact)]
    return OledSupplementaryParserPreflightManifest(
        paper_id=plan.paper_id,
        source_request_digest=plan.source_request_digest,
        source_mapping_result_digest=plan.source_mapping_result_digest,
        source_context_digest=plan.source_context_digest,
        recovery_plan_digest=plan.recovery_plan_digest,
        intake_plan_digest=plan.intake_plan_digest,
        parse_confirmed=parse_confirmed,
        reviewed_by="reviewer-02",
        reviewed_at=_REVIEWED_AT,
        selected_recovery_item_ids=selected,
        sources=[{"source_id": "supp-source-001", "local_pdf_path": str(pdf_path)}],
    )


def test_preflight_revalidates_pdf_and_preserves_explicit_locator_without_paths(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf", page_count=2)
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)

    artifact = prepare_oled_supplementary_parser_preflight_artifact(
        recovery_artifact=recovery_artifact,
        source_intake_artifact=source_intake_artifact,
        parse_manifest=parse_manifest,
        generated_at=_REVIEWED_AT,
    )

    assert artifact.preflight_plan.source_count == 1
    assert artifact.preflight_plan.item_count == 1
    assert artifact.preflight_plan.source_envelopes[0].page_count == 2
    assert artifact.preflight_plan.items[0].target_locator == "S1"
    assert artifact.preflight_plan.items[0].parse_scope == "full_source_then_locator_review"
    assert artifact.pdf_page_count_validated is True
    assert artifact.pdf_content_parsed is False
    assert artifact.mineru_called is False
    assert artifact.candidate_regenerated is False
    assert artifact.dataset_written is False
    rendered = artifact.model_dump_json()
    assert str(pdf_path) not in rendered
    assert "local_pdf_path" not in rendered


def test_preflight_plan_is_deterministic_and_uses_no_fabricated_page_range(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)

    first = build_oled_supplementary_parser_preflight_plan(
        recovery_artifact.plan,
        source_intake_artifact.intake_plan,
        parse_manifest,
    )
    second = build_oled_supplementary_parser_preflight_plan(
        recovery_artifact.plan,
        source_intake_artifact.intake_plan,
        parse_manifest,
    )

    assert first.preflight_plan_digest == second.preflight_plan_digest
    item_payload = first.items[0].model_dump(mode="json")
    assert "start_page" not in item_payload
    assert "end_page" not in item_payload
    assert item_payload["parse_scope"] == "full_source_then_locator_review"


def test_preflight_rejects_manual_target_even_when_its_source_was_approved(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    manual_item_id = next(
        item.recovery_item_id
        for item in source_intake_artifact.intake_plan.items
        if item.target_locator is None
    )
    parse_manifest = _parse_manifest(
        source_intake_artifact,
        pdf_path,
        selected_recovery_item_ids=[manual_item_id],
    )

    with pytest.raises(ValueError, match="only permits explicit recovery targets"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            source_intake_artifact.intake_plan,
            parse_manifest,
        )


def test_preflight_rejects_pdf_replaced_after_source_intake(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf", text="original")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)
    _write_pdf(pdf_path, text="replaced")

    with pytest.raises(ValueError, match="hash does not match|no longer matches"):
        prepare_oled_supplementary_parser_preflight_artifact(
            recovery_artifact=recovery_artifact,
            source_intake_artifact=source_intake_artifact,
            parse_manifest=parse_manifest,
            generated_at=_REVIEWED_AT,
        )


def test_preflight_rejects_path_swap_during_bound_page_count_and_hash_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf", page_count=2, text="approved")
    approved_bytes = pdf_path.read_bytes()
    replacement_bytes = _write_pdf(tmp_path / "replacement.pdf", page_count=5, text="replacement").read_bytes()
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)
    real_pdfplumber = parser_preflight_domain.importlib.import_module("pdfplumber")

    def swap_path_away_and_back(handle: object, *args: object, **kwargs: object) -> object:
        replacement_path = tmp_path / "swap-replacement.pdf"
        restored_path = tmp_path / "swap-restored.pdf"
        replacement_path.write_bytes(replacement_bytes)
        restored_path.write_bytes(approved_bytes)
        os.replace(replacement_path, pdf_path)
        try:
            return real_pdfplumber.open(handle, *args, **kwargs)
        finally:
            os.replace(restored_path, pdf_path)

    monkeypatch.setattr(
        parser_preflight_domain,
        "_load_pdfplumber",
        lambda: SimpleNamespace(open=swap_path_away_and_back),
    )

    with pytest.raises(ValueError, match="changed during parser preflight"):
        prepare_oled_supplementary_parser_preflight_artifact(
            recovery_artifact=recovery_artifact,
            source_intake_artifact=source_intake_artifact,
            parse_manifest=parse_manifest,
            generated_at=_REVIEWED_AT,
        )

    # A path replacement changes the open descriptor's metadata, so the
    # preflight fails closed instead of combining a page count from the
    # replacement with the approved file's hash.
    assert pdf_path.read_bytes() == approved_bytes


def test_preflight_rejects_unconfirmed_or_digest_mismatched_manifest(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    with pytest.raises(ValueError, match="parse_confirmed"):
        _parse_manifest(source_intake_artifact, pdf_path, parse_confirmed=False)
    payload = _parse_manifest(source_intake_artifact, pdf_path).model_dump(mode="json")
    payload["intake_plan_digest"] = "sha256:" + "0" * 64
    mismatched = OledSupplementaryParserPreflightManifest.model_validate(payload)

    with pytest.raises(ValueError, match="intake_plan_digest does not match"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            source_intake_artifact.intake_plan,
            mismatched,
        )


def test_preflight_revalidates_model_copies_against_recovery_chain(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    forged_item = next(item for item in source_intake_artifact.intake_plan.items if item.target_locator is None).model_copy(
        update={"target_locator": "S999"}
    )
    forged_plan = source_intake_artifact.intake_plan.model_copy(
        update={
            "items": [
                forged_item if item.recovery_item_id == forged_item.recovery_item_id else item
                for item in source_intake_artifact.intake_plan.items
            ]
        }
    )
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)

    with pytest.raises(ValueError, match="manual supplementary intake item must not assert target_locator"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            forged_plan,
            parse_manifest,
        )


def test_preflight_rejects_symlinked_rebinding_and_page_limit(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf", page_count=2)
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    with pytest.raises(ValueError, match="page count exceeds"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            source_intake_artifact.intake_plan,
            _parse_manifest(source_intake_artifact, pdf_path),
            max_pdf_pages=1,
        )
    link_path = tmp_path / "paper016_si_link.pdf"
    try:
        os.symlink(pdf_path, link_path)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")

    with pytest.raises(ValueError, match="must not be a symlink"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            source_intake_artifact.intake_plan,
            _parse_manifest(source_intake_artifact, link_path),
        )


def test_preflight_fails_closed_when_pdfplumber_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    monkeypatch.setattr(
        parser_preflight_domain.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("not installed")),
    )

    with pytest.raises(ValueError, match="requires pdfplumber"):
        build_oled_supplementary_parser_preflight_plan(
            recovery_artifact.plan,
            source_intake_artifact.intake_plan,
            _parse_manifest(source_intake_artifact, pdf_path),
        )


def _write_input_jsons(
    tmp_path: Path,
    recovery_artifact: OledSupplementaryEvidenceRecoveryArtifact,
    source_intake_artifact: OledSupplementarySourceIntakeArtifact,
    parse_manifest: OledSupplementaryParserPreflightManifest,
) -> tuple[Path, Path, Path]:
    recovery_path = tmp_path / "recovery.json"
    intake_path = tmp_path / "source-intake.json"
    manifest_path = tmp_path / "parse-manifest.json"
    write_json(recovery_path, recovery_artifact.model_dump(mode="json"))
    write_json(intake_path, source_intake_artifact.model_dump(mode="json"))
    write_json(manifest_path, parse_manifest.model_dump(mode="json"))
    return recovery_path, intake_path, manifest_path


def test_file_entrypoint_and_cli_write_only_redacted_preflight_artifact(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)
    recovery_path, intake_path, manifest_path = _write_input_jsons(
        tmp_path,
        recovery_artifact,
        source_intake_artifact,
        parse_manifest,
    )
    output_path = tmp_path / "parser-preflight.json"
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "--recovery-artifact",
            str(recovery_path),
            "--source-intake-artifact",
            str(intake_path),
            "--parse-manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    assert str(pdf_path) not in stdout.getvalue()
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert str(pdf_path) not in json.dumps(stored, sort_keys=True)
    assert stored["preflight_plan"]["source_envelopes"][0]["page_count"] == 2
    assert stored["preflight_plan"]["items"][0]["target_locator"] == "S1"


@pytest.mark.parametrize("collision_target", ["recovery", "intake", "manifest", "pdf"])
def test_file_entrypoint_rejects_output_collisions_without_mutating_inputs(
    tmp_path: Path,
    collision_target: str,
) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)
    recovery_path, intake_path, manifest_path = _write_input_jsons(
        tmp_path,
        recovery_artifact,
        source_intake_artifact,
        parse_manifest,
    )
    protected_paths = {
        "recovery": recovery_path,
        "intake": intake_path,
        "manifest": manifest_path,
        "pdf": pdf_path,
    }
    original_bytes = {path: path.read_bytes() for path in protected_paths.values()}
    original_pdf_sha256 = hashlib.sha256(original_bytes[pdf_path]).hexdigest()

    with pytest.raises(ValueError, match="must not overwrite an input artifact or local PDF"):
        prepare_oled_supplementary_parser_preflight_from_files(
            recovery_artifact_json=recovery_path,
            source_intake_artifact_json=intake_path,
            parse_manifest_json=manifest_path,
            output_json=protected_paths[collision_target],
            generated_at=_REVIEWED_AT,
        )

    for path, expected_bytes in original_bytes.items():
        assert path.read_bytes() == expected_bytes
    assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == original_pdf_sha256


def test_cli_collision_failure_does_not_leak_local_pdf_path(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    parse_manifest = _parse_manifest(source_intake_artifact, pdf_path)
    recovery_path, intake_path, manifest_path = _write_input_jsons(
        tmp_path,
        recovery_artifact,
        source_intake_artifact,
        parse_manifest,
    )
    original_pdf_bytes = pdf_path.read_bytes()
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "--recovery-artifact",
            str(recovery_path),
            "--source-intake-artifact",
            str(intake_path),
            "--parse-manifest",
            str(manifest_path),
            "--output",
            str(pdf_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "must not overwrite an input artifact or local PDF" in stderr.getvalue()
    assert str(pdf_path) not in stderr.getvalue()
    assert pdf_path.read_bytes() == original_pdf_bytes


def test_artifact_rejects_execution_side_effect_flags(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    source_intake_artifact = _source_intake_artifact(recovery_artifact, pdf_path)
    artifact = prepare_oled_supplementary_parser_preflight_artifact(
        recovery_artifact=recovery_artifact,
        source_intake_artifact=source_intake_artifact,
        parse_manifest=_parse_manifest(source_intake_artifact, pdf_path),
        generated_at=_REVIEWED_AT,
    )
    tampered = artifact.model_dump(mode="json")
    tampered["mineru_called"] = True

    with pytest.raises(ValueError, match="execution side effect"):
        OledSupplementaryParserPreflightArtifact.model_validate(tampered)
