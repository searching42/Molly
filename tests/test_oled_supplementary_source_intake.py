from __future__ import annotations

import hashlib
import json
import os
from io import StringIO
from pathlib import Path

import pytest

import ai4s_agent.domains.oled_supplementary_source_intake as supplementary_source_intake_domain
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
from ai4s_agent.domains.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryPlan,
    OledSupplementaryRecoveryStatus,
    OledSupplementaryTargetKind,
)
from ai4s_agent.domains.oled_supplementary_source_intake import (
    DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES,
    OledSupplementaryLocalSource,
    OledSupplementarySourceEnvelope,
    OledSupplementarySourceIntakeItem,
    OledSupplementarySourceIntakeManifest,
    OledSupplementarySourceParseEligibility,
    build_oled_supplementary_source_intake_plan,
    inspect_oled_supplementary_source_pdf,
)
from ai4s_agent.oled_llm_context_request import OledLLMContextRequestArtifact
from ai4s_agent.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryArtifact,
    prepare_oled_supplementary_evidence_recovery_artifact,
)
from ai4s_agent.oled_supplementary_source_intake import (
    OledSupplementarySourceIntakeArtifact,
    main,
    prepare_oled_supplementary_source_intake_artifact,
)


_REVIEWED_AT = "2026-07-13T08:00:00Z"


def _packet(*, packet_id: str, text: str, source_hash: str) -> OledSemanticMappingPacket:
    return OledSemanticMappingPacket(
        packet_id=packet_id,
        source_candidate_hash=source_hash,
        source_evidence_anchor=f"paper016:content_list:p3:{packet_id}:text",
        source_candidate_type=OledMineruCandidateType.TEXT,
        paper_id="paper016",
        raw_text=text,
        allowed_property_ids=["delta_e_st_ev"],
        allowed_layers=[OledCausalLayer.MOLECULE.value],
    )


def _recovery_artifact(*, include_manual_item: bool = False) -> OledSupplementaryEvidenceRecoveryArtifact:
    explicit_text = "The calculated values are reported in Supplementary Table S1."
    packets = [_packet(packet_id="packet:explicit", text=explicit_text, source_hash="source-explicit")]
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
        packets.append(_packet(packet_id="packet:manual", text=manual_text, source_hash="source-manual"))
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
                rationale_summary="The supplied main document cites unavailable supplementary evidence.",
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


def _manifest(
    plan: OledSupplementaryEvidenceRecoveryPlan,
    *,
    pdf_path: Path | None,
    decisions: list[dict[str, object]] | None = None,
    expected_pdf_sha256: str = "",
) -> OledSupplementarySourceIntakeManifest:
    if decisions is None:
        assert pdf_path is not None
        decisions = [
            {
                "recovery_item_id": item.item_id,
                "decision": "approved",
                "source_id": "supp-source-001",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
            }
            for item in plan.items
        ]
    source_ids = {str(item.get("source_id") or "") for item in decisions if item["decision"] == "approved"}
    sources: list[dict[str, object]] = []
    if source_ids:
        assert pdf_path is not None
        sources = [
            {
                "source_id": source_id,
                "local_pdf_path": str(pdf_path),
                "expected_pdf_sha256": expected_pdf_sha256,
                "provenance_category": "publisher-supplied",
                "access_policy": "reviewer-approved-local-copy",
                "provenance_note": "Manually supplied supplementary source.",
            }
            for source_id in sorted(source_ids)
        ]
    return OledSupplementarySourceIntakeManifest(
        paper_id=plan.paper_id,
        source_request_digest=plan.source_request_digest,
        source_mapping_result_digest=plan.source_mapping_result_digest,
        source_context_digest=plan.source_context_digest,
        recovery_plan_digest=plan.plan_digest,
        intake_confirmed=True,
        sources=sources,
        decisions=decisions,
    )


def _write_pdf(path: Path, *, body: bytes = b"supplementary") -> Path:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n" + body + b"\nendobj\n%%EOF\n")
    return path


def test_explicit_recovery_item_binds_a_verified_local_pdf_without_leaking_path(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)

    plan = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)

    assert plan.paper_id == recovery_artifact.paper_id
    assert plan.recovery_plan_digest == recovery_artifact.plan_digest
    assert plan.source_count == 1
    assert plan.item_count == 1
    assert plan.approved_item_count == 1
    assert plan.deferred_item_count == 0
    assert plan.rejected_item_count == 0
    assert plan.source_envelopes[0].pdf_sha256 == f"sha256:{hashlib.sha256(pdf_path.read_bytes()).hexdigest()}"
    assert plan.source_envelopes[0].byte_size == pdf_path.stat().st_size
    assert plan.source_envelopes[0].page_count_validated is False
    item = plan.items[0]
    assert item.recovery_status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    assert item.target_locator == "S1"
    assert item.parse_eligibility == OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_TARGETED_SOURCE_PARSE
    rendered = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    assert str(pdf_path) not in rendered
    assert "local_pdf_path" not in rendered
    assert plan.network_accessed is False
    assert plan.pdf_content_parsed is False
    assert plan.pdf_page_count_validated is False
    assert plan.mineru_called is False
    assert plan.candidate_regenerated is False
    assert plan.dataset_written is False


def test_one_local_pdf_can_bind_multiple_items_while_manual_item_stays_manual(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)

    plan = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)

    assert plan.source_count == 1
    assert plan.item_count == 2
    assert {item.source_id for item in plan.items} == {"supp-source-001"}
    explicit_item = next(
        item for item in plan.items if item.recovery_status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    )
    manual_item = next(
        item for item in plan.items if item.recovery_status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    )
    assert explicit_item.parse_eligibility == OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_TARGETED_SOURCE_PARSE
    assert manual_item.target_locator is None
    assert manual_item.parse_eligibility == OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_MANUAL_SOURCE_REVIEW
    assert manual_item.recovery_status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED


def test_all_recovery_items_require_one_explicit_decision(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(
        recovery_artifact.plan,
        pdf_path=pdf_path,
        decisions=[
            {
                "recovery_item_id": recovery_artifact.plan.items[0].item_id,
                "decision": "approved",
                "source_id": "supp-source-001",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
            }
        ],
    )

    with pytest.raises(ValueError, match="cover every recovery item"):
        build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)


def test_manifest_rejects_unknown_or_duplicate_recovery_item_decisions(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    base_decision = {
        "recovery_item_id": recovery_artifact.plan.items[0].item_id,
        "decision": "approved",
        "source_id": "supp-source-001",
        "reviewed_by": "reviewer-01",
        "reviewed_at": _REVIEWED_AT,
    }
    with pytest.raises(ValueError, match="duplicate supplementary recovery_item_id decision"):
        _manifest(recovery_artifact.plan, pdf_path=pdf_path, decisions=[base_decision, dict(base_decision)])
    unknown = dict(base_decision)
    unknown["recovery_item_id"] = "supplementary-recovery:unknown"
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path, decisions=[unknown])
    with pytest.raises(ValueError, match="cover every recovery item"):
        build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)


def test_manifest_digest_binding_fails_closed(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    payload = _manifest(recovery_artifact.plan, pdf_path=pdf_path).model_dump(mode="json")
    payload["recovery_plan_digest"] = "sha256:" + "0" * 64
    manifest = OledSupplementarySourceIntakeManifest.model_validate(payload)

    with pytest.raises(ValueError, match="recovery_plan_digest does not match"):
        build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)


def test_deferred_item_stays_unbound_and_never_becomes_parse_eligible() -> None:
    recovery_artifact = _recovery_artifact()
    item_id = recovery_artifact.plan.items[0].item_id
    manifest = _manifest(
        recovery_artifact.plan,
        pdf_path=None,
        decisions=[
            {
                "recovery_item_id": item_id,
                "decision": "deferred",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
                "review_note": "Awaiting an approved local source.",
            }
        ],
    )

    plan = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)

    assert plan.source_count == 0
    assert plan.approved_item_count == 0
    assert plan.deferred_item_count == 1
    assert plan.items[0].source_id is None
    assert plan.items[0].source_pdf_sha256 is None
    assert plan.items[0].parse_eligibility == OledSupplementarySourceParseEligibility.NOT_ELIGIBLE


def test_revalidation_blocks_model_copy_tampering_of_confirmation_and_manual_target(
    tmp_path: Path,
) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)
    manual_item = next(
        item
        for item in recovery_artifact.plan.items
        if item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    )
    forged_manual = manual_item.model_copy(
        update={
            "status": OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND,
            "target_kind": OledSupplementaryTargetKind.TABLE,
            "target_locator": "S999",
            "reference_label": "Supplementary Table S999",
            "recommended_next_action": "provide_approved_local_supplementary_source",
        }
    )
    forged_plan = recovery_artifact.plan.model_copy(
        update={
            "items": [
                forged_manual if item.item_id == manual_item.item_id else item
                for item in recovery_artifact.plan.items
            ]
        }
    )
    with pytest.raises(ValueError, match="plan digest does not match"):
        build_oled_supplementary_source_intake_plan(forged_plan, manifest)
    unconfirmed_manifest = manifest.model_copy(update={"intake_confirmed": False})
    with pytest.raises(ValueError, match="intake_confirmed=true"):
        build_oled_supplementary_source_intake_plan(
            recovery_artifact.plan,
            unconfirmed_manifest,
        )


def test_public_intake_models_reject_forged_manual_locators_and_page_count_claims(
    tmp_path: Path,
) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    plan = build_oled_supplementary_source_intake_plan(
        recovery_artifact.plan,
        _manifest(recovery_artifact.plan, pdf_path=pdf_path),
    )
    manual = next(
        item for item in plan.items if item.recovery_status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    )
    forged_manual_payload = manual.model_dump(mode="json")
    forged_manual_payload["target_locator"] = "S999"
    with pytest.raises(ValueError, match="must not assert target_locator"):
        OledSupplementarySourceIntakeItem.model_validate(forged_manual_payload)
    forged_explicit_payload = next(
        item for item in plan.items if item.recovery_status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    ).model_dump(mode="json")
    forged_explicit_payload["target_locator"] = None
    with pytest.raises(ValueError, match="requires target_locator"):
        OledSupplementarySourceIntakeItem.model_validate(forged_explicit_payload)
    forged_envelope_payload = plan.source_envelopes[0].model_dump(mode="json")
    forged_envelope_payload["page_count_validated"] = True
    with pytest.raises(ValueError, match="must not claim PDF page-count"):
        OledSupplementarySourceEnvelope.model_validate(forged_envelope_payload)


def test_audit_metadata_rejects_api_key_like_text() -> None:
    with pytest.raises(ValueError, match="credential-like"):
        OledSupplementaryLocalSource(
            source_id="supp-source-001",
            local_pdf_path="operator-source.pdf",
            provenance_category="publisher-supplied",
            access_policy="reviewer-approved-local-copy",
            provenance_note="api_key=not-allowed",
        )


@pytest.mark.parametrize(
    "kind",
    ["empty", "not_pdf", "missing_eof", "wrong_suffix", "directory"],
)
def test_local_pdf_envelope_validation_fails_closed(kind: str, tmp_path: Path) -> None:
    source_path = tmp_path / ("source.txt" if kind == "wrong_suffix" else "source.pdf")
    if kind == "empty":
        source_path.write_bytes(b"")
    elif kind == "not_pdf":
        source_path.write_bytes(b"not a PDF")
    elif kind == "missing_eof":
        source_path.write_bytes(b"%PDF-1.4\n1 0 obj\n")
    elif kind == "wrong_suffix":
        _write_pdf(source_path)
    else:
        source_path.mkdir()
    source = OledSupplementaryLocalSource(
        source_id="supp-source-001",
        local_pdf_path=str(source_path),
        provenance_category="publisher-supplied",
        access_policy="reviewer-approved-local-copy",
    )

    with pytest.raises(ValueError):
        inspect_oled_supplementary_source_pdf(source)


def test_local_pdf_hash_size_and_symlink_checks_fail_closed(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "source.pdf")
    source = OledSupplementaryLocalSource(
        source_id="supp-source-001",
        local_pdf_path=str(pdf_path),
        expected_pdf_sha256="sha256:" + "0" * 64,
        provenance_category="publisher-supplied",
        access_policy="reviewer-approved-local-copy",
    )
    with pytest.raises(ValueError, match="hash does not match"):
        inspect_oled_supplementary_source_pdf(source)
    with pytest.raises(ValueError, match="exceeds size limit"):
        inspect_oled_supplementary_source_pdf(
            source.model_copy(update={"expected_pdf_sha256": ""}),
            max_pdf_bytes=1,
        )
    link_path = tmp_path / "source-link.pdf"
    try:
        os.symlink(pdf_path, link_path)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    symlink_source = source.model_copy(
        update={"local_pdf_path": str(link_path), "expected_pdf_sha256": ""}
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        inspect_oled_supplementary_source_pdf(symlink_source)


def test_local_pdf_inspection_requires_no_follow_support(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "source.pdf")
    source = OledSupplementaryLocalSource(
        source_id="supp-source-001",
        local_pdf_path=str(pdf_path),
        provenance_category="publisher-supplied",
        access_policy="reviewer-approved-local-copy",
    )
    monkeypatch.delattr(supplementary_source_intake_domain.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="requires O_NOFOLLOW support"):
        inspect_oled_supplementary_source_pdf(source)


def test_two_distinct_local_sources_require_explicit_item_bindings(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact(include_manual_item=True)
    explicit_item = next(
        item
        for item in recovery_artifact.plan.items
        if item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    )
    manual_item = next(
        item
        for item in recovery_artifact.plan.items
        if item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    )
    source_a = _write_pdf(tmp_path / "source-a.pdf", body=b"source-a")
    source_b = _write_pdf(tmp_path / "source-b.pdf", body=b"source-b")
    manifest = OledSupplementarySourceIntakeManifest(
        paper_id=recovery_artifact.plan.paper_id,
        source_request_digest=recovery_artifact.plan.source_request_digest,
        source_mapping_result_digest=recovery_artifact.plan.source_mapping_result_digest,
        source_context_digest=recovery_artifact.plan.source_context_digest,
        recovery_plan_digest=recovery_artifact.plan.plan_digest,
        intake_confirmed=True,
        sources=[
            {
                "source_id": "supp-source-a",
                "local_pdf_path": str(source_a),
                "provenance_category": "publisher-supplied",
                "access_policy": "reviewer-approved-local-copy",
            },
            {
                "source_id": "supp-source-b",
                "local_pdf_path": str(source_b),
                "provenance_category": "repository-supplied",
                "access_policy": "reviewer-approved-local-copy",
            },
        ],
        decisions=[
            {
                "recovery_item_id": explicit_item.item_id,
                "decision": "approved",
                "source_id": "supp-source-a",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
            },
            {
                "recovery_item_id": manual_item.item_id,
                "decision": "approved",
                "source_id": "supp-source-b",
                "reviewed_by": "reviewer-01",
                "reviewed_at": _REVIEWED_AT,
            },
        ],
    )

    plan = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)

    assert [source.source_id for source in plan.source_envelopes] == ["supp-source-a", "supp-source-b"]
    assert {item.source_id for item in plan.items} == {"supp-source-a", "supp-source-b"}


def test_plan_digest_is_reproducible_and_manifest_paths_remain_private(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)

    first = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)
    second = build_oled_supplementary_source_intake_plan(recovery_artifact.plan, manifest)

    assert first.intake_plan_digest == second.intake_plan_digest
    assert first.source_envelopes[0].byte_size < DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES
    assert str(pdf_path) not in first.model_dump_json()


def test_artifact_and_cli_write_only_redacted_local_source_metadata(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)

    artifact = prepare_oled_supplementary_source_intake_artifact(
        recovery_artifact=recovery_artifact,
        intake_manifest=manifest,
        generated_at=_REVIEWED_AT,
    )
    rendered = artifact.model_dump_json()
    assert str(pdf_path) not in rendered
    assert "local_pdf_path" not in rendered
    assert artifact.run_id == recovery_artifact.run_id
    assert artifact.intake_plan.pdf_content_parsed is False

    recovery_path = tmp_path / "recovery.json"
    manifest_path = tmp_path / "intake-manifest.json"
    output_path = tmp_path / "intake-artifact.json"
    write_json(recovery_path, recovery_artifact.model_dump(mode="json"))
    write_json(manifest_path, manifest.model_dump(mode="json"))
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "--recovery-artifact",
            str(recovery_path),
            "--intake-manifest",
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
    assert str(tmp_path) not in stdout.getvalue()
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert str(pdf_path) not in json.dumps(stored, sort_keys=True)
    assert stored["intake_plan"]["items"][0]["parse_eligibility"] == "eligible_for_targeted_source_parse"


def test_artifact_side_effect_flags_and_cli_failures_do_not_leak_source_paths(tmp_path: Path) -> None:
    recovery_artifact = _recovery_artifact()
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    manifest = _manifest(recovery_artifact.plan, pdf_path=pdf_path)
    artifact = prepare_oled_supplementary_source_intake_artifact(
        recovery_artifact=recovery_artifact,
        intake_manifest=manifest,
        generated_at=_REVIEWED_AT,
    )
    tampered = artifact.model_dump(mode="json")
    tampered["mineru_called"] = True
    with pytest.raises(ValueError, match="execution side effect"):
        OledSupplementarySourceIntakeArtifact.model_validate(tampered)

    recovery_path = tmp_path / "recovery.json"
    manifest_path = tmp_path / "intake-manifest.json"
    output_path = tmp_path / "intake-artifact.json"
    bad_source_path = tmp_path / "not-present.pdf"
    bad_manifest = _manifest(recovery_artifact.plan, pdf_path=bad_source_path)
    write_json(recovery_path, recovery_artifact.model_dump(mode="json"))
    write_json(manifest_path, bad_manifest.model_dump(mode="json"))
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "--recovery-artifact",
            str(recovery_path),
            "--intake-manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert str(bad_source_path) not in stderr.getvalue()
