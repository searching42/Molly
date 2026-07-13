from __future__ import annotations

import json
from io import StringIO

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPacketMappingProposal,
    OledLLMPaperMappingRequest,
    OledPaperContextElement,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidateType
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSemanticMappingPacket,
)
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY
from ai4s_agent.domains.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryPlan,
    OledSupplementaryRecoveryStatus,
    OledSupplementaryTargetKind,
    build_oled_supplementary_evidence_recovery_plan,
)
from ai4s_agent.oled_llm_context_request import OledLLMContextRequestArtifact
from ai4s_agent.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryArtifact,
    main,
    prepare_oled_supplementary_evidence_recovery_from_files,
)


def _packet(
    *,
    packet_id: str = "packet:d01:source",
    source_hash: str = "source-d01-hash",
    source_anchor: str = "paper016:content_list:p3:b34:text",
    text: str,
) -> OledSemanticMappingPacket:
    return OledSemanticMappingPacket(
        packet_id=packet_id,
        source_candidate_hash=source_hash,
        source_evidence_anchor=source_anchor,
        source_candidate_type=OledMineruCandidateType.TEXT,
        paper_id="paper016",
        raw_text=text,
        allowed_property_ids=["delta_e_st_ev"],
        allowed_layers=[OledCausalLayer.MOLECULE.value],
    )


def _context(
    *,
    text: str,
    element_id: str = "el_p3_0039",
    source_hash: str = "sha256:main-paper",
    page: int = 3,
) -> OledPaperContextElement:
    return OledPaperContextElement(
        element_id=element_id,
        page=page,
        element_type="paragraph",
        text=text,
        source_hash=source_hash,
    )


def _candidate(
    *,
    source_hash: str = "source-d01-hash",
    candidate_id: str = "schema:deterministic:d01",
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper016",
        source_candidate_hash=source_hash,
        source_evidence_anchor="paper016:content_list:p3:b34:text",
        target_layer=OledCausalLayer.MOLECULE,
        property_id="delta_e_st_ev",
        value=0.5,
        unit="eV",
        reported_value_text="0.50",
        reported_decimal_places=2,
    )


def _request(
    packet: OledSemanticMappingPacket,
    *,
    context: list[OledPaperContextElement],
    candidates: list[OledSchemaCandidate] | None = None,
) -> OledLLMPaperMappingRequest:
    return OledLLMPaperMappingRequest(
        paper_id="paper016",
        packets=[packet],
        document_context=context,
        ontology=DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties(),
        deterministic_schema_candidates=candidates or [],
    )


def _source_check_result(
    request: OledLLMPaperMappingRequest,
    *,
    missing_evidence: list[str] | None = None,
    packet_results: list[OledLLMPacketMappingProposal] | None = None,
) -> OledLLMContextMappingResult:
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status="ready_for_human_review",
        request_digest=request.request_digest,
        packet_results=packet_results
        or [
            OledLLMPacketMappingProposal(
                packet_id=request.packets[0].packet_id,
                action="needs_source_check",
                scope_classification="property_bearing",
                source_check_questions=[
                    "Inspect the unavailable supplementary material for the cited assignment."
                ],
                source_check_missing_evidence=missing_evidence
                or ["supplementary_information"],
                rationale_summary="The supplied main document leaves the cited source unavailable.",
            )
        ],
    )


def test_d01_style_reference_builds_content_bound_explicit_table_recovery_plan() -> None:
    text = (
        "The calculated Delta E_ST values of the synthesized host materials were "
        "0.45-0.52 eV (Supplementary Table S1)."
    )
    packet = _packet(text=text)
    request = _request(
        packet,
        context=[_context(text=text)],
        candidates=[
            _candidate(),
            _candidate(
                source_hash="unrelated-source-hash",
                candidate_id="schema:deterministic:unrelated",
            ),
        ],
    )

    plan = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    )

    assert plan.item_count == 1
    item = plan.items[0]
    assert item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    assert item.target_kind == OledSupplementaryTargetKind.TABLE
    assert item.target_locator == "S1"
    assert item.reference_label == "Supplementary Table S1"
    assert item.source_packet_id == packet.packet_id
    assert item.source_candidate_hash == packet.source_candidate_hash
    assert item.affected_deterministic_candidate_ids == ["schema:deterministic:d01"]
    assert item.reference_anchors[0].element_id == "el_p3_0039"
    assert item.reference_anchors[0].source_hash == "sha256:main-paper"
    assert item.reference_anchors[0].page == 3
    assert item.reference_anchors[0].matched_text == "Supplementary Table S1"
    assert item.reference_anchors[0].match_start == text.index("Supplementary Table S1")
    assert item.reference_anchors[0].match_end == (
        item.reference_anchors[0].match_start + len("Supplementary Table S1")
    )
    assert plan.executable is False
    assert plan.offline_only is True
    assert plan.network_accessed is False
    assert plan.llm_called is False
    assert plan.mineru_called is False
    assert plan.supplementary_downloaded is False
    assert plan.automatic_candidate_merge is False
    assert plan.reviewed_evidence_staging is False
    assert plan.device_only_admitted is False
    assert plan.gold_records_created is False
    assert plan.dataset_written is False
    assert plan.plan_digest == build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).plan_digest


def test_figure_reference_preserves_figure_kind_and_locator() -> None:
    text = "The optimized geometry is shown in Supplementary Fig. S27."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    assert item.target_kind == OledSupplementaryTargetKind.FIGURE
    assert item.target_locator == "S27"
    assert item.reference_label == "Supplementary Figure S27"


def test_supporting_information_table_reference_is_explicit_without_normalizing_locator() -> None:
    text = "The raw data are available in Supporting Information Table 1."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    assert item.target_kind == OledSupplementaryTargetKind.TABLE
    assert item.target_locator == "1"
    assert item.reference_anchors[0].matched_text == "Supporting Information Table 1"


def test_generic_supplementary_information_requires_manual_locator_without_guessing_s1() -> None:
    text = "Experimental details are provided in the Supplementary Information."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_kind == OledSupplementaryTargetKind.INFORMATION
    assert item.target_locator is None
    assert item.reference_label == "Supplementary Information"
    assert "S1" not in json.dumps(item.model_dump(mode="json"))


def test_unqualified_table_s_number_stays_manual_even_when_packet_and_context_match() -> None:
    text = "The complete values are listed in Table S1."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_kind == OledSupplementaryTargetKind.TABLE
    assert item.target_locator is None
    assert item.reference_label == "Table S1"


@pytest.mark.parametrize(
    ("packet_text", "context_text", "target_kind"),
    [
        (
            "The complete values are listed in Table S1.",
            "The complete values are listed in Supplementary Table S1.",
            OledSupplementaryTargetKind.TABLE,
        ),
        (
            "The geometry is shown in Fig. S27.",
            "The geometry is shown in Supplementary Fig. S27.",
            OledSupplementaryTargetKind.FIGURE,
        ),
    ],
)
def test_bare_packet_reference_is_not_promoted_by_explicit_bound_context(
    packet_text: str,
    context_text: str,
    target_kind: OledSupplementaryTargetKind,
) -> None:
    packet = _packet(text=packet_text, source_anchor="shared-source-anchor")
    request = _request(
        packet,
        context=[_context(text=context_text, element_id="shared-source-anchor")],
    )

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_kind == target_kind
    assert item.target_locator is None


def test_packet_fields_cannot_be_joined_to_fabricate_an_explicit_reference() -> None:
    packet = _packet(
        text="The complete values are listed in Supplementary",
        source_anchor="shared-source-anchor",
    ).model_copy(update={"caption": "Table S1."})
    context_text = "The complete values are listed in Supplementary Table S1."
    request = _request(
        packet,
        context=[_context(text=context_text, element_id="shared-source-anchor")],
    )

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_locator is None


def test_supplementary_reference_range_never_collapses_to_the_first_locator() -> None:
    text = "The calculations are summarized in Supplementary Table S1-S3."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_locator is None
    assert item.reference_anchors == []


def test_textual_substring_without_shared_provenance_cannot_bind_a_recovery_reference() -> None:
    packet_text = "The calculations are summarized in Supplementary Table S1."
    packet = _packet(
        text=packet_text,
        source_hash="packet-source-hash",
        source_anchor="packet-source-anchor",
    )
    request = _request(
        packet,
        context=[
            _context(
                text=f"{packet_text} A separate experimental control follows.",
                element_id="unrelated-context-anchor",
                source_hash="unrelated-context-hash",
            )
        ],
    )

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_kind == OledSupplementaryTargetKind.UNKNOWN
    assert item.target_locator is None
    assert item.reference_anchors == []


def test_canonical_full_text_binding_tolerates_mineru_formatting_without_substring_matching() -> None:
    packet_text = "The calculation con fi rmed the assignment in Supplementary Table S1."
    context_text = (
        "The calculation con<sup>fi</sup>rmed the assignment in Supplementary Table S1."
    )
    packet = _packet(
        text=packet_text,
        source_hash="packet-source-hash",
        source_anchor="packet-source-anchor",
    )
    request = _request(
        packet,
        context=[
            _context(
                text=context_text,
                element_id="context-source-anchor",
                source_hash="context-source-hash",
            )
        ],
    )

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND
    assert item.target_locator == "S1"
    assert item.reference_anchors[0].matched_text == "Supplementary Table S1"


def test_unrelated_document_reference_cannot_be_assigned_to_source_check_packet() -> None:
    packet = _packet(
        text="The calculated values require an unavailable supplementary source for assignment."
    )
    request = _request(
        packet,
        context=[
            _context(
                text="Supplementary Table S1 summarizes a separate experimental control.",
                element_id="el_p8_0042",
                page=8,
            )
        ],
    )

    item = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    ).items[0]

    assert item.status == OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED
    assert item.target_kind == OledSupplementaryTargetKind.UNKNOWN
    assert item.target_locator is None
    assert item.reference_anchors == []


def test_non_supplementary_source_check_does_not_create_recovery_item() -> None:
    text = "The compound label requires a cited external reference."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])

    plan = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request, missing_evidence=["external_reference"]),
    )

    assert plan.items == []
    assert plan.item_count == 0


@pytest.mark.parametrize("mismatch", ["paper_id", "request_digest", "packet_set"])
def test_request_result_binding_mismatches_fail_closed(mismatch: str) -> None:
    text = "See Supplementary Table S1."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])
    result = _source_check_result(request)
    if mismatch == "paper_id":
        result = result.model_copy(update={"paper_id": "paper-other"})
    elif mismatch == "request_digest":
        result = result.model_copy(update={"request_digest": "wrong-request-digest"})
    else:
        result = result.model_copy(update={"packet_results": []})

    with pytest.raises(ValueError, match="does not match|packet set"):
        build_oled_supplementary_evidence_recovery_plan(request, result)


def test_invalid_mapping_result_fails_closed() -> None:
    text = "See Supplementary Table S1."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])
    invalid_result = _source_check_result(request).model_copy(
        update={"status": "invalid_response"}
    )

    with pytest.raises(ValueError, match="valid mapping result"):
        build_oled_supplementary_evidence_recovery_plan(request, invalid_result)


def test_repeated_bound_references_keep_all_anchors_and_plan_is_strictly_round_trippable() -> None:
    text = "The calculated values are reported in Supplementary Table S1."
    packet = _packet(text=text)
    request = _request(
        packet,
        context=[
            _context(text=text, element_id="el_p3_0039", page=3),
            _context(text=text, element_id="el_p4_0051", page=4),
        ],
    )
    plan = build_oled_supplementary_evidence_recovery_plan(
        request,
        _source_check_result(request),
    )

    assert len(plan.items[0].reference_anchors) == 2
    reloaded = OledSupplementaryEvidenceRecoveryPlan.model_validate_json(plan.model_dump_json())
    assert reloaded == plan
    bad_payload = plan.model_dump(mode="json")
    bad_payload["unknown_field"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        OledSupplementaryEvidenceRecoveryPlan.model_validate(bad_payload)


def test_file_artifact_and_cli_write_only_a_content_bound_offline_plan(tmp_path) -> None:
    text = "The calculated values are reported in Supplementary Table S1."
    packet = _packet(text=text)
    request = _request(packet, context=[_context(text=text)])
    request_artifact = OledLLMContextRequestArtifact(
        run_id="source-run",
        paper_id=request.paper_id,
        generated_at="2026-07-13T00:00:00Z",
        request_digest=request.request_digest,
        request=request,
    )
    result = _source_check_result(request)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "recovery.json"
    write_json(request_path, request_artifact.model_dump(mode="json"))
    write_json(result_path, result.model_dump(mode="json"))

    artifact = prepare_oled_supplementary_evidence_recovery_from_files(
        request_artifact_json=request_path,
        mapping_result_json=result_path,
        output_json=output_path,
        run_id="recovery-run",
        generated_at="2026-07-13T01:00:00Z",
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact.plan.item_count == 1
    assert written["plan_digest"] == artifact.plan.plan_digest
    assert written["network_accessed"] is False
    assert written["supplementary_downloaded"] is False
    assert written["source_context_digest"] == artifact.plan.source_context_digest
    assert written["device_only_admitted"] is False
    assert str(tmp_path) not in json.dumps(written)
    unsafe_payload = artifact.model_dump(mode="json")
    unsafe_payload["device_only_admitted"] = True
    with pytest.raises(ValidationError, match="execution side effect"):
        OledSupplementaryEvidenceRecoveryArtifact.model_validate(unsafe_payload)

    cli_output_path = tmp_path / "cli-recovery.json"
    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "--llm-context-request",
            str(request_path),
            "--llm-context-result",
            str(result_path),
            "--output",
            str(cli_output_path),
            "--run-id",
            "cli-recovery-run",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue())["item_count"] == 1
    assert cli_output_path.exists()
