from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent.domains import oled_supplementary_semantic_review as semantic_domain
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementaryColumnMappingReviewItem,
    OledSupplementaryScopeSemanticNoteReviewItem,
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticDecisionManifest,
    OledSupplementarySemanticReviewPacket,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    build_oled_supplementary_scoped_candidate_response_from_files,
)
from ai4s_agent.oled_supplementary_semantic_review import (
    build_oled_supplementary_semantic_adjudication_from_files,
    build_oled_supplementary_semantic_review_packet_from_files,
    main,
    render_oled_supplementary_semantic_review_packet_from_files,
)
from tests.test_oled_supplementary_scoped_candidate_response import (
    _SEMANTIC_NOTE,
    _build_chain,
    _recompute_request_after_table_change,
    _response_payload,
    _sha256_file,
    _stable_hash,
)


_RESPONSE_GENERATED_AT = "2026-07-13T22:00:00+08:00"
_PACKET_GENERATED_AT = "2026-07-13T22:10:00+08:00"
_REVIEWED_AT = "2026-07-13T22:20:00+08:00"
_ADJUDICATED_AT = "2026-07-13T22:30:00+08:00"


def _build_validated_response_chain(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    request_path, response_path, response_artifact_path, request, response = _build_chain(
        tmp_path
    )
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )
    return request_path, response_path, response_artifact_path, request, response


def _build_packet(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    OledSupplementarySemanticReviewPacket,
]:
    request_path, response_path, response_artifact_path, _, _ = (
        _build_validated_response_chain(tmp_path)
    )
    packet_path = tmp_path / "supplementary-semantic-review.json"
    packet = build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=packet_path,
        generated_at=_PACKET_GENERATED_AT,
    )
    return request_path, response_path, response_artifact_path, packet_path, packet


def _positive_decision_for_item(item: Any) -> tuple[str, str]:
    if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem):
        return (
            "resolve_semantic_note_as_reported",
            "The reported HOMO/LUMO labels and values are accepted without rewriting them.",
        )
    assert isinstance(item, OledSupplementaryColumnMappingReviewItem)
    disposition = item.disposition_summary.disposition.value
    return {
        "propose_known_property": ("accept_known_mapping", ""),
        "needs_ontology_review": ("confirm_ontology_review", ""),
        "needs_source_check": ("confirm_source_check", ""),
        "exclude_from_dataset": ("accept_exclusion", ""),
    }[disposition]


def _decision_payload(
    packet: OledSupplementarySemanticReviewPacket,
    packet_path: Path,
) -> dict[str, Any]:
    decisions = []
    for item in packet.review_items:
        decision, review_note = _positive_decision_for_item(item)
        decisions.append(
            {
                "review_item_id": item.review_item_id,
                "review_item_digest": item.review_item_digest,
                "item_kind": item.item_kind.value,
                "decision": decision,
                "review_note": review_note,
            }
        )
    return {
        "schema_version": "oled_supplementary_semantic_decision_manifest.v1",
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "review_packet_sha256": _sha256_file(packet_path),
        "review_packet_digest": packet.review_packet_digest,
        "reviewed_by": "Benton",
        "reviewed_at": _REVIEWED_AT,
        "adjudication_confirmed": True,
        "decisions": decisions,
    }


def _write_decisions(
    tmp_path: Path,
    packet: OledSupplementarySemanticReviewPacket,
    packet_path: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = _decision_payload(packet, packet_path)
    path = tmp_path / "supplementary-semantic-decisions.json"
    write_json(path, payload)
    return path, payload


def _build_complete_review_chain(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    OledSupplementarySemanticReviewPacket,
    dict[str, Any],
]:
    request_path, response_path, response_artifact_path, packet_path, packet = (
        _build_packet(tmp_path)
    )
    decision_path, decision_payload = _write_decisions(tmp_path, packet, packet_path)
    return (
        request_path,
        response_path,
        response_artifact_path,
        packet_path,
        decision_path,
        packet,
        decision_payload,
    )


def _adjudicate(
    tmp_path: Path,
    chain: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        OledSupplementarySemanticReviewPacket,
        dict[str, Any],
    ],
    *,
    output_name: str = "supplementary-semantic-adjudication.json",
) -> OledSupplementarySemanticAdjudicationArtifact:
    request_path, response_path, response_artifact_path, packet_path, decision_path, _, _ = (
        chain
    )
    return build_oled_supplementary_semantic_adjudication_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        review_packet_json=packet_path,
        decision_manifest_json=decision_path,
        output_json=tmp_path / output_name,
        generated_at=_ADJUDICATED_AT,
    )


def _recompute_adjudication_digest(payload: dict[str, Any]) -> None:
    payload["adjudication_artifact_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "adjudication_artifact_digest"
        }
    )


def _markdown_group_block(markdown: str, column_name: str) -> str:
    heading_marker = f": {semantic_domain._md_code(column_name)}"
    start = markdown.index(heading_marker)
    end = markdown.find("\n#### G", start + len(heading_marker))
    return markdown[start:] if end == -1 else markdown[start:end]


def test_paper016_packet_compacts_review_without_losing_any_cell(
    tmp_path: Path,
) -> None:
    request_path, response_path, response_artifact_path, packet_path, packet = (
        _build_packet(tmp_path)
    )

    assert packet.status.value == "ready_for_human_semantic_review"
    assert packet.scope_count == 1
    assert packet.review_item_count == 8
    assert packet.mapping_review_item_count == 7
    assert packet.semantic_note_review_item_count == 1
    assert packet.source_cell_count == 49
    assert packet.known_property_cell_count == 35
    assert packet.ontology_review_cell_count == 14
    assert packet.source_check_cell_count == 0
    assert packet.exclusion_cell_count == 0
    assert packet.request_artifact_sha256 == _sha256_file(request_path)
    assert packet.response_manifest_sha256 == _sha256_file(response_path)
    assert packet.response_artifact_sha256 == _sha256_file(response_artifact_path)

    mapping_items = [
        item
        for item in packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
    ]
    semantic_items = [
        item
        for item in packet.review_items
        if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem)
    ]
    assert len(mapping_items) == 7
    assert {item.column_name for item in mapping_items} == set(
        packet.scopes[0].matched_table.headers[1:]
    )
    assert {item.member_cell_count for item in mapping_items} == {7}
    all_cells = [cell for item in mapping_items for cell in item.member_cells]
    assert len(all_cells) == len({cell.source_cell_digest for cell in all_cells}) == 49
    assert len({cell.cell_disposition_digest for cell in all_cells}) == 49
    assert {(cell.row_index, cell.column_index) for cell in all_cells} == {
        (row_index, column_index)
        for row_index in range(7)
        for column_index in range(1, 8)
    }
    assert len(semantic_items) == 1
    assert semantic_items[0].semantic_note == _SEMANTIC_NOTE
    assert semantic_items[0].affected_mapping_review_item_ids == sorted(
        item.review_item_id for item in mapping_items
    )
    exact_literals = {
        cell.reported_value_text: cell.reported_decimal_places
        for cell in all_cells
        if cell.reported_value_text in {"2.80", "3.30", "0.1280", "-1.70", "-5.50"}
    }
    assert exact_literals == {
        "2.80": 2,
        "3.30": 2,
        "0.1280": 4,
        "-1.70": 2,
        "-5.50": 2,
    }
    assert packet.complete_source_context_included is True
    assert packet.strict_cell_partition_validated is True
    assert packet.human_semantic_review_completed is False
    assert packet.schema_candidates_created is False
    assert packet.reviewed_evidence_staging is False
    assert packet.direct_admission_eligible is False
    assert packet.device_only_admitted is False
    assert packet.gold_records_created is False
    assert packet.dataset_written is False
    assert OledSupplementarySemanticReviewPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    ) == packet


def test_markdown_shows_source_table_once_and_only_eight_review_items(
    tmp_path: Path,
) -> None:
    *_, packet_path, packet = _build_packet(tmp_path)
    markdown_path = tmp_path / "supplementary-semantic-review.md"

    rendered = render_oled_supplementary_semantic_review_packet_from_files(
        review_packet_json=packet_path,
        output_markdown=markdown_path,
    )

    assert rendered == packet
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.count("### Bound table") == 1
    assert markdown.count("Supplementary Table S1. TD-DFT properties") == 1
    assert markdown.count("### Mandatory scope semantic note") == 1
    assert markdown.count("#### G") == 7
    assert "Source cells: 49" in markdown
    assert "Mapping groups: 7" in markdown
    assert "Semantic-note items: 1" in markdown
    assert f"Run ID: <code>{packet.run_id}</code>" in markdown
    assert f"Exact packet-file SHA-256: <code>{_sha256_file(packet_path)}</code>" in markdown
    assert markdown.count("Review-item digest:") == 8
    assert "0.1280" in markdown
    assert "`dp` means decimal places" in markdown
    assert "Subject column: <code>column_1</code> (0-based index 0)" in markdown
    group_headings = [
        f"#### G{index:02d}: <code>{header}</code>"
        for index, header in enumerate(
            packet.scopes[0].matched_table.headers[1:],
            start=1,
        )
    ]
    assert all(heading in markdown for heading in group_headings)


def test_markdown_distinguishes_missing_source_unit_from_explicit_ev(
    tmp_path: Path,
) -> None:
    *_, packet_path, packet = _build_packet(tmp_path)
    markdown_path = tmp_path / "supplementary-semantic-review.md"
    render_oled_supplementary_semantic_review_packet_from_files(
        review_packet_json=packet_path,
        output_markdown=markdown_path,
    )

    ontology_groups = {
        item.column_name: item
        for item in packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
        and item.disposition_summary.disposition.value == "needs_ontology_review"
    }
    missing_unit_group = ontology_groups["$f(S_0-S_1)^b$"]
    explicit_ev_group = ontology_groups[
        "$\\Delta E_{\\text{HOMO} \\rightarrow \\text{LUMO}}$ (eV)"
    ]
    assert missing_unit_group.disposition_summary.reported_unit == ""
    assert explicit_ev_group.disposition_summary.reported_unit == "eV"

    serialized_packet = json.loads(packet_path.read_text(encoding="utf-8"))
    serialized_missing_group = next(
        item
        for item in serialized_packet["review_items"]
        if item["review_item_id"] == missing_unit_group.review_item_id
    )
    assert serialized_missing_group["disposition_summary"]["reported_unit"] == ""

    markdown = markdown_path.read_text(encoding="utf-8")
    missing_unit_block = _markdown_group_block(
        markdown,
        missing_unit_group.column_name,
    )
    explicit_ev_block = _markdown_group_block(
        markdown,
        explicit_ev_group.column_name,
    )
    assert (
        "reported unit: <code>no explicit unit in source header</code>"
        in missing_unit_block
    )
    assert "reported unit: <code>unitless</code>" not in missing_unit_block
    assert "reported unit: <code>eV</code>" in explicit_ev_block
    assert "no explicit unit in source header" not in explicit_ev_block


def test_markdown_preserves_explicit_dimensionless_source_unit(
    tmp_path: Path,
) -> None:
    request_path, response_path, response_artifact_path, request, _ = _build_chain(
        tmp_path
    )
    table = request["scopes"][0]["matched_table"]
    old_header = "$f(S_0-S_1)^b$"
    new_header = "Oscillator strength (unitless)"
    table["headers"][-1] = new_header
    for row in table["rows"]:
        row[new_header] = row.pop(old_header)
    _recompute_request_after_table_change(request)
    write_json(request_path, request)
    write_json(response_path, _response_payload(request_path, request))
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )
    packet_path = tmp_path / "dimensionless-packet.json"
    packet = build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=packet_path,
        generated_at=_PACKET_GENERATED_AT,
    )
    markdown_path = tmp_path / "dimensionless-packet.md"
    render_oled_supplementary_semantic_review_packet_from_files(
        review_packet_json=packet_path,
        output_markdown=markdown_path,
    )

    dimensionless_group = next(
        item
        for item in packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
        and item.column_name == new_header
    )
    assert dimensionless_group.disposition_summary.reported_unit == "unitless"
    serialized_packet = json.loads(packet_path.read_text(encoding="utf-8"))
    serialized_dimensionless_group = next(
        item
        for item in serialized_packet["review_items"]
        if item["review_item_id"] == dimensionless_group.review_item_id
    )
    assert (
        serialized_dimensionless_group["disposition_summary"]["reported_unit"]
        == "unitless"
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    dimensionless_block = _markdown_group_block(markdown, new_header)
    assert "reported unit: <code>unitless</code>" in dimensionless_block
    assert "no explicit unit in source header" not in dimensionless_block


def test_review_packet_cannot_predate_the_validated_pr_h_artifact(
    tmp_path: Path,
) -> None:
    request_path, response_path, response_artifact_path, _, _ = (
        _build_validated_response_chain(tmp_path)
    )
    output_path = tmp_path / "predated-semantic-review.json"

    with pytest.raises(ValueError, match="predates PR-H"):
        build_oled_supplementary_semantic_review_packet_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            response_artifact_json=response_artifact_path,
            output_json=output_path,
            generated_at="2026-07-13T21:59:59+08:00",
        )

    assert not output_path.exists()


def test_complete_human_decisions_expand_back_to_49_exact_cell_results(
    tmp_path: Path,
) -> None:
    chain = _build_complete_review_chain(tmp_path)

    artifact = _adjudicate(tmp_path, chain)

    assert artifact.status.value == "review_complete_with_unresolved_items"
    assert artifact.review_item_count == 8
    assert artifact.group_count == 7
    assert artifact.semantic_note_count == 1
    assert artifact.cell_count == 49
    assert artifact.later_eligible_group_count == 5
    assert artifact.later_eligible_cell_count == 35
    assert artifact.ontology_review_pending_group_count == 2
    assert artifact.ontology_review_pending_cell_count == 14
    assert artifact.source_check_pending_group_count == 0
    assert artifact.source_check_pending_cell_count == 0
    assert artifact.unresolved_review_item_count == 2
    assert len({cell.source_cell.source_cell_digest for cell in artifact.adjudicated_cells}) == 49
    assert sum(
        cell.eligible_for_later_materialization_review
        for cell in artifact.adjudicated_cells
    ) == 35
    assert sum(cell.ontology_review_pending for cell in artifact.adjudicated_cells) == 14
    assert artifact.adjudicated_semantic_notes[0].semantic_note_resolved is True
    assert artifact.human_semantic_review_completed is True
    for field_name in (
        "table_transcription_validated",
        "table_exhaustiveness_validated",
        "scientific_content_validated",
        "physical_semantics_validated",
        "material_identity_resolved",
        "source_values_corrected",
        "ontology_extensions_applied",
        "schema_candidates_created",
        "automatic_candidate_merge",
        "reviewed_evidence_staging",
        "direct_admission_eligible",
        "device_only_admitted",
        "gold_records_created",
        "dataset_written",
        "network_accessed",
        "external_service_called",
        "llm_called",
        "mineru_called",
    ):
        assert getattr(artifact, field_name) is False


def test_unresolved_scope_semantics_blocks_every_known_mapping(
    tmp_path: Path,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decisions = deepcopy(chain[6])
    semantic_entry = next(
        item
        for item in decisions["decisions"]
        if item["item_kind"] == "scope_semantic_note"
    )
    semantic_entry["decision"] = "needs_source_check"
    semantic_entry["review_note"] = "The reported HOMO/LUMO meaning needs source review."
    write_json(chain[4], decisions)

    artifact = _adjudicate(tmp_path, chain)

    assert artifact.status.value == "review_complete_with_unresolved_items"
    assert artifact.later_eligible_group_count == 0
    assert artifact.later_eligible_cell_count == 0
    assert artifact.unresolved_review_item_count == 3
    assert all(
        group.blocked_by_scope_semantics
        for group in artifact.adjudicated_groups
    )


def test_serialized_adjudication_cannot_bypass_unresolved_scope_semantics(
    tmp_path: Path,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decisions = deepcopy(chain[6])
    semantic_entry = next(
        item
        for item in decisions["decisions"]
        if item["item_kind"] == "scope_semantic_note"
    )
    semantic_entry["decision"] = "needs_source_check"
    semantic_entry["review_note"] = "The reported semantics require source review."
    write_json(chain[4], decisions)
    payload = _adjudicate(tmp_path, chain).model_dump(mode="json")
    known_group_ids = {
        group["review_item_id"]
        for group in payload["adjudicated_groups"]
        if group["disposition_summary"]["disposition"]
        == "propose_known_property"
    }
    for group in payload["adjudicated_groups"]:
        if group["review_item_id"] in known_group_ids:
            group["blocked_by_scope_semantics"] = False
            group["eligible_for_later_materialization_review"] = True
    for cell in payload["adjudicated_cells"]:
        if cell["decision_source_review_item_id"] in known_group_ids:
            cell["blocked_by_scope_semantics"] = False
            cell["eligible_for_later_materialization_review"] = True
    payload["later_eligible_group_count"] = len(known_group_ids)
    payload["later_eligible_cell_count"] = 35
    _recompute_adjudication_digest(payload)

    with pytest.raises(ValidationError, match="scope semantic blocking mismatch"):
        OledSupplementarySemanticAdjudicationArtifact.model_validate(payload)


def test_scope_rejection_terminally_inactivates_every_group(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decisions = deepcopy(chain[6])
    semantic_entry = next(
        item
        for item in decisions["decisions"]
        if item["item_kind"] == "scope_semantic_note"
    )
    semantic_entry["decision"] = "reject_scope"
    semantic_entry["review_note"] = "Reject this scope after semantic review."
    write_json(chain[4], decisions)

    artifact = _adjudicate(tmp_path, chain)

    assert artifact.status.value == "review_complete_no_eligible_mappings"
    assert artifact.unresolved_review_item_count == 0
    assert artifact.later_eligible_cell_count == 0
    assert artifact.ontology_review_pending_cell_count == 0
    assert artifact.source_check_pending_cell_count == 0
    assert artifact.rejected_group_count == 7
    assert artifact.rejected_cell_count == 49
    assert all(group.rejected_by_scope for group in artifact.adjudicated_groups)


def test_serialized_adjudication_rejects_kind_incompatible_group_decision(
    tmp_path: Path,
) -> None:
    payload = _adjudicate(
        tmp_path,
        _build_complete_review_chain(tmp_path),
    ).model_dump(mode="json")
    target = next(
        group
        for group in payload["adjudicated_groups"]
        if group["disposition_summary"]["disposition"]
        == "needs_ontology_review"
    )
    target["decision"] = "accept_known_mapping"
    target["ontology_review_pending"] = False
    for cell in payload["adjudicated_cells"]:
        if cell["decision_source_review_item_id"] == target["review_item_id"]:
            cell["decision"] = "accept_known_mapping"
            cell["ontology_review_pending"] = False
    payload["ontology_review_pending_group_count"] -= 1
    payload["ontology_review_pending_cell_count"] -= 7
    payload["unresolved_review_item_count"] -= 1
    _recompute_adjudication_digest(payload)

    with pytest.raises(ValidationError, match="decision is incompatible"):
        OledSupplementarySemanticAdjudicationArtifact.model_validate(payload)


def test_serialized_adjudication_rejects_mapping_summary_tamper(
    tmp_path: Path,
) -> None:
    payload = _adjudicate(
        tmp_path,
        _build_complete_review_chain(tmp_path),
    ).model_dump(mode="json")
    target = next(
        group
        for group in payload["adjudicated_groups"]
        if group["disposition_summary"]["disposition"]
        == "propose_known_property"
    )
    target["disposition_summary"]["proposal_note"] = "Unbound altered proposal"
    _recompute_adjudication_digest(payload)

    with pytest.raises(ValidationError, match="cell disposition binding mismatch"):
        OledSupplementarySemanticAdjudicationArtifact.model_validate(payload)


def test_decision_entries_may_follow_human_facing_order(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decisions = deepcopy(chain[6])
    decisions["decisions"] = list(reversed(decisions["decisions"]))
    write_json(chain[4], decisions)

    artifact = _adjudicate(tmp_path, chain)

    assert artifact.cell_count == 49


def test_different_proposal_note_in_one_column_splits_the_review_group(
    tmp_path: Path,
) -> None:
    request_path, response_path, response_artifact_path, request, response = _build_chain(
        tmp_path
    )
    response["scope_results"][0]["cell_dispositions"][0]["proposal_note"] = (
        "Row-specific mapping caveat"
    )
    write_json(response_path, response)
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )
    packet = build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=tmp_path / "split-packet.json",
        generated_at=_PACKET_GENERATED_AT,
    )

    homo_groups = [
        item
        for item in packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
        and item.column_name == request["scopes"][0]["matched_table"]["headers"][1]
    ]
    assert packet.mapping_review_item_count == 8
    assert packet.review_item_count == 9
    assert sorted(item.member_cell_count for item in homo_groups) == [1, 6]
    assert packet.source_cell_count == 49


def test_markdown_displays_every_known_mapping_context_and_proposal_note(
    tmp_path: Path,
) -> None:
    request_path, _, response_artifact_path, request, _ = _build_chain(tmp_path)
    table = request["scopes"][0]["matched_table"]
    old_header = table["headers"][1]
    new_header = "PL peak (nm)"
    table["headers"][1] = new_header
    for row in table["rows"]:
        row[new_header] = row.pop(old_header)
    _recompute_request_after_table_change(request)
    write_json(request_path, request)
    response = _response_payload(request_path, request)
    for disposition in response["scope_results"][0]["cell_dispositions"]:
        if disposition["column_name"] != new_header:
            continue
        disposition.pop("proposed_target_layer")
        disposition.pop("ontology_review_reason")
        disposition.update(
            {
                "disposition": "propose_known_property",
                "proposal_note": "Film values; retain the reported host context.",
                "property_id": "photoluminescence_peak_nm",
                "property_label": new_header,
                "target_layer": "interaction",
                "reported_unit": "nm",
                "canonical_unit": "nm",
                "comparison_context": {
                    "measurement_temperature": 300,
                    "measurement_temperature_unit": "K",
                    "host_material": "mCP",
                    "dopant_concentration": "10",
                    "dopant_concentration_unit": "wt%",
                    "sample_form": "thin film",
                    "excitation_wavelength": 365,
                    "excitation_wavelength_unit": "nm",
                    "lifetime_fit_method": "not applicable",
                },
            }
        )
    response_path = tmp_path / "context-response.json"
    write_json(response_path, response)
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )
    packet_path = tmp_path / "context-packet.json"
    build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=packet_path,
        generated_at=_PACKET_GENERATED_AT,
    )
    markdown_path = tmp_path / "context-packet.md"
    render_oled_supplementary_semantic_review_packet_from_files(
        review_packet_json=packet_path,
        output_markdown=markdown_path,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    for literal in (
        "photoluminescence_peak_nm",
        "PL peak (nm)",
        "interaction",
        "Film values; retain the reported host context.",
        '"host_material":"mCP"',
        '"dopant_concentration":"10"',
        '"sample_form":"thin film"',
        '"excitation_wavelength":365',
        '"lifetime_fit_method":"not applicable"',
    ):
        assert literal in markdown


def test_markdown_escape_makes_source_markup_and_display_controls_inert() -> None:
    rendered = semantic_domain._md_code("<script>alert(1)</script>|a\nb\u202e")

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&#124;" in rendered
    assert " ⏎ " in rendered
    assert "\\u202E" in rendered


def test_rejects_self_consistent_pr_h_tamper_not_present_in_original_manifest(
    tmp_path: Path,
) -> None:
    request_path, response_path, response_artifact_path, _, _ = (
        _build_validated_response_chain(tmp_path)
    )
    payload = json.loads(response_artifact_path.read_text(encoding="utf-8"))
    payload["scope_results"][0]["cell_dispositions"][0]["proposal_note"] = (
        "Tampered after PR-H validation"
    )
    reconstructed_manifest = {
        "schema_version": "oled_supplementary_scoped_candidate_response_manifest.v1",
        "run_id": payload["run_id"],
        "paper_id": payload["paper_id"],
        "request_artifact_sha256": payload["request_artifact_sha256"],
        "request_digest": payload["request_digest"],
        "producer": payload["producer"],
        "response_complete": True,
        "scope_results": payload["scope_results"],
    }
    payload["response_manifest_digest"] = _stable_hash(reconstructed_manifest)
    payload["response_artifact_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "response_artifact_digest"
        }
    )
    write_json(response_artifact_path, payload)

    with pytest.raises(ValueError, match="PR-H artifact binding mismatch"):
        build_oled_supplementary_semantic_review_packet_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            response_artifact_json=response_artifact_path,
            output_json=tmp_path / "tampered-packet.json",
            generated_at=_PACKET_GENERATED_AT,
        )

    assert not (tmp_path / "tampered-packet.json").exists()


def test_packet_model_rejects_a_valid_but_orphaned_mapping_group(
    tmp_path: Path,
) -> None:
    *_, packet = _build_packet(tmp_path / "first")
    second = tmp_path / "second"
    second.mkdir(parents=True)
    request_path, response_path, response_artifact_path, request, _ = _build_chain(
        second
    )
    request["scopes"][0]["review_item_id"] = (
        "supplementary-locator-review:supplementary-recovery:item-002"
    )
    _recompute_request_after_table_change(request)
    write_json(request_path, request)
    write_json(response_path, _response_payload(request_path, request))
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )
    orphan_packet = build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=second / "packet.json",
        generated_at=_PACKET_GENERATED_AT,
    )
    orphan_group = next(
        item
        for item in orphan_packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
    )
    payload = packet.model_dump(mode="json")
    replaced = False
    for index, item in enumerate(payload["review_items"]):
        if item["item_kind"] == "column_mapping_group":
            payload["review_items"][index] = orphan_group.model_dump(mode="json")
            replaced = True
            break
    assert replaced
    payload["review_items"].sort(key=lambda item: item["review_item_id"])
    payload["review_packet_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "review_packet_digest"
        }
    )

    with pytest.raises(ValidationError, match="exactly cover packet scopes"):
        OledSupplementarySemanticReviewPacket.model_validate(payload)


@pytest.mark.parametrize("coverage_error", ["missing", "duplicate", "unknown"])
def test_decisions_must_cover_every_review_item_exactly_once(
    tmp_path: Path,
    coverage_error: str,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decision_path = chain[4]
    decisions = deepcopy(chain[6])
    if coverage_error == "missing":
        decisions["decisions"].pop()
    elif coverage_error == "duplicate":
        decisions["decisions"][-1] = deepcopy(decisions["decisions"][0])
    else:
        decisions["decisions"][-1]["review_item_id"] = "semantic-review:unknown"
    decisions["decisions"].sort(key=lambda item: item["review_item_id"])
    write_json(decision_path, decisions)

    with pytest.raises((ValidationError, ValueError)):
        _adjudicate(tmp_path, chain)

    assert not (tmp_path / "supplementary-semantic-adjudication.json").exists()


@pytest.mark.parametrize("tamper", ["item_digest", "item_kind"])
def test_decision_items_remain_exact_bound_to_the_packet(
    tmp_path: Path,
    tamper: str,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decision_path = chain[4]
    decisions = deepcopy(chain[6])
    if tamper == "item_digest":
        decisions["decisions"][0]["review_item_digest"] = "sha256:" + "f" * 64
    else:
        current = decisions["decisions"][0]["item_kind"]
        decisions["decisions"][0]["item_kind"] = (
            "scope_semantic_note"
            if current == "column_mapping_group"
            else "column_mapping_group"
        )
    write_json(decision_path, decisions)

    with pytest.raises(ValueError, match="item binding mismatch"):
        _adjudicate(tmp_path, chain)


def test_positive_decision_is_specific_to_the_disposition_kind(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    packet = chain[5]
    decision_path = chain[4]
    decisions = deepcopy(chain[6])
    ontology_id = next(
        item.review_item_id
        for item in packet.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
        and item.disposition_summary.disposition.value == "needs_ontology_review"
    )
    entry = next(
        item for item in decisions["decisions"] if item["review_item_id"] == ontology_id
    )
    entry["decision"] = "accept_known_mapping"
    write_json(decision_path, decisions)

    with pytest.raises(ValueError, match="incompatible with review item kind"):
        _adjudicate(tmp_path, chain)


def test_semantic_note_cannot_be_silently_accepted_as_a_mapping(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decision_path = chain[4]
    decisions = deepcopy(chain[6])
    semantic_entry = next(
        item
        for item in decisions["decisions"]
        if item["item_kind"] == "scope_semantic_note"
    )
    semantic_entry["decision"] = "accept_known_mapping"
    semantic_entry["review_note"] = ""
    write_json(decision_path, decisions)

    with pytest.raises(ValueError, match="incompatible with review item kind"):
        _adjudicate(tmp_path, chain)


def test_changed_response_manifest_bytes_invalidate_packet_at_adjudication(
    tmp_path: Path,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    response_path = chain[1]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PR-H artifact binding mismatch"):
        _adjudicate(tmp_path, chain)


def test_decision_manifest_binds_exact_review_packet_bytes(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    packet_path = chain[3]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact review packet bytes"):
        _adjudicate(tmp_path, chain)


@pytest.mark.parametrize("protected_input", ["request", "manifest", "response"])
def test_packet_output_must_not_overwrite_any_upstream_input(
    tmp_path: Path,
    protected_input: str,
) -> None:
    request_path, response_path, response_artifact_path, _, _ = (
        _build_validated_response_chain(tmp_path)
    )
    protected = {
        "request": request_path,
        "manifest": response_path,
        "response": response_artifact_path,
    }[protected_input]
    original = protected.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_supplementary_semantic_review_packet_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            response_artifact_json=response_artifact_path,
            output_json=protected,
            generated_at=_PACKET_GENERATED_AT,
        )

    assert protected.read_bytes() == original


@pytest.mark.parametrize(
    "protected_index",
    [0, 1, 2, 3, 4],
    ids=["request", "manifest", "response", "packet", "decisions"],
)
def test_adjudication_output_must_not_overwrite_any_input(
    tmp_path: Path,
    protected_index: int,
) -> None:
    chain = _build_complete_review_chain(tmp_path)
    protected = chain[protected_index]
    assert isinstance(protected, Path)
    original = protected.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_supplementary_semantic_adjudication_from_files(
            request_artifact_json=chain[0],
            response_manifest_json=chain[1],
            response_artifact_json=chain[2],
            review_packet_json=chain[3],
            decision_manifest_json=chain[4],
            output_json=protected,
            generated_at=_ADJUDICATED_AT,
        )

    assert protected.read_bytes() == original


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "token=abc123",
        "Bearer abc12345",
        "sk-abcdef123456",
        "https://example.invalid/review",
        "/operator/private/review.json",
        "../operator/private/review.json",
        "```python\nimport os\n```",
        "review\u202eidentity",
    ],
)
def test_reviewer_text_rejects_credentials_paths_urls_and_executable_content(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    *_, packet_path, packet = _build_packet(tmp_path)
    decisions = _decision_payload(packet, packet_path)
    decisions["decisions"][0]["review_note"] = unsafe_text
    decision_path = tmp_path / "unsafe-decisions.json"
    write_json(decision_path, decisions)

    with pytest.raises(ValidationError):
        OledSupplementarySemanticDecisionManifest.model_validate(decisions)


def test_allows_bounded_scientific_reviewer_text(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decision_path = chain[4]
    decisions = deepcopy(chain[6])
    decisions["reviewed_by"] = "Benton"
    decisions["decisions"][0]["review_note"] = (
        "HOMO/LUMO at B3LYP/6-31G(d,p); values and trailing zeros remain as reported."
    )
    write_json(decision_path, decisions)

    artifact = _adjudicate(tmp_path, chain)

    assert artifact.cell_count == 49


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    chain = _build_complete_review_chain(tmp_path)
    decision_path = chain[4]
    decision_path.write_text(
        '{"schema_version":"oled_supplementary_semantic_decision_manifest.v1",'
        '"schema_version":"oled_supplementary_semantic_decision_manifest.v1"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        _adjudicate(tmp_path, chain)


def test_cli_success_and_failure_outputs_are_redacted(tmp_path: Path) -> None:
    request_path, response_path, response_artifact_path, _, _ = (
        _build_validated_response_chain(tmp_path)
    )
    packet_path = tmp_path / "cli-packet.json"
    stdout = StringIO()

    exit_code = main(
        [
            "packet",
            "--request-artifact",
            str(request_path),
            "--response-manifest",
            str(response_path),
            "--response-artifact",
            str(response_artifact_path),
            "--output",
            str(packet_path),
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "ready_for_human_semantic_review" in output
    assert '"review_item_count": 8' in output
    assert str(tmp_path) not in output
    assert "paper016" not in output
    assert _SEMANTIC_NOTE not in output
    assert "0.1280" not in output
    assert "claude-sonnet-test-snapshot-20260713" not in output

    packet = OledSupplementarySemanticReviewPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    decisions = _decision_payload(packet, packet_path)
    decisions["decisions"][0]["review_note"] = "token=secret-value"
    decision_path = tmp_path / "cli-unsafe-decisions.json"
    write_json(decision_path, decisions)
    failed_output = tmp_path / "cli-failed-adjudication.json"
    stdout = StringIO()
    exit_code = main(
        [
            "adjudicate",
            "--request-artifact",
            str(request_path),
            "--response-manifest",
            str(response_path),
            "--response-artifact",
            str(response_artifact_path),
            "--review-packet",
            str(packet_path),
            "--decision-manifest",
            str(decision_path),
            "--output",
            str(failed_output),
        ],
        stdout=stdout,
    )
    output = stdout.getvalue()
    assert exit_code == 2
    assert "supplementary_semantic_review_failed" in output
    assert "secret-value" not in output
    assert str(tmp_path) not in output
    assert not failed_output.exists()
