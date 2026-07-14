from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    OledSupplementaryMaterialIdentityStructureCandidate,
    build_oled_supplementary_material_identity_chemistry_validation,
    oled_supplementary_material_identity_evidence_response_artifact_digest,
    oled_supplementary_material_identity_evidence_response_manifest_digest,
)
from ai4s_agent.oled_supplementary_material_identity_evidence_response import (
    build_oled_supplementary_material_identity_evidence_response_from_files,
    main,
)
from tests.test_oled_supplementary_material_identity_candidate_request import (
    _build_request,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_PRODUCED_AT = "2026-07-13T23:20:00+08:00"
_VALIDATED_AT = "2026-07-13T23:30:00+08:00"
_CCO_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
_CCO_STANDARD_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
_STEREO_SMILES = "F/C=C(/F)Cl"
_STEREO_SMILES_INCHIKEY = "CJENPNUXCMYXPT-OWOJBTEDSA-N"
_MULTIFRAGMENT_SMILES = "CC.Cl"
_MULTIFRAGMENT_SMILES_INCHIKEY = "NARWYSCMDPLCIQ-UHFFFAOYSA-N"


def _group_binding(group: Any) -> dict[str, Any]:
    return {
        "identity_group_id": group.identity_group_id,
        "identity_group_digest": group.identity_group_digest,
        "contract_version": group.contract_version,
        "scope_id": group.scope_id,
        "source_id": group.source_id,
        "source_pdf_sha256": group.source_pdf_sha256,
        "parsed_document_sha256": group.parsed_document_sha256,
        "table_id": group.table_id,
        "table_content_digest": group.table_content_digest,
        "pdf_page_number_one_based": group.pdf_page_number_one_based,
        "source_transcription_review_item_id": (
            group.source_transcription_review_item_id
        ),
        "source_transcription_review_item_digest": (
            group.source_transcription_review_item_digest
        ),
        "row_index": group.row_index,
        "subject_column_index": group.subject_column_index,
        "subject_header_binding": group.subject_header_binding.model_dump(mode="json"),
        "reported_subject_text": group.reported_subject_text,
        "identity_dependent_cell_count": group.identity_dependent_cell_count,
        "identity_dependent_source_cell_digests": list(
            group.identity_dependent_source_cell_digests
        ),
    }


def _s27_anchor(group: Any) -> dict[str, Any]:
    return {
        "source_id": group.source_id,
        "source_pdf_sha256": group.source_pdf_sha256,
        "pdf_page_number_one_based": 27,
        "anchor_kind": "figure",
        "singleton_locator": "Supplementary Fig. S27",
        "panel_label": group.reported_subject_text,
        "evidence_roles": [
            "structure_representation",
            "subject_to_structure_link",
        ],
        "source_representation_kind": "authored_description",
        "source_representation": (
            f"Source-depicted structure labelled {group.reported_subject_text}"
        ),
        "source_excerpt": (
            "Supplementary Fig. S27 Synthetic route of TDBA-based host materials."
        ),
    }


def _anchor_only_result(group: Any) -> dict[str, Any]:
    return {
        **_group_binding(group),
        "disposition": "record_structure_anchor_only",
        "evidence_anchors": [_s27_anchor(group)],
        "proposal_note": "Structure anchor retained for later PDF-backed review.",
    }


def _source_check_result(group: Any) -> dict[str, Any]:
    return {
        **_group_binding(group),
        "disposition": "needs_source_check",
        "source_check_reason": "no_exact_structure_evidence",
        "review_note": "No machine structure candidate is asserted for this row.",
    }


def _candidate_result(group: Any, *, structure_text: str = "C(C)O") -> dict[str, Any]:
    return {
        **_group_binding(group),
        "disposition": "propose_structure_candidate",
        "evidence_anchors": [_s27_anchor(group)],
        "structure_candidate": {
            "candidate_origin": "diagram_derived",
            "structure_encoding_kind": "smiles",
            "structure_candidate_text": structure_text,
            "canonical_isomeric_smiles_candidate": "CCO",
            "inchikey_candidate": _CCO_INCHIKEY,
        },
        "proposal_note": (
            "Synthetic chemistry fixture only; source-to-structure match remains unreviewed."
        ),
    }


def _source_reported_inchi_result(group: Any) -> dict[str, Any]:
    anchor = _s27_anchor(group)
    anchor.update(
        {
            "pdf_page_number_one_based": 27,
            "anchor_kind": "text",
            "singleton_locator": "Supplementary Text S51",
            "source_representation_kind": "inchi_literal",
            "source_representation": _CCO_STANDARD_INCHI,
            "source_excerpt": (
                f"{group.reported_subject_text}: {_CCO_STANDARD_INCHI}"
            ),
        }
    )
    return {
        **_group_binding(group),
        "disposition": "propose_structure_candidate",
        "evidence_anchors": [anchor],
        "structure_candidate": {
            "candidate_origin": "source_reported_inchi",
            "structure_encoding_kind": "inchi",
            "structure_candidate_text": _CCO_STANDARD_INCHI,
            "canonical_isomeric_smiles_candidate": "CCO",
            "inchikey_candidate": _CCO_INCHIKEY,
        },
        "proposal_note": "Synthetic source-reported InChI validation fixture.",
    }


def _source_reported_stereo_smiles_result(group: Any) -> dict[str, Any]:
    anchor = _s27_anchor(group)
    anchor["source_representation_kind"] = "smiles_literal"
    anchor["source_representation"] = _STEREO_SMILES
    return {
        **_group_binding(group),
        "disposition": "propose_structure_candidate",
        "evidence_anchors": [anchor],
        "structure_candidate": {
            "candidate_origin": "source_reported_smiles",
            "structure_encoding_kind": "smiles",
            "structure_candidate_text": _STEREO_SMILES,
            "canonical_isomeric_smiles_candidate": _STEREO_SMILES,
            "inchikey_candidate": _STEREO_SMILES_INCHIKEY,
        },
        "proposal_note": "Synthetic source-reported stereochemical SMILES fixture.",
    }


def _source_reported_multifragment_smiles_result(group: Any) -> dict[str, Any]:
    anchor = _s27_anchor(group)
    anchor["source_representation_kind"] = "smiles_literal"
    anchor["source_representation"] = _MULTIFRAGMENT_SMILES
    return {
        **_group_binding(group),
        "disposition": "propose_structure_candidate",
        "evidence_anchors": [anchor],
        "structure_candidate": {
            "candidate_origin": "source_reported_smiles",
            "structure_encoding_kind": "smiles",
            "structure_candidate_text": _MULTIFRAGMENT_SMILES,
            "canonical_isomeric_smiles_candidate": _MULTIFRAGMENT_SMILES,
            "inchikey_candidate": _MULTIFRAGMENT_SMILES_INCHIKEY,
        },
        "proposal_note": "Synthetic source-reported multi-fragment SMILES fixture.",
    }


def _manifest_payload(
    chain: dict[str, Any],
    request: Any,
    request_path: Path,
    *,
    result_factory: Callable[[Any], dict[str, Any]] = _anchor_only_result,
) -> dict[str, Any]:
    packet = chain["transcription_packet"]
    return {
        "schema_version": (
            "oled_supplementary_material_identity_evidence_response_manifest.v1"
        ),
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "request_artifact_sha256": _sha256_file(request_path),
        "material_identity_request_digest": request.material_identity_request_digest,
        "transcription_review_packet_sha256": _sha256_file(
            chain["transcription_packet_path"]
        ),
        "transcription_review_packet_digest": packet.review_packet_digest,
        "source_pdf_evidence_digest": packet.source_pdf_evidence_digest,
        "producer": {
            "kind": "human",
            "client_id": "test_operator",
            "model_provider_id": "",
            "model_snapshot_id": "",
            "prompt_contract_version": (
                "oled_supplementary_material_identity_evidence_response_prompt.v1"
            ),
            "prompt_sha256": "",
            "produced_at": _PRODUCED_AT,
        },
        "response_complete": True,
        "group_results": [
            result_factory(group) for group in request.identity_groups
        ],
    }


def _write_inputs(
    tmp_path: Path,
    *,
    result_factory: Callable[[Any], dict[str, Any]] = _anchor_only_result,
) -> tuple[dict[str, Any], Any, Path, Path]:
    chain, request, request_path = _build_request(tmp_path)
    response_path = tmp_path / "material-identity-evidence-response-manifest.json"
    write_json(
        response_path,
        _manifest_payload(
            chain,
            request,
            request_path,
            result_factory=result_factory,
        ),
    )
    return chain, request, request_path, response_path


def _build_from_files(
    tmp_path: Path,
    *,
    result_factory: Callable[[Any], dict[str, Any]] = _anchor_only_result,
) -> tuple[Any, dict[str, Any], Any, Path, Path, Path]:
    chain, request, request_path, response_path = _write_inputs(
        tmp_path,
        result_factory=result_factory,
    )
    output_path = tmp_path / "material-identity-evidence-response.json"
    artifact = build_oled_supplementary_material_identity_evidence_response_from_files(
        request_artifact_json=request_path,
        transcription_review_packet_json=chain["transcription_packet_path"],
        response_manifest_json=response_path,
        output_json=output_path,
        generated_at=_VALIDATED_AT,
    )
    return artifact, chain, request, request_path, response_path, output_path


def test_paper016_anchor_only_response_preserves_exact_partition(
    tmp_path: Path,
) -> None:
    artifact, _, _, _, _, output_path = _build_from_files(tmp_path)

    assert artifact.status.value == "ready_for_human_material_identity_review"
    assert artifact.identity_group_count == 7
    assert artifact.identity_dependent_cell_count == 35
    assert artifact.bounded_transcription_validated_cell_count == 49
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    assert artifact.structure_candidate_count == 0
    assert artifact.structure_anchor_only_count == 7
    assert artifact.evidence_anchor_count == 7
    assert artifact.chemistry_validated_candidate_count == 0
    assert not artifact.chemistry_tool_called
    assert not artifact.source_pdf_read
    assert not artifact.source_location_content_validated
    assert not artifact.source_match_validated
    assert not artifact.material_identity_resolved
    assert not artifact.canonical_smiles_assigned
    assert not artifact.inchikey_assigned
    assert not artifact.registry_written
    assert not artifact.dataset_written
    assert artifact.joint_exact_input_revalidation_required
    assert not artifact.standalone_upstream_partition_revalidation_supported
    assert not artifact.standalone_source_pdf_metadata_revalidation_supported
    assert output_path.is_file()


def test_structure_candidate_is_rdkit_validated_but_not_resolved(
    tmp_path: Path,
) -> None:
    first = True

    def result(group: Any) -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return _candidate_result(group)
        return _source_check_result(group)

    artifact, _, _, _, _, _ = _build_from_files(
        tmp_path,
        result_factory=result,
    )

    assert artifact.structure_candidate_count == 1
    assert artifact.source_check_count == 6
    assert artifact.chemistry_validated_candidate_count == 1
    assert artifact.chemistry_tool_called
    validation = artifact.validated_results[0].chemistry_validation
    assert validation is not None
    assert validation.candidate.canonical_isomeric_smiles_candidate == "CCO"
    assert validation.candidate.inchikey_candidate == _CCO_INCHIKEY
    assert validation.parse_succeeded
    assert validation.sanitization_succeeded
    assert not validation.source_match_validated
    assert not validation.material_identity_resolved


def test_source_reported_inchi_candidate_replays_through_full_validation(
    tmp_path: Path,
) -> None:
    first = True

    def result(group: Any) -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return _source_reported_inchi_result(group)
        return _source_check_result(group)

    artifact, _, _, _, _, _ = _build_from_files(
        tmp_path,
        result_factory=result,
    )

    validation = artifact.validated_results[0].chemistry_validation
    assert validation is not None
    assert validation.standard_inchi_candidate == _CCO_STANDARD_INCHI
    assert validation.candidate.canonical_isomeric_smiles_candidate == "CCO"
    assert validation.candidate.inchikey_candidate == _CCO_INCHIKEY
    assert not validation.source_match_validated


def test_source_reported_stereo_smiles_is_not_misclassified_as_a_path(
    tmp_path: Path,
) -> None:
    first = True

    def result(group: Any) -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return _source_reported_stereo_smiles_result(group)
        return _source_check_result(group)

    artifact, _, _, _, _, _ = _build_from_files(
        tmp_path,
        result_factory=result,
    )

    validation = artifact.validated_results[0].chemistry_validation
    assert validation is not None
    assert validation.candidate.structure_candidate_text == _STEREO_SMILES
    assert validation.candidate.inchikey_candidate == _STEREO_SMILES_INCHIKEY


def test_explicit_multifragment_smiles_is_chemistry_not_authored_url_text(
    tmp_path: Path,
) -> None:
    first = True

    def result(group: Any) -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return _source_reported_multifragment_smiles_result(group)
        return _source_check_result(group)

    artifact, _, _, _, _, _ = _build_from_files(
        tmp_path,
        result_factory=result,
    )

    validation = artifact.validated_results[0].chemistry_validation
    assert validation is not None
    assert "multi_fragment_structure" in [
        code.value for code in validation.finding_codes
    ]


def test_anchor_only_cannot_relabel_urlish_authored_text_as_chemistry(
    tmp_path: Path,
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    payload["group_results"][0]["evidence_anchors"][0].update(
        {
            "source_representation_kind": "smiles_literal",
            "source_representation": "CO.CO",
        }
    )

    with pytest.raises(ValidationError):
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
            payload
        )


@pytest.mark.parametrize(
    ("structure_text", "canonical", "inchikey", "finding"),
    [
        (
            "C.C",
            "C.C",
            "CREMABGTGYGIQB-UHFFFAOYSA-N",
            "multi_fragment_structure",
        ),
        (
            "[NH4+]",
            "[NH4+]",
            "QGZKDVFQNNGYKY-UHFFFAOYSA-O",
            "formal_charge_present",
        ),
        (
            "CC(O)Cl",
            "CC(O)Cl",
            "KJESGYZFVCIMDE-UHFFFAOYSA-N",
            "unassigned_atom_stereochemistry",
        ),
    ],
)
def test_chemistry_findings_are_retained_without_normalizing_them_away(
    structure_text: str,
    canonical: str,
    inchikey: str,
    finding: str,
) -> None:
    candidate = OledSupplementaryMaterialIdentityStructureCandidate(
        candidate_origin="diagram_derived",
        structure_encoding_kind="smiles",
        structure_candidate_text=structure_text,
        canonical_isomeric_smiles_candidate=canonical,
        inchikey_candidate=inchikey,
    )
    validation = build_oled_supplementary_material_identity_chemistry_validation(
        candidate
    )
    assert finding in [item.value for item in validation.finding_codes]
    assert not validation.source_match_validated


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["group_results"].pop(),
        lambda payload: payload["group_results"].append(
            deepcopy(payload["group_results"][0])
        ),
        lambda payload: payload["group_results"][0].update(
            {"identity_group_digest": "sha256:" + "1" * 64}
        ),
        lambda payload: payload["group_results"][0].update(
            {"reported_subject_text": "different-subject"}
        ),
        lambda payload: payload["group_results"][0].update(
            {
                "identity_dependent_source_cell_digests": (
                    payload["group_results"][1][
                        "identity_dependent_source_cell_digests"
                    ]
                )
            }
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {"source_pdf_sha256": "sha256:" + "2" * 64}
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {"pdf_page_number_one_based": 55}
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {"singleton_locator": "Supplementary Fig. S27-S29"}
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {"evidence_roles": ["structure_representation"]}
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {
                "panel_label": "another-material",
                "source_representation": "An unlabeled depicted structure",
            }
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {
                "panel_label": "another-material",
                "source_representation": (
                    "Source-depicted structure labelled TDBA-Ph"
                ),
            }
        ),
        lambda payload: payload["group_results"][0]["evidence_anchors"][0].update(
            {
                "panel_label": "another-material",
                "source_representation": (
                    "Source-depicted structure labelled TDBA(Ph)"
                ),
            }
        ),
    ],
)
def test_group_source_locator_and_role_tampering_fails_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    mutate(payload)
    response_path = tmp_path / "tampered-response.json"
    write_json(response_path, payload)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises((ValueError, ValidationError)):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=response_path,
            output_json=output_path,
            generated_at=_VALIDATED_AT,
        )
    assert not output_path.exists()


def test_subject_and_structure_roles_cannot_be_laundered_across_anchors(
    tmp_path: Path,
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    group = request.identity_groups[0]
    subject_anchor = _s27_anchor(group)
    subject_anchor.update(
        {
            "anchor_kind": "text",
            "singleton_locator": "Supplementary Text S27",
            "evidence_roles": ["subject_to_structure_link"],
            "source_representation": (
                f"Reported row subject {group.reported_subject_text}"
            ),
            "source_excerpt": (
                f"The row is labelled {group.reported_subject_text}."
            ),
        }
    )
    unrelated_structure_anchor = _s27_anchor(group)
    unrelated_structure_anchor.update(
        {
            "singleton_locator": "Supplementary Fig. S28",
            "panel_label": "another-material",
            "evidence_roles": ["structure_representation"],
            "source_representation": "An unrelated depicted structure",
            "source_excerpt": "Supplementary Fig. S28 unrelated synthesis.",
        }
    )
    payload["group_results"][0]["evidence_anchors"] = [
        subject_anchor,
        unrelated_structure_anchor,
    ]

    with pytest.raises(ValidationError):
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
            payload
        )


def test_systematic_name_derived_candidate_requires_text_anchor(tmp_path: Path) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path, result_factory=_source_check_result)
    candidate = _candidate_result(request.identity_groups[0])
    candidate["structure_candidate"]["candidate_origin"] = "systematic_name_derived"
    payload["group_results"][0] = candidate

    with pytest.raises(ValidationError):
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(payload)


@pytest.mark.parametrize(
    "field_update",
    [
        {"structure_candidate_text": "not-a-smiles"},
        {"canonical_isomeric_smiles_candidate": "CCC"},
        {"inchikey_candidate": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"},
    ],
)
def test_invalid_or_inconsistent_structure_candidates_are_rejected(
    tmp_path: Path,
    field_update: dict[str, str],
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path, result_factory=_source_check_result)
    candidate = _candidate_result(request.identity_groups[0])
    candidate["structure_candidate"].update(field_update)
    payload["group_results"][0] = candidate

    with pytest.raises(ValidationError):
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(payload)


def test_duplicate_candidate_across_groups_creates_findings_without_merge(
    tmp_path: Path,
) -> None:
    count = 0

    def result(group: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count <= 2:
            return _candidate_result(group)
        return _source_check_result(group)

    artifact, _, _, _, _, _ = _build_from_files(tmp_path, result_factory=result)

    assert artifact.structure_candidate_count == 2
    assert artifact.collision_finding_count == 2
    assert {item.finding_kind.value for item in artifact.collision_findings} == {
        "duplicate_canonical_smiles_across_groups",
        "duplicate_inchikey_across_groups",
    }
    assert all(not item.automatic_merge_performed for item in artifact.collision_findings)
    assert not artifact.automatic_candidate_merge
    assert not artifact.cross_paper_identity_merge


def test_external_producer_records_claude_cli_and_deepseek_separately(
    tmp_path: Path,
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    payload["producer"] = {
        "kind": "external_llm_assisted",
        "client_id": "claude_cli",
        "model_provider_id": "deepseek",
        "model_snapshot_id": "deepseek-v4-pro-pinned",
        "prompt_contract_version": (
            "oled_supplementary_material_identity_evidence_response_prompt.v1"
        ),
        "prompt_sha256": "sha256:" + "3" * 64,
        "produced_at": _PRODUCED_AT,
    }
    response_path = tmp_path / "external-response.json"
    write_json(response_path, payload)
    output_path = tmp_path / "validated.json"

    artifact = build_oled_supplementary_material_identity_evidence_response_from_files(
        request_artifact_json=request_path,
        transcription_review_packet_json=chain["transcription_packet_path"],
        response_manifest_json=response_path,
        output_json=output_path,
        generated_at=_VALIDATED_AT,
    )

    assert artifact.external_llm_response_ingested
    assert artifact.producer.client_id == "claude_cli"
    assert artifact.producer.model_provider_id == "deepseek"
    assert artifact.producer.model_snapshot_id == "deepseek-v4-pro-pinned"


@pytest.mark.parametrize(
    "unsafe_note",
    [
        "token=abc123",
        "read /tmp/private.json",
        "https://example.com/structure",
        "run python -c 'print(1)'",
        "safe prefix\u202Ehidden suffix",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        '<a href="javascript&colon;alert(1)">click</a>',
        "[click](javascript&colon;alert(1))",
        "[click](jav&#x61;script:alert(1))",
        "javascript&#58;alert(1)",
        "data:text/html,<svg onload=alert(1)>",
        "ghp_" + "abcdefghijklmnopqrstuvwxyz123456",
        "AKIA" + "1234567890ABCDEF",
        "AIza" + "abcdefghijklmnopqrstuvwxyz123456789",
        "sk_" + "live_51AbCdEfGhIjKlMnOpQrStUvWxYz123456",
        "npm_" + "abcdefghijklmnopqrstuvwxyz1234567890",
        "hf_" + "abcdefghijklmnopqrstuvwxyz1234567890",
        "pypi-" + "AgEIcHlwaS5vcmcCJGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6",
        "dop_" + "v1_abcdefghijklmnopqrstuvwxyz1234567890abcdef",
    ],
)
def test_response_authored_text_rejects_credentials_paths_code_and_display_controls(
    tmp_path: Path,
    unsafe_note: str,
) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    payload["group_results"][0]["proposal_note"] = unsafe_note

    with pytest.raises(ValidationError):
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(payload)


def test_manifest_content_digest_is_order_normalized(tmp_path: Path) -> None:
    chain, request, request_path = _build_request(tmp_path)
    payload = _manifest_payload(chain, request, request_path)
    forward = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(payload)
    payload["group_results"].reverse()
    reverse = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(payload)
    assert (
        oled_supplementary_material_identity_evidence_response_manifest_digest(forward)
        == oled_supplementary_material_identity_evidence_response_manifest_digest(reverse)
    )


def test_standalone_artifact_rejects_rehashed_derived_count_and_boundary_tampering(
    tmp_path: Path,
) -> None:
    artifact, _, _, _, _, _ = _build_from_files(tmp_path)

    for field_name, value in (
        ("structure_anchor_only_count", 6),
        ("upstream_ontology_review_pending_cell_count", 15),
        ("material_identity_resolved", True),
        ("registry_written", True),
        ("dataset_written", True),
        ("rdkit_version", "forged-version"),
        ("joint_exact_input_revalidation_required", False),
        ("standalone_upstream_partition_revalidation_supported", True),
        ("standalone_source_pdf_metadata_revalidation_supported", True),
    ):
        provisional = artifact.model_copy(
            update={
                field_name: value,
                "response_artifact_digest": "sha256:" + "0" * 64,
            }
        )
        payload = provisional.model_dump(mode="json")
        payload["response_artifact_digest"] = (
            oled_supplementary_material_identity_evidence_response_artifact_digest(
                provisional
            )
        )
        with pytest.raises(ValidationError):
            OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_validate(
                payload
            )


def test_file_entry_rejects_output_collision_without_changing_input(
    tmp_path: Path,
) -> None:
    chain, _, request_path, response_path = _write_inputs(tmp_path)
    original = request_path.read_bytes()

    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=response_path,
            output_json=request_path,
            generated_at=_VALIDATED_AT,
        )
    assert request_path.read_bytes() == original


def test_file_entry_rejects_symlink_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    chain, _, request_path, response_path = _write_inputs(tmp_path)
    symlink_path = tmp_path / "response-link.json"
    symlink_path.symlink_to(response_path)

    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=symlink_path,
            output_json=tmp_path / "symlink-output.json",
            generated_at=_VALIDATED_AT,
        )

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=duplicate_path,
            output_json=tmp_path / "duplicate-output.json",
            generated_at=_VALIDATED_AT,
        )

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=nonfinite_path,
            output_json=tmp_path / "nonfinite-output.json",
            generated_at=_VALIDATED_AT,
        )


def test_file_entry_rejects_symlinked_ancestor_directory(tmp_path: Path) -> None:
    chain, _, request_path, response_path = _write_inputs(tmp_path)
    alias_dir = tmp_path / "input-alias"
    alias_dir.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=alias_dir / request_path.name,
            transcription_review_packet_json=(
                alias_dir / Path(chain["transcription_packet_path"]).name
            ),
            response_manifest_json=alias_dir / response_path.name,
            output_json=tmp_path / "ancestor-symlink-output.json",
            generated_at=_VALIDATED_AT,
        )


def test_file_entry_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    chain, _, request_path, response_path = _write_inputs(tmp_path)
    real_output_dir = tmp_path / "real-output"
    real_output_dir.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_output_dir, target_is_directory=True)
    redirected_output = real_output_dir / "must-not-exist.json"

    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_evidence_response_from_files(
            request_artifact_json=request_path,
            transcription_review_packet_json=chain["transcription_packet_path"],
            response_manifest_json=response_path,
            output_json=output_alias / redirected_output.name,
            generated_at=_VALIDATED_AT,
        )
    assert not redirected_output.exists()


def test_cli_emits_redacted_success_and_failure_summaries(tmp_path: Path) -> None:
    chain, _, request_path, response_path = _write_inputs(tmp_path)
    output_path = tmp_path / "cli-output.json"
    stdout = StringIO()
    rc = main(
        [
            "--request-artifact",
            str(request_path),
            "--transcription-review-packet",
            str(chain["transcription_packet_path"]),
            "--response-manifest",
            str(response_path),
            "--output",
            str(output_path),
        ],
        stdout=stdout,
    )
    assert rc == 0
    summary = json.loads(stdout.getvalue())
    assert summary["status"] == "ready_for_human_material_identity_review"
    assert summary["identity_group_count"] == 7
    assert "TDBA" not in stdout.getvalue()
    assert str(tmp_path) not in stdout.getvalue()

    failure = StringIO()
    rc = main(
        [
            "--request-artifact",
            str(request_path),
            "--transcription-review-packet",
            str(chain["transcription_packet_path"]),
            "--response-manifest",
            str(response_path),
            "--output",
            str(output_path),
        ],
        stdout=failure,
    )
    assert rc == 2
    failure_summary = json.loads(failure.getvalue())
    assert failure_summary == {
        "error_code": "supplementary_material_identity_evidence_response_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in failure.getvalue()
