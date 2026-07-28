from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains import (
    oled_supplementary_material_identity_candidate_request as identity_domain,
)
from ai4s_agent.domains import (
    oled_supplementary_source_transcription_review as transcription_domain,
)
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    build_oled_supplementary_material_identity_candidate_request_artifact,
    render_oled_supplementary_material_identity_candidate_request_markdown,
)
from ai4s_agent.oled_supplementary_material_identity_candidate_request import (
    build_oled_supplementary_material_identity_candidate_request_from_files,
    main,
    render_oled_supplementary_material_identity_candidate_request_from_files,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file
from tests.test_oled_supplementary_source_transcription_review import (
    _adjudicate_domain,
    _build_domain_packet,
    _transcription_decision_payload,
)


_IDENTITY_REQUEST_GENERATED_AT = "2026-07-13T23:10:00+08:00"


def _build_pr_j_chain(tmp_path: Path) -> dict[str, Any]:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path)
    decision_payload = _transcription_decision_payload(packet, packet_path)
    decision_path = tmp_path / "source-transcription-decisions.json"
    write_json(decision_path, decision_payload)
    adjudication = _adjudicate_domain(
        chain,
        packet,
        packet_path,
        decision_payload,
    )
    adjudication_path = tmp_path / "source-transcription-adjudication.json"
    write_json(adjudication_path, adjudication.model_dump(mode="json"))
    chain.update(
        {
            "transcription_packet": packet,
            "transcription_packet_path": packet_path,
            "transcription_decision_path": decision_path,
            "transcription_adjudication": adjudication,
            "transcription_adjudication_path": adjudication_path,
        }
    )
    return chain


def _file_kwargs(
    chain: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    return {
        "request_artifact_json": chain["request_path"],
        "response_manifest_json": chain["response_path"],
        "response_artifact_json": chain["response_artifact_path"],
        "semantic_review_packet_json": chain["semantic_packet_path"],
        "semantic_decision_manifest_json": chain["semantic_decision_path"],
        "semantic_adjudication_json": chain["semantic_adjudication_path"],
        "transcription_review_packet_json": chain["transcription_packet_path"],
        "transcription_decision_manifest_json": chain[
            "transcription_decision_path"
        ],
        "transcription_adjudication_json": chain[
            "transcription_adjudication_path"
        ],
        "output_json": output_path,
        "generated_at": _IDENTITY_REQUEST_GENERATED_AT,
    }


def _build_request(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    Path,
]:
    chain = _build_pr_j_chain(tmp_path)
    output_path = tmp_path / "material-identity-candidate-request.json"
    artifact = (
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **_file_kwargs(chain, output_path)
        )
    )
    return chain, artifact, output_path


def _recompute_group(payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(payload)
    payload["identity_group_id"] = "material-identity-placeholder"
    payload["identity_group_digest"] = "sha256:" + "0" * 64
    payload["subject_header_binding"] = (
        transcription_domain.OledSupplementarySourceHeaderReviewBinding.model_validate(
            payload["subject_header_binding"]
        )
    )
    payload["identity_dependent_cells"] = [
        identity_domain.OledSupplementaryMaterialIdentityDependentCell.model_validate(
            cell
        )
        for cell in payload["identity_dependent_cells"]
    ]
    provisional = identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_construct(
        **payload
    )
    payload["identity_group_id"] = identity_domain._material_identity_group_id(
        provisional
    )
    provisional = identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_construct(
        **payload
    )
    payload["identity_group_digest"] = (
        identity_domain._material_identity_group_digest(provisional)
    )
    identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
        payload
    )
    return payload


def _recompute_request(payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(payload)
    payload["material_identity_request_digest"] = "sha256:" + "0" * 64
    payload["status"] = (
        identity_domain.OledSupplementaryMaterialIdentityCandidateRequestStatus(
            payload["status"]
        )
    )
    payload["identity_groups"] = [
        identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
            group
        )
        for group in payload["identity_groups"]
    ]
    provisional = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_construct(
        **payload
    )
    payload["material_identity_request_digest"] = (
        identity_domain._material_identity_request_digest(provisional)
    )
    return payload


def _forge_transcription_eligible_roster(
    chain: dict[str, Any],
    *,
    replacement_digest: str,
) -> None:
    payload = chain["transcription_adjudication"].model_dump(mode="json")
    table = payload["adjudicated_tables"][0]
    old_digest = table["later_identity_review_eligible_source_cell_digests"][0]
    assert replacement_digest in table["source_cell_digests"]
    assert replacement_digest not in table[
        "later_identity_review_eligible_source_cell_digests"
    ]
    for field_name in (
        "upstream_later_eligible_source_cell_digests",
        "later_identity_review_eligible_source_cell_digests",
    ):
        roster = table[field_name]
        roster[roster.index(old_digest)] = replacement_digest
        roster.sort()
    payload["adjudication_artifact_digest"] = "sha256:" + "0" * 64
    payload["status"] = (
        transcription_domain.OledSupplementarySourceTranscriptionAdjudicationStatus(
            payload["status"]
        )
    )
    payload["adjudicated_tables"] = [
        transcription_domain.OledSupplementaryAdjudicatedSourceTranscription.model_validate(
            adjudicated_table
        )
        for adjudicated_table in payload["adjudicated_tables"]
    ]
    provisional = transcription_domain.OledSupplementarySourceTranscriptionAdjudicationArtifact.model_construct(
        **payload
    )
    payload["adjudication_artifact_digest"] = (
        transcription_domain._source_transcription_adjudication_digest(provisional)
    )
    forged = transcription_domain.OledSupplementarySourceTranscriptionAdjudicationArtifact.model_validate(
        payload
    )
    write_json(
        chain["transcription_adjudication_path"],
        forged.model_dump(mode="json"),
    )


def test_paper016_builds_seven_exact_row_groups_without_resolving_identity(
    tmp_path: Path,
) -> None:
    chain, artifact, output_path = _build_request(tmp_path)

    assert artifact.status.value == "ready_for_material_identity_evidence_proposal"
    assert artifact.source_count == artifact.scope_count == 1
    assert artifact.accepted_transcription_scope_count == 1
    assert artifact.identity_group_count == 7
    assert artifact.identity_dependent_cell_count == 35
    assert artifact.bounded_transcription_validated_cell_count == 49
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    assert [group.reported_subject_text for group in artifact.identity_groups] == [
        "TDBA",
        "TDBA-Ph",
        "mTDBA-Ph",
        "mTDBA-2Ph",
        "TDBA-Si",
        "mTDBA-Si",
        "mTDBA-2Si",
    ]
    assert [group.row_index for group in artifact.identity_groups] == list(range(7))
    assert all(group.identity_dependent_cell_count == 5 for group in artifact.identity_groups)
    assert all(
        [cell.column_index for cell in group.identity_dependent_cells]
        == [1, 2, 4, 5, 6]
        for group in artifact.identity_groups
    )
    assert all(
        group.subject_header_binding.parser_key == "column_1"
        and group.subject_header_binding.source_visible_header_candidate == ""
        and group.subject_header_binding.binding_kind.value
        == "parser_placeholder_candidate_for_blank_header"
        for group in artifact.identity_groups
    )
    all_digests = [
        digest
        for group in artifact.identity_groups
        for digest in group.identity_dependent_source_cell_digests
    ]
    assert len(all_digests) == len(set(all_digests)) == 35
    ontology_digests = set(
        chain["transcription_packet"].review_items[
            0
        ].upstream_ontology_review_pending_source_cell_digests
    )
    assert not ontology_digests.intersection(all_digests)
    assert artifact.transcription_adjudication_artifact_sha256 == _sha256_file(
        chain["transcription_adjudication_path"]
    )
    assert output_path.exists()
    assert OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_request_keeps_every_resolution_and_materialization_boundary_closed(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)

    assert artifact.request_only
    assert artifact.offline_only
    assert artifact.upstream_chain_replayed
    assert artifact.source_transcription_adjudication_replayed
    assert artifact.strict_eligible_cell_intersection_validated
    assert artifact.strict_row_partition_validated
    assert artifact.paper_local_row_scope_only
    assert artifact.reported_subject_literals_preserved
    assert artifact.bounded_source_transcription_accepted
    assert artifact.human_identity_review_required
    assert artifact.source_pdf_remains_authoritative
    assert artifact.material_identity_evidence_proposal_requested
    forbidden_true = (
        "response_received",
        "source_pdf_read",
        "raw_parsed_document_read",
        "source_structure_evidence_included",
        "identity_evidence_validated",
        "material_identity_resolved",
        "canonical_smiles_assigned",
        "inchikey_assigned",
        "cross_paper_identity_merge",
        "automatic_candidate_merge",
        "table_exhaustiveness_validated",
        "scientific_content_validated",
        "physical_semantics_validated",
        "schema_candidates_created",
        "registry_written",
        "reviewed_evidence_staging",
        "direct_admission_eligible",
        "training_eligible",
        "device_only_admitted",
        "gold_records_created",
        "dataset_written",
        "network_accessed",
        "external_service_called",
        "llm_called",
        "mineru_called",
    )
    assert all(not getattr(artifact, field_name) for field_name in forbidden_true)
    assert all(
        group.identity_evidence_required
        and group.human_identity_review_required
        and not group.source_structure_evidence_included
        and not group.material_identity_resolved
        and not group.canonical_smiles_assigned
        and not group.inchikey_assigned
        for group in artifact.identity_groups
    )
    serialized = json.dumps(artifact.model_dump(mode="json"), sort_keys=True)
    assert '"canonical_smiles"' not in serialized
    assert '"inchikey"' not in serialized


def test_same_reported_subject_in_two_rows_still_has_distinct_row_keys(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    first = artifact.identity_groups[0]
    second_payload = artifact.identity_groups[1].model_dump(mode="json")
    second_payload["reported_subject_text"] = first.reported_subject_text
    second = identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
        _recompute_group(second_payload)
    )

    assert first.reported_subject_text == second.reported_subject_text
    assert first.row_index != second.row_index
    assert first.identity_group_id != second.identity_group_id


def test_rejects_self_consistent_unknown_pr_j_eligible_cell_on_replay(
    tmp_path: Path,
) -> None:
    chain = _build_pr_j_chain(tmp_path)
    ontology_digest = chain[
        "transcription_packet"
    ].review_items[0].upstream_ontology_review_pending_source_cell_digests[0]
    _forge_transcription_eligible_roster(
        chain,
        replacement_digest=ontology_digest,
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="replay mismatch"):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **_file_kwargs(chain, output_path)
        )

    assert not output_path.exists()


def test_rejects_stale_semantic_adjudication_bytes_without_output(
    tmp_path: Path,
) -> None:
    chain = _build_pr_j_chain(tmp_path)
    semantic_path = chain["semantic_adjudication_path"]
    semantic_path.write_text(
        semantic_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="binding mismatch"):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **_file_kwargs(chain, output_path)
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("material_identity_resolved", True),
        ("canonical_smiles_assigned", True),
        ("registry_written", True),
        ("direct_admission_eligible", True),
        ("device_only_admitted", True),
        ("gold_records_created", True),
        ("dataset_written", True),
    ),
)
def test_standalone_model_rejects_downstream_boundary_tampering(
    tmp_path: Path,
    field_name: str,
    value: bool,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    payload = artifact.model_dump(mode="json")
    payload[field_name] = value
    payload = _recompute_request(payload)

    with pytest.raises(ValidationError, match="crossed a downstream boundary"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )


def test_standalone_model_rejects_duplicate_or_moved_group_cells(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    duplicate = artifact.identity_groups[0].model_dump(mode="json")
    duplicate["identity_dependent_cells"][1]["column_index"] = duplicate[
        "identity_dependent_cells"
    ][0]["column_index"]
    duplicate["identity_dependent_cells"][1]["column_name"] = duplicate[
        "identity_dependent_cells"
    ][0]["column_name"]
    duplicate["identity_dependent_cells"].sort(
        key=lambda cell: (cell["column_index"], cell["source_cell_digest"])
    )

    with pytest.raises(ValidationError, match="source coordinates repeat"):
        identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
            duplicate
        )

    subject = artifact.identity_groups[0].model_dump(mode="json")
    subject["identity_dependent_cells"][0]["column_index"] = subject[
        "subject_column_index"
    ]
    with pytest.raises(ValidationError, match="subject column"):
        identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
            subject
        )

    moved = artifact.identity_groups[0].model_dump(mode="json")
    moved["identity_dependent_cells"][0]["row_index"] += 1
    with pytest.raises(ValidationError, match="moved to another row"):
        identity_domain.OledSupplementaryMaterialIdentityCandidateGroup.model_validate(
            moved
        )


def test_standalone_model_rejects_duplicate_logical_rows_and_provenance_conflicts(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    payload = artifact.model_dump(mode="json")
    duplicate_row = deepcopy(payload["identity_groups"][1])
    duplicate_row["row_index"] = payload["identity_groups"][0]["row_index"]
    duplicate_row["reported_subject_text"] = "same-row-second-claim"
    for cell in duplicate_row["identity_dependent_cells"]:
        cell["row_index"] = duplicate_row["row_index"]
    payload["identity_groups"][1] = _recompute_group(duplicate_row)
    payload["identity_groups"].sort(
        key=lambda group: (
            group["scope_id"],
            group["table_id"],
            group["row_index"],
            group["identity_group_id"],
        )
    )
    payload = _recompute_request(payload)
    with pytest.raises(ValidationError, match="logical rows must be unique"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )

    payload = artifact.model_dump(mode="json")
    conflicting_pdf = deepcopy(payload["identity_groups"][1])
    conflicting_pdf["source_pdf_sha256"] = "sha256:" + "f" * 64
    payload["identity_groups"][1] = _recompute_group(conflicting_pdf)
    payload = _recompute_request(payload)
    with pytest.raises(ValidationError, match="inconsistent source provenance"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )

    payload = artifact.model_dump(mode="json")
    conflicting_scope = deepcopy(payload["identity_groups"][1])
    conflicting_scope["pdf_page_number_one_based"] += 1
    payload["identity_groups"][1] = _recompute_group(conflicting_scope)
    payload = _recompute_request(payload)
    with pytest.raises(ValidationError, match="scope provenance is inconsistent"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )


def test_standalone_model_rejects_cross_scope_and_column_provenance_conflicts(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    payload = artifact.model_dump(mode="json")
    second_scope = deepcopy(payload["identity_groups"][-1])
    second_scope["scope_id"] = "supplementary-scoped-request:second-scope"
    second_scope["parsed_document_sha256"] = "sha256:" + "d" * 64
    second_scope["source_transcription_review_item_id"] = (
        "supplementary-source-transcription:second-item"
    )
    second_scope["source_transcription_review_item_digest"] = "sha256:" + "c" * 64
    payload["identity_groups"][-1] = _recompute_group(second_scope)
    payload["identity_groups"].sort(
        key=lambda group: (
            group["scope_id"],
            group["table_id"],
            group["row_index"],
            group["identity_group_id"],
        )
    )
    payload["scope_count"] = 2
    payload["accepted_transcription_scope_count"] = 2
    payload = _recompute_request(payload)
    with pytest.raises(ValidationError, match="inconsistent source provenance"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )

    payload = artifact.model_dump(mode="json")
    reused_item = deepcopy(payload["identity_groups"][-1])
    reused_item["scope_id"] = "supplementary-scoped-request:second-scope"
    payload["identity_groups"][-1] = _recompute_group(reused_item)
    payload["identity_groups"].sort(
        key=lambda group: (
            group["scope_id"],
            group["table_id"],
            group["row_index"],
            group["identity_group_id"],
        )
    )
    payload["scope_count"] = 2
    payload["accepted_transcription_scope_count"] = 2
    payload = _recompute_request(payload)
    with pytest.raises(
        ValidationError,
        match="transcription review item provenance is inconsistent",
    ):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )

    payload = artifact.model_dump(mode="json")
    conflicting_column = deepcopy(payload["identity_groups"][0])
    conflicting_column["identity_dependent_cells"][0]["column_name"] = (
        "CONFLICTING HEADER"
    )
    payload["identity_groups"][0] = _recompute_group(conflicting_column)
    payload = _recompute_request(payload)
    with pytest.raises(ValidationError, match="column provenance is inconsistent"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )


def test_standalone_model_rejects_impossible_bounded_cell_partition(
    tmp_path: Path,
) -> None:
    _, artifact, _ = _build_request(tmp_path)
    payload = artifact.model_dump(mode="json")
    payload["bounded_transcription_validated_cell_count"] = 35
    payload = _recompute_request(payload)

    with pytest.raises(ValidationError, match="partition exceeds"):
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            payload
        )


def test_markdown_distinguishes_blank_source_header_and_escapes_source_text(
    tmp_path: Path,
) -> None:
    _, artifact, output_path = _build_request(tmp_path)
    markdown = render_oled_supplementary_material_identity_candidate_request_markdown(
        artifact,
        request_artifact_sha256=_sha256_file(output_path),
    )

    assert "no explicit source header (parser key column_1)" in markdown
    assert "Identity groups: **7**" in markdown
    assert "Identity-dependent cells: **35**" in markdown
    assert "Ontology-pending cells excluded: **14**" in markdown
    assert "Structure evidence included: `false`" in markdown
    assert "Material identity resolved: `false`" in markdown

    payload = artifact.model_dump(mode="json")
    group_payload = payload["identity_groups"][0]
    group_payload["reported_subject_text"] = "<script>alert(1)</script>|alias`"
    payload["identity_groups"][0] = _recompute_group(group_payload)
    payload = _recompute_request(payload)
    escaped_artifact = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        payload
    )
    escaped = render_oled_supplementary_material_identity_candidate_request_markdown(
        escaped_artifact,
        request_artifact_sha256="sha256:" + "a" * 64,
    )
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&#124;" in escaped
    assert "&#96;" in escaped

    payload = artifact.model_dump(mode="json")
    bidi_group = payload["identity_groups"][0]
    bidi_group["reported_subject_text"] = "TDBA\u202Egpj.exe"
    payload["identity_groups"][0] = _recompute_group(bidi_group)
    payload = _recompute_request(payload)
    bidi_artifact = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        payload
    )
    bidi_markdown = render_oled_supplementary_material_identity_candidate_request_markdown(
        bidi_artifact,
        request_artifact_sha256="sha256:" + "b" * 64,
    )
    assert "\u202E" not in bidi_markdown
    assert "\\u202E" in bidi_markdown


def test_file_renderer_writes_fresh_validated_markdown(
    tmp_path: Path,
) -> None:
    _, artifact, request_path = _build_request(tmp_path)
    markdown_path = tmp_path / "material-identity-candidate-request.md"

    rendered = render_oled_supplementary_material_identity_candidate_request_from_files(
        request_artifact_json=request_path,
        output_markdown=markdown_path,
    )

    assert rendered == artifact
    assert "# Supplementary material-identity evidence request" in markdown_path.read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fresh"):
        render_oled_supplementary_material_identity_candidate_request_from_files(
            request_artifact_json=request_path,
            output_markdown=markdown_path,
        )


def test_file_entry_rejects_output_input_collision_without_changing_input(
    tmp_path: Path,
) -> None:
    chain = _build_pr_j_chain(tmp_path)
    protected_path = chain["transcription_adjudication_path"]
    before = protected_path.read_bytes()

    with pytest.raises(ValueError, match="overwrite an input"):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **_file_kwargs(chain, protected_path)
        )

    assert protected_path.read_bytes() == before


def test_file_entry_rejects_symlink_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    chain = _build_pr_j_chain(tmp_path)
    output_path = tmp_path / "must-not-exist.json"
    semantic_link = tmp_path / "semantic-adjudication-link.json"
    semantic_link.symlink_to(chain["semantic_adjudication_path"])
    kwargs = _file_kwargs(chain, output_path)
    kwargs["semantic_adjudication_json"] = semantic_link
    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **kwargs
        )
    assert not output_path.exists()

    duplicate = tmp_path / "duplicate-keys.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}\n')
    kwargs = _file_kwargs(chain, output_path)
    kwargs["semantic_decision_manifest_json"] = duplicate
    with pytest.raises(ValueError, match="duplicate keys"):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **kwargs
        )
    assert not output_path.exists()

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n')
    kwargs["semantic_decision_manifest_json"] = nonfinite
    with pytest.raises(ValueError, match="NaN"):
        build_oled_supplementary_material_identity_candidate_request_from_files(
            **kwargs
        )
    assert not output_path.exists()


def test_cli_reports_redacted_failure_and_success_summary(tmp_path: Path) -> None:
    chain = _build_pr_j_chain(tmp_path)
    output_path = tmp_path / "material-identity-candidate-request.json"
    args = _cli_build_args(chain, output_path)
    stream = StringIO()

    assert main(args, stdout=stream) == 0
    success = json.loads(stream.getvalue())
    assert success == {
        "device_only_cell_count": 0,
        "identity_dependent_cell_count": 35,
        "identity_group_count": 7,
        "paper_id": "paper016",
        "status": "ready_for_material_identity_evidence_proposal",
        "upstream_ontology_review_pending_cell_count": 14,
    }

    missing = tmp_path / "operator-secret-token=abc.json"
    failed_args = _cli_build_args(chain, tmp_path / "failure-output.json")
    missing_index = failed_args.index(str(chain["semantic_adjudication_path"]))
    failed_args[missing_index] = str(missing)
    stream = StringIO()
    assert main(failed_args, stdout=stream) == 2
    failure = stream.getvalue()
    assert "supplementary_material_identity_candidate_request_failed" in failure
    assert str(missing) not in failure
    assert "abc" not in failure
    assert not (tmp_path / "failure-output.json").exists()


def _cli_build_args(chain: dict[str, Any], output_path: Path) -> list[str]:
    return [
        "build",
        "--request-artifact",
        str(chain["request_path"]),
        "--response-manifest",
        str(chain["response_path"]),
        "--response-artifact",
        str(chain["response_artifact_path"]),
        "--semantic-review-packet",
        str(chain["semantic_packet_path"]),
        "--semantic-decision-manifest",
        str(chain["semantic_decision_path"]),
        "--semantic-adjudication",
        str(chain["semantic_adjudication_path"]),
        "--transcription-review-packet",
        str(chain["transcription_packet_path"]),
        "--transcription-decision-manifest",
        str(chain["transcription_decision_path"]),
        "--transcription-adjudication",
        str(chain["transcription_adjudication_path"]),
        "--output",
        str(output_path),
    ]
