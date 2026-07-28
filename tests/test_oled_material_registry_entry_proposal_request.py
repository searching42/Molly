from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_material_registry_entry_proposal_request as proposal_runner
from ai4s_agent.domains import (
    oled_material_registry_entry_proposal_request as proposal_domain,
)
from ai4s_agent.domains.oled_material_registry_entry_proposal_request import (
    OledMaterialRegistryEntryProposalBatchFindingKind,
    OledMaterialRegistryEntryProposalRequestArtifact,
    OledMaterialRegistryEntryProposalRequestStatus,
    OledMaterialRegistryEntryReviewDecision,
    build_oled_material_registry_entry_proposal_request_artifact,
    oled_material_registry_entry_proposal_request_artifact_digest,
    oled_material_registry_entry_proposal_request_item_digest,
    oled_material_registry_entry_review_contract_digest,
    render_oled_material_registry_entry_proposal_request_markdown,
    validate_oled_material_registry_entry_proposal_request_inputs,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    build_oled_material_registry_entry,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityStructureEncodingKind,
    _rdkit_chemistry_observation,
)
from ai4s_agent.oled_material_registry_entry_proposal_request import (
    build_oled_material_registry_entry_proposal_request_from_files,
    render_oled_material_registry_entry_proposal_request_from_files,
)
from ai4s_agent.oled_material_registry_resolution_request import (
    build_oled_material_registry_resolution_request_from_files,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    build_oled_supplementary_material_identity_adjudication_from_files,
)
from tests.test_oled_material_registry_adjudication import (
    _adjudicate,
    _exact_match_request,
    _multi_candidate_no_match_request,
    _no_match_request,
    _request_with_entries,
)
from tests.test_oled_material_registry_resolution_request import (
    _REQUEST_AT,
    _snapshot,
    _write_snapshot,
)
from tests.test_oled_supplementary_material_identity_evidence_response import (
    _candidate_result,
)
from tests.test_oled_supplementary_material_identity_review import (
    _adjudication_kwargs,
    _build_packet,
    _write_decisions,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_PROPOSAL_AT = "2026-07-14T00:25:00+08:00"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MATERIAL_ID_RE = re.compile(r"material:[0-9a-f]{32}")


def _proposal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    request_factory: Any = _no_match_request,
    decision: str = "propose_new_entity",
    selected_material_id: str = "",
) -> tuple[Any, Any, Any, Path, Path]:
    request, request_path = request_factory(tmp_path, monkeypatch)
    adjudication, _, adjudication_path = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision=decision,
        selected_material_id=selected_material_id,
    )
    artifact = build_oled_material_registry_entry_proposal_request_artifact(
        resolution_request=request,
        resolution_request_sha256=_sha256_file(request_path),
        registry_adjudication=adjudication,
        registry_adjudication_sha256=_sha256_file(adjudication_path),
        generated_at=_PROPOSAL_AT,
    )
    return artifact, request, adjudication, request_path, adjudication_path


def _unique_candidate_result(group: Any) -> dict[str, Any]:
    structures = (
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "c1ccccc1",
        "c1ccncc1",
    )
    structure_text = structures[group.row_index]
    observation = _rdkit_chemistry_observation(
        encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
        structure_text=structure_text,
    )
    result = _candidate_result(group, structure_text=structure_text)
    result["structure_candidate"].update(
        {
            "canonical_isomeric_smiles_candidate": observation[
                "canonical_isomeric_smiles"
            ],
            "inchikey_candidate": observation["inchikey"],
        }
    )
    return result


def _seven_unique_no_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    _, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=_unique_candidate_result,
    )

    def accept_all_candidates(payload: dict[str, Any]) -> None:
        for entry in payload["decisions"]:
            entry["decision"] = "accept_structure_candidate"
            entry["candidate_source_match"] = "matches_source"
            entry["review_note"] = (
                "Human accepted this synthetic source-bound structure candidate."
            )
            for assessment in entry["anchor_assessments"]:
                assessment["assessment"] = "supports_claim"
                assessment["review_note"] = ""

    decision_path, _ = _write_decisions(
        context,
        mutate=accept_all_candidates,
        filename="seven-unique-material-identity-decisions.json",
    )
    source_path = context["review_dir"] / "seven-unique-adjudication.json"
    source = build_oled_supplementary_material_identity_adjudication_from_files(
        **_adjudication_kwargs(
            context,
            decision_path=decision_path,
            output_path=source_path,
        )
    )
    assert source.later_registry_review_eligible_group_count == 7
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    request_path = tmp_path / "seven-unique-registry-resolution-request.json"
    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=request_path,
        generated_at=_REQUEST_AT,
    )
    return request, request_path


def _rehash_artifact(
    artifact: OledMaterialRegistryEntryProposalRequestArtifact,
    **updates: Any,
) -> OledMaterialRegistryEntryProposalRequestArtifact:
    tampered = artifact.model_copy(update=updates, deep=True)
    return tampered.model_copy(
        update={
            "proposal_request_artifact_digest": (
                oled_material_registry_entry_proposal_request_artifact_digest(
                    tampered
                )
            )
        },
        deep=True,
    )


def test_single_new_entity_proposal_is_exact_bound_and_request_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, request, adjudication, request_path, adjudication_path = (
        _proposal_artifact(tmp_path, monkeypatch)
    )

    assert artifact.status == (
        OledMaterialRegistryEntryProposalRequestStatus
        .READY_FOR_HUMAN_REGISTRY_ENTRY_REVIEW
    )
    assert artifact.entry_review_item_count == 1
    assert artifact.entry_review_cell_count == 5
    assert artifact.source_resolution_item_count == 1
    assert artifact.source_adjudicated_item_count == 1
    assert artifact.batch_conflict_finding_count == 0
    assert artifact.device_only_cell_count == 0
    assert artifact.resolution_request_sha256 == _sha256_file(request_path)
    assert artifact.registry_adjudication_sha256 == _sha256_file(adjudication_path)
    assert artifact.resolution_request_digest == request.request_artifact_digest
    assert artifact.registry_adjudication_digest == (
        adjudication.adjudication_artifact_digest
    )
    assert not artifact.standalone_input_bytes_revalidation_supported
    assert _SHA256_RE.fullmatch(artifact.proposal_request_artifact_digest)

    item = artifact.entry_review_items[0]
    proposal = item.source_adjudicated_item.new_entity_proposal
    assert proposal is not None
    assert _MATERIAL_ID_RE.fullmatch(item.proposed_material_id)
    assert item.proposed_canonical_name == proposal.reported_subject_literal
    assert item.proposed_aliases == []
    assert item.identity_dependent_cell_count == 5
    assert item.source_graph_human_accepted
    assert item.local_snapshot_match_replayed
    assert not item.material_id_reserved
    assert not item.material_id_assigned
    assert not item.canonical_name_approved
    assert not item.registry_entry_created
    assert not item.registry_written

    contract = artifact.review_contract
    assert contract.allowed_decisions == list(OledMaterialRegistryEntryReviewDecision)
    assert contract.single_molecular_entity_policy_only
    assert contract.aliases_require_exact_source_support
    assert not contract.reported_subject_auto_approved_as_canonical_name
    assert not contract.global_chemical_novelty_assessed
    assert not contract.registry_write_requested
    assert OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
        artifact.model_dump(mode="json")
    ) == artifact


def test_seven_unique_proposals_cover_35_cells_without_batch_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, request, adjudication, request_path, adjudication_path = (
        _proposal_artifact(
            tmp_path,
            monkeypatch,
            request_factory=_seven_unique_no_match_request,
        )
    )

    assert artifact.status == (
        OledMaterialRegistryEntryProposalRequestStatus
        .READY_FOR_HUMAN_REGISTRY_ENTRY_REVIEW
    )
    assert artifact.entry_review_item_count == 7
    assert artifact.entry_review_cell_count == 35
    assert artifact.batch_conflict_findings == []
    assert artifact.batch_conflict_finding_count == 0
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert len({item.proposed_material_id for item in artifact.entry_review_items}) == 7
    assert len(
        {item.proposed_canonical_name for item in artifact.entry_review_items}
    ) == 7
    assert [
        item.source_adjudicated_item.request_item.adjudicated_group.review_item
        .validated_result.bound_identity_group.row_index
        for item in artifact.entry_review_items
    ] == list(range(7))
    assert all(
        item.identity_dependent_cell_count == 5
        for item in artifact.entry_review_items
    )

    rebuilt = build_oled_material_registry_entry_proposal_request_artifact(
        resolution_request=request,
        resolution_request_sha256=_sha256_file(request_path),
        registry_adjudication=adjudication,
        registry_adjudication_sha256=_sha256_file(adjudication_path),
        generated_at="2026-07-14T00:26:00+08:00",
    )
    assert [item.proposed_material_id for item in rebuilt.entry_review_items] == [
        item.proposed_material_id for item in artifact.entry_review_items
    ]
    assert [item.entry_review_item_digest for item in rebuilt.entry_review_items] == [
        item.entry_review_item_digest for item in artifact.entry_review_items
    ]


def test_non_new_entity_decision_yields_an_empty_review_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(
        tmp_path,
        monkeypatch,
        request_factory=_exact_match_request,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )

    assert artifact.status == (
        OledMaterialRegistryEntryProposalRequestStatus.NO_NEW_ENTITY_PROPOSALS
    )
    assert artifact.entry_review_items == []
    assert artifact.entry_review_item_count == 0
    assert artifact.entry_review_cell_count == 0
    assert artifact.existing_entity_mapping_excluded_count == 1
    assert artifact.unresolved_excluded_count == 0
    assert artifact.conflict_deferred_excluded_count == 0
    assert artifact.batch_conflict_findings == []


def test_duplicate_structure_batch_is_explicit_and_never_auto_merged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(
        tmp_path,
        monkeypatch,
        request_factory=_multi_candidate_no_match_request,
    )

    assert artifact.status == (
        OledMaterialRegistryEntryProposalRequestStatus
        .BATCH_CONFLICTS_REQUIRE_HUMAN_REVIEW
    )
    assert artifact.entry_review_item_count == 7
    assert artifact.entry_review_cell_count == 35
    assert artifact.batch_conflict_finding_count == 2
    assert {
        finding.finding_kind for finding in artifact.batch_conflict_findings
    } == {
        OledMaterialRegistryEntryProposalBatchFindingKind
        .DUPLICATE_CANONICAL_SMILES,
        OledMaterialRegistryEntryProposalBatchFindingKind.DUPLICATE_INCHIKEY,
    }
    assert all(
        len(finding.affected_entry_review_item_ids) == 7
        and finding.blocks_automatic_approval
        and not finding.automatic_merge_performed
        for finding in artifact.batch_conflict_findings
    )
    assert len({item.proposed_material_id for item in artifact.entry_review_items}) == 7


def test_proposed_material_id_already_occupied_in_snapshot_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    seed, _, _, _, _ = _proposal_artifact(seed_root, monkeypatch)
    occupied_material_id = seed.entry_review_items[0].proposed_material_id

    occupied_root = tmp_path / "occupied"
    occupied_root.mkdir()

    def unrelated_entry(_group: Any, _candidate: Any) -> list[Any]:
        return [
            build_oled_material_registry_entry(
                material_id=occupied_material_id,
                canonical_name="unrelated-existing-entry",
                canonical_isomeric_smiles="C",
            )
        ]

    request, request_path = _request_with_entries(
        occupied_root,
        monkeypatch,
        unrelated_entry,
    )
    adjudication, _, adjudication_path = _adjudicate(
        occupied_root,
        request,
        request_path,
        decision="propose_new_entity",
    )

    with pytest.raises(ValueError, match="already occupied"):
        build_oled_material_registry_entry_proposal_request_artifact(
            resolution_request=request,
            resolution_request_sha256=_sha256_file(request_path),
            registry_adjudication=adjudication,
            registry_adjudication_sha256=_sha256_file(adjudication_path),
            generated_at=_PROPOSAL_AT,
        )


def test_entry_review_source_order_key_exactly_replays_pr_n_group_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)
    item = artifact.entry_review_items[0]
    bound = (
        item.source_adjudicated_item.request_item.adjudicated_group.review_item
        .validated_result.bound_identity_group
    )

    assert proposal_domain._entry_review_item_source_sort_key(item) == (
        bound.scope_id,
        bound.table_id,
        bound.row_index,
        bound.identity_group_id,
    )


def test_rehashed_material_id_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)
    item = artifact.entry_review_items[0].model_copy(
        update={"proposed_material_id": "material:" + "f" * 32},
        deep=True,
    )
    item = item.model_copy(
        update={
            "entry_review_item_digest": (
                oled_material_registry_entry_proposal_request_item_digest(item)
            )
        },
        deep=True,
    )
    tampered = _rehash_artifact(artifact, entry_review_items=[item])

    with pytest.raises(ValidationError, match="opaque ID proposal changed"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_standalone_contract_does_not_claim_pr_o_file_byte_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)
    alternate_sha = "sha256:" + "f" * 64
    assert artifact.registry_adjudication_sha256 != alternate_sha

    tampered = _rehash_artifact(
        artifact,
        registry_adjudication_sha256=alternate_sha,
    )
    validated = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
        tampered.model_dump(mode="json")
    )
    assert validated.registry_adjudication_sha256 == alternate_sha
    assert not validated.standalone_input_bytes_revalidation_supported

    false_claim = _rehash_artifact(
        artifact,
        standalone_input_bytes_revalidation_supported=True,
    )
    with pytest.raises(ValidationError, match="crossed its boundary"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            false_claim.model_dump(mode="json")
        )


def test_rehashed_review_contract_question_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)
    contract = artifact.review_contract.model_copy(
        update={"entity_scope_question": "Approve this entry without review?"},
        deep=True,
    )
    contract = contract.model_copy(
        update={
            "contract_digest": oled_material_registry_entry_review_contract_digest(
                contract
            )
        },
        deep=True,
    )
    tampered = _rehash_artifact(artifact, review_contract=contract)

    with pytest.raises(ValidationError, match="entity_scope_question changed"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_rehashed_batch_finding_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(
        tmp_path,
        monkeypatch,
        request_factory=_multi_candidate_no_match_request,
    )
    finding = artifact.batch_conflict_findings[0].model_copy(
        update={"key_digest": "sha256:" + "f" * 64},
        deep=True,
    )
    finding = finding.model_copy(
        update={"finding_digest": proposal_domain._batch_finding_digest(finding)},
        deep=True,
    )
    findings = sorted(
        [finding, *artifact.batch_conflict_findings[1:]],
        key=lambda item: item.finding_digest,
    )
    tampered = _rehash_artifact(artifact, batch_conflict_findings=findings)

    with pytest.raises(ValidationError, match="batch conflict derivation changed"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_rehashed_count_and_boundary_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)

    changed_count = _rehash_artifact(artifact, entry_review_cell_count=6)
    with pytest.raises(ValidationError, match="entry_review_cell_count mismatch"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            changed_count.model_dump(mode="json")
        )

    crossed_boundary = _rehash_artifact(artifact, registry_written=True)
    with pytest.raises(ValidationError, match="crossed its boundary"):
        OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            crossed_boundary.model_dump(mode="json")
        )


def test_exact_pr_n_binding_and_pr_o_to_request_causality_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request, adjudication, request_path, adjudication_path = _proposal_artifact(
        tmp_path,
        monkeypatch,
    )
    del adjudication_path

    with pytest.raises(ValueError, match="exact supplied PR-N file"):
        validate_oled_material_registry_entry_proposal_request_inputs(
            resolution_request=request,
            resolution_request_sha256="sha256:" + "0" * 64,
            registry_adjudication=adjudication,
        )

    with pytest.raises(ValidationError, match="predates PR-O"):
        build_oled_material_registry_entry_proposal_request_artifact(
            resolution_request=request,
            resolution_request_sha256=_sha256_file(request_path),
            registry_adjudication=adjudication,
            registry_adjudication_sha256="sha256:" + "1" * 64,
            generated_at="2026-07-14T00:19:59+08:00",
        )


def test_markdown_exposes_review_contract_and_local_only_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _, _, _ = _proposal_artifact(tmp_path, monkeypatch)
    markdown = render_oled_material_registry_entry_proposal_request_markdown(
        artifact,
        artifact_sha256="sha256:" + "a" * 64,
    )
    item = artifact.entry_review_items[0]

    assert markdown.startswith("# OLED Local Material Registry Entry Review Request")
    assert "does not claim that any molecule or material is globally novel" in markdown
    assert artifact.review_contract.local_snapshot_only_notice in markdown
    assert artifact.review_contract.entity_scope_question in markdown
    assert artifact.review_contract.preferred_name_question in markdown
    assert artifact.review_contract.alias_question in markdown
    assert "approve_local_registry_entry_candidate" in markdown
    assert "the original JSON bytes are not embedded" in markdown
    assert "standalone_input_bytes_revalidation_supported=false" in markdown
    assert "## E01:" in markdown
    assert item.proposed_material_id in markdown
    assert item.proposed_canonical_name in markdown
    assert "proposed aliases: `[]`" in markdown
    assert (
        "No Registry, observation, Gold, dataset, or training write occurs here"
        in markdown
    )


def test_markdown_for_empty_and_conflicted_requests_is_unambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty, _, _, _, _ = _proposal_artifact(
        tmp_path / "empty",
        monkeypatch,
        request_factory=_exact_match_request,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )
    conflict_root = tmp_path / "conflict"
    conflict_root.mkdir()
    conflicted, _, _, _, _ = _proposal_artifact(
        conflict_root,
        monkeypatch,
        request_factory=_multi_candidate_no_match_request,
    )

    empty_markdown = render_oled_material_registry_entry_proposal_request_markdown(
        empty,
        artifact_sha256="sha256:" + "b" * 64,
    )
    conflict_markdown = render_oled_material_registry_entry_proposal_request_markdown(
        conflicted,
        artifact_sha256="sha256:" + "c" * 64,
    )
    assert "No PR-O new-entity proposal is eligible" in empty_markdown
    assert "## Batch conflicts" in conflict_markdown
    assert "duplicate_canonical_smiles" in conflict_markdown
    assert "duplicate_inchikey" in conflict_markdown


def test_file_entry_builds_exact_byte_bound_artifact_and_renders_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, request_path, adjudication_path = _proposal_artifact(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "entry-proposal-request.json"
    artifact = build_oled_material_registry_entry_proposal_request_from_files(
        resolution_request_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=output_path,
        generated_at=_PROPOSAL_AT,
    )

    assert artifact.resolution_request_sha256 == _sha256_file(request_path)
    assert artifact.registry_adjudication_sha256 == _sha256_file(adjudication_path)
    assert OledMaterialRegistryEntryProposalRequestArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact

    markdown_path = tmp_path / "entry-proposal-review.md"
    rendered = render_oled_material_registry_entry_proposal_request_from_files(
        proposal_request_json=output_path,
        output_markdown=markdown_path,
    )
    assert rendered == artifact
    markdown = markdown_path.read_text(encoding="utf-8")
    assert _sha256_file(output_path) in markdown
    assert artifact.entry_review_items[0].proposed_material_id in markdown


def test_file_entry_rejects_changed_pr_n_bytes_without_publishing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, request_path, adjudication_path = _proposal_artifact(
        tmp_path,
        monkeypatch,
    )
    request_path.write_text(
        request_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="exact supplied PR-N file"):
        build_oled_material_registry_entry_proposal_request_from_files(
            resolution_request_json=request_path,
            registry_adjudication_json=adjudication_path,
            output_json=output_path,
            generated_at=_PROPOSAL_AT,
        )

    assert not output_path.exists()


def test_file_entry_never_overwrites_an_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, request_path, adjudication_path = _proposal_artifact(
        tmp_path,
        monkeypatch,
    )
    original_request = request_path.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_material_registry_entry_proposal_request_from_files(
            resolution_request_json=request_path,
            registry_adjudication_json=adjudication_path,
            output_json=request_path,
            generated_at=_PROPOSAL_AT,
        )

    assert request_path.read_bytes() == original_request


def test_domain_and_file_entry_are_generic_not_paper016_specific() -> None:
    source = inspect.getsource(proposal_domain) + inspect.getsource(proposal_runner)

    assert "paper016" not in source
    assert proposal_domain._entry_review_item_id("generic-resolution-item") == (
        "material-registry-entry-review:generic-resolution-item"
    )
    allocation = proposal_domain._material_id_allocation_digest(
        registry_id="generic-material-registry",
        proposal_digest="sha256:" + "d" * 64,
    )
    assert _SHA256_RE.fullmatch(allocation)
    assert _MATERIAL_ID_RE.fullmatch(proposal_domain._proposed_material_id(allocation))
