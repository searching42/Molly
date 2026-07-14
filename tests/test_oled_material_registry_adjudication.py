from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent import oled_material_registry_adjudication as adjudication_runner
from ai4s_agent.domains.oled_material_registry_adjudication import (
    OledMaterialRegistryAdjudicationArtifact,
    OledMaterialRegistryDecisionManifest,
    oled_material_registry_adjudication_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    build_oled_material_registry_entry,
)
from ai4s_agent.oled_material_registry_adjudication import (
    build_oled_material_registry_adjudication_from_files,
    main,
    render_oled_material_registry_adjudication_review_from_files,
)
from ai4s_agent.oled_material_registry_resolution_request import (
    build_oled_material_registry_resolution_request_from_files,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    build_oled_supplementary_material_identity_adjudication_from_files,
)
from tests.test_oled_material_registry_resolution_request import (
    _REQUEST_AT,
    _accepted_candidate,
    _accepted_source,
    _snapshot,
    _write_snapshot,
)
from tests.test_oled_supplementary_material_identity_review import (
    _adjudication_kwargs,
    _build_packet,
    _write_decisions,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_REVIEWED_AT = "2026-07-14T00:15:00+08:00"
_ADJUDICATED_AT = "2026-07-14T00:20:00+08:00"


def _request_with_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_factory: Callable[[Any, Any], list[OledMaterialRegistryEntry]],
) -> tuple[Any, Path]:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    group, candidate = _accepted_candidate(source)
    entries = entry_factory(group, candidate)
    snapshot_path = _write_snapshot(tmp_path, _snapshot(entries))
    request_path = tmp_path / "material-registry-resolution-request.json"
    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=request_path,
        generated_at=_REQUEST_AT,
    )
    return request, request_path


def _no_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    return _request_with_entries(tmp_path, monkeypatch, lambda _group, _candidate: [])


def _exact_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    def entries(group: Any, candidate: Any) -> list[OledMaterialRegistryEntry]:
        reported = (
            group.review_item.validated_result.bound_identity_group.reported_subject_text
        )
        return [
            build_oled_material_registry_entry(
                material_id="material-0001",
                canonical_name=reported,
                aliases=["accepted-fixture-alias"],
                canonical_isomeric_smiles=(
                    candidate.canonical_isomeric_smiles_candidate
                ),
            )
        ]

    return _request_with_entries(tmp_path, monkeypatch, entries)


def _alias_only_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    def entries(group: Any, _candidate: Any) -> list[OledMaterialRegistryEntry]:
        reported = (
            group.review_item.validated_result.bound_identity_group.reported_subject_text
        )
        return [
            build_oled_material_registry_entry(
                material_id="material-alias-only",
                canonical_name=reported,
                canonical_isomeric_smiles="c1ccccc1",
            )
        ]

    return _request_with_entries(tmp_path, monkeypatch, entries)


def _duplicate_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    def entries(group: Any, candidate: Any) -> list[OledMaterialRegistryEntry]:
        reported = (
            group.review_item.validated_result.bound_identity_group.reported_subject_text
        )
        return [
            build_oled_material_registry_entry(
                material_id=material_id,
                canonical_name=reported if index == 0 else f"duplicate-{index}",
                aliases=[] if index == 0 else [reported],
                canonical_isomeric_smiles=(
                    candidate.canonical_isomeric_smiles_candidate
                ),
            )
            for index, material_id in enumerate(("material-0001", "material-0002"))
        ]

    return _request_with_entries(tmp_path, monkeypatch, entries)


def _decision_payload(
    request: Any,
    request_path: Path,
    *,
    decision: str,
    selected_material_id: str = "",
    conflict_reason: str = "none",
) -> dict[str, Any]:
    entries = []
    for item in request.resolution_items:
        candidate_ids = sorted(
            {
                *item.canonical_smiles_candidate_material_ids,
                *item.inchikey_candidate_material_ids,
            }
        )
        entries.append(
            {
                "resolution_item_id": item.resolution_item_id,
                "resolution_item_digest": item.resolution_item_digest,
                "decision": decision,
                "selected_existing_material_id": selected_material_id,
                "conflict_reason": conflict_reason,
                "reviewed_structural_candidate_material_ids": candidate_ids,
                "reviewed_alias_hit_digests": sorted(
                    hit.alias_hit_digest for hit in item.exact_alias_literal_hits
                ),
                "reviewed_registry_conflict_digests": (
                    item.related_registry_conflict_digests
                ),
                "review_note": "Human reviewed the complete bounded Registry evidence.",
            }
        )
    entries.sort(key=lambda entry: entry["resolution_item_id"])
    return {
        "schema_version": "oled_material_registry_decision_manifest.v1",
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "request_artifact_sha256": _sha256_file(request_path),
        "request_artifact_digest": request.request_artifact_digest,
        "source_adjudication_sha256": request.source_adjudication_sha256,
        "source_adjudication_digest": request.source_adjudication_digest,
        "registry_snapshot_sha256": request.registry_snapshot_sha256,
        "registry_snapshot_digest": request.registry_snapshot_digest,
        "reviewed_by": "human_registry_reviewer",
        "reviewed_at": _REVIEWED_AT,
        "adjudication_confirmed": True,
        "decisions": entries,
    }


def _write_decision_manifest(
    tmp_path: Path,
    request: Any,
    request_path: Path,
    *,
    decision: str,
    selected_material_id: str = "",
    conflict_reason: str = "none",
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload = _decision_payload(
        request,
        request_path,
        decision=decision,
        selected_material_id=selected_material_id,
        conflict_reason=conflict_reason,
    )
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "material-registry-decisions.json"
    write_json(path, payload)
    return path, payload


def _adjudicate(
    tmp_path: Path,
    request: Any,
    request_path: Path,
    *,
    decision: str,
    selected_material_id: str = "",
    conflict_reason: str = "none",
) -> tuple[Any, Path, Path]:
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision=decision,
        selected_material_id=selected_material_id,
        conflict_reason=conflict_reason,
    )
    output_path = tmp_path / "material-registry-adjudication.json"
    artifact = build_oled_material_registry_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=output_path,
        generated_at=_ADJUDICATED_AT,
    )
    return artifact, decision_path, output_path


def test_no_match_can_propose_new_entity_without_assigning_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    artifact, decision_path, output_path = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="propose_new_entity",
    )

    assert artifact.status.value == "review_complete_with_pending_new_entity_proposals"
    assert artifact.review_item_count == 1
    assert artifact.upstream_pr_m_review_item_count == 7
    assert artifact.source_registry_eligible_group_count == 1
    assert artifact.source_registry_eligible_cell_count == 5
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.new_entity_proposal_count == 1
    assert artifact.new_entity_proposal_cell_count == 5
    assert artifact.later_observation_staging_eligible_group_count == 0
    assert artifact.device_only_cell_count == 0
    item = artifact.adjudicated_items[0]
    assert item.new_registry_entity_proposed
    assert not item.material_identity_resolved
    assert not item.canonical_material_id_assigned
    assert item.selected_registry_entry is None
    proposal = item.new_entity_proposal
    assert proposal is not None
    assert proposal.reported_subject_literal
    assert proposal.standard_inchi_candidate.startswith("InChI=1S/")
    assert not proposal.material_id_assigned
    assert not proposal.canonical_name_assigned
    assert not proposal.aliases_assigned
    assert not proposal.registry_entry_created
    assert not proposal.registry_written
    assert not artifact.registry_written
    assert not artifact.observations_materialized
    assert artifact.request_artifact_sha256 == _sha256_file(request_path)
    assert artifact.decision_manifest_sha256 == _sha256_file(decision_path)
    assert OledMaterialRegistryAdjudicationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_exact_match_can_map_existing_entity_for_later_staging_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )

    assert artifact.status.value == "existing_entity_mappings_ready_for_later_staging"
    assert artifact.existing_entity_mapping_count == 1
    assert artifact.existing_entity_mapping_cell_count == 5
    assert artifact.later_observation_staging_eligible_group_count == 1
    assert artifact.later_observation_staging_eligible_cell_count == 5
    item = artifact.adjudicated_items[0]
    assert item.existing_registry_entity_mapped
    assert item.material_identity_resolved
    assert item.canonical_material_id_assigned
    assert item.cross_paper_identity_mapping_human_confirmed
    assert item.eligible_for_later_observation_staging
    assert item.selected_registry_entry is not None
    assert item.selected_registry_entry.material_id == "material-0001"
    assert item.new_entity_proposal is None
    assert not item.registry_written
    assert not artifact.reviewed_evidence_staging
    assert not artifact.gold_records_created
    assert not artifact.dataset_written


def test_alias_only_hit_cannot_map_an_existing_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _alias_only_request(tmp_path, monkeypatch)
    item = request.resolution_items[0]
    assert item.match_status.value == "no_exact_structural_candidate"
    assert item.canonical_smiles_candidate_material_ids == []
    assert item.inchikey_candidate_material_ids == []
    assert len(item.exact_alias_literal_hits) == 1
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-alias-only",
    )

    with pytest.raises(ValueError, match="not allowed"):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_ADJUDICATED_AT,
        )


def test_alias_only_hit_can_be_acknowledged_without_blocking_new_entity_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _alias_only_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="propose_new_entity",
    )

    entry = artifact.adjudicated_items[0].decision_entry
    assert len(entry.reviewed_alias_hit_digests) == 1
    assert entry.reviewed_structural_candidate_material_ids == []
    assert artifact.new_entity_proposal_count == 1


def test_no_match_can_remain_unresolved_without_excluding_the_source_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )

    assert artifact.status.value == "review_complete_with_unresolved_items"
    assert artifact.kept_unresolved_count == 1
    assert artifact.kept_unresolved_cell_count == 5
    item = artifact.adjudicated_items[0]
    assert item.kept_unresolved
    assert not item.material_identity_resolved
    assert item.selected_registry_entry is None
    assert item.new_entity_proposal is None


def test_duplicate_structural_keys_must_be_deferred_without_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _duplicate_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="defer_conflict",
        conflict_reason="duplicate_structural_key",
    )

    assert artifact.conflict_deferred_count == 1
    assert artifact.conflict_deferred_cell_count == 5
    item = artifact.adjudicated_items[0]
    assert item.conflict_deferred
    assert item.decision_entry.reviewed_structural_candidate_material_ids == [
        "material-0001",
        "material-0002",
    ]
    assert len(item.decision_entry.reviewed_registry_conflict_digests) == 3
    assert len(item.reviewed_registry_conflict_findings) == 3
    assert item.selected_registry_entry is None
    assert not item.automatic_candidate_merge
    assert not artifact.automatic_candidate_merge


@pytest.mark.parametrize(
    ("request_factory", "reason", "error"),
    (
        (
            _no_match_request,
            "duplicate_structural_key",
            "duplicate-structural-key.*lacks evidence",
        ),
        (
            _exact_match_request,
            "structural_key_disagreement",
            "structural-key-disagreement.*lacks evidence",
        ),
        (
            _no_match_request,
            "reported_name_collision",
            "reported-name-collision.*lacks evidence",
        ),
    ),
)
def test_conflict_reason_must_match_bounded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_factory: Callable[[Path, pytest.MonkeyPatch], tuple[Any, Path]],
    reason: str,
    error: str,
) -> None:
    request, request_path = request_factory(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="defer_conflict",
        conflict_reason=reason,
    )

    with pytest.raises(ValueError, match=error):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_ADJUDICATED_AT,
        )


def test_human_entity_scope_conflict_can_be_deferred_without_precomputed_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="defer_conflict",
        conflict_reason="entity_scope_or_chemistry_conflict",
    )

    assert artifact.conflict_deferred_count == 1
    assert artifact.adjudicated_items[0].reviewed_registry_conflict_findings == []


@pytest.mark.parametrize(
    ("request_factory", "decision", "selected", "reason", "error"),
    (
        (
            _no_match_request,
            "map_to_existing_entity",
            "material-0001",
            "none",
            "not allowed",
        ),
        (
            _exact_match_request,
            "propose_new_entity",
            "",
            "none",
            "not allowed",
        ),
        (
            _duplicate_match_request,
            "map_to_existing_entity",
            "material-0001",
            "none",
            "not allowed",
        ),
    ),
)
def test_unsafe_resolution_transitions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_factory: Callable[[Path, pytest.MonkeyPatch], tuple[Any, Path]],
    decision: str,
    selected: str,
    reason: str,
    error: str,
) -> None:
    request, request_path = request_factory(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision=decision,
        selected_material_id=selected,
        conflict_reason=reason,
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match=error):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("field_name", "replacement", "error"),
    (
        ("run_id", "wrong-run", "run_id"),
        ("paper_id", "wrong-paper", "paper_id"),
        ("request_artifact_sha256", "sha256:" + "a" * 64, "request_artifact_sha256"),
        ("request_artifact_digest", "sha256:" + "b" * 64, "request_artifact_digest"),
        ("source_adjudication_digest", "sha256:" + "c" * 64, "source_adjudication_digest"),
        ("registry_snapshot_digest", "sha256:" + "d" * 64, "registry_snapshot_digest"),
    ),
)
def test_manifest_binding_tamper_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    replacement: str,
    error: str,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
        mutate=lambda payload: payload.__setitem__(field_name, replacement),
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match=error):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda payload: payload["decisions"].clear(), "coverage"),
        (
            lambda payload: payload["decisions"].append(
                {**payload["decisions"][0], "resolution_item_id": "unknown:item"}
            ),
            "coverage",
        ),
        (
            lambda payload: payload["decisions"][0].__setitem__(
                "resolution_item_digest",
                "sha256:" + "e" * 64,
            ),
            "digest",
        ),
        (
            lambda payload: payload["decisions"][0][
                "reviewed_structural_candidate_material_ids"
            ].clear(),
            "structural candidate acknowledgement",
        ),
        (
            lambda payload: payload["decisions"][0][
                "reviewed_alias_hit_digests"
            ].clear(),
            "alias-hit acknowledgement",
        ),
        (
            lambda payload: payload["decisions"][0][
                "reviewed_registry_conflict_digests"
            ].clear(),
            "conflict acknowledgement",
        ),
    ),
)
def test_complete_item_and_acknowledgement_coverage_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    request, request_path = _duplicate_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="defer_conflict",
        conflict_reason="duplicate_structural_key",
        mutate=mutation,
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises((ValueError, ValidationError), match=error):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )
    assert not output_path.exists()


def test_selected_existing_material_id_must_be_a_surfaced_structural_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-9999",
    )

    with pytest.raises(ValueError, match="not a structural candidate"):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_ADJUDICATED_AT,
        )


def test_review_timestamp_and_adjudication_timestamp_are_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
        mutate=lambda payload: payload.__setitem__(
            "reviewed_at",
            "2026-07-14T00:09:00+08:00",
        ),
    )
    with pytest.raises(ValueError, match="predates its request"):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "must-not-exist-1.json",
            generated_at=_ADJUDICATED_AT,
        )

    decision_path.unlink()
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )
    with pytest.raises(ValidationError, match="predates human review"):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "must-not-exist-2.json",
            generated_at="2026-07-14T00:14:00+08:00",
        )


@pytest.mark.parametrize(
    "unsafe_text",
    (
        "token=do-not-store",
        "<script>alert(1)</script>",
        "https://example.test/private-review",
        "/Users/operator/private-review",
    ),
)
def test_reviewer_fields_reject_sensitive_or_active_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_text: str,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    payload = _decision_payload(
        request,
        request_path,
        decision="keep_unresolved",
    )
    payload["decisions"][0]["review_note"] = unsafe_text

    with pytest.raises(ValidationError):
        OledMaterialRegistryDecisionManifest.model_validate(payload)


def test_review_markdown_shows_exact_allowed_decisions_and_acknowledgements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    markdown_path = tmp_path / "material-registry-review.md"

    render_oled_material_registry_adjudication_review_from_files(
        request_artifact_json=request_path,
        output_markdown=markdown_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert markdown.index("PR-M accepted paper-local evidence") < markdown.index(
        "PR-O human Registry decision instructions"
    )
    assert "`map_to_existing_entity`" in markdown
    assert "`keep_unresolved`" in markdown
    assert "`defer_conflict`" in markdown
    decision_section = markdown[markdown.index("### D01:") :]
    assert "`propose_new_entity`" not in decision_section
    assert "reviewed_structural_candidate_material_ids" in markdown
    assert "reviewed_alias_hit_digests" in markdown
    assert "reviewed_registry_conflict_digests" in markdown
    assert "allowed `conflict_reason` for `defer_conflict`" in markdown
    assert "`entity_scope_or_chemistry_conflict`" in decision_section
    assert "`duplicate_structural_key`" not in decision_section
    assert "Registry and observation writes remain disabled" in markdown
    assert "<script" not in markdown.lower()


def test_adjudication_artifact_tamper_breaks_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )

    payload = artifact.model_dump(mode="json")
    payload["registry_written"] = True
    with pytest.raises(ValidationError, match="boundary"):
        OledMaterialRegistryAdjudicationArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["existing_entity_mapping_count"] = 0
    with pytest.raises(ValidationError, match="count"):
        OledMaterialRegistryAdjudicationArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["adjudicated_items"][0]["selected_registry_entry"][
        "canonical_name"
    ] = "changed-name"
    with pytest.raises(ValidationError, match="digest"):
        OledMaterialRegistryAdjudicationArtifact.model_validate(payload)


def test_adjudication_artifact_rebinds_decision_manifest_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    artifact, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )
    tampered = artifact.model_copy(
        update={"decision_manifest_digest": "sha256:" + "f" * 64},
        deep=True,
    )
    tampered = tampered.model_copy(
        update={
            "adjudication_artifact_digest": (
                oled_material_registry_adjudication_artifact_digest(tampered)
            )
        },
        deep=True,
    )

    with pytest.raises(ValidationError, match="decision manifest digest mismatch"):
        OledMaterialRegistryAdjudicationArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_adjudication_artifact_rebinds_embedded_decision_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    original, _, _ = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )
    alternate_dir = tmp_path / "alternate"
    alternate_dir.mkdir()
    alternate, _, _ = _adjudicate(
        alternate_dir,
        request,
        request_path,
        decision="keep_unresolved",
    )
    assert alternate.decision_manifest_digest != original.decision_manifest_digest
    assert alternate.adjudicated_items[0].decision_entry.decision.value == (
        "keep_unresolved"
    )
    tampered = alternate.model_copy(
        update={"decision_manifest_digest": original.decision_manifest_digest},
        deep=True,
    )
    tampered = tampered.model_copy(
        update={
            "adjudication_artifact_digest": (
                oled_material_registry_adjudication_artifact_digest(tampered)
            )
        },
        deep=True,
    )

    with pytest.raises(ValidationError, match="decision manifest digest mismatch"):
        OledMaterialRegistryAdjudicationArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


@pytest.mark.parametrize("protected_kind", ("request", "decision"))
def test_adjudication_output_cannot_overwrite_either_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )
    protected = {"request": request_path, "decision": decision_path}[protected_kind]
    before = protected.read_bytes()

    with pytest.raises(ValueError, match="overwrite"):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=protected,
            generated_at=_ADJUDICATED_AT,
        )
    assert protected.read_bytes() == before


def test_symlinked_inputs_and_output_parent_fail_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )
    request_alias = tmp_path / "request-alias.json"
    request_alias.symlink_to(request_path)
    output_path = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_alias,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )
    assert not output_path.exists()

    output_alias = tmp_path / "output-alias"
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_alias.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_alias / "must-not-exist.json",
            generated_at=_ADJUDICATED_AT,
        )
    assert not (real_output / "must-not-exist.json").exists()


@pytest.mark.parametrize("operation", ("render", "adjudicate"))
def test_output_parent_replacement_mid_operation_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, _ = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    suffix = ".md" if operation == "render" else ".json"
    output_path = output_parent / f"must-not-exist{suffix}"

    if operation == "render":
        original = (
            adjudication_runner.render_oled_material_registry_adjudication_review_markdown
        )

        def replace_after_work(*args: Any, **kwargs: Any) -> str:
            result = original(*args, **kwargs)
            output_parent.rename(displaced)
            output_parent.symlink_to(redirected, target_is_directory=True)
            return result

        monkeypatch.setattr(
            adjudication_runner,
            "render_oled_material_registry_adjudication_review_markdown",
            replace_after_work,
        )
        call = lambda: render_oled_material_registry_adjudication_review_from_files(
            request_artifact_json=request_path,
            output_markdown=output_path,
        )
    else:
        original = adjudication_runner.build_oled_material_registry_adjudication_artifact

        def replace_after_work(**kwargs: Any) -> Any:
            result = original(**kwargs)
            output_parent.rename(displaced)
            output_parent.symlink_to(redirected, target_is_directory=True)
            return result

        monkeypatch.setattr(
            adjudication_runner,
            "build_oled_material_registry_adjudication_artifact",
            replace_after_work,
        )
        call = lambda: build_oled_material_registry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )

    with pytest.raises(ValueError, match="parent changed"):
        call()
    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()


def test_cli_failure_is_redacted_and_does_not_publish_output(tmp_path: Path) -> None:
    sensitive = tmp_path / "token=do-not-disclose.json"
    output_path = tmp_path / "must-not-exist.json"
    stream = StringIO()

    status = main(
        [
            "adjudicate",
            "--request-artifact",
            str(sensitive),
            "--decision-manifest",
            str(tmp_path / "missing-decisions.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    assert status == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "material_registry_adjudication_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()
    assert not output_path.exists()


def test_empty_pr_n_request_accepts_only_an_empty_decision_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)

    def remove_candidate_acceptance(payload: dict[str, Any]) -> None:
        packet = context["packet"]
        candidate_item = next(
            item
            for item in packet.review_items
            if item.candidate_depiction_asset is not None
        )
        entry = next(
            decision
            for decision in payload["decisions"]
            if decision["review_item_id"] == candidate_item.review_item_id
        )
        entry["decision"] = "needs_source_check"
        entry["candidate_source_match"] = "not_checked"
        entry["review_note"] = "Registry eligibility requires a later source check."
        for assessment in entry["anchor_assessments"]:
            assessment["assessment"] = "not_checked"
            assessment["review_note"] = "The source anchor remains unchecked."

    decision_path, _ = _write_decisions(
        context,
        mutate=remove_candidate_acceptance,
        filename="no-registry-eligible-decisions.json",
    )
    source_path = context["review_dir"] / "no-registry-eligible-adjudication.json"
    source = build_oled_supplementary_material_identity_adjudication_from_files(
        **_adjudication_kwargs(
            context,
            decision_path=decision_path,
            output_path=source_path,
        )
    )
    assert source.later_registry_review_eligible_group_count == 0
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    request_path = tmp_path / "empty-request.json"
    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=request_path,
        generated_at=_REQUEST_AT,
    )
    assert request.resolution_item_count == 0
    manifest_payload = _decision_payload(
        request,
        request_path,
        decision="keep_unresolved",
    )
    manifest_path = tmp_path / "empty-decisions.json"
    write_json(manifest_path, manifest_payload)

    artifact = build_oled_material_registry_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=manifest_path,
        output_json=tmp_path / "empty-adjudication.json",
        generated_at=_ADJUDICATED_AT,
    )

    assert artifact.status.value == "no_registry_eligible_candidates"
    assert artifact.upstream_pr_m_review_item_count == 7
    assert artifact.source_registry_eligible_group_count == 0
    assert artifact.source_registry_eligible_cell_count == 0
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.review_item_count == 0
    assert artifact.adjudicated_items == []
    assert artifact.later_observation_staging_eligible_cell_count == 0


def test_build_does_not_mutate_request_or_decision_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    decision_path, payload = _write_decision_manifest(
        tmp_path,
        request,
        request_path,
        decision="keep_unresolved",
    )
    manifest = OledMaterialRegistryDecisionManifest.model_validate(payload)
    request_before = deepcopy(request.model_dump(mode="json"))
    manifest_before = deepcopy(manifest.model_dump(mode="json"))

    artifact = build_oled_material_registry_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=tmp_path / "adjudication.json",
        generated_at=_ADJUDICATED_AT,
    )

    assert request.model_dump(mode="json") == request_before
    assert manifest.model_dump(mode="json") == manifest_before
    assert artifact.adjudicated_items[0].decision_entry.model_dump(mode="json") == (
        manifest_before["decisions"][0]
    )
