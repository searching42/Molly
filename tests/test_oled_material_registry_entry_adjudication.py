from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryEntryAdjudicationArtifact,
    OledMaterialRegistryEntryDecisionManifest,
    build_oled_material_registry_entry_adjudication_artifact,
    oled_material_registry_entry_adjudication_artifact_digest,
)
from ai4s_agent.oled_material_registry_entry_adjudication import (
    build_oled_material_registry_entry_adjudication_from_files,
)
from tests.test_oled_material_registry_entry_proposal_request import (
    _PROPOSAL_AT,
    _multi_candidate_no_match_request,
    _proposal_artifact,
    _seven_unique_no_match_request,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_REVIEWED_AT = "2026-07-14T00:30:00+08:00"
_ADJUDICATED_AT = "2026-07-14T00:35:00+08:00"
_FILE_SHA = "sha256:" + "a" * 64


def _manifest_payload(
    request: Any,
    request_sha256: str,
    *,
    decision: str = "approve_local_registry_entry_candidate",
) -> dict[str, Any]:
    findings_by_item = {
        item.entry_review_item_id: [] for item in request.entry_review_items
    }
    for finding in request.batch_conflict_findings:
        for item_id in finding.affected_entry_review_item_ids:
            findings_by_item[item_id].append(finding.finding_digest)
    decisions = []
    for item in request.entry_review_items:
        approved = decision == "approve_local_registry_entry_candidate"
        source = item.source_adjudicated_item
        decisions.append(
            {
                "entry_review_item_id": item.entry_review_item_id,
                "entry_review_item_digest": item.entry_review_item_digest,
                "review_contract_digest": item.review_contract_digest,
                "decision": decision,
                "approved_material_id": item.proposed_material_id if approved else "",
                "approved_canonical_name": (
                    item.proposed_canonical_name if approved else ""
                ),
                "approved_aliases": item.proposed_aliases if approved else [],
                "reviewed_existing_name_hit_digests": sorted(
                    hit.alias_hit_digest
                    for hit in source.request_item.exact_alias_literal_hits
                ),
                "reviewed_snapshot_conflict_finding_digests": sorted(
                    finding.finding_digest
                    for finding in source.reviewed_registry_conflict_findings
                ),
                "reviewed_batch_conflict_finding_digests": sorted(
                    findings_by_item[item.entry_review_item_id]
                ),
                "single_entity_scope_confirmed": approved,
                "material_id_approved": approved,
                "canonical_name_approved": approved,
                "aliases_approved": approved,
                "review_contract_acknowledged": True,
                "review_note": "Human completed the exact bound Registry-entry review.",
            }
        )
    return {
        "schema_version": "oled_material_registry_entry_decision_manifest.v1",
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "request_artifact_sha256": request_sha256,
        "request_artifact_digest": request.proposal_request_artifact_digest,
        "registry_snapshot_sha256": request.registry_snapshot_sha256,
        "registry_snapshot_digest": request.registry_snapshot_digest,
        "review_contract_digest": request.review_contract.contract_digest,
        "reviewed_by": "human-reviewer",
        "reviewed_at": _REVIEWED_AT,
        "adjudication_confirmed": True,
        "decisions": decisions,
    }


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    request_factory: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    kwargs = {}
    if request_factory is not None:
        kwargs["request_factory"] = request_factory
    request, *_ = _proposal_artifact(tmp_path, monkeypatch, **kwargs)
    manifest_payload = _manifest_payload(request, _FILE_SHA)
    manifest = OledMaterialRegistryEntryDecisionManifest.model_validate(
        manifest_payload
    )
    artifact = build_oled_material_registry_entry_adjudication_artifact(
        request=request,
        request_artifact_sha256=_FILE_SHA,
        decision_manifest=manifest,
        decision_manifest_sha256="sha256:" + "b" * 64,
        generated_at=_ADJUDICATED_AT,
    )
    return artifact, request, manifest_payload


def test_approved_candidate_preserves_exact_proposal_without_writing_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, request, _ = _build(tmp_path, monkeypatch)
    item = artifact.adjudicated_items[0]
    candidate = item.approved_entry_candidate
    assert candidate is not None
    assert candidate.registry_entry.material_id == request.entry_review_items[0].proposed_material_id
    assert candidate.registry_entry.canonical_name == request.entry_review_items[0].proposed_canonical_name
    assert candidate.registry_entry.aliases == []
    assert item.eligible_for_registry_write_preflight is True
    assert artifact.material_id_reserved is False
    assert artifact.material_id_assigned is False
    assert artifact.registry_entry_created is False
    assert artifact.registry_written is False
    assert artifact.standalone_input_bytes_revalidation_supported is False


def test_seven_source_ordered_approvals_cover_35_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _ = _build(
        tmp_path,
        monkeypatch,
        request_factory=_seven_unique_no_match_request,
    )
    assert artifact.review_item_count == 7
    assert artifact.review_cell_count == 35
    assert artifact.approved_entry_candidate_count == 7
    assert artifact.registry_write_preflight_eligible_count == 7
    assert artifact.status.value == "approved_entry_candidates_ready_for_write_preflight"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("approved_material_id", "material:" + "f" * 32),
        ("approved_canonical_name", "unreported alternate name"),
        ("approved_aliases", ["unreported alias"]),
    ],
)
def test_approval_cannot_rewrite_exact_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    replacement: Any,
) -> None:
    _, request, payload = _build(tmp_path, monkeypatch)
    payload["decisions"][0][field_name] = replacement
    manifest = OledMaterialRegistryEntryDecisionManifest.model_validate(payload)
    with pytest.raises(ValueError, match="differ.*from the exact PR-V proposal"):
        build_oled_material_registry_entry_adjudication_artifact(
            request=request,
            request_artifact_sha256=_FILE_SHA,
            decision_manifest=manifest,
            decision_manifest_sha256="sha256:" + "b" * 64,
            generated_at=_ADJUDICATED_AT,
        )


def test_missing_contract_acknowledgement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, payload = _build(tmp_path, monkeypatch)
    payload["decisions"][0]["review_contract_acknowledged"] = False
    with pytest.raises(ValidationError, match="contract must be acknowledged"):
        OledMaterialRegistryEntryDecisionManifest.model_validate(payload)


def test_standalone_artifact_rebinds_embedded_decision_manifest_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _ = _build(tmp_path, monkeypatch)
    tampered = artifact.model_copy(
        update={"decision_manifest_digest": "sha256:" + "c" * 64}, deep=True
    )
    tampered = tampered.model_copy(
        update={
            "adjudication_artifact_digest": (
                oled_material_registry_entry_adjudication_artifact_digest(tampered)
            )
        }
    )
    with pytest.raises(ValidationError, match="decision manifest digest mismatch"):
        OledMaterialRegistryEntryAdjudicationArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_file_entry_binds_bytes_and_publishes_fresh_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, *_ = _proposal_artifact(tmp_path, monkeypatch)
    request_path = write_json(tmp_path / "request.json", request.model_dump(mode="json"))
    decision_path = write_json(
        tmp_path / "decisions.json",
        _manifest_payload(request, _sha256_file(request_path)),
    )
    output_path = tmp_path / "adjudication.json"
    artifact = build_oled_material_registry_entry_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=output_path,
        generated_at=_ADJUDICATED_AT,
    )
    assert artifact.request_artifact_sha256 == _sha256_file(request_path)
    assert artifact.decision_manifest_sha256 == _sha256_file(decision_path)
    assert output_path.is_file()
    with pytest.raises(ValueError, match="fresh"):
        build_oled_material_registry_entry_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=output_path,
            generated_at=_ADJUDICATED_AT,
        )


def test_nonapproval_creates_no_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, *_ = _proposal_artifact(tmp_path, monkeypatch)
    payload = _manifest_payload(request, _FILE_SHA, decision="keep_unresolved")
    artifact = build_oled_material_registry_entry_adjudication_artifact(
        request=request,
        request_artifact_sha256=_FILE_SHA,
        decision_manifest=OledMaterialRegistryEntryDecisionManifest.model_validate(
            payload
        ),
        decision_manifest_sha256="sha256:" + "b" * 64,
        generated_at=_ADJUDICATED_AT,
    )
    assert artifact.approved_entry_candidate_count == 0
    assert artifact.kept_unresolved_count == artifact.review_item_count
    assert all(item.approved_entry_candidate is None for item in artifact.adjudicated_items)


def test_acknowledged_batch_conflict_still_blocks_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, *_ = _proposal_artifact(
        tmp_path,
        monkeypatch,
        request_factory=_multi_candidate_no_match_request,
    )
    payload = _manifest_payload(request, _FILE_SHA)
    with pytest.raises(ValueError, match="batch conflicts.*cannot be approved"):
        build_oled_material_registry_entry_adjudication_artifact(
            request=request,
            request_artifact_sha256=_FILE_SHA,
            decision_manifest=(
                OledMaterialRegistryEntryDecisionManifest.model_validate(payload)
            ),
            decision_manifest_sha256="sha256:" + "b" * 64,
            generated_at=_ADJUDICATED_AT,
        )
