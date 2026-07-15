from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent import oled_reviewed_evidence_staging_preflight as preflight_runner
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import OledMeasurementCondition
from ai4s_agent.domains.oled_observation_materialization_candidate import (
    OledObservationMaterializationCandidateItem,
    _canonicalize_observation,
    _observation_candidate_item_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerEntry,
    OledReviewedEvidenceLedgerEntryStatus,
    OledReviewedEvidencePreflightDisposition,
    OledReviewedEvidenceStagingPreflightArtifact,
    _derive_preflight_item,
    _derive_preflight_items,
    _ledger_entry_digest,
    _ledger_projection_payload,
    _projection_id_from_fields,
    _projection_payload_digest,
    _semantic_contract_digest,
    _source_claim_id_from_fields,
    build_empty_oled_reviewed_evidence_ledger_snapshot,
    build_oled_reviewed_evidence_ledger_entry_from_candidate,
    build_oled_reviewed_evidence_ledger_snapshot,
    build_oled_reviewed_evidence_semantic_contract_snapshot,
    oled_reviewed_evidence_staging_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    _source_cell_payload,
    _stable_hash as _semantic_hash,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryProposalComparisonContext,
)
from ai4s_agent.oled_observation_materialization_candidate import (
    build_oled_observation_materialization_candidate_from_files,
)
from ai4s_agent.oled_reviewed_evidence_staging_preflight import (
    build_oled_reviewed_evidence_staging_preflight_from_files,
    main,
)
from tests.test_oled_observation_materialization_candidate import (
    _build_exact_chain,
    _file_kwargs,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_LEDGER_AT = "2026-07-14T00:41:00+08:00"
_PREFLIGHT_AT = "2026-07-14T00:42:00+08:00"


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Any, Path]:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    materialization_path = tmp_path / "observation-materialization-candidates.json"
    materialization = build_oled_observation_materialization_candidate_from_files(
        **_file_kwargs(chain, materialization_path)
    )
    ledger = build_empty_oled_reviewed_evidence_ledger_snapshot(
        generated_at=_LEDGER_AT
    )
    ledger_path = tmp_path / "reviewed-evidence-ledger-snapshot.json"
    write_json(ledger_path, ledger.model_dump(mode="json"))
    return materialization, materialization_path, ledger, ledger_path


def _build_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger: Any | None = None,
) -> tuple[Any, Any, Path, Path]:
    materialization, materialization_path, default_ledger, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    selected_ledger = ledger or default_ledger
    if ledger is not None:
        write_json(ledger_path, selected_ledger.model_dump(mode="json"))
    output_path = tmp_path / "reviewed-evidence-staging-preflight.json"
    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )
    return materialization, artifact, output_path, ledger_path


def _entry_for_candidate(
    materialization: Any,
    candidate: Any,
    *,
    status: OledReviewedEvidenceLedgerEntryStatus = (
        OledReviewedEvidenceLedgerEntryStatus.ACTIVE
    ),
) -> OledReviewedEvidenceLedgerEntry:
    return build_oled_reviewed_evidence_ledger_entry_from_candidate(
        candidate=candidate,
        source_materialization_artifact_digest=materialization.artifact_digest,
        semantic_contract=build_oled_reviewed_evidence_semantic_contract_snapshot(),
        status=status,
        created_at=_LEDGER_AT,
    )


def _alternate_source_entry(
    entry: OledReviewedEvidenceLedgerEntry,
    *,
    normalized_value: float | int | str | None = None,
    preserve_value: bool = True,
    preserve_candidate_id: bool = False,
    semantic_contract_digest: str | None = None,
) -> OledReviewedEvidenceLedgerEntry:
    source_pdf_sha = "sha256:" + "a" * 64
    source_cell_digest = "sha256:" + "b" * 64
    source_candidate_digest = "sha256:" + "c" * 64
    source_claim_id = _source_claim_id_from_fields(
        source_pdf_sha256=source_pdf_sha,
        source_cell_digest=source_cell_digest,
    )
    selected_contract_digest = semantic_contract_digest or entry.semantic_contract_digest
    projection_id = _projection_id_from_fields(
        source_claim_id=source_claim_id,
        source_candidate_digest=source_candidate_digest,
        selected_material_id=entry.selected_material_id,
        registry_entry_digest=entry.registry_entry_digest,
        cell_disposition_digest=entry.cell_disposition_digest,
        semantic_contract_digest=selected_contract_digest,
    )
    payload = entry.model_dump(mode="python")
    payload.update(
        {
            "entry_id": f"reviewed-evidence:{projection_id.split(':', 1)[-1]}",
            "source_claim_id": source_claim_id,
            "projection_id": projection_id,
            "source_candidate_id": (
                entry.source_candidate_id
                if preserve_candidate_id
                else "observation-candidate:alternate-source"
            ),
            "source_candidate_digest": source_candidate_digest,
            "source_pdf_sha256": source_pdf_sha,
            "source_cell_digest": source_cell_digest,
            "semantic_contract_digest": selected_contract_digest,
            "normalized_value": (
                entry.normalized_value if preserve_value else normalized_value
            ),
            "entry_digest": "sha256:" + "0" * 64,
        }
    )
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["projection_payload_digest"] = _projection_payload_digest(
        _ledger_projection_payload(provisional)
    )
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["entry_digest"] = _ledger_entry_digest(provisional)
    return OledReviewedEvidenceLedgerEntry.model_validate(payload)


def test_paper016_candidates_are_grouped_and_ready_without_becoming_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, artifact, output_path, _ = _build_preflight(
        tmp_path,
        monkeypatch,
    )

    assert artifact.status.value == "ready_for_reviewed_evidence_ledger_write"
    assert artifact.source_candidate_count == 5
    assert artifact.source_row_group_count == 1
    assert artifact.source_row_groups[0].observation_count == 5
    assert artifact.ledger_write_count == 5
    assert artifact.exact_replay_count == 0
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    assert not artifact.reviewed_evidence_staged
    assert not artifact.ledger_written
    assert not artifact.confidence_score_invented
    assert not artifact.gold_records_created
    assert not artifact.dataset_written
    assert artifact.materialization_artifact_digest == materialization.artifact_digest
    assert artifact.semantic_contract.contract_digest.startswith("sha256:")
    assert all(
        item.disposition == OledReviewedEvidencePreflightDisposition.NEW_CLAIM_READY
        for item in artifact.preflight_items
    )
    assert all(
        item.gold_blocker_codes
        == ["missing_confidence_assessment", "scientific_consistency_not_reviewed"]
        for item in artifact.preflight_items
    )
    by_text = {
        item.source_candidate.property_observation.reported_value_text: item
        for item in artifact.preflight_items
    }
    assert by_text["-1.70"].source_candidate.property_observation.reported_decimal_places == 2
    assert OledReviewedEvidenceStagingPreflightArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_exact_replay_is_a_deterministic_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    entries = [
        _entry_for_candidate(materialization, candidate)
        for candidate in materialization.observation_candidates
    ]
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=entries,
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:with-exact-replay",
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=tmp_path / "exact-replay-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status.value == "no_ledger_changes_required"
    assert artifact.exact_replay_count == 5
    assert artifact.ledger_write_count == 0
    assert all(not item.ledger_write_required for item in artifact.preflight_items)


def test_exact_replay_rejects_rehashed_ledger_projection_payload_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    candidate = materialization.observation_candidates[0]
    entry = _entry_for_candidate(materialization, candidate)
    payload = entry.model_dump(mode="python")
    payload["reported_unit"] = "tampered-unit"
    payload["projection_payload_digest"] = "sha256:" + "0" * 64
    payload["entry_digest"] = "sha256:" + "0" * 64
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["projection_payload_digest"] = _projection_payload_digest(
        _ledger_projection_payload(provisional)
    )
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["entry_digest"] = _ledger_entry_digest(provisional)
    tampered_entry = OledReviewedEvidenceLedgerEntry.model_validate(payload)
    assert tampered_entry.projection_id == entry.projection_id

    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[tampered_entry],
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:projection-payload-tamper",
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))
    output_path = tmp_path / "projection-payload-tamper-preflight.json"

    with pytest.raises(
        ValueError,
        match="exact replay ledger projection payload does not match",
    ):
        build_oled_reviewed_evidence_staging_preflight_from_files(
            materialization_artifact_json=materialization_path,
            ledger_snapshot_json=ledger_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("preserve_value", "expected_disposition", "expected_status"),
    (
        (
            True,
            OledReviewedEvidencePreflightDisposition.CONSISTENT_DUPLICATE_READY,
            "ready_for_reviewed_evidence_ledger_write",
        ),
        (
            False,
            OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE,
            "manual_exception_review_required",
        ),
    ),
)
def test_cross_source_duplicate_and_conflict_are_not_silently_collapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_value: bool,
    expected_disposition: OledReviewedEvidencePreflightDisposition,
    expected_status: str,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    candidate = materialization.observation_candidates[0]
    entry = _entry_for_candidate(materialization, candidate)
    alternate = _alternate_source_entry(
        entry,
        preserve_value=preserve_value,
        normalized_value=123.456,
    )
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[alternate],
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:cross-source",
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=tmp_path / f"{expected_disposition.value}.json",
        generated_at=_PREFLIGHT_AT,
    )
    item = next(
        item
        for item in artifact.preflight_items
        if item.source_candidate.candidate_id == candidate.candidate_id
    )

    assert item.disposition == expected_disposition
    assert item.ledger_write_required
    assert artifact.status.value == expected_status
    if preserve_value:
        assert not item.quarantine_on_write
        assert not item.manual_exception_review_required
    else:
        assert item.quarantine_on_write
        assert item.manual_exception_review_required
        assert "unresolved_value_conflict" in item.gold_blocker_codes


def test_source_candidate_id_collision_is_reported_but_not_used_as_global_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    candidate = materialization.observation_candidates[0]
    entry = _alternate_source_entry(
        _entry_for_candidate(materialization, candidate),
        preserve_candidate_id=True,
    )
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[entry],
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:candidate-id-collision",
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=tmp_path / "candidate-id-collision.json",
        generated_at=_PREFLIGHT_AT,
    )
    item = next(
        item
        for item in artifact.preflight_items
        if item.source_candidate.candidate_id == candidate.candidate_id
    )

    assert artifact.candidate_id_collision_count == 1
    assert item.candidate_id_collision_detected
    assert item.source_claim_id != entry.source_claim_id
    assert item.projection_id != entry.projection_id


def test_same_source_changed_projection_requires_exception_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    candidate = materialization.observation_candidates[0]
    entry = _entry_for_candidate(materialization, candidate)
    payload = entry.model_dump(mode="python")
    payload["source_candidate_digest"] = "sha256:" + "d" * 64
    payload["projection_id"] = _projection_id_from_fields(
        source_claim_id=entry.source_claim_id,
        source_candidate_digest=payload["source_candidate_digest"],
        selected_material_id=entry.selected_material_id,
        registry_entry_digest=entry.registry_entry_digest,
        cell_disposition_digest=entry.cell_disposition_digest,
        semantic_contract_digest=entry.semantic_contract_digest,
    )
    payload["entry_id"] = f"reviewed-evidence:{payload['projection_id'].split(':', 1)[-1]}"
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["projection_payload_digest"] = _projection_payload_digest(
        _ledger_projection_payload(provisional)
    )
    payload["entry_digest"] = "sha256:" + "0" * 64
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["entry_digest"] = _ledger_entry_digest(provisional)
    changed_projection = OledReviewedEvidenceLedgerEntry.model_validate(payload)
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[changed_projection],
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:changed-projection",
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=tmp_path / "changed-projection.json",
        generated_at=_PREFLIGHT_AT,
    )
    item = next(
        item
        for item in artifact.preflight_items
        if item.source_candidate.candidate_id == candidate.candidate_id
    )

    assert item.disposition == OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW
    assert item.manual_exception_review_required
    assert not item.ledger_write_required
    assert artifact.revision_review_count == 1


def test_cross_contract_comparison_requires_explicit_migration_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, materialization_path, _, ledger_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    candidate = materialization.observation_candidates[0]
    current_contract = build_oled_reviewed_evidence_semantic_contract_snapshot()
    contract_payload = current_contract.model_dump(mode="python")
    contract_payload["comparison_context_policy"] += " Historical test revision."
    contract_payload["contract_digest"] = "sha256:" + "0" * 64
    provisional_contract = current_contract.model_construct(**contract_payload)
    contract_payload["contract_digest"] = _semantic_contract_digest(
        provisional_contract
    )
    historical_contract = type(current_contract).model_validate(contract_payload)
    historical_entry = _alternate_source_entry(
        _entry_for_candidate(materialization, candidate),
        semantic_contract_digest=historical_contract.contract_digest,
    )
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[historical_entry],
        generated_at=_LEDGER_AT,
        snapshot_id="reviewed-evidence-ledger:historical-contract",
        semantic_contracts=[historical_contract],
    )
    write_json(ledger_path, ledger.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=tmp_path / "semantic-contract-migration.json",
        generated_at=_PREFLIGHT_AT,
    )
    item = next(
        item
        for item in artifact.preflight_items
        if item.source_candidate.candidate_id == candidate.candidate_id
    )

    assert item.disposition == (
        OledReviewedEvidencePreflightDisposition
        .SEMANTIC_CONTRACT_MIGRATION_REQUIRED
    )
    assert item.manual_exception_review_required
    assert not item.ledger_write_required
    assert "semantic_contract_migration_required" in item.gold_blocker_codes
    assert artifact.semantic_contract_migration_count == 1


def test_incomplete_photophysical_context_is_queryable_but_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, _, ledger, _ = _inputs(tmp_path, monkeypatch)
    source = materialization.observation_candidates[0]
    mapping = source.mapping_summary.model_copy(
        update={
            "property_id": "photoluminescence_peak_nm",
            "property_label": "PL peak (nm)",
            "reported_unit": "nm",
            "canonical_unit": "nm",
            "target_layer": OledCausalLayer.MOLECULE,
            "comparison_context": OledSupplementaryProposalComparisonContext.model_validate(
                {
                    "measurement_temperature": None,
                    "host_material": None,
                    "dopant_concentration": None,
                    "sample_form": None,
                    "excitation_wavelength": None,
                    "lifetime_fit_method": None,
                }
            ),
        }
    )
    evidence = source.property_observation.evidence_sources[0].model_copy(
        update={"layer": OledCausalLayer.MOLECULE}
    )
    observation = source.property_observation.model_copy(
        update={
            "property_label": "photoluminescence_peak_nm",
            "value": 520.0,
            "unit": "nm",
            "reported_value_text": "520.0",
            "reported_decimal_places": 1,
            "condition": OledMeasurementCondition(),
            "evidence_sources": [evidence],
            "metadata": {
                **source.property_observation.metadata,
                "property_id": "photoluminescence_peak_nm",
                "reported_property_label": "PL peak (nm)",
                "canonical_unit": "nm",
            },
        }
    )
    canonical = _canonicalize_observation(OledCausalLayer.MOLECULE, observation)
    semantic_source_cell = source.semantic_source_cell.model_copy(
        update={
            "cell_value": "520.0",
            "reported_value_text": "520.0",
            "reported_decimal_places": 1,
        }
    )
    semantic_source_cell = semantic_source_cell.model_copy(
        update={
            "source_cell_digest": _semantic_hash(
                _source_cell_payload(semantic_source_cell)
            )
        }
    )
    payload = {
        field_name: getattr(source, field_name)
        for field_name in OledObservationMaterializationCandidateItem.model_fields
    }
    payload.update(
        {
            "mapping_summary": mapping,
            "semantic_source_cell": semantic_source_cell,
            "source_cell_digest": semantic_source_cell.source_cell_digest,
            "candidate_id": (
                "observation-candidate:"
                f"{semantic_source_cell.source_cell_digest[7:]}"
            ),
            "property_observation": observation,
            "canonical_observation": canonical,
            "comparison_context_status": canonical.comparison_context_status,
            "comparison_context_required_fields": sorted(
                canonical.comparison_context_required_fields
            ),
            "comparison_context_missing_fields": sorted(
                canonical.comparison_context_missing_fields
            ),
            "comparison_ready": canonical.is_comparison_ready,
            "candidate_digest": "sha256:" + "0" * 64,
        }
    )
    provisional = OledObservationMaterializationCandidateItem.model_construct(**payload)
    payload["candidate_digest"] = _observation_candidate_item_digest(provisional)
    incomplete = OledObservationMaterializationCandidateItem.model_validate(payload)
    contract = build_oled_reviewed_evidence_semantic_contract_snapshot()

    item = _derive_preflight_item(
        incomplete,
        materialization,
        ledger,
        contract,
    )

    assert item.disposition == (
        OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE
    )
    assert item.ledger_write_required
    assert item.quarantine_on_write
    assert not item.comparison_ready
    assert "incomplete_comparison_context" in item.gold_blocker_codes
    assert item.source_candidate.comparison_context_missing_fields


def test_device_layer_candidate_fails_closed_before_preflight_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, _, ledger, _ = _inputs(tmp_path, monkeypatch)
    candidate = materialization.observation_candidates[0]
    device_candidate = candidate.model_copy(
        update={
            "canonical_observation": candidate.canonical_observation.model_copy(
                update={"layer": OledCausalLayer.DEVICE}
            )
        }
    )
    forged = materialization.model_copy(
        update={"observation_candidates": [device_candidate]}
    )

    with pytest.raises(ValueError, match="device-only"):
        _derive_preflight_items(
            forged,
            ledger,
            build_oled_reviewed_evidence_semantic_contract_snapshot(),
        )


def test_item_tamper_fails_even_after_outer_artifact_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact, _, _ = _build_preflight(tmp_path, monkeypatch)
    changed_item = artifact.preflight_items[0].model_copy(
        update={
            "disposition": OledReviewedEvidencePreflightDisposition.EXACT_REPLAY,
            "ledger_write_required": False,
            "preflight_item_digest": "sha256:" + "e" * 64,
        }
    )
    provisional = artifact.model_copy(
        update={
            "preflight_items": [changed_item, *artifact.preflight_items[1:]],
            "preflight_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = provisional.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_reviewed_evidence_staging_preflight_artifact_digest(provisional)
            )
        }
    )

    with pytest.raises(ValidationError, match="item digest|derivation"):
        OledReviewedEvidenceStagingPreflightArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_semantic_contract_tamper_fails_after_contract_and_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact, _, _ = _build_preflight(tmp_path, monkeypatch)
    contract_payload = artifact.semantic_contract.model_dump(mode="python")
    contract_payload["comparison_context_policy"] += " Forged policy."
    contract_payload["contract_digest"] = "sha256:" + "0" * 64
    provisional_contract = artifact.semantic_contract.model_construct(
        **contract_payload
    )
    contract_payload["contract_digest"] = _semantic_contract_digest(
        provisional_contract
    )
    changed_contract = type(artifact.semantic_contract).model_validate(
        contract_payload
    )
    provisional = artifact.model_copy(
        update={
            "semantic_contract": changed_contract,
            "preflight_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = provisional.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_reviewed_evidence_staging_preflight_artifact_digest(provisional)
            )
        }
    )

    with pytest.raises(ValidationError, match="semantic contract"):
        OledReviewedEvidenceStagingPreflightArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_preflight_cannot_predate_pr_q_or_ledger_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, materialization_path, _, ledger_path = _inputs(tmp_path, monkeypatch)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="predates PR-Q|predates ledger"):
        build_oled_reviewed_evidence_staging_preflight_from_files(
            materialization_artifact_json=materialization_path,
            ledger_snapshot_json=ledger_path,
            output_json=output_path,
            generated_at="2026-07-14T00:39:59+08:00",
        )

    assert not output_path.exists()


def test_output_cannot_overwrite_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, materialization_path, _, ledger_path = _inputs(tmp_path, monkeypatch)
    before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="output must not overwrite"):
        build_oled_reviewed_evidence_staging_preflight_from_files(
            materialization_artifact_json=materialization_path,
            ledger_snapshot_json=ledger_path,
            output_json=ledger_path,
            generated_at=_PREFLIGHT_AT,
        )

    assert ledger_path.read_bytes() == before


def test_cli_failure_is_redacted_and_does_not_publish_output(tmp_path: Path) -> None:
    sensitive = tmp_path / "secret=do-not-disclose.json"
    output_path = tmp_path / "must-not-exist.json"
    stream = StringIO()

    status = main(
        [
            "--materialization-candidates",
            str(sensitive),
            "--ledger-snapshot",
            str(tmp_path / "missing-ledger.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    assert status == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "reviewed_evidence_staging_preflight_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()
    assert not output_path.exists()


def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, materialization_path, _, ledger_path = _inputs(tmp_path, monkeypatch)
    output_dir = tmp_path / "preflight-output"
    output_dir.mkdir()
    output_path = output_dir / "preflight.json"
    replacement = tmp_path / "replacement-output"
    original_publish = preflight_runner._publish_with_pinned_parent

    def replace_parent_then_publish(*args: Any, **kwargs: Any) -> None:
        output_dir.rename(replacement)
        output_dir.mkdir()
        original_publish(*args, **kwargs)

    monkeypatch.setattr(
        preflight_runner,
        "_publish_with_pinned_parent",
        replace_parent_then_publish,
    )

    with pytest.raises(ValueError, match="parent changed"):
        build_oled_reviewed_evidence_staging_preflight_from_files(
            materialization_artifact_json=materialization_path,
            ledger_snapshot_json=ledger_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )

    assert not output_path.exists()
    assert not (replacement / "preflight.json").exists()


def test_file_sha_bindings_record_exact_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, materialization_path, _, ledger_path = _inputs(tmp_path, monkeypatch)
    output_path = tmp_path / "sha-bound-preflight.json"

    artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.materialization_artifact_sha256 == _sha256_file(
        materialization_path
    )
    assert artifact.ledger_snapshot_sha256 == _sha256_file(ledger_path)
