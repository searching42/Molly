from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.domains.oled_reviewed_evidence_facet_review_request import (
    OledReviewedEvidenceFacetReviewRequestArtifact,
    OledReviewedEvidenceFacetReviewRequestStatus,
    oled_reviewed_evidence_facet_review_request_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    build_oled_reviewed_evidence_ledger_snapshot,
)
from ai4s_agent.oled_reviewed_evidence_facet_review_request import (
    build_oled_reviewed_evidence_facet_review_request_from_files,
    main,
)
from ai4s_agent.oled_reviewed_evidence_ledger_postwrite_verifier import (
    build_oled_reviewed_evidence_ledger_postwrite_verification_from_files,
)
from tests.test_oled_reviewed_evidence_ledger_postwrite_verifier import (
    _write_result,
)
from tests.test_oled_reviewed_evidence_staging_preflight import (
    _alternate_source_entry,
    _entry_for_candidate,
    _inputs,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_VERIFY_AT = "2026-07-14T00:44:00+08:00"
_REQUEST_AT = "2026-07-14T00:45:00+08:00"


def _verification_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger=None,
    inputs=None,
):
    _, receipt_path, snapshot_path = _write_result(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )
    verification_path = tmp_path / "postwrite-verification.json"
    verification = (
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=verification_path,
            generated_at=_VERIFY_AT,
        )
    )
    return verification, verification_path


def test_request_groups_five_active_observations_by_exact_source_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification, verification_path = _verification_result(tmp_path, monkeypatch)
    output_path = tmp_path / "facet-review-request.json"

    artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=output_path,
        generated_at=_REQUEST_AT,
    )

    assert artifact.status == OledReviewedEvidenceFacetReviewRequestStatus.READY
    assert artifact.eligible_observation_count == 5
    assert artifact.review_group_count == 1
    assert artifact.review_groups[0].observation_count == 5
    assert artifact.excluded_quarantined_count == 0
    assert artifact.device_only_count == 0
    assert artifact.postwrite_verification_sha256 == _sha256_file(
        verification_path
    )
    assert artifact.postwrite_verification == verification
    assert artifact.review_contract.requested_facets == [
        "confidence_sufficiency",
        "scientific_consistency",
    ]
    assert not artifact.review_contract.numeric_confidence_score_requested
    assert not artifact.review_contract.confidence_is_calibrated_probability
    assert all(
        observation.gold_blocker_codes
        == [
            "missing_confidence_assessment",
            "scientific_consistency_not_reviewed",
        ]
        for observation in artifact.review_groups[0].observations
    )
    assert all(
        observation.comparison_context is None
        and observation.comparison_context_hash is None
        for observation in artifact.review_groups[0].observations
    )
    assert OledReviewedEvidenceFacetReviewRequestArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_exact_replay_active_entries_remain_eligible_for_unfinished_facets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    materialization, _, _, _ = inputs
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[
            _entry_for_candidate(materialization, candidate)
            for candidate in materialization.observation_candidates
        ],
        generated_at="2026-07-14T00:41:00+08:00",
        snapshot_id="reviewed-evidence-ledger:facet-review-noop",
    )
    _, verification_path = _verification_result(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=tmp_path / "noop-facet-request.json",
        generated_at=_REQUEST_AT,
    )

    assert artifact.eligible_observation_count == 5
    assert artifact.review_group_count == 1
    assert artifact.excluded_quarantined_count == 0


def test_quarantined_conflict_is_excluded_from_facet_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    materialization, _, _, _ = inputs
    candidate = materialization.observation_candidates[0]
    historical = _alternate_source_entry(
        _entry_for_candidate(materialization, candidate),
        preserve_value=False,
        normalized_value=123.456,
    )
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[historical],
        generated_at="2026-07-14T00:41:00+08:00",
        snapshot_id="reviewed-evidence-ledger:facet-review-conflict",
    )
    _, verification_path = _verification_result(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=tmp_path / "conflict-facet-request.json",
        generated_at=_REQUEST_AT,
    )

    assert artifact.eligible_observation_count == 4
    assert artifact.excluded_quarantined_count == 1
    candidate_projection = next(
        item.projection_id
        for item in artifact.postwrite_verification.write_artifact.preflight_artifact.preflight_items
        if item.source_candidate.candidate_id == candidate.candidate_id
    )
    assert all(
        observation.projection_id != candidate_projection
        for group in artifact.review_groups
        for observation in group.observations
    )


def test_review_observation_preserves_reported_precision_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path = _verification_result(tmp_path, monkeypatch)
    artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=tmp_path / "precision-facet-request.json",
        generated_at=_REQUEST_AT,
    )
    by_text = {
        observation.reported_value_text: observation
        for observation in artifact.review_groups[0].observations
    }

    assert by_text["-1.70"].reported_decimal_places == 2
    assert by_text["-1.70"].source_pdf_sha256.startswith("sha256:")
    assert by_text["-1.70"].table_id
    assert by_text["-1.70"].column_name
    assert by_text["-1.70"].selected_registry_entry.material_id == (
        by_text["-1.70"].selected_material_id
    )


def test_group_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path = _verification_result(tmp_path, monkeypatch)
    artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=tmp_path / "facet-request.json",
        generated_at=_REQUEST_AT,
    )
    forged = artifact.model_copy(
        update={
            "eligible_observation_count": 4,
            "request_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "request_artifact_digest": (
                oled_reviewed_evidence_facet_review_request_artifact_digest(
                    forged
                )
            )
        }
    )

    with pytest.raises(ValidationError, match="eligible_observation_count"):
        OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_request_timestamp_reversal_and_input_overwrite_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path = _verification_result(tmp_path, monkeypatch)
    original = verification_path.read_bytes()

    with pytest.raises(ValueError, match="predates PR-T"):
        build_oled_reviewed_evidence_facet_review_request_from_files(
            postwrite_verification_json=verification_path,
            output_json=tmp_path / "early-request.json",
            generated_at="2026-07-14T00:43:59+08:00",
        )
    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_reviewed_evidence_facet_review_request_from_files(
            postwrite_verification_json=verification_path,
            output_json=verification_path,
            generated_at=_REQUEST_AT,
        )

    assert verification_path.read_bytes() == original


def test_request_cli_failure_is_redacted(tmp_path: Path) -> None:
    secret = "secret-facet-review-token"
    output_path = tmp_path / "output.json"
    stream = StringIO()

    code = main(
        [
            "--postwrite-verification",
            str(tmp_path / f"missing-{secret}.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    payload = json.loads(stream.getvalue())
    assert code == 2
    assert payload["error_code"] == "reviewed_evidence_facet_review_request_failed"
    assert secret not in stream.getvalue()
    assert not output_path.exists()
