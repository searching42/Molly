from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledConfidenceSufficiencyDecision,
    OledReviewedEvidenceFacetAdjudicationArtifact,
    OledReviewedEvidenceFacetAdjudicationStatus,
    OledReviewedEvidenceFacetDecisionEntry,
    OledReviewedEvidenceFacetDecisionManifest,
    OledScientificConsistencyDecision,
    oled_reviewed_evidence_facet_adjudication_artifact_digest,
)
from ai4s_agent.oled_reviewed_evidence_facet_adjudication import (
    build_oled_reviewed_evidence_facet_adjudication_from_files,
    main,
)
from ai4s_agent.oled_reviewed_evidence_facet_review_request import (
    build_oled_reviewed_evidence_facet_review_request_from_files,
)
from tests.test_oled_reviewed_evidence_facet_review_request import (
    _verification_result,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_REQUEST_AT = "2026-07-14T00:45:00+08:00"
_REVIEW_AT = "2026-07-14T00:46:00+08:00"
_ADJUDICATION_AT = "2026-07-14T00:47:00+08:00"


def _request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, verification_path = _verification_result(tmp_path, monkeypatch)
    request_path = tmp_path / "facet-review-request.json"
    request = build_oled_reviewed_evidence_facet_review_request_from_files(
        postwrite_verification_json=verification_path,
        output_json=request_path,
        generated_at=_REQUEST_AT,
    )
    return request, request_path


def _manifest(
    request,
    request_path: Path,
    *,
    scientific: OledScientificConsistencyDecision = (
        OledScientificConsistencyDecision.CONSISTENT
    ),
    confidence: OledConfidenceSufficiencyDecision = (
        OledConfidenceSufficiencyDecision.SUFFICIENT
    ),
) -> OledReviewedEvidenceFacetDecisionManifest:
    decisions = []
    for group in request.review_groups:
        for observation in group.observations:
            decisions.append(
                OledReviewedEvidenceFacetDecisionEntry(
                    review_group_id=group.review_group_id,
                    group_digest=group.group_digest,
                    entry_id=observation.entry_id,
                    observation_digest=observation.observation_digest,
                    scientific_consistency=scientific,
                    confidence_sufficiency=confidence,
                    review_note="Reviewer checked the exact source-bound observation.",
                )
            )
    return OledReviewedEvidenceFacetDecisionManifest(
        run_id=request.run_id,
        paper_id=request.paper_id,
        request_artifact_sha256=_sha256_file(request_path),
        request_artifact_digest=request.request_artifact_digest,
        postwrite_verification_sha256=request.postwrite_verification_sha256,
        postwrite_verification_digest=request.postwrite_verification_digest,
        reviewed_by="Benton",
        reviewed_at=_REVIEW_AT,
        adjudication_confirmed=True,
        decisions=sorted(decisions, key=lambda item: item.entry_id),
    )


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest=None,
):
    request, request_path = _request(tmp_path, monkeypatch)
    selected = manifest or _manifest(request, request_path)
    decision_path = tmp_path / "facet-decisions.json"
    write_json(decision_path, selected.model_dump(mode="json"))
    output_path = tmp_path / "facet-adjudication.json"
    artifact = build_oled_reviewed_evidence_facet_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=output_path,
        generated_at=_ADJUDICATION_AT,
    )
    return request, selected, artifact, request_path, decision_path, output_path


def test_all_accepted_decisions_become_gold_preflight_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _, artifact, _, decision_path, output_path = _build(
        tmp_path,
        monkeypatch,
    )

    assert artifact.status == (
        OledReviewedEvidenceFacetAdjudicationStatus
        .READY_FOR_GOLD_ADMISSION_PREFLIGHT
    )
    assert artifact.review_group_count == request.review_group_count == 1
    assert artifact.reviewed_observation_count == 5
    assert artifact.scientific_consistent_count == 5
    assert artifact.confidence_sufficient_count == 5
    assert artifact.gold_admission_preflight_eligible_count == 5
    assert artifact.blocked_observation_count == 0
    assert artifact.decision_manifest_sha256 == _sha256_file(decision_path)
    assert all(
        item.retained_gold_blocker_codes == []
        and item.eligible_for_gold_admission_preflight
        and not item.gold_record_created
        for item in artifact.adjudicated_observations
    )
    assert OledReviewedEvidenceFacetAdjudicationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_mixed_decisions_retain_specific_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decisions = list(manifest.decisions)
    decisions[0] = decisions[0].model_copy(
        update={
            "scientific_consistency": (
                OledScientificConsistencyDecision.INCONSISTENT
            ),
            "confidence_sufficiency": (
                OledConfidenceSufficiencyDecision.INSUFFICIENT
            ),
        }
    )
    decisions[1] = decisions[1].model_copy(
        update={
            "scientific_consistency": (
                OledScientificConsistencyDecision.NEEDS_SOURCE_CHECK
            ),
        }
    )
    manifest = manifest.model_copy(update={"decisions": decisions})
    decision_path = tmp_path / "mixed-decisions.json"
    write_json(decision_path, manifest.model_dump(mode="json"))

    artifact = build_oled_reviewed_evidence_facet_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=tmp_path / "mixed-adjudication.json",
        generated_at=_ADJUDICATION_AT,
    )

    assert artifact.status == (
        OledReviewedEvidenceFacetAdjudicationStatus
        .REVIEW_COMPLETE_WITH_BLOCKED_EVIDENCE
    )
    assert artifact.gold_admission_preflight_eligible_count == 3
    assert artifact.blocked_observation_count == 2
    by_entry = {
        item.request_observation.entry_id: item
        for item in artifact.adjudicated_observations
    }
    assert by_entry[decisions[0].entry_id].retained_gold_blocker_codes == [
        "confidence_evidence_insufficient",
        "scientific_consistency_inconsistent",
    ]
    assert by_entry[decisions[1].entry_id].retained_gold_blocker_codes == [
        "scientific_consistency_source_check_required"
    ]
    assert by_entry[decisions[1].entry_id].source_check_required


@pytest.mark.parametrize("drop_last", [True, False])
def test_partial_or_extra_decision_roster_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop_last: bool,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decisions = list(manifest.decisions)
    if drop_last:
        decisions.pop()
    else:
        decisions.append(
            decisions[-1].model_copy(
                update={
                    "entry_id": "reviewed-evidence-entry:unexpected",
                }
            )
        )
        decisions.sort(key=lambda item: item.entry_id)
    decision_path = tmp_path / "bad-roster.json"
    write_json(
        decision_path,
        manifest.model_copy(update={"decisions": decisions}).model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="coverage"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "bad-roster-output.json",
            generated_at=_ADJUDICATION_AT,
        )


@pytest.mark.parametrize(
    "field",
    ["review_group_id", "group_digest", "observation_digest"],
)
def test_stale_group_or_observation_binding_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decisions = list(manifest.decisions)
    replacement = (
        "reviewed-evidence-facet-review-group:stale"
        if field == "review_group_id"
        else "sha256:" + "1" * 64
    )
    decisions[0] = decisions[0].model_copy(update={field: replacement})
    decision_path = tmp_path / f"stale-{field}.json"
    write_json(
        decision_path,
        manifest.model_copy(update={"decisions": decisions}).model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="binding"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / f"stale-{field}-output.json",
            generated_at=_ADJUDICATION_AT,
        )


def test_request_reformatting_is_rejected_by_exact_sha_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decision_path = tmp_path / "decisions.json"
    write_json(decision_path, manifest.model_dump(mode="json"))
    request_path.write_text(
        json.dumps(request.model_dump(mode="json"), separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="request_artifact_sha256"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "reformatted-output.json",
            generated_at=_ADJUDICATION_AT,
        )


def test_input_overwrite_and_symbolic_input_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decision_path = tmp_path / "decisions.json"
    write_json(decision_path, manifest.model_dump(mode="json"))

    with pytest.raises(ValueError, match="overwrite|distinct"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=request_path,
            generated_at=_ADJUDICATION_AT,
        )

    request_link = tmp_path / "request-link.json"
    request_link.symlink_to(request_path)
    with pytest.raises(ValueError, match="symbolic|symlink"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_link,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "symlink-output.json",
            generated_at=_ADJUDICATION_AT,
        )


def test_review_timestamp_before_request_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path).model_copy(
        update={"reviewed_at": "2026-07-14T00:44:59+08:00"}
    )
    decision_path = tmp_path / "early-decisions.json"
    write_json(decision_path, manifest.model_dump(mode="json"))

    with pytest.raises(ValueError, match="predates"):
        build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decision_path,
            output_json=tmp_path / "early-output.json",
            generated_at=_ADJUDICATION_AT,
        )


def test_outer_rehash_cannot_hide_eligibility_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, artifact, _, _, _ = _build(tmp_path, monkeypatch)
    first = artifact.adjudicated_observations[0].model_copy(
        update={"eligible_for_gold_admission_preflight": False}
    )
    forged = artifact.model_copy(
        update={
            "adjudicated_observations": [
                first,
                *artifact.adjudicated_observations[1:],
            ],
            "adjudication_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "adjudication_artifact_digest": (
                oled_reviewed_evidence_facet_adjudication_artifact_digest(forged)
            )
        }
    )

    with pytest.raises(ValidationError, match="eligible"):
        OledReviewedEvidenceFacetAdjudicationArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_cli_redacts_validation_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path).model_copy(
        update={"request_artifact_digest": "sha256:" + "2" * 64}
    )
    decision_path = tmp_path / "invalid-decisions.json"
    write_json(decision_path, manifest.model_dump(mode="json"))
    stream = StringIO()

    exit_code = main(
        [
            "--request-artifact",
            str(request_path),
            "--decision-manifest",
            str(decision_path),
            "--output",
            str(tmp_path / "cli-output.json"),
        ],
        stdout=stream,
    )

    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "reviewed_evidence_facet_adjudication_failed",
        "error_type": "ValueError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    domain_path = (
        Path(__file__).parents[1]
        / "src"
        / "ai4s_agent"
        / "domains"
        / "oled_reviewed_evidence_facet_adjudication.py"
    )
    runner_path = (
        Path(__file__).parents[1]
        / "src"
        / "ai4s_agent"
        / "oled_reviewed_evidence_facet_adjudication.py"
    )
    assert "paper016" not in domain_path.read_text(encoding="utf-8")
    assert "paper016" not in runner_path.read_text(encoding="utf-8")
