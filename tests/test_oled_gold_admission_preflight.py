from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_gold_admission_preflight import (
    OledGoldAdmissionPreflightArtifact,
    OledGoldAdmissionPreflightStatus,
    oled_gold_admission_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledConfidenceSufficiencyDecision,
    OledScientificConsistencyDecision,
)
from ai4s_agent.oled_gold_admission_preflight import (
    build_oled_gold_admission_preflight_from_files,
    main,
)
from ai4s_agent.oled_reviewed_evidence_facet_adjudication import (
    build_oled_reviewed_evidence_facet_adjudication_from_files,
)
from tests.test_oled_reviewed_evidence_facet_adjudication import (
    _ADJUDICATION_AT,
    _build,
    _manifest,
    _request,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_PREFLIGHT_AT = "2026-07-14T00:48:00+08:00"


def _all_accepted_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, adjudication, _, _, adjudication_path = _build(tmp_path, monkeypatch)
    return adjudication, adjudication_path


def _mixed_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    block_all: bool = False,
):
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decisions = list(manifest.decisions)
    if block_all:
        decisions = [
            decision.model_copy(
                update={
                    "scientific_consistency": (
                        OledScientificConsistencyDecision.INCONSISTENT
                    ),
                    "confidence_sufficiency": (
                        OledConfidenceSufficiencyDecision.INSUFFICIENT
                    ),
                }
            )
            for decision in decisions
        ]
    else:
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
    decision_path = tmp_path / "facet-decisions-mixed.json"
    write_json(decision_path, manifest.model_dump(mode="json"))
    adjudication_path = tmp_path / "facet-adjudication-mixed.json"
    adjudication = build_oled_reviewed_evidence_facet_adjudication_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decision_path,
        output_json=adjudication_path,
        generated_at=_ADJUDICATION_AT,
    )
    return adjudication, adjudication_path


def test_all_accepted_facets_build_exact_gold_admission_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication, adjudication_path = _all_accepted_adjudication(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "gold-admission-preflight.json"

    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status == OledGoldAdmissionPreflightStatus.CANDIDATES_READY
    assert artifact.source_reviewed_observation_count == 5
    assert artifact.eligible_candidate_count == 5
    assert artifact.blocked_observation_count == 0
    assert artifact.facet_adjudication_sha256 == _sha256_file(adjudication_path)
    assert artifact.facet_adjudication == adjudication
    assert all(
        candidate.scientific_consistency == "consistent"
        and candidate.confidence_sufficiency == "sufficient"
        and candidate.evidence_refs
        and candidate.selected_registry_entry.material_id
        == candidate.selected_material_id
        and candidate.categorical_confidence_only
        and not candidate.numeric_confidence_score_assigned
        and not candidate.legacy_numeric_confidence_record_constructed
        and not candidate.gold_record_created
        for candidate in artifact.candidates
    )
    assert OledGoldAdmissionPreflightArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_mixed_adjudication_excludes_blocked_evidence_and_counts_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _mixed_adjudication(tmp_path, monkeypatch)

    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=tmp_path / "mixed-gold-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status == (
        OledGoldAdmissionPreflightStatus.PARTIAL_WITH_BLOCKED_EVIDENCE
    )
    assert artifact.eligible_candidate_count == 3
    assert artifact.blocked_observation_count == 2
    assert artifact.blocked_scientific_inconsistent_count == 1
    assert artifact.blocked_scientific_source_check_count == 1
    assert artifact.blocked_confidence_insufficient_count == 1
    assert artifact.blocked_confidence_source_check_count == 0
    candidate_entries = {candidate.source_entry_id for candidate in artifact.candidates}
    blocked_entries = {
        item.request_observation.entry_id
        for item in artifact.facet_adjudication.adjudicated_observations
        if not item.eligible_for_gold_admission_preflight
    }
    assert candidate_entries.isdisjoint(blocked_entries)


def test_no_eligible_evidence_emits_empty_preflight_not_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _mixed_adjudication(
        tmp_path,
        monkeypatch,
        block_all=True,
    )

    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=tmp_path / "empty-gold-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status == OledGoldAdmissionPreflightStatus.NO_ELIGIBLE_EVIDENCE
    assert artifact.eligible_candidate_count == 0
    assert artifact.blocked_observation_count == 5
    assert artifact.candidates == []
    assert not artifact.gold_records_created
    assert not artifact.gold_published


def test_reported_precision_registry_and_source_provenance_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _all_accepted_adjudication(tmp_path, monkeypatch)
    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=tmp_path / "precision-gold-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )
    by_text = {candidate.reported_value_text: candidate for candidate in artifact.candidates}

    candidate = by_text["-1.70"]
    assert candidate.reported_decimal_places == 2
    assert candidate.source_pdf_sha256.startswith("sha256:")
    assert candidate.table_id
    assert candidate.source_cell_digest.startswith("sha256:")
    assert candidate.selected_registry_entry.entry_digest == candidate.registry_entry_digest
    assert candidate.registry_entry_payload_digest != candidate.registry_entry_digest


def test_reformatted_adjudication_fails_exact_sha_binding_via_embedded_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication, adjudication_path = _all_accepted_adjudication(
        tmp_path,
        monkeypatch,
    )
    reformatted = tmp_path / "facet-adjudication-reformatted.json"
    reformatted.write_text(
        json.dumps(adjudication.model_dump(mode="json"), separators=(",", ":")),
        encoding="utf-8",
    )

    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=reformatted,
        output_json=tmp_path / "reformatted-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.facet_adjudication_sha256 == _sha256_file(reformatted)
    assert artifact.facet_adjudication_sha256 != _sha256_file(adjudication_path)
    assert artifact.facet_adjudication_digest == adjudication.adjudication_artifact_digest


def test_candidate_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _all_accepted_adjudication(tmp_path, monkeypatch)
    artifact = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=tmp_path / "gold-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )
    first = artifact.candidates[0].model_copy(
        update={"reported_value_text": "tampered"}
    )
    forged = artifact.model_copy(
        update={
            "candidates": [first, *artifact.candidates[1:]],
            "preflight_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_gold_admission_preflight_artifact_digest(forged)
            )
        }
    )

    with pytest.raises(ValidationError, match="candidate"):
        OledGoldAdmissionPreflightArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_timestamp_reversal_input_overwrite_and_symlink_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _all_accepted_adjudication(tmp_path, monkeypatch)

    with pytest.raises(ValidationError, match="predates"):
        build_oled_gold_admission_preflight_from_files(
            facet_adjudication_json=adjudication_path,
            output_json=tmp_path / "early-preflight.json",
            generated_at="2026-07-14T00:46:59+08:00",
        )
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_gold_admission_preflight_from_files(
            facet_adjudication_json=adjudication_path,
            output_json=adjudication_path,
            generated_at=_PREFLIGHT_AT,
        )
    link = tmp_path / "facet-adjudication-link.json"
    link.symlink_to(adjudication_path)
    with pytest.raises(ValueError, match="symbolic|symlink"):
        build_oled_gold_admission_preflight_from_files(
            facet_adjudication_json=link,
            output_json=tmp_path / "link-preflight.json",
            generated_at=_PREFLIGHT_AT,
        )


def test_cli_redacts_failure_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adjudication_path = _all_accepted_adjudication(tmp_path, monkeypatch)
    output_path = tmp_path / "existing.json"
    output_path.write_text("{}\n", encoding="utf-8")
    stream = StringIO()

    exit_code = main(
        [
            "--facet-adjudication",
            str(adjudication_path),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_admission_preflight_failed",
        "error_type": "ValueError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = root / "src/ai4s_agent/domains/oled_gold_admission_preflight.py"
    runner = root / "src/ai4s_agent/oled_gold_admission_preflight.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
