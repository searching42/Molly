from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_locator_adjudication import (
    OledSupplementaryLocatorAdjudicationArtifact,
    OledSupplementaryLocatorDecisionEntry,
    OledSupplementaryLocatorDecisionManifest,
)
from ai4s_agent.oled_supplementary_locator_adjudication import (
    adjudicate_oled_supplementary_locator_from_files,
    main,
)


_GENERATED_AT = "2026-07-13T14:00:00+08:00"
_SEMANTIC_NOTE = "HOMO/LUMO labels are preserved as reported but require semantic review"


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _review_table_payload(table_id: str = "table_p38_0178") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "table_id": table_id,
        "page": 38,
        "caption": "Supplementary Table S1. Photophysical properties",
        "headers": ["Emitter", "HOMO (eV)", "LUMO (eV)"],
        "rows": [
            {"Emitter": "Compound 1", "HOMO (eV)": "-1.59", "LUMO (eV)": "-5.49"}
        ],
        "footnotes": ["Values are preserved as reported."],
        "source_bbox": {"x0": 10.0, "y0": 20.0, "x1": 500.0, "y1": 700.0},
        "row_count": 1,
        "column_count": 3,
        "table_content_digest": "",
    }
    payload["table_content_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "table_content_digest"}
    )
    return payload


def _review_item_payload(
    item_number: int = 1,
    *,
    match_status: str = "exact_match",
) -> dict[str, Any]:
    table_id = f"table_p38_{177 + item_number:04d}"
    item: dict[str, Any] = {
        "review_item_id": f"supplementary-locator-review:supplementary-recovery:item-{item_number:03d}",
        "recovery_item_id": f"supplementary-recovery:item-{item_number:03d}",
        "source_id": "supp-source-001",
        "source_pdf_sha256": "sha256:" + "1" * 64,
        "parsed_document_sha256": "sha256:" + "2" * 64,
        "parser_backend": "mineru_api:hybrid-engine",
        "target_kind": "table",
        "target_locator": f"S{item_number}",
        "canonical_locator": f"S{item_number}",
        "match_status": match_status,
        "candidate_table_ids": [],
        "matched_table": None,
        "parser_warning_codes": [],
        "review_decision": "pending",
        "review_guidance": "Review the bound locator against the source.",
    }
    if match_status == "exact_match":
        table = _review_table_payload(table_id)
        table["caption"] = f"Supplementary Table S{item_number}. Photophysical properties"
        table["table_content_digest"] = _stable_hash(
            {key: value for key, value in table.items() if key != "table_content_digest"}
        )
        item["candidate_table_ids"] = [table_id]
        item["matched_table"] = table
    elif match_status == "ambiguous":
        item["candidate_table_ids"] = [f"{table_id}-a", f"{table_id}-b"]
    elif match_status == "unsupported_target_kind":
        item["target_kind"] = "figure"
        item["canonical_locator"] = ""
    elif match_status == "unsupported_locator_format":
        item["canonical_locator"] = ""
    return item


def _review_payload(
    *,
    statuses: list[str] | None = None,
) -> dict[str, Any]:
    selected_statuses = statuses or ["exact_match"]
    items = [
        _review_item_payload(index, match_status=status)
        for index, status in enumerate(selected_statuses, start=1)
    ]
    items.sort(key=lambda item: item["review_item_id"])
    exact_count = sum(item["match_status"] == "exact_match" for item in items)
    all_resolved = exact_count == len(items)
    payload: dict[str, Any] = {
        "artifact_version": "oled_supplementary_locator_review.v1",
        "run_id": "supp-mineru-run-001",
        "paper_id": "paper016",
        "generated_at": _GENERATED_AT,
        "execution_artifact_sha256": "sha256:" + "3" * 64,
        "execution_artifact_digest": "sha256:" + "4" * 64,
        "locator_manifest_sha256": "sha256:" + "5" * 64,
        "preflight_plan_digest": "sha256:" + "6" * 64,
        "endpoint_profile_name": "node45-loopback",
        "backend": "hybrid-engine",
        "status": (
            "ready_for_human_review"
            if all_resolved
            else "manual_locator_review_required"
        ),
        "source_count": 1,
        "item_count": len(items),
        "exact_match_count": exact_count,
        "unresolved_item_count": len(items) - exact_count,
        "review_items": items,
        "review_artifact_digest": "",
        "review_only": True,
        "human_review_required": True,
        "offline_only": True,
        "scientific_content_included": True,
        "parsed_output_read": True,
        "locator_resolution_attempted": True,
        "locator_resolved": all_resolved,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "pdf_content_read": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["review_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "review_artifact_digest"}
    )
    return payload


def _decision_payload(
    review_payload: dict[str, Any],
    review_path: Path,
    *,
    decisions: list[str] | None = None,
    semantic_notes: list[str] | None = None,
) -> dict[str, Any]:
    selected_decisions = decisions or ["accept_locator"] * len(review_payload["review_items"])
    selected_notes = semantic_notes or [""] * len(review_payload["review_items"])
    entries = []
    for item, decision, semantic_note in zip(
        review_payload["review_items"],
        selected_decisions,
        selected_notes,
        strict=True,
    ):
        entries.append(
            {
                "review_item_id": item["review_item_id"],
                "decision": decision,
                "reviewed_by": "Benton",
                "reviewed_at": _GENERATED_AT,
                "review_note": (
                    "Source must be checked before this locator can be used."
                    if decision != "accept_locator"
                    else ""
                ),
                "semantic_note": semantic_note,
            }
        )
    return {
        "schema_version": "oled_supplementary_locator_decision_manifest.v1",
        "run_id": review_payload["run_id"],
        "paper_id": review_payload["paper_id"],
        "review_artifact_sha256": _sha256_file(review_path),
        "review_artifact_digest": review_payload["review_artifact_digest"],
        "adjudication_confirmed": True,
        "decisions": entries,
    }


def _write_inputs(
    tmp_path: Path,
    *,
    statuses: list[str] | None = None,
    decisions: list[str] | None = None,
    semantic_notes: list[str] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    review_payload = _review_payload(statuses=statuses)
    review_path = tmp_path / "locator-review.json"
    write_json(review_path, review_payload)
    decision_payload = _decision_payload(
        review_payload,
        review_path,
        decisions=decisions,
        semantic_notes=semantic_notes,
    )
    decision_path = tmp_path / "locator-decisions.json"
    write_json(decision_path, decision_payload)
    return (
        review_path,
        decision_path,
        tmp_path / "locator-adjudication.json",
        review_payload,
        decision_payload,
    )


def _adjudicate(
    paths: tuple[Path, Path, Path, dict[str, Any], dict[str, Any]],
) -> OledSupplementaryLocatorAdjudicationArtifact:
    review_path, decision_path, output_path, _, _ = paths
    return adjudicate_oled_supplementary_locator_from_files(
        review_artifact_json=review_path,
        decision_manifest_json=decision_path,
        output_json=output_path,
        generated_at=_GENERATED_AT,
    )


def test_accept_locator_preserves_semantic_note_without_validating_semantics(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, semantic_notes=[_SEMANTIC_NOTE])

    artifact = _adjudicate(paths)

    assert artifact.status.value == "all_locators_accepted"
    assert artifact.accepted_count == 1
    assert artifact.semantic_review_required_count == 1
    assert artifact.candidate_proposal_eligible_count == 1
    assert artifact.physical_semantics_validated is False
    assert artifact.table_transcription_validated is False
    assert artifact.scientific_content_validated is False
    assert artifact.candidate_regenerated is False
    assert artifact.reviewed_evidence_staging is False
    assert artifact.device_only_admitted is False
    assert artifact.gold_records_created is False
    assert artifact.dataset_written is False
    item = artifact.adjudicated_items[0]
    assert item.semantic_note == _SEMANTIC_NOTE
    assert item.semantic_review_required is True
    assert item.locator_accepted is True
    assert item.eligible_for_later_scoped_candidate_proposal is True
    assert item.direct_admission_eligible is False
    assert item.evidence_content_mutated is False
    output_text = paths[2].read_text(encoding="utf-8")
    assert "Compound 1" not in output_text
    assert "Photophysical properties" not in output_text
    assert '"rows"' not in output_text
    assert '"caption"' not in output_text
    assert _SEMANTIC_NOTE in output_text
    OledSupplementaryLocatorAdjudicationArtifact.model_validate_json(output_text)


@pytest.mark.parametrize(
    "match_status",
    [
        "not_found",
        "ambiguous",
        "unsupported_target_kind",
        "unsupported_locator_format",
    ],
)
def test_accept_locator_requires_an_exact_match(tmp_path: Path, match_status: str) -> None:
    paths = _write_inputs(tmp_path, statuses=[match_status])

    with pytest.raises(ValueError, match="only exact supplementary locator matches may be accepted"):
        _adjudicate(paths)

    assert not paths[2].exists()


def test_decisions_must_cover_every_review_item_exactly_once(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, statuses=["exact_match", "exact_match"])
    decision_payload = deepcopy(paths[4])
    decision_payload["decisions"] = decision_payload["decisions"][:1]
    write_json(paths[1], decision_payload)

    with pytest.raises(ValueError, match="exactly cover review items"):
        _adjudicate(paths)

    decision_payload = deepcopy(paths[4])
    decision_payload["decisions"][1]["review_item_id"] = "supplementary-locator-review:unknown"
    write_json(paths[1], decision_payload)
    with pytest.raises(ValueError, match="exactly cover review items"):
        _adjudicate(paths)


def test_duplicate_decision_ids_are_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, statuses=["exact_match", "exact_match"])
    decision_payload = deepcopy(paths[4])
    decision_payload["decisions"][1]["review_item_id"] = decision_payload["decisions"][0][
        "review_item_id"
    ]
    write_json(paths[1], decision_payload)

    with pytest.raises(ValidationError, match="duplicate supplementary locator"):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("run_id", "different-run", "identity does not match"),
        ("paper_id", "paper999", "identity does not match"),
        ("review_artifact_sha256", "sha256:" + "a" * 64, "exact review artifact bytes"),
        ("review_artifact_digest", "sha256:" + "b" * 64, "canonical review content"),
    ],
)
def test_manifest_must_bind_review_identity_bytes_and_content(
    tmp_path: Path,
    field: str,
    replacement: str,
    error: str,
) -> None:
    paths = _write_inputs(tmp_path)
    decision_payload = deepcopy(paths[4])
    decision_payload[field] = replacement
    write_json(paths[1], decision_payload)

    with pytest.raises(ValueError, match=error):
        _adjudicate(paths)


def test_canonically_equivalent_review_reserialization_breaks_exact_byte_binding(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    prior_sha256 = _sha256_file(paths[0])
    paths[0].write_text(
        json.dumps(paths[3], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert _sha256_file(paths[0]) != prior_sha256

    with pytest.raises(ValueError, match="exact review artifact bytes"):
        _adjudicate(paths)


@pytest.mark.parametrize("decision", ["reject_locator", "needs_source_check"])
def test_non_accept_decisions_require_a_review_note(decision: str) -> None:
    with pytest.raises(ValidationError, match="require review_note"):
        OledSupplementaryLocatorDecisionEntry(
            review_item_id="supplementary-locator-review:item-001",
            decision=decision,
            reviewed_by="Benton",
            reviewed_at=_GENERATED_AT,
        )


def test_mixed_legal_decisions_return_partial_status_and_do_not_admit_data(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(
        tmp_path,
        statuses=["exact_match", "exact_match", "not_found"],
        decisions=["accept_locator", "reject_locator", "needs_source_check"],
        semantic_notes=[_SEMANTIC_NOTE, "", ""],
    )

    artifact = _adjudicate(paths)

    assert artifact.status.value == "partially_accepted"
    assert artifact.accepted_count == 1
    assert artifact.rejected_count == 1
    assert artifact.needs_source_check_count == 1
    assert artifact.candidate_proposal_eligible_count == 1
    assert artifact.dataset_written is False


def test_no_accepted_locators_is_a_valid_completed_adjudication(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        statuses=["not_found", "ambiguous"],
        decisions=["needs_source_check", "reject_locator"],
    )

    artifact = _adjudicate(paths)

    assert artifact.status.value == "no_locators_accepted"
    assert artifact.accepted_count == 0
    assert artifact.candidate_proposal_eligible_count == 0
    assert artifact.adjudication_complete is True


@pytest.mark.parametrize("extra_field", ["corrected_locator", "proposed_page", "corrected_value"])
def test_decisions_forbid_corrections_and_proposals(tmp_path: Path, extra_field: str) -> None:
    paths = _write_inputs(tmp_path)
    decision_payload = deepcopy(paths[4])
    decision_payload["decisions"][0][extra_field] = "forbidden"
    write_json(paths[1], decision_payload)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _adjudicate(paths)


def test_semantic_note_allows_scientific_slashes_but_rejects_paths_and_urls() -> None:
    decision = OledSupplementaryLocatorDecisionEntry(
        review_item_id="supplementary-locator-review:item-001",
        decision="accept_locator",
        reviewed_by="Benton",
        reviewed_at=_GENERATED_AT,
        semantic_note=_SEMANTIC_NOTE,
    )
    assert decision.semantic_note == _SEMANTIC_NOTE

    for unsafe_note in (
        "Inspect /operator/private/source.pdf",
        "Inspect (/operator/private/source.pdf)",
        r"Inspect \\server\share\source.pdf",
        "Inspect https://example.invalid/source",
        "Inspect file:///operator/private/source.pdf",
    ):
        with pytest.raises(ValidationError, match="URL or absolute path"):
            OledSupplementaryLocatorDecisionEntry(
                review_item_id="supplementary-locator-review:item-001",
                decision="accept_locator",
                reviewed_by="Benton",
                reviewed_at=_GENERATED_AT,
                semantic_note=unsafe_note,
            )

    with pytest.raises(ValidationError, match="control characters"):
        OledSupplementaryLocatorDecisionEntry(
            review_item_id="supplementary-locator-review:item-001",
            decision="accept_locator",
            reviewed_by="Benton",
            reviewed_at=_GENERATED_AT,
            semantic_note="Line one\nLine two",
        )


@pytest.mark.parametrize("input_index", [0, 1])
def test_symlink_inputs_are_rejected(tmp_path: Path, input_index: int) -> None:
    paths = _write_inputs(tmp_path)
    original_path = paths[input_index]
    symlink_path = tmp_path / f"linked-{input_index}.json"
    symlink_path.symlink_to(original_path)
    arguments = [paths[0], paths[1]]
    arguments[input_index] = symlink_path

    with pytest.raises(ValueError, match="input is unavailable"):
        adjudicate_oled_supplementary_locator_from_files(
            review_artifact_json=arguments[0],
            decision_manifest_json=arguments[1],
            output_json=paths[2],
            generated_at=_GENERATED_AT,
        )


@pytest.mark.parametrize("collision_index", [0, 1])
def test_output_must_not_overwrite_either_input(tmp_path: Path, collision_index: int) -> None:
    paths = _write_inputs(tmp_path)
    protected_path = paths[collision_index]
    before = protected_path.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        adjudicate_oled_supplementary_locator_from_files(
            review_artifact_json=paths[0],
            decision_manifest_json=paths[1],
            output_json=protected_path,
            generated_at=_GENERATED_AT,
        )

    assert protected_path.read_bytes() == before


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths[2].write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output must be fresh"):
        _adjudicate(paths)

    assert paths[2].read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.parametrize("tamper", ["table", "item", "artifact"])
def test_review_content_digest_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    paths = _write_inputs(tmp_path)
    review_payload = deepcopy(paths[3])
    if tamper == "table":
        review_payload["review_items"][0]["matched_table"]["rows"][0]["HOMO (eV)"] = "-9.99"
    elif tamper == "item":
        review_payload["review_items"][0]["target_locator"] = "S99"
    else:
        review_payload["backend"] = "different-backend"
    write_json(paths[0], review_payload)
    decision_payload = deepcopy(paths[4])
    decision_payload["review_artifact_sha256"] = _sha256_file(paths[0])
    write_json(paths[1], decision_payload)

    with pytest.raises(ValidationError, match="digest mismatch"):
        _adjudicate(paths)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("accepted_count", 0, "accepted_count mismatch"),
        ("physical_semantics_validated", True, "downstream boundary"),
        ("candidate_regenerated", True, "downstream boundary"),
    ],
)
def test_output_integrity_rejects_count_and_boundary_flag_tampering(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    paths = _write_inputs(tmp_path, semantic_notes=[_SEMANTIC_NOTE])
    _adjudicate(paths)
    payload = json.loads(paths[2].read_text(encoding="utf-8"))
    payload[field] = value
    payload["adjudication_artifact_digest"] = _stable_hash(
        {key: item for key, item in payload.items() if key != "adjudication_artifact_digest"}
    )

    with pytest.raises(ValidationError, match=error):
        OledSupplementaryLocatorAdjudicationArtifact.model_validate(payload)


def test_cli_redacts_paths_notes_and_table_content_on_success_and_failure(tmp_path: Path) -> None:
    success_paths = _write_inputs(tmp_path / "success", semantic_notes=[_SEMANTIC_NOTE])
    success_stdout = StringIO()

    success_code = main(
        [
            "--review-artifact",
            str(success_paths[0]),
            "--decision-manifest",
            str(success_paths[1]),
            "--output",
            str(success_paths[2]),
        ],
        stdout=success_stdout,
    )

    assert success_code == 0
    success_text = success_stdout.getvalue()
    assert "all_locators_accepted" in success_text
    assert str(tmp_path) not in success_text
    assert _SEMANTIC_NOTE not in success_text
    assert "Compound 1" not in success_text

    failure_stdout = StringIO()
    failure_code = main(
        [
            "--review-artifact",
            str(tmp_path / "private-review.json"),
            "--decision-manifest",
            str(success_paths[1]),
            "--output",
            str(tmp_path / "private-output.json"),
        ],
        stdout=failure_stdout,
    )
    assert failure_code == 2
    failure_text = failure_stdout.getvalue()
    assert str(tmp_path) not in failure_text
    assert "supplementary_locator_adjudication_failed" in failure_text


def test_cli_returns_success_for_a_completed_rejection(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        statuses=["not_found"],
        decisions=["reject_locator"],
    )
    stdout = StringIO()

    exit_code = main(
        [
            "--review-artifact",
            str(paths[0]),
            "--decision-manifest",
            str(paths[1]),
            "--output",
            str(paths[2]),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "no_locators_accepted" in stdout.getvalue()


def test_manifest_confirmation_and_timezone_are_required(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    decision_payload = deepcopy(paths[4])
    decision_payload["adjudication_confirmed"] = False
    with pytest.raises(ValidationError, match="adjudication_confirmed=true"):
        OledSupplementaryLocatorDecisionManifest.model_validate(decision_payload)

    decision_payload = deepcopy(paths[4])
    decision_payload["decisions"][0]["reviewed_at"] = "2026-07-13T14:00:00"
    with pytest.raises(ValidationError, match="include a timezone"):
        OledSupplementaryLocatorDecisionManifest.model_validate(decision_payload)


@pytest.mark.parametrize("coercible_value", ["true", "yes", "on", 1])
def test_manifest_confirmation_requires_a_literal_json_boolean(
    tmp_path: Path,
    coercible_value: Any,
) -> None:
    paths = _write_inputs(tmp_path)
    decision_payload = deepcopy(paths[4])
    decision_payload["adjudication_confirmed"] = coercible_value

    with pytest.raises(ValidationError, match="valid boolean"):
        OledSupplementaryLocatorDecisionManifest.model_validate(decision_payload)


@pytest.mark.parametrize("input_kind", ["review", "decision"])
def test_duplicate_json_object_keys_are_rejected(tmp_path: Path, input_kind: str) -> None:
    paths = _write_inputs(tmp_path)
    target_path = paths[0] if input_kind == "review" else paths[1]
    original = target_path.read_text(encoding="utf-8")
    if input_kind == "review":
        duplicate = original.replace(
            '"paper_id": "paper016",',
            '"paper_id": "paper016",\n  "paper_id": "paper999",',
            1,
        )
    else:
        duplicate = original.replace(
            '"decision": "accept_locator",',
            '"decision": "accept_locator",\n      "decision": "reject_locator",',
            1,
        )
    assert duplicate != original
    target_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate keys"):
        _adjudicate(paths)
