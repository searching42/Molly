from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_supplementary_scoped_candidate_response as response_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_reported_values import reported_decimal_places
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION,
    SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION,
    OledSupplementaryScopedCandidateRequestArtifact,
    _dataset_ontology_snapshot,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    build_oled_supplementary_scoped_candidate_response_from_files,
    main,
)


_GENERATED_AT = "2026-07-13T22:00:00+08:00"
_SEMANTIC_NOTE = "HOMO/LUMO labels are preserved as reported but require semantic review"
_KNOWN_COLUMNS = {
    "HOMO (eV)": "homo_ev",
    "LUMO (eV)": "lumo_ev",
    "$S_1$ (eV)": "s1_ev",
    "$T_1$ (eV)": "t1_ev",
    "$\\Delta E_{ST}^a$ (eV)": "delta_e_st_ev",
}


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _table_payload() -> dict[str, Any]:
    headers = [
        "column_1",
        "HOMO (eV)",
        "LUMO (eV)",
        "$\\Delta E_{\\text{HOMO} \\rightarrow \\text{LUMO}}$ (eV)",
        "$S_1$ (eV)",
        "$T_1$ (eV)",
        "$\\Delta E_{ST}^a$ (eV)",
        "$f(S_0-S_1)^b$",
    ]
    row_values = [
        ("TDBA", "-1.59", "-5.49", "3.90", "3.37", "2.85", "0.52", "0.1332"),
        ("TDBA-Ph", "-1.67", "-5.49", "3.83", "3.32", "2.80", "0.52", "0.1321"),
        ("mTDBA-Ph", "-1.61", "-5.37", "3.75", "3.23", "2.75", "0.48", "0.1047"),
        ("mTDBA-2Ph", "-1.63", "-5.28", "3.65", "3.12", "2.67", "0.45", "0.0986"),
        ("TDBA-Si", "-1.70", "-5.50", "3.80", "3.30", "2.78", "0.52", "0.1280"),
        ("mTDBA-Si", "-1.63", "-5.38", "3.76", "3.23", "2.76", "0.48", "0.1111"),
        ("mTDBA-2Si", "-1.65", "-5.31", "3.65", "3.13", "2.68", "0.45", "0.1188"),
    ]
    payload: dict[str, Any] = {
        "table_id": "table_p38_0178",
        "page": 38,
        "caption": "Supplementary Table S1. TD-DFT properties",
        "headers": headers,
        "rows": [dict(zip(headers, values, strict=True)) for values in row_values],
        "footnotes": ["Delta EST = S1 - T1. f is oscillator strength."],
        "source_bbox": {"x0": 10.0, "y0": 20.0, "x1": 500.0, "y1": 700.0},
        "row_count": 7,
        "column_count": 8,
        "table_content_digest": "",
    }
    payload["table_content_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "table_content_digest"}
    )
    return payload


def _scope_id(scope: dict[str, Any]) -> str:
    identity = {
        "review_item_id": scope["review_item_id"],
        "source_review_item_digest": scope["source_review_item_digest"],
        "source_id": scope["source_id"],
        "table_id": scope["matched_table"]["table_id"],
        "table_content_digest": scope["matched_table"]["table_content_digest"],
        "semantic_note": scope["semantic_note"],
    }
    return f"supplementary-scoped-request:{_stable_hash(identity)[7:31]}"


def _request_payload() -> dict[str, Any]:
    table = _table_payload()
    scope: dict[str, Any] = {
        "scope_id": "",
        "review_item_id": "supplementary-locator-review:supplementary-recovery:item-001",
        "recovery_item_id": "supplementary-recovery:item-001",
        "source_review_item_digest": "sha256:" + "5" * 64,
        "source_id": "supp-source-001",
        "source_pdf_sha256": "sha256:" + "6" * 64,
        "parsed_document_sha256": "sha256:" + "7" * 64,
        "parser_backend": "mineru_api:hybrid-engine",
        "target_kind": "table",
        "target_locator": "S1",
        "canonical_locator": "S1",
        "match_status": "exact_match",
        "matched_table": table,
        "parser_warning_codes": ["parser_warning_present"],
        "semantic_note": _SEMANTIC_NOTE,
        "semantic_review_required": True,
        "dataset_scope": "molecule_interaction_properties_only",
        "allowed_layers": ["interaction", "molecule"],
        "proposal_instructions": [
            "Read each bound table caption, headers, rows, and footnotes together.",
            (
                "Treat the source PDF as authoritative; the copied parsed table remains an "
                "unvalidated transcription."
            ),
            "Propose only molecule- or interaction-layer properties; exclude device-only records.",
            (
                "Preserve every reported label and cell string verbatim, including signs "
                "and trailing zeros."
            ),
            "Do not swap, correct, or normalize HOMO/LUMO or any other reported label or value.",
            "Treat each non-empty semantic_note as a mandatory unresolved issue for that scope.",
            (
                "Bind every proposed observation to scope_id, table_id, zero-based row_index, "
                "column_name, and the exact cell_value."
            ),
            (
                "Use only a same-row reported subject cell; do not infer canonical identity, "
                "structure, SMILES, or material role."
            ),
            (
                "Do not force an unsupported property into the ontology; request ontology "
                "review instead."
            ),
            (
                "Do not infer that table cells omitted from a proposal are absent, invalid, "
                "or irrelevant."
            ),
            (
                "Every proposal remains pending human review and must not be compiled, merged, "
                "staged, admitted, converted to gold, or written to a dataset."
            ),
            (
                "Return data only; do not return executable code, scripts, credentials, URLs, "
                "or local paths."
            ),
        ],
        "human_review_required": True,
        "source_pdf_remains_authoritative": True,
        "parsed_table_is_authoritative": False,
        "reported_labels_must_be_preserved": True,
        "reported_values_must_be_preserved": True,
        "table_exhaustiveness_validated": False,
        "table_transcription_validated": False,
        "scientific_content_validated": False,
        "physical_semantics_validated": False,
        "schema_mapping_performed": False,
        "schema_candidates_created": False,
        "direct_admission_eligible": False,
    }
    scope["scope_id"] = _scope_id(scope)
    ontology = [item.model_dump(mode="json") for item in _dataset_ontology_snapshot()]
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION,
        "run_id": "supp-mineru-run-001",
        "paper_id": "paper016",
        "generated_at": _GENERATED_AT,
        "review_artifact_sha256": "sha256:" + "1" * 64,
        "review_artifact_digest": "sha256:" + "2" * 64,
        "adjudication_artifact_sha256": "sha256:" + "3" * 64,
        "adjudication_artifact_digest": "sha256:" + "4" * 64,
        "status": "ready_for_semantic_proposal",
        "source_count": 1,
        "scope_count": 1,
        "semantic_review_required_count": 1,
        "ontology_version": SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION,
        "ontology": ontology,
        "ontology_snapshot_digest": _stable_hash(ontology),
        "scopes": [scope],
        "request_digest": "",
        "request_only": True,
        "candidate_proposal_requested": True,
        "response_received": False,
        "response_validation_implemented": False,
        "offline_only": True,
        "human_review_required": True,
        "review_artifact_read": True,
        "adjudication_artifact_read": True,
        "matched_table_content_copied": True,
        "scientific_content_included": True,
        "parsed_output_read": False,
        "pdf_content_read": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["request_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "request_digest"}
    )
    OledSupplementaryScopedCandidateRequestArtifact.model_validate(payload)
    return payload


def _write_request(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = _request_payload()
    path = tmp_path / "scoped-candidate-request.json"
    write_json(path, payload)
    return path, payload


def _recompute_request_after_table_change(request: dict[str, Any]) -> None:
    scope = request["scopes"][0]
    table = scope["matched_table"]
    table["table_content_digest"] = _stable_hash(
        {key: value for key, value in table.items() if key != "table_content_digest"}
    )
    scope["scope_id"] = _scope_id(scope)
    request["request_digest"] = _stable_hash(
        {key: value for key, value in request.items() if key != "request_digest"}
    )
    OledSupplementaryScopedCandidateRequestArtifact.model_validate(request)


def _response_payload(request_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    scope = request["scopes"][0]
    table = scope["matched_table"]
    dispositions: list[dict[str, Any]] = []
    for row_index, row in enumerate(table["rows"]):
        for column_index, column_name in enumerate(table["headers"][1:], start=1):
            cell_value = row[column_name]
            common: dict[str, Any] = {
                "scope_id": scope["scope_id"],
                "table_id": table["table_id"],
                "table_content_digest": table["table_content_digest"],
                "row_index": row_index,
                "column_index": column_index,
                "column_name": column_name,
                "cell_value": cell_value,
                "reported_value_text": cell_value,
                "reported_decimal_places": (
                    reported_decimal_places(cell_value)
                    if _is_strict_numeric_lexeme(cell_value)
                    else None
                ),
                "subject_column_index": 0,
                "subject_column_name": table["headers"][0],
                "reported_subject_text": row[table["headers"][0]],
                "proposal_note": "",
            }
            property_id = _KNOWN_COLUMNS.get(column_name)
            if not _is_strict_numeric_lexeme(cell_value):
                dispositions.append(
                    {
                        **common,
                        "disposition": "needs_source_check",
                        "source_check_reason": "unsupported_numeric_form",
                    }
                )
            elif property_id:
                dispositions.append(
                    {
                        **common,
                        "disposition": "propose_known_property",
                        "property_id": property_id,
                        "property_label": column_name,
                        "target_layer": "molecule",
                        "reported_unit": "eV",
                        "canonical_unit": "eV",
                        "comparison_context": None,
                    }
                )
            else:
                dispositions.append(
                    {
                        **common,
                        "disposition": "needs_ontology_review",
                        "property_label": column_name,
                        "proposed_target_layer": "molecule",
                        "reported_unit": "" if column_name.startswith("$f(") else "eV",
                        "ontology_review_reason": "property_missing_from_pinned_ontology",
                    }
                )
    return {
        "schema_version": "oled_supplementary_scoped_candidate_response_manifest.v1",
        "run_id": request["run_id"],
        "paper_id": request["paper_id"],
        "request_artifact_sha256": _sha256_file(request_path),
        "request_digest": request["request_digest"],
        "producer": {
            "kind": "external_llm_assisted",
            "provider_id": "anthropic",
            "model_snapshot_id": "claude-sonnet-test-snapshot-20260713",
            "prompt_contract_version": "oled-supplementary-response.v1",
            "prompt_sha256": "sha256:" + "9" * 64,
            "produced_at": _GENERATED_AT,
        },
        "response_complete": True,
        "scope_results": [
            {
                "scope_id": scope["scope_id"],
                "source_review_item_digest": scope["source_review_item_digest"],
                "source_pdf_sha256": scope["source_pdf_sha256"],
                "parsed_document_sha256": scope["parsed_document_sha256"],
                "table_id": table["table_id"],
                "table_content_digest": table["table_content_digest"],
                "semantic_note": scope["semantic_note"],
                "semantic_note_status": "unresolved",
                "subject_column_index": 0,
                "subject_column_name": table["headers"][0],
                "cell_dispositions": dispositions,
            }
        ],
    }


def _is_strict_numeric_lexeme(value: str) -> bool:
    try:
        reported_decimal_places(value)
    except ValueError:
        return False
    return True


def _write_response(
    tmp_path: Path,
    request_path: Path,
    request: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path]:
    payload = _response_payload(request_path, request)
    response_path = tmp_path / "candidate-response-manifest.json"
    write_json(response_path, payload)
    return response_path, payload, tmp_path / "validated-candidate-response.json"


def _build_chain(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    request_path, request = _write_request(tmp_path)
    response_path, response, output_path = _write_response(tmp_path, request_path, request)
    return request_path, response_path, output_path, request, response


def _recompute_artifact_digest(payload: dict[str, Any]) -> None:
    payload["response_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "response_artifact_digest"}
    )


def test_validates_complete_paper016_response_without_crossing_review_boundary(
    tmp_path: Path,
) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
        generated_at=_GENERATED_AT,
    )

    assert artifact.status.value == "ready_for_human_semantic_review"
    assert artifact.scope_count == 1
    assert artifact.cell_disposition_count == 49
    assert artifact.known_property_proposal_count == 35
    assert artifact.ontology_review_count == 14
    assert artifact.source_check_count == 0
    assert artifact.exclusion_count == 0
    assert artifact.semantic_review_required_count == 1
    assert artifact.request_artifact_sha256 == _sha256_file(request_path)
    assert artifact.response_manifest_sha256 == _sha256_file(response_path)
    scope = artifact.scope_results[0]
    assert scope.semantic_note == _SEMANTIC_NOTE
    assert scope.semantic_note_status.value == "unresolved"
    exact_values = {
        item.cell_value: item.reported_decimal_places
        for item in scope.cell_dispositions
        if item.cell_value in {"2.80", "3.30", "0.1280", "-1.70", "-5.50"}
    }
    assert exact_values == {
        "2.80": 2,
        "3.30": 2,
        "0.1280": 4,
        "-1.70": 2,
        "-5.50": 2,
    }
    assert artifact.external_llm_response_ingested is True
    assert artifact.validator_llm_called is False
    assert artifact.validator_network_accessed is False
    assert artifact.table_transcription_validated is False
    assert artifact.physical_semantics_validated is False
    assert artifact.human_semantic_review_completed is False
    assert artifact.schema_mapping_proposed is True
    assert artifact.schema_mapping_adjudicated is False
    assert artifact.schema_candidates_created is False
    assert artifact.direct_admission_eligible is False
    assert artifact.device_only_admitted is False
    assert artifact.gold_records_created is False
    assert artifact.dataset_written is False
    dumped = artifact.model_dump(mode="json")
    assert "schema_candidates" not in dumped
    assert OledSupplementaryScopedCandidateResponseArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_rejects_incomplete_duplicate_or_unknown_cell_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    cells = response["scope_results"][0]["cell_dispositions"]
    if mutation == "missing":
        cells.pop()
    elif mutation == "duplicate":
        cells.append(deepcopy(cells[0]))
    else:
        fabricated = deepcopy(cells[0])
        fabricated["row_index"] = 99
        cells[0] = fabricated
    write_json(response_path, response)

    with pytest.raises((ValidationError, ValueError)):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


def test_response_cannot_select_a_property_column_as_subject(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["scope_results"][0]["subject_column_index"] = 1
    response["scope_results"][0]["subject_column_name"] = "HOMO (eV)"
    write_json(response_path, response)

    with pytest.raises(ValueError, match="requires the first column as subject"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize("subject_mode", ["numeric", "blank"])
def test_request_first_column_requires_reported_nonnumeric_subjects(
    tmp_path: Path,
    subject_mode: str,
) -> None:
    request = _request_payload()
    table = request["scopes"][0]["matched_table"]
    subject_header = table["headers"][0]
    if subject_mode == "numeric":
        for index, row in enumerate(table["rows"], start=1):
            row[subject_header] = str(index)
        error_pattern = "cannot be purely numeric"
    else:
        table["rows"][0][subject_header] = " "
        error_pattern = "reported_subject_text is required|requires a subject for every row"
    _recompute_request_after_table_change(request)
    request_path = tmp_path / f"{subject_mode}-subject-request.json"
    write_json(request_path, request)
    response_path, _, output_path = _write_response(tmp_path, request_path, request)

    with pytest.raises(ValueError, match=error_pattern):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("shape_mutation", "error_pattern"),
    [
        ("duplicate_header", "requires unique table headers"),
        ("empty_header", "requires unique table headers"),
        ("nonrectangular_row", "requires complete rectangular table rows"),
    ],
)
def test_narrow_table_shape_fails_closed_before_cell_mapping(
    tmp_path: Path,
    shape_mutation: str,
    error_pattern: str,
) -> None:
    request = _request_payload()
    request_path = tmp_path / f"{shape_mutation}-request.json"
    write_json(request_path, request)
    response = _response_payload(request_path, request)
    table = request["scopes"][0]["matched_table"]
    if shape_mutation == "duplicate_header":
        table["headers"][2] = table["headers"][1]
    elif shape_mutation == "empty_header":
        old_header = table["headers"][1]
        table["headers"][1] = ""
        for row in table["rows"]:
            row[""] = row.pop(old_header)
    else:
        del table["rows"][0][table["headers"][1]]
    _recompute_request_after_table_change(request)
    write_json(request_path, request)
    response["request_artifact_sha256"] = _sha256_file(request_path)
    response["request_digest"] = request["request_digest"]
    response_scope = response["scope_results"][0]
    response_scope["scope_id"] = request["scopes"][0]["scope_id"]
    response_scope["table_content_digest"] = table["table_content_digest"]
    for item in response_scope["cell_dispositions"]:
        item["scope_id"] = request["scopes"][0]["scope_id"]
        item["table_content_digest"] = table["table_content_digest"]
    response_path = tmp_path / f"{shape_mutation}-response.json"
    output_path = tmp_path / f"{shape_mutation}-output.json"
    write_json(response_path, response)

    with pytest.raises(ValueError, match=error_pattern):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("target", "replacement", "error_pattern"),
    [
        ("request_artifact_sha256", "sha256:" + "a" * 64, "exact request bytes"),
        ("request_digest", "sha256:" + "b" * 64, "canonical request content"),
        ("source_pdf_sha256", "sha256:" + "c" * 64, "source_pdf_sha256"),
        ("table_content_digest", "sha256:" + "d" * 64, "table_content_digest"),
        ("semantic_note", "different scientific note", "semantic_note"),
        ("reported_subject_text", "different subject", "subject text"),
        ("property_id", "eqe_percent", "pinned ontology"),
        ("property_id", "lumo_ev", "reported column label"),
        ("target_layer", "device", "request scope"),
        ("canonical_unit", "meV", "canonical_unit"),
    ],
)
def test_rejects_request_scope_cell_and_mapping_tampering(
    tmp_path: Path,
    target: str,
    replacement: Any,
    error_pattern: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    scope = response["scope_results"][0]
    if target in {"request_artifact_sha256", "request_digest"}:
        response[target] = replacement
    elif target in {
        "source_pdf_sha256",
        "table_content_digest",
        "semantic_note",
    }:
        scope[target] = replacement
    else:
        scope["cell_dispositions"][0][target] = replacement
    write_json(response_path, response)

    with pytest.raises((ValidationError, ValueError), match=error_pattern):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


@pytest.mark.parametrize(
    ("cell_value", "reported_value_text", "decimal_places"),
    [
        ("2.80", "2.8", 1),
        ("0.1280", "0.128", 3),
        ("-1.70", "−1.70", 2),
    ],
)
def test_rejects_reformatted_reported_literals(
    tmp_path: Path,
    cell_value: str,
    reported_value_text: str,
    decimal_places: int,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    item = next(
        item
        for item in response["scope_results"][0]["cell_dispositions"]
        if item["cell_value"] == cell_value
    )
    item["reported_value_text"] = reported_value_text
    item["reported_decimal_places"] = decimal_places
    write_json(response_path, response)

    with pytest.raises(ValidationError, match="exactly match cell_value"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


def test_rejects_request_byte_change_even_when_canonical_content_is_unchanged(
    tmp_path: Path,
) -> None:
    request_path, response_path, output_path, request, _ = _build_chain(tmp_path)
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact request bytes"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


def test_non_scalar_numeric_cell_requires_explicit_source_check(tmp_path: Path) -> None:
    request = _request_payload()
    scope = request["scopes"][0]
    scope["matched_table"]["rows"][0]["HOMO (eV)"] = "-1.59 ± 0.02"
    _recompute_request_after_table_change(request)
    request_path = tmp_path / "non-scalar-request.json"
    write_json(request_path, request)
    response_path, _, output_path = _write_response(tmp_path, request_path, request)

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )

    assert artifact.cell_disposition_count == 49
    assert artifact.known_property_proposal_count == 34
    assert artifact.source_check_count == 1
    source_check = next(
        item
        for item in artifact.scope_results[0].cell_dispositions
        if item.cell_value == "-1.59 ± 0.02"
    )
    assert source_check.disposition.value == "needs_source_check"
    assert source_check.reported_decimal_places is None
    assert source_check.source_check_reason.value == "unsupported_numeric_form"


def test_unicode_minus_known_property_preserves_literal_and_normalizes_for_check(
    tmp_path: Path,
) -> None:
    request = _request_payload()
    scope = request["scopes"][0]
    scope["matched_table"]["rows"][0]["HOMO (eV)"] = "−1.59"
    _recompute_request_after_table_change(request)
    request_path = tmp_path / "unicode-minus-request.json"
    write_json(request_path, request)
    response_path, _, output_path = _write_response(tmp_path, request_path, request)

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )

    item = next(
        item
        for item in artifact.scope_results[0].cell_dispositions
        if item.row_index == 0 and item.column_name == "HOMO (eV)"
    )
    assert item.reported_value_text == "−1.59"
    assert item.reported_decimal_places == 2


def test_required_comparison_context_requires_every_explicit_field(tmp_path: Path) -> None:
    request = _request_payload()
    scope = request["scopes"][0]
    table = scope["matched_table"]
    old_header = table["headers"][1]
    new_header = "PL peak (nm)"
    table["headers"][1] = new_header
    for index, row in enumerate(table["rows"]):
        row[new_header] = f"{500 + index}.0"
        del row[old_header]
    _recompute_request_after_table_change(request)
    request_path = tmp_path / "context-request.json"
    write_json(request_path, request)
    response = _response_payload(request_path, request)
    first = response["scope_results"][0]["cell_dispositions"][0]
    first.clear()
    first.update(
        {
            "disposition": "propose_known_property",
            "scope_id": scope["scope_id"],
            "table_id": table["table_id"],
            "table_content_digest": table["table_content_digest"],
            "row_index": 0,
            "column_index": 1,
            "column_name": new_header,
            "cell_value": "500.0",
            "reported_value_text": "500.0",
            "reported_decimal_places": 1,
            "subject_column_index": 0,
            "subject_column_name": table["headers"][0],
            "reported_subject_text": table["rows"][0][table["headers"][0]],
            "proposal_note": "",
            "property_id": "photoluminescence_peak_nm",
            "property_label": new_header,
            "target_layer": "molecule",
            "reported_unit": "nm",
            "canonical_unit": "nm",
            "comparison_context": {
                "measurement_temperature": None,
                "host_material": None,
                "dopant_concentration": None,
                "sample_form": None,
                "excitation_wavelength": None,
                "lifetime_fit_method": None,
            },
        }
    )
    response_path = tmp_path / "context-response.json"
    write_json(response_path, response)
    output_path = tmp_path / "context-output.json"

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )
    assert artifact.known_property_proposal_count == 29

    response["scope_results"][0]["cell_dispositions"][0]["comparison_context"].pop(
        "lifetime_fit_method"
    )
    write_json(response_path, response)
    missing_output = tmp_path / "context-missing-output.json"
    with pytest.raises(
        ValidationError,
        match="explicitly include every photophysical context field",
    ):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=missing_output,
        )

    context = response["scope_results"][0]["cell_dispositions"][0][
        "comparison_context"
    ]
    context["lifetime_fit_method"] = None
    context["measurement_temperature"] = True
    write_json(response_path, response)
    boolean_output = tmp_path / "context-boolean-output.json"
    with pytest.raises(ValidationError, match="must not be boolean"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=boolean_output,
        )

    assert not boolean_output.exists()


def test_rejects_reported_unit_that_disagrees_with_source_header(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["scope_results"][0]["cell_dispositions"][0]["reported_unit"] = "meV"
    write_json(response_path, response)

    with pytest.raises(ValueError, match="reported_unit does not match the source header"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


@pytest.mark.parametrize(
    ("new_header", "claimed_unit"),
    [
        ("PLQY (%)^a", "fraction"),
        ("PLQY [%]", "fraction"),
        ("PLQY [fraction]", "%"),
        ("PLQY / %", "fraction"),
    ],
)
def test_source_header_unit_syntax_cannot_be_bypassed(
    tmp_path: Path,
    new_header: str,
    claimed_unit: str,
) -> None:
    request = _request_payload()
    scope = request["scopes"][0]
    table = scope["matched_table"]
    old_header = table["headers"][1]
    table["headers"][1] = new_header
    for row in table["rows"]:
        row[new_header] = "0.80"
        del row[old_header]
    _recompute_request_after_table_change(request)
    request_path = tmp_path / "footnoted-unit-request.json"
    write_json(request_path, request)
    response = _response_payload(request_path, request)
    first = response["scope_results"][0]["cell_dispositions"][0]
    first.clear()
    first.update(
        {
            "disposition": "propose_known_property",
            "scope_id": scope["scope_id"],
            "table_id": table["table_id"],
            "table_content_digest": table["table_content_digest"],
            "row_index": 0,
            "column_index": 1,
            "column_name": new_header,
            "cell_value": "0.80",
            "reported_value_text": "0.80",
            "reported_decimal_places": 2,
            "subject_column_index": 0,
            "subject_column_name": table["headers"][0],
            "reported_subject_text": table["rows"][0][table["headers"][0]],
            "proposal_note": "",
            "property_id": "plqy",
            "property_label": new_header,
            "target_layer": "interaction",
            "reported_unit": claimed_unit,
            "canonical_unit": "fraction",
            "comparison_context": None,
        }
    )
    response_path = tmp_path / "footnoted-unit-response.json"
    output_path = tmp_path / "footnoted-unit-output.json"
    write_json(response_path, response)

    with pytest.raises(ValueError, match="reported_unit does not match the source header"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


def test_known_device_only_column_requires_explicit_device_exclusion(tmp_path: Path) -> None:
    request = _request_payload()
    scope = request["scopes"][0]
    table = scope["matched_table"]
    old_header = table["headers"][1]
    new_header = "EQE (%)"
    table["headers"][1] = new_header
    for row in table["rows"]:
        row[new_header] = "21.0"
        del row[old_header]
    _recompute_request_after_table_change(request)
    request_path = tmp_path / "device-only-request.json"
    write_json(request_path, request)
    response = _response_payload(request_path, request)
    device_items = [
        item
        for item in response["scope_results"][0]["cell_dispositions"]
        if item["column_name"] == new_header
    ]
    for item in device_items:
        item.pop("property_label")
        item.pop("proposed_target_layer")
        item.pop("reported_unit")
        item.pop("ontology_review_reason")
        item["disposition"] = "exclude_from_dataset"
        item["exclusion_reason"] = "device_only"
    response_path = tmp_path / "device-only-response.json"
    output_path = tmp_path / "device-only-output.json"
    write_json(response_path, response)

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )
    assert artifact.exclusion_count == 7
    assert artifact.device_only_admitted is False

    first = device_items[0]
    first.pop("exclusion_reason")
    first.update(
        {
            "disposition": "needs_ontology_review",
            "property_label": new_header,
            "proposed_target_layer": "molecule",
            "reported_unit": "%",
            "ontology_review_reason": "property_missing_from_pinned_ontology",
        }
    )
    write_json(response_path, response)
    invalid_output = tmp_path / "device-only-invalid-output.json"
    with pytest.raises(ValueError, match="known device-only columns require"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=invalid_output,
        )

    assert not invalid_output.exists()


@pytest.mark.parametrize(
    "unsafe_note",
    [
        "https://example.invalid/response",
        "www.example.com/response",
        "example.com/response",
        "mailto:secret@example.com",
        "doi.org/10.1/example",
        "/operator/private/response.json",
        "artifacts/response.json",
        "token=abc123",
        "Bearer abc12345",
        "sk-abcdef123456",
        "```python\nimport os\n```",
        "import subprocess",
        "os.popen('id')",
        "__import__('os').system('id')",
        "python -c print(1)",
        "curl example.invalid",
        "wget example.invalid",
        "powershell -Command Get-ChildItem",
    ],
)
def test_rejects_sensitive_executable_or_file_text(
    tmp_path: Path,
    unsafe_note: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["scope_results"][0]["cell_dispositions"][0]["proposal_note"] = unsafe_note
    write_json(response_path, response)

    with pytest.raises(ValidationError):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


@pytest.mark.parametrize(
    "producer_field",
    ["provider_id", "model_snapshot_id", "prompt_contract_version"],
)
def test_rejects_credentials_in_external_producer_provenance(
    tmp_path: Path,
    producer_field: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["producer"][producer_field] = "sk-abcdef123456"
    write_json(response_path, response)

    with pytest.raises(ValidationError, match="credential-like text"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


def test_external_llm_provenance_requires_exact_prompt_hash(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["producer"].pop("prompt_sha256")
    write_json(response_path, response)

    with pytest.raises(ValidationError, match="prompt provenance"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("produced_at", "generated_at", "error_pattern"),
    [
        ("2026-07-12T22:00:00+08:00", _GENERATED_AT, "predates its bound request"),
        ("2026-07-14T22:00:00+08:00", _GENERATED_AT, "timestamps are invalid"),
    ],
)
def test_response_provenance_timestamps_follow_causal_order(
    tmp_path: Path,
    produced_at: str,
    generated_at: str,
    error_pattern: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["producer"]["produced_at"] = produced_at
    write_json(response_path, response)

    with pytest.raises((ValidationError, ValueError), match=error_pattern):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
            generated_at=generated_at,
        )

    assert not output_path.exists()


def test_allows_safe_scientific_response_text(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["scope_results"][0]["cell_dispositions"][0]["proposal_note"] = (
        "HOMO/LUMO at B3LYP/6-31G(d,p); SK-TADF01 is a material label and charge "
        "bearer is scientific text."
    )
    write_json(response_path, response)

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )

    assert artifact.cell_disposition_count == 49


@pytest.mark.parametrize("invented_field", ["canonical_smiles", "material_role", "device_stack"])
def test_rejects_invented_identity_role_or_device_fields(
    tmp_path: Path,
    invented_field: str,
) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response["scope_results"][0]["cell_dispositions"][0][invented_field] = "invented"
    write_json(response_path, response)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


def test_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, response = _build_chain(tmp_path)
    response_path.write_text(
        '{"schema_version":"oled_supplementary_scoped_candidate_response_manifest.v1",'
        '"schema_version":"oled_supplementary_scoped_candidate_response_manifest.v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    response_path.write_text(
        json.dumps(response, ensure_ascii=False)[:-1] + ',"unexpected":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains NaN"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )


def test_rejects_symlink_input(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)
    link_path = tmp_path / "response-link.json"
    link_path.symlink_to(response_path)

    with pytest.raises(ValueError, match="input is unavailable"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=link_path,
            output_json=output_path,
        )


def test_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    request_path, _, output_path, _, _ = _build_chain(tmp_path)
    fifo_path = tmp_path / "response-fifo.json"
    os.mkfifo(fifo_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai4s_agent.oled_supplementary_scoped_candidate_response",
            "--request-artifact",
            str(request_path),
            "--response-manifest",
            str(fifo_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "supplementary_scoped_candidate_response_failed" in result.stdout
    assert not output_path.exists()


@pytest.mark.parametrize("protected_input", ["request", "response"])
def test_output_must_not_overwrite_an_input(tmp_path: Path, protected_input: str) -> None:
    request_path, response_path, _, _, _ = _build_chain(tmp_path)
    output_path = request_path if protected_input == "request" else response_path
    original = output_path.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert output_path.read_bytes() == original


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)
    output_path.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output must be fresh"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "operator-owned\n"


def test_output_parent_swap_fails_without_publishing_or_leaking_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, response_path, _, _, _ = _build_chain(tmp_path)
    output_parent = tmp_path / "output-parent"
    moved_parent = tmp_path / "moved-output-parent"
    output_parent.mkdir()
    output_path = output_parent / "artifact.json"
    real_link = os.link

    def swap_parent_then_link(*args: Any, **kwargs: Any) -> None:
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        real_link(*args, **kwargs)

    monkeypatch.setattr(response_runner.os, "link", swap_parent_then_link)

    with pytest.raises(ValueError, match="output parent changed during write"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )

    assert not output_path.exists()
    assert not (moved_parent / output_path.name).exists()
    assert not list(moved_parent.glob(f".{output_path.name}.*.tmp"))


def test_stable_symlinked_output_parent_uses_its_pinned_real_directory(
    tmp_path: Path,
) -> None:
    request_path, response_path, _, _, _ = _build_chain(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    alias_parent = tmp_path / "output-parent-alias"
    real_parent.mkdir()
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    output_path = alias_parent / "artifact.json"

    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
    )

    assert artifact.cell_disposition_count == 49
    assert (real_parent / output_path.name).is_file()


def test_cli_success_and_failure_are_redacted(tmp_path: Path) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)
    stdout = StringIO()

    exit_code = main(
        [
            "--request-artifact",
            str(request_path),
            "--response-manifest",
            str(response_path),
            "--output",
            str(output_path),
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "ready_for_human_semantic_review" in output
    assert str(tmp_path) not in output
    assert "paper016" not in output
    assert _SEMANTIC_NOTE not in output
    assert "0.1280" not in output
    assert "claude-sonnet-test-snapshot-20260713" not in output

    failed_output_path = tmp_path / "failed-output.json"
    response_path.write_text('{"token":"secret-value"}', encoding="utf-8")
    stdout = StringIO()
    exit_code = main(
        [
            "--request-artifact",
            str(request_path),
            "--response-manifest",
            str(response_path),
            "--output",
            str(failed_output_path),
        ],
        stdout=stdout,
    )
    output = stdout.getvalue()
    assert exit_code == 2
    assert "supplementary_scoped_candidate_response_failed" in output
    assert "secret-value" not in output
    assert str(tmp_path) not in output
    assert not failed_output_path.exists()


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("physical_semantics_validated", True),
        ("semantic_notes_resolved", True),
        ("schema_mapping_adjudicated", True),
        ("schema_candidates_created", True),
        ("direct_admission_eligible", True),
        ("device_only_admitted", True),
        ("gold_records_created", True),
        ("dataset_written", True),
    ],
)
def test_artifact_rejects_downstream_flag_tampering_even_with_recomputed_digest(
    tmp_path: Path,
    field_name: str,
    replacement: bool,
) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)
    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
        generated_at=_GENERATED_AT,
    )
    payload = deepcopy(artifact.model_dump(mode="json"))
    payload[field_name] = replacement
    _recompute_artifact_digest(payload)

    with pytest.raises(ValidationError, match="downstream boundary"):
        OledSupplementaryScopedCandidateResponseArtifact.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("ontology_digest", "ontology snapshot"),
        ("cell_scope", "cell scope mismatch"),
        ("cell_table", "cell table mismatch"),
        ("cell_table_digest", "cell table digest mismatch"),
        ("known_property", "reported column label"),
        ("known_layer", "layer outside"),
        ("cell_literal", "manifest content digest"),
    ],
)
def test_artifact_replays_internal_bindings_after_outer_digest_recomputation(
    tmp_path: Path,
    mutation: str,
    error_pattern: str,
) -> None:
    request_path, response_path, output_path, _, _ = _build_chain(tmp_path)
    artifact = build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=output_path,
        generated_at=_GENERATED_AT,
    )
    payload = deepcopy(artifact.model_dump(mode="json"))
    item = payload["scope_results"][0]["cell_dispositions"][0]
    if mutation == "ontology_digest":
        payload["ontology_snapshot_digest"] = "sha256:" + "f" * 64
    elif mutation == "cell_scope":
        item["scope_id"] = "supplementary-scoped-request:tampered"
    elif mutation == "cell_table":
        item["table_id"] = "table_tampered"
    elif mutation == "cell_table_digest":
        item["table_content_digest"] = "sha256:" + "e" * 64
    elif mutation == "known_property":
        item["property_id"] = "lumo_ev"
    elif mutation == "known_layer":
        item["target_layer"] = "device"
    else:
        item["cell_value"] = "-9.99"
        item["reported_value_text"] = "-9.99"
        item["reported_decimal_places"] = 2
    _recompute_artifact_digest(payload)

    with pytest.raises((ValidationError, ValueError), match=error_pattern):
        OledSupplementaryScopedCandidateResponseArtifact.model_validate(payload)


def test_file_binding_rejects_unknown_scope(tmp_path: Path) -> None:
    request_path, response_path, output_path, request, _ = _build_chain(tmp_path)
    response = _response_payload(request_path, request)
    response["scope_results"][0]["scope_id"] = "supplementary-scoped-request:unknown"
    for item in response["scope_results"][0]["cell_dispositions"]:
        item["scope_id"] = "supplementary-scoped-request:unknown"
    write_json(response_path, response)

    with pytest.raises(ValueError, match="scope binding mismatch"):
        build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=request_path,
            response_manifest_json=response_path,
            output_json=output_path,
        )
