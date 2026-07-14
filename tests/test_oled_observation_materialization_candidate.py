from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent import oled_observation_materialization_candidate as candidate_runner
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledComparisonContextStatus,
    OledMeasurementCondition,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    build_oled_material_registry_entry,
)
from ai4s_agent.domains.oled_observation_materialization_candidate import (
    OledObservationMaterializationCandidateArtifact,
    _canonicalize_observation,
    _observation_candidate_item_digest,
    oled_observation_materialization_candidate_artifact_digest,
)
from ai4s_agent.oled_material_registry_resolution_request import (
    build_oled_material_registry_resolution_request_from_files,
)
from ai4s_agent.oled_observation_materialization_candidate import (
    build_oled_observation_materialization_candidate_from_files,
    main,
)
from ai4s_agent.oled_observation_staging_preflight import (
    build_oled_observation_staging_preflight_from_files,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    build_oled_supplementary_material_identity_adjudication_from_files,
)
from tests.test_oled_material_registry_adjudication import _adjudicate
from tests.test_oled_material_registry_resolution_request import (
    _REQUEST_AT,
    _accepted_candidate,
    _snapshot,
    _write_snapshot,
)
from tests.test_oled_observation_staging_preflight import _PREFLIGHT_AT
from tests.test_oled_supplementary_material_identity_evidence_response import (
    _candidate_result,
    _source_check_result,
)
from tests.test_oled_supplementary_material_identity_review import (
    _REVIEWED_AT,
    _adjudication_kwargs,
    _build_packet,
    _decision_payload,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_MATERIALIZED_AT = "2026-07-14T00:40:00+08:00"


def _build_exact_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_row: int = 4,
    map_existing: bool = True,
) -> dict[str, Any]:
    def result_factory(group: Any) -> dict[str, Any]:
        if group.row_index == candidate_row:
            return _candidate_result(group)
        return _source_check_result(group)

    packet, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=result_factory,
    )
    decision_payload = _decision_payload(packet, context["packet_path"])
    item_by_id = {item.review_item_id: item for item in packet.review_items}
    for decision in decision_payload["decisions"]:
        item = item_by_id[decision["review_item_id"]]
        row_index = item.validated_result.bound_identity_group.row_index
        if row_index == candidate_row:
            decision["decision"] = "accept_structure_candidate"
            decision["candidate_source_match"] = "matches_source"
        else:
            decision["decision"] = "confirm_source_check"
            decision["candidate_source_match"] = "not_applicable"
        decision["review_note"] = ""
    decision_payload["reviewed_at"] = _REVIEWED_AT
    decision_path = context["review_dir"] / "material-identity-decisions.json"
    write_json(decision_path, decision_payload)
    source_adjudication_path = (
        context["review_dir"] / "material-identity-adjudication.json"
    )
    source_adjudication = (
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=source_adjudication_path,
            )
        )
    )
    accepted_group, structure = _accepted_candidate(source_adjudication)
    reported = (
        accepted_group.review_item.validated_result.bound_identity_group.reported_subject_text
    )
    entry = build_oled_material_registry_entry(
        material_id="material-0001",
        canonical_name=reported,
        aliases=["accepted-fixture-alias"],
        canonical_isomeric_smiles=structure.canonical_isomeric_smiles_candidate,
    )
    snapshot_path = _write_snapshot(
        tmp_path,
        _snapshot([entry] if map_existing else []),
    )
    registry_request_path = tmp_path / "material-registry-resolution-request.json"
    registry_request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_adjudication_path,
        registry_snapshot_json=snapshot_path,
        output_json=registry_request_path,
        generated_at=_REQUEST_AT,
    )
    _, _, registry_adjudication_path = _adjudicate(
        tmp_path,
        registry_request,
        registry_request_path,
        decision=("map_to_existing_entity" if map_existing else "keep_unresolved"),
        selected_material_id=("material-0001" if map_existing else ""),
    )
    staging_path = tmp_path / "observation-staging-preflight.json"
    build_oled_observation_staging_preflight_from_files(
        request_artifact_json=registry_request_path,
        registry_adjudication_json=registry_adjudication_path,
        output_json=staging_path,
        generated_at=_PREFLIGHT_AT,
    )
    chain = context["chain"]
    return {
        "staging_path": staging_path,
        "identity_request_path": context["request_path"],
        "semantic_adjudication_path": chain["semantic_adjudication_path"],
        "transcription_packet_path": chain["transcription_packet_path"],
        "transcription_adjudication_path": chain[
            "transcription_adjudication_path"
        ],
    }


def _file_kwargs(
    chain: dict[str, Any],
    output_path: Path,
    *,
    generated_at: str = _MATERIALIZED_AT,
) -> dict[str, Any]:
    return {
        "staging_preflight_json": chain["staging_path"],
        "material_identity_request_json": chain["identity_request_path"],
        "semantic_adjudication_json": chain["semantic_adjudication_path"],
        "transcription_review_packet_json": chain["transcription_packet_path"],
        "transcription_adjudication_json": chain[
            "transcription_adjudication_path"
        ],
        "output_json": output_path,
        "generated_at": generated_at,
    }


def test_exact_chain_materializes_five_source_bound_observation_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    output_path = tmp_path / "observation-materialization-candidates.json"

    artifact = build_oled_observation_materialization_candidate_from_files(
        **_file_kwargs(chain, output_path)
    )

    assert artifact.status.value == "ready_for_reviewed_evidence_staging_preflight"
    assert artifact.observation_candidate_count == 5
    assert artifact.comparison_ready_candidate_count == 5
    assert artifact.comparison_context_not_required_count == 5
    assert artifact.comparison_context_complete_count == 0
    assert artifact.comparison_context_incomplete_count == 0
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    assert artifact.source_property_values_present
    assert artifact.material_id_attached_to_observations
    assert artifact.observations_materialized
    assert not artifact.reviewed_evidence_staging
    assert not artifact.direct_admission_eligible
    assert not artifact.gold_records_created
    assert not artifact.dataset_written
    assert artifact.staging_preflight_sha256 == _sha256_file(chain["staging_path"])
    assert artifact.material_identity_request_sha256 == _sha256_file(
        chain["identity_request_path"]
    )
    by_text = {
        item.property_observation.reported_value_text: item
        for item in artifact.observation_candidates
    }
    assert {"-1.70", "-5.50", "3.30", "2.78", "0.52"} == set(by_text)
    assert by_text["-1.70"].property_observation.value == -1.7
    assert by_text["-1.70"].property_observation.reported_decimal_places == 2
    assert by_text["-1.70"].selected_existing_material_id == "material-0001"
    assert by_text["-1.70"].mapping_summary.property_label == "HOMO (eV)"
    assert by_text["-1.70"].property_observation.property_label == "homo_ev"
    assert by_text["-1.70"].canonical_observation.property_id == "homo_ev"
    assert by_text["-1.70"].canonical_observation.normalized_unit == "eV"
    assert OledObservationMaterializationCandidateArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_same_model_but_different_pr_k_bytes_fail_exact_hash_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    original = chain["identity_request_path"]
    alternate = tmp_path / "same-pr-k-with-different-bytes.json"
    alternate.write_text(original.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    chain["identity_request_path"] = alternate
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="exact file bound"):
        build_oled_observation_materialization_candidate_from_files(
            **_file_kwargs(chain, output_path)
        )

    assert not output_path.exists()


def test_unresolved_registry_identity_produces_no_observation_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch, map_existing=False)

    artifact = build_oled_observation_materialization_candidate_from_files(
        **_file_kwargs(chain, tmp_path / "no-resolved-candidates.json")
    )

    assert artifact.status.value == "no_resolved_observation_candidates"
    assert artifact.source_staging_item_count == 0
    assert artifact.source_staging_cell_count == 0
    assert artifact.observation_candidate_count == 0
    assert artifact.observation_candidates == []
    assert not artifact.source_property_values_present
    assert not artifact.material_id_attached_to_observations
    assert not artifact.observations_materialized
    assert not artifact.reviewed_evidence_staging
    assert not artifact.dataset_written


def test_same_model_but_different_pr_j_packet_bytes_fail_exact_hash_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    original = chain["transcription_packet_path"]
    alternate = tmp_path / "same-pr-j-packet-with-different-bytes.json"
    alternate.write_text(original.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    chain["transcription_packet_path"] = alternate
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="exact file bound"):
        build_oled_observation_materialization_candidate_from_files(
            **_file_kwargs(chain, output_path)
        )

    assert not output_path.exists()


def test_materialized_value_tamper_fails_even_after_rehashing_item_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    artifact = build_oled_observation_materialization_candidate_from_files(
        **_file_kwargs(chain, tmp_path / "valid.json")
    )
    original_candidate = artifact.observation_candidates[0]
    changed_observation = original_candidate.property_observation.model_copy(
        update={"value": 7.77}
    )
    provisional_item = original_candidate.model_copy(
        update={
            "property_observation": changed_observation,
            "candidate_digest": "sha256:" + "0" * 64,
        }
    )
    changed_item = provisional_item.model_copy(
        update={
            "candidate_digest": _observation_candidate_item_digest(provisional_item)
        }
    )
    changed_candidates = [changed_item, *artifact.observation_candidates[1:]]
    provisional = artifact.model_copy(
        update={
            "observation_candidates": changed_candidates,
            "artifact_digest": "sha256:" + "0" * 64,
        }
    )
    changed_artifact = provisional.model_copy(
        update={
            "artifact_digest": (
                oled_observation_materialization_candidate_artifact_digest(provisional)
            )
        }
    )

    with pytest.raises(ValidationError, match="reported_value_text|canonical replay"):
        OledObservationMaterializationCandidateArtifact.model_validate(
            changed_artifact.model_dump(mode="json")
        )


def test_materialization_cannot_predate_staging_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="predates PR-P"):
        build_oled_observation_materialization_candidate_from_files(
            **_file_kwargs(
                chain,
                output_path,
                generated_at="2026-07-14T00:29:59+08:00",
            )
        )

    assert not output_path.exists()


def test_output_cannot_overwrite_any_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    protected = chain["semantic_adjudication_path"]
    before = protected.read_bytes()

    with pytest.raises(ValueError, match="output must not overwrite"):
        build_oled_observation_materialization_candidate_from_files(
            **_file_kwargs(chain, protected)
        )

    assert protected.read_bytes() == before


def test_incomplete_photophysical_context_remains_explicitly_incomplete() -> None:
    observation = OledPropertyObservation(
        property_label="photoluminescence_peak_nm",
        value=520.0,
        unit="nm",
        reported_value_text="520.0",
        reported_decimal_places=1,
        condition=OledMeasurementCondition(
            measurement_temperature=None,
            measurement_temperature_unit=None,
            host_material=None,
            dopant_concentration=None,
            dopant_concentration_unit=None,
            sample_form=None,
            excitation_wavelength=None,
            excitation_wavelength_unit=None,
            lifetime_fit_method=None,
        ),
    )

    canonical = _canonicalize_observation(OledCausalLayer.MOLECULE, observation)

    assert canonical.comparison_context_status == OledComparisonContextStatus.INCOMPLETE
    assert not canonical.is_comparison_ready
    assert canonical.comparison_context_hash is None
    assert set(canonical.comparison_context_missing_fields) == {
        "measurement_temperature",
        "host_material",
        "dopant_concentration",
        "sample_form",
        "excitation_wavelength",
        "lifetime_fit_method",
    }


def test_cli_failure_is_redacted_and_does_not_publish_output(tmp_path: Path) -> None:
    sensitive = tmp_path / "token=do-not-disclose.json"
    output_path = tmp_path / "must-not-exist.json"
    stream = StringIO()

    status = main(
        [
            "--staging-preflight",
            str(sensitive),
            "--material-identity-request",
            str(tmp_path / "missing-k.json"),
            "--semantic-adjudication",
            str(tmp_path / "missing-i.json"),
            "--transcription-review-packet",
            str(tmp_path / "missing-j-packet.json"),
            "--transcription-adjudication",
            str(tmp_path / "missing-j-adjudication.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    assert status == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "observation_materialization_candidate_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()
    assert not output_path.exists()


def test_output_parent_replacement_fails_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_exact_chain(tmp_path, monkeypatch)
    output_dir = tmp_path / "candidate-output"
    output_dir.mkdir()
    output_path = output_dir / "candidate.json"
    replacement = tmp_path / "replacement-output"
    original_publish = candidate_runner._publish_packet_text

    def replace_parent_then_publish(*args: Any, **kwargs: Any) -> None:
        output_dir.rename(replacement)
        output_dir.mkdir()
        original_publish(*args, **kwargs)

    monkeypatch.setattr(
        candidate_runner,
        "_publish_packet_text",
        replace_parent_then_publish,
    )

    with pytest.raises(ValueError, match="parent changed"):
        build_oled_observation_materialization_candidate_from_files(
            **_file_kwargs(chain, output_path)
        )

    assert not output_path.exists()
    assert not (replacement / "candidate.json").exists()
