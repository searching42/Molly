from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_observation_staging_preflight as preflight_runner
from ai4s_agent.domains.oled_observation_staging_preflight import (
    OledObservationStagingPreflightArtifact,
    _observation_staging_item_digest,
    build_oled_observation_staging_preflight_artifact,
    oled_observation_staging_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_adjudication import (
    _adjudicated_item_digest,
    oled_material_registry_adjudication_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    build_oled_material_registry_entry,
)
from ai4s_agent.oled_observation_staging_preflight import (
    build_oled_observation_staging_preflight_from_files,
    main,
)
from tests.test_oled_material_registry_adjudication import (
    _adjudicate,
    _exact_match_request,
    _no_match_request,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_PREFLIGHT_AT = "2026-07-14T00:30:00+08:00"


def _mapped_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Any, Path]:
    request, request_path = _exact_match_request(tmp_path, monkeypatch)
    adjudication, _, adjudication_path = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="map_to_existing_entity",
        selected_material_id="material-0001",
    )
    return request, request_path, adjudication, adjudication_path


def test_exact_mapping_builds_cell_reference_preflight_without_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path, adjudication, adjudication_path = _mapped_inputs(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "observation-staging-preflight.json"

    artifact = build_oled_observation_staging_preflight_from_files(
        request_artifact_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status.value == "ready_for_exact_source_value_replay"
    assert artifact.request_artifact_sha256 == _sha256_file(request_path)
    assert artifact.registry_adjudication_sha256 == _sha256_file(
        adjudication_path
    )
    assert artifact.request_artifact_digest == request.request_artifact_digest
    assert artifact.registry_adjudication_digest == (
        adjudication.adjudication_artifact_digest
    )
    assert artifact.source_resolution_item_count == 1
    assert artifact.source_adjudicated_item_count == 1
    assert artifact.staging_item_count == 1
    assert artifact.staging_cell_count == 5
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    item = artifact.staging_items[0]
    assert item.selected_existing_material_id == "material-0001"
    assert item.selected_registry_entry.material_id == "material-0001"
    assert item.identity_dependent_cell_count == 5
    assert len(item.identity_dependent_cells) == 5
    assert item.eligible_for_exact_source_value_replay
    assert item.human_registry_mapping_confirmed
    assert not item.source_property_values_present
    assert not item.material_id_attached_to_observations
    assert not artifact.source_property_values_present
    assert not artifact.observations_materialized
    assert not artifact.schema_candidates_created
    assert not artifact.reviewed_evidence_staging
    assert not artifact.gold_records_created
    assert not artifact.dataset_written
    assert OledObservationStagingPreflightArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_new_entity_proposal_is_counted_but_excluded_from_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    adjudication, _, adjudication_path = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision="propose_new_entity",
    )

    artifact = build_oled_observation_staging_preflight_from_files(
        request_artifact_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=tmp_path / "no-mapping-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.status.value == "no_existing_entity_mappings"
    assert artifact.staging_item_count == 0
    assert artifact.staging_cell_count == 0
    assert artifact.staging_items == []
    assert artifact.new_entity_proposal_excluded_group_count == 1
    assert artifact.new_entity_proposal_excluded_cell_count == 5
    assert artifact.unresolved_excluded_group_count == 0
    assert artifact.conflict_deferred_excluded_group_count == 0
    assert not artifact.registry_written
    assert not artifact.observations_materialized
    assert adjudication.new_entity_proposal_count == 1


@pytest.mark.parametrize(
    ("decision", "group_field", "cell_field"),
    (
        ("keep_unresolved", "unresolved_excluded_group_count", "unresolved_excluded_cell_count"),
        ("defer_conflict", "conflict_deferred_excluded_group_count", "conflict_deferred_excluded_cell_count"),
    ),
)
def test_unresolved_and_deferred_items_remain_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    group_field: str,
    cell_field: str,
) -> None:
    request, request_path = _no_match_request(tmp_path, monkeypatch)
    kwargs = {"conflict_reason": "entity_scope_or_chemistry_conflict"}
    if decision != "defer_conflict":
        kwargs = {}
    _, _, adjudication_path = _adjudicate(
        tmp_path,
        request,
        request_path,
        decision=decision,
        **kwargs,
    )

    artifact = build_oled_observation_staging_preflight_from_files(
        request_artifact_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=tmp_path / f"{decision}-preflight.json",
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.staging_items == []
    assert getattr(artifact, group_field) == 1
    assert getattr(artifact, cell_field) == 5


def test_exact_request_file_bytes_must_match_pr_o_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    substituted_request = tmp_path / "same-model-different-bytes.json"
    substituted_request.write_text(
        "\n" + request_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert _sha256_file(substituted_request) != _sha256_file(request_path)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="exact supplied PR-N file"):
        build_oled_observation_staging_preflight_from_files(
            request_artifact_json=substituted_request,
            registry_adjudication_json=adjudication_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert not output_path.exists()


def test_standalone_artifact_revalidates_semantic_chain_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    artifact = build_oled_observation_staging_preflight_from_files(
        request_artifact_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=tmp_path / "preflight.json",
        generated_at=_PREFLIGHT_AT,
    )
    tampered = artifact.model_copy(
        update={"request_artifact_sha256": "sha256:" + "f" * 64},
        deep=True,
    )
    tampered = tampered.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_observation_staging_preflight_artifact_digest(tampered)
            )
        },
        deep=True,
    )

    with pytest.raises(ValidationError, match="request_artifact_sha256 mismatch"):
        OledObservationStagingPreflightArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_redundant_staging_literals_are_rederived_from_pr_o(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    artifact = build_oled_observation_staging_preflight_from_files(
        request_artifact_json=request_path,
        registry_adjudication_json=adjudication_path,
        output_json=tmp_path / "preflight.json",
        generated_at=_PREFLIGHT_AT,
    )
    changed_item = artifact.staging_items[0].model_copy(
        update={"reported_subject_text": "different-reported-subject"},
        deep=True,
    )
    changed_item = changed_item.model_copy(
        update={
            "staging_item_digest": _observation_staging_item_digest(changed_item)
        },
        deep=True,
    )
    tampered = artifact.model_copy(
        update={"staging_items": [changed_item]},
        deep=True,
    )
    tampered = tampered.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_observation_staging_preflight_artifact_digest(tampered)
            )
        },
        deep=True,
    )

    with pytest.raises(ValidationError, match="item derivation mismatch"):
        OledObservationStagingPreflightArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_selected_entry_must_replay_exact_pr_n_snapshot_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path, adjudication, adjudication_path = _mapped_inputs(
        tmp_path,
        monkeypatch,
    )
    source_item = adjudication.adjudicated_items[0]
    selected = source_item.selected_registry_entry
    assert selected is not None
    changed_entry = build_oled_material_registry_entry(
        material_id=selected.material_id,
        canonical_name="changed-but-chemically-equivalent-name",
        aliases=selected.aliases,
        canonical_isomeric_smiles=selected.canonical_isomeric_smiles,
    )
    changed_item = source_item.model_copy(
        update={"selected_registry_entry": changed_entry},
        deep=True,
    )
    changed_item = changed_item.model_copy(
        update={"adjudicated_item_digest": _adjudicated_item_digest(changed_item)},
        deep=True,
    )
    changed_adjudication = adjudication.model_copy(
        update={"adjudicated_items": [changed_item]},
        deep=True,
    )
    changed_adjudication = changed_adjudication.model_copy(
        update={
            "adjudication_artifact_digest": (
                oled_material_registry_adjudication_artifact_digest(
                    changed_adjudication
                )
            )
        },
        deep=True,
    )
    assert changed_adjudication.selected_existing_entries_replayed_from_snapshot

    with pytest.raises(ValueError, match="differs from PR-N snapshot"):
        build_oled_observation_staging_preflight_artifact(
            request=request,
            request_artifact_sha256=_sha256_file(request_path),
            registry_adjudication=changed_adjudication,
            registry_adjudication_sha256=_sha256_file(adjudication_path),
            generated_at=_PREFLIGHT_AT,
        )


@pytest.mark.parametrize("protected_kind", ("request", "adjudication"))
def test_output_cannot_overwrite_either_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    protected = {
        "request": request_path,
        "adjudication": adjudication_path,
    }[protected_kind]
    before = protected.read_bytes()

    with pytest.raises(ValueError, match="overwrite"):
        build_oled_observation_staging_preflight_from_files(
            request_artifact_json=request_path,
            registry_adjudication_json=adjudication_path,
            output_json=protected,
            generated_at=_PREFLIGHT_AT,
        )
    assert protected.read_bytes() == before


def test_symlinked_input_and_output_parent_fail_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    request_alias = tmp_path / "request-alias.json"
    request_alias.symlink_to(request_path)
    output_path = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        build_oled_observation_staging_preflight_from_files(
            request_artifact_json=request_alias,
            registry_adjudication_json=adjudication_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert not output_path.exists()

    output_alias = tmp_path / "output-alias"
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_alias.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_observation_staging_preflight_from_files(
            request_artifact_json=request_path,
            registry_adjudication_json=adjudication_path,
            output_json=output_alias / "must-not-exist.json",
            generated_at=_PREFLIGHT_AT,
        )
    assert not (real_output / "must-not-exist.json").exists()


def test_output_parent_replacement_mid_build_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, _, adjudication_path = _mapped_inputs(tmp_path, monkeypatch)
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.json"
    original = preflight_runner.build_oled_observation_staging_preflight_artifact

    def replace_after_work(**kwargs: Any) -> Any:
        result = original(**kwargs)
        output_parent.rename(displaced)
        output_parent.symlink_to(redirected, target_is_directory=True)
        return result

    monkeypatch.setattr(
        preflight_runner,
        "build_oled_observation_staging_preflight_artifact",
        replace_after_work,
    )

    with pytest.raises(ValueError, match="parent changed"):
        build_oled_observation_staging_preflight_from_files(
            request_artifact_json=request_path,
            registry_adjudication_json=adjudication_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()


def test_cli_failure_is_redacted_and_does_not_publish_output(
    tmp_path: Path,
) -> None:
    sensitive = tmp_path / "token=do-not-disclose.json"
    output_path = tmp_path / "must-not-exist.json"
    stream = StringIO()

    status = main(
        [
            "--request-artifact",
            str(sensitive),
            "--registry-adjudication",
            str(tmp_path / "missing-adjudication.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    assert status == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "observation_staging_preflight_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()
    assert not output_path.exists()
