from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_material_registry_successor_preflight as preflight_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains import (
    oled_material_registry_successor_preflight as preflight_domain,
)
from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryEntryDecisionManifest,
    build_oled_material_registry_entry_adjudication_artifact,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    build_oled_material_registry_entry,
    build_oled_material_registry_snapshot,
)
from ai4s_agent.domains.oled_material_registry_successor_preflight import (
    OledMaterialRegistrySuccessorPreflightArtifact,
    build_oled_material_registry_successor_preflight_artifact,
    oled_material_registry_successor_preflight_artifact_digest,
)
from ai4s_agent.oled_material_registry_successor_preflight import (
    build_oled_material_registry_successor_preflight_from_files,
)
from tests.test_oled_material_registry_entry_adjudication import (
    _ADJUDICATED_AT,
    _FILE_SHA,
    _build,
    _manifest_payload,
)
from tests.test_oled_material_registry_entry_proposal_request import (
    _proposal_artifact,
    _seven_unique_no_match_request,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_PREFLIGHT_AT = "2026-07-14T00:40:00+08:00"
_CURRENT_AT = "2026-07-14T00:36:00+08:00"


def _approved_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seven: bool = False,
) -> Any:
    artifact, _, _ = _build(
        tmp_path,
        monkeypatch,
        request_factory=_seven_unique_no_match_request if seven else None,
    )
    return artifact


def _original_snapshot(artifact: Any) -> Any:
    return artifact.request.resolution_request.registry_snapshot


def _snapshot_with_entries(artifact: Any, entries: list[Any]) -> Any:
    original = _original_snapshot(artifact)
    return build_oled_material_registry_snapshot(
        registry_id=original.registry_id,
        registry_version="current-registry-0002",
        generated_at=_CURRENT_AT,
        entries=entries,
    )


def _build_preflight(
    artifact: Any,
    snapshot: Any | None = None,
) -> OledMaterialRegistrySuccessorPreflightArtifact:
    selected_snapshot = snapshot or _original_snapshot(artifact)
    return build_oled_material_registry_successor_preflight_artifact(
        entry_adjudication=artifact,
        entry_adjudication_sha256="sha256:" + "c" * 64,
        current_registry_snapshot=selected_snapshot,
        current_registry_snapshot_sha256="sha256:" + "d" * 64,
        generated_at=_PREFLIGHT_AT,
    )


def test_seven_approved_candidates_plan_seven_append_only_additions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch, seven=True)
    artifact = _build_preflight(adjudication)

    assert artifact.status.value == "ready_for_registry_successor_write"
    assert artifact.eligible_candidate_count == 7
    assert artifact.eligible_candidate_cell_count == 35
    assert artifact.planned_addition_count == 7
    assert artifact.planned_addition_cell_count == 35
    assert artifact.prior_entry_count == 0
    assert artifact.expected_entry_count == 7
    assert artifact.expected_successor_snapshot is not None
    assert artifact.expected_successor_snapshot.entry_count == 7
    assert artifact.expected_successor_snapshot.snapshot_digest == (
        artifact.expected_successor_snapshot_digest
    )
    assert artifact.successor_registry_version.startswith("successor-")
    assert not artifact.registry_written
    assert not artifact.registry_head_activated
    assert not artifact.observations_materialized
    assert not artifact.gold_records_created
    assert not artifact.dataset_written
    assert not artifact.device_only_records_admitted
    assert artifact.standalone_input_bytes_revalidation_supported is False
    assert artifact.current_snapshot_lineage_receipt_bound is False


def test_unrelated_current_entries_are_preserved_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    prior_entry = build_oled_material_registry_entry(
        material_id="material:" + "a" * 32,
        canonical_name="unrelated-current-entry",
        canonical_isomeric_smiles="N#N",
    )
    current = _snapshot_with_entries(adjudication, [prior_entry])
    artifact = _build_preflight(adjudication, current)

    assert artifact.prior_entry_count == 1
    assert artifact.planned_addition_count == 1
    assert artifact.expected_entry_count == 2
    assert artifact.expected_successor_snapshot is not None
    preserved = {
        entry.material_id: entry
        for entry in artifact.expected_successor_snapshot.entries
    }[prior_entry.material_id]
    assert preserved.model_dump(mode="json") == prior_entry.model_dump(mode="json")


def test_empty_approved_roster_is_an_explicit_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, *_ = _proposal_artifact(tmp_path, monkeypatch)
    manifest_payload = _manifest_payload(
        request,
        _FILE_SHA,
        decision="keep_unresolved",
    )
    adjudication = build_oled_material_registry_entry_adjudication_artifact(
        request=request,
        request_artifact_sha256=_FILE_SHA,
        decision_manifest=OledMaterialRegistryEntryDecisionManifest.model_validate(
            manifest_payload
        ),
        decision_manifest_sha256="sha256:" + "b" * 64,
        generated_at=_ADJUDICATED_AT,
    )
    artifact = _build_preflight(adjudication)

    assert artifact.status.value == "no_registry_changes_required"
    assert artifact.planned_addition_count == 0
    assert artifact.expected_entry_count == artifact.prior_entry_count
    assert artifact.expected_successor_snapshot is None
    assert artifact.expected_successor_snapshot_digest == ""
    assert artifact.successor_registry_version == ""


def test_current_snapshot_material_id_collision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    candidate = adjudication.adjudicated_items[0].approved_entry_candidate
    occupied = build_oled_material_registry_entry(
        material_id=candidate.registry_entry.material_id,
        canonical_name="occupied-current-entry",
        canonical_isomeric_smiles="C",
    )
    with pytest.raises(ValueError, match="material ID is already occupied"):
        _build_preflight(adjudication, _snapshot_with_entries(adjudication, [occupied]))


def test_current_snapshot_name_collision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    candidate = adjudication.adjudicated_items[0].approved_entry_candidate
    occupied = build_oled_material_registry_entry(
        material_id="material:" + "f" * 32,
        canonical_name=candidate.registry_entry.canonical_name,
        canonical_isomeric_smiles="C",
    )
    with pytest.raises(ValueError, match="name conflicts"):
        _build_preflight(adjudication, _snapshot_with_entries(adjudication, [occupied]))


def test_current_snapshot_graph_collision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    candidate = adjudication.adjudicated_items[0].approved_entry_candidate
    occupied = build_oled_material_registry_entry(
        material_id="material:" + "e" * 32,
        canonical_name="same-graph-current-entry",
        canonical_isomeric_smiles=candidate.registry_entry.canonical_isomeric_smiles,
    )
    with pytest.raises(ValueError, match="canonical structure already exists"):
        _build_preflight(adjudication, _snapshot_with_entries(adjudication, [occupied]))


def test_snapshot_older_than_pr_w_bound_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    original = _original_snapshot(adjudication)
    stale = build_oled_material_registry_snapshot(
        registry_id=original.registry_id,
        registry_version="stale-registry",
        generated_at="2026-07-12T00:00:00+08:00",
        entries=[],
    )
    with pytest.raises(ValueError, match="predates the PR-W snapshot"):
        _build_preflight(adjudication, stale)


def test_preflight_timestamp_cannot_predate_pr_w(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="timestamp reversal"):
        build_oled_material_registry_successor_preflight_artifact(
            entry_adjudication=adjudication,
            entry_adjudication_sha256="sha256:" + "c" * 64,
            current_registry_snapshot=_original_snapshot(adjudication),
            current_registry_snapshot_sha256="sha256:" + "d" * 64,
            generated_at="2026-07-14T00:34:00+08:00",
        )


def test_rehashed_successor_version_tamper_fails_semantic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build_preflight(_approved_artifact(tmp_path, monkeypatch))
    tampered = artifact.model_copy(
        update={"successor_registry_version": "successor-rehashed-tamper"},
        deep=True,
    )
    tampered = tampered.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_material_registry_successor_preflight_artifact_digest(tampered)
            )
        }
    )
    with pytest.raises(ValidationError, match="successor_registry_version mismatch"):
        OledMaterialRegistrySuccessorPreflightArtifact.model_validate(
            tampered.model_dump(mode="json")
        )


def test_file_entry_binds_exact_bytes_and_never_overwrites_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    snapshot = _original_snapshot(adjudication)
    adjudication_path = write_json(
        tmp_path / "entry-adjudication.json",
        adjudication.model_dump(mode="json"),
    )
    snapshot_path = write_json(
        tmp_path / "current-registry.json",
        snapshot.model_dump(mode="json"),
    )
    output_path = tmp_path / "registry-successor-preflight.json"
    artifact = build_oled_material_registry_successor_preflight_from_files(
        entry_adjudication_json=adjudication_path,
        current_registry_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )
    assert artifact.entry_adjudication_sha256 == _sha256_file(adjudication_path)
    assert artifact.current_registry_snapshot_sha256 == _sha256_file(snapshot_path)
    assert output_path.is_file()

    before = snapshot_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_material_registry_successor_preflight_from_files(
            entry_adjudication_json=adjudication_path,
            current_registry_snapshot_json=snapshot_path,
            output_json=snapshot_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert snapshot_path.read_bytes() == before


def test_symlinked_input_and_output_parent_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    adjudication_path = write_json(
        tmp_path / "entry-adjudication.json",
        adjudication.model_dump(mode="json"),
    )
    snapshot_path = write_json(
        tmp_path / "current-registry.json",
        _original_snapshot(adjudication).model_dump(mode="json"),
    )
    snapshot_alias = tmp_path / "current-registry-alias.json"
    snapshot_alias.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_preflight_from_files(
            entry_adjudication_json=adjudication_path,
            current_registry_snapshot_json=snapshot_alias,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_PREFLIGHT_AT,
        )

    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_preflight_from_files(
            entry_adjudication_json=adjudication_path,
            current_registry_snapshot_json=snapshot_path,
            output_json=output_alias / "must-not-exist.json",
            generated_at=_PREFLIGHT_AT,
        )
    assert not (real_parent / "must-not-exist.json").exists()


@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    adjudication_path = write_json(
        tmp_path / "entry-adjudication.json",
        adjudication.model_dump(mode="json"),
    )
    snapshot_path = write_json(
        tmp_path / "current-registry.json",
        _original_snapshot(adjudication).model_dump(mode="json"),
    )
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.json"
    original_builder = (
        preflight_runner.build_oled_material_registry_successor_preflight_artifact
    )

    def replace_parent_after_build(**kwargs: Any) -> Any:
        result = original_builder(**kwargs)
        output_parent.rename(displaced)
        if replacement_kind == "symlink":
            output_parent.symlink_to(redirected, target_is_directory=True)
        else:
            output_parent.mkdir()
        return result

    monkeypatch.setattr(
        preflight_runner,
        "build_oled_material_registry_successor_preflight_artifact",
        replace_parent_after_build,
    )
    with pytest.raises(ValueError, match="parent changed"):
        build_oled_material_registry_successor_preflight_from_files(
            entry_adjudication_json=adjudication_path,
            current_registry_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()
    assert not (output_parent / output_path.name).exists()


def test_production_implementation_is_generic_not_paper016_specific() -> None:
    source = inspect.getsource(preflight_domain) + inspect.getsource(preflight_runner)
    assert "paper016" not in source
    assert "TDBA" not in source
