from __future__ import annotations

import hashlib
import inspect
import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_material_registry_successor_writer as writer_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains import oled_material_registry_successor_writer as writer_domain
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    build_oled_material_registry_entry,
)
from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryEntryDecisionManifest,
    build_oled_material_registry_entry_adjudication_artifact,
)
from ai4s_agent.domains.oled_material_registry_successor_preflight import (
    build_oled_material_registry_successor_preflight_artifact,
)
from ai4s_agent.domains.oled_material_registry_successor_writer import (
    MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
    OledMaterialRegistrySuccessorWriteArtifact,
    build_oled_material_registry_successor_write_artifact,
    material_registry_snapshot_publication_bytes,
    oled_material_registry_successor_write_artifact_digest,
)
from ai4s_agent.oled_material_registry_successor_writer import (
    build_oled_material_registry_successor_write_from_files,
    main,
)
from tests.test_oled_material_registry_successor_preflight import (
    _PREFLIGHT_AT,
    _approved_artifact,
    _build_preflight,
    _original_snapshot,
    _snapshot_with_entries,
)
from tests.test_oled_material_registry_entry_adjudication import (
    _ADJUDICATED_AT,
    _FILE_SHA,
    _manifest_payload,
)
from tests.test_oled_material_registry_entry_proposal_request import (
    _proposal_artifact,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_WRITE_AT = "2026-07-14T00:45:00+08:00"


def _build_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seven: bool = False,
    current_snapshot: Any | None = None,
    adjudication: Any | None = None,
) -> tuple[Any, Any, Any]:
    selected_adjudication = adjudication or _approved_artifact(
        tmp_path,
        monkeypatch,
        seven=seven,
    )
    prior = current_snapshot or _original_snapshot(selected_adjudication)
    preflight = _build_preflight(selected_adjudication, prior)
    artifact = build_oled_material_registry_successor_write_artifact(
        preflight_artifact=preflight,
        preflight_artifact_sha256="sha256:" + "e" * 64,
        prior_registry_snapshot=prior,
        prior_registry_snapshot_sha256=preflight.current_registry_snapshot_sha256,
        generated_at=_WRITE_AT,
    )
    return artifact, preflight, prior


def _file_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seven: bool = False,
) -> tuple[Any, Path, Any, Path, Path]:
    adjudication = _approved_artifact(tmp_path, monkeypatch, seven=seven)
    snapshot = _original_snapshot(adjudication)
    snapshot_path = write_json(
        tmp_path / "current-registry-snapshot.json",
        snapshot.model_dump(mode="json"),
    )
    preflight = build_oled_material_registry_successor_preflight_artifact(
        entry_adjudication=adjudication,
        entry_adjudication_sha256="sha256:" + "c" * 64,
        current_registry_snapshot=snapshot,
        current_registry_snapshot_sha256=_sha256_file(snapshot_path),
        generated_at=_PREFLIGHT_AT,
    )
    preflight_path = write_json(
        tmp_path / "registry-successor-preflight.json",
        preflight.model_dump(mode="json"),
    )
    return preflight, preflight_path, snapshot, snapshot_path, tmp_path / "registry-write"


def test_seven_entry_receipt_publishes_exact_pr_x_successor_without_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, preflight, prior = _build_write(
        tmp_path,
        monkeypatch,
        seven=True,
    )

    assert artifact.status.value == "registry_successor_snapshot_published"
    assert artifact.prior_entry_count == 0
    assert artifact.added_entry_count == 7
    assert artifact.added_entry_cell_count == 35
    assert artifact.published_entry_count == 7
    assert artifact.published_successor_snapshot == (
        preflight.expected_successor_snapshot
    )
    assert artifact.prior_registry_snapshot == prior
    assert artifact.registry_written is True
    assert artifact.material_id_assigned is True
    assert artifact.registry_entry_created is True
    assert artifact.registry_head_activated is False
    assert artifact.activation_receipt_created is False
    assert artifact.observations_materialized is False
    assert artifact.gold_records_created is False
    assert artifact.dataset_written is False
    assert artifact.device_only_records_admitted is False
    assert artifact.standalone_input_bytes_revalidation_supported is False


def test_unrelated_prior_entry_is_preserved_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    prior_entry = build_oled_material_registry_entry(
        material_id="material:" + "a" * 32,
        canonical_name="unrelated-current-entry",
        canonical_isomeric_smiles="N#N",
    )
    prior = _snapshot_with_entries(adjudication, [prior_entry])
    artifact, _, _ = _build_write(
        tmp_path,
        monkeypatch,
        current_snapshot=prior,
        adjudication=adjudication,
    )

    published = {
        entry.material_id: entry
        for entry in artifact.published_successor_snapshot.entries
    }
    assert artifact.prior_entry_count == 1
    assert artifact.added_entry_count == 1
    assert artifact.published_entry_count == 2
    assert published[prior_entry.material_id].model_dump(mode="json") == (
        prior_entry.model_dump(mode="json")
    )


def test_receipt_rejects_rehashed_successor_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _ = _build_write(tmp_path, monkeypatch)
    forged = artifact.model_copy(
        update={
            "added_entry_cell_count": artifact.added_entry_cell_count + 1,
            "write_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": (
                oled_material_registry_successor_write_artifact_digest(forged)
            )
        }
    )
    with pytest.raises(ValidationError, match="added_entry_cell_count mismatch"):
        OledMaterialRegistrySuccessorWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_receipt_rejects_parent_sha_unbound_from_pr_x(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _, _ = _build_write(tmp_path, monkeypatch)
    forged = artifact.model_copy(
        update={
            "prior_registry_snapshot_sha256": "sha256:" + "f" * 64,
            "write_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": (
                oled_material_registry_successor_write_artifact_digest(forged)
            )
        }
    )
    with pytest.raises(ValidationError, match="prior_registry_snapshot_sha256"):
        OledMaterialRegistrySuccessorWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_no_change_preflight_is_not_published(
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
    prior = _original_snapshot(adjudication)
    preflight = _build_preflight(adjudication, prior)
    with pytest.raises(ValueError, match="no snapshot to publish"):
        build_oled_material_registry_successor_write_artifact(
            preflight_artifact=preflight,
            preflight_artifact_sha256="sha256:" + "e" * 64,
            prior_registry_snapshot=prior,
            prior_registry_snapshot_sha256=(
                preflight.current_registry_snapshot_sha256
            ),
            generated_at=_WRITE_AT,
        )


def test_write_timestamp_cannot_predate_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudication = _approved_artifact(tmp_path, monkeypatch)
    prior = _original_snapshot(adjudication)
    preflight = _build_preflight(adjudication, prior)
    with pytest.raises(ValidationError, match="timestamp reversal"):
        build_oled_material_registry_successor_write_artifact(
            preflight_artifact=preflight,
            preflight_artifact_sha256="sha256:" + "e" * 64,
            prior_registry_snapshot=prior,
            prior_registry_snapshot_sha256=(
                preflight.current_registry_snapshot_sha256
            ),
            generated_at="2026-07-14T00:39:00+08:00",
        )


def test_file_publication_is_atomic_and_exactly_two_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
        seven=True,
    )
    artifact = build_oled_material_registry_successor_write_from_files(
        preflight_artifact_json=preflight_path,
        current_registry_snapshot_json=snapshot_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
        MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    }
    receipt_path = output_dir / MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME
    published_path = output_dir / MATERIAL_REGISTRY_SNAPSHOT_FILENAME
    assert OledMaterialRegistrySuccessorWriteArtifact.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    ) == artifact
    assert published_path.read_bytes() == material_registry_snapshot_publication_bytes(
        preflight.expected_successor_snapshot
    )
    assert artifact.published_snapshot_file_sha256 == (
        f"sha256:{hashlib.sha256(published_path.read_bytes()).hexdigest()}"
    )
    assert artifact.preflight_artifact_sha256 == _sha256_file(preflight_path)


def test_output_cannot_replace_an_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, _ = _file_inputs(tmp_path, monkeypatch)
    before = snapshot_path.read_bytes()
    with pytest.raises(ValueError, match="cannot replace an input"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=snapshot_path,
            generated_at=_WRITE_AT,
        )
    assert snapshot_path.read_bytes() == before


def test_compare_and_swap_rejects_semantically_same_reformatted_parent_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Registry bytes do not match PR-X"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


def test_input_change_between_derivation_and_publication_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    real_builder = writer_runner.build_oled_material_registry_successor_write_artifact

    def mutate_after_build(**kwargs: Any) -> Any:
        result = real_builder(**kwargs)
        snapshot_path.write_text(
            snapshot_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        writer_runner,
        "build_oled_material_registry_successor_write_artifact",
        mutate_after_build,
    )
    with pytest.raises(ValueError, match="input changed before publication"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


def test_existing_output_directory_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="must be fresh"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert marker.read_text(encoding="utf-8") == "user data"


def test_partial_publication_is_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    real_write = writer_runner._write_fresh_bytes_at
    call_count = 0

    def fail_second_write(directory_descriptor: int, filename: str, payload: bytes):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected publication failure")
        return real_write(directory_descriptor, filename, payload)

    monkeypatch.setattr(writer_runner, "_write_fresh_bytes_at", fail_second_write)
    with pytest.raises(ValueError, match="directory publication failed"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".registry-write.*.tmp"))


def test_temp_directory_name_swap_cannot_publish_or_clean_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    real_rename = writer_runner._rename_directory_noreplace_at
    displaced_name = "displaced-owned-directory"
    replacement_marker = b"replacement must survive"
    injected = False

    def swap_at_rename(parent_descriptor: int, temp_name: str, output_name: str):
        nonlocal injected
        if injected:
            return real_rename(parent_descriptor, temp_name, output_name)
        injected = True
        os.rename(
            temp_name,
            displaced_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.mkdir(temp_name, mode=0o700, dir_fd=parent_descriptor)
        replacement_descriptor = os.open(
            temp_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            writer_runner._write_fresh_bytes_at(
                replacement_descriptor,
                "replacement.txt",
                replacement_marker,
            )
            os.fsync(replacement_descriptor)
        finally:
            os.close(replacement_descriptor)
        return real_rename(parent_descriptor, temp_name, output_name)

    monkeypatch.setattr(writer_runner, "_rename_directory_noreplace_at", swap_at_rename)
    with pytest.raises(ValueError, match="published directory inode mismatch"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()
    displaced = tmp_path / displaced_name
    assert {path.name for path in displaced.iterdir()} == {
        MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
        MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    }
    replacements = list(tmp_path.glob(".registry-write.*.tmp"))
    assert len(replacements) == 1
    assert (replacements[0] / "replacement.txt").read_bytes() == replacement_marker


def test_atomic_noreplace_preserves_concurrent_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )
    real_commit = writer_runner._atomic_rename_owned_directory_noreplace
    marker = b"concurrent target must survive"

    def create_target_before_commit(**kwargs: Any):
        parent_descriptor = kwargs["parent_descriptor"]
        output_name = kwargs["output_name"]
        os.mkdir(output_name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(
            output_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            writer_runner._write_fresh_bytes_at(descriptor, "target.txt", marker)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return real_commit(**kwargs)

    monkeypatch.setattr(
        writer_runner,
        "_atomic_rename_owned_directory_noreplace",
        create_target_before_commit,
    )
    with pytest.raises(ValueError, match="must be fresh"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert (output_dir / "target.txt").read_bytes() == marker
    assert not (output_dir / MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME).exists()


@pytest.mark.parametrize("replacement_kind", ("directory", "symlink"))
def test_output_parent_replacement_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        workspace,
        monkeypatch,
    )
    displaced = tmp_path / "workspace-displaced"
    redirected = tmp_path / "workspace-redirected"
    redirected.mkdir()
    real_commit = writer_runner._atomic_rename_owned_directory_noreplace

    def replace_parent_before_commit(**kwargs: Any):
        workspace.rename(displaced)
        if replacement_kind == "symlink":
            workspace.symlink_to(redirected, target_is_directory=True)
        else:
            workspace.mkdir()
        return real_commit(**kwargs)

    monkeypatch.setattr(
        writer_runner,
        "_atomic_rename_owned_directory_noreplace",
        replace_parent_before_commit,
    )
    with pytest.raises(ValueError, match="output parent changed"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not (displaced / output_dir.name).exists()
    assert not (redirected / output_dir.name).exists()
    assert not (workspace / output_dir.name).exists()


def test_unavailable_atomic_noreplace_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
    )

    def unavailable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ValueError("atomic no-replace directory rename is unavailable")

    monkeypatch.setattr(
        writer_runner,
        "_rename_directory_noreplace_at",
        unavailable,
    )
    with pytest.raises(ValueError, match="atomic no-replace"):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".registry-write.*.tmp"))


def test_symlinked_input_and_output_parent_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, _, snapshot_path, _ = _file_inputs(tmp_path, monkeypatch)
    snapshot_alias = tmp_path / "snapshot-alias.json"
    snapshot_alias.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_alias,
            output_dir=tmp_path / "must-not-exist",
            generated_at=_WRITE_AT,
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=preflight_path,
            current_registry_snapshot_json=snapshot_path,
            output_dir=linked_parent / "must-not-exist",
            generated_at=_WRITE_AT,
        )
    assert not (real_parent / "must-not-exist").exists()


def test_cli_failure_is_redacted_and_does_not_publish(tmp_path: Path) -> None:
    secret = "secret-token-abc123"
    output_dir = tmp_path / "output"
    stream = StringIO()
    code = main(
        [
            "--successor-preflight",
            str(tmp_path / f"missing-{secret}.json"),
            "--current-registry-snapshot",
            str(tmp_path / "missing-registry.json"),
            "--output-dir",
            str(output_dir),
        ],
        stdout=stream,
    )
    payload = json.loads(stream.getvalue())
    assert code == 2
    assert payload["error_code"] == "material_registry_successor_write_failed"
    assert secret not in stream.getvalue()
    assert not output_dir.exists()


def test_production_implementation_is_generic_not_paper016_specific() -> None:
    source = inspect.getsource(writer_domain) + inspect.getsource(writer_runner)
    assert "paper016" not in source
    assert "TDBA" not in source
