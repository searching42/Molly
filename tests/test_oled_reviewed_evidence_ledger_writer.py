from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_reviewed_evidence_ledger_writer as writer_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_reviewed_evidence_ledger_writer import (
    OledReviewedEvidenceLedgerWriteArtifact,
    OledReviewedEvidenceLedgerWriteStatus,
    oled_reviewed_evidence_ledger_write_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerEntry,
    OledReviewedEvidenceLedgerEntryStatus,
    OledReviewedEvidenceLedgerSnapshot,
    _ledger_entry_digest,
    _ledger_projection_payload,
    _projection_id_from_fields,
    _projection_payload_digest,
    build_oled_reviewed_evidence_ledger_snapshot,
)
from ai4s_agent.oled_reviewed_evidence_ledger_writer import (
    build_oled_reviewed_evidence_ledger_write_from_files,
    main,
)
from ai4s_agent.oled_reviewed_evidence_staging_preflight import (
    build_oled_reviewed_evidence_staging_preflight_from_files,
)
from tests.test_oled_reviewed_evidence_staging_preflight import (
    _alternate_source_entry,
    _entry_for_candidate,
    _inputs,
)


_PREFLIGHT_AT = "2026-07-14T00:42:00+08:00"
_WRITE_AT = "2026-07-14T00:43:00+08:00"


def _preflight_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger: OledReviewedEvidenceLedgerSnapshot | None = None,
    inputs: tuple[object, Path, OledReviewedEvidenceLedgerSnapshot, Path]
    | None = None,
) -> tuple[object, Path, Path, Path]:
    materialization, materialization_path, genesis, ledger_path = (
        inputs or _inputs(tmp_path, monkeypatch)
    )
    selected = ledger or genesis
    write_json(ledger_path, selected.model_dump(mode="json"))
    preflight_path = tmp_path / "reviewed-evidence-staging-preflight.json"
    build_oled_reviewed_evidence_staging_preflight_from_files(
        materialization_artifact_json=materialization_path,
        ledger_snapshot_json=ledger_path,
        output_json=preflight_path,
        generated_at=_PREFLIGHT_AT,
    )
    return materialization, preflight_path, ledger_path, tmp_path / "ledger-write"


def test_genesis_write_appends_five_active_entries_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )

    artifact = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    assert artifact.status == OledReviewedEvidenceLedgerWriteStatus.LEDGER_UPDATED
    assert artifact.prior_entry_count == 0
    assert artifact.added_entry_count == 5
    assert artifact.active_entry_count_added == 5
    assert artifact.quarantined_entry_count_added == 0
    assert artifact.next_entry_count == 5
    assert all(
        entry.status == OledReviewedEvidenceLedgerEntryStatus.ACTIVE
        for entry in artifact.next_ledger_snapshot.entries
    )
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "reviewed_evidence_ledger_snapshot.json",
        "reviewed_evidence_ledger_write.json",
    ]
    assert OledReviewedEvidenceLedgerWriteArtifact.model_validate_json(
        (output_dir / "reviewed_evidence_ledger_write.json").read_text(
            encoding="utf-8"
        )
    ) == artifact
    assert OledReviewedEvidenceLedgerSnapshot.model_validate_json(
        (output_dir / "reviewed_evidence_ledger_snapshot.json").read_text(
            encoding="utf-8"
        )
    ) == artifact.next_ledger_snapshot


def test_exact_replay_keeps_the_snapshot_semantically_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    materialization, _, genesis, _ = inputs
    entries = [
        _entry_for_candidate(materialization, candidate)
        for candidate in materialization.observation_candidates
    ]
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=entries,
        generated_at="2026-07-14T00:41:00+08:00",
        snapshot_id="reviewed-evidence-ledger:exact-replay",
    )
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    assert artifact.status == OledReviewedEvidenceLedgerWriteStatus.NO_CHANGES_REQUIRED
    assert artifact.added_entry_count == 0
    assert artifact.exact_replay_noop_count == 5
    assert artifact.next_ledger_snapshot == artifact.prior_ledger_snapshot


def test_value_conflict_is_appended_only_as_quarantined(
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
        snapshot_id="reviewed-evidence-ledger:conflict",
    )
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    added = [
        entry
        for entry in artifact.next_ledger_snapshot.entries
        if entry.entry_id in artifact.added_entry_ids
    ]
    assert artifact.quarantined_entry_count_added == 1
    assert artifact.active_entry_count_added == 4
    assert sum(
        entry.status == OledReviewedEvidenceLedgerEntryStatus.QUARANTINED
        for entry in added
    ) == 1


def test_compare_and_swap_rejects_changed_current_ledger_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )
    changed = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[
            _entry_for_candidate(
                materialization,
                materialization.observation_candidates[0],
            )
        ],
        generated_at="2026-07-14T00:41:30+08:00",
        snapshot_id="reviewed-evidence-ledger:changed-after-preflight",
    )
    write_json(ledger_path, changed.model_dump(mode="json"))

    with pytest.raises(ValueError, match="compare-and-swap ledger bytes"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert not output_dir.exists()


def test_unreviewed_same_source_revision_is_refused_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    materialization, _, _, _ = inputs
    candidate = materialization.observation_candidates[0]
    original = _entry_for_candidate(materialization, candidate)
    payload = original.model_dump(mode="python")
    payload["source_candidate_digest"] = "sha256:" + "d" * 64
    payload["projection_id"] = _projection_id_from_fields(
        source_claim_id=original.source_claim_id,
        source_candidate_digest=payload["source_candidate_digest"],
        selected_material_id=original.selected_material_id,
        registry_entry_digest=original.registry_entry_digest,
        cell_disposition_digest=original.cell_disposition_digest,
        semantic_contract_digest=original.semantic_contract_digest,
    )
    payload["entry_id"] = (
        f"reviewed-evidence:{payload['projection_id'].split(':', 1)[-1]}"
    )
    payload["projection_payload_digest"] = "sha256:" + "0" * 64
    payload["entry_digest"] = "sha256:" + "0" * 64
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["projection_payload_digest"] = _projection_payload_digest(
        _ledger_projection_payload(provisional)
    )
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["entry_digest"] = _ledger_entry_digest(provisional)
    revision = OledReviewedEvidenceLedgerEntry.model_validate(payload)
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[revision],
        generated_at="2026-07-14T00:41:00+08:00",
        snapshot_id="reviewed-evidence-ledger:unreviewed-revision",
    )
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    with pytest.raises(ValueError, match="roster-bound exception decision"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert not output_dir.exists()


def test_existing_output_directory_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="must be fresh"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert marker.read_text(encoding="utf-8") == "user data"


def test_partial_directory_publication_is_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
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

    monkeypatch.setattr(
        writer_runner,
        "_write_fresh_bytes_at",
        fail_second_write,
    )

    with pytest.raises(ValueError, match="directory publication failed"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".ledger-write.*.tmp"))


def test_temp_directory_name_swap_cannot_publish_or_clean_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
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

    monkeypatch.setattr(
        writer_runner,
        "_rename_directory_noreplace_at",
        swap_at_rename,
    )

    with pytest.raises(ValueError, match="published directory inode mismatch"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert not output_dir.exists()
    displaced = tmp_path / displaced_name
    assert displaced.is_dir()
    assert {
        path.name for path in displaced.iterdir()
    } == {
        "reviewed_evidence_ledger_snapshot.json",
        "reviewed_evidence_ledger_write.json",
    }
    replacements = list(tmp_path.glob(".ledger-write.*.tmp"))
    assert len(replacements) == 1
    assert (replacements[0] / "replacement.txt").read_bytes() == replacement_marker


def test_atomic_noreplace_preserves_target_created_after_fresh_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )
    real_commit = writer_runner._atomic_rename_owned_directory_noreplace
    target_marker = b"concurrent target must survive"

    def create_target_before_commit(**kwargs):
        parent_descriptor = kwargs["parent_descriptor"]
        output_name = kwargs["output_name"]
        os.mkdir(output_name, mode=0o700, dir_fd=parent_descriptor)
        target_descriptor = os.open(
            output_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            writer_runner._write_fresh_bytes_at(
                target_descriptor,
                "target.txt",
                target_marker,
            )
            os.fsync(target_descriptor)
        finally:
            os.close(target_descriptor)
        return real_commit(**kwargs)

    monkeypatch.setattr(
        writer_runner,
        "_atomic_rename_owned_directory_noreplace",
        create_target_before_commit,
    )

    with pytest.raises(ValueError, match="must be fresh"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert output_dir.is_dir()
    assert (output_dir / "target.txt").read_bytes() == target_marker
    assert not (output_dir / "reviewed_evidence_ledger_write.json").exists()
    assert not list(tmp_path.glob(".ledger-write.*.tmp"))


def test_symbolic_output_parent_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, _ = _preflight_inputs(tmp_path, monkeypatch)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output_dir = linked_parent / "ledger-write"

    with pytest.raises(ValueError, match="symbolic|unsafe"):
        build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=preflight_path,
            current_ledger_snapshot_json=ledger_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    assert not (real_parent / "ledger-write").exists()


def test_write_artifact_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )
    artifact = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    forged = artifact.model_copy(
        update={
            "active_entry_count_added": 4,
            "write_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": (
                oled_reviewed_evidence_ledger_write_artifact_digest(forged)
            )
        }
    )

    with pytest.raises(ValidationError, match="active_entry_count_added"):
        OledReviewedEvidenceLedgerWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_receipt_rejects_prior_file_hash_unbound_from_pr_r(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
    )
    artifact = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    forged = artifact.model_copy(
        update={
            "prior_ledger_snapshot_sha256": "sha256:" + "a" * 64,
            "write_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": (
                oled_reviewed_evidence_ledger_write_artifact_digest(forged)
            )
        }
    )

    with pytest.raises(ValidationError, match="prior ledger bytes"):
        OledReviewedEvidenceLedgerWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_cli_failure_is_redacted_and_does_not_publish(
    tmp_path: Path,
) -> None:
    secret = "secret-token-abc123"
    output_dir = tmp_path / "output"
    stream = StringIO()

    code = main(
        [
            "--staging-preflight",
            str(tmp_path / f"missing-{secret}.json"),
            "--current-ledger-snapshot",
            str(tmp_path / "missing-ledger.json"),
            "--output-dir",
            str(output_dir),
        ],
        stdout=stream,
    )

    payload = json.loads(stream.getvalue())
    assert code == 2
    assert payload["error_code"] == "reviewed_evidence_ledger_write_failed"
    assert secret not in stream.getvalue()
    assert not output_dir.exists()
