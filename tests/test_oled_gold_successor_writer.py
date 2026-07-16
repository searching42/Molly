from __future__ import annotations

import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_gold_successor_writer as writer_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_gold_successor_writer import (
    GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    GOLD_SUCCESSOR_WRITE_FILENAME,
    OledGoldSuccessorWriteArtifact,
    build_oled_gold_successor_write_artifact,
    gold_successor_write_receipt_publication_bytes,
    oled_gold_successor_write_artifact_digest,
)
from ai4s_agent.oled_gold_successor_preflight import (
    build_oled_gold_successor_preflight_from_files,
)
from ai4s_agent.oled_gold_successor_writer import (
    build_oled_gold_successor_write_from_files,
    main,
)
from test_oled_gold_successor_preflight import (
    _PREFLIGHT_AT,
    _inputs,
)
from test_oled_supplementary_scoped_candidate_response import _sha256_file


_WRITE_AT = "2026-07-14T01:10:00+08:00"


def _file_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Any, Path, Path, Path, Path]:
    verification, verification_path, candidate_path, current, current_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    preflight_path = tmp_path / "gold-successor-preflight.json"
    preflight = build_oled_gold_successor_preflight_from_files(
        verification_artifact_json=verification_path,
        candidate_snapshot_json=candidate_path,
        current_gold_snapshot_json=current_path,
        output_json=preflight_path,
        generated_at=_PREFLIGHT_AT,
    )
    return (
        preflight,
        preflight_path,
        verification,
        verification_path,
        candidate_path,
        current_path,
        tmp_path / "gold-successor-publication",
    )


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OledGoldSuccessorWriteArtifact:
    (
        preflight,
        preflight_path,
        verification,
        verification_path,
        candidate_path,
        current_path,
        _,
    ) = _file_inputs(tmp_path, monkeypatch)
    return build_oled_gold_successor_write_artifact(
        preflight_artifact=preflight,
        preflight_artifact_sha256=_sha256_file(preflight_path),
        verification_artifact=verification,
        verification_artifact_sha256=_sha256_file(verification_path),
        candidate_snapshot=preflight.candidate_snapshot,
        candidate_snapshot_sha256=_sha256_file(candidate_path),
        prior_gold_snapshot=preflight.current_gold_snapshot,
        prior_gold_snapshot_sha256=_sha256_file(current_path),
        generated_at=_WRITE_AT,
    )


def test_receipt_publishes_and_activates_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)

    assert (
        artifact.status.value
        == "categorical_gold_successor_snapshot_published_and_activated"
    )
    assert artifact.prior_generation == 0
    assert artifact.published_generation == 1
    assert artifact.prior_entry_count == 0
    assert artifact.added_entry_count == 5
    assert artifact.published_entry_count == 5
    assert artifact.published_successor_snapshot == (
        artifact.preflight_artifact.expected_successor_snapshot
    )
    assert artifact.activated_snapshot_id == artifact.successor_snapshot_id
    assert artifact.activated_snapshot_digest == artifact.successor_snapshot_digest
    assert artifact.activation_receipt_created
    assert artifact.categorical_gold_snapshot_activated
    assert not artifact.gold_head_activated
    assert not artifact.mutable_gold_head_pointer_written
    assert not artifact.prior_snapshot_mutated
    assert not artifact.numeric_confidence_score_assigned
    assert not artifact.curated_dataset_written
    assert not artifact.training_eligible


def test_receipt_rejects_rehashed_activation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)
    forged = artifact.model_copy(
        update={
            "activated_snapshot_digest": "sha256:" + "f" * 64,
            "write_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": oled_gold_successor_write_artifact_digest(
                forged
            )
        }
    )
    with pytest.raises(ValidationError, match="activated_snapshot_digest mismatch"):
        OledGoldSuccessorWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_receipt_rejects_parent_sha_unbound_from_pr_ae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)
    forged = artifact.model_copy(
        update={
            "prior_gold_snapshot_sha256": "sha256:" + "f" * 64,
            "write_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": oled_gold_successor_write_artifact_digest(
                forged
            )
        }
    )
    with pytest.raises(
        ValidationError,
        match="prior_gold_snapshot_sha256 mismatch",
    ):
        OledGoldSuccessorWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_write_timestamp_cannot_predate_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        preflight,
        preflight_path,
        verification,
        verification_path,
        candidate_path,
        current_path,
        _,
    ) = _file_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="timestamp reversal"):
        build_oled_gold_successor_write_artifact(
            preflight_artifact=preflight,
            preflight_artifact_sha256=_sha256_file(preflight_path),
            verification_artifact=verification,
            verification_artifact_sha256=_sha256_file(verification_path),
            candidate_snapshot=preflight.candidate_snapshot,
            candidate_snapshot_sha256=_sha256_file(candidate_path),
            prior_gold_snapshot=preflight.current_gold_snapshot,
            prior_gold_snapshot_sha256=_sha256_file(current_path),
            generated_at="2026-07-14T00:59:59+08:00",
        )


def test_file_publication_is_atomic_and_exactly_two_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        preflight,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    artifact = build_oled_gold_successor_write_from_files(
        successor_preflight_json=preflight_path,
        verification_artifact_json=verification_path,
        candidate_snapshot_json=candidate_path,
        current_gold_snapshot_json=current_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        GOLD_SUCCESSOR_WRITE_FILENAME,
        GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    }
    receipt_path = output_dir / GOLD_SUCCESSOR_WRITE_FILENAME
    snapshot_path = output_dir / GOLD_SUCCESSOR_SNAPSHOT_FILENAME
    assert receipt_path.read_bytes() == gold_successor_write_receipt_publication_bytes(
        artifact
    )
    assert OledGoldSuccessorWriteArtifact.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    ) == artifact
    assert artifact.published_successor_snapshot == (
        preflight.expected_successor_snapshot
    )
    assert artifact.published_snapshot_file_sha256 == (
        f"sha256:{hashlib.sha256(snapshot_path.read_bytes()).hexdigest()}"
    )


@pytest.mark.parametrize(
    ("selected_input", "message"),
    (
        ("verification", "verification_sha"),
        ("candidate", "candidate_sha"),
        ("current", "current_sha"),
    ),
)
def test_cas_rejects_reformatted_bound_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_input: str,
    message: str,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    paths = {
        "verification": verification_path,
        "candidate": candidate_path,
        "current": current_path,
    }
    selected = paths[selected_input]
    selected.write_text(
        selected.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


def test_any_input_change_before_publication_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    real_builder = writer_runner.build_oled_gold_successor_write_artifact

    def mutate_after_build(**kwargs: Any) -> Any:
        result = real_builder(**kwargs)
        preflight_path.write_text(
            preflight_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        writer_runner,
        "build_oled_gold_successor_write_artifact",
        mutate_after_build,
    )
    with pytest.raises(ValueError, match="input changed before publication"):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


def test_existing_output_and_partial_publication_fail_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="must be fresh"):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert marker.read_text(encoding="utf-8") == "user data"

    marker.unlink()
    output_dir.rmdir()
    real_write = writer_runner._write_fresh_bytes_at
    calls = 0

    def fail_second_write(directory_descriptor: int, filename: str, payload: bytes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_write(directory_descriptor, filename, payload)

    monkeypatch.setattr(writer_runner, "_write_fresh_bytes_at", fail_second_write)
    with pytest.raises(ValueError, match="directory publication failed"):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".gold-successor-publication.*.tmp"))


def test_atomic_noreplace_preserves_concurrent_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
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
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert (output_dir / "target.txt").read_bytes() == marker


def test_symlinked_input_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    current_link = tmp_path / "current-link.json"
    current_link.symlink_to(current_path)
    with pytest.raises(ValueError):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_link,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        _,
    ) = _file_inputs(tmp_path, monkeypatch)
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    output_dir = output_parent / "publication"
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    real_builder = writer_runner.build_oled_gold_successor_write_artifact

    def replace_parent_after_build(**kwargs: Any) -> Any:
        result = real_builder(**kwargs)
        output_parent.rename(displaced)
        if replacement_kind == "symlink":
            output_parent.symlink_to(redirected, target_is_directory=True)
        else:
            output_parent.mkdir()
        return result

    monkeypatch.setattr(
        writer_runner,
        "build_oled_gold_successor_write_artifact",
        replace_parent_after_build,
    )
    with pytest.raises(ValueError, match="output parent changed|symbolic"):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()
    assert not (redirected / output_dir.name).exists()


def test_output_cannot_replace_input_and_cli_failure_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        _,
    ) = _file_inputs(tmp_path, monkeypatch)
    before = current_path.read_bytes()
    with pytest.raises(ValueError, match="cannot replace an input"):
        build_oled_gold_successor_write_from_files(
            successor_preflight_json=preflight_path,
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_path,
            output_dir=current_path,
            generated_at=_WRITE_AT,
        )
    assert current_path.read_bytes() == before

    bad_current = tmp_path / "bad-current.json"
    write_json(bad_current, {})
    stream = StringIO()
    exit_code = main(
        [
            "--successor-preflight",
            str(preflight_path),
            "--verification-artifact",
            str(verification_path),
            "--candidate-snapshot",
            str(candidate_path),
            "--current-gold-snapshot",
            str(bad_current),
            "--output-dir",
            str(tmp_path / "cli-output"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_successor_write_failed",
        "error_type": "ValidationError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = root / "src/ai4s_agent/domains/oled_gold_successor_writer.py"
    runner = root / "src/ai4s_agent/oled_gold_successor_writer.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
