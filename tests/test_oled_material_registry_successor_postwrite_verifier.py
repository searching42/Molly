from __future__ import annotations

import hashlib
import inspect
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import (
    oled_material_registry_successor_postwrite_verifier as verifier_runner,
)
from ai4s_agent.domains import (
    oled_material_registry_successor_postwrite_verifier as verifier_domain,
)
from ai4s_agent.domains.oled_material_registry_successor_postwrite_verifier import (
    OledMaterialRegistrySuccessorPostwriteVerificationArtifact,
    _independently_replay_registry_postwrite,
    build_oled_material_registry_successor_postwrite_verification_artifact,
    material_registry_write_receipt_publication_bytes,
    oled_material_registry_successor_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_successor_writer import (
    MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
    material_registry_snapshot_publication_bytes,
)
from ai4s_agent.oled_material_registry_successor_postwrite_verifier import (
    build_oled_material_registry_successor_postwrite_verification_from_files,
    main,
)
from ai4s_agent.oled_material_registry_successor_writer import (
    build_oled_material_registry_successor_write_from_files,
)
from tests.test_oled_material_registry_successor_writer import (
    _WRITE_AT,
    _build_write,
    _file_inputs,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_VERIFY_AT = "2026-07-14T00:50:00+08:00"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seven: bool = False,
) -> tuple[Any, Path, Path]:
    _, preflight_path, _, snapshot_path, output_dir = _file_inputs(
        tmp_path,
        monkeypatch,
        seven=seven,
    )
    receipt = build_oled_material_registry_successor_write_from_files(
        preflight_artifact_json=preflight_path,
        current_registry_snapshot_json=snapshot_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    return (
        receipt,
        output_dir / MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
        output_dir / MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    )


def test_postwrite_verifier_replays_seven_entry_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, snapshot_path = _write_result(
        tmp_path,
        monkeypatch,
        seven=True,
    )
    output_path = tmp_path / "registry-postwrite-verification.json"
    artifact = (
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    )

    assert artifact.status.value == "registry_successor_publication_verified"
    assert artifact.prior_entry_count == 0
    assert artifact.verified_added_entry_count == 7
    assert artifact.verified_added_entry_cell_count == 35
    assert artifact.published_entry_count == 7
    assert artifact.verified_added_material_ids == receipt.added_material_ids
    assert artifact.verified_added_entry_digests == receipt.added_entry_digests
    assert artifact.published_snapshot == receipt.published_successor_snapshot
    assert artifact.write_artifact_sha256 == _sha256_file(receipt_path)
    assert artifact.published_snapshot_sha256 == _sha256_file(snapshot_path)
    assert artifact.published_registry_snapshot_verified is True
    assert artifact.eligible_for_explicit_pr_n_input is True
    assert artifact.registry_written is False
    assert artifact.registry_head_activated is False
    assert artifact.activation_receipt_created is False
    assert artifact.observations_materialized is False
    assert artifact.gold_records_created is False
    assert artifact.dataset_written is False
    assert OledMaterialRegistrySuccessorPostwriteVerificationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_direct_verifier_replays_single_entry_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _build_write(tmp_path, monkeypatch)
    published = receipt.published_successor_snapshot
    artifact = build_oled_material_registry_successor_postwrite_verification_artifact(
        write_artifact=receipt,
        write_artifact_sha256=_sha256_bytes(
            material_registry_write_receipt_publication_bytes(receipt)
        ),
        published_snapshot=published,
        published_snapshot_sha256=_sha256_bytes(
            material_registry_snapshot_publication_bytes(published)
        ),
        generated_at=_VERIFY_AT,
    )
    assert artifact.verified_added_entry_count == 1
    assert artifact.published_entry_count == 1


def test_different_valid_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _build_write(tmp_path, monkeypatch)
    different = receipt.prior_registry_snapshot
    with pytest.raises(ValueError, match="snapshot file SHA-256 mismatch"):
        build_oled_material_registry_successor_postwrite_verification_artifact(
            write_artifact=receipt,
            write_artifact_sha256=_sha256_bytes(
                material_registry_write_receipt_publication_bytes(receipt)
            ),
            published_snapshot=different,
            published_snapshot_sha256=_sha256_bytes(
                material_registry_snapshot_publication_bytes(different)
            ),
            generated_at=_VERIFY_AT,
        )


def test_reformatted_published_snapshot_file_fails_exact_sha_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="snapshot file SHA-256 mismatch"):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    assert not output_path.exists()


def test_reformatted_receipt_file_fails_exact_sha_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="receipt file SHA-256 mismatch"):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    assert not output_path.exists()


def test_independent_replay_does_not_trust_writer_verification_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _build_write(tmp_path, monkeypatch)
    untrusted = receipt.model_copy(
        update={
            "append_only_transition_verified": False,
            "published_payloads_revalidated": False,
        },
        deep=True,
    )
    replay = _independently_replay_registry_postwrite(
        untrusted,
        receipt.published_successor_snapshot,
        write_artifact_sha256=_sha256_bytes(
            material_registry_write_receipt_publication_bytes(untrusted)
        ),
        published_snapshot_sha256=receipt.published_snapshot_file_sha256,
    )
    assert replay["added_material_ids"] == receipt.added_material_ids


def test_verification_artifact_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    artifact = (
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "verification.json",
            generated_at=_VERIFY_AT,
        )
    )
    forged = artifact.model_copy(
        update={
            "verified_added_entry_cell_count": (
                artifact.verified_added_entry_cell_count + 1
            ),
            "verification_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "verification_artifact_digest": (
                oled_material_registry_successor_postwrite_verification_artifact_digest(
                    forged
                )
            )
        }
    )
    with pytest.raises(ValidationError, match="verified_added_entry_cell_count"):
        OledMaterialRegistrySuccessorPostwriteVerificationArtifact.model_validate(
            forged.model_dump(mode="json")
        )
    assert receipt.added_entry_count == artifact.verified_added_entry_count


def test_verification_timestamp_cannot_predate_pr_y(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _build_write(tmp_path, monkeypatch)
    published = receipt.published_successor_snapshot
    with pytest.raises(ValidationError, match="predates PR-Y"):
        build_oled_material_registry_successor_postwrite_verification_artifact(
            write_artifact=receipt,
            write_artifact_sha256=_sha256_bytes(
                material_registry_write_receipt_publication_bytes(receipt)
            ),
            published_snapshot=published,
            published_snapshot_sha256=_sha256_bytes(
                material_registry_snapshot_publication_bytes(published)
            ),
            generated_at="2026-07-14T00:44:00+08:00",
        )


@pytest.mark.parametrize("protected", ("receipt", "snapshot"))
def test_output_cannot_overwrite_either_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected: str,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    output_path = {"receipt": receipt_path, "snapshot": snapshot_path}[protected]
    before = output_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    assert output_path.read_bytes() == before


def test_symlinked_input_and_output_parent_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    snapshot_alias = tmp_path / "snapshot-alias.json"
    snapshot_alias.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_alias,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_VERIFY_AT,
        )

    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=linked_parent / "must-not-exist.json",
            generated_at=_VERIFY_AT,
        )
    assert not (real_parent / "must-not-exist.json").exists()


@pytest.mark.parametrize("replacement_kind", ("directory", "symlink"))
def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    output_parent = tmp_path / "verification-output"
    output_parent.mkdir()
    displaced = tmp_path / "verification-output-displaced"
    redirected = tmp_path / "verification-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.json"
    original_builder = (
        verifier_runner
        .build_oled_material_registry_successor_postwrite_verification_artifact
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
        verifier_runner,
        "build_oled_material_registry_successor_postwrite_verification_artifact",
        replace_parent_after_build,
    )
    with pytest.raises(ValueError, match="parent changed"):
        build_oled_material_registry_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()
    assert not (output_parent / output_path.name).exists()


def test_cli_failure_is_redacted_and_does_not_publish(tmp_path: Path) -> None:
    secret = "secret-token-abc123"
    output_path = tmp_path / "output.json"
    stream = StringIO()
    code = main(
        [
            "--write-artifact",
            str(tmp_path / f"missing-{secret}.json"),
            "--published-registry-snapshot",
            str(tmp_path / "missing-snapshot.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )
    payload = json.loads(stream.getvalue())
    assert code == 2
    assert payload["error_code"] == "material_registry_postwrite_verification_failed"
    assert secret not in stream.getvalue()
    assert not output_path.exists()


def test_production_implementation_is_generic_not_paper016_specific() -> None:
    source = inspect.getsource(verifier_domain) + inspect.getsource(verifier_runner)
    assert "paper016" not in source
    assert "TDBA" not in source
