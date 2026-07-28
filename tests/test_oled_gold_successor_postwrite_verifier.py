from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import (
    oled_gold_successor_postwrite_verifier as verifier_runner,
)
from ai4s_agent.domains.oled_gold_successor_postwrite_verifier import (
    OledGoldSuccessorPostwriteVerificationArtifact,
    build_oled_gold_successor_postwrite_verification_artifact,
    independently_replay_gold_successor_postwrite,
    oled_gold_successor_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    categorical_gold_snapshot_publication_bytes,
)
from ai4s_agent.domains.oled_gold_successor_writer import (
    GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    GOLD_SUCCESSOR_WRITE_FILENAME,
    gold_successor_write_receipt_publication_bytes,
)
from ai4s_agent.oled_gold_successor_postwrite_verifier import (
    build_oled_gold_successor_postwrite_verification_from_files,
    main,
)
from ai4s_agent.oled_gold_successor_writer import (
    build_oled_gold_successor_write_from_files,
)
from test_oled_gold_successor_writer import _WRITE_AT, _file_inputs
from test_oled_supplementary_scoped_candidate_response import _sha256_file


_VERIFY_AT = "2026-07-14T01:20:00+08:00"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path]:
    (
        _,
        preflight_path,
        _,
        verification_path,
        candidate_path,
        current_path,
        output_dir,
    ) = _file_inputs(tmp_path, monkeypatch)
    receipt = build_oled_gold_successor_write_from_files(
        successor_preflight_json=preflight_path,
        verification_artifact_json=verification_path,
        candidate_snapshot_json=candidate_path,
        current_gold_snapshot_json=current_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    return (
        receipt,
        output_dir / GOLD_SUCCESSOR_WRITE_FILENAME,
        output_dir / GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    )


def test_verifier_replays_publication_and_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    output_path = tmp_path / "gold-successor-verification.json"
    artifact = build_oled_gold_successor_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_VERIFY_AT,
    )

    assert artifact.status.value == "categorical_gold_successor_publication_verified"
    assert artifact.prior_generation == 0
    assert artifact.verified_generation == 1
    assert artifact.prior_entry_count == 0
    assert artifact.verified_added_entry_count == 5
    assert artifact.published_entry_count == 5
    assert artifact.verified_added_gold_entry_ids == receipt.added_gold_entry_ids
    assert artifact.verified_added_entry_digests == receipt.added_entry_digests
    assert artifact.verified_activated_snapshot_id == receipt.activated_snapshot_id
    assert artifact.snapshot_activation_receipt_verified
    assert artifact.eligible_for_explicit_dataset_admission_input
    assert not artifact.gold_snapshot_written
    assert not artifact.categorical_gold_snapshot_activated
    assert not artifact.curated_dataset_written
    assert not artifact.training_eligible
    assert artifact.write_artifact_sha256 == _sha256_file(receipt_path)
    assert artifact.published_snapshot_sha256 == _sha256_file(snapshot_path)
    assert OledGoldSuccessorPostwriteVerificationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_direct_verifier_replays_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _publication(tmp_path, monkeypatch)
    snapshot = receipt.published_successor_snapshot
    artifact = build_oled_gold_successor_postwrite_verification_artifact(
        write_artifact=receipt,
        write_artifact_sha256=_sha256_bytes(
            gold_successor_write_receipt_publication_bytes(receipt)
        ),
        published_snapshot=snapshot,
        published_snapshot_sha256=_sha256_bytes(
            categorical_gold_snapshot_publication_bytes(snapshot)
        ),
        generated_at=_VERIFY_AT,
    )
    assert artifact.verified_added_entry_count == 5
    assert artifact.verified_successor_snapshot_id == snapshot.snapshot_id


def test_independent_replay_does_not_trust_writer_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _publication(tmp_path, monkeypatch)
    untrusted = receipt.model_copy(
        update={
            "append_only_transition_verified": False,
            "published_payloads_revalidated": False,
            "activation_receipt_created": False,
            "categorical_gold_snapshot_activated": False,
        },
        deep=True,
    )
    replay = independently_replay_gold_successor_postwrite(
        untrusted,
        receipt.published_successor_snapshot,
        write_artifact_sha256=_sha256_bytes(
            gold_successor_write_receipt_publication_bytes(untrusted)
        ),
        published_snapshot_sha256=receipt.published_snapshot_file_sha256,
    )
    assert replay["added_gold_entry_ids"] == receipt.added_gold_entry_ids


@pytest.mark.parametrize("target", ("receipt", "snapshot"))
def test_reformatted_publication_file_fails_exact_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    selected = receipt_path if target == "receipt" else snapshot_path
    selected.write_text(
        selected.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        build_oled_gold_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_VERIFY_AT,
        )


def test_different_valid_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _publication(tmp_path, monkeypatch)
    different = receipt.prior_gold_snapshot
    with pytest.raises(ValueError, match="snapshot file SHA-256 mismatch"):
        build_oled_gold_successor_postwrite_verification_artifact(
            write_artifact=receipt,
            write_artifact_sha256=_sha256_bytes(
                gold_successor_write_receipt_publication_bytes(receipt)
            ),
            published_snapshot=different,
            published_snapshot_sha256=_sha256_bytes(
                categorical_gold_snapshot_publication_bytes(different)
            ),
            generated_at=_VERIFY_AT,
        )


def test_activation_binding_tamper_fails_independent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _publication(tmp_path, monkeypatch)
    tampered = receipt.model_copy(
        update={"activated_snapshot_digest": "sha256:" + "f" * 64},
        deep=True,
    )
    with pytest.raises(ValueError, match="activation binding mismatch"):
        independently_replay_gold_successor_postwrite(
            tampered,
            receipt.published_successor_snapshot,
            write_artifact_sha256=_sha256_bytes(
                gold_successor_write_receipt_publication_bytes(tampered)
            ),
            published_snapshot_sha256=receipt.published_snapshot_file_sha256,
        )


def test_verification_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    artifact = build_oled_gold_successor_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=snapshot_path,
        output_json=tmp_path / "successor-verification.json",
        generated_at=_VERIFY_AT,
    )
    forged = artifact.model_copy(
        update={
            "verified_added_entry_count": artifact.verified_added_entry_count + 1,
            "verification_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "verification_artifact_digest": (
                oled_gold_successor_postwrite_verification_artifact_digest(forged)
            )
        }
    )
    with pytest.raises(ValidationError, match="verified_added_entry_count mismatch"):
        OledGoldSuccessorPostwriteVerificationArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_timestamp_overwrite_symlink_and_cli_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="predates PR-AF"):
        build_oled_gold_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "early.json",
            generated_at="2026-07-14T01:09:59+08:00",
        )
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_gold_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=receipt_path,
            generated_at=_VERIFY_AT,
        )
    snapshot_link = tmp_path / "snapshot-link.json"
    snapshot_link.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_gold_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_link,
            output_json=tmp_path / "link-output.json",
            generated_at=_VERIFY_AT,
        )

    bad_snapshot = tmp_path / "bad-snapshot.json"
    bad_snapshot.write_text("{}\n", encoding="utf-8")
    stream = StringIO()
    exit_code = main(
        [
            "--write-artifact",
            str(receipt_path),
            "--published-categorical-gold-snapshot",
            str(bad_snapshot),
            "--output",
            str(tmp_path / "cli-output.json"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_successor_postwrite_verification_failed",
        "error_type": "ValidationError",
        "status": "error",
    }


@pytest.mark.parametrize("replacement_kind", ("directory", "symlink"))
def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    output_parent = tmp_path / "verification-output"
    output_parent.mkdir()
    output_path = output_parent / "must-not-exist.json"
    displaced = tmp_path / "verification-output-displaced"
    redirected = tmp_path / "verification-output-redirected"
    redirected.mkdir()
    original_builder = (
        verifier_runner.build_oled_gold_successor_postwrite_verification_artifact
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
        "build_oled_gold_successor_postwrite_verification_artifact",
        replace_parent_after_build,
    )
    with pytest.raises(ValueError, match="parent changed"):
        build_oled_gold_successor_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = (
        root
        / "src/ai4s_agent/domains/oled_gold_successor_postwrite_verifier.py"
    )
    runner = root / "src/ai4s_agent/oled_gold_successor_postwrite_verifier.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
