from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
    gold_candidate_write_receipt_publication_bytes,
    independently_replay_gold_candidate_postwrite,
    oled_gold_candidate_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    GOLD_CANDIDATE_WRITE_FILENAME,
    gold_candidate_snapshot_publication_bytes,
)
from ai4s_agent.oled_gold_candidate_postwrite_verifier import (
    build_oled_gold_candidate_postwrite_verification_from_files,
    main,
)
from ai4s_agent.oled_gold_candidate_writer import (
    build_oled_gold_candidate_write_from_files,
)
from tests.test_oled_gold_candidate_writer import _WRITE_AT, _preflight
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_VERIFY_AT = "2026-07-14T00:50:00+08:00"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, preflight_path = _preflight(tmp_path, monkeypatch)
    output_dir = tmp_path / "publication"
    receipt = build_oled_gold_candidate_write_from_files(
        preflight_artifact_json=preflight_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    return (
        receipt,
        output_dir / GOLD_CANDIDATE_WRITE_FILENAME,
        output_dir / GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    )


def test_verifier_replays_exact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    output_path = tmp_path / "verification.json"

    artifact = build_oled_gold_candidate_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_VERIFY_AT,
    )

    assert artifact.status.value == "gold_candidate_publication_verified"
    assert artifact.verified_candidate_count == 5
    assert artifact.verified_candidate_ids == receipt.published_candidate_ids
    assert artifact.verified_candidate_digests == receipt.published_candidate_digests
    assert artifact.write_artifact_sha256 == _sha256_file(receipt_path)
    assert artifact.published_snapshot_sha256 == _sha256_file(snapshot_path)
    assert artifact.eligible_for_explicit_gold_publication_input
    assert not artifact.gold_records_created
    assert not artifact.curated_dataset_written
    assert OledGoldCandidatePostwriteVerificationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_independent_replay_does_not_trust_writer_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _publication(tmp_path, monkeypatch)
    untrusted = receipt.model_copy(
        update={
            "exact_candidate_roster_replayed": False,
            "published_payloads_revalidated": False,
        }
    )
    replay = independently_replay_gold_candidate_postwrite(
        untrusted,
        receipt.published_snapshot,
        write_artifact_sha256=_sha256_bytes(
            gold_candidate_write_receipt_publication_bytes(untrusted)
        ),
        published_snapshot_sha256=receipt.published_snapshot_file_sha256,
    )
    assert replay["candidate_ids"] == receipt.published_candidate_ids


@pytest.mark.parametrize("target", ["receipt", "snapshot"])
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
        build_oled_gold_candidate_postwrite_verification_from_files(
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
    from ai4s_agent.domains.oled_gold_candidate_writer import (
        OledGoldCandidateSnapshot,
        oled_gold_candidate_snapshot_digest,
    )
    provisional = receipt.published_snapshot.model_copy(
        update={
            "generated_at": "2026-07-14T00:49:01+08:00",
            "snapshot_digest": "sha256:" + "0" * 64,
        }
    )
    different = OledGoldCandidateSnapshot.model_validate(
        provisional.model_copy(
            update={
                "snapshot_digest": oled_gold_candidate_snapshot_digest(
                    provisional
                )
            }
        ).model_dump(mode="json")
    )

    with pytest.raises(ValueError, match="snapshot file SHA-256 mismatch"):
        independently_replay_gold_candidate_postwrite(
            receipt,
            different,
            write_artifact_sha256=_sha256_bytes(
                gold_candidate_write_receipt_publication_bytes(receipt)
            ),
            published_snapshot_sha256=_sha256_bytes(
                gold_candidate_snapshot_publication_bytes(different)
            ),
        )


def test_verification_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    artifact = build_oled_gold_candidate_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=snapshot_path,
        output_json=tmp_path / "verification.json",
        generated_at=_VERIFY_AT,
    )
    forged = artifact.model_copy(
        update={
            "verified_candidate_count": artifact.verified_candidate_count + 1,
            "verification_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "verification_artifact_digest": (
                oled_gold_candidate_postwrite_verification_artifact_digest(forged)
            )
        }
    )
    with pytest.raises(ValidationError, match="verified_candidate_count"):
        OledGoldCandidatePostwriteVerificationArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_timestamp_overwrite_symlink_and_cli_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="predates"):
        build_oled_gold_candidate_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "early.json",
            generated_at="2026-07-14T00:48:59+08:00",
        )
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_gold_candidate_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=receipt_path,
            generated_at=_VERIFY_AT,
        )
    link = tmp_path / "snapshot-link.json"
    link.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_gold_candidate_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=link,
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
            "--published-gold-candidate-snapshot",
            str(bad_snapshot),
            "--output",
            str(tmp_path / "cli-output.json"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_candidate_postwrite_verification_failed",
        "error_type": "ValidationError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = root / "src/ai4s_agent/domains/oled_gold_candidate_postwrite_verifier.py"
    runner = root / "src/ai4s_agent/oled_gold_candidate_postwrite_verifier.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
