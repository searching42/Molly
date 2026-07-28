from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_reviewed_evidence_ledger_postwrite_verifier import (
    OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
    OledReviewedEvidenceLedgerPostwriteVerificationStatus,
    oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    build_oled_reviewed_evidence_ledger_snapshot,
)
from ai4s_agent.oled_reviewed_evidence_ledger_postwrite_verifier import (
    build_oled_reviewed_evidence_ledger_postwrite_verification_from_files,
    main,
)
from ai4s_agent.oled_reviewed_evidence_ledger_writer import (
    build_oled_reviewed_evidence_ledger_write_from_files,
)
from tests.test_oled_reviewed_evidence_ledger_writer import (
    _preflight_inputs,
)
from tests.test_oled_reviewed_evidence_staging_preflight import (
    _alternate_source_entry,
    _entry_for_candidate,
    _inputs,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_WRITE_AT = "2026-07-14T00:43:00+08:00"
_VERIFY_AT = "2026-07-14T00:44:00+08:00"


def _write_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger=None,
    inputs=None,
):
    _, preflight_path, ledger_path, output_dir = _preflight_inputs(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )
    receipt = build_oled_reviewed_evidence_ledger_write_from_files(
        preflight_artifact_json=preflight_path,
        current_ledger_snapshot_json=ledger_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )
    return (
        receipt,
        output_dir / "reviewed_evidence_ledger_write.json",
        output_dir / "reviewed_evidence_ledger_snapshot.json",
    )


def test_postwrite_verifier_replays_genesis_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    output_path = tmp_path / "postwrite-verification.json"

    artifact = (
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )
    )

    assert artifact.status == OledReviewedEvidenceLedgerPostwriteVerificationStatus.VERIFIED
    assert artifact.prior_entry_count == 0
    assert artifact.verified_added_entry_count == 5
    assert artifact.verified_active_entry_count == 5
    assert artifact.verified_quarantined_entry_count == 0
    assert artifact.verified_exact_replay_noop_count == 0
    assert artifact.published_entry_count == 5
    assert artifact.write_artifact_sha256 == _sha256_file(receipt_path)
    assert artifact.published_snapshot_sha256 == _sha256_file(snapshot_path)
    assert artifact.published_snapshot == receipt.next_ledger_snapshot
    assert OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def test_postwrite_verifier_confirms_exact_replay_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    materialization, _, _, _ = inputs
    ledger = build_oled_reviewed_evidence_ledger_snapshot(
        entries=[
            _entry_for_candidate(materialization, candidate)
            for candidate in materialization.observation_candidates
        ],
        generated_at="2026-07-14T00:41:00+08:00",
        snapshot_id="reviewed-evidence-ledger:postwrite-noop",
    )
    receipt, receipt_path, snapshot_path = _write_result(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = (
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "noop-verification.json",
            generated_at=_VERIFY_AT,
        )
    )

    assert artifact.verified_added_entry_count == 0
    assert artifact.verified_exact_replay_noop_count == 5
    assert artifact.published_snapshot == receipt.prior_ledger_snapshot


def test_postwrite_verifier_confirms_conflict_is_quarantined(
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
        snapshot_id="reviewed-evidence-ledger:postwrite-conflict",
    )
    _, receipt_path, snapshot_path = _write_result(
        tmp_path,
        monkeypatch,
        ledger=ledger,
        inputs=inputs,
    )

    artifact = (
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "conflict-verification.json",
            generated_at=_VERIFY_AT,
        )
    )

    assert artifact.verified_added_entry_count == 5
    assert artifact.verified_active_entry_count == 4
    assert artifact.verified_quarantined_entry_count == 1


def test_different_valid_snapshot_is_rejected_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, receipt_path, _ = _write_result(tmp_path, monkeypatch)
    wrong_snapshot_path = tmp_path / "wrong-snapshot.json"
    write_json(
        wrong_snapshot_path,
        receipt.prior_ledger_snapshot.model_dump(mode="json"),
    )
    output_path = tmp_path / "wrong-snapshot-verification.json"

    with pytest.raises(ValueError, match="does not match the exact PR-S receipt"):
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=wrong_snapshot_path,
            output_json=output_path,
            generated_at=_VERIFY_AT,
        )

    assert not output_path.exists()


def test_postwrite_verification_timestamp_reversal_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    output_path = tmp_path / "timestamp-reversal.json"

    with pytest.raises(ValueError, match="predates PR-S"):
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at="2026-07-14T00:42:59+08:00",
        )

    assert not output_path.exists()


def test_verification_count_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    artifact = (
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "verification.json",
            generated_at=_VERIFY_AT,
        )
    )
    forged = artifact.model_copy(
        update={
            "verified_active_entry_count": 4,
            "verification_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "verification_artifact_digest": (
                oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest(
                    forged
                )
            )
        }
    )

    with pytest.raises(ValidationError, match="verified_active_entry_count"):
        OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_postwrite_output_cannot_overwrite_an_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt_path, snapshot_path = _write_result(tmp_path, monkeypatch)
    original = receipt_path.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        build_oled_reviewed_evidence_ledger_postwrite_verification_from_files(
            write_artifact_json=receipt_path,
            published_snapshot_json=snapshot_path,
            output_json=receipt_path,
            generated_at=_VERIFY_AT,
        )

    assert receipt_path.read_bytes() == original


def test_postwrite_cli_failure_is_redacted(
    tmp_path: Path,
) -> None:
    secret = "secret-postwrite-token"
    output_path = tmp_path / "output.json"
    stream = StringIO()

    code = main(
        [
            "--write-artifact",
            str(tmp_path / f"missing-{secret}.json"),
            "--published-ledger-snapshot",
            str(tmp_path / "missing-snapshot.json"),
            "--output",
            str(output_path),
        ],
        stdout=stream,
    )

    payload = json.loads(stream.getvalue())
    assert code == 2
    assert payload["error_code"] == "reviewed_evidence_postwrite_verification_failed"
    assert secret not in stream.getvalue()
    assert not output_path.exists()
