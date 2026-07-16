from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent import oled_gold_successor_preflight as preflight_runner
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
    OledGoldSuccessorPreflightArtifact,
    build_oled_categorical_gold_genesis_snapshot,
    build_oled_gold_successor_preflight_artifact,
    oled_categorical_gold_snapshot_digest,
    oled_gold_successor_preflight_artifact_digest,
)
from ai4s_agent.oled_gold_candidate_postwrite_verifier import (
    build_oled_gold_candidate_postwrite_verification_from_files,
)
from ai4s_agent.oled_gold_successor_preflight import (
    build_oled_gold_successor_preflight_from_files,
    main,
)
from test_oled_gold_candidate_postwrite_verifier import (
    _VERIFY_AT,
    _publication,
)
from test_oled_supplementary_scoped_candidate_response import _sha256_file


_GENESIS_AT = "2026-07-14T00:00:00+08:00"
_PREFLIGHT_AT = "2026-07-14T01:00:00+08:00"


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, Any, Path]:
    _, receipt_path, candidate_path = _publication(tmp_path, monkeypatch)
    verification_path = tmp_path / "verification.json"
    verification = build_oled_gold_candidate_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=candidate_path,
        output_json=verification_path,
        generated_at=_VERIFY_AT,
    )
    genesis = build_oled_categorical_gold_genesis_snapshot(
        gold_registry_id="oled-categorical-gold-main",
        generated_at=_GENESIS_AT,
    )
    genesis_path = write_json(
        tmp_path / "current-gold.json",
        genesis.model_dump(mode="json"),
    )
    return verification, verification_path, candidate_path, genesis, genesis_path


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OledGoldSuccessorPreflightArtifact:
    verification, verification_path, candidate_path, genesis, genesis_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    return build_oled_gold_successor_preflight_artifact(
        verification_artifact=verification,
        verification_artifact_sha256=_sha256_file(verification_path),
        candidate_snapshot=verification.published_snapshot,
        candidate_snapshot_sha256=_sha256_file(candidate_path),
        current_gold_snapshot=genesis,
        current_gold_snapshot_sha256=_sha256_file(genesis_path),
        generated_at=_PREFLIGHT_AT,
    )


def test_genesis_is_explicit_empty_categorical_snapshot() -> None:
    genesis = build_oled_categorical_gold_genesis_snapshot(
        gold_registry_id="oled-categorical-gold-main",
        generated_at=_GENESIS_AT,
    )

    assert genesis.generation == 0
    assert genesis.entry_count == 0
    assert genesis.entries == []
    assert genesis.parent_snapshot_digest is None
    assert genesis.source_verification_digest is None
    assert genesis.categorical_confidence_only
    assert not genesis.numeric_confidence_score_assigned
    assert not genesis.curated_dataset_written


def test_verified_candidate_snapshot_plans_categorical_gold_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)

    assert artifact.status.value == "ready_for_gold_successor_write"
    assert artifact.candidate_count == 5
    assert artifact.planned_addition_count == 5
    assert artifact.prior_entry_count == 0
    assert artifact.expected_entry_count == 5
    assert artifact.expected_successor_snapshot is not None
    successor = artifact.expected_successor_snapshot
    assert successor.generation == 1
    assert successor.parent_snapshot_digest == artifact.current_gold_snapshot_digest
    assert successor.source_verification_digest == (
        artifact.verification_artifact_digest
    )
    assert successor.snapshot_digest == artifact.expected_successor_snapshot_digest
    assert all(
        entry.confidence_sufficiency == "sufficient"
        and entry.scientific_consistency == "consistent"
        and entry.categorical_confidence_only
        and not entry.numeric_confidence_score_assigned
        for entry in successor.entries
    )
    assert not artifact.gold_snapshot_written
    assert not artifact.gold_head_activated
    assert not artifact.curated_dataset_written
    assert not artifact.training_eligible


def test_current_snapshot_digest_is_exact_cas_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)
    assert artifact.expected_successor_snapshot is not None
    assert artifact.expected_successor_snapshot.parent_snapshot_digest == (
        artifact.current_gold_snapshot.snapshot_digest
    )
    assert artifact.current_gold_snapshot_digest == (
        artifact.current_gold_snapshot.snapshot_digest
    )
    assert artifact.current_snapshot_parent_bound


def test_replaying_already_present_candidate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)
    assert artifact.expected_successor_snapshot is not None
    with pytest.raises(ValueError, match="already exists"):
        build_oled_gold_successor_preflight_artifact(
            verification_artifact=artifact.verification_artifact,
            verification_artifact_sha256=artifact.verification_artifact_sha256,
            candidate_snapshot=artifact.candidate_snapshot,
            candidate_snapshot_sha256=artifact.candidate_snapshot_sha256,
            current_gold_snapshot=artifact.expected_successor_snapshot,
            current_gold_snapshot_sha256="sha256:" + "e" * 64,
            generated_at="2026-07-14T01:10:00+08:00",
        )


def test_non_genesis_snapshot_requires_parent_and_verification_lineage() -> None:
    genesis = build_oled_categorical_gold_genesis_snapshot(
        gold_registry_id="oled-categorical-gold-main",
        generated_at=_GENESIS_AT,
    )
    invalid = genesis.model_copy(
        update={
            "generation": 1,
            "snapshot_digest": "sha256:" + "0" * 64,
        }
    )
    invalid = invalid.model_copy(
        update={"snapshot_digest": oled_categorical_gold_snapshot_digest(invalid)}
    )
    with pytest.raises(ValidationError, match="requires a parent digest"):
        OledCategoricalGoldSnapshot.model_validate(
            invalid.model_dump(mode="json")
        )


def test_preflight_timestamp_cannot_predate_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification, verification_path, candidate_path, genesis, genesis_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(ValueError, match="timestamp reversal"):
        build_oled_gold_successor_preflight_artifact(
            verification_artifact=verification,
            verification_artifact_sha256=_sha256_file(verification_path),
            candidate_snapshot=verification.published_snapshot,
            candidate_snapshot_sha256=_sha256_file(candidate_path),
            current_gold_snapshot=genesis,
            current_gold_snapshot_sha256=_sha256_file(genesis_path),
            generated_at="2026-07-14T00:49:59+08:00",
        )


def test_rehashed_expected_count_tamper_fails_semantic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(tmp_path, monkeypatch)
    forged = artifact.model_copy(
        update={
            "expected_entry_count": artifact.expected_entry_count + 1,
            "preflight_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "preflight_artifact_digest": (
                oled_gold_successor_preflight_artifact_digest(forged)
            )
        }
    )
    with pytest.raises(ValidationError, match="expected_entry_count mismatch"):
        OledGoldSuccessorPreflightArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_file_entry_binds_all_exact_inputs_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, candidate_path, _, genesis_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "gold-successor-preflight.json"
    artifact = build_oled_gold_successor_preflight_from_files(
        verification_artifact_json=verification_path,
        candidate_snapshot_json=candidate_path,
        current_gold_snapshot_json=genesis_path,
        output_json=output_path,
        generated_at=_PREFLIGHT_AT,
    )

    assert artifact.verification_artifact_sha256 == _sha256_file(verification_path)
    assert artifact.candidate_snapshot_sha256 == _sha256_file(candidate_path)
    assert artifact.current_gold_snapshot_sha256 == _sha256_file(genesis_path)
    assert output_path.is_file()
    before = genesis_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_gold_successor_preflight_from_files(
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=genesis_path,
            output_json=genesis_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert genesis_path.read_bytes() == before


def test_candidate_reformat_and_symlinked_current_snapshot_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, candidate_path, _, genesis_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    original_candidate_bytes = candidate_path.read_bytes()
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        build_oled_gold_successor_preflight_from_files(
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=genesis_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_PREFLIGHT_AT,
        )

    candidate_path.write_bytes(original_candidate_bytes)
    current_link = tmp_path / "current-link.json"
    current_link.symlink_to(genesis_path)
    with pytest.raises(ValueError):
        build_oled_gold_successor_preflight_from_files(
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=current_link,
            output_json=tmp_path / "link-output.json",
            generated_at=_PREFLIGHT_AT,
        )


@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_output_parent_replacement_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _, verification_path, candidate_path, _, genesis_path = _inputs(
        tmp_path,
        monkeypatch,
    )
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.json"
    original_builder = preflight_runner.build_oled_gold_successor_preflight_artifact

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
        "build_oled_gold_successor_preflight_artifact",
        replace_parent_after_build,
    )
    with pytest.raises(ValueError):
        build_oled_gold_successor_preflight_from_files(
            verification_artifact_json=verification_path,
            candidate_snapshot_json=candidate_path,
            current_gold_snapshot_json=genesis_path,
            output_json=output_path,
            generated_at=_PREFLIGHT_AT,
        )
    assert not output_path.exists()
    assert not (redirected / output_path.name).exists()


def test_cli_failure_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, candidate_path, _, _ = _inputs(tmp_path, monkeypatch)
    invalid_current = tmp_path / "invalid-current.json"
    invalid_current.write_text("{}\n", encoding="utf-8")
    stream = StringIO()
    exit_code = main(
        [
            "--verification-artifact",
            str(verification_path),
            "--candidate-snapshot",
            str(candidate_path),
            "--current-gold-snapshot",
            str(invalid_current),
            "--output",
            str(tmp_path / "cli-output.json"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_successor_preflight_failed",
        "error_type": "ValidationError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = root / "src/ai4s_agent/domains/oled_gold_successor_preflight.py"
    runner = root / "src/ai4s_agent/oled_gold_successor_preflight.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
