from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.domains.oled_categorical_gold_dataset_admission import (
    OledCategoricalGoldDatasetAdmissionArtifact,
    OledCategoricalGoldDatasetAdmissionDecisionStatus,
    build_oled_categorical_gold_dataset_admission_artifact,
    gold_successor_postwrite_verification_publication_bytes,
    oled_categorical_gold_dataset_admission_artifact_digest,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.domains.oled_gold_successor_preflight import (
    categorical_gold_snapshot_publication_bytes,
)
from ai4s_agent.oled_categorical_gold_dataset_admission import (
    build_oled_categorical_gold_dataset_admission_from_files,
    main,
)
from ai4s_agent.oled_gold_successor_postwrite_verifier import (
    build_oled_gold_successor_postwrite_verification_from_files,
)
from test_oled_gold_successor_postwrite_verifier import (
    _VERIFY_AT,
    _publication,
    _sha256_bytes,
)


_ADMIT_AT = "2026-07-14T01:30:00+08:00"


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path]:
    _, receipt_path, snapshot_path = _publication(tmp_path, monkeypatch)
    verification_path = tmp_path / "gold-successor-verification.json"
    verification = build_oled_gold_successor_postwrite_verification_from_files(
        write_artifact_json=receipt_path,
        published_snapshot_json=snapshot_path,
        output_json=verification_path,
        generated_at=_VERIFY_AT,
    )
    return verification, verification_path, snapshot_path


def test_admission_binds_complete_snapshot_and_only_publishes_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification, verification_path, snapshot_path = _inputs(
        tmp_path, monkeypatch
    )
    output = tmp_path / "categorical-gold-dataset-admission.json"
    artifact = build_oled_categorical_gold_dataset_admission_from_files(
        verification_artifact_json=verification_path,
        published_snapshot_json=snapshot_path,
        output_json=output,
        generated_at=_ADMIT_AT,
    )

    assert artifact.status.value == "categorical_gold_dataset_admission_complete"
    assert artifact.input_entry_count == len(
        verification.published_snapshot.entries
    )
    assert artifact.admitted_entry_count + artifact.not_admitted_entry_count == (
        artifact.input_entry_count
    )
    assert [item.gold_entry_id for item in artifact.decisions] == [
        item.gold_entry_id for item in verification.published_snapshot.entries
    ]
    assert all(len(item.view_eligibility) == 4 for item in artifact.decisions)
    assert artifact.exact_verification_artifact_bytes_bound
    assert artifact.pr_ag_verification_replayed
    assert not artifact.dataset_view_rows_written
    assert not artifact.dataset_materialized
    assert not artifact.split_assignments_created
    assert not artifact.features_materialized
    assert not artifact.training_eligible
    assert OledCategoricalGoldDatasetAdmissionArtifact.model_validate_json(
        output.read_text(encoding="utf-8")
    ) == artifact


def test_fixture_decisions_follow_exact_causal_layer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, snapshot_path = _inputs(tmp_path, monkeypatch)
    artifact = build_oled_categorical_gold_dataset_admission_from_files(
        verification_artifact_json=verification_path,
        published_snapshot_json=snapshot_path,
        output_json=tmp_path / "admission.json",
        generated_at=_ADMIT_AT,
    )
    expected = {
        OledCausalLayer.MOLECULE: [OledDatasetViewKind.CURATED_INTRINSIC],
        OledCausalLayer.INTERACTION: [],
        OledCausalLayer.MEASUREMENT: [
            OledDatasetViewKind.RAW_ALL_MEASUREMENTS
        ],
    }
    for decision in artifact.decisions:
        assert decision.eligible_view_kinds == expected[decision.target_layer]
        assert (
            decision.status
            == (
                OledCategoricalGoldDatasetAdmissionDecisionStatus.ADMITTED
                if expected[decision.target_layer]
                else OledCategoricalGoldDatasetAdmissionDecisionStatus.NOT_ADMITTED
            )
        )
        assert OledDatasetViewKind.CURATED_DEVICE_BASELINE not in (
            decision.eligible_view_kinds
        )
        assert OledDatasetViewKind.BEST_REPORTED not in (
            decision.eligible_view_kinds
        )


def test_direct_builder_reconstructs_exact_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification, _, _ = _inputs(tmp_path, monkeypatch)
    snapshot = verification.published_snapshot
    artifact = build_oled_categorical_gold_dataset_admission_artifact(
        verification_artifact=verification,
        verification_artifact_sha256=_sha256_bytes(
            gold_successor_postwrite_verification_publication_bytes(verification)
        ),
        published_snapshot=snapshot,
        published_snapshot_sha256=_sha256_bytes(
            categorical_gold_snapshot_publication_bytes(snapshot)
        ),
        generated_at=_ADMIT_AT,
    )
    assert artifact.admitted_snapshot_id == snapshot.snapshot_id
    assert artifact.published_snapshot_digest == snapshot.snapshot_digest


@pytest.mark.parametrize("target", ("verification", "snapshot"))
def test_reformatted_input_fails_exact_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _, verification_path, snapshot_path = _inputs(tmp_path, monkeypatch)
    selected = (
        verification_path if target == "verification" else snapshot_path
    )
    selected.write_text(
        selected.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        build_oled_categorical_gold_dataset_admission_from_files(
            verification_artifact_json=verification_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "must-not-exist.json",
            generated_at=_ADMIT_AT,
        )


def test_different_valid_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification, _, _ = _inputs(tmp_path, monkeypatch)
    different = verification.write_artifact.prior_gold_snapshot
    with pytest.raises(ValueError, match="snapshot differs from PR-AG"):
        build_oled_categorical_gold_dataset_admission_artifact(
            verification_artifact=verification,
            verification_artifact_sha256=_sha256_bytes(
                gold_successor_postwrite_verification_publication_bytes(
                    verification
                )
            ),
            published_snapshot=different,
            published_snapshot_sha256=_sha256_bytes(
                categorical_gold_snapshot_publication_bytes(different)
            ),
            generated_at=_ADMIT_AT,
        )


def test_decision_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, snapshot_path = _inputs(tmp_path, monkeypatch)
    artifact = build_oled_categorical_gold_dataset_admission_from_files(
        verification_artifact_json=verification_path,
        published_snapshot_json=snapshot_path,
        output_json=tmp_path / "admission.json",
        generated_at=_ADMIT_AT,
    )
    forged = artifact.model_copy(deep=True)
    forged.decisions[0].property_id = "forged-property"
    forged.admission_artifact_digest = (
        oled_categorical_gold_dataset_admission_artifact_digest(forged)
    )
    with pytest.raises(ValidationError):
        OledCategoricalGoldDatasetAdmissionArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_timestamp_overwrite_symlink_and_cli_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, verification_path, snapshot_path = _inputs(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="predates PR-AG"):
        build_oled_categorical_gold_dataset_admission_from_files(
            verification_artifact_json=verification_path,
            published_snapshot_json=snapshot_path,
            output_json=tmp_path / "early.json",
            generated_at="2026-07-14T01:19:59+08:00",
        )
    with pytest.raises(ValueError, match="overwrite"):
        build_oled_categorical_gold_dataset_admission_from_files(
            verification_artifact_json=verification_path,
            published_snapshot_json=snapshot_path,
            output_json=verification_path,
            generated_at=_ADMIT_AT,
        )
    snapshot_link = tmp_path / "snapshot-link.json"
    snapshot_link.symlink_to(snapshot_path)
    with pytest.raises(ValueError):
        build_oled_categorical_gold_dataset_admission_from_files(
            verification_artifact_json=verification_path,
            published_snapshot_json=snapshot_link,
            output_json=tmp_path / "link-output.json",
            generated_at=_ADMIT_AT,
        )

    bad = tmp_path / "bad-verification.json"
    bad.write_text("{}\n", encoding="utf-8")
    stream = StringIO()
    exit_code = main(
        [
            "--gold-successor-verification",
            str(bad),
            "--published-categorical-gold-snapshot",
            str(snapshot_path),
            "--output",
            str(tmp_path / "cli-output.json"),
        ],
        stdout=stream,
    )
    payload = json.loads(stream.getvalue())
    assert exit_code == 2
    assert payload == {
        "error_code": "categorical_gold_dataset_admission_failed",
        "error_type": "ValidationError",
        "status": "error",
    }
    assert str(bad) not in stream.getvalue()
