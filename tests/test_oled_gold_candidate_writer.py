from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.domains.oled_gold_candidate_writer import (
    GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    GOLD_CANDIDATE_WRITE_FILENAME,
    OledGoldCandidateSnapshot,
    OledGoldCandidateWriteArtifact,
    OledGoldCandidateWriteStatus,
    gold_candidate_snapshot_publication_bytes,
    oled_gold_candidate_write_artifact_digest,
)
from ai4s_agent.oled_gold_admission_preflight import (
    build_oled_gold_admission_preflight_from_files,
)
from ai4s_agent import oled_gold_candidate_writer as writer
from ai4s_agent.oled_gold_candidate_writer import (
    build_oled_gold_candidate_write_from_files,
    main,
)
from tests.test_oled_gold_admission_preflight import (
    _PREFLIGHT_AT,
    _all_accepted_adjudication,
    _mixed_adjudication,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_WRITE_AT = "2026-07-14T00:49:00+08:00"


def _preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    empty: bool = False,
):
    if empty:
        _, adjudication_path = _mixed_adjudication(
            tmp_path,
            monkeypatch,
            block_all=True,
        )
    else:
        _, adjudication_path = _all_accepted_adjudication(tmp_path, monkeypatch)
    preflight_path = tmp_path / "gold-admission-preflight.json"
    preflight = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=adjudication_path,
        output_json=preflight_path,
        generated_at=_PREFLIGHT_AT,
    )
    return preflight, preflight_path


def test_writer_publishes_exact_snapshot_and_receipt_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, preflight_path = _preflight(tmp_path, monkeypatch)
    output_dir = tmp_path / "gold-candidate-publication"

    artifact = build_oled_gold_candidate_write_from_files(
        preflight_artifact_json=preflight_path,
        output_dir=output_dir,
        generated_at=_WRITE_AT,
    )

    receipt_path = output_dir / GOLD_CANDIDATE_WRITE_FILENAME
    snapshot_path = output_dir / GOLD_CANDIDATE_SNAPSHOT_FILENAME
    assert artifact.status == OledGoldCandidateWriteStatus.PUBLISHED
    assert artifact.preflight_artifact == preflight
    assert artifact.preflight_artifact_sha256 == _sha256_file(preflight_path)
    assert artifact.published_candidate_count == 5
    assert artifact.published_candidate_ids == sorted(
        candidate.candidate_id for candidate in preflight.candidates
    )
    assert artifact.published_snapshot_file_sha256 == _sha256_file(snapshot_path)
    assert snapshot_path.read_bytes() == gold_candidate_snapshot_publication_bytes(
        artifact.published_snapshot
    )
    assert set(path.name for path in output_dir.iterdir()) == {
        GOLD_CANDIDATE_WRITE_FILENAME,
        GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    }
    assert OledGoldCandidateWriteArtifact.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    ) == artifact
    assert OledGoldCandidateSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    ) == artifact.published_snapshot
    assert not artifact.gold_records_created
    assert not artifact.curated_dataset_written
    assert not artifact.training_eligible


def test_empty_eligible_roster_cannot_be_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch, empty=True)

    with pytest.raises(ValueError, match="empty roster"):
        build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=preflight_path,
            output_dir=tmp_path / "empty-publication",
            generated_at=_WRITE_AT,
        )


def test_existing_output_and_symbolic_input_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="fresh"):
        build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=preflight_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )

    link = tmp_path / "preflight-link.json"
    link.symlink_to(preflight_path)
    with pytest.raises(ValueError, match="symbolic|symlink"):
        build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=link,
            output_dir=tmp_path / "link-output",
            generated_at=_WRITE_AT,
        )


def test_preflight_change_before_publication_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch)
    output_dir = tmp_path / "changed-input-output"
    original = writer._read_bound_json
    calls = 0

    def changing_read(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            payload = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight_path.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(writer, "_read_bound_json", changing_read)

    with pytest.raises(ValueError, match="changed before publication"):
        build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=preflight_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert not output_dir.exists()


def test_target_created_in_rename_window_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch)
    output_dir = tmp_path / "raced-output"
    original = writer._atomic_rename_owned_directory_noreplace

    def create_target_then_rename(**kwargs):
        output_dir.mkdir()
        (output_dir / "foreign.txt").write_text("foreign", encoding="utf-8")
        return original(**kwargs)

    monkeypatch.setattr(
        writer,
        "_atomic_rename_owned_directory_noreplace",
        create_target_then_rename,
    )

    with pytest.raises(ValueError, match="fresh"):
        build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=preflight_path,
            output_dir=output_dir,
            generated_at=_WRITE_AT,
        )
    assert (output_dir / "foreign.txt").read_text(encoding="utf-8") == "foreign"


def test_snapshot_or_receipt_tamper_fails_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch)
    artifact = build_oled_gold_candidate_write_from_files(
        preflight_artifact_json=preflight_path,
        output_dir=tmp_path / "publication",
        generated_at=_WRITE_AT,
    )
    forged_snapshot = artifact.published_snapshot.model_copy(
        update={"entry_count": 4}
    )
    forged = artifact.model_copy(
        update={
            "published_snapshot": forged_snapshot,
            "write_artifact_digest": "sha256:" + "0" * 64,
        }
    )
    forged = forged.model_copy(
        update={
            "write_artifact_digest": oled_gold_candidate_write_artifact_digest(
                forged
            )
        }
    )

    with pytest.raises(ValidationError, match="snapshot|count"):
        OledGoldCandidateWriteArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_cli_redacts_failure_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preflight_path = _preflight(tmp_path, monkeypatch, empty=True)
    stream = StringIO()

    exit_code = main(
        [
            "--gold-admission-preflight",
            str(preflight_path),
            "--output-dir",
            str(tmp_path / "cli-output"),
        ],
        stdout=stream,
    )

    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "gold_candidate_write_failed",
        "error_type": "ValueError",
        "status": "error",
    }


def test_production_implementation_is_not_paper016_specific() -> None:
    root = Path(__file__).parents[1]
    domain = root / "src/ai4s_agent/domains/oled_gold_candidate_writer.py"
    runner = root / "src/ai4s_agent/oled_gold_candidate_writer.py"
    assert "paper016" not in domain.read_text(encoding="utf-8")
    assert "paper016" not in runner.read_text(encoding="utf-8")
