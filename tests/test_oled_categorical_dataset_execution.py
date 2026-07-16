from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetBaselineStatus,
    OledCategoricalDatasetExecutionArtifact,
    build_oled_categorical_dataset_split_assignments,
    oled_categorical_dataset_execution_artifact_digest,
    oled_categorical_dataset_view_row_digest,
    run_oled_categorical_dataset_baselines,
)
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.oled_categorical_dataset_execution import (
    build_oled_categorical_dataset_execution_from_files,
    main,
)
from ai4s_agent import oled_categorical_dataset_execution as execution_runner
from ai4s_agent.oled_categorical_gold_dataset_admission import (
    build_oled_categorical_gold_dataset_admission_from_files,
)
from test_oled_categorical_gold_dataset_admission import (
    _ADMIT_AT,
    _inputs,
)


_EXECUTE_AT = "2026-07-14T01:40:00+08:00"


def _admission_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _, verification_path, snapshot_path = _inputs(tmp_path, monkeypatch)
    path = tmp_path / "categorical-gold-dataset-admission.json"
    build_oled_categorical_gold_dataset_admission_from_files(
        verification_artifact_json=verification_path,
        published_snapshot_json=snapshot_path,
        output_json=path,
        generated_at=_ADMIT_AT,
    )
    return path


def test_execution_materializes_only_pr_ah_admitted_rows_and_runs_smoke_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_path = _admission_path(tmp_path, monkeypatch)
    artifact, output_dir = build_oled_categorical_dataset_execution_from_files(
        admission_artifact_json=admission_path,
        output_root=tmp_path / "datasets",
        generated_at=_EXECUTE_AT,
    )

    assert artifact.status.value == "categorical_dataset_snapshot_materialized"
    assert artifact.materialized_row_count == 5
    assert artifact.admitted_decision_count == 5
    assert artifact.excluded_decision_count == 0
    assert artifact.material_group_count == 1
    assert artifact.rows_by_view == {
        OledDatasetViewKind.CURATED_INTRINSIC: 5
    }
    assert artifact.rows_by_split == {"train": 5}
    assert all(
        row.view_kind == OledDatasetViewKind.CURATED_INTRINSIC
        and row.target_layer.value == "molecule"
        and row.source_gold_entry_id
        and row.evidence_refs
        and len(row.features) == 128
        for row in artifact.rows
    )
    assert all(
        summary.status
        == OledCategoricalDatasetBaselineStatus.TRAIN_ONLY
        for summary in artifact.baseline_summaries
    )
    assert len(artifact.baseline_predictions) == 5
    assert len(artifact.baseline_metrics) == 5
    assert not artifact.benchmark_validated
    assert not artifact.training_eligible
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "baseline_metrics.json",
        "baseline_predictions.jsonl",
        "report.md",
        "rows.jsonl",
        "snapshot.json",
        "split_assignments.jsonl",
    ]
    assert OledCategoricalDatasetExecutionArtifact.model_validate_json(
        (output_dir / "snapshot.json").read_text(encoding="utf-8")
    ) == artifact
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "completed_without_holdout" in report
    assert "not a benchmark-validation" in report

    with pytest.raises(ValueError, match="already exists"):
        build_oled_categorical_dataset_execution_from_files(
            admission_artifact_json=admission_path,
            output_root=tmp_path / "datasets",
            generated_at=_EXECUTE_AT,
        )


def test_material_group_split_produces_holdout_metrics_when_data_supports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_path = _admission_path(tmp_path, monkeypatch)
    artifact, _ = build_oled_categorical_dataset_execution_from_files(
        admission_artifact_json=admission_path,
        output_root=tmp_path / "datasets",
        generated_at=_EXECUTE_AT,
    )
    source = next(
        row for row in artifact.rows if row.property_id == "delta_e_st_ev"
    )
    rows = [source]
    for index, value in ((2, 0.61), (3, 0.43)):
        clone = source.model_copy(
            update={
                "row_id": f"oled-categorical-dataset-row:synthetic-{index}",
                "selected_material_id": f"material-{index:04d}",
                "target_value": value,
                "row_digest": "sha256:" + "0" * 64,
            },
            deep=True,
        )
        clone = clone.model_copy(
            update={
                "row_digest": oled_categorical_dataset_view_row_digest(clone)
            }
        )
        rows.append(type(source).model_validate(clone.model_dump(mode="json")))
    rows = sorted(rows, key=lambda item: item.row_id)
    assignments = build_oled_categorical_dataset_split_assignments(rows)
    summaries, predictions, metrics = run_oled_categorical_dataset_baselines(
        rows, assignments
    )

    assert {item.split for item in assignments} == {
        "train",
        "validation",
        "test",
    }
    assert summaries[0].status == OledCategoricalDatasetBaselineStatus.EVALUATED
    assert summaries[0].prediction_count == 3
    assert len(predictions) == 3
    assert {item.split for item in metrics} == {
        "train",
        "validation",
        "test",
    }


def test_reformatted_admission_and_artifact_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_path = _admission_path(tmp_path, monkeypatch)
    admission_path.write_text(
        admission_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        build_oled_categorical_dataset_execution_from_files(
            admission_artifact_json=admission_path,
            output_root=tmp_path / "datasets",
            generated_at=_EXECUTE_AT,
        )

    admission_path.write_text(
        admission_path.read_text(encoding="utf-8").rstrip() + "\n",
        encoding="utf-8",
    )
    artifact, _ = build_oled_categorical_dataset_execution_from_files(
        admission_artifact_json=admission_path,
        output_root=tmp_path / "datasets",
        generated_at=_EXECUTE_AT,
    )
    forged = artifact.model_copy(deep=True)
    forged.rows_by_split = {"test": artifact.materialized_row_count}
    forged.execution_artifact_digest = (
        oled_categorical_dataset_execution_artifact_digest(forged)
    )
    with pytest.raises(ValidationError, match="rows_by_split mismatch"):
        OledCategoricalDatasetExecutionArtifact.model_validate(
            forged.model_dump(mode="json")
        )


def test_concurrent_target_created_before_commit_survives_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_path = _admission_path(tmp_path, monkeypatch)
    output_root = tmp_path / "datasets"
    original = execution_runner._atomic_rename_owned_directory_noreplace
    concurrent_target: Path | None = None

    def create_target_then_commit(**kwargs: object) -> None:
        nonlocal concurrent_target
        parent_descriptor = int(kwargs["parent_descriptor"])
        output_name = str(kwargs["output_name"])
        os.mkdir(output_name, dir_fd=parent_descriptor)
        target_descriptor = os.open(
            output_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            marker_descriptor = os.open(
                "concurrent.marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=target_descriptor,
            )
            os.write(marker_descriptor, b"concurrent-owner\n")
            os.fsync(marker_descriptor)
            os.close(marker_descriptor)
            os.fsync(target_descriptor)
        finally:
            os.close(target_descriptor)
        concurrent_target = output_root / output_name
        original(**kwargs)

    monkeypatch.setattr(
        execution_runner,
        "_atomic_rename_owned_directory_noreplace",
        create_target_then_commit,
    )
    with pytest.raises(ValueError, match="already exists"):
        build_oled_categorical_dataset_execution_from_files(
            admission_artifact_json=admission_path,
            output_root=output_root,
            generated_at=_EXECUTE_AT,
        )

    assert concurrent_target is not None
    assert concurrent_target.is_dir()
    assert (
        concurrent_target / "concurrent.marker"
    ).read_text(encoding="utf-8") == "concurrent-owner\n"
    assert sorted(path.name for path in concurrent_target.iterdir()) == [
        "concurrent.marker"
    ]
    assert not any(
        path.name.startswith(".oled-categorical-dataset-snapshot")
        for path in output_root.iterdir()
    )


def test_output_parent_replacement_or_symlink_redirect_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_path = _admission_path(tmp_path, monkeypatch)
    output_root = tmp_path / "datasets"
    moved_root = tmp_path / "datasets-original"
    redirected_root = tmp_path / "datasets-redirected"
    redirected_root.mkdir()
    original = execution_runner._publish_versioned_dataset_directory

    def replace_parent_then_publish(*args: object, **kwargs: object) -> None:
        output_root.rename(moved_root)
        output_root.symlink_to(redirected_root, target_is_directory=True)
        original(*args, **kwargs)

    monkeypatch.setattr(
        execution_runner,
        "_publish_versioned_dataset_directory",
        replace_parent_then_publish,
    )
    with pytest.raises(ValueError, match="output parent changed"):
        build_oled_categorical_dataset_execution_from_files(
            admission_artifact_json=admission_path,
            output_root=output_root,
            generated_at=_EXECUTE_AT,
        )

    assert output_root.is_symlink()
    assert list(moved_root.iterdir()) == []
    assert list(redirected_root.iterdir()) == []


def test_cli_failure_is_stable_and_redacted(tmp_path: Path) -> None:
    bad = tmp_path / "private-admission.json"
    bad.write_text("{}\n", encoding="utf-8")
    stream = StringIO()
    exit_code = main(
        [
            "--dataset-admission",
            str(bad),
            "--output-root",
            str(tmp_path / "datasets"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "categorical_dataset_execution_failed",
        "error_type": "ValidationError",
        "status": "error",
    }
    assert str(bad) not in stream.getvalue()
