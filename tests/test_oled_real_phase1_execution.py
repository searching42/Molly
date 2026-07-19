from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
    OledCategoricalDatasetExecutionStatus,
    OledCategoricalDatasetViewRow,
    build_oled_categorical_dataset_split_assignments,
    oled_categorical_dataset_execution_artifact_digest,
    oled_categorical_dataset_view_row_digest,
    run_oled_categorical_dataset_baselines,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.oled_real_phase1_execution import (
    main,
    run_oled_real_phase1_execution_from_files,
)


def _snapshot_path(tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = []
    for material_index in range(4):
        for property_index, (property_id, base_value) in enumerate(
            (("delta_e_st_ev", 0.4), ("s1_ev", 3.0), ("t1_ev", 2.6))
        ):
            token = f"{material_index:02d}{property_index:02d}"
            row = OledCategoricalDatasetViewRow.model_construct(
                row_id=f"oled-categorical-dataset-row:execution-{token}",
                source_admission_decision_id=f"admission:execution-{token}",
                source_admission_decision_digest="sha256:" + f"{material_index + 1:064x}",
                source_gold_entry_id=f"gold-entry:execution-{token}",
                source_gold_entry_digest="sha256:" + f"{property_index + 10:064x}",
                source_candidate_id=f"candidate:execution-{token}",
                source_candidate_digest=(
                    "sha256:"
                    + f"{material_index * 3 + property_index + 20:064x}"
                ),
                selected_material_id=f"material:execution-{material_index:02d}",
                canonical_isomeric_smiles="C" * (material_index + 1),
                registry_entry_digest="sha256:" + f"{material_index + 30:064x}",
                view_kind=OledDatasetViewKind.CURATED_INTRINSIC,
                property_id=property_id,
                target_layer=OledCausalLayer.MOLECULE,
                target_value=base_value + material_index * 0.1,
                target_unit="eV",
                reported_value_text=str(base_value + material_index * 0.1),
                reported_decimal_places=2,
                reported_unit="eV",
                comparison_context_status="not_required",
                evidence_refs=[f"evidence:{token}"],
                feature_type="morgan_ecfp",
                features={
                    "ecfp_000": float(material_index % 2),
                    "ecfp_001": float(material_index // 2),
                    "ecfp_002": float(property_index == 0),
                    "ecfp_003": 1.0,
                },
                row_digest="sha256:" + "0" * 64,
            )
            row = row.model_copy(
                update={"row_digest": oled_categorical_dataset_view_row_digest(row)}
            )
            rows.append(OledCategoricalDatasetViewRow.model_validate(row.model_dump(mode="json")))
    rows.sort(key=lambda row: row.row_id)
    assignments = build_oled_categorical_dataset_split_assignments(rows)
    summaries, predictions, metrics = run_oled_categorical_dataset_baselines(
        rows, assignments
    )
    expanded = OledCategoricalDatasetExecutionArtifact.model_construct(
        run_id="real-phase1-test",
        paper_id="paper-test",
        generated_at="2026-07-14T01:40:00+08:00",
        source_admission_sha256="sha256:" + "1" * 64,
        source_admission_digest="sha256:" + "2" * 64,
        source_gold_snapshot_id="gold-snapshot:test",
        source_gold_snapshot_digest="sha256:" + "3" * 64,
        dataset_snapshot_id="dataset-snapshot:test",
        status=OledCategoricalDatasetExecutionStatus.MATERIALIZED,
        admitted_decision_count=len(rows),
        materialized_row_count=len(rows),
        excluded_decision_count=0,
        material_group_count=4,
        rows_by_view={OledDatasetViewKind.CURATED_INTRINSIC: len(rows)},
        rows_by_property={property_id: 4 for property_id in ("delta_e_st_ev", "s1_ev", "t1_ev")},
        rows_by_split={"test": 3, "train": 6, "validation": 3},
        rows=rows,
        split_assignments=assignments,
        baseline_summaries=summaries,
        baseline_predictions=predictions,
        baseline_metrics=metrics,
        execution_artifact_digest="sha256:" + "0" * 64,
    )
    expanded = expanded.model_copy(
        update={
            "execution_artifact_digest": oled_categorical_dataset_execution_artifact_digest(
                expanded
            )
        }
    )
    validated = OledCategoricalDatasetExecutionArtifact.model_validate(
        expanded.model_dump(mode="json")
    )
    path = tmp_path / "expanded-snapshot.json"
    path.write_text(
        json.dumps(validated.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_real_execution_fits_models_and_ranks_holdout_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot_path(tmp_path, monkeypatch)
    result = run_oled_real_phase1_execution_from_files(
        dataset_snapshot_json=snapshot,
        output_root=tmp_path / "executions",
        property_ids=["delta_e_st_ev", "s1_ev"],
        generated_at="2026-07-19T12:00:00+08:00",
    )

    assert result.trained_model_count == 2
    assert result.ranked_candidate_count == 2
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "execution.json",
        "metrics.json",
        "model__delta_e_st_ev.json",
        "model__s1_ev.json",
        "predictions.jsonl",
        "ranked_candidates.csv",
        "report.md",
    ]
    summary = json.loads((result.output_dir / "execution.json").read_text())
    assert summary["claims"] == {
        "benchmark_validated": False,
        "holdout_only_ranking": True,
        "model_registered": False,
        "production_ready": False,
        "real_model_fit": True,
    }
    model = json.loads((result.output_dir / "model__s1_ev.json").read_text())
    assert model["model_kind"] == "linear_kernel_ridge.v1"
    assert len(model["training_row_ids"]) == 2
    ranked = (result.output_dir / "ranked_candidates.csv").read_text()
    assert "material:execution-02" in ranked
    assert "material:execution-03" in ranked
    assert "material:execution-00" not in ranked
    assert "material:execution-01" not in ranked

    with pytest.raises(ValueError, match="already exists"):
        run_oled_real_phase1_execution_from_files(
            dataset_snapshot_json=snapshot,
            output_root=tmp_path / "executions",
            property_ids=["delta_e_st_ev", "s1_ev"],
            generated_at="2026-07-19T12:01:00+08:00",
        )


def test_real_execution_rejects_tampered_snapshot_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot_path(tmp_path, monkeypatch)
    payload = json.loads(snapshot.read_text())
    payload["rows"][0]["target_value"] = 999.0
    snapshot.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row digest mismatch"):
        run_oled_real_phase1_execution_from_files(
            dataset_snapshot_json=snapshot,
            output_root=tmp_path / "executions",
        )
    assert not (tmp_path / "executions").exists() or not any(
        (tmp_path / "executions").iterdir()
    )


def test_real_execution_cli_failure_is_redacted(tmp_path: Path) -> None:
    private_path = tmp_path / "private-snapshot.json"
    private_path.write_text("{}\n", encoding="utf-8")
    stream = StringIO()
    exit_code = main(
        [
            "--dataset-snapshot",
            str(private_path),
            "--output-root",
            str(tmp_path / "executions"),
        ],
        stdout=stream,
    )
    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "real_phase1_execution_failed",
        "error_type": "ValidationError",
        "status": "error",
    }
    assert str(private_path) not in stream.getvalue()
