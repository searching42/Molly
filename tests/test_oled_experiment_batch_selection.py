from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from ai4s_agent import oled_experiment_batch_selection as batch_runner
from ai4s_agent.oled_experiment_batch_selection import (
    load_oled_experiment_batch_selection_inputs,
    run_oled_experiment_batch_selection_from_files,
)


_PROPERTY_IDS = ("delta_e_st_ev", "s1_ev")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _shortlist_bytes(rows: list[dict[str, object]]) -> bytes:
    fieldnames = [
        "rank",
        "material_id",
        "registry_entry_digest",
        "canonical_name",
        "canonical_isomeric_smiles",
        "aggregate_percentile",
        *[f"predicted_{property_id}" for property_id in _PROPERTY_IDS],
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _rows() -> list[dict[str, object]]:
    return [
        {
            "rank": 1,
            "material_id": "material:batch-a",
            "registry_entry_digest": "sha256:" + "1" * 64,
            "canonical_name": "ethane",
            "canonical_isomeric_smiles": "CC",
            "aggregate_percentile": 0.95,
            "predicted_delta_e_st_ev": 0.10,
            "predicted_s1_ev": 3.10,
        },
        {
            "rank": 2,
            "material_id": "material:batch-b",
            "registry_entry_digest": "sha256:" + "2" * 64,
            "canonical_name": "benzene",
            "canonical_isomeric_smiles": "c1ccccc1",
            "aggregate_percentile": 0.90,
            "predicted_delta_e_st_ev": 0.12,
            "predicted_s1_ev": 3.20,
        },
        {
            "rank": 3,
            "material_id": "material:batch-c",
            "registry_entry_digest": "sha256:" + "3" * 64,
            "canonical_name": "ethanol",
            "canonical_isomeric_smiles": "CCO",
            "aggregate_percentile": 0.80,
            "predicted_delta_e_st_ev": 0.18,
            "predicted_s1_ev": 3.00,
        },
    ]


def _sources() -> dict[str, object]:
    return {
        "phase1_execution_id": "oled-real-phase1-execution:fixture",
        "phase1_execution_sha256": "sha256:" + "a" * 64,
        "dataset_snapshot_id": "oled-dataset-snapshot:fixture",
        "dataset_snapshot_digest": "sha256:" + "b" * 64,
        "dataset_snapshot_sha256": "sha256:" + "c" * 64,
        "registry_id": "oled-registry:fixture",
        "registry_version": "registry-version:fixture",
        "registry_snapshot_digest": "sha256:" + "d" * 64,
        "registry_snapshot_sha256": "sha256:" + "e" * 64,
        "model_sha256": {
            "delta_e_st_ev": "sha256:" + "f" * 64,
            "s1_ev": "sha256:" + "0" * 64,
        },
    }


def _screening_paths(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shortlist_rows = _rows() if rows is None else rows
    shortlist = _shortlist_bytes(shortlist_rows)
    shortlist_path = tmp_path / "ranked_shortlist.csv"
    shortlist_path.write_bytes(shortlist)
    sources = _sources()
    config = {
        "property_ids": list(_PROPERTY_IDS),
        "directions": {
            "delta_e_st_ev": "minimize",
            "s1_ev": "maximize",
        },
        "constraints": {},
        "feature_policy": "exact_pr_ao_model_feature_contract",
        "feature_generator_profile": {"feature_type": "morgan_ecfp", "n_bits": 128},
        "scoring_policy": "pareto_then_mean_rank_percentile.v1",
    }
    screening_id = "oled-registry-screening:" + batch_runner._stable_hash(
        {
            "phase1_execution_id": sources["phase1_execution_id"],
            "phase1_execution_sha256": sources["phase1_execution_sha256"],
            "dataset_snapshot_digest": sources["dataset_snapshot_digest"],
            "dataset_snapshot_sha256": sources["dataset_snapshot_sha256"],
            "registry_snapshot_digest": sources["registry_snapshot_digest"],
            "registry_snapshot_sha256": sources["registry_snapshot_sha256"],
            "config": config,
        }
    )
    receipt = {
        "screening_version": "oled_registry_candidate_screening.v1",
        "screening_id": screening_id,
        "generated_at": "2026-07-20T09:00:00+08:00",
        "status": "completed",
        "sources": sources,
        "config": config,
        "counts": {
            "registry_candidate_count": 5,
            "eligible_candidate_count": len(shortlist_rows),
            "excluded_candidate_count": 2,
            "prediction_count": len(shortlist_rows),
            "shortlist_count": len(shortlist_rows),
        },
        "reason_code_counts": {},
        "artifacts": {
            "eligible_candidates.csv": "sha256:" + "4" * 64,
            "excluded_candidates.jsonl": "sha256:" + "5" * 64,
            "predictions.jsonl": "sha256:" + "6" * 64,
            "ranked_shortlist.csv": _sha256(shortlist),
        },
        "claims": {
            "independent_registry_candidate_pool": True,
            "training_identity_exclusion_applied": True,
            "experimental_validation_claimed": False,
            "benchmark_validated": False,
            "production_ready": False,
            "model_registered": False,
            "registry_mutated": False,
        },
    }
    screening_path = tmp_path / "screening.json"
    screening_path.write_bytes(_json_bytes(receipt))
    return screening_path, shortlist_path


def _cost_manifest_path(
    tmp_path: Path,
    *,
    screening_path: Path,
    shortlist_path: Path,
    rows: list[dict[str, object]],
    costs: list[int],
) -> Path:
    receipt = json.loads(screening_path.read_text(encoding="utf-8"))
    manifest = {
        "cost_manifest_version": "oled_candidate_cost_manifest.v1",
        "screening_id": receipt["screening_id"],
        "ranked_shortlist_sha256": _sha256(shortlist_path.read_bytes()),
        "currency": "USD",
        "entries": [
            {
                "material_id": row["material_id"],
                "registry_entry_digest": row["registry_entry_digest"],
                "cost_minor": cost,
            }
            for row, cost in zip(rows, costs, strict=True)
        ],
    }
    path = tmp_path / "candidate-costs.json"
    path.write_bytes(_json_bytes(manifest))
    return path


def test_loader_binds_exact_screening_shortlist_and_optional_cost_manifest(
    tmp_path: Path,
) -> None:
    rows = _rows()
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)
    costs_path = _cost_manifest_path(
        tmp_path,
        screening_path=screening_path,
        shortlist_path=shortlist_path,
        rows=rows,
        costs=[500, 800, 100],
    )

    loaded = load_oled_experiment_batch_selection_inputs(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        candidate_cost_manifest_json=costs_path,
    )

    assert loaded.screening_id.startswith("oled-registry-screening:")
    assert loaded.screening_sha256 == _sha256(screening_path.read_bytes())
    assert loaded.shortlist_sha256 == _sha256(shortlist_path.read_bytes())
    assert loaded.cost_manifest_sha256 == _sha256(costs_path.read_bytes())
    assert loaded.property_ids == _PROPERTY_IDS
    assert loaded.cost_currency == "USD"
    assert loaded.costs_by_candidate[("material:batch-a", "sha256:" + "1" * 64)] == 500


def test_ready_batch_is_diverse_deterministic_and_recommendation_only(
    tmp_path: Path,
) -> None:
    screening_path, shortlist_path = _screening_paths(tmp_path)

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=tmp_path / "batches",
        target_batch_size=2,
        maximums=["delta_e_st_ev=0.20"],
        minimums=["s1_ev=3.00"],
        max_pairwise_tanimoto=0.0,
        generated_at="2026-07-20T10:00:00+08:00",
    )

    assert result.status == "ready"
    assert result.selected_count == 2
    assert result.eligible_count == 3
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "batch_selection.json",
        "experiment_batch.csv",
        "experiment_handoff.md",
    ]
    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert [item["material_id"] for item in receipt["selection"]["selected_candidates"]] == [
        "material:batch-a",
        "material:batch-b",
    ]
    assert receipt["claims"] == {
        "recommendation_only": True,
        "experiment_started": False,
        "experiment_completed": False,
        "experiment_executed": False,
        "procurement_started": False,
        "procurement_performed": False,
        "synthesis_started": False,
        "synthesis_performed": False,
        "measurement_started": False,
        "measurement_performed": False,
        "experimental_validation_claimed": False,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
        "model_registered": False,
    }
    assert receipt["artifacts"]["experiment_batch.csv"] == _sha256(
        (result.output_dir / "experiment_batch.csv").read_bytes()
    )
    fingerprint = receipt["config"]["diversity"]["fingerprint"]
    assert fingerprint["generator"] == "rdkit.AllChem.GetMorganFingerprintAsBitVect.v1"
    assert fingerprint["radius"] == 2
    assert fingerprint["n_bits"] == 2048
    assert fingerprint["use_chirality"] is False
    assert fingerprint["use_features"] is False
    assert "does not claim or start procurement" in (
        result.output_dir / "experiment_handoff.md"
    ).read_text(encoding="utf-8")


def test_budget_manifest_is_exactly_bound_and_selection_respects_minor_units(
    tmp_path: Path,
) -> None:
    rows = _rows()
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)
    costs_path = _cost_manifest_path(
        tmp_path,
        screening_path=screening_path,
        shortlist_path=shortlist_path,
        rows=rows,
        costs=[500, 800, 100],
    )

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=tmp_path / "batches",
        target_batch_size=2,
        max_budget_minor=600,
        max_pairwise_tanimoto=1.0,
        candidate_cost_manifest_json=costs_path,
        generated_at="2026-07-20T10:01:00+08:00",
    )

    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert result.status == "ready"
    assert result.total_cost_minor == 600
    assert [item["material_id"] for item in receipt["selection"]["selected_candidates"]] == [
        "material:batch-a",
        "material:batch-c",
    ]
    assert receipt["selection"]["currency"] == "USD"

    malformed = json.loads(costs_path.read_text(encoding="utf-8"))
    malformed["screening_id"] = "oled-registry-screening:other"
    costs_path.write_bytes(_json_bytes(malformed))
    with pytest.raises(ValueError, match="screening binding mismatch"):
        load_oled_experiment_batch_selection_inputs(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            candidate_cost_manifest_json=costs_path,
        )


def test_valid_infeasible_request_publishes_not_ready_without_partial_batch(
    tmp_path: Path,
) -> None:
    rows = _rows()[:2]
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=tmp_path / "batches",
        target_batch_size=3,
        max_pairwise_tanimoto=0.0,
        generated_at="2026-07-20T10:02:00+08:00",
    )

    assert result.status == "not_ready"
    assert result.selected_count == 0
    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert receipt["selection"]["selected_candidates"] == []
    assert "insufficient_eligible_candidates" in receipt["selection"]["not_ready_reasons"]
    csv_lines = (result.output_dir / "experiment_batch.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(csv_lines) == 1
    assert "No partial material batch is provided" in (
        result.output_dir / "experiment_handoff.md"
    ).read_text(encoding="utf-8")


def test_missing_costs_are_a_not_ready_outcome_under_a_money_budget(
    tmp_path: Path,
) -> None:
    rows = _rows()
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)
    costs_path = _cost_manifest_path(
        tmp_path,
        screening_path=screening_path,
        shortlist_path=shortlist_path,
        rows=rows[:1],
        costs=[500],
    )

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=tmp_path / "batches",
        target_batch_size=2,
        max_budget_minor=1_000,
        max_pairwise_tanimoto=1.0,
        candidate_cost_manifest_json=costs_path,
        generated_at="2026-07-20T10:02:30+08:00",
    )

    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert result.status == "not_ready"
    assert result.selected_count == 0
    reasons = {
        item["material_id"]: item["reason_codes"]
        for item in receipt["selection"]["candidate_decisions"]
    }
    assert reasons["material:batch-b"] == ["candidate_cost_unavailable"]
    assert reasons["material:batch-c"] == ["candidate_cost_unavailable"]


def test_partial_cost_manifest_never_reports_a_partial_total_without_budget(
    tmp_path: Path,
) -> None:
    rows = _rows()
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)
    costs_path = _cost_manifest_path(
        tmp_path,
        screening_path=screening_path,
        shortlist_path=shortlist_path,
        rows=rows[:1],
        costs=[500],
    )

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=tmp_path / "batches",
        target_batch_size=2,
        max_pairwise_tanimoto=1.0,
        candidate_cost_manifest_json=costs_path,
        generated_at="2026-07-20T10:02:40+08:00",
    )

    assert result.status == "ready"
    assert result.total_cost_minor is None
    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert receipt["selection"]["total_cost_minor"] is None


def test_tampered_shortlist_or_invalid_options_fail_before_publication(
    tmp_path: Path,
) -> None:
    screening_path, shortlist_path = _screening_paths(tmp_path)
    shortlist_path.write_bytes(shortlist_path.read_bytes() + b"\n")
    output_root = tmp_path / "batches"

    with pytest.raises(ValueError, match="shortlist SHA-256 mismatch"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=1,
        )
    assert not output_root.exists() or not any(output_root.iterdir())

    screening_path, shortlist_path = _screening_paths(tmp_path / "fresh")
    with pytest.raises(ValueError, match="unknown property"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=1,
            minimums=["plqy=0.9"],
        )
    with pytest.raises(ValueError, match="required for a multi-material"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=2,
        )
    assert not output_root.exists() or not any(output_root.iterdir())


def test_duplicate_shortlist_chemical_identity_fails_before_publication(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows[2] = {
        **rows[2],
        "material_id": "material:batch-duplicate",
        "registry_entry_digest": "sha256:" + "9" * 64,
        "canonical_isomeric_smiles": rows[0]["canonical_isomeric_smiles"],
    }
    screening_path, shortlist_path = _screening_paths(tmp_path, rows)
    output_root = tmp_path / "batches"

    with pytest.raises(ValueError, match="chemical identity is duplicated"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=1,
        )

    assert not output_root.exists() or not any(output_root.iterdir())


def test_diversity_requires_rdkit_and_existing_batch_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screening_path, shortlist_path = _screening_paths(tmp_path)
    output_root = tmp_path / "batches"
    original_chem = batch_runner.Chem
    original_all_chem = batch_runner.AllChem
    original_data_structs = batch_runner.DataStructs
    original_rd_base = batch_runner.rdBase
    monkeypatch.setattr(batch_runner, "Chem", None)
    monkeypatch.setattr(batch_runner, "AllChem", None)
    monkeypatch.setattr(batch_runner, "DataStructs", None)
    monkeypatch.setattr(batch_runner, "rdBase", None)
    with pytest.raises(ValueError, match="requires RDKit"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=2,
            max_pairwise_tanimoto=0.5,
        )
    assert not output_root.exists() or not any(output_root.iterdir())
    monkeypatch.setattr(batch_runner, "Chem", original_chem)
    monkeypatch.setattr(batch_runner, "AllChem", original_all_chem)
    monkeypatch.setattr(batch_runner, "DataStructs", original_data_structs)
    monkeypatch.setattr(batch_runner, "rdBase", original_rd_base)

    result = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        output_root=output_root,
        target_batch_size=1,
        generated_at="2026-07-20T10:03:00+08:00",
    )
    marker = result.output_dir / "marker.txt"
    marker.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=1,
            generated_at="2026-07-20T10:04:00+08:00",
        )
    assert marker.read_text(encoding="utf-8") == "do not replace\n"


def test_output_parent_replacement_after_input_load_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screening_path, shortlist_path = _screening_paths(tmp_path)
    output_root = tmp_path / "batches"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved-batches"
    original_select = batch_runner._select_batch
    replaced = False

    def replace_parent(*args: object, **kwargs: object):
        nonlocal replaced
        outcome = original_select(*args, **kwargs)
        if not replaced:
            output_root.rename(moved)
            replacement.mkdir()
            replacement.rename(output_root)
            replaced = True
        return outcome

    monkeypatch.setattr(batch_runner, "_select_batch", replace_parent)
    with pytest.raises(ValueError, match="output parent changed"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=output_root,
            target_batch_size=1,
    )
    assert replaced is True
    assert not any(output_root.iterdir())
    assert not any(moved.iterdir())


def test_output_symlink_component_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    screening_path, shortlist_path = _screening_paths(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic or unsafe"):
        run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_path,
            ranked_shortlist_csv=shortlist_path,
            output_root=redirected_parent / "batches",
            target_batch_size=1,
        )

    assert not (real_parent / "batches").exists()
