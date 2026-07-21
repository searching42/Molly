from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ai4s_agent import oled_generated_candidate_evaluation as evaluator
from ai4s_agent.oled_generated_candidate_evaluation import (
    run_oled_generated_candidate_evaluation_from_files,
    verify_oled_generated_candidate_evaluation_from_files,
)
from tests.test_oled_inverse_design import _run, _shortfall_inputs, _source_csv


def _evaluation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, object]:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound PR-AT test config\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "reinvent-output.csv",
        [("generated-one", "CCCCC"), ("generated-two", "COC")],
    )
    inverse = _run(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        config=config,
        raw_output=raw_output,
    )
    return publication, batch_receipt, inverse


def _run_evaluation(
    *,
    tmp_path: Path,
    publication: object,
    batch_receipt: Path,
    inverse: object,
) -> object:
    return run_oled_generated_candidate_evaluation_from_files(
        inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        output_root=tmp_path / "generated-evaluations",
        generated_at="2026-07-21T16:00:00+08:00",
    )


def test_evaluates_generated_candidates_and_globally_reranks_without_registry_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt, inverse = _evaluation_inputs(tmp_path, monkeypatch)

    def extreme_predictions(prepared: object, smiles: str) -> dict[str, float]:
        del smiles
        return {
            property_id: (1_000_000.0 if prepared.directions[property_id] == "maximize" else -1_000_000.0)  # type: ignore[attr-defined]
            for property_id in prepared.property_ids  # type: ignore[attr-defined]
        }

    monkeypatch.setattr(evaluator, "_predict_candidate_smiles", extreme_predictions)
    result = _run_evaluation(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        inverse=inverse,
    )

    receipt = json.loads(
        (result.output_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert receipt["counts"]["generated_source_count"] == 2
    assert receipt["counts"]["generated_prediction_count"] == 2
    assert receipt["counts"]["generated_exclusion_count"] == 0
    assert receipt["claims"]["registry_and_generated_pool_globally_ranked"] is True
    assert receipt["claims"]["generated_candidates_assigned_registry_material_ids"] is False
    assert receipt["next_required_step"] == "pr_arb_candidate_decision_successor"

    with (result.output_dir / "ranked_shortlist.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert "material_id" not in (reader.fieldnames or [])
        assert "registry_entry_digest" not in (reader.fieldnames or [])
    assert rows
    assert {row["source_kind"] for row in rows} == {"inverse_design"}
    assert all(row["candidate_id"].startswith("oled-generated:") for row in rows)

    complete = [
        json.loads(line)
        for line in (result.output_dir / "complete_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_kind"] for row in complete} == {
        "registry",
        "inverse_design",
    }
    verified = verify_oled_generated_candidate_evaluation_from_files(
        evaluation_json=result.output_dir / "evaluation.json",
        inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
    )
    assert verified.evaluation_id == result.evaluation_id
    assert verified.generated_prediction_count == 2


def test_verifier_rejects_fully_resigned_successor_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt, inverse = _evaluation_inputs(tmp_path, monkeypatch)
    result = _run_evaluation(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        inverse=inverse,
    )
    shortlist_path = result.output_dir / "ranked_shortlist.csv"
    forged_shortlist = shortlist_path.read_bytes().replace(b"CCCCC", b"CCCCN")
    shortlist_path.write_bytes(forged_shortlist)
    receipt_path = result.output_dir / "evaluation.json"
    forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged_receipt["artifacts"]["ranked_shortlist.csv"] = evaluator._sha256_bytes(  # type: ignore[attr-defined]
        forged_shortlist
    )
    receipt_path.write_bytes(evaluator._json_bytes(forged_receipt))  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="exact replay mismatch"):
        verify_oled_generated_candidate_evaluation_from_files(
            evaluation_json=receipt_path,
            inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
            ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
            phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
            dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
            registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        )
