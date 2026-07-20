from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai4s_agent import oled_experiment_batch_selection as batch_runner
from ai4s_agent.oled_experiment_batch_selection import (
    OledExperimentBatchSelectionInputs,
    OledExperimentBatchSelectionResult,
    _parse_ranked_shortlist,
    load_oled_experiment_batch_selection_inputs,
    run_oled_experiment_batch_selection_from_files,
)
from ai4s_agent.oled_registry_candidate_screening import (
    run_oled_registry_candidate_screening_from_files,
)
from tests.test_oled_registry_candidate_screening import _screening_inputs


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


@dataclass(frozen=True)
class _ScreeningPublication:
    screening_receipt: Path
    ranked_shortlist: Path
    phase1_execution_dir: Path
    dataset_snapshot: Path
    registry_snapshot: Path


def _screening_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _ScreeningPublication:
    """Build a real PR-AP publication and retain its exact replay inputs."""

    anchor_root = tmp_path / "pr-ap-anchor"
    anchor_root.mkdir(parents=True, exist_ok=True)
    source = _screening_inputs(anchor_root, monkeypatch)
    result = run_oled_registry_candidate_screening_from_files(
        phase1_execution_dir=source.execution_dir,
        dataset_snapshot_json=source.dataset_snapshot,
        registry_snapshot_json=source.registry_snapshot,
        output_root=tmp_path / "screenings",
        generated_at="2026-07-20T09:00:00+08:00",
    )
    return _ScreeningPublication(
        screening_receipt=result.output_dir / "screening.json",
        ranked_shortlist=result.output_dir / "ranked_shortlist.csv",
        phase1_execution_dir=source.execution_dir,
        dataset_snapshot=source.dataset_snapshot,
        registry_snapshot=source.registry_snapshot,
    )


def _shortlist_rows(publication: _ScreeningPublication) -> list[dict[str, object]]:
    with publication.ranked_shortlist.open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _load_batch_inputs(
    publication: _ScreeningPublication,
    *,
    candidate_cost_manifest_json: Path | None = None,
) -> OledExperimentBatchSelectionInputs:
    return load_oled_experiment_batch_selection_inputs(
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
    )


def _run_batch(
    publication: _ScreeningPublication,
    **kwargs: object,
) -> OledExperimentBatchSelectionResult:
    return run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        **kwargs,
    )


def _cost_manifest_path(
    tmp_path: Path,
    *,
    publication: _ScreeningPublication,
    rows: list[dict[str, object]],
    costs: list[int],
) -> Path:
    receipt = json.loads(publication.screening_receipt.read_text(encoding="utf-8"))
    manifest = {
        "cost_manifest_version": "oled_candidate_cost_manifest.v1",
        "screening_id": receipt["screening_id"],
        "ranked_shortlist_sha256": _sha256(publication.ranked_shortlist.read_bytes()),
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    costs_path = _cost_manifest_path(
        tmp_path,
        publication=publication,
        rows=rows,
        costs=[500, 800],
    )

    loaded = _load_batch_inputs(
        publication,
        candidate_cost_manifest_json=costs_path,
    )

    assert loaded.screening_id.startswith("oled-registry-screening:")
    assert loaded.screening_sha256 == _sha256(
        publication.screening_receipt.read_bytes()
    )
    assert loaded.shortlist_sha256 == _sha256(
        publication.ranked_shortlist.read_bytes()
    )
    assert loaded.cost_manifest_sha256 == _sha256(costs_path.read_bytes())
    assert loaded.property_ids == _PROPERTY_IDS
    assert loaded.cost_currency == "USD"
    assert loaded.costs_by_candidate[
        (str(rows[0]["material_id"]), str(rows[0]["registry_entry_digest"]))
    ] == 500


def test_fully_resigned_shortlist_fails_exact_pr_ap_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-consistent receipt/CSV pair cannot replace PR-AP's publication."""

    publication = _screening_publication(tmp_path, monkeypatch)
    forged_rows = _shortlist_rows(publication)
    forged_rows[0] = {
        **forged_rows[0],
        "material_id": "material:forged-shortlist-entry",
        "registry_entry_digest": "sha256:" + "f" * 64,
        "canonical_name": "forged candidate",
        "canonical_isomeric_smiles": "CO",
        "aggregate_percentile": "0.500000",
        "predicted_delta_e_st_ev": "0.123000",
        "predicted_s1_ev": "3.456000",
    }
    forged_shortlist = _shortlist_bytes(forged_rows)
    publication.ranked_shortlist.write_bytes(forged_shortlist)
    forged_receipt = json.loads(
        publication.screening_receipt.read_text(encoding="utf-8")
    )
    original_screening_id = forged_receipt["screening_id"]
    forged_receipt["artifacts"]["ranked_shortlist.csv"] = _sha256(forged_shortlist)
    publication.screening_receipt.write_bytes(_json_bytes(forged_receipt))

    assert forged_receipt["screening_id"] == original_screening_id
    with pytest.raises(ValueError, match="PR-AP screening exact replay mismatch"):
        _load_batch_inputs(publication)


def test_ready_batch_is_diverse_deterministic_and_recommendation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)

    result = _run_batch(
        publication,
        output_root=tmp_path / "batches",
        target_batch_size=2,
        max_pairwise_tanimoto=1.0,
        generated_at="2026-07-20T10:00:00+08:00",
    )

    assert result.status == "ready"
    assert result.selected_count == 2
    assert result.eligible_count == 2
    assert result.candidate_supply_count == 2
    assert result.inverse_design_should_trigger is False
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "batch_selection.json",
        "candidate_decision_dossier.csv",
        "experiment_batch.csv",
        "experiment_handoff.md",
    ]
    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert [
        item["material_id"] for item in receipt["selection"]["selected_candidates"]
    ] == [row["material_id"] for row in rows]
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
    assert receipt["artifacts"]["candidate_decision_dossier.csv"] == _sha256(
        (result.output_dir / "candidate_decision_dossier.csv").read_bytes()
    )
    fingerprint = receipt["config"]["diversity"]["fingerprint"]
    assert fingerprint["generator"] == "rdkit.AllChem.GetMorganFingerprintAsBitVect.v1"
    assert fingerprint["radius"] == 2
    assert fingerprint["n_bits"] == 2048
    assert fingerprint["use_chirality"] is False
    assert fingerprint["use_features"] is False
    trace = receipt["selection"]["greedy_trace"]
    assert trace["termination"] == "target_reached"
    assert trace["steps"][0]["provisional_choice"]["material_id"] == rows[0][
        "material_id"
    ]
    assert trace["steps"][0]["finalized_choice"]["material_id"] == rows[0][
        "material_id"
    ]
    second_step = trace["steps"][1]
    assert second_step["finalized_choice"]["material_id"] == rows[1]["material_id"]
    assert second_step["evaluations"][0]["max_pairwise_tanimoto"] == 1.0
    assert second_step["evaluations"][0]["diversity_status"] == (
        "within_max_pairwise_tanimoto"
    )
    handoff = (result.output_dir / "experiment_handoff.md").read_text(
        encoding="utf-8"
    )
    assert "maximum pairwise Morgan/Tanimoto similarity is `1.0`" in handoff
    assert "does not claim or start procurement" in handoff


def test_candidate_decision_dossier_explains_ranking_only_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    shortlist_rows = _shortlist_rows(publication)

    result = _run_batch(
        publication,
        output_root=tmp_path / "batches",
        target_batch_size=1,
        generated_at="2026-07-20T10:00:30+08:00",
    )

    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert receipt["batch_selection_version"] == "oled_experiment_batch_selection.v2"
    decisions = receipt["selection"]["candidate_decisions"]
    assert [item["material_id"] for item in decisions] == [
        row["material_id"] for row in shortlist_rows
    ]
    selected, unselected = decisions
    assert selected["selection_status"] == "selected"
    assert selected["selection_order"] == 1
    assert "highest_ranked_feasible_candidate" in selected["reason_codes"]
    assert "no_additional_batch_constraints_requested" in selected["reason_codes"]
    assert unselected["selection_status"] == "eligible_not_selected"
    assert unselected["reason_codes"] == [
        "eligible_but_target_batch_filled",
        "not_selected_by_rank_anchored_greedy_policy",
    ]
    properties = {item["property_id"]: item for item in selected["properties"]}
    assert properties["delta_e_st_ev"]["display_name"] == "Singlet-triplet energy gap"
    assert properties["delta_e_st_ev"]["unit"] == "eV"
    assert properties["delta_e_st_ev"]["objective_direction"] == "minimize"
    assert properties["s1_ev"]["display_name"] == "First singlet excited-state energy"
    assert properties["s1_ev"]["unit"] == "eV"
    assert properties["s1_ev"]["objective_direction"] == "maximize"
    assert all(
        item["batch_constraint"]["status"] == "not_requested"
        for item in selected["properties"]
    )

    with (result.output_dir / "experiment_batch.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        batch_reader = csv.DictReader(stream)
        batch_rows = list(batch_reader)
    assert batch_reader.fieldnames is not None
    assert batch_reader.fieldnames[:10] == [
        "selection_order",
        "source_rank",
        "material_id",
        "registry_entry_digest",
        "canonical_name",
        "canonical_isomeric_smiles",
        "aggregate_percentile",
        "cost_minor",
        "currency",
        "maximum_similarity_to_prior",
    ]
    assert len(batch_rows) == 1
    assert batch_rows[0]["selection_status"] == "selected"
    assert "highest_ranked_feasible_candidate" in batch_rows[0]["selection_reason_codes"]
    assert batch_rows[0]["property_unit_delta_e_st_ev"] == "eV"

    with (result.output_dir / "candidate_decision_dossier.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        dossier_reader = csv.DictReader(stream)
        dossier_rows = list(dossier_reader)
    assert dossier_reader.fieldnames is not None
    assert dossier_reader.fieldnames.count("selection_status") == 1
    assert [row["material_id"] for row in dossier_rows] == [
        row["material_id"] for row in shortlist_rows
    ]
    assert dossier_rows[0]["selection_status"] == "selected"
    assert dossier_rows[0]["property_name_delta_e_st_ev"] == "Singlet-triplet energy gap"
    assert dossier_rows[0]["property_unit_s1_ev"] == "eV"
    assert dossier_rows[0]["batch_constraint_status_s1_ev"] == "not_requested"
    assert dossier_rows[0]["selection_terminal_status"] == "chosen_at_step:1"
    assert dossier_rows[0]["preflight_budget_status"] == "not_requested"
    assert json.loads(dossier_rows[0]["selection_step_evidence_json"])[0][
        "decision_at_step"
    ] == "chosen"
    handoff = (result.output_dir / "experiment_handoff.md").read_text(
        encoding="utf-8"
    )
    assert "No additional batch property constraints were supplied" in handoff
    assert "Top-N is therefore selected" in handoff
    assert "bound ranking" not in handoff
    assert shortlist_rows[1]["canonical_name"] in handoff


def test_candidate_decision_dossier_reports_per_property_batch_constraint_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    shortlist_rows = _shortlist_rows(publication)
    property_id = "s1_ev"
    value = float(shortlist_rows[0][f"predicted_{property_id}"])
    passing = _run_batch(
        publication,
        output_root=tmp_path / "passing-batches",
        target_batch_size=1,
        minimums=[f"{property_id}={value:.17g}"],
        generated_at="2026-07-20T10:00:40+08:00",
    )
    receipt = json.loads(
        (passing.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    decisions = receipt["selection"]["candidate_decisions"]
    selected = next(item for item in decisions if item["selected"])
    selected_property = next(
        item for item in selected["properties"] if item["property_id"] == property_id
    )
    assert selected_property["batch_constraint"] == {
        "min": value,
        "max": None,
        "status": "passed",
    }

    failing = _run_batch(
        publication,
        output_root=tmp_path / "failing-batches",
        target_batch_size=1,
        minimums=[f"{property_id}={value + 0.1:.17g}"],
        generated_at="2026-07-20T10:00:41+08:00",
    )
    failed_receipt = json.loads(
        (failing.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    excluded = failed_receipt["selection"]["candidate_decisions"][0]
    excluded_property = next(
        item for item in excluded["properties"] if item["property_id"] == property_id
    )
    assert excluded_property["batch_constraint"] == {
        "min": value + 0.1,
        "max": None,
        "status": "failed_minimum",
    }
    assert f"hard_constraint_failed:{property_id}:min" in excluded["reason_codes"]
    assert excluded["selection_status"] == "excluded_by_batch_policy"


def test_candidate_decision_dossier_bytes_are_deterministic_for_exact_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    kwargs = {
        "target_batch_size": 1,
        "generated_at": "2026-07-20T10:00:50+08:00",
    }
    first = _run_batch(publication, output_root=tmp_path / "first", **kwargs)
    second = _run_batch(publication, output_root=tmp_path / "second", **kwargs)

    assert first.batch_id == second.batch_id
    assert {
        path.name: path.read_bytes() for path in first.output_dir.iterdir()
    } == {
        path.name: path.read_bytes() for path in second.output_dir.iterdir()
    }


def test_property_presentation_contract_changes_batch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    kwargs = {
        "target_batch_size": 1,
        "generated_at": "2026-07-20T10:00:55+08:00",
    }
    first = _run_batch(publication, output_root=tmp_path / "first", **kwargs)
    original_descriptor = batch_runner._property_descriptor

    def changed_descriptor(property_id: str) -> dict[str, str]:
        descriptor = original_descriptor(property_id)
        if property_id == "s1_ev":
            return {
                **descriptor,
                "display_name": "Frozen alternate S1 presentation",
            }
        return descriptor

    monkeypatch.setattr(batch_runner, "_property_descriptor", changed_descriptor)
    second = _run_batch(publication, output_root=tmp_path / "second", **kwargs)

    assert first.batch_id != second.batch_id
    second_receipt = json.loads(
        (second.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert second_receipt["config"]["property_presentation"]["s1_ev"][
        "display_name"
    ] == "Frozen alternate S1 presentation"
    assert "Frozen alternate S1 presentation" in (
        second.output_dir / "candidate_decision_dossier.csv"
    ).read_text(encoding="utf-8")


def test_budget_manifest_is_exactly_bound_and_selection_respects_minor_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    costs_path = _cost_manifest_path(
        tmp_path,
        publication=publication,
        rows=rows,
        costs=[700, 500],
    )

    result = _run_batch(
        publication,
        output_root=tmp_path / "batches",
        target_batch_size=1,
        max_budget_minor=600,
        max_pairwise_tanimoto=1.0,
        candidate_cost_manifest_json=costs_path,
        generated_at="2026-07-20T10:01:00+08:00",
    )

    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert result.status == "ready"
    assert result.total_cost_minor == 500
    assert [
        item["material_id"] for item in receipt["selection"]["selected_candidates"]
    ] == [rows[1]["material_id"]]
    assert receipt["selection"]["currency"] == "USD"
    decisions_by_material = {
        item["material_id"]: item for item in receipt["selection"]["candidate_decisions"]
    }
    too_expensive = decisions_by_material[rows[0]["material_id"]][
        "selection_evidence"
    ]
    assert too_expensive["preflight"] == {
        "initial_eligibility": "excluded_before_greedy_selection",
        "hard_constraint_reason_codes": [],
        "candidate_cost_minor": 700,
        "currency": "USD",
        "max_budget_minor": 600,
        "budget_status": "exceeds_per_candidate_limit",
        "max_pairwise_tanimoto": 1.0,
    }
    assert too_expensive["selection_steps"] == []
    selected_evidence = decisions_by_material[rows[1]["material_id"]][
        "selection_evidence"
    ]
    assert selected_evidence["preflight"]["candidate_cost_minor"] == 500
    assert selected_evidence["preflight"]["budget_status"] == (
        "within_per_candidate_limit"
    )
    assert selected_evidence["selection_steps"][0][
        "cumulative_cost_if_selected_minor"
    ] == 500
    assert "maximum cumulative selected-batch cost is `600` `USD` minor units" in (
        result.output_dir / "experiment_handoff.md"
    ).read_text(encoding="utf-8")

    malformed = json.loads(costs_path.read_text(encoding="utf-8"))
    malformed["screening_id"] = "oled-registry-screening:other"
    costs_path.write_bytes(_json_bytes(malformed))
    with pytest.raises(ValueError, match="screening binding mismatch"):
        _load_batch_inputs(
            publication,
            candidate_cost_manifest_json=costs_path,
        )


def test_greedy_trace_reports_cumulative_budget_and_diversity_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    costs_path = _cost_manifest_path(
        tmp_path,
        publication=publication,
        rows=rows,
        costs=[400, 400],
    )

    budget_result = _run_batch(
        publication,
        output_root=tmp_path / "budget-batches",
        target_batch_size=2,
        max_budget_minor=600,
        max_pairwise_tanimoto=1.0,
        candidate_cost_manifest_json=costs_path,
        generated_at="2026-07-20T10:01:10+08:00",
    )
    budget_receipt = json.loads(
        (budget_result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert budget_result.status == "not_ready"
    assert budget_receipt["selection"]["selected_candidates"] == []
    assert budget_receipt["selection"]["candidate_supply"] == {
        "shortlist_candidate_count": 2,
        "property_eligible_candidate_count": 2,
        "target_batch_size": 2,
        "candidate_shortfall_count": 0,
        "candidate_quantity_status": "sufficient_after_property_constraints",
        "inverse_design_should_trigger": False,
        "inverse_design_reason": "candidate_quantity_sufficient_no_generation_requested",
        "generation_executed": False,
        "required_return_path": [],
        "ready_batch_status": "not_ready",
        "non_supply_policy_prevented_ready_batch": True,
    }
    budget_trace = budget_receipt["selection"]["greedy_trace"]
    assert budget_trace["termination"] == "no_feasible_candidate"
    assert budget_trace["provisional_selected"] == [
        {
            "material_id": rows[0]["material_id"],
            "registry_entry_digest": rows[0]["registry_entry_digest"],
        }
    ]
    assert budget_trace["finalized_selected"] == []
    rejected_by_budget = budget_trace["steps"][1]["evaluations"][0]
    assert rejected_by_budget["candidate_cost_minor"] == 400
    assert rejected_by_budget["provisional_cost_minor_before"] == 400
    assert rejected_by_budget["cumulative_cost_if_selected_minor"] == 800
    assert rejected_by_budget["budget_status"] == "exceeds_remaining_budget"
    assert rejected_by_budget["decision_at_step"] == "infeasible_budget"
    budget_decisions = {
        item["material_id"]: item
        for item in budget_receipt["selection"]["candidate_decisions"]
    }
    assert "exceeded the remaining batch budget" in budget_decisions[
        rows[1]["material_id"]
    ]["selection_basis"]
    provisional = budget_decisions[rows[0]["material_id"]]
    assert provisional["selection_status"] == "eligible_but_batch_not_ready"
    assert provisional["selection_evidence"]["terminal_status"] == (
        "provisionally_chosen_not_finalized_at_step:1"
    )
    assert provisional["selection_evidence"]["selection_steps"][0][
        "decision_at_step"
    ] == "provisionally_chosen"
    assert "not finalized" in provisional["selection_basis"]
    budget_handoff = (budget_result.output_dir / "experiment_handoff.md").read_text(
        encoding="utf-8"
    )
    assert "Inverse-design routing requested: `False`" in budget_handoff
    assert "candidate quantity is sufficient" in budget_handoff

    monkeypatch.setattr(batch_runner, "_tanimoto_similarity", lambda *_: 0.9)
    diversity_result = _run_batch(
        publication,
        output_root=tmp_path / "diversity-batches",
        target_batch_size=2,
        max_pairwise_tanimoto=0.3,
        generated_at="2026-07-20T10:01:11+08:00",
    )
    diversity_receipt = json.loads(
        (diversity_result.output_dir / "batch_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert diversity_result.status == "not_ready"
    assert diversity_receipt["selection"]["candidate_supply"][
        "inverse_design_should_trigger"
    ] is False
    rejected_by_diversity = diversity_receipt["selection"]["greedy_trace"][
        "steps"
    ][1]["evaluations"][0]
    assert rejected_by_diversity["maximum_similarity_to_prior"] == 0.9
    assert rejected_by_diversity["max_pairwise_tanimoto"] == 0.3
    assert rejected_by_diversity["diversity_status"] == "exceeds_max_pairwise_tanimoto"
    assert rejected_by_diversity["decision_at_step"] == "infeasible_diversity"


def test_valid_infeasible_request_publishes_not_ready_without_partial_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)

    result = _run_batch(
        publication,
        output_root=tmp_path / "batches",
        target_batch_size=3,
        max_pairwise_tanimoto=1.0,
        generated_at="2026-07-20T10:02:00+08:00",
    )

    assert result.status == "not_ready"
    assert result.selected_count == 0
    assert result.candidate_supply_count == 2
    assert result.inverse_design_should_trigger is True
    receipt = json.loads(
        (result.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert receipt["selection"]["selected_candidates"] == []
    assert receipt["selection"]["candidate_supply"] == {
        "shortlist_candidate_count": 2,
        "property_eligible_candidate_count": 2,
        "target_batch_size": 3,
        "candidate_shortfall_count": 1,
        "candidate_quantity_status": "insufficient_after_property_constraints",
        "inverse_design_should_trigger": True,
        "inverse_design_reason": "candidate_quantity_insufficient_after_property_constraints",
        "generation_executed": False,
        "required_return_path": [
            "inverse_design",
            "controlled_prediction",
            "filter",
            "rank",
            "candidate_decision_dossier",
        ],
        "ready_batch_status": "not_ready",
        "non_supply_policy_prevented_ready_batch": False,
    }
    assert "insufficient_eligible_candidates" in receipt["selection"]["not_ready_reasons"]
    csv_lines = (result.output_dir / "experiment_batch.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(csv_lines) == 1
    handoff = (result.output_dir / "experiment_handoff.md").read_text(
        encoding="utf-8"
    )
    assert "No partial Top-N batch is provided" in handoff
    assert "Inverse-design routing requested: `True`" in handoff
    assert "No generation is executed here" in handoff


def test_missing_costs_are_a_not_ready_outcome_under_a_money_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    costs_path = _cost_manifest_path(
        tmp_path,
        publication=publication,
        rows=rows[:1],
        costs=[500],
    )

    result = _run_batch(
        publication,
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
    assert reasons[rows[1]["material_id"]] == ["candidate_cost_unavailable"]


def test_partial_cost_manifest_never_reports_a_partial_total_without_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    costs_path = _cost_manifest_path(
        tmp_path,
        publication=publication,
        rows=rows[:1],
        costs=[500],
    )

    result = _run_batch(
        publication,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    publication.ranked_shortlist.write_bytes(
        publication.ranked_shortlist.read_bytes() + b"\n"
    )
    output_root = tmp_path / "batches"

    with pytest.raises(ValueError, match="shortlist SHA-256 mismatch"):
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=1,
        )
    assert not output_root.exists() or not any(output_root.iterdir())

    publication = _screening_publication(tmp_path / "fresh", monkeypatch)
    with pytest.raises(ValueError, match="unknown property"):
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=1,
            minimums=["plqy=0.9"],
        )
    with pytest.raises(ValueError, match="required for a multi-material"):
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=2,
        )
    assert not output_root.exists() or not any(output_root.iterdir())


def test_duplicate_shortlist_chemical_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    rows = _shortlist_rows(publication)
    duplicate = {
        **rows[1],
        "rank": 3,
        "material_id": "material:batch-duplicate",
        "registry_entry_digest": "sha256:" + "9" * 64,
        "canonical_isomeric_smiles": rows[0]["canonical_isomeric_smiles"],
    }

    with pytest.raises(ValueError, match="chemical identity is duplicated"):
        _parse_ranked_shortlist(
            _shortlist_bytes([rows[0], rows[1], duplicate]),
            property_ids=_PROPERTY_IDS,
        )


def test_diversity_requires_rdkit_and_existing_batch_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
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
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=2,
            max_pairwise_tanimoto=0.5,
        )
    assert not output_root.exists() or not any(output_root.iterdir())
    monkeypatch.setattr(batch_runner, "Chem", original_chem)
    monkeypatch.setattr(batch_runner, "AllChem", original_all_chem)
    monkeypatch.setattr(batch_runner, "DataStructs", original_data_structs)
    monkeypatch.setattr(batch_runner, "rdBase", original_rd_base)

    result = _run_batch(
        publication,
        output_root=output_root,
        target_batch_size=1,
        generated_at="2026-07-20T10:03:00+08:00",
    )
    marker = result.output_dir / "marker.txt"
    marker.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=1,
            generated_at="2026-07-20T10:04:00+08:00",
        )
    assert marker.read_text(encoding="utf-8") == "do not replace\n"


def test_output_parent_replacement_after_input_load_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
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
        _run_batch(
            publication,
            output_root=output_root,
            target_batch_size=1,
    )
    assert replaced is True
    assert not any(output_root.iterdir())
    assert not any(moved.iterdir())


def test_output_symlink_component_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic or unsafe"):
        _run_batch(
            publication,
            output_root=redirected_parent / "batches",
            target_batch_size=1,
        )

    assert not (real_parent / "batches").exists()
