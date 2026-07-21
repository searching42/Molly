from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ai4s_agent import oled_candidate_decision as decision_runner
from ai4s_agent.oled_candidate_decision import (
    _Candidate,
    _payloads,
    _parse_shortlist,
    _select,
    run_oled_candidate_decision_from_files,
    verify_oled_candidate_decision_from_files,
)
from ai4s_agent.oled_generated_candidate_evaluation import _json_bytes, _sha256_bytes
from tests.test_oled_generated_candidate_evaluation import (
    _evaluation_inputs,
    _run_evaluation,
)


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, object, object]:
    publication, batch_receipt, inverse = _evaluation_inputs(tmp_path, monkeypatch)
    evaluation = _run_evaluation(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        inverse=inverse,
    )
    return publication, batch_receipt, inverse, evaluation


def _run(
    tmp_path: Path,
    publication: object,
    batch_receipt: Path,
    inverse: object,
    evaluation: object,
) -> object:
    return run_oled_candidate_decision_from_files(
        evaluation_json=evaluation.output_dir / "evaluation.json",  # type: ignore[attr-defined]
        inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        output_root=tmp_path / "candidate-decisions",
        generated_at="2026-07-21T18:00:00+08:00",
    )


def test_consumes_exact_pr_at_and_emits_narrow_explainable_top_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt, inverse, evaluation = _inputs(tmp_path, monkeypatch)
    result = _run(tmp_path, publication, batch_receipt, inverse, evaluation)

    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "candidate_decision.json",
        "candidate_decision_dossier.csv",
        "report.md",
        "top_candidates.csv",
    ]
    receipt = json.loads(
        (result.output_dir / "candidate_decision.json").read_text(encoding="utf-8")
    )
    assert receipt["decision_version"] == "oled_candidate_decision.v2"
    assert receipt["config"]["candidate_source_types"] == ["registry", "generated"]
    assert receipt["config"]["target_top_n"] == 3
    assert receipt["config"]["max_pairwise_tanimoto"] == 1.0
    assert receipt["claims"]["human_candidate_adjudication_performed"] is False
    assert receipt["claims"]["computational_validation_claimed"] is False
    assert receipt["claims"]["registry_mutated"] is False
    assert receipt["counts"]["selected_candidate_count"] <= receipt["counts"]["target_top_n"]
    assert receipt["next_required_step"] in {
        "end_to_end_candidate_flow_complete",
        "bounded_closed_loop_controller_may_continue",
    }

    with (result.output_dir / "candidate_decision_dossier.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    assert "candidate_id" in fields
    assert "source_kind" in fields
    assert "material_id" not in fields
    assert "registry_entry_digest" not in fields
    assert any(field.startswith("predicted_") for field in fields)
    assert any(field.endswith("_unit") for field in fields)
    assert any(field.endswith("_direction") for field in fields)
    assert any(field.endswith("_decision_status") for field in fields)
    assert rows
    assert {row["source_kind"] for row in rows}.issubset({"registry", "generated"})
    assert all(row["reason_codes"] for row in rows)

    verified = verify_oled_candidate_decision_from_files(
        decision_json=result.output_dir / "candidate_decision.json",
        evaluation_json=evaluation.output_dir / "evaluation.json",  # type: ignore[attr-defined]
        inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
    )
    assert verified.decision_id == result.decision_id
    assert verified.status == result.status


def test_verifier_rejects_fully_resigned_dossier_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt, inverse, evaluation = _inputs(tmp_path, monkeypatch)
    result = _run(tmp_path, publication, batch_receipt, inverse, evaluation)
    dossier = result.output_dir / "candidate_decision_dossier.csv"
    forged = dossier.read_bytes().replace(b"selected_by_global_rank", b"selected_by_fake_rank")
    if forged == dossier.read_bytes():
        forged = dossier.read_bytes() + b"forged\n"
    dossier.write_bytes(forged)
    receipt_path = result.output_dir / "candidate_decision.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["candidate_decision_dossier.csv"] = _sha256_bytes(forged)
    receipt_path.write_bytes(_json_bytes(receipt))

    with pytest.raises(ValueError, match="exact replay mismatch"):
        verify_oled_candidate_decision_from_files(
            decision_json=receipt_path,
            evaluation_json=evaluation.output_dir / "evaluation.json",  # type: ignore[attr-defined]
            inverse_design_json=inverse.output_dir / "inverse_design.json",  # type: ignore[attr-defined]
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
            ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
            phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
            dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
            registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        )


def test_candidate_source_roster_rejects_unplanned_generic_variant() -> None:
    payload = (
        "rank,candidate_id,source_kind,source_candidate_id,source_identity_digest,"
        "source_publication_id,canonical_name,canonical_isomeric_smiles,standard_inchi,"
        "inchikey,aggregate_percentile,predicted_p1\n"
        "1,candidate-1,external_database,source-1,digest-1,publication-1,name,CC,"
        "InChI=1S/C2H6/c1-2/h1-2H3,OTMSDBZUPAUEDD-UHFFFAOYSA-N,1.0,2.0\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="identity/source"):
        _parse_shortlist(payload, ("p1",))


def _candidate(rank: int, candidate_id: str) -> _Candidate:
    return _Candidate(
        rank=rank,
        candidate_id=candidate_id,
        source_kind="generated",
        source_candidate_id=candidate_id,
        source_identity_digest=f"digest-{candidate_id}",
        source_publication_id="publication-1",
        canonical_name=candidate_id,
        canonical_isomeric_smiles="CC",
        standard_inchi="InChI=1S/C2H6/c1-2/h1-2H3",
        inchikey=f"inchikey-{candidate_id}",
        aggregate_percentile=1.0 - rank / 10.0,
        predictions={"p1": float(rank)},
    )


def _selection_kwargs(candidates: list[_Candidate], target_count: int) -> dict[str, object]:
    return {
        "candidates": candidates,
        "property_ids": ("p1",),
        "screening_constraints": {},
        "batch_constraints": {},
        "directions": {"p1": "maximize"},
        "presentation": {
            "p1": {
                "display_name": "Property 1",
                "unit": "a.u.",
                "physical_interpretation": "test property",
                "ontology_status": "mapped",
            }
        },
        "target_count": target_count,
        "max_budget": None,
        "costs_by_candidate": {},
    }


def test_incomplete_top_n_publishes_no_final_selection() -> None:
    candidates = [_candidate(1, "a"), _candidate(2, "b")]
    selected, decisions, total_cost = _select(
        **_selection_kwargs(candidates, 3),  # type: ignore[arg-type]
        max_similarity=None,
    )
    assert selected == []
    assert total_cost is None
    assert {item["selection_status"] for item in decisions} == {
        "eligible_but_incomplete_batch"
    }
    assert all("provisional_not_final" in item["reason_codes"] for item in decisions)

    config = {
        "target_top_n": 3,
        "candidate_source_types": ["registry", "generated"],
        "constraints": {},
        "directions": {"p1": "maximize"},
        "max_pairwise_tanimoto": None,
        "max_budget_minor": None,
        "currency": None,
        "selection_policy": "rank_anchored_greedy_max_min_tanimoto.v1",
        "property_presentation": _selection_kwargs(candidates, 3)["presentation"],
    }
    payloads = _payloads(
        decision_id="oled-candidate-decision:test",
        generated_at="2026-07-21T20:00:00+08:00",
        status="incomplete",
        evaluation_receipt={"evaluation_id": "evaluation-1"},
        evaluation_sha256="sha256:" + "1" * 64,
        batch_receipt={"batch_id": "batch-1"},
        batch_sha256="sha256:" + "2" * 64,
        config=config,
        property_ids=("p1",),
        selected=selected,
        decisions=decisions,
        total_cost=total_cost,
    )
    assert list(csv.DictReader(payloads["top_candidates.csv"].decode().splitlines())) == []
    receipt = json.loads(payloads["candidate_decision.json"])
    assert receipt["selected_candidates"] == []
    assert receipt["counts"]["selected_candidate_count"] == 0
    assert not any(
        item["selection_status"] == "selected"
        for item in receipt["candidate_decisions"]
    )


def test_selection_inherits_rank_anchored_greedy_max_min_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate(1, "a"),
        _candidate(2, "b"),
        _candidate(3, "c"),
    ]
    similarities = {
        frozenset(("a", "b")): 0.80,
        frozenset(("a", "c")): 0.20,
        frozenset(("b", "c")): 0.30,
    }
    monkeypatch.setattr(
        decision_runner,
        "_fingerprints",
        lambda values: {item.candidate_id: item.candidate_id for item in values},
    )
    monkeypatch.setattr(
        decision_runner,
        "_tanimoto_similarity",
        lambda left, right: similarities[frozenset((left, right))],
    )

    selected, decisions, _ = _select(
        **_selection_kwargs(candidates, 2),  # type: ignore[arg-type]
        max_similarity=0.90,
    )

    assert [item.candidate_id for item in selected] == ["a", "c"]
    selected_decisions = [
        item for item in decisions if item["selection_status"] == "selected"
    ]
    assert [item["candidate"]["candidate_id"] for item in selected_decisions] == [
        "a",
        "c",
    ]
