from __future__ import annotations

from typing import Any

import pytest

from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    _component_split_assignments,
    validate_candidates,
)
from tests.test_structured_dataset_confirmation import fixture_rows


def test_chemical_validation_keeps_invalid_duplicate_and_ood_findings() -> None:
    candidates = [
        {"candidate_id": "c1", "smiles": "not-a-smiles"},
        {"candidate_id": "c2", "smiles": "CCO"},
        {"candidate_id": "c3", "smiles": "OCC"},
        {"candidate_id": "c4", "smiles": "c1ccccc1"},
    ]
    results, summary = validate_candidates(
        candidates, fixture_rows(), seed=41, ad_similarity_threshold=0.95,
    )

    assert len(results) == len(candidates)
    assert results[0]["valid"] is False
    assert results[2]["duplicate"] is True
    assert results[3]["training_exact_duplicate"] is True
    assert summary["no_silent_candidate_loss"] is True
    assert summary["ood_count"] >= 1


def _sample(row: str, molecule: str, paper: str) -> dict[str, Any]:
    return {
        "row_id": row,
        "inchikey": molecule,
        "paper_id": paper,
        "features": [1.0],
        "target": 1.0,
    }


def test_component_split_never_leaks_shared_molecule_or_paper() -> None:
    samples = [
        _sample("r1", "m1", "p1"),
        _sample("r2", "m2", "p1"),
        _sample("r3", "m2", "p2"),
        _sample("r4", "m3", "p3"),
        _sample("r5", "m4", "p4"),
        _sample("r6", "m5", "p5"),
        _sample("r7", "m6", "p6"),
    ]

    assignments, components = _component_split_assignments(samples, seed=7)
    split_by_row = {item["row_id"]: item["split"] for item in assignments}

    assert split_by_row["r1"] == split_by_row["r2"] == split_by_row["r3"]
    assert len({item["split"] for item in assignments}) == 3
    assert all(item["row_ids"] for item in components)


def test_component_split_is_order_independent_and_seed_bound() -> None:
    samples = [
        _sample(f"r{index}", f"m{index}", f"p{index}")
        for index in range(1, 8)
    ]

    first, first_components = _component_split_assignments(samples, seed=17)
    replay, replay_components = _component_split_assignments(
        list(reversed(samples)), seed=17
    )
    changed, _ = _component_split_assignments(samples, seed=18)

    assert first == replay
    assert first_components == replay_components
    assert first != changed


def test_component_split_fails_closed_without_independent_holdouts() -> None:
    samples = [
        _sample("r1", "m1", "p1"),
        _sample("r2", "m2", "p1"),
        _sample("r3", "m3", "p2"),
        _sample("r4", "m4", "p2"),
    ]

    with pytest.raises(StructuredDatasetCanaryError, match="three independent"):
        _component_split_assignments(samples, seed=7)
