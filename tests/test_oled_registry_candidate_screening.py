from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

import pytest

from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistrySnapshot,
    _rdkit_chemistry_observation,
    _rdkit_runtime_versions,
    oled_material_registry_entry_digest,
    oled_material_registry_snapshot_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityStructureEncodingKind,
)
from ai4s_agent.oled_real_phase1_execution import (
    run_oled_real_phase1_execution_from_files,
)
from ai4s_agent.oled_registry_candidate_screening import (
    _load_screening_inputs,
    _parse_constraints,
    _rank_candidates,
    _screen_registry_candidates,
    main,
    run_oled_registry_candidate_screening_from_files,
)
from tests.test_oled_real_phase1_execution import _snapshot_path


@dataclass(frozen=True)
class _ScreeningInputs:
    execution_dir: Path
    dataset_snapshot: Path
    registry_snapshot: Path


def _screening_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _ScreeningInputs:
    dataset_path = _snapshot_path(tmp_path, monkeypatch)
    execution = run_oled_real_phase1_execution_from_files(
        dataset_snapshot_json=dataset_path,
        output_root=tmp_path / "phase1-executions",
        property_ids=["delta_e_st_ev", "s1_ev"],
        generated_at="2026-07-20T08:10:00+08:00",
    )
    registry_path = tmp_path / "registry-snapshot.json"
    registry_path.write_text(
        json.dumps(_registry_snapshot().model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return _ScreeningInputs(
        execution_dir=execution.output_dir,
        dataset_snapshot=dataset_path,
        registry_snapshot=registry_path,
    )


def _registry_snapshot() -> OledMaterialRegistrySnapshot:
    entries = [_registry_entry(index) for index in range(4)]
    toolkit_version, inchi_backend_version = _rdkit_runtime_versions()
    snapshot = OledMaterialRegistrySnapshot.model_construct(
        registry_id="oled-registry:screening-test",
        registry_version="registry-version:screening-test",
        generated_at="2026-07-20T08:05:00+08:00",
        toolkit_version=toolkit_version,
        inchi_backend_version=inchi_backend_version,
        entry_count=len(entries),
        entries=entries,
        snapshot_digest="sha256:" + "0" * 64,
        read_only_snapshot=True,
    )
    snapshot = snapshot.model_copy(
        update={"snapshot_digest": oled_material_registry_snapshot_digest(snapshot)}
    )
    return OledMaterialRegistrySnapshot.model_validate(snapshot.model_dump(mode="json"))


def _registry_entry(index: int) -> OledMaterialRegistryEntry:
    smiles = "C" * (index + 1)
    chemistry = _rdkit_chemistry_observation(
        encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
        structure_text=smiles,
    )
    entry = OledMaterialRegistryEntry.model_construct(
        material_id=f"material:execution-{index:02d}",
        canonical_name=f"screening material {index}",
        aliases=[],
        canonical_isomeric_smiles=chemistry["canonical_isomeric_smiles"],
        standard_inchi=chemistry["standard_inchi"],
        inchikey=chemistry["inchikey"],
        entry_digest="sha256:" + f"{index + 30:064x}",
    )
    entry = entry.model_copy(
        update={"entry_digest": oled_material_registry_entry_digest(entry)}
    )
    return OledMaterialRegistryEntry.model_validate(entry.model_dump(mode="json"))


def test_loads_exact_inputs_and_rederives_training_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    prepared = _load_screening_inputs(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
    )

    assert prepared.property_ids == ("delta_e_st_ev", "s1_ev")
    assert prepared.training_material_ids == frozenset(
        {"material:execution-00", "material:execution-01"}
    )
    assert len(prepared.training_registry_digests) == 2
    assert len(prepared.training_smiles) == 2
    assert len(prepared.registry.entries) == 4


def test_excludes_train_materials_and_predicts_complete_nontrain_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    prepared = _load_screening_inputs(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
    )
    eligible, excluded, predictions = _screen_registry_candidates(prepared)

    assert [item["material_id"] for item in eligible] == [
        "material:execution-02",
        "material:execution-03",
    ]
    assert {
        item["material_id"]: item["reason_codes"] for item in excluded
    } == {
        "material:execution-00": [
            "training_material_id_overlap",
            "training_smiles_overlap",
        ],
        "material:execution-01": [
            "training_material_id_overlap",
            "training_smiles_overlap",
        ],
    }
    assert [item["material_id"] for item in predictions] == [
        "material:execution-02",
        "material:execution-03",
    ]
    assert all(
        set(item["predictions"]) == {"delta_e_st_ev", "s1_ev"}
        for item in predictions
    )


def test_digest_and_smiles_training_overlap_have_independent_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    prepared = _load_screening_inputs(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
    )
    target = prepared.registry.entries[-1]
    prepared = replace(
        prepared,
        training_material_ids=frozenset(),
        training_registry_digests=frozenset({target.entry_digest}),
        training_smiles=frozenset({target.canonical_isomeric_smiles}),
    )

    _, excluded, _ = _screen_registry_candidates(prepared)

    assert next(
        item for item in excluded if item["material_id"] == target.material_id
    )["reason_codes"] == [
        "training_registry_digest_overlap",
        "training_smiles_overlap",
    ]


def test_constraints_reject_duplicates_unknown_properties_and_empty_ranges() -> None:
    assert _parse_constraints(
        minimums=["s1_ev=2.8"],
        maximums=["delta_e_st_ev=0.2"],
        property_ids=("delta_e_st_ev", "s1_ev"),
    ) == {
        "delta_e_st_ev": {"max": 0.2},
        "s1_ev": {"min": 2.8},
    }
    with pytest.raises(ValueError, match="duplicate minimum"):
        _parse_constraints(
            minimums=["s1_ev=2.8", "s1_ev=2.9"],
            maximums=[],
            property_ids=("s1_ev",),
        )
    with pytest.raises(ValueError, match="unknown property"):
        _parse_constraints(
            minimums=["plqy=0.8"],
            maximums=[],
            property_ids=("s1_ev",),
        )
    with pytest.raises(ValueError, match="empty feasible range"):
        _parse_constraints(
            minimums=["s1_ev=3.0"],
            maximums=["s1_ev=2.0"],
            property_ids=("s1_ev",),
        )


def test_pareto_percentiles_and_material_tie_break_are_deterministic() -> None:
    predictions = [
        {"material_id": "material:b", "predictions": {"gap": 0.2, "s1": 3.2}},
        {"material_id": "material:c", "predictions": {"gap": 0.3, "s1": 2.8}},
        {"material_id": "material:a", "predictions": {"gap": 0.1, "s1": 3.0}},
    ]

    scored, shortlist = _rank_candidates(
        predictions,
        property_ids=("gap", "s1"),
        directions={"gap": "minimize", "s1": "maximize"},
        constraints={},
    )

    by_id = {item["material_id"]: item for item in scored}
    assert by_id["material:c"]["pareto_dominated"] is True
    assert by_id["material:a"]["aggregate_percentile"] == 0.75
    assert by_id["material:b"]["aggregate_percentile"] == 0.75
    assert [item["material_id"] for item in shortlist] == [
        "material:a",
        "material:b",
    ]
    assert [item["rank"] for item in shortlist] == [1, 2]


def test_constraint_failure_remains_in_predictions_but_not_shortlist() -> None:
    predictions = [
        {"material_id": "material:a", "predictions": {"gap": 0.1, "s1": 3.0}},
        {"material_id": "material:b", "predictions": {"gap": 0.2, "s1": 3.2}},
    ]
    constraints = _parse_constraints(
        minimums=[],
        maximums=["gap=0.15"],
        property_ids=("gap", "s1"),
    )

    scored, shortlist = _rank_candidates(
        predictions,
        property_ids=("gap", "s1"),
        directions={"gap": "minimize", "s1": "maximize"},
        constraints=constraints,
    )

    by_id = {item["material_id"]: item for item in scored}
    assert by_id["material:b"]["hard_constraints_passed"] is False
    assert by_id["material:b"]["decision_reason_codes"] == [
        "hard_constraint_failed:gap:max"
    ]
    assert [item["material_id"] for item in shortlist] == ["material:a"]


def test_file_runner_publishes_complete_versioned_screening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    result = run_oled_registry_candidate_screening_from_files(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
        output_root=tmp_path / "screenings",
        maximums=["delta_e_st_ev=1.0"],
        generated_at="2026-07-20T09:00:00+08:00",
    )

    assert result.eligible_candidate_count == 2
    assert result.excluded_candidate_count == 2
    assert result.prediction_count == 2
    assert result.shortlist_count >= 1
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "eligible_candidates.csv",
        "excluded_candidates.jsonl",
        "predictions.jsonl",
        "ranked_shortlist.csv",
        "report.md",
        "screening.json",
    ]
    receipt = json.loads(
        (result.output_dir / "screening.json").read_text(encoding="utf-8")
    )
    assert receipt["counts"] == {
        "eligible_candidate_count": 2,
        "excluded_candidate_count": 2,
        "prediction_count": 2,
        "registry_candidate_count": 4,
        "shortlist_count": result.shortlist_count,
    }
    assert receipt["claims"] == {
        "benchmark_validated": False,
        "experimental_validation_claimed": False,
        "independent_registry_candidate_pool": True,
        "model_registered": False,
        "production_ready": False,
        "registry_mutated": False,
        "training_identity_exclusion_applied": True,
    }
    with pytest.raises(ValueError, match="already exists"):
        run_oled_registry_candidate_screening_from_files(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
            output_root=tmp_path / "screenings",
            maximums=["delta_e_st_ev=1.0"],
        )


def test_cli_failure_is_stable_redacted_and_publishes_nothing(tmp_path: Path) -> None:
    private_execution = tmp_path / "private-execution"
    private_execution.mkdir()
    stream = StringIO()

    exit_code = main(
        [
            "--phase1-execution-dir",
            str(private_execution),
            "--dataset-snapshot",
            str(tmp_path / "private-dataset.json"),
            "--registry-snapshot",
            str(tmp_path / "private-registry.json"),
            "--output-root",
            str(tmp_path / "screenings"),
        ],
        stdout=stream,
    )

    assert exit_code == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "registry_candidate_screening_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert "private" not in stream.getvalue()
    assert not (tmp_path / "screenings").exists() or not any(
        (tmp_path / "screenings").iterdir()
    )


@pytest.mark.parametrize("target", ["execution", "model", "dataset", "registry"])
def test_exact_input_byte_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    path = {
        "execution": inputs.execution_dir / "execution.json",
        "model": inputs.execution_dir / "model__s1_ev.json",
        "dataset": inputs.dataset_snapshot,
        "registry": inputs.registry_snapshot,
    }[target]
    path.write_bytes(path.read_bytes() + b" \n")

    with pytest.raises(ValueError):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
