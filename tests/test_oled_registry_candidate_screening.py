from __future__ import annotations

import hashlib
import json
import shutil
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
from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
    oled_categorical_dataset_execution_artifact_digest,
    oled_categorical_dataset_view_row_digest,
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
from ai4s_agent import oled_registry_candidate_screening as screening_runner
from ai4s_agent.trainability import generate_baseline_features
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
    _rewrite_fixture_with_exact_128_bit_features(dataset_path)
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


def _rewrite_fixture_with_exact_128_bit_features(dataset_path: Path) -> None:
    snapshot = OledCategoricalDatasetExecutionArtifact.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    generated = generate_baseline_features(
        [row.canonical_isomeric_smiles for row in snapshot.rows],
        n_bits=128,
        radius=2,
    )
    rows = []
    for row, vector in zip(snapshot.rows, generated.matrix, strict=True):
        rewritten = row.model_copy(
            update={
                "feature_type": generated.feature_type,
                "features": {
                    f"ecfp_{index:03d}": value
                    for index, value in enumerate(vector)
                },
                "row_digest": "sha256:" + "0" * 64,
            }
        )
        rewritten = rewritten.model_copy(
            update={
                "row_digest": oled_categorical_dataset_view_row_digest(rewritten)
            }
        )
        rows.append(rewritten)
    rewritten_snapshot = snapshot.model_copy(
        update={
            "rows": rows,
            "execution_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    rewritten_snapshot = rewritten_snapshot.model_copy(
        update={
            "execution_artifact_digest": (
                oled_categorical_dataset_execution_artifact_digest(
                    rewritten_snapshot
                )
            )
        }
    )
    rewritten_snapshot = OledCategoricalDatasetExecutionArtifact.model_validate(
        rewritten_snapshot.model_dump(mode="json")
    )
    dataset_path.write_text(
        json.dumps(rewritten_snapshot.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
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
    assert len(prepared.training_standard_inchi) == 2
    assert len(prepared.training_inchikey) == 2
    assert prepared.feature_generator_profile["n_bits"] == 128
    assert prepared.feature_generator_profile["radius"] == 2
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
            "training_inchikey_overlap",
            "training_material_id_overlap",
            "training_smiles_overlap",
            "training_standard_inchi_overlap",
        ],
        "material:execution-01": [
            "training_inchikey_overlap",
            "training_material_id_overlap",
            "training_smiles_overlap",
            "training_standard_inchi_overlap",
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


def test_inchi_training_overlap_has_independent_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    prepared = _load_screening_inputs(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
    )
    target = prepared.registry.entries[0]
    prepared = replace(
        prepared,
        training_material_ids=frozenset(),
        training_registry_digests=frozenset(),
        training_smiles=frozenset(),
        training_standard_inchi=frozenset({target.standard_inchi}),
        training_inchikey=frozenset({target.inchikey}),
    )

    _, excluded, _ = _screen_registry_candidates(prepared)

    assert next(
        item for item in excluded if item["material_id"] == target.material_id
    )["reason_codes"] == [
        "training_inchikey_overlap",
        "training_standard_inchi_overlap",
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
    assert receipt["config"]["feature_generator_profile"] == {
        "fallback_reason": "",
        "feature_type": "morgan_ecfp",
        "feature_version": "morgan_or_hashed_ecfp_128.v1",
        "generator": "rdkit.AllChem.GetMorganFingerprintAsBitVect.v1",
        "n_bits": 128,
        "radius": 2,
        "rdkit_version": _rdkit_runtime_versions()[0],
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


@pytest.mark.parametrize(
    "target",
    [
        "execution",
        "model",
        "predictions",
        "metrics",
        "ranking",
        "report",
        "dataset",
        "registry",
    ],
)
def test_exact_input_byte_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    path = {
        "execution": inputs.execution_dir / "execution.json",
        "model": inputs.execution_dir / "model__s1_ev.json",
        "predictions": inputs.execution_dir / "predictions.jsonl",
        "metrics": inputs.execution_dir / "metrics.json",
        "ranking": inputs.execution_dir / "ranked_candidates.csv",
        "report": inputs.execution_dir / "report.md",
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


def test_resigned_direction_change_requires_matching_execution_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    receipt_path = inputs.execution_dir / "execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["config"]["directions"]["s1_ev"] = "minimize"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution ID mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_fully_resigned_direction_change_fails_exact_execution_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    receipt_path = inputs.execution_dir / "execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["config"]["directions"]["s1_ev"] = "minimize"
    execution_id = "oled-real-phase1-execution:" + screening_runner._stable_hash(
        {
            "source_snapshot_digest": receipt["source"]["dataset_snapshot_digest"],
            "source_snapshot_sha256": receipt["source"]["dataset_snapshot_sha256"],
            "config": receipt["config"],
        }
    )
    receipt["execution_id"] = execution_id
    for model_path in sorted(inputs.execution_dir.glob("model__*.json")):
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["execution_id"] = execution_id
        model_path.write_bytes(screening_runner._json_bytes(model))
        receipt["artifacts"][model_path.name] = _sha256(model_path)
    receipt_path.write_bytes(screening_runner._json_bytes(receipt))
    resigned_dir = inputs.execution_dir.parent / execution_id
    inputs.execution_dir.rename(resigned_dir)

    with pytest.raises(ValueError, match="exact replay mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=resigned_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_execution_file_roster_mismatch_fails_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    (inputs.execution_dir / "unbound-marker.txt").write_text(
        "unexpected\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact replay mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_execution_directory_basename_must_equal_execution_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    renamed = inputs.execution_dir.parent / "not-the-execution-id"
    inputs.execution_dir.rename(renamed)

    with pytest.raises(ValueError, match="directory name mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=renamed,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_resigned_model_change_fails_deterministic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    model_path = inputs.execution_dir / "model__s1_ev.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["target_mean"] += 1.0
    model_path.write_text(
        json.dumps(model, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path = inputs.execution_dir / "execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][model_path.name] = _sha256(model_path)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact replay mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_fully_resigned_train_feature_substitution_fails_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    snapshot = OledCategoricalDatasetExecutionArtifact.model_validate_json(
        inputs.dataset_snapshot.read_text(encoding="utf-8")
    )
    split_by_row = {
        assignment.row_id: assignment.split
        for assignment in snapshot.split_assignments
    }
    forged_rows = []
    changed = False
    for row in snapshot.rows:
        if split_by_row[row.row_id] == "train" and not changed:
            features = dict(row.features)
            features["ecfp_000"] = 1.0 - features["ecfp_000"]
            forged_row = row.model_copy(
                update={"features": features, "row_digest": "sha256:" + "0" * 64}
            )
            forged_row = forged_row.model_copy(
                update={"row_digest": oled_categorical_dataset_view_row_digest(forged_row)}
            )
            forged_rows.append(forged_row)
            changed = True
        else:
            forged_rows.append(row)
    forged = snapshot.model_copy(
        update={
            "rows": forged_rows,
            "execution_artifact_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={
            "execution_artifact_digest": (
                oled_categorical_dataset_execution_artifact_digest(forged)
            )
        }
    )
    forged = OledCategoricalDatasetExecutionArtifact.model_validate(
        forged.model_dump(mode="json")
    )
    forged_path = tmp_path / "forged-dataset.json"
    forged_path.write_text(
        json.dumps(forged.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    execution = run_oled_real_phase1_execution_from_files(
        dataset_snapshot_json=forged_path,
        output_root=tmp_path / "forged-executions",
        property_ids=["delta_e_st_ev", "s1_ev"],
        generated_at="2026-07-20T08:10:00+08:00",
    )

    with pytest.raises(ValueError, match="training feature regeneration mismatch"):
        _load_screening_inputs(
            phase1_execution_dir=execution.output_dir,
            dataset_snapshot_json=forged_path,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_resigned_registry_duplicate_chemical_identity_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    registry = OledMaterialRegistrySnapshot.model_validate_json(
        inputs.registry_snapshot.read_text(encoding="utf-8")
    )
    duplicate = registry.entries[0].model_copy(
        update={
            "material_id": "material:execution-duplicate",
            "canonical_name": "duplicate chemical identity",
            "entry_digest": "sha256:" + "0" * 64,
        }
    )
    duplicate = duplicate.model_copy(
        update={"entry_digest": oled_material_registry_entry_digest(duplicate)}
    )
    duplicate = OledMaterialRegistryEntry.model_validate(
        duplicate.model_dump(mode="json")
    )
    forged_entries = sorted(
        [*registry.entries, duplicate], key=lambda entry: entry.material_id
    )
    forged = registry.model_copy(
        update={
            "entry_count": len(forged_entries),
            "entries": forged_entries,
            "snapshot_digest": "sha256:" + "0" * 64,
        },
        deep=True,
    )
    forged = forged.model_copy(
        update={"snapshot_digest": oled_material_registry_snapshot_digest(forged)}
    )
    forged = OledMaterialRegistrySnapshot.model_validate(
        forged.model_dump(mode="json")
    )
    inputs.registry_snapshot.write_text(
        json.dumps(forged.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Registry chemical identity is duplicated"):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def test_execution_directory_replacement_between_reads_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement-execution"
    moved = tmp_path / "moved-execution"
    shutil.copytree(inputs.execution_dir, replacement)
    original_read = screening_runner._read_bound_json
    replaced = False

    def replace_directory_after_read(*args: object, **kwargs: object):
        nonlocal replaced
        result = original_read(*args, **kwargs)
        if not replaced:
            inputs.execution_dir.rename(moved)
            replacement.rename(inputs.execution_dir)
            replaced = True
        return result

    monkeypatch.setattr(
        screening_runner,
        "_read_bound_json",
        replace_directory_after_read,
    )

    with pytest.raises(ValueError, match="execution directory changed"):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
