from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
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
    _screen_registry_candidates,
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
