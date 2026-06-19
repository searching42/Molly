from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.snapshot_material import build_execution_snapshot
from ai4s_agent.storage import ProjectStorage


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_snapshot_material_splits_execution_payload_audit_metadata_and_resource_manifest(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    scorer = tmp_path / "scorer.py"
    input_csv.write_text("SMILES,plqy\nCCO,0.5\n", encoding="utf-8")
    scorer.write_text("def score(row): return 1\n", encoding="utf-8")

    snapshot = build_execution_snapshot(
        run_id="r-material",
        task_id="train_model",
        adapter="train_model_baseline_adapter",
        run_plan={"run_id": "r-material", "tasks": []},
        task_options={"epochs": 5},
        execution_payload={
            "run_id": "r-material",
            "input_csv": str(input_csv),
            "scorer_path": str(scorer),
            "output_dir": str(tmp_path / "generated"),
            "actor": "alice",
            "confirmed": True,
        },
        artifact_paths={"uploaded_dataset": str(input_csv)},
        run_dir=tmp_path,
        approved_gates=[GateName.TRAIN_CONFIG.value],
    )

    assert snapshot["schema_version"] == 2
    assert snapshot["snapshot_hash"]
    assert snapshot["payload"] == snapshot["execution_payload"]
    assert "actor" not in snapshot["execution_payload"]
    assert "confirmed" not in snapshot["execution_payload"]
    assert snapshot["audit_metadata"] == {"actor": "alice", "confirmed": True}

    resource_manifest = snapshot["resource_manifest"]
    assert "uploaded_dataset" in resource_manifest
    assert resource_manifest["uploaded_dataset"]["artifact_id"] == "uploaded_dataset"
    assert resource_manifest["uploaded_dataset"]["sha256"]
    scorer_entries = [entry for entry in resource_manifest.values() if entry["key"] == "scorer_path"]
    assert len(scorer_entries) == 1
    assert scorer_entries[0]["sha256"]
    assert not any(entry["key"] == "output_dir" for entry in resource_manifest.values())


def test_run_plan_snapshot_rejects_changed_auxiliary_resource_after_gate(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    scorer = tmp_path / "resources" / "scorer.py"
    dataset.parent.mkdir(parents=True)
    scorer.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    scorer.write_text("def score(row): return 1\n", encoding="utf-8")
    run_plan = expand_run_plan(
        run_id="r-aux-resource-change",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )
    executor = RunPlanExecutor(storage=storage)
    task_options = {"train_model": {"scorer_path": str(scorer)}}

    first = executor.execute(
        project_id="proj-open-003",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
        task_options=task_options,
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("proj-open-003", "r-aux-resource-change")
    assert state is not None
    snapshot = state.details["execution_snapshot"]
    assert snapshot["execution_payload"]["scorer_path"] == str(scorer)
    scorer_entries = [entry for entry in snapshot["resource_manifest"].values() if entry["key"] == "scorer_path"]
    assert len(scorer_entries) == 1
    original_sha = scorer_entries[0]["sha256"]

    scorer.write_text("def score(row): return 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="execution snapshot changed"):
        executor.resume_after_gate(
            project_id="proj-open-003",
            run_plan=run_plan,
            approved_gates=[GateName.TRAIN_CONFIG.value],
            actor="user",
        )

    current = json.loads(json.dumps(storage.read_stage_state("proj-open-003", "r-aux-resource-change").details))
    scorer_after = [entry for entry in current["execution_snapshot"]["resource_manifest"].values() if entry["key"] == "scorer_path"]
    assert scorer_after[0]["sha256"] == original_sha
