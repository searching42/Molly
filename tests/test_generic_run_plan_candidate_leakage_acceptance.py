from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "check_public_dataset_leakage"
RUN_ID = "candidate-leakage-generic-queue-demo"
PROJECT_ID = "demo-project"
OVERLAP_SMILES = "CCN(CC)c1ccc(N)cc1"
NON_OVERLAP_SMILES = "O=C1N(c2ccccc2)C(=O)c2ccccc21"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_leakage_fixtures(tmp_path: Path, *, overlap: bool = True) -> dict[str, Path]:
    candidate_csv = _write_csv(
        tmp_path / "candidate_training_dataset.csv",
        [
            {"smiles": OVERLAP_SMILES, "plqy": 0.865, "homo": -5.405, "lumo": -2.605},
            {"smiles": NON_OVERLAP_SMILES, "plqy": 0.72, "homo": -5.7, "lumo": -3.0},
        ],
    )
    benchmark_a_rows = (
        [{"smiles": OVERLAP_SMILES, "target": 0.80}]
        if overlap
        else [{"smiles": "N#Cc1ccccc1", "target": 0.80}]
    )
    public_benchmark_a_csv = _write_csv(tmp_path / "public_benchmark_a.csv", benchmark_a_rows)
    public_benchmark_b_csv = _write_csv(tmp_path / "public_benchmark_b.csv", [{"smiles": "CCO", "target": 0.50}])
    return {
        "candidate_training_dataset": candidate_csv,
        "public_benchmark_a": public_benchmark_a_csv,
        "public_benchmark_b": public_benchmark_b_csv,
    }


def _run_leakage_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    candidate_csv: Path | None,
    public_csvs: list[Path],
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["candidate_training_dataset"],
    )
    input_artifacts: dict[str, str] = {}
    if candidate_csv is not None:
        input_artifacts["candidate_training_dataset"] = str(candidate_csv)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={
            TASK_ID: {
                "output_dir": str(output_dir),
                "public_dataset_csvs": [str(path) for path in public_csvs],
            }
        },
        max_iterations=3,
    )


def test_generic_queue_executes_candidate_leakage_check(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "leakage-out"
    fixtures = _write_leakage_fixtures(tmp_path, overlap=True)

    summary = _run_leakage_queue(
        queue_root=queue_root,
        storage=storage,
        candidate_csv=fixtures["candidate_training_dataset"],
        public_csvs=[fixtures["public_benchmark_a"], fixtures["public_benchmark_b"]],
        output_dir=output_dir,
    )

    final_job = summary["final_job"]
    final_lease = summary["final_lease"]
    assert summary["ok"] is True
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["completed", "idle"]
    assert final_job["task"]["task_id"] == "run_plan_execute"
    assert [task["task_id"] for task in final_job["task"]["run_plan"]["tasks"]] == [TASK_ID]
    assert final_job["status"] == "succeeded"
    assert final_lease["status"] == "completed"
    assert final_job["result"]["executed_tasks"] == [TASK_ID]

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == [TASK_ID]
    assert "benchmark_contamination_report" in registry

    report_json = output_dir / f"{RUN_ID}_benchmark_contamination_report.json"
    report_md = output_dir / f"{RUN_ID}_benchmark_contamination_report.md"
    assert report_json.exists()
    assert report_md.exists()
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["benchmark_contamination_report"] == report_json

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "overlap_detected"
    assert report["dataset_count"] == 2
    assert report["total_overlap_count"] == 1
    assert report["total_overlap_smiles"] == [OVERLAP_SMILES]
    assert report["datasets"][0]["dataset_id"] == "public_benchmark_a"
    assert report["datasets"][0]["overlap_count"] == 1
    assert report["datasets"][0]["overlap_smiles"] == [OVERLAP_SMILES]
    assert report["datasets"][1]["dataset_id"] == "public_benchmark_b"
    assert report["datasets"][1]["overlap_count"] == 0
    assert any("contamination" in note or "leakage" in note for note in report["notes"])

    markdown = report_md.read_text(encoding="utf-8")
    assert "public_benchmark_a" in markdown
    assert "public_benchmark_b" in markdown
    assert OVERLAP_SMILES in markdown
    assert "Unique overlapping SMILES: 1" in markdown
    assert "contamination" in markdown and "leakage" in markdown

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("promotion" in name for name in created_files)
    assert not any("publication" in name for name in created_files)
    assert not any("release" in name for name in created_files)
    assert not any("global_append" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_clear_dataset_path_succeeds_with_clear_status(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "leakage-out"
    fixtures = _write_leakage_fixtures(tmp_path, overlap=False)

    summary = _run_leakage_queue(
        queue_root=queue_root,
        storage=storage,
        candidate_csv=fixtures["candidate_training_dataset"],
        public_csvs=[fixtures["public_benchmark_a"], fixtures["public_benchmark_b"]],
        output_dir=output_dir,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    report = json.loads((output_dir / f"{RUN_ID}_benchmark_contamination_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "clear"
    assert report["total_overlap_count"] == 0
    assert report["total_overlap_smiles"] == []
    assert all(dataset["overlap_count"] == 0 for dataset in report["datasets"])

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_public_dataset_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "leakage-out"
    fixtures = _write_leakage_fixtures(tmp_path, overlap=True)

    summary = _run_leakage_queue(
        queue_root=queue_root,
        storage=storage,
        candidate_csv=fixtures["candidate_training_dataset"],
        public_csvs=[fixtures["public_benchmark_a"], tmp_path / "missing_public_benchmark.csv"],
        output_dir=output_dir,
    )

    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert summary["final_job"]["status"] == "failed"
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "public_dataset_missing"
    assert "public benchmark" in state.error["message"]
    assert "benchmark_contamination_report" not in registry
    assert not output_dir.exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_candidate_dataset_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "leakage-out"
    fixtures = _write_leakage_fixtures(tmp_path, overlap=True)

    summary = _run_leakage_queue(
        queue_root=queue_root,
        storage=storage,
        candidate_csv=tmp_path / "missing_candidate_training_dataset.csv",
        public_csvs=[fixtures["public_benchmark_a"], fixtures["public_benchmark_b"]],
        output_dir=output_dir,
    )

    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert summary["final_job"]["status"] == "failed"
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "training_dataset_missing"
    assert "training dataset" in state.error["message"]
    assert "benchmark_contamination_report" not in registry
    assert not output_dir.exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_candidate_leakage_acceptance as acceptance

    source = inspect.getsource(acceptance)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_imports = (
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
        "sentence_transformers",
    )
    forbidden_call_names = {"urlopen", "Popen", "run", "call", "check_call", "check_output"}
    assert not any(any(token in module for token in forbidden_imports) for module in imported_modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_call_names
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_call_names
                if isinstance(node.func.value, ast.Name):
                    assert node.func.value.id != "requests"
    for forbidden_text in (
        "confirm_" + "extracted_dataset",
        "train_" + "model",
        "predict_" + "candidates",
        "resume_" + "after_gate",
    ):
        assert forbidden_text not in source
