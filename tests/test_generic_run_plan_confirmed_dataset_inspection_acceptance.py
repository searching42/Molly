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


TASK_ID = "inspect_dataset"
RUN_ID = "confirmed-dataset-inspection-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_dataset_csv(path: Path, *, rows: int = 36) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "plqy", "homo", "lumo"])
        writer.writeheader()
        for idx in range(rows):
            writer.writerow(
                {
                    "smiles": f"CC{idx}N",
                    "plqy": f"{0.50 + idx * 0.01:.3f}",
                    "homo": f"{-5.20 - idx * 0.01:.3f}",
                    "lumo": f"{-2.40 - idx * 0.01:.3f}",
                }
            )
    return path


def _run_inspection_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    input_artifacts: dict[str, str],
    run_id: str = RUN_ID,
) -> dict:
    run_plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=list(input_artifacts),
    )
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        max_iterations=3,
    )


def test_generic_queue_inspects_confirmed_dataset(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    confirmed_csv = _write_dataset_csv(tmp_path / "confirmed_training_dataset.csv")

    summary = _run_inspection_queue(
        queue_root=queue_root,
        storage=storage,
        input_artifacts={"confirmed_training_dataset": str(confirmed_csv)},
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
    assert "dataset_profile" in registry
    assert "property_catalog" in registry

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    profile_path = run_dir / registry["dataset_profile"]
    catalog_path = run_dir / registry["property_catalog"]
    assert profile_path == catalog_path
    assert profile_path.exists()
    result = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = result["dataset_profile"]
    assert profile["input_csv"] == str(confirmed_csv)
    assert profile["row_count"] == 36
    assert profile["column_count"] == 4
    assert profile["headers"] == ["smiles", "plqy", "homo", "lumo"]
    assert profile["smiles_col"] == "smiles"
    assert profile["duplicate_smiles_rows"] == 0
    assert profile["smiles_missing_rows"] == 0
    property_ids = {item["property_id"] for item in result["property_candidates"]}
    assert {"plqy", "homo", "lumo"}.issubset(property_ids)

    _assert_no_downstream_outputs(tmp_path)


def test_existing_uploaded_dataset_path_still_works(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    dataset_csv = _write_dataset_csv(tmp_path / "uploaded_dataset.csv")
    run_id = "uploaded-dataset-inspection-generic-queue-demo"

    summary = _run_inspection_queue(
        queue_root=queue_root,
        storage=storage,
        input_artifacts={"uploaded_dataset": str(dataset_csv)},
        run_id=run_id,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    registry = storage.read_artifact_registry(PROJECT_ID, run_id)
    assert "dataset_profile" in registry
    assert "property_catalog" in registry


def test_missing_confirmed_dataset_path_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    missing_csv = tmp_path / "missing_confirmed_training_dataset.csv"

    summary = _run_inspection_queue(
        queue_root=queue_root,
        storage=storage,
        input_artifacts={"confirmed_training_dataset": str(missing_csv)},
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
    assert state.error["code"] == "input_not_found"
    assert "input_csv not found" in state.error["message"]
    assert "dataset_profile" not in registry
    assert "property_catalog" not in registry
    _assert_no_downstream_outputs(tmp_path)


def _assert_no_downstream_outputs(tmp_path: Path) -> None:
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("cleaned_train_dataset" in name for name in created_files)
    assert not any("trainability_report" in name for name in created_files)
    assert not any("baseline_report" in name for name in created_files)
    assert not any("model_metadata" in name for name in created_files)
    assert not any("candidate_predictions" in name for name in created_files)
    assert not any("ranked_candidates" in name for name in created_files)
    assert not any("report_markdown" in name or "report_html" in name for name in created_files)
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any(("public" + "ation") in name for name in created_files)
    assert not any(("rel" + "ease") in name for name in created_files)
    assert not any(("global_" + "append") in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_confirmed_dataset_inspection_acceptance as acceptance

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
        "clean_" + "dataset",
        "check_" + "trainability",
        "run_" + "baseline",
        "train_" + "model",
        "generate_" + "candidates",
        "predict_" + "candidates",
        "filter_" + "rank",
        "render_" + "report",
        "public" + "ation",
        "rel" + "ease",
        "global_" + "append",
    ):
        assert forbidden_text not in source
