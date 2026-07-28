from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "prepare_literature_corpus_sources"
RUN_ID = "source-manifest-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _run_source_manifest_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    output_dir: Path,
    task_options: dict | None = None,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
    )
    options = {
        TASK_ID: {
            "output_dir": str(output_dir),
            "search_queries": [
                "OLED thermally activated delayed fluorescence emitters PLQY",
                "organic light emitting diode red emitter HOMO LUMO",
            ],
            "dois": ["10.0000/example-doi"],
            "urls": ["https://example.org/local-placeholder"],
            "dataset_registries": [
                {
                    "registry": "local_registry",
                    "record_id": "oled-fixture-001",
                    "title": "Local OLED fixture registry entry",
                    "license": "test-fixture",
                }
            ],
        }
    }
    if task_options is not None:
        options = task_options
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        task_options=options,
        max_iterations=3,
    )


def test_generic_queue_executes_source_manifest_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "source-manifest-out"

    summary = _run_source_manifest_queue(queue_root=queue_root, storage=storage, output_dir=output_dir)

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
    assert "corpus_source_manifest" in registry
    assert "corpus_source_manifest_md" in registry

    manifest_json = output_dir / f"{RUN_ID}_corpus_source_manifest.json"
    manifest_md = output_dir / f"{RUN_ID}_corpus_source_manifest.md"
    assert manifest_json.exists()
    assert manifest_md.exists()
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["corpus_source_manifest"] == manifest_json
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["corpus_source_manifest_md"] == manifest_md

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["source_count"] == 5
    assert manifest["source_type_counts"] == {
        "dataset_registry": 1,
        "doi": 1,
        "search_query": 2,
        "url": 1,
    }
    assert {source["source_type"] for source in manifest["sources"]} == {
        "dataset_registry",
        "doi",
        "search_query",
        "url",
    }
    assert {source["status"] for source in manifest["sources"]} == {"pending_acquisition"}
    assert any("does not fetch network content" in note for note in manifest["notes"])
    assert "Network acquisition, PDF download, and license review remain explicit" in manifest_md.read_text(encoding="utf-8")

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parsed_document" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_source_inputs_fail_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "source-manifest-out"

    summary = _run_source_manifest_queue(
        queue_root=queue_root,
        storage=storage,
        output_dir=output_dir,
        task_options={TASK_ID: {}},
    )

    final_job = summary["final_job"]
    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert final_job["status"] == "failed"
    assert final_job["error"] == {"reason": "run-plan execution failed"}

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.status == RunStatus.FAILED
    assert state.error == {"code": "no_sources", "message": "at least one literature source is required"}
    assert storage.read_artifact_registry(PROJECT_ID, RUN_ID) == {}
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parsed_document" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_or_subprocess_imports() -> None:
    import tests.test_generic_run_plan_source_manifest_acceptance as acceptance

    source = inspect.getsource(acceptance)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_imports = ("requests", "urllib", "openai", "mineru", "pdfplumber", "subprocess")
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


def test_phase3_executor_has_no_network_pdf_or_subprocess_imports() -> None:
    import ai4s_agent.phase3_executor as phase3_executor

    source = inspect.getsource(phase3_executor)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_imports = ("requests", "urllib", "openai", "mineru", "pdfplumber", "subprocess")
    assert not any(any(token in module for token in forbidden_imports) for module in imported_modules)
