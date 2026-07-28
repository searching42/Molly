from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import ParsedDocument, ParsedDocumentElement, ParsedTable, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "index_corpus"
RUN_ID = "corpus-index-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_parsed_document(path: Path) -> Path:
    parsed = ParsedDocument(
        paper_id="oled-fixture-paper-1",
        source_path="synthetic-oled-parsed-document",
        parser_backend="synthetic_fixture",
        metadata={"title": "Synthetic OLED ParsedDocument fixture"},
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="paragraph_0001",
                page=1,
                type="paragraph",
                text=(
                    "OLED TADF emitters with high PLQY require balanced HOMO and LUMO "
                    "levels for red-shifted emission."
                ),
                source_hash="sha256:synthetic-paragraph",
            )
        ],
        tables=[
            ParsedTable(
                table_id="table_0001",
                caption="Synthetic OLED emitter measurements",
                headers=["Compound", "SMILES", "PLQY (%)", "HOMO", "LUMO"],
                rows=[
                    {
                        "Compound": "Fixture-A",
                        "SMILES": "CCN(CC)c1ccc(N)cc1",
                        "PLQY (%)": "86",
                        "HOMO": "-5.4",
                        "LUMO": "-2.6",
                    },
                    {
                        "Compound": "Fixture-B",
                        "SMILES": "O=C1N(c2ccccc2)C(=O)c2ccccc21",
                        "PLQY (%)": "72",
                        "HOMO": "-5.7",
                        "LUMO": "-3.0",
                    },
                ],
                page=1,
                markdown=(
                    "| Compound | SMILES | PLQY (%) | HOMO | LUMO |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| Fixture-A | CCN(CC)c1ccc(N)cc1 | 86 | -5.4 | -2.6 |\n"
                    "| Fixture-B | O=C1N(c2ccccc2)C(=O)c2ccccc21 | 72 | -5.7 | -3.0 |"
                ),
            )
        ],
    )
    path.write_text(parsed.model_dump_json(), encoding="utf-8")
    return path


def _run_index_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    parsed_document_json: Path | None,
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["parsed_document"],
    )
    input_artifacts = {"parsed_document": str(parsed_document_json)} if parsed_document_json is not None else {}
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={TASK_ID: {"output_dir": str(output_dir)}},
        max_iterations=3,
    )


def test_generic_queue_executes_corpus_indexing_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "corpus-index-out"
    parsed_document_json = _write_parsed_document(tmp_path / "parsed_document.json")

    summary = _run_index_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
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
    assert "corpus_index" in registry
    assert "evidence_chunks" in registry
    assert "corpus_index_report" in registry

    corpus_index_json = output_dir / f"{RUN_ID}_corpus_index.json"
    chunks_jsonl = output_dir / f"{RUN_ID}_evidence_chunks.jsonl"
    index_report_json = output_dir / f"{RUN_ID}_index_report.json"
    assert corpus_index_json.exists()
    assert chunks_jsonl.exists()
    assert index_report_json.exists()
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["corpus_index"] == corpus_index_json
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["evidence_chunks"] == chunks_jsonl
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["corpus_index_report"] == index_report_json

    index_payload = json.loads(corpus_index_json.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in chunks_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(index_report_json.read_text(encoding="utf-8"))
    assert index_payload["chunk_count"] == 2
    assert report["chunk_count"] == 2
    assert {chunk["element_type"] for chunk in chunks} == {"paragraph", "table"}
    table_chunk = next(chunk for chunk in chunks if chunk["element_type"] == "table")
    assert "table" in table_chunk["retrieval_channels"]
    combined_text = "\n".join([json.dumps(index_payload), *(chunk["text"] for chunk in chunks)])
    for signal in ("OLED", "PLQY", "SMILES", "HOMO", "LUMO"):
        assert signal in combined_text

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_parsed_document_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "corpus-index-out"

    summary = _run_index_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=tmp_path / "missing_parsed_document.json",
        output_dir=output_dir,
    )

    final_job = summary["final_job"]
    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert final_job["status"] == "failed"
    assert final_job["error"] == {"reason": "run-plan execution failed"}

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error == {
        "code": "no_parsed_documents",
        "message": "parsed_document_json or corpus_manifest_json is required",
    }
    assert "corpus_index" not in registry
    assert "evidence_chunks" not in registry
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_or_subprocess_imports() -> None:
    import tests.test_generic_run_plan_corpus_index_acceptance as acceptance

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
