from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CorpusChunk, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "build_dense_index"
RUN_ID = "dense-index-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_evidence_chunks(path: Path) -> Path:
    chunks = [
        CorpusChunk(
            chunk_id="chunk_paragraph_0001",
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            element_id="paragraph_0001",
            element_type="paragraph",
            text=(
                "OLED TADF emitters with high PLQY require balanced HOMO and LUMO "
                "levels for red-shifted emission."
            ),
            retrieval_channels=["bm25"],
            citation_context="Synthetic OLED paragraph fixture.",
            metadata={"section": "summary"},
        ),
        CorpusChunk(
            chunk_id="chunk_table_0001",
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            element_id="table_0001",
            element_type="table",
            text=(
                "Compound Fixture-A SMILES CCN(CC)c1ccc(N)cc1 PLQY 86 HOMO -5.4 LUMO -2.6\n"
                "Compound Fixture-B SMILES O=C1N(c2ccccc2)C(=O)c2ccccc21 PLQY 72 HOMO -5.7 LUMO -3.0"
            ),
            markdown=(
                "| Compound | SMILES | PLQY (%) | HOMO | LUMO |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| Fixture-A | CCN(CC)c1ccc(N)cc1 | 86 | -5.4 | -2.6 |\n"
                "| Fixture-B | O=C1N(c2ccccc2)C(=O)c2ccccc21 | 72 | -5.7 | -3.0 |"
            ),
            table_id="table_0001",
            retrieval_channels=["bm25", "table"],
            citation_context="Synthetic OLED table fixture.",
            metadata={
                "caption": "Synthetic OLED emitter measurements",
                "headers": ["Compound", "SMILES", "PLQY (%)", "HOMO", "LUMO"],
                "rows": [
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
            },
        ),
    ]
    path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks) + "\n", encoding="utf-8")
    return path


def _run_dense_index_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    chunks_jsonl: Path | None,
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["evidence_chunks"],
    )
    input_artifacts = {"evidence_chunks": str(chunks_jsonl)} if chunks_jsonl is not None else {}
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={
            TASK_ID: {
                "output_dir": str(output_dir),
                "dimension": 16,
                "embedding_backend": "deterministic_hash_embedding",
            }
        },
        max_iterations=3,
    )


def test_generic_queue_executes_deterministic_dense_index_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "dense-index-out"
    chunks_jsonl = _write_evidence_chunks(tmp_path / "evidence_chunks.jsonl")

    summary = _run_dense_index_queue(
        queue_root=queue_root,
        storage=storage,
        chunks_jsonl=chunks_jsonl,
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
    assert final_job["task"]["task_options"][TASK_ID]["embedding_backend"] == "deterministic_hash_embedding"

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == [TASK_ID]
    assert "dense_index" in registry
    assert "dense_index_summary" in registry

    dense_index_json = output_dir / f"{RUN_ID}_dense_index.json"
    summary_md = output_dir / f"{RUN_ID}_dense_index_summary.md"
    assert dense_index_json.exists()
    assert summary_md.exists()
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["dense_index"] == dense_index_json
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["dense_index_summary"] == summary_md

    dense_index = json.loads(dense_index_json.read_text(encoding="utf-8"))
    assert dense_index["embedding_backend"] == "deterministic_hash_embedding"
    assert dense_index["embedding_model"] == ""
    assert dense_index["dimension"] == 16
    assert dense_index["chunk_count"] == 2
    assert set(dense_index["vectors"]) == {"chunk_paragraph_0001", "chunk_table_0001"}
    for vector in dense_index["vectors"].values():
        assert len(vector) == 16
        assert any(value != 0 for value in vector)
    summary_text = summary_md.read_text(encoding="utf-8").lower()
    assert "deterministic hash" in summary_text
    assert "offline fallback" in summary_text
    assert "sentence_transformers" not in dense_index["embedding_backend"]

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_chunks_input_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "dense-index-out"

    summary = _run_dense_index_queue(
        queue_root=queue_root,
        storage=storage,
        chunks_jsonl=tmp_path / "missing_evidence_chunks.jsonl",
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
    assert state.error
    assert state.error["code"] == "invalid_dense_index_inputs"
    assert "chunks_jsonl" in state.error["message"]
    assert "dense_index" not in registry
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_empty_chunks_input_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "dense-index-out"
    chunks_jsonl = tmp_path / "evidence_chunks.jsonl"
    chunks_jsonl.write_text("", encoding="utf-8")

    summary = _run_dense_index_queue(
        queue_root=queue_root,
        storage=storage,
        chunks_jsonl=chunks_jsonl,
        output_dir=output_dir,
    )

    final_job = summary["final_job"]
    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert final_job["status"] == "failed"

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "invalid_dense_index_inputs"
    assert "did not contain any chunks" in state.error["message"]
    assert "dense_index" not in registry


def test_acceptance_test_has_no_network_pdf_dense_or_subprocess_imports() -> None:
    import tests.test_generic_run_plan_dense_index_acceptance as acceptance

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


def test_phase3_executor_has_no_network_pdf_dense_or_subprocess_imports() -> None:
    import ai4s_agent.phase3_executor as phase3_executor

    source = inspect.getsource(phase3_executor)
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
    assert not any(any(token in module for token in forbidden_imports) for module in imported_modules)
