from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CorpusChunk, CorpusMultiIndex, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "retrieve_evidence"
RUN_ID = "retrieve-evidence-generic-queue-demo"
PROJECT_ID = "demo-project"
QUERY = "OLED PLQY HOMO LUMO SMILES"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _synthetic_chunks() -> list[CorpusChunk]:
    return [
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


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_retrieval_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    chunks = _synthetic_chunks()
    chunks_jsonl = _write_jsonl(tmp_path / "evidence_chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    corpus_index_json = tmp_path / "corpus_index.json"
    corpus_index_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "chunk_count": len(chunks),
                "channels": ["bm25", "table"],
                "chunks_jsonl": str(chunks_jsonl),
            }
        ),
        encoding="utf-8",
    )
    multi_index = CorpusMultiIndex(
        run_id=RUN_ID,
        chunk_count=len(chunks),
        chunks_jsonl=str(chunks_jsonl),
        indices={
            "text": {
                "oled": ["chunk_paragraph_0001", "chunk_table_0001"],
                "tadf": ["chunk_paragraph_0001"],
                "plqy": ["chunk_paragraph_0001", "chunk_table_0001"],
                "homo": ["chunk_paragraph_0001", "chunk_table_0001"],
                "lumo": ["chunk_paragraph_0001", "chunk_table_0001"],
                "smiles": ["chunk_table_0001"],
            },
            "property": {
                "plqy": ["chunk_table_0001"],
                "homo": ["chunk_table_0001"],
                "lumo": ["chunk_table_0001"],
            },
            "table": {
                "table_0001": ["chunk_table_0001"],
                "smiles": ["chunk_table_0001"],
                "plqy": ["chunk_table_0001"],
            },
            "chemical": {
                "ccn(cc)c1ccc(n)cc1": ["chunk_table_0001"],
                "o=c1n(c2ccccc2)c(=o)c2ccccc21": ["chunk_table_0001"],
            },
        },
        channel_counts={"text": 6, "property": 3, "table": 3, "chemical": 2},
        created_at="2026-01-01T00:00:00Z",
        notes=["Synthetic deterministic multi-index fixture."],
    )
    multi_index_json = tmp_path / "multi_index.json"
    multi_index_json.write_text(multi_index.model_dump_json(), encoding="utf-8")
    return chunks_jsonl, corpus_index_json, multi_index_json


def _run_retrieve_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    corpus_index_json: Path | None,
    multi_index_json: Path | None,
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["corpus_index", "multi_index"],
    )
    input_artifacts: dict[str, str] = {}
    if corpus_index_json is not None:
        input_artifacts["corpus_index"] = str(corpus_index_json)
    if multi_index_json is not None:
        input_artifacts["multi_index"] = str(multi_index_json)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={TASK_ID: {"output_dir": str(output_dir), "query": QUERY, "topk": 2}},
        max_iterations=3,
    )


def test_generic_queue_executes_retrieve_evidence_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "retrieval-out"
    chunks_jsonl, corpus_index_json, multi_index_json = _write_retrieval_fixtures(tmp_path)

    summary = _run_retrieve_queue(
        queue_root=queue_root,
        storage=storage,
        corpus_index_json=corpus_index_json,
        multi_index_json=multi_index_json,
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
    task_options = final_job["task"]["task_options"][TASK_ID]
    assert "dense_index" not in final_job["task"]["input_artifacts"]
    assert "dense_index_json" not in task_options

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == [TASK_ID]
    assert "evidence_hits" in registry
    assert "retrieval_log" in registry

    evidence_hits_json = output_dir / f"{RUN_ID}_evidence_hits.json"
    retrieval_log_jsonl = output_dir / f"{RUN_ID}_retrieval_log.jsonl"
    assert evidence_hits_json.exists()
    assert retrieval_log_jsonl.exists()
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["evidence_hits"] == evidence_hits_json
    assert storage.run_dir(PROJECT_ID, RUN_ID) / registry["retrieval_log"] == retrieval_log_jsonl

    hits_payload = json.loads(evidence_hits_json.read_text(encoding="utf-8"))
    logs = [json.loads(line) for line in retrieval_log_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert hits_payload["query"] == QUERY
    assert len(hits_payload["hits"]) == 2
    table_hits = [hit for hit in hits_payload["hits"] if hit["element_type"] == "table"]
    assert table_hits
    assert table_hits[0]["retrieval_channel"] == "table"
    assert set(table_hits[0]["metadata"]["multi_index_channels"]) & {"property", "table"}
    assert table_hits[0]["metadata"]["dense_score"] == 0.0
    assert logs == [
        {
            "run_id": RUN_ID,
            "query": QUERY,
            "channel": "bm25",
            "topk": 2,
            "hit_count": 2,
            "created_at": logs[0]["created_at"],
        }
    ]

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert chunks_jsonl.name in created_files
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_index_input_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "retrieval-out"
    _, _, multi_index_json = _write_retrieval_fixtures(tmp_path)

    summary = _run_retrieve_queue(
        queue_root=queue_root,
        storage=storage,
        corpus_index_json=tmp_path / "missing_corpus_index.json",
        multi_index_json=multi_index_json,
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
    assert state.error == {"code": "missing_index", "message": "chunks_jsonl or corpus_index_json is required"}
    assert "evidence_hits" not in registry
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_dense_or_subprocess_imports() -> None:
    import tests.test_generic_run_plan_retrieve_evidence_acceptance as acceptance

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
