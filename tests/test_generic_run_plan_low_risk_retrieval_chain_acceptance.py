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


RUN_ID = "low-risk-retrieval-chain-generic-queue-demo"
PROJECT_ID = "demo-project"
CHAIN_TASKS = ["index_corpus", "build_multi_index", "build_dense_index", "retrieve_evidence"]
QUERY = "OLED PLQY HOMO LUMO SMILES"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_parsed_document(path: Path) -> Path:
    parsed = ParsedDocument(
        paper_id="oled-chain-fixture-paper-1",
        source_path="synthetic-oled-chain-parsed-document",
        parser_backend="synthetic_fixture",
        metadata={"title": "Synthetic OLED retrieval chain fixture"},
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="paragraph_0001",
                page=1,
                type="paragraph",
                text=(
                    "OLED TADF emitters with high PLQY, balanced HOMO and LUMO, "
                    "and inspectable SMILES strings are useful for red-shifted emission review."
                ),
                source_hash="sha256:synthetic-chain-paragraph",
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


def _run_chain_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    parsed_document_json: Path,
    output_root: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=CHAIN_TASKS,
        available_artifacts=["parsed_document"],
    )
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts={"parsed_document": str(parsed_document_json)},
        task_options={
            "index_corpus": {
                "output_dir": str(output_root / "01_index"),
            },
            "build_multi_index": {
                "output_dir": str(output_root / "02_multi_index"),
            },
            "build_dense_index": {
                "output_dir": str(output_root / "03_dense_index"),
                "dimension": 16,
                "embedding_backend": "deterministic_hash_embedding",
            },
            "retrieve_evidence": {
                "output_dir": str(output_root / "04_retrieve"),
                "query": QUERY,
                "topk": 2,
            },
        },
        max_iterations=3,
    )


def test_generic_queue_executes_low_risk_retrieval_chain(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "retrieval-chain-out"
    parsed_document_json = _write_parsed_document(tmp_path / "parsed_document.json")

    summary = _run_chain_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
        output_root=output_root,
    )

    final_job = summary["final_job"]
    final_lease = summary["final_lease"]
    assert summary["ok"] is True
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["completed", "idle"]
    assert final_job["task"]["task_id"] == "run_plan_execute"
    assert [task["task_id"] for task in final_job["task"]["run_plan"]["tasks"]] == CHAIN_TASKS
    assert final_job["status"] == "succeeded"
    assert final_lease["status"] == "completed"
    assert final_job["result"]["executed_tasks"] == CHAIN_TASKS
    assert final_job["task"]["task_options"]["build_dense_index"]["embedding_backend"] == "deterministic_hash_embedding"

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == "retrieve_evidence"
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == CHAIN_TASKS
    for artifact_id in (
        "corpus_index",
        "evidence_chunks",
        "corpus_index_report",
        "multi_index",
        "multi_index_summary",
        "dense_index",
        "dense_index_summary",
        "evidence_hits",
        "retrieval_log",
    ):
        assert artifact_id in registry

    index_dir = output_root / "01_index"
    multi_dir = output_root / "02_multi_index"
    dense_dir = output_root / "03_dense_index"
    retrieve_dir = output_root / "04_retrieve"
    expected_files = {
        "corpus_index": index_dir / f"{RUN_ID}_corpus_index.json",
        "evidence_chunks": index_dir / f"{RUN_ID}_evidence_chunks.jsonl",
        "corpus_index_report": index_dir / f"{RUN_ID}_index_report.json",
        "multi_index": multi_dir / f"{RUN_ID}_multi_index.json",
        "multi_index_summary": multi_dir / f"{RUN_ID}_multi_index_summary.md",
        "dense_index": dense_dir / f"{RUN_ID}_dense_index.json",
        "dense_index_summary": dense_dir / f"{RUN_ID}_dense_index_summary.md",
        "evidence_hits": retrieve_dir / f"{RUN_ID}_evidence_hits.json",
        "retrieval_log": retrieve_dir / f"{RUN_ID}_retrieval_log.jsonl",
    }
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    for artifact_id, path in expected_files.items():
        assert path.exists()
        assert run_dir / registry[artifact_id] == path

    chunks = [
        json.loads(line)
        for line in expected_files["evidence_chunks"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {chunk["element_type"] for chunk in chunks} == {"paragraph", "table"}
    table_chunk = next(chunk for chunk in chunks if chunk["element_type"] == "table")
    assert "table" in table_chunk["retrieval_channels"]

    multi_index = json.loads(expected_files["multi_index"].read_text(encoding="utf-8"))
    assert set(multi_index["channel_counts"]) == {"text", "property", "table", "chemical"}
    assert "oled" in multi_index["indices"]["text"]
    assert "plqy" in multi_index["indices"]["property"]
    assert "table_0001" in multi_index["indices"]["table"]
    assert multi_index["channel_counts"]["chemical"] == 2

    dense_index = json.loads(expected_files["dense_index"].read_text(encoding="utf-8"))
    assert dense_index["embedding_backend"] == "deterministic_hash_embedding"
    assert dense_index["embedding_model"] == ""
    assert dense_index["dimension"] == 16
    assert dense_index["chunk_count"] == 2
    for vector in dense_index["vectors"].values():
        assert len(vector) == 16
        assert any(value != 0 for value in vector)

    hits_payload = json.loads(expected_files["evidence_hits"].read_text(encoding="utf-8"))
    logs = [
        json.loads(line)
        for line in expected_files["retrieval_log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    table_hits = [hit for hit in hits_payload["hits"] if hit["element_type"] == "table"]
    assert table_hits
    assert table_hits[0]["retrieval_channel"] == "table"
    assert table_hits[0]["metadata"]["multi_index_channels"]
    assert logs[0]["query"] == QUERY
    assert logs[0]["hit_count"] == 2
    assert logs[0]["channel"] == "bm25+dense"

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_parsed_document_fails_at_index_stage(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "retrieval-chain-out"

    summary = _run_chain_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=tmp_path / "missing_parsed_document.json",
        output_root=output_root,
    )

    final_job = summary["final_job"]
    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert final_job["status"] == "failed"
    assert final_job["error"] == {"reason": "run-plan execution failed"}
    assert final_job.get("result", {}).get("executed_tasks", []) == []

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == "index_corpus"
    assert state.status == RunStatus.FAILED
    assert state.error == {
        "code": "no_parsed_documents",
        "message": "parsed_document_json or corpus_manifest_json is required",
    }
    assert "multi_index" not in registry
    assert "dense_index" not in registry
    assert "evidence_hits" not in registry
    assert not (output_root / "02_multi_index").exists()
    assert not (output_root / "03_dense_index").exists()
    assert not (output_root / "04_retrieve").exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_dense_or_subprocess_imports() -> None:
    import tests.test_generic_run_plan_low_risk_retrieval_chain_acceptance as acceptance

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
