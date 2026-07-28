from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CorpusChunk, EvidenceHit, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


RUN_ID = "candidate-dataset-chain-generic-queue-demo"
PROJECT_ID = "demo-project"
EXTRACT_TASK_ID = "extract_records"
NORMALIZE_TASK_ID = "normalize_extracted_units"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _table_chunk() -> CorpusChunk:
    return CorpusChunk(
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
        citation_context="synthetic-oled-paper p.1 table_0001",
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
    )


def _table_hit() -> EvidenceHit:
    return EvidenceHit(
        source_id="synthetic-oled-paper",
        page=1,
        element_id="table_0001",
        element_type="table",
        retrieval_channel="table",
        score=2.0,
        text_or_table_ref="chunk_table_0001",
        citation_context="synthetic-oled-paper p.1 table_0001",
        metadata={"chunk_id": "chunk_table_0001", "table_id": "table_0001"},
    )


def _non_table_hit() -> EvidenceHit:
    return EvidenceHit(
        source_id="synthetic-oled-paper",
        page=1,
        element_id="paragraph_0001",
        element_type="paragraph",
        retrieval_channel="bm25",
        score=1.0,
        text_or_table_ref="chunk_paragraph_0001",
        citation_context="synthetic-oled-paper p.1 paragraph_0001",
        metadata={"chunk_id": "chunk_paragraph_0001"},
    )


def _write_evidence_fixtures(tmp_path: Path, *, hits: list[EvidenceHit] | None = None) -> tuple[Path, Path]:
    chunks_jsonl = _write_jsonl(tmp_path / "evidence_chunks.jsonl", [_table_chunk().model_dump(mode="json")])
    evidence_hits_json = tmp_path / "evidence_hits.json"
    evidence_hits_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "query": "OLED PLQY HOMO LUMO SMILES",
                "hits": [hit.model_dump(mode="json") for hit in (hits if hits is not None else [_table_hit()])],
            }
        ),
        encoding="utf-8",
    )
    return evidence_hits_json, chunks_jsonl


def _run_candidate_chain_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    evidence_hits_json: Path | None,
    chunks_jsonl: Path,
    output_root: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[EXTRACT_TASK_ID, NORMALIZE_TASK_ID],
        available_artifacts=["evidence_hits", "evidence_chunks"],
    )
    input_artifacts = {"evidence_chunks": str(chunks_jsonl)}
    if evidence_hits_json is not None:
        input_artifacts["evidence_hits"] = str(evidence_hits_json)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={
            EXTRACT_TASK_ID: {
                "output_dir": str(output_root / "01_extract"),
                "confidence_threshold": 0.6,
            },
            NORMALIZE_TASK_ID: {
                "output_dir": str(output_root / "02_normalize"),
            },
        },
        max_iterations=3,
    )


def test_generic_queue_executes_candidate_dataset_chain(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "candidate-chain-out"
    evidence_hits_json, chunks_jsonl = _write_evidence_fixtures(tmp_path)

    summary = _run_candidate_chain_queue(
        queue_root=queue_root,
        storage=storage,
        evidence_hits_json=evidence_hits_json,
        chunks_jsonl=chunks_jsonl,
        output_root=output_root,
    )

    final_job = summary["final_job"]
    final_lease = summary["final_lease"]
    assert summary["ok"] is True
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["completed", "idle"]
    assert final_job["task"]["task_id"] == "run_plan_execute"
    assert [task["task_id"] for task in final_job["task"]["run_plan"]["tasks"]] == [
        EXTRACT_TASK_ID,
        NORMALIZE_TASK_ID,
    ]
    assert final_job["status"] == "succeeded"
    assert final_lease["status"] == "completed"
    assert final_job["result"]["executed_tasks"] == [EXTRACT_TASK_ID, NORMALIZE_TASK_ID]

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == NORMALIZE_TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == [EXTRACT_TASK_ID, NORMALIZE_TASK_ID]
    for artifact_id in (
        "extracted_records",
        "rejected_records",
        "extraction_confidence_report",
        "candidate_training_dataset",
        "normalized_extracted_records",
        "unit_normalization_report",
    ):
        assert artifact_id in registry

    extract_dir = output_root / "01_extract"
    normalize_dir = output_root / "02_normalize"
    extracted_records_jsonl = extract_dir / f"{RUN_ID}_extracted_records.jsonl"
    rejected_records_jsonl = extract_dir / f"{RUN_ID}_rejected_records.jsonl"
    extraction_report_json = extract_dir / f"{RUN_ID}_extraction_confidence_report.json"
    extraction_summary_md = extract_dir / f"{RUN_ID}_extraction_summary.md"
    candidate_csv = extract_dir / f"{RUN_ID}_candidate_training_dataset.csv"
    normalized_jsonl = normalize_dir / f"{RUN_ID}_normalized_extracted_records.jsonl"
    normalized_candidate_csv = normalize_dir / f"{RUN_ID}_normalized_candidate_training_dataset.csv"
    normalization_report_json = normalize_dir / f"{RUN_ID}_unit_normalization_report.json"
    normalization_report_md = normalize_dir / f"{RUN_ID}_unit_normalization_report.md"
    for path in (
        extracted_records_jsonl,
        rejected_records_jsonl,
        extraction_report_json,
        extraction_summary_md,
        candidate_csv,
        normalized_jsonl,
        normalized_candidate_csv,
        normalization_report_json,
        normalization_report_md,
    ):
        assert path.exists()

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["extracted_records"] == extracted_records_jsonl
    assert run_dir / registry["candidate_training_dataset"] == normalized_candidate_csv
    assert run_dir / registry["normalized_extracted_records"] == normalized_jsonl
    assert run_dir / registry["unit_normalization_report"] == normalization_report_json

    extracted_records = [
        json.loads(line)
        for line in extracted_records_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized_records = [
        json.loads(line)
        for line in normalized_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(extracted_records) == 2
    assert len(normalized_records) == 2
    assert [record["status"] for record in extracted_records] == ["candidate", "candidate"]
    assert [record["status"] for record in normalized_records] == ["candidate", "candidate"]
    assert [record["properties"]["plqy"] for record in extracted_records] == [86.0, 72.0]
    assert [record["properties"]["plqy"] for record in normalized_records] == [0.86, 0.72]
    assert [record["properties"]["homo"] for record in normalized_records] == [-5.4, -5.7]
    assert [record["properties"]["lumo"] for record in normalized_records] == [-2.6, -3.0]

    normalized_rows = list(csv.DictReader(normalized_candidate_csv.open(encoding="utf-8")))
    assert len(normalized_rows) == 2
    assert [float(row["plqy"]) for row in normalized_rows] == [0.86, 0.72]
    assert [float(row["homo"]) for row in normalized_rows] == [-5.4, -5.7]
    assert [float(row["lumo"]) for row in normalized_rows] == [-2.6, -3.0]

    normalization_report = json.loads(normalization_report_json.read_text(encoding="utf-8"))
    percent_conversions = [
        item for item in normalization_report["conversions"] if item["rule"] == "percent_to_fraction"
    ]
    assert len(percent_conversions) == 2
    assert {item["source_unit"] for item in percent_conversions} == {"%"}
    assert {item["canonical_unit"] for item in percent_conversions} == {"fraction"}
    assert {item["canonical_value"] for item in percent_conversions} == {0.86, 0.72}

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


def test_missing_evidence_hits_fails_at_extraction_stage(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "candidate-chain-out"
    _, chunks_jsonl = _write_evidence_fixtures(tmp_path)

    summary = _run_candidate_chain_queue(
        queue_root=queue_root,
        storage=storage,
        evidence_hits_json=tmp_path / "missing_evidence_hits.json",
        chunks_jsonl=chunks_jsonl,
        output_root=output_root,
    )

    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert summary["final_job"]["status"] == "failed"
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == EXTRACT_TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] in {"missing_evidence_hits", "invalid_extraction_inputs"}
    assert "evidence_hits" in state.error["message"]
    assert "normalized_extracted_records" not in registry
    assert "unit_normalization_report" not in registry
    assert not (output_root / "02_normalize").exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_non_table_evidence_safely_produces_zero_candidate_records(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "candidate-chain-out"
    evidence_hits_json, chunks_jsonl = _write_evidence_fixtures(tmp_path, hits=[_non_table_hit()])

    summary = _run_candidate_chain_queue(
        queue_root=queue_root,
        storage=storage,
        evidence_hits_json=evidence_hits_json,
        chunks_jsonl=chunks_jsonl,
        output_root=output_root,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    assert summary["final_job"]["result"]["executed_tasks"] == [EXTRACT_TASK_ID, NORMALIZE_TASK_ID]
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "confirmed_dataset" not in registry

    extracted_records_jsonl = output_root / "01_extract" / f"{RUN_ID}_extracted_records.jsonl"
    normalized_jsonl = output_root / "02_normalize" / f"{RUN_ID}_normalized_extracted_records.jsonl"
    normalized_candidate_csv = output_root / "02_normalize" / f"{RUN_ID}_normalized_candidate_training_dataset.csv"
    normalization_report_json = output_root / "02_normalize" / f"{RUN_ID}_unit_normalization_report.json"
    assert extracted_records_jsonl.read_text(encoding="utf-8") == ""
    assert normalized_jsonl.read_text(encoding="utf-8") == ""
    assert list(csv.DictReader(normalized_candidate_csv.open(encoding="utf-8"))) == []
    normalization_report = json.loads(normalization_report_json.read_text(encoding="utf-8"))
    assert normalization_report["input_record_count"] == 0
    assert normalization_report["normalized_record_count"] == 0
    assert normalization_report["conversion_count"] == 0

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_candidate_dataset_chain_acceptance as acceptance

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
