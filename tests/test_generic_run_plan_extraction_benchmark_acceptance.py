from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import EvidenceHit, ExtractedRecord, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "evaluate_extraction_benchmark"
RUN_ID = "extraction-benchmark-generic-queue-demo"
PROJECT_ID = "demo-project"
SMILES_A = "CCN(CC)c1ccc(N)cc1"
SMILES_B = "O=C1N(c2ccccc2)C(=O)c2ccccc21"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


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


def _hit() -> EvidenceHit:
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


def _record(
    *,
    record_id: str,
    smiles: str,
    plqy: float,
    homo: float,
    lumo: float,
    row_index: int,
) -> ExtractedRecord:
    return ExtractedRecord(
        record_id=record_id,
        smiles=smiles,
        properties={"plqy": plqy, "homo": homo, "lumo": lumo},
        source_id="synthetic-oled-paper",
        paper_id="synthetic-oled-paper",
        page=1,
        table_id="table_0001",
        row_index=row_index,
        evidence_ref="chunk_table_0001",
        citation_context="synthetic-oled-paper p.1 table_0001",
        confidence=0.9,
        confidence_factors={"unit_normalized": True},
        raw_values={
            "SMILES": smiles,
            "plqy": str(plqy),
            "homo": str(homo),
            "lumo": str(lumo),
        },
        status="candidate",
    )


def _normalized_records() -> list[ExtractedRecord]:
    return [
        _record(record_id="rec_000001", smiles=SMILES_A, plqy=0.86, homo=-5.4, lumo=-2.6, row_index=0),
        _record(record_id="rec_000002", smiles=SMILES_B, plqy=0.72, homo=-5.7, lumo=-3.0, row_index=1),
    ]


def _write_benchmark_fixtures(tmp_path: Path) -> dict[str, Path]:
    evidence_hits_json = tmp_path / "evidence_hits.json"
    evidence_hits_json.write_text(
        json.dumps({"run_id": RUN_ID, "query": "OLED PLQY HOMO LUMO", "hits": [_hit().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    normalized_records_jsonl = _write_jsonl(
        tmp_path / "normalized_extracted_records.jsonl",
        [record.model_dump(mode="json") for record in _normalized_records()],
    )
    gold_records_jsonl = _write_jsonl(
        tmp_path / "gold_records.jsonl",
        [
            _record(record_id="gold_000001", smiles=SMILES_A, plqy=0.86, homo=-5.4, lumo=-2.6, row_index=0).model_dump(
                mode="json"
            ),
            _record(record_id="gold_000002", smiles="CCO", plqy=0.5, homo=-5.0, lumo=-2.0, row_index=2).model_dump(
                mode="json"
            ),
        ],
    )
    conflict_report_json = tmp_path / "conflict_report.json"
    conflict_report_json.write_text(
        json.dumps({"input_record_count": 4, "merged_record_count": 2, "conflict_count": 1}),
        encoding="utf-8",
    )
    confidence_report_json = tmp_path / "extraction_confidence_report.json"
    confidence_report_json.write_text(
        json.dumps({"attempted_hit_count": 1, "extracted_record_count": 2, "rejected_record_count": 1}),
        encoding="utf-8",
    )
    provenance_report_json = tmp_path / "citation_provenance_report.json"
    provenance_report_json.write_text(
        json.dumps({"source_count": 1, "unknown_license_count": 1, "notes": ["synthetic provenance fixture"]}),
        encoding="utf-8",
    )
    unit_report_json = tmp_path / "unit_normalization_report.json"
    unit_report_json.write_text(
        json.dumps({"input_record_count": 2, "normalized_record_count": 2, "warning_count": 1}),
        encoding="utf-8",
    )
    candidate_csv = _write_csv(
        tmp_path / "candidate_training_dataset.csv",
        [
            {"smiles": SMILES_A, "plqy": 0.86, "homo": -5.4, "lumo": -2.6},
            {"smiles": SMILES_B, "plqy": 0.72, "homo": -5.7, "lumo": -3.0},
        ],
    )
    model_metrics_before_json = tmp_path / "model_metrics_before.json"
    model_metrics_before_json.write_text(json.dumps({"metrics": {"rmse": 0.40}}), encoding="utf-8")
    model_metrics_after_json = tmp_path / "model_metrics_after.json"
    model_metrics_after_json.write_text(json.dumps({"metrics": {"rmse": 0.35}}), encoding="utf-8")
    return {
        "evidence_hits": evidence_hits_json,
        "normalized_extracted_records": normalized_records_jsonl,
        "gold_records": gold_records_jsonl,
        "conflict_report": conflict_report_json,
        "extraction_confidence_report": confidence_report_json,
        "citation_provenance_report": provenance_report_json,
        "unit_normalization_report": unit_report_json,
        "candidate_training_dataset": candidate_csv,
        "model_metrics_before": model_metrics_before_json,
        "model_metrics_after": model_metrics_after_json,
    }


def _run_benchmark_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    fixtures: dict[str, Path],
    output_dir: Path,
    include_gold: bool = True,
    evidence_hits_json: Path | None = None,
    conflict_report_json: Path | None = None,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["evidence_hits", "normalized_extracted_records", "conflict_report"],
    )
    input_artifacts: dict[str, str] = {
        "normalized_extracted_records": str(fixtures["normalized_extracted_records"]),
    }
    if evidence_hits_json is not None:
        input_artifacts["evidence_hits"] = str(evidence_hits_json)
    if conflict_report_json is not None:
        input_artifacts["conflict_report"] = str(conflict_report_json)
    options = {
        "output_dir": str(output_dir),
        "candidate_training_dataset_csv": str(fixtures["candidate_training_dataset"]),
        "extraction_confidence_report_json": str(fixtures["extraction_confidence_report"]),
        "citation_provenance_report_json": str(fixtures["citation_provenance_report"]),
        "unit_normalization_report_json": str(fixtures["unit_normalization_report"]),
        "model_metrics_before_json": str(fixtures["model_metrics_before"]),
        "model_metrics_after_json": str(fixtures["model_metrics_after"]),
    }
    if include_gold:
        options["gold_records_jsonl"] = str(fixtures["gold_records"])
        options["gold_evidence_refs"] = ["chunk_table_0001"]
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={TASK_ID: options},
        max_iterations=3,
    )


def test_generic_queue_executes_extraction_benchmark_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "benchmark-out"
    fixtures = _write_benchmark_fixtures(tmp_path)

    summary = _run_benchmark_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
        evidence_hits_json=fixtures["evidence_hits"],
        conflict_report_json=fixtures["conflict_report"],
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
    assert "extraction_benchmark_report" in registry

    report_json = output_dir / f"{RUN_ID}_extraction_benchmark_report.json"
    report_md = output_dir / f"{RUN_ID}_extraction_benchmark_report.md"
    assert report_json.exists()
    assert report_md.exists()
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["extraction_benchmark_report"] == report_json

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["retrieval_recall"] == 1.0
    assert report["extraction_precision"] == 0.5
    assert report["conflict_rate"] == 0.25
    assert report["confirmation_workload_count"] == 4
    assert report["trainable_labels_gained"] == 6
    assert report["downstream_model_performance_delta"] == {"rmse": -0.05}
    assert report["metric_statuses"]["retrieval_recall"] == "computed"
    assert report["metric_statuses"]["extraction_precision"] == "computed"
    assert report["metric_statuses"]["downstream_model_performance_delta"] == "computed"
    assert report["counts"]["evidence_hits"] == 1
    assert report["counts"]["extracted_records"] == 2
    assert report["counts"]["conflicts"] == 1

    markdown = report_md.read_text(encoding="utf-8")
    assert "retrieval_recall" in markdown
    assert "extraction_precision" in markdown
    assert "conflict_rate" in markdown
    assert "confirmation_workload_count" in markdown
    assert "trainable_labels_gained" in markdown
    assert "rmse" in markdown

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


def test_missing_gold_inputs_produce_explicit_missing_gold_statuses(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "benchmark-out"
    fixtures = _write_benchmark_fixtures(tmp_path)

    summary = _run_benchmark_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
        include_gold=False,
        evidence_hits_json=fixtures["evidence_hits"],
        conflict_report_json=fixtures["conflict_report"],
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    report = json.loads((output_dir / f"{RUN_ID}_extraction_benchmark_report.json").read_text(encoding="utf-8"))
    assert report["retrieval_recall"] is None
    assert report["extraction_precision"] is None
    assert report["metric_statuses"]["retrieval_recall"] == "missing_gold_evidence"
    assert report["metric_statuses"]["extraction_precision"] == "missing_gold_records"
    assert report["counts"]["gold_evidence"] == 0
    assert report["counts"]["gold_records"] == 0

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_required_artifacts_fail_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "benchmark-out"
    fixtures = _write_benchmark_fixtures(tmp_path)

    summary = _run_benchmark_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
        evidence_hits_json=tmp_path / "missing_evidence_hits.json",
        conflict_report_json=fixtures["conflict_report"],
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
    assert "evidence" in state.error["message"]
    assert "extraction_benchmark_report" not in registry
    assert not output_dir.exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_extraction_benchmark_acceptance as acceptance

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
