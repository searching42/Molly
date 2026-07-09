from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import ExtractedRecord, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "normalize_extracted_units"
RUN_ID = "normalize-units-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _extracted_records() -> list[ExtractedRecord]:
    return [
        ExtractedRecord(
            record_id="rec_000001",
            smiles="CCN(CC)c1ccc(N)cc1",
            properties={"plqy": 86.0, "homo": -5.4, "lumo": -2.6},
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            table_id="table_0001",
            row_index=0,
            evidence_ref="chunk_table_0001",
            citation_context="synthetic-oled-paper p.1 table_0001",
            confidence=0.91,
            confidence_factors={"table_row": True},
            raw_values={
                "Compound": "Fixture-A",
                "SMILES": "CCN(CC)c1ccc(N)cc1",
                "PLQY (%)": "86",
                "HOMO": "-5.4",
                "LUMO": "-2.6",
            },
            status="candidate",
        ),
        ExtractedRecord(
            record_id="rec_000002",
            smiles="O=C1N(c2ccccc2)C(=O)c2ccccc21",
            properties={"plqy": 72.0, "homo": -5.7, "lumo": -3.0},
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            table_id="table_0001",
            row_index=1,
            evidence_ref="chunk_table_0001",
            citation_context="synthetic-oled-paper p.1 table_0001",
            confidence=0.88,
            confidence_factors={"table_row": True},
            raw_values={
                "Compound": "Fixture-B",
                "SMILES": "O=C1N(c2ccccc2)C(=O)c2ccccc21",
                "PLQY (%)": "72",
                "HOMO": "-5.7",
                "LUMO": "-3.0",
            },
            status="candidate",
        ),
    ]


def _write_extracted_records(path: Path) -> Path:
    return _write_jsonl(path, [record.model_dump(mode="json") for record in _extracted_records()])


def _run_normalize_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    extracted_records_jsonl: Path | None,
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["extracted_records"],
    )
    input_artifacts: dict[str, str] = {}
    if extracted_records_jsonl is not None:
        input_artifacts["extracted_records"] = str(extracted_records_jsonl)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={TASK_ID: {"output_dir": str(output_dir)}},
        max_iterations=3,
    )


def test_generic_queue_executes_unit_normalization_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "normalize-out"
    extracted_records_jsonl = _write_extracted_records(tmp_path / "extracted_records.jsonl")

    summary = _run_normalize_queue(
        queue_root=queue_root,
        storage=storage,
        extracted_records_jsonl=extracted_records_jsonl,
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
    for artifact_id in (
        "normalized_extracted_records",
        "candidate_training_dataset",
        "unit_normalization_report",
    ):
        assert artifact_id in registry

    normalized_jsonl = output_dir / f"{RUN_ID}_normalized_extracted_records.jsonl"
    candidate_csv = output_dir / f"{RUN_ID}_normalized_candidate_training_dataset.csv"
    report_json = output_dir / f"{RUN_ID}_unit_normalization_report.json"
    report_md = output_dir / f"{RUN_ID}_unit_normalization_report.md"
    for path in (normalized_jsonl, candidate_csv, report_json, report_md):
        assert path.exists()

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["normalized_extracted_records"] == normalized_jsonl
    assert run_dir / registry["candidate_training_dataset"] == candidate_csv
    assert run_dir / registry["unit_normalization_report"] == report_json

    records = [
        json.loads(line)
        for line in normalized_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["status"] for record in records] == ["candidate", "candidate"]
    assert [record["properties"]["plqy"] for record in records] == [0.86, 0.72]
    assert [record["properties"]["homo"] for record in records] == [-5.4, -5.7]
    assert [record["properties"]["lumo"] for record in records] == [-2.6, -3.0]
    assert all(record["raw_values"]["PLQY (%)"] in {"86", "72"} for record in records)

    csv_rows = list(csv.DictReader(candidate_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 2
    assert [float(row["plqy"]) for row in csv_rows] == [0.86, 0.72]
    assert [float(row["homo"]) for row in csv_rows] == [-5.4, -5.7]
    assert [float(row["lumo"]) for row in csv_rows] == [-2.6, -3.0]

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["input_record_count"] == 2
    assert report["normalized_record_count"] == 2
    assert report["conversion_count"] == 2
    percent_conversions = [item for item in report["conversions"] if item["rule"] == "percent_to_fraction"]
    assert len(percent_conversions) == 2
    assert {item["source_unit"] for item in percent_conversions} == {"%"}
    assert {item["canonical_unit"] for item in percent_conversions} == {"fraction"}
    assert {item["source_header"] for item in percent_conversions} == {"PLQY (%)"}
    assert {item["canonical_value"] for item in percent_conversions} == {0.86, 0.72}
    report_text = report_md.read_text(encoding="utf-8")
    assert "candidate data" in report_text
    assert "human confirmation before training dataset promotion" in report_text

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_extracted_records_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "normalize-out"

    summary = _run_normalize_queue(
        queue_root=queue_root,
        storage=storage,
        extracted_records_jsonl=tmp_path / "missing_extracted_records.jsonl",
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
    assert state.error["code"] in {"missing_extracted_records", "invalid_extracted_records"}
    assert "extracted" in state.error["message"]
    assert "normalized_extracted_records" not in registry
    assert "candidate_training_dataset" not in registry

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_empty_extracted_records_input_succeeds_with_empty_candidate_outputs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "normalize-out"
    extracted_records_jsonl = tmp_path / "empty_extracted_records.jsonl"
    extracted_records_jsonl.write_text("", encoding="utf-8")

    summary = _run_normalize_queue(
        queue_root=queue_root,
        storage=storage,
        extracted_records_jsonl=extracted_records_jsonl,
        output_dir=output_dir,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "confirmed_dataset" not in registry

    normalized_jsonl = output_dir / f"{RUN_ID}_normalized_extracted_records.jsonl"
    candidate_csv = output_dir / f"{RUN_ID}_normalized_candidate_training_dataset.csv"
    report_json = output_dir / f"{RUN_ID}_unit_normalization_report.json"
    assert normalized_jsonl.read_text(encoding="utf-8") == ""
    csv_rows = list(csv.DictReader(candidate_csv.open(encoding="utf-8")))
    assert csv_rows == []
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["input_record_count"] == 0
    assert report["normalized_record_count"] == 0
    assert report["conversion_count"] == 0

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_normalize_units_acceptance as acceptance

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
