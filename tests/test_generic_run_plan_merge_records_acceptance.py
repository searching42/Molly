from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CitationLicenseReport, ExtractedRecord, LiteratureSourceProvenance, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "merge_extracted_records"
RUN_ID = "merge-records-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _record(
    *,
    record_id: str,
    smiles: str,
    source_id: str,
    plqy: float,
    homo: float,
    lumo: float,
    row_index: int,
) -> ExtractedRecord:
    return ExtractedRecord(
        record_id=record_id,
        smiles=smiles,
        properties={"plqy": plqy, "homo": homo, "lumo": lumo},
        source_id=source_id,
        paper_id=source_id,
        page=1,
        table_id="table_0001",
        row_index=row_index,
        evidence_ref=f"{source_id}:table_0001:{row_index}",
        citation_context=f"{source_id} p.1 table_0001",
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
        _record(
            record_id="rec_000001",
            smiles="CCN(CC)c1ccc(N)cc1",
            source_id="paper-a",
            plqy=0.86,
            homo=-5.40,
            lumo=-2.60,
            row_index=0,
        ),
        _record(
            record_id="rec_000002",
            smiles="CCN(CC)c1ccc(N)cc1",
            source_id="paper-b",
            plqy=0.87,
            homo=-5.41,
            lumo=-2.61,
            row_index=1,
        ),
        _record(
            record_id="rec_000003",
            smiles="O=C1N(c2ccccc2)C(=O)c2ccccc21",
            source_id="paper-c",
            plqy=0.72,
            homo=-5.70,
            lumo=-3.00,
            row_index=2,
        ),
        _record(
            record_id="rec_000004",
            smiles="O=C1N(c2ccccc2)C(=O)c2ccccc21",
            source_id="paper-d",
            plqy=0.90,
            homo=-5.71,
            lumo=-3.01,
            row_index=3,
        ),
    ]


def _citation_report() -> CitationLicenseReport:
    sources = [
        LiteratureSourceProvenance(
            source_id=f"paper-{suffix}",
            paper_id=f"paper-{suffix}",
            title=f"Synthetic OLED paper {suffix}",
            source_path=f"synthetic-paper-{suffix}",
            source_hash=f"sha256:paper-{suffix}",
            parser_backend="synthetic_fixture",
            citation=f"Synthetic OLED paper {suffix}, 2026",
            doi=f"10.0000/synthetic-{suffix}",
            license="CC-BY-4.0",
            license_requires_review=False,
            evidence_count=1,
            extracted_record_count=1,
        )
        for suffix in ("a", "b", "c", "d")
    ]
    return CitationLicenseReport(
        run_id=RUN_ID,
        source_count=len(sources),
        evidence_count=4,
        extracted_record_count=4,
        unknown_license_count=0,
        sources=sources,
        generated_at="2026-07-09T00:00:00Z",
        notes=["Synthetic provenance fixture tracks licenses for review and does not grant reuse permission."],
    )


def _write_merge_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    normalized_records_jsonl = _write_jsonl(
        tmp_path / "normalized_extracted_records.jsonl",
        [record.model_dump(mode="json") for record in _normalized_records()],
    )
    citation_report_json = tmp_path / "citation_provenance_report.json"
    citation_report_json.write_text(_citation_report().model_dump_json(), encoding="utf-8")
    return normalized_records_jsonl, citation_report_json


def _run_merge_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    normalized_records_jsonl: Path | None,
    citation_report_json: Path | None,
    output_dir: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["normalized_extracted_records", "citation_provenance_report"],
    )
    input_artifacts: dict[str, str] = {}
    if normalized_records_jsonl is not None:
        input_artifacts["normalized_extracted_records"] = str(normalized_records_jsonl)
    if citation_report_json is not None:
        input_artifacts["citation_provenance_report"] = str(citation_report_json)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={
            TASK_ID: {
                "output_dir": str(output_dir),
                "property_tolerances": {
                    "plqy": 0.02,
                    "homo": 0.02,
                    "lumo": 0.02,
                },
            }
        },
        max_iterations=3,
    )


def test_generic_queue_executes_merge_records_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "merge-out"
    normalized_records_jsonl, citation_report_json = _write_merge_fixtures(tmp_path)

    summary = _run_merge_queue(
        queue_root=queue_root,
        storage=storage,
        normalized_records_jsonl=normalized_records_jsonl,
        citation_report_json=citation_report_json,
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
    for artifact_id in ("merged_records", "conflict_report", "candidate_training_dataset"):
        assert artifact_id in registry

    merged_records_jsonl = output_dir / f"{RUN_ID}_merged_records.jsonl"
    conflict_report_json = output_dir / f"{RUN_ID}_conflict_report.json"
    conflict_report_md = output_dir / f"{RUN_ID}_conflict_report.md"
    candidate_csv = output_dir / f"{RUN_ID}_merged_candidate_training_dataset.csv"
    for path in (merged_records_jsonl, conflict_report_json, conflict_report_md, candidate_csv):
        assert path.exists()

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["merged_records"] == merged_records_jsonl
    assert run_dir / registry["conflict_report"] == conflict_report_json
    assert run_dir / registry["candidate_training_dataset"] == candidate_csv

    merged_records = [
        json.loads(line)
        for line in merged_records_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(merged_records) == 2
    non_conflicting = next(record for record in merged_records if record["status"] == "merged")
    conflict_record = next(record for record in merged_records if record["status"] == "conflict")
    assert non_conflicting["smiles"] == "CCN(CC)c1ccc(N)cc1"
    assert non_conflicting["properties"] == {"homo": -5.405, "lumo": -2.605, "plqy": 0.865}
    assert conflict_record["smiles"] == "O=C1N(c2ccccc2)C(=O)c2ccccc21"
    assert conflict_record["property_status"]["plqy"] == "conflict"
    assert conflict_record["conflict_ids"]

    conflict_report = json.loads(conflict_report_json.read_text(encoding="utf-8"))
    assert conflict_report["input_record_count"] == 4
    assert conflict_report["merged_record_count"] == 2
    assert conflict_report["conflict_count"] >= 1
    assert any(item["property_id"] == "plqy" for item in conflict_report["conflicts"])
    assert any("pending human review" in note for note in conflict_report["notes"])
    assert "Conflicted values are excluded" in conflict_report_md.read_text(encoding="utf-8")

    csv_rows = list(csv.DictReader(candidate_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 1
    assert csv_rows[0]["smiles"] == "CCN(CC)c1ccc(N)cc1"
    assert float(csv_rows[0]["plqy"]) == 0.865
    assert "O=C1N(c2ccccc2)C(=O)c2ccccc21" not in candidate_csv.read_text(encoding="utf-8")

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


def test_missing_records_fail_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "merge-out"
    _, citation_report_json = _write_merge_fixtures(tmp_path)

    summary = _run_merge_queue(
        queue_root=queue_root,
        storage=storage,
        normalized_records_jsonl=tmp_path / "missing_normalized_extracted_records.jsonl",
        citation_report_json=citation_report_json,
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
    assert state.error["code"] in {"missing_extracted_records", "invalid_extracted_records"}
    assert "extracted" in state.error["message"]
    assert "merged_records" not in registry
    assert "conflict_report" not in registry
    assert "candidate_training_dataset" not in registry

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_empty_records_input_succeeds_with_empty_candidate_outputs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "merge-out"
    normalized_records_jsonl = tmp_path / "normalized_extracted_records.jsonl"
    normalized_records_jsonl.write_text("", encoding="utf-8")
    citation_report_json = tmp_path / "citation_provenance_report.json"
    citation_report_json.write_text(_citation_report().model_dump_json(), encoding="utf-8")

    summary = _run_merge_queue(
        queue_root=queue_root,
        storage=storage,
        normalized_records_jsonl=normalized_records_jsonl,
        citation_report_json=citation_report_json,
        output_dir=output_dir,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    merged_records_jsonl = output_dir / f"{RUN_ID}_merged_records.jsonl"
    candidate_csv = output_dir / f"{RUN_ID}_merged_candidate_training_dataset.csv"
    conflict_report_json = output_dir / f"{RUN_ID}_conflict_report.json"
    assert merged_records_jsonl.read_text(encoding="utf-8") == ""
    assert list(csv.DictReader(candidate_csv.open(encoding="utf-8"))) == []
    conflict_report = json.loads(conflict_report_json.read_text(encoding="utf-8"))
    assert conflict_report["input_record_count"] == 0
    assert conflict_report["merged_record_count"] == 0
    assert conflict_report["conflict_count"] == 0
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "confirmed_dataset" not in registry

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_merge_records_acceptance as acceptance

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
