from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.storage import ProjectStorage


def test_phase3_executor_prepares_literature_source_manifest_and_registers_artifacts(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    run_plan = expand_run_plan(
        run_id="r-phase3-sources",
        requested_tasks=["prepare_literature_corpus_sources"],
        available_artifacts=[],
    )
    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-open-007",
        run_plan=run_plan,
        task_options={
            "prepare_literature_corpus_sources": {
                "search_queries": ["TADF OLED PLQY"],
                "dois": ["10.1000/example"],
            }
        },
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-open-007", "r-phase3-sources")
    assert "corpus_source_manifest" in registry
    assert "corpus_source_manifest_md" in registry
    manifest_path = storage.run_dir("proj-open-007", "r-phase3-sources") / registry["corpus_source_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_count"] == 2


def test_phase3_executor_resumes_parse_document_and_registers_parser_outputs(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")
    mineru_output = tmp_path / "mineru_output"
    mineru_output.mkdir()
    (mineru_output / "paper.md").write_text(
        "# Test Paper\n\n| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |\n",
        encoding="utf-8",
    )
    (mineru_output / "layout.json").write_text(
        json.dumps(
            {
                "pages": [{"page": 1}],
                "tables": [
                    {
                        "page": 1,
                        "caption": "OLED measurements",
                        "headers": ["SMILES", "PLQY"],
                        "rows": [{"SMILES": "CCO", "PLQY": "0.8"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_plan = expand_run_plan(
        run_id="r-phase3-parse",
        requested_tasks=["parse_document"],
        available_artifacts=["pdf_corpus"],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-open-007",
        run_plan=run_plan,
        input_artifacts={"pdf_corpus": str(pdf)},
        task_options={"parse_document": {"mineru_output_dir": str(mineru_output)}},
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == "parse_document"
    resumed = executor.resume_after_gate(
        project_id="proj-open-007",
        run_plan=run_plan,
        approved_gates=[GateName.DATA_MINING.value],
        actor="reviewer",
        input_artifacts={"pdf_corpus": str(pdf)},
        task_options={"parse_document": {"mineru_output_dir": str(mineru_output)}},
    )

    assert resumed["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-open-007", "r-phase3-parse")
    assert "parsed_document" in registry
    assert "parsed_document_markdown" in registry
    assert "parsed_tables" in registry
    assert "parser_audit" in registry
    parsed_path = storage.run_dir("proj-open-007", "r-phase3-parse") / registry["parsed_document"]
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    assert parsed["paper_id"] == "paper"
    assert parsed["tables"][0]["headers"] == ["SMILES", "PLQY"]


def test_phase3_executor_registers_normalized_candidate_training_dataset(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    records_jsonl = tmp_path / "extracted_records.jsonl"
    record = {
        "record_id": "rec_000001",
        "smiles": "CCO",
        "properties": {"plqy_percent": 80.0},
        "source_id": "paper-1",
        "paper_id": "paper-1",
        "page": 1,
        "table_id": "table_1",
        "row_index": 0,
        "evidence_ref": "paper-1:table_1",
        "citation_context": "paper-1 p.1 table_1",
        "confidence": 0.95,
        "confidence_factors": {},
        "raw_values": {"SMILES": "CCO", "PLQY (%)": "80"},
    }
    records_jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    run_plan = expand_run_plan(
        run_id="r-phase3-normalize",
        requested_tasks=["normalize_extracted_units"],
        available_artifacts=["extracted_records"],
    )

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-open-007",
        run_plan=run_plan,
        input_artifacts={"extracted_records": str(records_jsonl)},
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-open-007", "r-phase3-normalize")
    assert "normalized_extracted_records" in registry
    assert "candidate_training_dataset" in registry
    assert "unit_normalization_report" in registry
    candidate_path = storage.run_dir("proj-open-007", "r-phase3-normalize") / registry["candidate_training_dataset"]
    assert candidate_path.name.endswith("_normalized_candidate_training_dataset.csv")
    assert "CCO" in candidate_path.read_text(encoding="utf-8")


def test_phase3_executor_registers_extraction_confirmation_record(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    candidate_csv = tmp_path / "candidate_training_dataset.csv"
    candidate_csv.write_text("smiles,plqy\nCCO,0.8\n", encoding="utf-8")
    conflict_report = tmp_path / "conflict_report.json"
    conflict_report.write_text(json.dumps({"conflict_count": 0}), encoding="utf-8")
    provenance_report = tmp_path / "citation_provenance_report.json"
    provenance_report.write_text(json.dumps({"unknown_license_count": 0}), encoding="utf-8")
    run_plan = expand_run_plan(
        run_id="r-phase3-confirm",
        requested_tasks=["confirm_extracted_dataset"],
        available_artifacts=[
            "candidate_training_dataset",
            "conflict_report",
            "citation_provenance_report",
        ],
    )
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="proj-open-007",
        run_plan=run_plan,
        input_artifacts={
            "candidate_training_dataset": str(candidate_csv),
            "conflict_report": str(conflict_report),
            "citation_provenance_report": str(provenance_report),
        },
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    resumed = executor.resume_after_gate(
        project_id="proj-open-007",
        run_plan=run_plan,
        approved_gates=[GateName.DATA_MINING.value],
        actor="reviewer",
        input_artifacts={
            "candidate_training_dataset": str(candidate_csv),
            "conflict_report": str(conflict_report),
            "citation_provenance_report": str(provenance_report),
        },
    )

    assert resumed["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("proj-open-007", "r-phase3-confirm")
    assert "confirmed_training_dataset" in registry
    assert "extraction_confirmation_record" in registry
    confirmation_path = storage.run_dir("proj-open-007", "r-phase3-confirm") / registry["extraction_confirmation_record"]
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    assert confirmation["confirmed_by"] == "reviewer"
    assert confirmation["status"] == "confirmed"
