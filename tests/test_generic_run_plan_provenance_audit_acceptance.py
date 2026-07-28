from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import EvidenceHit, ExtractedRecord, ParsedDocument, ParsedDocumentElement, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "track_citation_provenance"
RUN_ID = "provenance-audit-generic-queue-demo"
PROJECT_ID = "demo-project"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _parsed_document(*, license_value: str | None = "CC-BY-4.0") -> ParsedDocument:
    metadata = {
        "title": "Synthetic OLED source paper",
        "doi": "10.0000/synthetic-oled-provenance",
        "source_hash": "sha256:synthetic-oled-source",
        "citation": "Synthetic OLED source paper, 2026",
        "parsed_at": "2026-07-09T00:00:00Z",
    }
    if license_value is not None:
        metadata["license"] = license_value
    return ParsedDocument(
        paper_id="synthetic-oled-paper",
        source_path="synthetic-oled-source",
        parser_backend="synthetic_fixture",
        metadata=metadata,
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="table_0001",
                page=1,
                type="table",
                text="Fixture OLED table with SMILES, PLQY, HOMO, and LUMO values.",
                markdown=(
                    "| Compound | SMILES | PLQY (%) | HOMO | LUMO |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| Fixture-A | CCN(CC)c1ccc(N)cc1 | 0.86 | -5.4 | -2.6 |\n"
                    "| Fixture-B | O=C1N(c2ccccc2)C(=O)c2ccccc21 | 0.72 | -5.7 | -3.0 |"
                ),
            )
        ],
        tables=[],
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


def _extracted_records() -> list[ExtractedRecord]:
    return [
        ExtractedRecord(
            record_id="rec_000001",
            smiles="CCN(CC)c1ccc(N)cc1",
            properties={"plqy": 0.86, "homo": -5.4, "lumo": -2.6},
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            table_id="table_0001",
            row_index=0,
            evidence_ref="chunk_table_0001",
            citation_context="synthetic-oled-paper p.1 table_0001",
            confidence=0.91,
            confidence_factors={"unit_normalized": True},
            raw_values={"SMILES": "CCN(CC)c1ccc(N)cc1", "PLQY (%)": "86", "HOMO": "-5.4", "LUMO": "-2.6"},
            status="candidate",
        ),
        ExtractedRecord(
            record_id="rec_000002",
            smiles="O=C1N(c2ccccc2)C(=O)c2ccccc21",
            properties={"plqy": 0.72, "homo": -5.7, "lumo": -3.0},
            source_id="synthetic-oled-paper",
            paper_id="synthetic-oled-paper",
            page=1,
            table_id="table_0001",
            row_index=1,
            evidence_ref="chunk_table_0001",
            citation_context="synthetic-oled-paper p.1 table_0001",
            confidence=0.88,
            confidence_factors={"unit_normalized": True},
            raw_values={
                "SMILES": "O=C1N(c2ccccc2)C(=O)c2ccccc21",
                "PLQY (%)": "72",
                "HOMO": "-5.7",
                "LUMO": "-3.0",
            },
            status="candidate",
        ),
    ]


def _write_provenance_fixtures(
    tmp_path: Path,
    *,
    license_value: str | None = "CC-BY-4.0",
) -> tuple[Path, Path, Path]:
    parsed_document_json = tmp_path / "parsed_document.json"
    parsed_document_json.write_text(_parsed_document(license_value=license_value).model_dump_json(), encoding="utf-8")
    evidence_hits_json = tmp_path / "evidence_hits.json"
    evidence_hits_json.write_text(
        json.dumps({"run_id": RUN_ID, "query": "OLED provenance", "hits": [_table_hit().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    extracted_records_jsonl = _write_jsonl(
        tmp_path / "extracted_records.jsonl",
        [record.model_dump(mode="json") for record in _extracted_records()],
    )
    return parsed_document_json, evidence_hits_json, extracted_records_jsonl


def _run_provenance_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    parsed_document_json: Path | None,
    evidence_hits_json: Path,
    extracted_records_jsonl: Path,
    output_dir: Path,
    default_license: str = "CC-BY-4.0",
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=["parsed_document", "evidence_hits", "extracted_records"],
    )
    input_artifacts = {
        "evidence_hits": str(evidence_hits_json),
        "extracted_records": str(extracted_records_jsonl),
    }
    if parsed_document_json is not None:
        input_artifacts["parsed_document"] = str(parsed_document_json)
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options={TASK_ID: {"output_dir": str(output_dir), "default_license": default_license}},
        max_iterations=3,
    )


def test_generic_queue_executes_provenance_audit_task(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "provenance-out"
    parsed_document_json, evidence_hits_json, extracted_records_jsonl = _write_provenance_fixtures(tmp_path)

    summary = _run_provenance_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
        evidence_hits_json=evidence_hits_json,
        extracted_records_jsonl=extracted_records_jsonl,
        output_dir=output_dir,
        default_license="CC-BY-4.0",
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
    assert "citation_provenance_report" in registry
    assert "audit_summary" in registry

    report_json = output_dir / f"{RUN_ID}_citation_provenance_report.json"
    audit_summary_md = output_dir / f"{RUN_ID}_audit_summary.md"
    assert report_json.exists()
    assert audit_summary_md.exists()
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["citation_provenance_report"] == report_json
    assert run_dir / registry["audit_summary"] == audit_summary_md

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["source_count"] == 1
    assert report["evidence_count"] == 1
    assert report["extracted_record_count"] == 2
    assert report["unknown_license_count"] == 0
    source = report["sources"][0]
    assert source["paper_id"] == "synthetic-oled-paper"
    assert source["title"] == "Synthetic OLED source paper"
    assert source["citation"] == "Synthetic OLED source paper, 2026"
    assert source["doi"] == "10.0000/synthetic-oled-provenance"
    assert source["license"] == "CC-BY-4.0"
    assert source["evidence_count"] == 1
    assert source["extracted_record_count"] == 2
    assert source["license_requires_review"] is False
    assert any("does not grant reuse permission" in note for note in report["notes"])

    summary_text = audit_summary_md.read_text(encoding="utf-8")
    assert "synthetic-oled-paper" in summary_text
    assert "Synthetic OLED source paper, 2026" in summary_text
    assert "CC-BY-4.0" in summary_text
    assert "| synthetic-oled-paper |" in summary_text
    assert "| no |" in summary_text
    assert "- Evidence hits: 1" in summary_text
    assert "- Extracted records: 2" in summary_text

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_unknown_license_requires_review(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "provenance-out"
    parsed_document_json, evidence_hits_json, extracted_records_jsonl = _write_provenance_fixtures(
        tmp_path,
        license_value=None,
    )

    summary = _run_provenance_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
        evidence_hits_json=evidence_hits_json,
        extracted_records_jsonl=extracted_records_jsonl,
        output_dir=output_dir,
        default_license="unknown",
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    report = json.loads((output_dir / f"{RUN_ID}_citation_provenance_report.json").read_text(encoding="utf-8"))
    assert report["unknown_license_count"] == 1
    assert report["sources"][0]["license"] == "unknown"
    assert report["sources"][0]["license_requires_review"] is True
    summary_text = (output_dir / f"{RUN_ID}_audit_summary.md").read_text(encoding="utf-8")
    assert "| yes |" in summary_text
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "confirmed_dataset" not in registry

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_missing_parsed_document_fails_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "provenance-out"
    _, evidence_hits_json, extracted_records_jsonl = _write_provenance_fixtures(tmp_path)

    summary = _run_provenance_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=tmp_path / "missing_parsed_document.json",
        evidence_hits_json=evidence_hits_json,
        extracted_records_jsonl=extracted_records_jsonl,
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
    assert state.error["code"] == "no_parsed_documents"
    assert "parsed_document" in state.error["message"]
    assert "citation_provenance_report" not in registry
    assert not output_dir.exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_provenance_audit_acceptance as acceptance

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
