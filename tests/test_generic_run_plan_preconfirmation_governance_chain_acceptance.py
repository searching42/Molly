from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CorpusChunk, EvidenceHit, ParsedDocument, ParsedDocumentElement, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


RUN_ID = "preconfirmation-governance-chain-generic-queue-demo"
PROJECT_ID = "demo-project"
EXTRACT_TASK_ID = "extract_records"
NORMALIZE_TASK_ID = "normalize_extracted_units"
PROVENANCE_TASK_ID = "track_citation_provenance"
MERGE_TASK_ID = "merge_extracted_records"
CHAIN_TASKS = [EXTRACT_TASK_ID, NORMALIZE_TASK_ID, PROVENANCE_TASK_ID, MERGE_TASK_ID]
SMILES_MERGED = "CCN(CC)c1ccc(N)cc1"
SMILES_CONFLICT = "O=C1N(c2ccccc2)C(=O)c2ccccc21"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="synthetic-oled-paper",
        source_path="synthetic-oled-source",
        parser_backend="synthetic_fixture",
        metadata={
            "title": "Synthetic OLED governance paper",
            "doi": "10.0000/synthetic-oled-governance",
            "citation": "Synthetic OLED governance paper, 2026",
            "source_hash": "sha256:synthetic-oled-governance",
            "license": "CC-BY-4.0",
            "parsed_at": "2026-07-09T00:00:00Z",
        },
        pages=[{"page": 1}],
        elements=[
            ParsedDocumentElement(
                element_id="table_0001",
                page=1,
                type="table",
                text="Synthetic table containing OLED emitter PLQY, HOMO, LUMO, and SMILES values.",
                markdown=(
                    "| Compound | SMILES | PLQY (%) | HOMO | LUMO |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    f"| Fixture-A1 | {SMILES_MERGED} | 86 | -5.40 | -2.60 |\n"
                    f"| Fixture-A2 | {SMILES_MERGED} | 87 | -5.41 | -2.61 |\n"
                    f"| Fixture-B1 | {SMILES_CONFLICT} | 72 | -5.70 | -3.00 |\n"
                    f"| Fixture-B2 | {SMILES_CONFLICT} | 90 | -5.71 | -3.01 |"
                ),
            )
        ],
        tables=[],
    )


def _table_rows() -> list[dict[str, str]]:
    return [
        {
            "Compound": "Fixture-A1",
            "SMILES": SMILES_MERGED,
            "PLQY (%)": "86",
            "HOMO": "-5.40",
            "LUMO": "-2.60",
        },
        {
            "Compound": "Fixture-A2",
            "SMILES": SMILES_MERGED,
            "PLQY (%)": "87",
            "HOMO": "-5.41",
            "LUMO": "-2.61",
        },
        {
            "Compound": "Fixture-B1",
            "SMILES": SMILES_CONFLICT,
            "PLQY (%)": "72",
            "HOMO": "-5.70",
            "LUMO": "-3.00",
        },
        {
            "Compound": "Fixture-B2",
            "SMILES": SMILES_CONFLICT,
            "PLQY (%)": "90",
            "HOMO": "-5.71",
            "LUMO": "-3.01",
        },
    ]


def _table_chunk() -> CorpusChunk:
    markdown_rows = "\n".join(
        f"| {row['Compound']} | {row['SMILES']} | {row['PLQY (%)']} | {row['HOMO']} | {row['LUMO']} |"
        for row in _table_rows()
    )
    return CorpusChunk(
        chunk_id="chunk_table_0001",
        source_id="synthetic-oled-paper",
        paper_id="synthetic-oled-paper",
        page=1,
        element_id="table_0001",
        element_type="table",
        text=(
            "OLED table with TADF emitters, PLQY, HOMO, LUMO, and SMILES values. "
            "Rows include close duplicate measurements and one conflicting PLQY pair."
        ),
        markdown=(
            "| Compound | SMILES | PLQY (%) | HOMO | LUMO |\n"
            "| --- | --- | --- | --- | --- |\n"
            f"{markdown_rows}"
        ),
        table_id="table_0001",
        retrieval_channels=["bm25", "table"],
        citation_context="synthetic-oled-paper p.1 table_0001",
        metadata={
            "caption": "Synthetic OLED pre-confirmation governance measurements",
            "headers": ["Compound", "SMILES", "PLQY (%)", "HOMO", "LUMO"],
            "rows": _table_rows(),
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


def _write_chain_fixtures(
    tmp_path: Path,
    *,
    parsed_document: bool = True,
    hits: list[EvidenceHit] | None = None,
) -> tuple[Path | None, Path, Path]:
    parsed_document_json = tmp_path / "parsed_document.json"
    if parsed_document:
        parsed_document_json.write_text(_parsed_document().model_dump_json(), encoding="utf-8")
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
    return (parsed_document_json if parsed_document else None), evidence_hits_json, chunks_jsonl


def _run_preconfirmation_chain_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    parsed_document_json: Path | None,
    evidence_hits_json: Path,
    chunks_jsonl: Path,
    output_root: Path,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=CHAIN_TASKS,
        available_artifacts=["parsed_document", "evidence_hits", "evidence_chunks"],
    )
    input_artifacts = {
        "evidence_hits": str(evidence_hits_json),
        "evidence_chunks": str(chunks_jsonl),
    }
    if parsed_document_json is not None:
        input_artifacts["parsed_document"] = str(parsed_document_json)
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
            PROVENANCE_TASK_ID: {
                "output_dir": str(output_root / "03_provenance"),
                "default_license": "CC-BY-4.0",
            },
            MERGE_TASK_ID: {
                "output_dir": str(output_root / "04_merge"),
                "property_tolerances": {
                    "plqy": 0.02,
                    "homo": 0.02,
                    "lumo": 0.02,
                },
            },
        },
        max_iterations=3,
    )


def test_generic_queue_executes_preconfirmation_governance_chain(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-governance-out"
    parsed_document_json, evidence_hits_json, chunks_jsonl = _write_chain_fixtures(tmp_path)

    summary = _run_preconfirmation_chain_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
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
    assert [task["task_id"] for task in final_job["task"]["run_plan"]["tasks"]] == CHAIN_TASKS
    assert final_job["status"] == "succeeded"
    assert final_lease["status"] == "completed"
    assert final_job["result"]["executed_tasks"] == CHAIN_TASKS

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == MERGE_TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == CHAIN_TASKS
    for artifact_id in (
        "extracted_records",
        "rejected_records",
        "extraction_confidence_report",
        "normalized_extracted_records",
        "unit_normalization_report",
        "citation_provenance_report",
        "audit_summary",
        "merged_records",
        "conflict_report",
        "candidate_training_dataset",
    ):
        assert artifact_id in registry

    extract_dir = output_root / "01_extract"
    normalize_dir = output_root / "02_normalize"
    provenance_dir = output_root / "03_provenance"
    merge_dir = output_root / "04_merge"
    extracted_records_jsonl = extract_dir / f"{RUN_ID}_extracted_records.jsonl"
    rejected_records_jsonl = extract_dir / f"{RUN_ID}_rejected_records.jsonl"
    extraction_report_json = extract_dir / f"{RUN_ID}_extraction_confidence_report.json"
    extraction_summary_md = extract_dir / f"{RUN_ID}_extraction_summary.md"
    extract_candidate_csv = extract_dir / f"{RUN_ID}_candidate_training_dataset.csv"
    normalized_jsonl = normalize_dir / f"{RUN_ID}_normalized_extracted_records.jsonl"
    normalized_candidate_csv = normalize_dir / f"{RUN_ID}_normalized_candidate_training_dataset.csv"
    normalization_report_json = normalize_dir / f"{RUN_ID}_unit_normalization_report.json"
    normalization_report_md = normalize_dir / f"{RUN_ID}_unit_normalization_report.md"
    provenance_report_json = provenance_dir / f"{RUN_ID}_citation_provenance_report.json"
    audit_summary_md = provenance_dir / f"{RUN_ID}_audit_summary.md"
    merged_records_jsonl = merge_dir / f"{RUN_ID}_merged_records.jsonl"
    conflict_report_json = merge_dir / f"{RUN_ID}_conflict_report.json"
    conflict_report_md = merge_dir / f"{RUN_ID}_conflict_report.md"
    final_candidate_csv = merge_dir / f"{RUN_ID}_merged_candidate_training_dataset.csv"
    for path in (
        extracted_records_jsonl,
        rejected_records_jsonl,
        extraction_report_json,
        extraction_summary_md,
        extract_candidate_csv,
        normalized_jsonl,
        normalized_candidate_csv,
        normalization_report_json,
        normalization_report_md,
        provenance_report_json,
        audit_summary_md,
        merged_records_jsonl,
        conflict_report_json,
        conflict_report_md,
        final_candidate_csv,
    ):
        assert path.exists()

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["normalized_extracted_records"] == normalized_jsonl
    assert run_dir / registry["citation_provenance_report"] == provenance_report_json
    assert run_dir / registry["merged_records"] == merged_records_jsonl
    assert run_dir / registry["conflict_report"] == conflict_report_json
    assert run_dir / registry["candidate_training_dataset"] == final_candidate_csv

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
    assert len(extracted_records) == 4
    assert len(normalized_records) == 4
    assert {record["status"] for record in extracted_records} == {"candidate"}
    assert {record["status"] for record in normalized_records} == {"candidate"}
    assert [record["properties"]["plqy"] for record in extracted_records] == [86.0, 87.0, 72.0, 90.0]
    assert [record["properties"]["plqy"] for record in normalized_records] == [0.86, 0.87, 0.72, 0.9]

    normalization_report = json.loads(normalization_report_json.read_text(encoding="utf-8"))
    assert normalization_report["conversion_count"] == 4
    assert all(item["rule"] == "percent_to_fraction" for item in normalization_report["conversions"])

    provenance_report = json.loads(provenance_report_json.read_text(encoding="utf-8"))
    assert provenance_report["source_count"] == 1
    assert provenance_report["unknown_license_count"] == 0
    assert provenance_report["sources"][0]["license"] == "CC-BY-4.0"
    assert provenance_report["sources"][0]["license_requires_review"] is False
    assert any("does not grant reuse permission" in note for note in provenance_report["notes"])

    merged_records = [
        json.loads(line)
        for line in merged_records_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(merged_records) == 2
    non_conflicting = next(record for record in merged_records if record["status"] == "merged")
    conflict_record = next(record for record in merged_records if record["status"] == "conflict")
    assert non_conflicting["smiles"] == SMILES_MERGED
    assert non_conflicting["properties"] == {"homo": -5.405, "lumo": -2.605, "plqy": 0.865}
    assert conflict_record["smiles"] == SMILES_CONFLICT
    assert conflict_record["property_status"]["plqy"] == "conflict"

    conflict_report = json.loads(conflict_report_json.read_text(encoding="utf-8"))
    assert conflict_report["input_record_count"] == 4
    assert conflict_report["merged_record_count"] == 2
    assert conflict_report["conflict_count"] >= 1
    assert any(item["property_id"] == "plqy" for item in conflict_report["conflicts"])
    assert any("pending human review" in note for note in conflict_report["notes"])

    csv_rows = list(csv.DictReader(final_candidate_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 1
    assert csv_rows[0]["smiles"] == SMILES_MERGED
    assert float(csv_rows[0]["plqy"]) == 0.865
    assert SMILES_CONFLICT not in final_candidate_csv.read_text(encoding="utf-8")

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


def test_missing_parsed_document_fails_before_merge(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-governance-out"
    _, evidence_hits_json, chunks_jsonl = _write_chain_fixtures(tmp_path, parsed_document=False)

    summary = _run_preconfirmation_chain_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=tmp_path / "missing_parsed_document.json",
        evidence_hits_json=evidence_hits_json,
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
    assert state.stage == PROVENANCE_TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "no_parsed_documents"
    assert "parsed_document" in state.error["message"]
    assert "extracted_records" in registry
    assert "normalized_extracted_records" in registry
    assert "citation_provenance_report" not in registry
    assert "merged_records" not in registry
    assert "conflict_report" not in registry
    history = [(item.stage, item.status) for item in state.history]
    assert (EXTRACT_TASK_ID, RunStatus.SUCCEEDED) in history
    assert (NORMALIZE_TASK_ID, RunStatus.SUCCEEDED) in history
    assert (PROVENANCE_TASK_ID, RunStatus.FAILED) in history
    assert all(item.stage != MERGE_TASK_ID for item in state.history)
    assert not (output_root / "04_merge").exists()

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_non_table_evidence_safely_produces_zero_candidate_records(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-governance-out"
    parsed_document_json, evidence_hits_json, chunks_jsonl = _write_chain_fixtures(tmp_path, hits=[_non_table_hit()])

    summary = _run_preconfirmation_chain_queue(
        queue_root=queue_root,
        storage=storage,
        parsed_document_json=parsed_document_json,
        evidence_hits_json=evidence_hits_json,
        chunks_jsonl=chunks_jsonl,
        output_root=output_root,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    assert summary["final_job"]["result"]["executed_tasks"] == CHAIN_TASKS
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == MERGE_TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert "confirmed_training_dataset" not in registry
    assert "confirmed_dataset" not in registry

    extracted_records_jsonl = output_root / "01_extract" / f"{RUN_ID}_extracted_records.jsonl"
    normalized_jsonl = output_root / "02_normalize" / f"{RUN_ID}_normalized_extracted_records.jsonl"
    merged_records_jsonl = output_root / "04_merge" / f"{RUN_ID}_merged_records.jsonl"
    final_candidate_csv = output_root / "04_merge" / f"{RUN_ID}_merged_candidate_training_dataset.csv"
    conflict_report_json = output_root / "04_merge" / f"{RUN_ID}_conflict_report.json"
    assert extracted_records_jsonl.read_text(encoding="utf-8") == ""
    assert normalized_jsonl.read_text(encoding="utf-8") == ""
    assert merged_records_jsonl.read_text(encoding="utf-8") == ""
    assert list(csv.DictReader(final_candidate_csv.open(encoding="utf-8"))) == []
    conflict_report = json.loads(conflict_report_json.read_text(encoding="utf-8"))
    assert conflict_report["input_record_count"] == 0
    assert conflict_report["merged_record_count"] == 0
    assert conflict_report["conflict_count"] == 0

    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("confirmed" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_preconfirmation_governance_chain_acceptance as acceptance

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
