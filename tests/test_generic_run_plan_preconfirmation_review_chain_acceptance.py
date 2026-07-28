from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import CorpusChunk, EvidenceHit, ExtractedRecord, ParsedDocument, ParsedDocumentElement, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


RUN_ID = "preconfirmation-review-chain-generic-queue-demo"
PROJECT_ID = "demo-project"
EXTRACT_TASK_ID = "extract_records"
NORMALIZE_TASK_ID = "normalize_extracted_units"
PROVENANCE_TASK_ID = "track_citation_provenance"
MERGE_TASK_ID = "merge_extracted_records"
BENCHMARK_TASK_ID = "evaluate_extraction_benchmark"
LEAKAGE_TASK_ID = "check_public_dataset_leakage"
CHAIN_TASKS = [
    EXTRACT_TASK_ID,
    NORMALIZE_TASK_ID,
    PROVENANCE_TASK_ID,
    MERGE_TASK_ID,
    BENCHMARK_TASK_ID,
    LEAKAGE_TASK_ID,
]
SMILES_MERGED = "CCN(CC)c1ccc(N)cc1"
SMILES_CONFLICT = "O=C1N(c2ccccc2)C(=O)c2ccccc21"


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


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="synthetic-oled-paper",
        source_path="synthetic-oled-source",
        parser_backend="synthetic_fixture",
        metadata={
            "title": "Synthetic OLED pre-confirmation review paper",
            "doi": "10.0000/synthetic-oled-review",
            "citation": "Synthetic OLED pre-confirmation review paper, 2026",
            "source_hash": "sha256:synthetic-oled-review",
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
        {"Compound": "Fixture-A1", "SMILES": SMILES_MERGED, "PLQY (%)": "86", "HOMO": "-5.40", "LUMO": "-2.60"},
        {"Compound": "Fixture-A2", "SMILES": SMILES_MERGED, "PLQY (%)": "87", "HOMO": "-5.41", "LUMO": "-2.61"},
        {"Compound": "Fixture-B1", "SMILES": SMILES_CONFLICT, "PLQY (%)": "72", "HOMO": "-5.70", "LUMO": "-3.00"},
        {"Compound": "Fixture-B2", "SMILES": SMILES_CONFLICT, "PLQY (%)": "90", "HOMO": "-5.71", "LUMO": "-3.01"},
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
            "caption": "Synthetic OLED pre-confirmation review measurements",
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
        raw_values={"SMILES": smiles, "plqy": str(plqy), "homo": str(homo), "lumo": str(lumo)},
        status="candidate",
    )


def _write_review_fixtures(tmp_path: Path, *, leakage_overlap: bool = True) -> dict[str, Path]:
    parsed_document_json = tmp_path / "parsed_document.json"
    parsed_document_json.write_text(_parsed_document().model_dump_json(), encoding="utf-8")
    evidence_chunks_jsonl = _write_jsonl(tmp_path / "evidence_chunks.jsonl", [_table_chunk().model_dump(mode="json")])
    evidence_hits_json = tmp_path / "evidence_hits.json"
    evidence_hits_json.write_text(
        json.dumps({"run_id": RUN_ID, "query": "OLED PLQY HOMO LUMO SMILES", "hits": [_table_hit().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    gold_records_jsonl = _write_jsonl(
        tmp_path / "gold_records.jsonl",
        [
            _record(record_id="gold_000001", smiles=SMILES_MERGED, plqy=0.86, homo=-5.4, lumo=-2.6, row_index=0).model_dump(
                mode="json"
            ),
            _record(record_id="gold_000002", smiles="CCO", plqy=0.5, homo=-5.0, lumo=-2.0, row_index=4).model_dump(
                mode="json"
            ),
        ],
    )
    public_a_rows = (
        [{"smiles": SMILES_MERGED, "target": 0.80}]
        if leakage_overlap
        else [{"smiles": "N#Cc1ccccc1", "target": 0.80}]
    )
    public_benchmark_a_csv = _write_csv(tmp_path / "public_benchmark_a.csv", public_a_rows)
    public_benchmark_b_csv = _write_csv(tmp_path / "public_benchmark_b.csv", [{"smiles": "CCO", "target": 0.50}])
    model_metrics_before_json = tmp_path / "model_metrics_before.json"
    model_metrics_before_json.write_text(json.dumps({"metrics": {"rmse": 0.40}}), encoding="utf-8")
    model_metrics_after_json = tmp_path / "model_metrics_after.json"
    model_metrics_after_json.write_text(json.dumps({"metrics": {"rmse": 0.35}}), encoding="utf-8")
    return {
        "parsed_document": parsed_document_json,
        "evidence_hits": evidence_hits_json,
        "evidence_chunks": evidence_chunks_jsonl,
        "gold_records": gold_records_jsonl,
        "public_benchmark_a": public_benchmark_a_csv,
        "public_benchmark_b": public_benchmark_b_csv,
        "model_metrics_before": model_metrics_before_json,
        "model_metrics_after": model_metrics_after_json,
    }


def _run_review_chain_queue(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    fixtures: dict[str, Path],
    output_root: Path,
    include_gold: bool = True,
    public_csvs: list[Path] | None = None,
) -> dict:
    run_plan = expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=CHAIN_TASKS,
        available_artifacts=["parsed_document", "evidence_hits", "evidence_chunks"],
    )
    benchmark_options = {
        "output_dir": str(output_root / "05_benchmark"),
        "model_metrics_before_json": str(fixtures["model_metrics_before"]),
        "model_metrics_after_json": str(fixtures["model_metrics_after"]),
    }
    if include_gold:
        benchmark_options["gold_records_jsonl"] = str(fixtures["gold_records"])
        benchmark_options["gold_evidence_refs"] = ["chunk_table_0001"]
    return run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts={
            "parsed_document": str(fixtures["parsed_document"]),
            "evidence_hits": str(fixtures["evidence_hits"]),
            "evidence_chunks": str(fixtures["evidence_chunks"]),
        },
        task_options={
            EXTRACT_TASK_ID: {
                "output_dir": str(output_root / "01_extract"),
                "confidence_threshold": 0.6,
            },
            NORMALIZE_TASK_ID: {"output_dir": str(output_root / "02_normalize")},
            PROVENANCE_TASK_ID: {
                "output_dir": str(output_root / "03_provenance"),
                "default_license": "CC-BY-4.0",
            },
            MERGE_TASK_ID: {
                "output_dir": str(output_root / "04_merge"),
                "property_tolerances": {"plqy": 0.02, "homo": 0.02, "lumo": 0.02},
            },
            BENCHMARK_TASK_ID: benchmark_options,
            LEAKAGE_TASK_ID: {
                "output_dir": str(output_root / "06_leakage"),
                "public_dataset_csvs": [
                    str(path)
                    for path in (
                        public_csvs
                        if public_csvs is not None
                        else [fixtures["public_benchmark_a"], fixtures["public_benchmark_b"]]
                    )
                ],
            },
        },
        max_iterations=3,
    )


def test_generic_queue_executes_full_preconfirmation_review_chain(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-review-out"
    fixtures = _write_review_fixtures(tmp_path, leakage_overlap=True)

    summary = _run_review_chain_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
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
    assert state.stage == LEAKAGE_TASK_ID
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
        "extraction_benchmark_report",
        "benchmark_contamination_report",
    ):
        assert artifact_id in registry

    paths = {
        "extracted": output_root / "01_extract" / f"{RUN_ID}_extracted_records.jsonl",
        "rejected": output_root / "01_extract" / f"{RUN_ID}_rejected_records.jsonl",
        "extraction_report": output_root / "01_extract" / f"{RUN_ID}_extraction_confidence_report.json",
        "extraction_summary": output_root / "01_extract" / f"{RUN_ID}_extraction_summary.md",
        "normalized": output_root / "02_normalize" / f"{RUN_ID}_normalized_extracted_records.jsonl",
        "normalization_report": output_root / "02_normalize" / f"{RUN_ID}_unit_normalization_report.json",
        "normalization_md": output_root / "02_normalize" / f"{RUN_ID}_unit_normalization_report.md",
        "provenance_report": output_root / "03_provenance" / f"{RUN_ID}_citation_provenance_report.json",
        "audit_summary": output_root / "03_provenance" / f"{RUN_ID}_audit_summary.md",
        "merged": output_root / "04_merge" / f"{RUN_ID}_merged_records.jsonl",
        "conflict_report": output_root / "04_merge" / f"{RUN_ID}_conflict_report.json",
        "conflict_md": output_root / "04_merge" / f"{RUN_ID}_conflict_report.md",
        "candidate_csv": output_root / "04_merge" / f"{RUN_ID}_merged_candidate_training_dataset.csv",
        "benchmark_report": output_root / "05_benchmark" / f"{RUN_ID}_extraction_benchmark_report.json",
        "benchmark_md": output_root / "05_benchmark" / f"{RUN_ID}_extraction_benchmark_report.md",
        "leakage_report": output_root / "06_leakage" / f"{RUN_ID}_benchmark_contamination_report.json",
        "leakage_md": output_root / "06_leakage" / f"{RUN_ID}_benchmark_contamination_report.md",
    }
    for path in paths.values():
        assert path.exists()

    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["candidate_training_dataset"] == paths["candidate_csv"]
    assert run_dir / registry["extraction_benchmark_report"] == paths["benchmark_report"]
    assert run_dir / registry["benchmark_contamination_report"] == paths["leakage_report"]

    extracted_records = [json.loads(line) for line in paths["extracted"].read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized_records = [json.loads(line) for line in paths["normalized"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {record["status"] for record in extracted_records} == {"candidate"}
    assert {record["status"] for record in normalized_records} == {"candidate"}
    assert [record["properties"]["plqy"] for record in extracted_records] == [86.0, 87.0, 72.0, 90.0]
    assert [record["properties"]["plqy"] for record in normalized_records] == [0.86, 0.87, 0.72, 0.9]

    provenance_report = json.loads(paths["provenance_report"].read_text(encoding="utf-8"))
    assert provenance_report["unknown_license_count"] == 0
    assert provenance_report["sources"][0]["license"] == "CC-BY-4.0"
    assert any("does not grant reuse permission" in note for note in provenance_report["notes"])

    merged_records = [json.loads(line) for line in paths["merged"].read_text(encoding="utf-8").splitlines() if line.strip()]
    non_conflicting = next(record for record in merged_records if record["status"] == "merged")
    conflict_record = next(record for record in merged_records if record["status"] == "conflict")
    assert non_conflicting["smiles"] == SMILES_MERGED
    assert non_conflicting["properties"] == {"homo": -5.405, "lumo": -2.605, "plqy": 0.865}
    assert conflict_record["smiles"] == SMILES_CONFLICT
    assert conflict_record["property_status"]["plqy"] == "conflict"

    conflict_report = json.loads(paths["conflict_report"].read_text(encoding="utf-8"))
    assert any(item["property_id"] == "plqy" for item in conflict_report["conflicts"])

    final_candidate_rows = list(csv.DictReader(paths["candidate_csv"].open(encoding="utf-8")))
    assert len(final_candidate_rows) == 1
    assert final_candidate_rows[0]["smiles"] == SMILES_MERGED

    benchmark_report = json.loads(paths["benchmark_report"].read_text(encoding="utf-8"))
    assert benchmark_report["retrieval_recall"] == 1.0
    assert 0.0 < benchmark_report["extraction_precision"] < 1.0
    assert benchmark_report["conflict_rate"] == 0.25
    assert benchmark_report["confirmation_workload_count"] >= 1
    assert benchmark_report["trainable_labels_gained"] > 0
    assert benchmark_report["downstream_model_performance_delta"] == {"rmse": -0.05}
    assert benchmark_report["metric_statuses"]["retrieval_recall"] == "computed"
    assert benchmark_report["metric_statuses"]["extraction_precision"] == "computed"

    leakage_report = json.loads(paths["leakage_report"].read_text(encoding="utf-8"))
    assert leakage_report["status"] == "overlap_detected"
    assert leakage_report["total_overlap_count"] > 0
    assert leakage_report["total_overlap_smiles"] == [SMILES_MERGED]

    _assert_no_forbidden_outputs(tmp_path)


def test_clear_leakage_path_succeeds(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-review-out"
    fixtures = _write_review_fixtures(tmp_path, leakage_overlap=False)

    summary = _run_review_chain_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_root=output_root,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    benchmark_report = output_root / "05_benchmark" / f"{RUN_ID}_extraction_benchmark_report.json"
    leakage_report = output_root / "06_leakage" / f"{RUN_ID}_benchmark_contamination_report.json"
    assert benchmark_report.exists()
    report = json.loads(leakage_report.read_text(encoding="utf-8"))
    assert report["status"] == "clear"
    assert report["total_overlap_count"] == 0
    _assert_no_forbidden_outputs(tmp_path)


def test_missing_gold_inputs_report_missing_gold_statuses_but_do_not_fail(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-review-out"
    fixtures = _write_review_fixtures(tmp_path, leakage_overlap=True)

    summary = _run_review_chain_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_root=output_root,
        include_gold=False,
    )

    assert summary["ok"] is True
    assert summary["final_job"]["status"] == "succeeded"
    benchmark_report = json.loads(
        (output_root / "05_benchmark" / f"{RUN_ID}_extraction_benchmark_report.json").read_text(encoding="utf-8")
    )
    assert benchmark_report["retrieval_recall"] is None
    assert benchmark_report["extraction_precision"] is None
    assert benchmark_report["metric_statuses"]["retrieval_recall"] == "missing_gold_evidence"
    assert benchmark_report["metric_statuses"]["extraction_precision"] == "missing_gold_records"
    assert (output_root / "06_leakage" / f"{RUN_ID}_benchmark_contamination_report.json").exists()
    _assert_no_forbidden_outputs(tmp_path)


def test_missing_public_benchmark_fails_only_at_leakage_stage(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_root = tmp_path / "preconfirmation-review-out"
    fixtures = _write_review_fixtures(tmp_path, leakage_overlap=True)

    summary = _run_review_chain_queue(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_root=output_root,
        public_csvs=[fixtures["public_benchmark_a"], tmp_path / "missing_public_benchmark.csv"],
    )

    assert summary["ok"] is False
    assert summary["terminal"] is True
    assert summary["loop_results"] == ["failed", "idle"]
    assert summary["final_job"]["status"] == "failed"
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == LEAKAGE_TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "public_dataset_missing"
    history = [(item.stage, item.status) for item in state.history]
    for task_id in CHAIN_TASKS[:-1]:
        assert (task_id, RunStatus.SUCCEEDED) in history
    assert "extraction_benchmark_report" in registry
    assert "benchmark_contamination_report" not in registry
    _assert_no_forbidden_outputs(tmp_path)


def _assert_no_forbidden_outputs(tmp_path: Path) -> None:
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


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_preconfirmation_review_chain_acceptance as acceptance

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
