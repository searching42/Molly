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
