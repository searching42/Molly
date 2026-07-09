from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

import pytest
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import GateName, RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "confirm_extracted_dataset"
RUN_ID = "confirmation-success-generic-queue-demo"
PROJECT_ID = "demo-project"
DATASET_ID = "synthetic_confirmed_oled_dataset"
ACTOR = "synthetic-reviewer"
SMILES = "CCN(CC)c1ccc(N)cc1"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_clean_confirmation_fixtures(tmp_path: Path, *, conflict_count: int = 0) -> dict[str, Path]:
    candidate_csv = tmp_path / "candidate_training_dataset.csv"
    candidate_csv.write_text(
        "smiles,plqy,homo,lumo\n"
        f"{SMILES},0.865,-5.405,-2.605\n",
        encoding="utf-8",
    )
    conflict_report_json = tmp_path / "conflict_report.json"
    conflict_report_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "input_record_count": 2,
                "merged_record_count": 1,
                "conflict_count": conflict_count,
                "conflicts": [
                    {
                        "conflict_id": "conflict_000001",
                        "smiles": "O=C1N(c2ccccc2)C(=O)c2ccccc21",
                        "property_id": "plqy",
                        "min_value": 0.72,
                        "max_value": 0.90,
                        "tolerance": 0.02,
                        "observations": [],
                    }
                ] if conflict_count else [],
                "notes": ["No unresolved conflicts remain for this synthetic confirmation fixture."],
            }
        ),
        encoding="utf-8",
    )
    citation_provenance_report_json = tmp_path / "citation_provenance_report.json"
    citation_provenance_report_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "source_count": 1,
                "evidence_count": 1,
                "extracted_record_count": 1,
                "unknown_license_count": 0,
                "sources": [
                    {
                        "source_id": "synthetic-oled-paper",
                        "paper_id": "synthetic-oled-paper",
                        "title": "Synthetic OLED confirmation fixture",
                        "license": "CC-BY-4.0",
                        "license_requires_review": False,
                        "evidence_count": 1,
                        "extracted_record_count": 1,
                    }
                ],
                "notes": ["Synthetic fixture for gate-controlled confirmation acceptance."],
            }
        ),
        encoding="utf-8",
    )
    return {
        "candidate_training_dataset": candidate_csv,
        "conflict_report": conflict_report_json,
        "citation_provenance_report": citation_provenance_report_json,
    }


def _confirmation_run_plan() -> RunPlan:
    return expand_run_plan(
        run_id=RUN_ID,
        requested_tasks=[TASK_ID],
        available_artifacts=[
            "candidate_training_dataset",
            "conflict_report",
            "citation_provenance_report",
        ],
    )


def _input_artifacts(fixtures: dict[str, Path]) -> dict[str, str]:
    return {
        "candidate_training_dataset": str(fixtures["candidate_training_dataset"]),
        "conflict_report": str(fixtures["conflict_report"]),
        "citation_provenance_report": str(fixtures["citation_provenance_report"]),
    }


def _task_options(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        TASK_ID: {
            "output_dir": str(output_dir),
            "dataset_id": DATASET_ID,
            "confirmed": True,
            "actor": ACTOR,
            "note": "Synthetic confirmation acceptance only.",
        }
    }


def _queue_confirmation_until_gate(
    *,
    queue_root: Path,
    storage: ProjectStorage,
    fixtures: dict[str, Path],
    output_dir: Path,
) -> tuple[dict, RunPlan]:
    run_plan = _confirmation_run_plan()
    summary = run_run_plan_via_local_queue(
        queue=_queue(queue_root),
        storage=storage,
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts=_input_artifacts(fixtures),
        task_options=_task_options(output_dir),
        max_iterations=3,
    )
    return summary, run_plan


def _resume_confirmation(
    *,
    storage: ProjectStorage,
    run_plan: RunPlan,
    fixtures: dict[str, Path],
    output_dir: Path,
    approved_gates: list[str],
) -> dict:
    return RunPlanExecutor(storage=storage).resume_after_gate(
        project_id=PROJECT_ID,
        run_plan=run_plan,
        approved_gates=approved_gates,
        actor=ACTOR,
        input_artifacts=_input_artifacts(fixtures),
        task_options=_task_options(output_dir),
    )


def test_generic_queue_pauses_before_data_mining_gate(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_clean_confirmation_fixtures(tmp_path)

    summary, _run_plan = _queue_confirmation_until_gate(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
    )

    final_job = summary["final_job"]
    final_lease = summary["final_lease"]
    assert summary["ok"] is True
    assert summary["waiting_user"] is True
    assert summary["waiting_task"] == TASK_ID
    assert summary["required_gates"] == [GateName.DATA_MINING.value]
    assert summary["loop_results"] == ["completed", "idle"]
    assert final_job["task"]["task_id"] == "run_plan_execute"
    assert [task["task_id"] for task in final_job["task"]["run_plan"]["tasks"]] == [TASK_ID]
    assert final_job["status"] == "succeeded"
    assert final_lease["status"] == "completed"
    assert final_job["result"]["status"] == RunStatus.WAITING_USER.value
    assert final_job["result"]["executed_tasks"] == []

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.WAITING_USER
    assert state.details["required_gates"] == [GateName.DATA_MINING.value]
    assert state.details["executed_tasks"] == []
    snapshot = state.details["execution_snapshot"]
    assert snapshot["task_id"] == TASK_ID
    assert snapshot["approved_gates"] == [GateName.DATA_MINING.value]
    assert snapshot["snapshot_hash"]
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    assert not output_dir.exists()


def test_resume_with_data_mining_gate_confirms_clean_synthetic_dataset(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_clean_confirmation_fixtures(tmp_path)
    _summary, run_plan = _queue_confirmation_until_gate(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
    )
    waiting_state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert waiting_state is not None
    waiting_snapshot = waiting_state.details["execution_snapshot"]

    result = _resume_confirmation(
        storage=storage,
        run_plan=run_plan,
        fixtures=fixtures,
        output_dir=output_dir,
        approved_gates=[GateName.DATA_MINING.value],
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == [TASK_ID]
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    gate_decisions = storage.read_gate_decisions(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == [TASK_ID]
    assert len(gate_decisions) == 1
    assert gate_decisions[0]["gate"] == GateName.DATA_MINING.value
    assert gate_decisions[0]["actor"] == ACTOR
    assert gate_decisions[0]["approved_snapshot_hash"] == waiting_snapshot["snapshot_hash"]
    assert gate_decisions[0]["approved_snapshot_id"] == waiting_snapshot["snapshot_id"]
    assert "confirmed_training_dataset" in registry
    assert "extraction_confirmation_record" in registry

    confirmed_csv = output_dir / f"{DATASET_ID}.csv"
    confirmation_json = output_dir / f"{DATASET_ID}_confirmation_record.json"
    confirmation_md = output_dir / f"{DATASET_ID}_human_confirmation_report.md"
    assert confirmed_csv.exists()
    assert confirmation_json.exists()
    assert confirmation_md.exists()
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    assert run_dir / registry["confirmed_training_dataset"] == confirmed_csv
    assert run_dir / registry["extraction_confirmation_record"] == confirmation_json
    assert confirmed_csv.read_text(encoding="utf-8") == fixtures["candidate_training_dataset"].read_text(encoding="utf-8")

    confirmation_record = json.loads(confirmation_json.read_text(encoding="utf-8"))
    assert confirmation_record["status"] == "confirmed"
    assert confirmation_record["confirmed_by"] == ACTOR
    assert confirmation_record["dataset_id"] == DATASET_ID
    assert confirmation_record["record_count"] == 1
    assert confirmation_record["conflict_count"] == 0
    assert confirmation_record["unknown_license_count"] == 0
    assert confirmation_record["source_reports"]["conflict_report_json"] == str(fixtures["conflict_report"])
    assert confirmation_record["source_reports"]["citation_provenance_report_json"] == str(
        fixtures["citation_provenance_report"]
    )
    markdown = confirmation_md.read_text(encoding="utf-8")
    assert DATASET_ID in markdown
    assert "Human Confirmation" in markdown or "confirmed" in markdown

    _assert_no_forbidden_outputs(tmp_path)


def test_unexpected_gate_approval_is_rejected(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_clean_confirmation_fixtures(tmp_path)
    _summary, run_plan = _queue_confirmation_until_gate(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="unexpected gate approval|gate approval required"):
        _resume_confirmation(
            storage=storage,
            run_plan=run_plan,
            fixtures=fixtures,
            output_dir=output_dir,
            approved_gates=[GateName.TRAIN_CONFIG.value],
        )

    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    assert not output_dir.exists()


def test_positive_confirmation_does_not_bypass_clean_review_requirements(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_clean_confirmation_fixtures(tmp_path, conflict_count=1)
    _summary, run_plan = _queue_confirmation_until_gate(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
    )

    result = _resume_confirmation(
        storage=storage,
        run_plan=run_plan,
        fixtures=fixtures,
        output_dir=output_dir,
        approved_gates=[GateName.DATA_MINING.value],
    )

    assert result["status"] == RunStatus.FAILED.value
    assert result["result"]["error"]["code"] == "confirmation_blocked"
    assert "unresolved_conflicts" in result["result"]["error"]["blocking_reasons"]
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def _assert_no_forbidden_outputs(tmp_path: Path) -> None:
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name and "confirmed_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)
    assert not any(("public" + "ation") in name for name in created_files)
    assert not any(("rel" + "ease") in name for name in created_files)
    assert not any(("global_" + "append") in name for name in created_files)


def _assert_no_confirmation_outputs(tmp_path: Path, output_dir: Path) -> None:
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not output_dir.exists()
    assert not any("confirmed_training_dataset" in name for name in created_files)
    assert not any("extraction_confirmation_record" in name for name in created_files)
    assert not any("human_confirmation_report" in name for name in created_files)
    _assert_no_forbidden_outputs(tmp_path)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_confirmation_success_acceptance as acceptance

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
        "train_" + "model",
        "predict_" + "candidates",
        "public" + "ation",
        "rel" + "ease",
        "global_" + "append",
    ):
        assert forbidden_text not in source
