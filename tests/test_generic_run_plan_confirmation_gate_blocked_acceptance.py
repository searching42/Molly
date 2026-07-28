from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

import pytest
from ai4s_agent.adapters.phase3 import confirm_extracted_dataset_adapter
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import GateName, RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "confirm_extracted_dataset"
RUN_ID = "confirmation-gate-blocked-generic-queue-demo"
PROJECT_ID = "demo-project"
MERGED_SMILES = "CCN(CC)c1ccc(N)cc1"
CONFLICT_SMILES = "O=C1N(c2ccccc2)C(=O)c2ccccc21"


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


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


def _write_confirmation_fixtures(
    tmp_path: Path,
    *,
    conflict_count: int,
    unknown_license_count: int,
) -> dict[str, Path]:
    candidate_csv = _write_csv(
        tmp_path / "candidate_training_dataset.csv",
        [{"smiles": MERGED_SMILES, "plqy": 0.865, "homo": -5.405, "lumo": -2.605}],
    )
    conflicts = [
        {
            "conflict_id": "conflict_000001",
            "smiles": CONFLICT_SMILES,
            "property_id": "plqy",
            "min_value": 0.72,
            "max_value": 0.90,
            "tolerance": 0.02,
            "observations": [],
        }
    ] if conflict_count else []
    conflict_report_json = tmp_path / "conflict_report.json"
    conflict_report_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "input_record_count": 4,
                "merged_record_count": 2,
                "conflict_count": conflict_count,
                "conflicts": conflicts,
                "notes": ["Conflicted values are excluded from candidate training CSV pending human review."],
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
                "unknown_license_count": unknown_license_count,
                "sources": [],
                "notes": ["License values are tracked for review and do not grant reuse permission."],
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
            "dataset_id": "blocked_confirmation_demo",
            "confirmed": True,
            "actor": "synthetic-reviewer",
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
        actor="synthetic-reviewer",
        input_artifacts=_input_artifacts(fixtures),
        task_options=_task_options(output_dir),
    )


def test_generic_queue_pauses_before_confirmation_gate(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_confirmation_fixtures(tmp_path, conflict_count=0, unknown_license_count=0)

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
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def test_resume_with_missing_gate_remains_blocked(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_confirmation_fixtures(tmp_path, conflict_count=0, unknown_license_count=0)
    _summary, run_plan = _queue_confirmation_until_gate(
        queue_root=queue_root,
        storage=storage,
        fixtures=fixtures,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="gate approval required"):
        _resume_confirmation(
            storage=storage,
            run_plan=run_plan,
            fixtures=fixtures,
            output_dir=output_dir,
            approved_gates=[],
        )

    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.WAITING_USER
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def test_approved_gate_with_unresolved_conflict_is_blocked(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_confirmation_fixtures(tmp_path, conflict_count=1, unknown_license_count=0)
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
    assert result["failed_task"] == TASK_ID
    assert result["executed_tasks"] == []
    assert result["result"]["error"]["code"] == "confirmation_blocked"
    assert "unresolved_conflicts" in result["result"]["error"]["blocking_reasons"]
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "confirmation_blocked"
    assert "unresolved_conflicts" in state.error["blocking_reasons"]
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def test_approved_gate_with_unknown_license_is_blocked(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    storage = ProjectStorage(tmp_path / "projects")
    output_dir = tmp_path / "confirmation-out"
    fixtures = _write_confirmation_fixtures(tmp_path, conflict_count=0, unknown_license_count=1)
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
    assert "license_review_required" in result["result"]["error"]["blocking_reasons"]
    state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.stage == TASK_ID
    assert state.status == RunStatus.FAILED
    assert state.error
    assert state.error["code"] == "confirmation_blocked"
    assert "license_review_required" in state.error["blocking_reasons"]
    assert "confirmed_training_dataset" not in registry
    assert "extraction_confirmation_record" not in registry
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def test_missing_actor_or_confirmed_flag_is_blocked(tmp_path: Path) -> None:
    fixtures = _write_confirmation_fixtures(tmp_path, conflict_count=0, unknown_license_count=0)
    output_dir = tmp_path / "confirmation-out"

    missing_actor = confirm_extracted_dataset_adapter(
        {
            "run_id": RUN_ID,
            "candidate_training_dataset_csv": str(fixtures["candidate_training_dataset"]),
            "conflict_report_json": str(fixtures["conflict_report"]),
            "citation_provenance_report_json": str(fixtures["citation_provenance_report"]),
            "output_dir": str(output_dir),
            "confirmed": True,
        }
    )
    false_confirmation = confirm_extracted_dataset_adapter(
        {
            "run_id": RUN_ID,
            "candidate_training_dataset_csv": str(fixtures["candidate_training_dataset"]),
            "conflict_report_json": str(fixtures["conflict_report"]),
            "citation_provenance_report_json": str(fixtures["citation_provenance_report"]),
            "output_dir": str(output_dir),
            "confirmed": False,
            "actor": "synthetic-reviewer",
        }
    )

    assert missing_actor["status"] == "failed"
    assert false_confirmation["status"] == "failed"
    assert missing_actor["error"]["code"] == "confirmation_required"
    assert false_confirmation["error"]["code"] == "confirmation_required"
    _assert_no_confirmation_outputs(tmp_path, output_dir)


def _assert_no_confirmation_outputs(tmp_path: Path, output_dir: Path) -> None:
    created_files = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert not output_dir.exists()
    assert not any("confirmed_training_dataset" in name for name in created_files)
    assert not any("extraction_confirmation_record" in name for name in created_files)
    assert not any("human_confirmation_report" in name for name in created_files)
    assert not any(name.lower().endswith(".pdf") for name in created_files)
    assert not any("parser_audit" in name for name in created_files)
    assert not any("training" in name and "candidate_training_dataset" not in name for name in created_files)
    assert not any("prediction" in name for name in created_files)
    assert not any("promotion" in name for name in created_files)
    assert not any("publication" in name for name in created_files)
    assert not any("release" in name for name in created_files)
    assert not any("global_append" in name for name in created_files)


def test_acceptance_test_has_no_network_pdf_llm_training_or_subprocess_usage() -> None:
    import tests.test_generic_run_plan_confirmation_gate_blocked_acceptance as acceptance

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
        "allow_" + "conflicts=True",
        "allow_" + "license_review=True",
    ):
        assert forbidden_text not in source
