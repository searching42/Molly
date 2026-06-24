from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_queue_cli import main
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.run_plan_queue_summary import RunPlanQueueExecutionSummary
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any]) -> None:
        self.execution = dict(execution)

    def execute(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return dict(self.execution)


def _run_plan(run_id: str = "run-a") -> RunPlan:
    return RunPlan(
        run_id=run_id,
        requested_tasks=["train_model"],
        tasks=[PlannedTask(task_id="train_model")],
        available_artifacts=[],
        missing_artifacts=[],
    )


def _queue(tmp_path: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))


def _storage(tmp_path: Path) -> ProjectStorage:
    return ProjectStorage(tmp_path / "workspace")


def _factory(execution: dict[str, Any]):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(execution)

    return factory


def _write_run_plan(path: Path) -> Path:
    path.write_text(json.dumps(_run_plan().model_dump(mode="json")), encoding="utf-8")
    return path


def test_succeeded_run_plan_queue_summary_validates(tmp_path: Path) -> None:
    summary = run_run_plan_via_local_queue(
        queue=_queue(tmp_path),
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        executor_factory=_factory({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}),
    )

    parsed = RunPlanQueueExecutionSummary.model_validate(summary)

    assert parsed.ok is True
    assert parsed.terminal is True
    assert parsed.queued_job_id
    assert parsed.final_job is not None
    assert parsed.final_lease is not None
    assert parsed.loop_results == ["completed", "idle"]
    assert parsed.error is None


def test_failed_run_plan_queue_summary_validates(tmp_path: Path) -> None:
    summary = run_run_plan_via_local_queue(
        queue=_queue(tmp_path),
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        executor_factory=_factory(
            {
                "ok": False,
                "run_id": "run-a",
                "status": "FAILED",
                "error": {"message": "adapter failed"},
            }
        ),
    )

    parsed = RunPlanQueueExecutionSummary.model_validate(summary)

    assert parsed.ok is False
    assert parsed.terminal is True
    assert parsed.final_job is not None
    assert parsed.final_job["error"] == {"reason": "adapter failed"}
    assert parsed.error is None


def test_non_terminal_run_plan_queue_summary_validates(tmp_path: Path) -> None:
    summary = run_run_plan_via_local_queue(
        queue=_queue(tmp_path),
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        max_iterations=1,
        executor_factory=_factory({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}),
    )

    parsed = RunPlanQueueExecutionSummary.model_validate(summary)

    assert parsed.ok is True
    assert parsed.terminal is False
    assert parsed.loop_results == ["completed"]


def test_cli_success_output_validates_against_summary_schema(tmp_path: Path) -> None:
    stdout = io.StringIO()
    code = main(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--queue-dir",
            str(tmp_path / "queue"),
            "--project-id",
            "proj-a",
            "--run-plan-json",
            str(_write_run_plan(tmp_path / "run_plan.json")),
        ],
        executor_factory=_factory({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    parsed = RunPlanQueueExecutionSummary.model_validate(json.loads(stdout.getvalue()))

    assert code == 0
    assert parsed.ok is True
    assert parsed.error is None


def test_cli_validation_error_output_uses_summary_schema(tmp_path: Path) -> None:
    bad_run_plan = tmp_path / "bad_run_plan.json"
    bad_run_plan.write_text("{bad json", encoding="utf-8")
    stdout = io.StringIO()
    code = main(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--queue-dir",
            str(tmp_path / "queue"),
            "--project-id",
            "proj-a",
            "--run-plan-json",
            str(bad_run_plan),
        ],
        executor_factory=_factory({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    parsed = RunPlanQueueExecutionSummary.model_validate(json.loads(stdout.getvalue()))

    assert code == 2
    assert parsed.ok is False
    assert parsed.terminal is False
    assert parsed.queued_job_id == ""
    assert parsed.final_job is None
    assert parsed.final_lease is None
    assert parsed.loop_results == []
    assert parsed.error == {
        "type": "validation_error",
        "message": "run-plan JSON is not valid JSON",
    }


def test_summary_schema_rejects_error_without_type_or_message() -> None:
    with pytest.raises(ValidationError, match="error must include non-empty type and message"):
        RunPlanQueueExecutionSummary.model_validate(
            {
                "ok": False,
                "terminal": False,
                "queued_job_id": "",
                "final_job": None,
                "final_lease": None,
                "loop_results": [],
                "error": {"type": "validation_error"},
            }
        )
