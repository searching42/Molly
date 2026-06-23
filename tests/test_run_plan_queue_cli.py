from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from typing import Any

from ai4s_agent import run_plan_queue_cli
from ai4s_agent.run_plan_queue_cli import main
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any]) -> None:
        self.execution = dict(execution)
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "project_id": project_id,
                "run_plan": run_plan,
                "input_artifacts": input_artifacts,
                "task_options": task_options,
            }
        )
        return dict(self.execution)


def _run_plan(run_id: str = "run-a") -> RunPlan:
    return RunPlan(
        run_id=run_id,
        requested_tasks=["train_model"],
        tasks=[PlannedTask(task_id="train_model")],
        available_artifacts=[],
        missing_artifacts=[],
    )


def _write_run_plan(path: Path, run_plan: RunPlan | None = None) -> Path:
    path.write_text(json.dumps((run_plan or _run_plan()).model_dump(mode="json")), encoding="utf-8")
    return path


def _factory(fake: FakeRunPlanExecutor):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return fake

    return factory


def _write_json(path: Path, payload: dict[str, Any] | list[Any] | str) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _argv(
    tmp_path: Path,
    *,
    queue_dir: Path | None = None,
    run_plan_json: Path | None = None,
    input_artifacts_json: Path | None = None,
    task_options_json: Path | None = None,
) -> list[str]:
    args = [
        "--workspace",
        str(tmp_path / "workspace"),
        "--queue-dir",
        str(queue_dir or (tmp_path / "queue")),
        "--project-id",
        "proj-a",
        "--run-plan-json",
        str(run_plan_json or _write_run_plan(tmp_path / "run_plan.json")),
    ]
    if input_artifacts_json is not None:
        args.extend(["--input-artifacts-json", str(input_artifacts_json)])
    if task_options_json is not None:
        args.extend(["--task-options-json", str(task_options_json)])
    args.extend(
        [
            "--max-iterations",
            "10",
        ]
    )
    return args


def _run_cli(argv: list[str], fake: FakeRunPlanExecutor) -> tuple[int, dict[str, Any], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(argv, executor_factory=_factory(fake), stdout=stdout, stderr=stderr)
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


def test_run_plan_queue_cli_returns_zero_for_waiting_user_execution(tmp_path: Path) -> None:
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    code, payload, stderr = _run_cli(_argv(tmp_path), fake)

    assert code == 0
    assert stderr == ""
    assert payload["ok"] is True
    assert payload["terminal"] is True
    assert payload["loop_results"] == ["completed", "idle"]
    assert payload["final_job"]["status"] == "succeeded"
    assert payload["final_lease"]["status"] == "completed"
    assert fake.calls[0]["project_id"] == "proj-a"


def test_run_plan_queue_cli_passes_input_artifacts_and_task_options(tmp_path: Path) -> None:
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})
    input_artifacts_json = _write_json(tmp_path / "input_artifacts.json", {"dataset": "datasets/input.csv"})
    task_options_json = _write_json(tmp_path / "task_options.json", {"train_model": {"epochs": 1}})

    code, payload, stderr = _run_cli(
        _argv(
            tmp_path,
            input_artifacts_json=input_artifacts_json,
            task_options_json=task_options_json,
        ),
        fake,
    )

    assert code == 0
    assert stderr == ""
    assert payload["ok"] is True
    assert fake.calls[0]["input_artifacts"] == {"dataset": "datasets/input.csv"}
    assert fake.calls[0]["task_options"] == {"train_model": {"epochs": 1}}


def test_run_plan_queue_cli_returns_one_for_failed_execution(tmp_path: Path) -> None:
    fake = FakeRunPlanExecutor(
        {
            "ok": False,
            "run_id": "run-a",
            "status": "FAILED",
            "error": {"message": "adapter failed"},
        }
    )

    code, payload, stderr = _run_cli(_argv(tmp_path), fake)

    assert code == 1
    assert stderr == ""
    assert payload["ok"] is False
    assert payload["final_job"]["status"] == "failed"
    assert payload["final_job"]["error"] == {"reason": "adapter failed"}


def test_run_plan_queue_cli_returns_two_for_invalid_run_plan_json(tmp_path: Path) -> None:
    run_plan_path = tmp_path / "bad_run_plan.json"
    run_plan_path.write_text("{bad json", encoding="utf-8")
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    code, payload, stderr = _run_cli(_argv(tmp_path, run_plan_json=run_plan_path), fake)

    assert code == 2
    assert stderr == ""
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "valid JSON" in payload["error"]["message"]
    assert fake.calls == []


def test_run_plan_queue_cli_returns_two_for_malformed_input_artifacts_json(tmp_path: Path) -> None:
    input_artifacts_path = tmp_path / "bad_input_artifacts.json"
    input_artifacts_path.write_text("{bad json", encoding="utf-8")
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    code, payload, stderr = _run_cli(_argv(tmp_path, input_artifacts_json=input_artifacts_path), fake)

    assert code == 2
    assert stderr == ""
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "input_artifacts JSON is not valid JSON" in payload["error"]["message"]
    assert fake.calls == []


def test_run_plan_queue_cli_returns_two_for_non_object_task_options_json(tmp_path: Path) -> None:
    task_options_path = _write_json(tmp_path / "task_options.json", [{"train_model": {"epochs": 1}}])
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    code, payload, stderr = _run_cli(_argv(tmp_path, task_options_json=task_options_path), fake)

    assert code == 2
    assert stderr == ""
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "task_options JSON root must be an object" in payload["error"]["message"]
    assert fake.calls == []


def test_run_plan_queue_cli_returns_two_for_non_dedicated_queue(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    queue.enqueue("proj-old", "run-old", {"task_id": "some_other_task"})
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    code, payload, stderr = _run_cli(_argv(tmp_path, queue_dir=queue_dir), fake)

    assert code == 2
    assert stderr == ""
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "empty/dedicated queue" in payload["error"]["message"]
    assert len(queue.list_jobs()) == 1
    assert fake.calls == []


def test_run_plan_queue_cli_does_not_import_flask_routes() -> None:
    source = inspect.getsource(run_plan_queue_cli)

    assert "flask" not in source.lower()
    assert "routes.run_plans" not in source
