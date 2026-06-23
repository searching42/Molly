from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from ai4s_agent.run_plan_queue_cli import main
from ai4s_agent.schemas import RunPlan
from ai4s_agent.storage import ProjectStorage


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "run_plan_queue_demo"


class FakeRunPlanExecutor:
    def __init__(self) -> None:
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
        return {"ok": True, "run_id": run_plan.run_id, "status": "WAITING_USER"}


def test_run_plan_queue_cli_demo_fixture_executes_with_fake_executor(tmp_path: Path) -> None:
    fake = FakeRunPlanExecutor()

    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return fake

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--queue-dir",
            str(tmp_path / "queue"),
            "--project-id",
            "demo-project",
            "--run-plan-json",
            str(FIXTURE_DIR / "run_plan.json"),
            "--input-artifacts-json",
            str(FIXTURE_DIR / "input_artifacts.json"),
            "--task-options-json",
            str(FIXTURE_DIR / "task_options.json"),
            "--max-iterations",
            "10",
        ],
        executor_factory=factory,
        stdout=stdout,
        stderr=stderr,
    )

    summary = json.loads(stdout.getvalue())
    assert code == 0
    assert stderr.getvalue() == ""
    assert summary["ok"] is True
    assert summary["terminal"] is True
    assert summary["queued_job_id"]
    assert summary["final_job"]["status"] == "succeeded"
    assert summary["final_lease"]["status"] == "completed"
    assert summary["loop_results"] == ["completed", "idle"]
    assert "command" not in summary["final_job"]["task"]
    assert "argv" not in summary["final_job"]["task"]
    assert fake.calls == [
        {
            "project_id": "demo-project",
            "run_plan": RunPlan.model_validate(json.loads((FIXTURE_DIR / "run_plan.json").read_text(encoding="utf-8"))),
            "input_artifacts": json.loads((FIXTURE_DIR / "input_artifacts.json").read_text(encoding="utf-8")),
            "task_options": json.loads((FIXTURE_DIR / "task_options.json").read_text(encoding="utf-8")),
        }
    ]


def test_run_plan_queue_cli_demo_fixture_contains_no_command_or_argv() -> None:
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        _assert_no_command_or_argv(json.loads(path.read_text(encoding="utf-8")))


def _assert_no_command_or_argv(value: Any) -> None:
    if isinstance(value, dict):
        assert "command" not in value
        assert "argv" not in value
        for item in value.values():
            _assert_no_command_or_argv(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_command_or_argv(item)
