from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.run_plan_queue_summary import (
    RunPlanQueueExecutionSummary,
    build_run_plan_queue_execution_summary,
)
from ai4s_agent.run_plan_task_runner import ExecutorFactory
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def main(
    argv: list[str] | None = None,
    *,
    executor_factory: ExecutorFactory | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        run_plan = _read_json_object(Path(args.run_plan_json), label="run-plan JSON")
        input_artifacts = _read_optional_json_object(args.input_artifacts_json, label="input_artifacts JSON")
        task_options = _read_optional_json_object(args.task_options_json, label="task_options JSON")
        queue = WorkerQueue(JsonWorkerQueueStore(Path(args.queue_dir)))
        storage = ProjectStorage(Path(args.workspace))
        summary = run_run_plan_via_local_queue(
            queue=queue,
            storage=storage,
            project_id=str(args.project_id),
            run_plan=run_plan,
            input_artifacts=input_artifacts,
            task_options=task_options,
            max_iterations=int(args.max_iterations),
            executor_factory=executor_factory,
        )
    except (OSError, ValidationError, ValueError) as exc:
        _write_json(
            build_run_plan_queue_execution_summary(
                ok=False,
                terminal=False,
                error={
                    "type": "validation_error",
                    "message": _error_message(exc),
                },
            ),
            output,
        )
        return 2
    parsed_summary = RunPlanQueueExecutionSummary.model_validate(summary)
    _write_json(parsed_summary.to_json_dict(), output)
    return 0 if parsed_summary.ok and parsed_summary.terminal else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.run_plan_queue_cli",
        description="Internal opt-in queued run-plan execution helper.",
    )
    parser.add_argument("--workspace", required=True, help="ProjectStorage workspace directory.")
    parser.add_argument("--queue-dir", required=True, help="Dedicated JsonWorkerQueueStore directory.")
    parser.add_argument("--project-id", required=True, help="Project id for the queued run-plan job.")
    parser.add_argument("--run-plan-json", required=True, help="Path to a run_plan JSON file.")
    parser.add_argument("--input-artifacts-json", help="Optional path to an input_artifacts JSON object.")
    parser.add_argument("--task-options-json", help="Optional path to a task_options JSON object.")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum local worker loop iterations.")
    return parser


def _read_optional_json_object(value: str | None, *, label: str) -> dict[str, Any] | None:
    if not value:
        return None
    return _read_json_object(Path(value), label=label)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _write_json(payload: dict[str, Any], output: TextIO) -> None:
    output.write(json.dumps(payload, sort_keys=True))
    output.write("\n")


def _error_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
