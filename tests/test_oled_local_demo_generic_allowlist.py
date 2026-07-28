from __future__ import annotations

import ast
import copy
import inspect

import pytest

import ai4s_agent.agents.oled_local_demo_generic_allowlist as allowlist
from ai4s_agent.agents.oled_local_demo_generic_allowlist import (
    ALLOWLISTED_RUN_PLAN_TASKS,
    LOCAL_DEMO_TASK_ID,
    is_oled_local_demo_run_plan_execute_job,
    oled_local_demo_task_options,
    validate_oled_local_demo_run_plan_execute_job,
    validate_oled_local_demo_run_plan_execute_task,
)
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import build_run_plan_execute_task


def _valid_task_payload() -> dict:
    return build_run_plan_execute_task(
        project_id="demo-project",
        run_id="oled-allowlist-demo",
        run_plan=expand_run_plan(
            run_id="oled-allowlist-demo",
            requested_tasks=[LOCAL_DEMO_TASK_ID],
        ),
        input_artifacts={},
        task_options={
            LOCAL_DEMO_TASK_ID: {
                "input_bundle": "/tmp/oled_demo_bundle.json",
                "output_dir": "/tmp/oled-agent-demo",
                "overwrite": False,
                "project_id": "demo-project",
            }
        },
    )


def _valid_job() -> dict:
    return {
        "job_id": "job-demo-project-oled-allowlist-demo",
        "project_id": "demo-project",
        "run_id": "oled-allowlist-demo",
        "status": "queued",
        "task": _valid_task_payload(),
    }


def test_valid_generic_run_plan_execute_task_validates() -> None:
    parsed = validate_oled_local_demo_run_plan_execute_task(_valid_task_payload())

    assert parsed.task_id == "run_plan_execute"
    assert parsed.kind == "run_plan_execute"
    assert [task.task_id for task in parsed.run_plan.tasks] == [LOCAL_DEMO_TASK_ID]
    assert parsed.run_plan.requested_tasks == [LOCAL_DEMO_TASK_ID]
    assert list(ALLOWLISTED_RUN_PLAN_TASKS) == [LOCAL_DEMO_TASK_ID]
    assert oled_local_demo_task_options(parsed) == {
        "input_bundle": "/tmp/oled_demo_bundle.json",
        "output_dir": "/tmp/oled-agent-demo",
        "overwrite": False,
        "project_id": "demo-project",
    }


def test_valid_generic_queue_job_validates_and_is_detected() -> None:
    job = _valid_job()

    parsed = validate_oled_local_demo_run_plan_execute_job(job)

    assert parsed.run_plan.run_id == "oled-allowlist-demo"
    assert is_oled_local_demo_run_plan_execute_job(job) is True


def test_invalid_jobs_are_not_detected() -> None:
    assert is_oled_local_demo_run_plan_execute_job({"task": {"task_id": "other_task"}}) is False
    assert is_oled_local_demo_run_plan_execute_job({"task": _disallowed_task_payload("run_baseline")}) is False
    assert is_oled_local_demo_run_plan_execute_job(object()) is False


def test_non_dict_task_payload_fails_clearly() -> None:
    with pytest.raises(ValueError, match="task payload must be an object"):
        validate_oled_local_demo_run_plan_execute_task(object())


def test_missing_job_task_fails_clearly() -> None:
    with pytest.raises(ValueError, match="job task must be an object"):
        validate_oled_local_demo_run_plan_execute_job({"job_id": "missing-task"})


def test_non_generic_task_id_fails_clearly() -> None:
    with pytest.raises(ValueError, match="task_id must be run_plan_execute"):
        validate_oled_local_demo_run_plan_execute_task({"task_id": "other_task"})


def test_wrong_generic_kind_fails_clearly() -> None:
    task = _valid_task_payload()
    task["kind"] = "other_kind"

    with pytest.raises(ValueError, match="task kind must be run_plan_execute"):
        validate_oled_local_demo_run_plan_execute_task(task)


def test_generic_run_plan_with_run_baseline_fails_clearly() -> None:
    with pytest.raises(ValueError, match="generic_run_plan_not_allowlisted_for_oled_local_demo"):
        validate_oled_local_demo_run_plan_execute_task(_disallowed_task_payload("run_baseline"))


def test_generic_run_plan_with_more_than_one_task_fails_clearly() -> None:
    task = _valid_task_payload()
    extra = copy.deepcopy(task["run_plan"]["tasks"][0])
    extra["task_id"] = "run_baseline"
    task["run_plan"]["tasks"].append(extra)

    with pytest.raises(ValueError, match="generic_run_plan_not_allowlisted_for_oled_local_demo"):
        validate_oled_local_demo_run_plan_execute_task(task)


def test_generic_run_plan_requested_tasks_must_match_allowlist() -> None:
    task = _valid_task_payload()
    task["run_plan"]["requested_tasks"] = ["run_baseline"]

    with pytest.raises(ValueError, match="generic_run_plan_not_allowlisted_for_oled_local_demo"):
        validate_oled_local_demo_run_plan_execute_task(task)


def test_oled_specific_worker_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="task_id must be run_plan_execute"):
        validate_oled_local_demo_run_plan_execute_task({"task_id": "execute_oled_local_demo_runplan"})


def test_allowlist_helper_safety_guards() -> None:
    source = inspect.getsource(allowlist)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "RunPlanExecutor",
        "RunPlanExecutorTaskRunner",
        "ProjectStorage",
        "LocalWorkerLoop",
        "WorkerQueuePoller",
        "ai4s_agent.adapters",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
    )
    forbidden_calls = (
        "queue.enqueue",
        "queue.cancel",
        "enqueue_retry_of_failed_job",
        "poll_once",
        "resume_after_gate",
    )

    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert not any(call in source for call in forbidden_calls)
    assert "admission" not in allowlist.__name__
    assert "receipt" not in allowlist.__name__
    assert "preflight" not in allowlist.__name__
    assert "writer" not in allowlist.__name__


def _disallowed_task_payload(task_id: str) -> dict:
    return build_run_plan_execute_task(
        project_id="demo-project",
        run_id=f"{task_id}-demo",
        run_plan=expand_run_plan(run_id=f"{task_id}-demo", requested_tasks=[task_id]),
        input_artifacts={},
        task_options={},
    )
