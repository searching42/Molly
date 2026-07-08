from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_enqueue as generic_enqueue
from ai4s_agent.agents.oled_local_demo_generic_enqueue import (
    enqueue_oled_local_demo_generic_run_plan_job,
    main,
)
from ai4s_agent.agents.oled_local_demo_generic_worker_loop import run_oled_local_demo_generic_worker_loop
from ai4s_agent.agents.oled_mvp_demo import write_local_input_bundle_template
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


EXPECTED_ARTIFACTS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _enqueue(
    tmp_path: Path,
    *,
    queue_root: Path | None = None,
    run_id: str = "oled-generic-queue-demo",
    input_bundle: Path | None = None,
    output_dir: Path | None = None,
    goal: str | None = None,
    overwrite: bool = False,
) -> dict:
    return enqueue_oled_local_demo_generic_run_plan_job(
        queue_root=queue_root or (tmp_path / "queue"),
        project_id="demo-project",
        run_id=run_id,
        input_bundle=input_bundle or (tmp_path / "missing-ok.json"),
        output_dir=output_dir or (tmp_path / "out"),
        goal=goal,
        overwrite=overwrite,
    )


def _run_worker_loop(tmp_path: Path, *, queue_root: Path, project_root: Path | None = None) -> dict:
    return run_oled_local_demo_generic_worker_loop(
        queue_root=queue_root,
        project_root=project_root or (tmp_path / "projects"),
        worker_id="generic-worker-1",
        max_iterations=3,
        now="2026-01-01T00:00:00Z",
    )


def test_helper_enqueues_exactly_one_generic_run_plan_job(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"

    result = _enqueue(
        tmp_path,
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=output_dir,
        goal="Find OLED emitters",
        overwrite=True,
    )

    assert result == {
        "enqueued": True,
        "executable": False,
        "executed": False,
        "generic_run_plan_queue": True,
        "input_bundle": str(tmp_path / "missing-ok.json"),
        "job_id": "job-demo-project-oled-generic-queue-demo",
        "job_status": "queued",
        "ok": True,
        "output_dir": str(output_dir),
        "overwrite": True,
        "project_id": "demo-project",
        "queue_root": str(queue_root),
        "queue_task_id": "run_plan_execute",
        "queue_task_kind": "run_plan_execute",
        "run_id": "oled-generic-queue-demo",
        "run_plan_tasks": ["execute_oled_local_demo"],
    }
    jobs = _queue(queue_root).list_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "queued"
    assert job["task"]["task_id"] == "run_plan_execute"
    assert job["task"]["kind"] == "run_plan_execute"
    assert job["task"]["task_id"] != "execute_oled_local_demo_runplan"
    assert job["task"]["run_plan"]["requested_tasks"] == ["execute_oled_local_demo"]
    assert [task["task_id"] for task in job["task"]["run_plan"]["tasks"]] == ["execute_oled_local_demo"]
    assert job["task"]["task_options"] == {
        "execute_oled_local_demo": {
            "goal": "Find OLED emitters",
            "input_bundle": str(tmp_path / "missing-ok.json"),
            "output_dir": str(output_dir),
            "overwrite": True,
            "project_id": "demo-project",
        }
    }


def test_cli_prints_compact_json(capsys, tmp_path: Path) -> None:
    exit_code = main(
        [
            "--queue-root",
            str(tmp_path / "queue"),
            "--project-id",
            "demo-project",
            "--run-id",
            "cli-demo",
            "--input-bundle",
            str(tmp_path / "missing-ok.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--goal",
            "Find OLED emitters",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["queue_task_id"] == "run_plan_execute"
    assert payload["queue_task_kind"] == "run_plan_execute"
    assert payload["run_plan_tasks"] == ["execute_oled_local_demo"]
    assert payload["enqueued"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "scenarios" not in payload


def test_missing_input_bundle_path_is_allowed_and_not_read(tmp_path: Path) -> None:
    missing_bundle = tmp_path / "missing.json"
    output_dir = tmp_path / "out"

    result = _enqueue(tmp_path, input_bundle=missing_bundle, output_dir=output_dir)

    assert result["input_bundle"] == str(missing_bundle)
    assert result["job_status"] == "queued"
    assert not output_dir.exists()
    assert not (tmp_path / "projects").exists()
    assert ProjectStorage(tmp_path / "projects").read_stage_state("demo-project", "oled-generic-queue-demo") is None
    assert not output_dir.exists()


def test_enqueue_command_does_not_read_input_bundle(monkeypatch, tmp_path: Path) -> None:
    bundle_path = write_local_input_bundle_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = _enqueue(tmp_path, input_bundle=bundle_path, output_dir=output_dir)

    assert result["ok"] is True
    assert str(bundle_path) not in opened_for_read
    assert not output_dir.exists()


def test_enqueued_generic_job_can_later_be_consumed_by_worker_loop(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    bundle_path = write_local_input_bundle_template(tmp_path / "oled_demo_bundle.json")
    _enqueue(tmp_path, queue_root=queue_root, input_bundle=bundle_path, output_dir=output_dir)

    result = _run_worker_loop(tmp_path, queue_root=queue_root, project_root=project_root)

    assert result["actions"] == ["completed", "idle"]
    assert result["executed_tasks"] == ["execute_oled_local_demo"]
    job = _queue(queue_root).status("job-demo-project-oled-generic-queue-demo")
    assert job is not None
    assert job["status"] == "succeeded"
    registry = ProjectStorage(project_root).read_artifact_registry("demo-project", "oled-generic-queue-demo")
    assert list(registry) == EXPECTED_ARTIFACTS


def test_existing_outputs_with_overwrite_false_fail_in_worker_loop_phase(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    bundle_path = write_local_input_bundle_template(tmp_path / "oled_demo_bundle.json")

    enqueue_result = _enqueue(tmp_path, queue_root=queue_root, input_bundle=bundle_path, output_dir=output_dir)
    loop_result = _run_worker_loop(tmp_path, queue_root=queue_root)

    assert enqueue_result["job_status"] == "queued"
    assert loop_result["actions"] == ["failed", "idle"]
    failed = _queue(queue_root).status("job-demo-project-oled-generic-queue-demo")
    assert failed is not None
    assert failed["error"] == {"reason": "local_demo_output_exists:oled_agent_mvp_demo_bundle.json"}


def test_existing_outputs_with_overwrite_true_succeeds_in_worker_loop_phase(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    bundle_path = write_local_input_bundle_template(tmp_path / "oled_demo_bundle.json")

    enqueue_result = _enqueue(
        tmp_path,
        queue_root=queue_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=True,
    )
    loop_result = _run_worker_loop(tmp_path, queue_root=queue_root)

    assert enqueue_result["job_status"] == "queued"
    assert loop_result["completed_jobs"] == ["job-demo-project-oled-generic-queue-demo"]
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["source"] == "local_input_bundle"


def test_generic_enqueue_module_safety_guards() -> None:
    source = inspect.getsource(generic_enqueue)
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

    assert "execute_oled_local_demo_runplan" not in source
    assert "poll_once" not in source
    assert "resume_after_gate" not in source
    assert "enqueue_retry_of_failed_job" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in generic_enqueue.__name__
    assert "receipt" not in generic_enqueue.__name__
    assert "preflight" not in generic_enqueue.__name__
    assert "writer" not in generic_enqueue.__name__
