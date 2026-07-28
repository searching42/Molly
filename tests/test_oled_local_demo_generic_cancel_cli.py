from __future__ import annotations

import ast
import builtins
import copy
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_cancel as generic_cancel
from ai4s_agent.agents.oled_local_demo_generic_cancel import (
    cancel_oled_local_demo_generic_run_plan_job,
    main,
)
from ai4s_agent.agents.oled_local_demo_generic_enqueue import enqueue_oled_local_demo_generic_run_plan_job
from ai4s_agent.agents.oled_local_demo_generic_worker_loop import run_oled_local_demo_generic_worker_loop
from ai4s_agent.agents.oled_mvp_demo import write_local_input_bundle_template
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _enqueue_generic_oled_job(
    *,
    queue_root: Path,
    project_id: str = "demo-project",
    run_id: str = "oled-generic-cancel-demo",
    input_bundle: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    return enqueue_oled_local_demo_generic_run_plan_job(
        queue_root=queue_root,
        project_id=project_id,
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _cancel(queue_root: Path, job_id: str) -> dict:
    return cancel_oled_local_demo_generic_run_plan_job(
        queue_root=queue_root,
        job_id=job_id,
        now="2026-01-01T00:10:00Z",
    )


def test_queued_generic_oled_job_can_be_cancelled(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    job = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=tmp_path / "out",
    )
    original_task = copy.deepcopy(_queue(queue_root).status(str(job["job_id"]))["task"])

    result = _cancel(queue_root, str(job["job_id"]))

    cancelled = _queue(queue_root).status(str(job["job_id"]))
    assert result == {
        "cancelled": True,
        "cancellation_requested": True,
        "executable": False,
        "executed": False,
        "generic_run_plan_queue": True,
        "job_id": job["job_id"],
        "job_status": "cancelled",
        "ok": True,
        "previous_status": "queued",
        "queue_root": str(queue_root),
        "queue_task_id": "run_plan_execute",
        "queue_task_kind": "run_plan_execute",
        "run_plan_tasks": ["execute_oled_local_demo"],
    }
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancellation_requested"] is True
    assert cancelled["task"] == original_task


def test_cli_cancels_queued_job_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    job = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=tmp_path / "out",
    )

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--job-id",
            str(job["job_id"]),
            "--now",
            "2026-01-01T00:10:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["job_status"] == "cancelled"
    assert payload["cancellation_requested"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload


def test_running_generic_oled_job_can_be_cancellation_requested_and_worker_loop_handles_it(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    job = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=_write_template(tmp_path / "bundle.json"),
        output_dir=tmp_path / "out",
    )
    acquired = _queue(queue_root).acquire("generic-worker-1", target_job_id=str(job["job_id"]))
    assert acquired is not None

    result = _cancel(queue_root, str(job["job_id"]))
    loop = run_oled_local_demo_generic_worker_loop(
        queue_root=queue_root,
        project_root=project_root,
        worker_id="generic-worker-1",
        max_iterations=2,
        now="2026-01-01T00:11:00Z",
    )

    cancelled = _queue(queue_root).status(str(job["job_id"]))
    lease = _queue(queue_root).lease_status(str(acquired["lease_id"]))
    assert result["previous_status"] == "running"
    assert result["job_status"] == "running"
    assert result["cancellation_requested"] is True
    assert loop["actions"] == ["cancelled", "idle"]
    assert loop["cancelled_jobs"] == [job["job_id"]]
    assert cancelled is not None
    assert cancelled["status"] == "failed"
    assert cancelled["error"] == {"reason": "cancelled"}
    assert lease is not None
    assert lease["status"] == "failed"
    assert not (tmp_path / "out").exists()


def test_cancel_rejects_missing_non_generic_specific_and_disallowed_jobs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = _queue(queue_root)
    non_generic = queue.enqueue("demo-project", "non-generic", {"task_id": "other_task"})
    oled_specific = queue.enqueue("demo-project", "specific", {"task_id": "execute_oled_local_demo_runplan"})
    disallowed = enqueue_run_plan_execute_job(
        queue,
        project_id="demo-project",
        run_plan=expand_run_plan(run_id="baseline-run", requested_tasks=["run_baseline"]),
        input_artifacts={},
        task_options={},
    )

    with pytest.raises(KeyError, match="job not found"):
        _cancel(queue_root, "missing-job")
    with pytest.raises(ValueError, match="job task_id must be run_plan_execute"):
        _cancel(queue_root, str(non_generic["job_id"]))
    with pytest.raises(ValueError, match="job task_id must be run_plan_execute"):
        _cancel(queue_root, str(oled_specific["job_id"]))
    with pytest.raises(ValueError, match="generic_run_plan_not_allowlisted_for_oled_local_demo"):
        _cancel(queue_root, str(disallowed["job_id"]))


def test_cancel_command_does_not_read_bundle_create_outputs_or_storage(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    bundle_path = _write_template(tmp_path / "bundle.json")
    job = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
    )
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = _cancel(queue_root, str(job["job_id"]))

    assert result["job_status"] == "cancelled"
    assert str(bundle_path) not in opened_for_read
    assert not output_dir.exists()
    assert not (tmp_path / "projects").exists()
    assert not list(tmp_path.glob("**/*adapter_result*.json"))


def test_generic_cancel_module_safety_guards() -> None:
    source = inspect.getsource(generic_cancel)
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
    assert "queue.enqueue" not in source
    assert "enqueue_retry_of_failed_job" not in source
    assert "resume_after_gate" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in generic_cancel.__name__
    assert "receipt" not in generic_cancel.__name__
    assert "preflight" not in generic_cancel.__name__
    assert "writer" not in generic_cancel.__name__
