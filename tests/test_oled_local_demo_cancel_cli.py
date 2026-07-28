from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_cancel as oled_local_demo_cancel
from ai4s_agent.agents.oled_local_demo_cancel import cancel_oled_local_demo_worker_job, main
from ai4s_agent.agents.oled_local_demo_enqueue import enqueue_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_worker_loop import run_oled_local_demo_worker_loop
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def _enqueue_oled_job(
    *,
    queue_root: Path,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    run_id: str = "oled-local-demo",
    overwrite: bool = False,
) -> dict[str, object]:
    return enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        goal="Find OLED emitters with high PLQY and red-shifted emission",
        overwrite=overwrite,
        created_at="2026-01-01T00:00:00Z",
    )


def test_queued_oled_local_demo_worker_job_can_be_cancelled(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    enqueue_result = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "missing_bundle_is_not_read.json",
        output_dir=output_dir,
    )

    result = cancel_oled_local_demo_worker_job(
        queue_root=queue_root,
        job_id=str(enqueue_result["job_id"]),
        now="2026-01-01T00:10:00Z",
    )

    assert result == {
        "cancelled": True,
        "cancellation_requested": True,
        "executed": False,
        "executable": False,
        "job_id": "job-demo-project-oled-local-demo",
        "job_status": "cancelled",
        "ok": True,
        "previous_status": "queued",
        "queue_root": str(queue_root),
        "task_id": TASK_ID,
    }
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    job = queue.status(str(enqueue_result["job_id"]))
    assert job is not None
    assert job["status"] == "cancelled"
    assert job["cancellation_requested"] is True
    assert job["updated_at"] == "2026-01-01T00:10:00Z"
    assert job["task"] == {
        "task_id": TASK_ID,
        "project_root": str(project_root),
        "input_bundle": str(tmp_path / "missing_bundle_is_not_read.json"),
        "output_dir": str(output_dir),
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "overwrite": False,
    }
    assert not project_root.exists()
    assert not output_dir.exists()


def test_cancel_cli_cancels_queued_job_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    enqueue_result = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        input_bundle=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--job-id",
            str(enqueue_result["job_id"]),
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
    assert payload["previous_status"] == "queued"
    assert payload["task_id"] == TASK_ID
    assert payload["cancellation_requested"] is True
    assert payload["cancelled"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload
    assert "executed_tasks" not in payload
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).status(str(enqueue_result["job_id"]))["status"] == "cancelled"  # type: ignore[index]


def test_cancel_rejects_wrong_task_and_missing_job(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    wrong = queue.enqueue("demo-project", "wrong-task", {"task_id": "wrong_task"})

    with pytest.raises(ValueError, match="unsupported_oled_local_demo_cancel_task:wrong_task"):
        cancel_oled_local_demo_worker_job(queue_root=queue_root, job_id=wrong["job_id"])

    with pytest.raises(KeyError, match="job not found"):
        cancel_oled_local_demo_worker_job(queue_root=queue_root, job_id="job-missing")


def test_running_oled_job_can_be_cancellation_requested_and_worker_loop_handles_it(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    enqueue_result = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        input_bundle=tmp_path / "missing_bundle_is_not_read.json",
        output_dir=output_dir,
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    acquired = queue.acquire("local-worker-1", target_job_id=str(enqueue_result["job_id"]))
    assert acquired is not None
    assert acquired["status"] == "running"

    result = cancel_oled_local_demo_worker_job(
        queue_root=queue_root,
        job_id=str(enqueue_result["job_id"]),
        now="2026-01-01T00:10:00Z",
    )

    assert result["previous_status"] == "running"
    assert result["job_status"] == "running"
    assert result["cancellation_requested"] is True
    assert result["cancelled"] is True
    running_job = queue.status(str(enqueue_result["job_id"]))
    assert running_job is not None
    assert running_job["status"] == "running"
    assert running_job["cancellation_requested"] is True

    loop_result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=2,
        now="2026-01-01T00:20:00Z",
    )

    assert loop_result["actions"] == ["cancelled", "idle"]
    assert loop_result["cancelled_jobs"] == [enqueue_result["job_id"]]
    assert loop_result["completed_jobs"] == []
    assert loop_result["failed_jobs"] == []
    finished = queue.status(str(enqueue_result["job_id"]))
    assert finished is not None
    assert finished["status"] == "failed"
    assert finished["error"]["reason"] == "cancelled"
    lease = queue.lease_status(str(finished["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"
    assert not output_dir.exists()


def test_cancel_command_does_not_read_bundle_create_outputs_or_modify_task_payload(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    forbidden_bundle = tmp_path / "do-not-open-bundle.json"
    enqueue_result = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=forbidden_bundle,
        output_dir=output_dir,
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    original_task = dict(queue.status(str(enqueue_result["job_id"]))["task"])  # type: ignore[index]
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = cancel_oled_local_demo_worker_job(
        queue_root=queue_root,
        job_id=str(enqueue_result["job_id"]),
    )

    assert result["job_status"] == "cancelled"
    assert str(forbidden_bundle) not in opened_for_read
    assert queue.status(str(enqueue_result["job_id"]))["task"] == original_task  # type: ignore[index]
    assert not output_dir.exists()
    assert not project_root.exists()
    assert not (project_root / "demo-project").exists()


def test_cancel_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_cancel)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "RunPlanExecutor",
        "ai4s_agent.adapters",
        "ai4s_agent.local_worker_loop",
        "ai4s_agent.worker_queue_poller",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
    )

    assert "poll_once" not in source
    assert "LocalWorkerLoop" not in source
    assert "resume_after_gate" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert not any(token in source for token in ("RunPlanExecutor", "execute_oled_local_demo_adapter"))
    assert "admission" not in oled_local_demo_cancel.__name__
    assert "receipt" not in oled_local_demo_cancel.__name__
    assert "preflight" not in oled_local_demo_cancel.__name__
    assert "writer" not in oled_local_demo_cancel.__name__
