from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_retry as oled_local_demo_retry
from ai4s_agent.agents.oled_local_demo_enqueue import enqueue_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_retry import main, retry_failed_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_worker_loop import run_oled_local_demo_worker_loop
from ai4s_agent.agents.oled_mvp_demo import write_local_input_bundle_template
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"
EXPECTED_OUTPUTS = [
    "oled_agent_mvp_demo_bundle.json",
    "oled_agent_mvp_demo_bundle.md",
    "oled_local_demo_execution_manifest.json",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _task(*, project_root: Path, input_bundle: Path, output_dir: Path, overwrite: bool = False) -> dict:
    return {
        "task_id": TASK_ID,
        "project_root": str(project_root),
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "overwrite": overwrite,
    }


def _failed_oled_job(
    tmp_path: Path,
    *,
    queue_root: Path | None = None,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    run_id: str = "oled-local-demo",
) -> tuple[WorkerQueue, dict, Path, Path, Path]:
    queue_root = queue_root or tmp_path / "queue"
    project_root = project_root or tmp_path / "projects"
    output_dir = output_dir or tmp_path / "out"
    bundle_path = _write_template(tmp_path / f"{run_id}_bundle.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    enqueue_result = enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id=run_id,
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=False,
    )
    loop_result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id=f"worker-{run_id}",
        max_iterations=2,
        target_run_id=run_id,
    )
    assert loop_result["actions"] == ["failed", "idle"]
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    source = queue.status(enqueue_result["job_id"])
    assert source is not None
    assert source["status"] == "failed"
    return queue, source, bundle_path, project_root, output_dir


def _write_job_field(queue: WorkerQueue, job_id: str, **changes: object) -> None:
    payload = queue.store.read_queue()
    for job in payload["jobs"]:
        if str(job.get("job_id") or "") == job_id:
            job.update(changes)
            break
    else:
        raise AssertionError(f"job not found: {job_id}")
    queue.store.write_queue(payload)


def test_failed_oled_local_demo_worker_job_can_be_retried(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue, source, _bundle_path, _project_root, _output_dir = _failed_oled_job(tmp_path, queue_root=queue_root)
    source_task = dict(source["task"])

    result = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="fixed stale output directory",
        now="2026-01-01T00:10:00Z",
    )

    assert result == {
        "enqueued": True,
        "executed": False,
        "executable": False,
        "ok": True,
        "queue_root": str(queue_root),
        "retry_job_id": "job-demo-project-oled-local-demo-2",
        "retry_of_job_id": source["job_id"],
        "retry_reason": "fixed stale output directory",
        "retry_request_id": "retry-001",
        "retry_requested_by": "benton",
        "retry_status": "queued",
        "source_job_id": source["job_id"],
        "source_status": "failed",
        "task_id": TASK_ID,
    }
    jobs = queue.list_jobs()
    assert len(jobs) == 2
    persisted_source = queue.status(source["job_id"])
    retry = queue.status(result["retry_job_id"])
    assert persisted_source is not None
    assert retry is not None
    assert persisted_source["status"] == "failed"
    assert retry["status"] == "queued"
    assert retry["task"] == source_task
    assert retry["retry_of_job_id"] == source["job_id"]
    assert retry["retry_root_job_id"] == source["job_id"]
    assert retry["retry_request_id"] == "retry-001"
    assert retry["retry_reason"] == "fixed stale output directory"
    assert retry["retry_requested_by"] == "benton"
    assert retry["created_at"] == "2026-01-01T00:10:00Z"


def test_retry_cli_creates_retry_child_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _queue, source, _bundle_path, _project_root, _output_dir = _failed_oled_job(tmp_path, queue_root=queue_root)

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--source-job-id",
            source["job_id"],
            "--retry-request-id",
            "retry-001",
            "--requested-by",
            "benton",
            "--reason",
            "fixed stale output directory",
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
    assert payload["source_status"] == "failed"
    assert payload["retry_status"] == "queued"
    assert payload["retry_of_job_id"] == source["job_id"]
    assert payload["task_id"] == TASK_ID
    assert payload["enqueued"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload
    assert "executed_tasks" not in payload


def test_retry_rejects_non_failed_missing_wrong_task_and_retry_child_sources(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    queued = queue.enqueue(
        "demo-project",
        "queued-demo",
        _task(project_root=tmp_path / "projects", input_bundle=tmp_path / "missing.json", output_dir=tmp_path / "out"),
    )

    with pytest.raises(ValueError, match="source job status must be failed"):
        retry_failed_oled_local_demo_worker_job(
            queue_root=queue_root,
            source_job_id=queued["job_id"],
            retry_request_id="retry-queued",
            requested_by="benton",
            reason="retry queued",
        )
    with pytest.raises(KeyError, match="source job not found"):
        retry_failed_oled_local_demo_worker_job(
            queue_root=queue_root,
            source_job_id="job-missing",
            retry_request_id="retry-missing",
            requested_by="benton",
            reason="retry missing",
        )

    wrong = queue.enqueue("demo-project", "wrong-task", {"task_id": "wrong_task"})
    acquired = queue.acquire("worker-a", target_job_id=wrong["job_id"])
    assert acquired is not None
    queue.fail(acquired["lease_id"], reason="wrong task failed")
    with pytest.raises(ValueError, match="unsupported_oled_local_demo_retry_task:wrong_task"):
        retry_failed_oled_local_demo_worker_job(
            queue_root=queue_root,
            source_job_id=wrong["job_id"],
            retry_request_id="retry-wrong",
            requested_by="benton",
            reason="retry wrong task",
        )

    _queue, source, _bundle_path, _project_root, _output_dir = _failed_oled_job(tmp_path, queue_root=queue_root, run_id="failed-demo")
    retry = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="retry source",
    )
    retry_job = queue.status(retry["retry_job_id"])
    assert retry_job is not None
    _write_job_field(queue, retry_job["job_id"], status="failed")
    with pytest.raises(ValueError, match="retry child jobs are not eligible for explicit retry"):
        retry_failed_oled_local_demo_worker_job(
            queue_root=queue_root,
            source_job_id=retry_job["job_id"],
            retry_request_id="retry-child",
            requested_by="benton",
            reason="retry child",
        )


def test_duplicate_retry_request_id_idempotency_and_cross_source_rejection(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _queue, first_source, _bundle_path, _project_root, first_output = _failed_oled_job(
        tmp_path,
        queue_root=queue_root,
        output_dir=tmp_path / "first-out",
        run_id="first-demo",
    )
    _queue, second_source, _bundle_path, _project_root, _second_output = _failed_oled_job(
        tmp_path,
        queue_root=queue_root,
        output_dir=tmp_path / "second-out",
        run_id="second-demo",
    )

    first = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=first_source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="retry first",
    )
    second = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=first_source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="retry first again",
    )
    assert first["retry_job_id"] == second["retry_job_id"]
    assert first["retry_reason"] == second["retry_reason"] == "retry first"

    with pytest.raises(ValueError, match="retry_request_id already belongs to a different source job"):
        retry_failed_oled_local_demo_worker_job(
            queue_root=queue_root,
            source_job_id=second_source["job_id"],
            retry_request_id="retry-001",
            requested_by="benton",
            reason="retry second",
        )

    assert (first_output / "oled_agent_mvp_demo_bundle.json").exists()


def test_retry_child_can_be_consumed_after_original_failure_is_fixed(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _queue, source, _bundle_path, _project_root, output_dir = _failed_oled_job(tmp_path, queue_root=queue_root)
    retry = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="removed stale output",
    )
    (output_dir / "oled_agent_mvp_demo_bundle.json").unlink()

    loop_result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="retry-worker",
        max_iterations=3,
        target_job_id=retry["retry_job_id"],
    )

    assert loop_result["actions"] == ["completed", "idle"]
    assert loop_result["completed_jobs"] == [retry["retry_job_id"]]
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).status(retry["retry_job_id"])["status"] == "succeeded"  # type: ignore[index]
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_retry_command_does_not_read_bundle_or_create_outputs(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _queue, source, bundle_path, project_root, output_dir = _failed_oled_job(tmp_path, queue_root=queue_root)
    (output_dir / "oled_agent_mvp_demo_bundle.json").unlink()
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=source["job_id"],
        retry_request_id="retry-001",
        requested_by="benton",
        reason="retry without reading bundle",
    )

    assert result["retry_status"] == "queued"
    assert str(bundle_path) not in opened_for_read
    assert not (output_dir / "oled_agent_mvp_demo_bundle.json").exists()
    assert not (project_root / "demo-project" / source["run_id"] / result["retry_job_id"] / "adapter_result.json").exists()


def test_retry_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_retry)
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
    assert "admission" not in oled_local_demo_retry.__name__
    assert "receipt" not in oled_local_demo_retry.__name__
    assert "preflight" not in oled_local_demo_retry.__name__
    assert "writer" not in oled_local_demo_retry.__name__
