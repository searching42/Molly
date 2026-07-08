from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_retry as generic_retry
from ai4s_agent.agents.oled_local_demo_generic_enqueue import enqueue_oled_local_demo_generic_run_plan_job
from ai4s_agent.agents.oled_local_demo_generic_retry import (
    main,
    retry_failed_oled_local_demo_generic_run_plan_job,
)
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
    run_id: str = "oled-generic-retry-demo",
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


def _run_generic_loop(
    *,
    queue_root: Path,
    project_root: Path,
    target_job_id: str | None = None,
    max_iterations: int = 3,
) -> dict:
    return run_oled_local_demo_generic_worker_loop(
        queue_root=queue_root,
        project_root=project_root,
        worker_id="generic-worker-1",
        max_iterations=max_iterations,
        target_job_id=target_job_id,
        now="2026-01-01T00:00:00Z",
    )


def _failed_stale_output_job(tmp_path: Path, *, run_id: str = "oled-generic-retry-demo") -> tuple[Path, Path, dict]:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / f"{run_id}-out"
    output_dir.mkdir()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    source = _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id=run_id,
        input_bundle=_write_template(tmp_path / f"{run_id}-bundle.json"),
        output_dir=output_dir,
        overwrite=False,
    )
    loop = _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(source["job_id"]))
    assert loop["actions"] == ["failed", "idle"]
    failed = _queue(queue_root).status(str(source["job_id"]))
    assert failed is not None
    assert failed["status"] == "failed"
    return queue_root, output_dir, failed


def _retry(queue_root: Path, source_job_id: str, *, retry_request_id: str = "retry-001") -> dict:
    return retry_failed_oled_local_demo_generic_run_plan_job(
        queue_root=queue_root,
        source_job_id=source_job_id,
        retry_request_id=retry_request_id,
        requested_by="benton",
        reason="fixed stale output directory",
        now="2026-01-01T00:10:00Z",
    )


def test_failed_generic_oled_job_can_be_retried(tmp_path: Path) -> None:
    queue_root, _output_dir, source = _failed_stale_output_job(tmp_path)

    result = _retry(queue_root, str(source["job_id"]))

    retry_job = _queue(queue_root).status(result["retry_job_id"])
    original = _queue(queue_root).status(str(source["job_id"]))
    assert result == {
        "enqueued": True,
        "executable": False,
        "executed": False,
        "generic_run_plan_queue": True,
        "ok": True,
        "queue_root": str(queue_root),
        "queue_task_id": "run_plan_execute",
        "queue_task_kind": "run_plan_execute",
        "retry_job_id": "job-demo-project-oled-generic-retry-demo-2",
        "retry_of_job_id": source["job_id"],
        "retry_reason": "fixed stale output directory",
        "retry_request_id": "retry-001",
        "retry_requested_by": "benton",
        "retry_root_job_id": source["job_id"],
        "retry_status": "queued",
        "run_plan_tasks": ["execute_oled_local_demo"],
        "source_job_id": source["job_id"],
        "source_status": "failed",
    }
    assert original is not None
    assert original["status"] == "failed"
    assert retry_job is not None
    assert retry_job["status"] == "queued"
    assert retry_job["task"] == source["task"]
    assert retry_job["retry_of_job_id"] == source["job_id"]
    assert retry_job["retry_root_job_id"] == source["job_id"]
    assert retry_job["retry_request_id"] == "retry-001"
    assert retry_job["retry_reason"] == "fixed stale output directory"
    assert retry_job["retry_requested_by"] == "benton"


def test_cli_creates_retry_child_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root, _output_dir, source = _failed_stale_output_job(tmp_path)

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--source-job-id",
            str(source["job_id"]),
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
    assert payload["retry_status"] == "queued"
    assert payload["run_plan_tasks"] == ["execute_oled_local_demo"]
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload


def test_retry_rejects_missing_non_failed_non_generic_and_disallowed_sources(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = _queue(queue_root)
    queued = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
        run_id="queued-demo",
    )
    non_generic = queue.enqueue("demo-project", "non-generic", {"task_id": "other_task"})
    acquired = queue.acquire("worker-a", target_job_id=str(non_generic["job_id"]))
    assert acquired is not None
    failed_non_generic = queue.fail(str(acquired["lease_id"]), reason="failed")
    disallowed = enqueue_run_plan_execute_job(
        queue,
        project_id="demo-project",
        run_plan=expand_run_plan(run_id="baseline-run", requested_tasks=["run_baseline"]),
        input_artifacts={},
        task_options={},
    )
    acquired = queue.acquire("worker-b", target_job_id=str(disallowed["job_id"]))
    assert acquired is not None
    failed_disallowed = queue.fail(str(acquired["lease_id"]), reason="failed")

    with pytest.raises(KeyError, match="source job not found"):
        _retry(queue_root, "missing-job")
    with pytest.raises(ValueError, match="source job status must be failed"):
        _retry(queue_root, str(queued["job_id"]))
    with pytest.raises(ValueError, match="source job task_id must be run_plan_execute"):
        _retry(queue_root, str(failed_non_generic["job_id"]))
    with pytest.raises(ValueError, match="generic_run_plan_not_allowlisted_for_oled_local_demo"):
        _retry(queue_root, str(failed_disallowed["job_id"]))


def test_retry_rejects_already_retry_child(tmp_path: Path) -> None:
    queue_root, _output_dir, source = _failed_stale_output_job(tmp_path)
    first = _retry(queue_root, str(source["job_id"]))
    acquired = _queue(queue_root).acquire("worker-a", target_job_id=first["retry_job_id"])
    assert acquired is not None
    _queue(queue_root).fail(str(acquired["lease_id"]), reason="retry failed")

    with pytest.raises(ValueError, match="retry child jobs are not eligible for explicit retry"):
        _retry(queue_root, first["retry_job_id"], retry_request_id="retry-002")


def test_duplicate_retry_request_id_semantics_follow_worker_queue(tmp_path: Path) -> None:
    queue_root, _output_dir, first_source = _failed_stale_output_job(tmp_path, run_id="first-run")
    _other_root, _other_output, second_source = _failed_stale_output_job(tmp_path, run_id="second-run")

    first = _retry(queue_root, str(first_source["job_id"]), retry_request_id="retry-001")
    second = _retry(queue_root, str(first_source["job_id"]), retry_request_id="retry-001")

    assert first["retry_job_id"] == second["retry_job_id"]
    with pytest.raises(ValueError, match="retry_request_id already belongs to a different source job"):
        _retry(queue_root, str(second_source["job_id"]), retry_request_id="retry-001")


def test_retry_child_can_later_be_consumed_after_failure_cause_is_fixed(tmp_path: Path) -> None:
    queue_root, output_dir, source = _failed_stale_output_job(tmp_path)
    retry = _retry(queue_root, str(source["job_id"]))
    (output_dir / "oled_agent_mvp_demo_bundle.json").unlink()

    loop = _run_generic_loop(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        target_job_id=retry["retry_job_id"],
    )

    retry_job = _queue(queue_root).status(retry["retry_job_id"])
    assert loop["actions"] == ["completed", "idle"]
    assert loop["executed_tasks"] == ["execute_oled_local_demo"]
    assert retry_job is not None
    assert retry_job["status"] == "succeeded"
    assert (output_dir / "oled_agent_mvp_demo_bundle.json").exists()
    assert (output_dir / "oled_agent_mvp_demo_bundle.md").exists()
    assert (output_dir / "oled_local_demo_execution_manifest.json").exists()


def test_retry_command_does_not_read_bundle_create_outputs_or_storage(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "manual-out"
    bundle_path = _write_template(tmp_path / "bundle.json")
    source = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="manual-failed-run",
    )
    acquired = _queue(queue_root).acquire("worker-a", target_job_id=str(source["job_id"]))
    assert acquired is not None
    failed = _queue(queue_root).fail(str(acquired["lease_id"]), reason="manual failure")
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = _retry(queue_root, str(failed["job_id"]))

    retry_job = _queue(queue_root).status(result["retry_job_id"])
    assert retry_job is not None
    assert retry_job["task"] == failed["task"]
    assert str(bundle_path) not in opened_for_read
    assert not output_dir.exists()
    assert not (tmp_path / "projects").exists()
    assert not list(tmp_path.glob("**/*adapter_result*.json"))


def test_generic_retry_module_safety_guards() -> None:
    source = inspect.getsource(generic_retry)
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
    assert "queue.enqueue(" not in source
    assert "queue.cancel" not in source
    assert "resume_after_gate" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in generic_retry.__name__
    assert "receipt" not in generic_retry.__name__
    assert "preflight" not in generic_retry.__name__
    assert "writer" not in generic_retry.__name__
