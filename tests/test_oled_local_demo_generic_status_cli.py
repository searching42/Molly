from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_status as generic_status
from ai4s_agent.agents.oled_local_demo_generic_enqueue import enqueue_oled_local_demo_generic_run_plan_job
from ai4s_agent.agents.oled_local_demo_generic_status import (
    inspect_oled_local_demo_generic_run_plan_jobs,
    main,
)
from ai4s_agent.agents.oled_local_demo_generic_worker_loop import run_oled_local_demo_generic_worker_loop
from ai4s_agent.agents.oled_mvp_demo import write_local_input_bundle_template
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


EXPECTED_ARTIFACTS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _enqueue_generic_oled_job(
    *,
    queue_root: Path,
    project_id: str = "demo-project",
    run_id: str = "oled-generic-status-demo",
    input_bundle: Path,
    output_dir: Path,
    goal: str | None = None,
    overwrite: bool = False,
) -> dict:
    return enqueue_oled_local_demo_generic_run_plan_job(
        queue_root=queue_root,
        project_id=project_id,
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        goal=goal,
        overwrite=overwrite,
    )


def _run_generic_loop(
    *,
    queue_root: Path,
    project_root: Path,
    worker_id: str = "generic-worker-1",
    target_job_id: str | None = None,
    max_iterations: int = 3,
) -> dict:
    return run_oled_local_demo_generic_worker_loop(
        queue_root=queue_root,
        project_root=project_root,
        worker_id=worker_id,
        max_iterations=max_iterations,
        target_job_id=target_job_id,
        now="2026-01-01T00:00:00Z",
    )


def test_status_lists_queued_generic_oled_jobs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queued = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=tmp_path / "out",
    )

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root)

    assert result == {
        "allowlist": ["execute_oled_local_demo"],
        "executed": False,
        "executable": False,
        "filters": {"job_id": "", "project_id": "", "run_id": "", "status": ""},
        "generic_run_plan_queue": True,
        "job_count": 1,
        "jobs": [
            {
                "attempts": 0,
                "cancellation_requested": False,
                "input_bundle": str(tmp_path / "missing-ok.json"),
                "job_id": queued["job_id"],
                "lease": {},
                "lease_id": "",
                "output_dir": str(tmp_path / "out"),
                "overwrite": False,
                "project_id": "demo-project",
                "queue_task_id": "run_plan_execute",
                "queue_task_kind": "run_plan_execute",
                "requested_tasks": ["execute_oled_local_demo"],
                "retry_of_job_id": "",
                "retry_request_id": "",
                "retry_root_job_id": "",
                "run_id": "oled-generic-status-demo",
                "run_plan_tasks": ["execute_oled_local_demo"],
                "status": "queued",
                "worker_id": "",
            }
        ],
        "ok": True,
        "project_root": "",
        "queue_root": str(queue_root),
        "status_counts": {"queued": 1},
    }


def test_status_cli_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=tmp_path / "out",
    )

    exit_code = main(["--queue-root", str(queue_root)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["job_count"] == 1
    assert payload["status_counts"] == {"queued": 1}
    assert payload["generic_run_plan_queue"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload


def test_status_filters_by_job_project_run_and_status(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    first = _enqueue_generic_oled_job(
        queue_root=queue_root,
        project_id="project-a",
        run_id="run-a",
        input_bundle=tmp_path / "first.json",
        output_dir=tmp_path / "first-out",
    )
    second = _enqueue_generic_oled_job(
        queue_root=queue_root,
        project_id="project-b",
        run_id="run-b",
        input_bundle=tmp_path / "second.json",
        output_dir=tmp_path / "second-out",
    )
    acquired = _queue(queue_root).acquire("worker-b", target_job_id=str(second["job_id"]))
    assert acquired is not None

    by_job = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, job_id=str(second["job_id"]))
    by_project = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, project_id="project-a")
    by_run = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, run_id="run-b")
    by_status = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, status="running")

    assert [job["job_id"] for job in by_job["jobs"]] == [second["job_id"]]
    assert [job["job_id"] for job in by_project["jobs"]] == [first["job_id"]]
    assert [job["job_id"] for job in by_run["jobs"]] == [second["job_id"]]
    assert [job["job_id"] for job in by_status["jobs"]] == [second["job_id"]]
    assert by_status["status_counts"] == {"running": 1}
    assert by_status["jobs"][0]["lease"]["status"] == "active"
    assert by_status["jobs"][0]["lease"]["worker_id"] == "worker-b"


def test_non_generic_and_disallowed_generic_jobs_are_ignored(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=tmp_path / "missing-ok.json",
        output_dir=tmp_path / "out",
    )
    queue = _queue(queue_root)
    queue.enqueue("demo-project", "specific-envelope", {"task_id": "execute_oled_local_demo_runplan"})
    enqueue_run_plan_execute_job(
        queue,
        project_id="demo-project",
        run_plan=expand_run_plan(run_id="baseline-run", requested_tasks=["run_baseline"]),
        input_artifacts={},
        task_options={},
    )

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root)

    assert result["job_count"] == 1
    assert result["jobs"][0]["run_id"] == "oled-generic-status-demo"
    assert result["jobs"][0]["queue_task_id"] == "run_plan_execute"


def test_status_counts_are_deterministic_for_mixed_states(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id="queued-demo",
        input_bundle=tmp_path / "queued.json",
        output_dir=tmp_path / "queued-out",
    )
    failed_out = tmp_path / "failed-out"
    failed_out.mkdir()
    (failed_out / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    failed = _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id="failed-demo",
        input_bundle=_write_template(tmp_path / "failed-bundle.json"),
        output_dir=failed_out,
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(failed["job_id"]))
    succeeded = _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id="succeeded-demo",
        input_bundle=_write_template(tmp_path / "succeeded-bundle.json"),
        output_dir=tmp_path / "succeeded-out",
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(succeeded["job_id"]))
    cancelled = _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id="cancelled-demo",
        input_bundle=tmp_path / "cancelled.json",
        output_dir=tmp_path / "cancelled-out",
    )
    _queue(queue_root).cancel(str(cancelled["job_id"]))

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root)

    assert result["status_counts"] == {
        "cancelled": 1,
        "failed": 1,
        "queued": 1,
        "succeeded": 1,
    }


def test_succeeded_job_includes_execution_and_project_storage_metadata(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    enqueued = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=_write_template(tmp_path / "bundle.json"),
        output_dir=output_dir,
        run_id="succeeded-demo",
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(enqueued["job_id"]))

    result = inspect_oled_local_demo_generic_run_plan_jobs(
        queue_root=queue_root,
        project_root=project_root,
        job_id=str(enqueued["job_id"]),
    )

    assert result["project_root"] == str(project_root)
    job = result["jobs"][0]
    assert job["status"] == "succeeded"
    assert job["executed_tasks"] == ["execute_oled_local_demo"]
    assert job["stage_state"]["stage"] == "execute_oled_local_demo"
    assert job["stage_state"]["status"] == "succeeded"
    assert job["artifact_ids"] == EXPECTED_ARTIFACTS


def test_failed_cancelled_and_retry_metadata_are_preserved(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "failed-out"
    output_dir.mkdir()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    failed = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=_write_template(tmp_path / "bundle.json"),
        output_dir=output_dir,
        run_id="failed-demo",
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(failed["job_id"]))
    retry = _queue(queue_root).enqueue_retry_of_failed_job(
        str(failed["job_id"]),
        retry_request_id="retry-001",
        requested_by="benton",
        reason="fixed stale output directory",
    )
    cancelled = _enqueue_generic_oled_job(
        queue_root=queue_root,
        run_id="cancelled-demo",
        input_bundle=tmp_path / "cancelled.json",
        output_dir=tmp_path / "cancelled-out",
    )
    _queue(queue_root).cancel(str(cancelled["job_id"]))

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root)

    by_id = {job["job_id"]: job for job in result["jobs"]}
    assert by_id[str(failed["job_id"])]["error"] == {
        "reason": "local_demo_output_exists:oled_agent_mvp_demo_bundle.json"
    }
    assert by_id[str(cancelled["job_id"])]["cancellation_requested"] is True
    assert by_id[str(cancelled["job_id"])]["status"] == "cancelled"
    assert by_id[str(retry["job_id"])]["retry_of_job_id"] == failed["job_id"]
    assert by_id[str(retry["job_id"])]["retry_root_job_id"] == failed["job_id"]
    assert by_id[str(retry["job_id"])]["retry_request_id"] == "retry-001"


def test_missing_queue_root_returns_empty_status(tmp_path: Path) -> None:
    queue_root = tmp_path / "missing-queue"

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root)

    assert result["job_count"] == 0
    assert result["jobs"] == []
    assert result["status_counts"] == {}
    assert result["executed"] is False
    assert result["executable"] is False


def test_status_does_not_read_input_bundle_or_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    bundle_path = _write_template(tmp_path / "bundle.json")
    enqueued = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="succeeded-demo",
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(enqueued["job_id"]))
    artifact_paths = [
        output_dir / "oled_agent_mvp_demo_bundle.json",
        output_dir / "oled_agent_mvp_demo_bundle.md",
        output_dir / "oled_local_demo_execution_manifest.json",
    ]
    opened_for_read: list[str] = []
    real_open = Path.open

    def tracking_open(self: Path, mode: str = "r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    result = inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, project_root=project_root)

    assert result["job_count"] == 1
    assert str(bundle_path) not in opened_for_read
    for artifact_path in artifact_paths:
        assert str(artifact_path) not in opened_for_read


def test_status_does_not_mutate_queue_or_project_storage_files(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    enqueued = _enqueue_generic_oled_job(
        queue_root=queue_root,
        input_bundle=_write_template(tmp_path / "bundle.json"),
        output_dir=tmp_path / "out",
        run_id="succeeded-demo",
    )
    _run_generic_loop(queue_root=queue_root, project_root=project_root, target_job_id=str(enqueued["job_id"]))
    queue = _queue(queue_root)
    before_queue = queue.store.queue_path.read_text(encoding="utf-8")
    before_leases = queue.store.leases_path.read_text(encoding="utf-8")
    stage_path = project_root / "projects" / "demo-project" / "runs" / "succeeded-demo" / "stage.json"
    registry_path = project_root / "projects" / "demo-project" / "runs" / "succeeded-demo" / "artifact_registry.json"
    before_stage = stage_path.read_text(encoding="utf-8")
    before_registry = registry_path.read_text(encoding="utf-8")

    inspect_oled_local_demo_generic_run_plan_jobs(queue_root=queue_root, project_root=project_root)

    assert queue.store.queue_path.read_text(encoding="utf-8") == before_queue
    assert queue.store.leases_path.read_text(encoding="utf-8") == before_leases
    assert stage_path.read_text(encoding="utf-8") == before_stage
    assert registry_path.read_text(encoding="utf-8") == before_registry


def test_generic_status_module_safety_guards() -> None:
    source = inspect.getsource(generic_status)
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
    assert "queue.cancel" not in source
    assert "enqueue_retry_of_failed_job" not in source
    assert "resume_after_gate" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in generic_status.__name__
    assert "receipt" not in generic_status.__name__
    assert "preflight" not in generic_status.__name__
    assert "writer" not in generic_status.__name__
