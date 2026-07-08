from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import ai4s_agent.agents.oled_local_demo_status as oled_local_demo_status
from ai4s_agent.agents.oled_local_demo_cancel import cancel_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_enqueue import enqueue_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_retry import retry_failed_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_status import inspect_oled_local_demo_worker_jobs, main
from ai4s_agent.agents.oled_local_demo_worker_loop import run_oled_local_demo_worker_loop
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"
EXPECTED_ARTIFACTS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _enqueue_oled_job(
    *,
    queue_root: Path,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    project_id: str = "demo-project",
    run_id: str = "oled-local-demo",
    overwrite: bool = False,
) -> dict[str, object]:
    return enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=project_root,
        project_id=project_id,
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        goal="Find OLED emitters with high PLQY and red-shifted emission",
        overwrite=overwrite,
        created_at="2026-01-01T00:00:00Z",
    )


def _stale_output_failure(
    tmp_path: Path,
    *,
    queue_root: Path,
    project_root: Path,
    run_id: str = "failed-demo",
) -> dict[str, object]:
    output_dir = tmp_path / f"{run_id}-out"
    output_dir.mkdir(parents=True)
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    enqueued = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=_write_template(tmp_path / f"{run_id}-bundle.json"),
        output_dir=output_dir,
        run_id=run_id,
        overwrite=False,
    )
    loop = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id=f"{run_id}-worker",
        max_iterations=2,
        target_run_id=run_id,
    )
    assert loop["actions"] == ["failed", "idle"]
    return enqueued


def test_status_lists_queued_oled_jobs_and_ignores_non_oled_jobs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    queued = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    queue.enqueue("demo-project", "other-run", {"task_id": "other_task", "input_bundle": "ignored.json"})

    result = inspect_oled_local_demo_worker_jobs(queue_root=queue_root)

    assert result == {
        "executed": False,
        "executable": False,
        "filters": {"job_id": "", "project_id": "", "run_id": "", "status": ""},
        "job_count": 1,
        "jobs": [
            {
                "attempts": 0,
                "cancellation_requested": False,
                "input_bundle": str(tmp_path / "missing.json"),
                "job_id": queued["job_id"],
                "lease": {},
                "lease_id": "",
                "output_dir": str(tmp_path / "out"),
                "overwrite": False,
                "project_id": "demo-project",
                "retry_of_job_id": "",
                "retry_request_id": "",
                "retry_root_job_id": "",
                "run_id": "oled-local-demo",
                "status": "queued",
                "task_id": TASK_ID,
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
    _enqueue_oled_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        input_bundle=tmp_path / "missing.json",
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
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload
    assert "executed_tasks" not in payload


def test_status_filters_by_job_project_run_and_status(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    first = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "first.json",
        output_dir=tmp_path / "first-out",
        project_id="project-a",
        run_id="run-a",
    )
    second = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "second.json",
        output_dir=tmp_path / "second-out",
        project_id="project-b",
        run_id="run-b",
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    acquired = queue.acquire("worker-b", target_job_id=str(second["job_id"]))
    assert acquired is not None

    by_job = inspect_oled_local_demo_worker_jobs(queue_root=queue_root, job_id=str(second["job_id"]))
    by_project = inspect_oled_local_demo_worker_jobs(queue_root=queue_root, project_id="project-a")
    by_run = inspect_oled_local_demo_worker_jobs(queue_root=queue_root, run_id="run-b")
    by_status = inspect_oled_local_demo_worker_jobs(queue_root=queue_root, status="running")

    assert [job["job_id"] for job in by_job["jobs"]] == [second["job_id"]]
    assert [job["job_id"] for job in by_project["jobs"]] == [first["job_id"]]
    assert [job["job_id"] for job in by_run["jobs"]] == [second["job_id"]]
    assert [job["job_id"] for job in by_status["jobs"]] == [second["job_id"]]
    assert by_status["status_counts"] == {"running": 1}
    assert by_status["jobs"][0]["lease"]["status"] == "active"
    assert by_status["jobs"][0]["lease"]["worker_id"] == "worker-b"


def test_status_counts_are_deterministic_for_mixed_states(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "queued.json",
        output_dir=tmp_path / "queued-out",
        run_id="queued-demo",
    )
    _stale_output_failure(tmp_path, queue_root=queue_root, project_root=project_root, run_id="failed-demo")
    succeeded = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=_write_template(tmp_path / "succeeded-bundle.json"),
        output_dir=tmp_path / "succeeded-out",
        run_id="succeeded-demo",
    )
    run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="succeeded-worker",
        max_iterations=2,
        target_job_id=str(succeeded["job_id"]),
    )
    cancel_target = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "cancelled.json",
        output_dir=tmp_path / "cancelled-out",
        run_id="cancelled-demo",
    )
    cancel_oled_local_demo_worker_job(queue_root=queue_root, job_id=str(cancel_target["job_id"]))

    result = inspect_oled_local_demo_worker_jobs(queue_root=queue_root)

    assert result["status_counts"] == {
        "cancelled": 1,
        "failed": 1,
        "queued": 1,
        "succeeded": 1,
    }


def test_succeeded_job_includes_project_storage_stage_and_artifact_ids_when_project_root_supplied(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    enqueued = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=_write_template(tmp_path / "bundle.json"),
        output_dir=tmp_path / "out",
        run_id="succeeded-demo",
    )
    run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="worker-a",
        max_iterations=2,
        target_job_id=str(enqueued["job_id"]),
    )

    result = inspect_oled_local_demo_worker_jobs(
        queue_root=queue_root,
        project_root=project_root,
        job_id=str(enqueued["job_id"]),
    )

    assert result["project_root"] == str(project_root)
    assert result["job_count"] == 1
    job = result["jobs"][0]
    assert job["status"] == "succeeded"
    assert job["stage_state"]["stage"] == "execute_oled_local_demo"
    assert job["stage_state"]["status"] == "succeeded"
    assert job["stage_state"]["updated_at"]
    assert job["artifact_ids"] == EXPECTED_ARTIFACTS


def test_failed_retry_and_cancelled_jobs_include_metadata(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    failed = _stale_output_failure(tmp_path, queue_root=queue_root, project_root=project_root, run_id="failed-demo")
    retry = retry_failed_oled_local_demo_worker_job(
        queue_root=queue_root,
        source_job_id=str(failed["job_id"]),
        retry_request_id="retry-001",
        requested_by="benton",
        reason="fixed stale output",
    )
    cancel_target = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "cancel.json",
        output_dir=tmp_path / "cancel-out",
        run_id="cancelled-demo",
    )
    cancel_oled_local_demo_worker_job(queue_root=queue_root, job_id=str(cancel_target["job_id"]))

    result = inspect_oled_local_demo_worker_jobs(queue_root=queue_root)
    by_id = {str(job["job_id"]): job for job in result["jobs"]}

    failed_job = by_id[str(failed["job_id"])]
    assert failed_job["status"] == "failed"
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in failed_job["error"]["reason"]
    retry_job = by_id[str(retry["retry_job_id"])]
    assert retry_job["status"] == "queued"
    assert retry_job["retry_of_job_id"] == failed["job_id"]
    assert retry_job["retry_root_job_id"] == failed["job_id"]
    assert retry_job["retry_request_id"] == "retry-001"
    cancelled_job = by_id[str(cancel_target["job_id"])]
    assert cancelled_job["status"] == "cancelled"
    assert cancelled_job["cancellation_requested"] is True


def test_missing_queue_root_returns_empty_status(tmp_path: Path) -> None:
    queue_root = tmp_path / "missing-queue"

    result = inspect_oled_local_demo_worker_jobs(queue_root=queue_root)

    assert result == {
        "executed": False,
        "executable": False,
        "filters": {"job_id": "", "project_id": "", "run_id": "", "status": ""},
        "job_count": 0,
        "jobs": [],
        "ok": True,
        "project_root": "",
        "queue_root": str(queue_root),
        "status_counts": {},
    }


def test_status_command_does_not_read_bundles_open_artifacts_or_mutate_state(monkeypatch, tmp_path: Path) -> None:
    bundle = local_input_bundle_template()
    forbidden_label = str(tmp_path / "do-not-open-artifact.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_label}
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    enqueued = _enqueue_oled_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
        run_id="status-read-guard",
    )
    run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="worker-a",
        max_iterations=2,
        target_job_id=str(enqueued["job_id"]),
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    before_jobs = queue.list_jobs()
    before_leases = queue.list_leases()
    stage_path = project_root / "projects" / "demo-project" / "runs" / "status-read-guard" / "stage.json"
    registry_path = project_root / "projects" / "demo-project" / "runs" / "status-read-guard" / "artifact_registry.json"
    before_stage = stage_path.read_text(encoding="utf-8")
    before_registry = registry_path.read_text(encoding="utf-8")
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = inspect_oled_local_demo_worker_jobs(
        queue_root=queue_root,
        project_root=project_root,
        job_id=str(enqueued["job_id"]),
    )

    assert result["job_count"] == 1
    assert str(bundle_path) not in opened_for_read
    assert forbidden_label not in opened_for_read
    artifact_paths = json.loads(before_registry)["artifacts"].values()
    for artifact_path in artifact_paths:
        assert str(project_root / "projects" / "demo-project" / "runs" / "status-read-guard" / artifact_path) not in opened_for_read
    assert queue.list_jobs() == before_jobs
    assert queue.list_leases() == before_leases
    assert stage_path.read_text(encoding="utf-8") == before_stage
    assert registry_path.read_text(encoding="utf-8") == before_registry


def test_status_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_status)
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
    assert ".cancel(" not in source
    assert ".enqueue(" not in source
    assert "enqueue_retry_of_failed_job" not in source
    assert "resume_after_gate" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert not any(token in source for token in ("RunPlanExecutor", "execute_oled_local_demo_adapter"))
    assert "admission" not in oled_local_demo_status.__name__
    assert "receipt" not in oled_local_demo_status.__name__
    assert "preflight" not in oled_local_demo_status.__name__
    assert "writer" not in oled_local_demo_status.__name__
