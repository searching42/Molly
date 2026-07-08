from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import ai4s_agent.agents.oled_local_demo_enqueue as oled_local_demo_enqueue
from ai4s_agent.agents.oled_local_demo_enqueue import enqueue_oled_local_demo_worker_job, main
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


def test_enqueue_oled_local_demo_worker_job_writes_one_queued_job(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    input_bundle = tmp_path / "missing_bundle_is_allowed.json"
    output_dir = tmp_path / "out"

    result = enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=input_bundle,
        output_dir=output_dir,
        goal="Find OLED emitters with high PLQY and red-shifted emission",
        overwrite=False,
        created_at="2026-01-01T00:00:00Z",
    )

    assert result == {
        "enqueued": True,
        "executed": False,
        "executable": False,
        "input_bundle": str(input_bundle),
        "job_id": "job-demo-project-oled-local-demo",
        "job_status": "queued",
        "ok": True,
        "output_dir": str(output_dir),
        "overwrite": False,
        "project_id": "demo-project",
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "run_id": "oled-local-demo",
        "task_id": TASK_ID,
    }
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    jobs = queue.list_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["job_id"] == result["job_id"]
    assert job["status"] == "queued"
    assert job["project_id"] == "demo-project"
    assert job["run_id"] == "oled-local-demo"
    assert job["created_at"] == "2026-01-01T00:00:00Z"
    assert job["task"] == {
        "task_id": TASK_ID,
        "project_root": str(project_root),
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "overwrite": False,
    }
    assert not project_root.exists()
    assert not output_dir.exists()


def test_enqueue_cli_prints_compact_json_and_does_not_execute(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--project-root",
            str(tmp_path / "projects"),
            "--project-id",
            "demo-project",
            "--run-id",
            "oled-local-demo",
            "--input-bundle",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(output_dir),
            "--goal",
            "CLI goal",
            "--overwrite",
            "--created-at",
            "2026-01-01T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["job_status"] == "queued"
    assert payload["task_id"] == TASK_ID
    assert payload["enqueued"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert payload["overwrite"] is True
    assert "actions" not in payload
    assert "executed_tasks" not in payload
    assert not output_dir.exists()
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).list_jobs()[0]["task"]["goal"] == "CLI goal"


def test_input_bundle_is_stored_but_not_opened(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    forbidden_bundle = tmp_path / "do-not-open-bundle.json"
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=forbidden_bundle,
        output_dir=tmp_path / "out",
    )

    assert result["input_bundle"] == str(forbidden_bundle)
    assert str(forbidden_bundle) not in opened_for_read
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).list_jobs()[0]["task"]["input_bundle"] == str(forbidden_bundle)


def test_enqueue_only_creates_no_project_storage_outputs_or_adapter_results(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"

    enqueue_oled_local_demo_worker_job(
        queue_root=tmp_path / "queue",
        project_root=project_root,
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=tmp_path / "missing.json",
        output_dir=output_dir,
    )

    assert not project_root.exists()
    assert not output_dir.exists()
    assert not (project_root / "demo-project").exists()
    assert not (output_dir / "oled_agent_mvp_demo_bundle.json").exists()


def test_enqueued_job_can_be_consumed_by_bounded_worker_loop(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    enqueue_result = enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=_write_template(tmp_path / "oled_demo_bundle.json"),
        output_dir=output_dir,
        overwrite=False,
    )

    loop_result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=3,
    )

    assert loop_result["actions"] == ["completed", "idle"]
    assert loop_result["completed_jobs"] == [enqueue_result["job_id"]]
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).status(enqueue_result["job_id"])["status"] == "succeeded"  # type: ignore[index]
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_existing_outputs_remain_worker_loop_failure_not_enqueue_failure(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    enqueue_result = enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=_write_template(tmp_path / "oled_demo_bundle.json"),
        output_dir=output_dir,
        overwrite=False,
    )

    assert enqueue_result["job_status"] == "queued"
    loop_result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=2,
    )
    assert loop_result["actions"] == ["failed", "idle"]
    assert loop_result["failed_jobs"] == [enqueue_result["job_id"]]
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in loop_result["failure_messages"][enqueue_result["job_id"]]
    assert WorkerQueue(JsonWorkerQueueStore(queue_root)).status(enqueue_result["job_id"])["status"] == "failed"  # type: ignore[index]


def test_enqueue_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_enqueue)
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
    assert "admission" not in oled_local_demo_enqueue.__name__
    assert "receipt" not in oled_local_demo_enqueue.__name__
    assert "preflight" not in oled_local_demo_enqueue.__name__
    assert "writer" not in oled_local_demo_enqueue.__name__
