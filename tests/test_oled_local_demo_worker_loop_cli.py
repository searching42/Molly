from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import ai4s_agent.agents.oled_local_demo_worker_loop as oled_local_demo_worker_loop
from ai4s_agent.agents.oled_local_demo_worker_loop import main, run_oled_local_demo_worker_loop
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


EXPECTED_ARTIFACTS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]
EXPECTED_OUTPUTS = [
    "oled_agent_mvp_demo_bundle.json",
    "oled_agent_mvp_demo_bundle.md",
    "oled_local_demo_execution_manifest.json",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _task(
    *,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    return {
        "task_id": "execute_oled_local_demo_runplan",
        "project_root": str(project_root),
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "overwrite": overwrite,
    }


def _enqueue_job(
    *,
    queue: WorkerQueue,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    project_id: str = "demo-project",
    run_id: str = "oled-local-demo",
    overwrite: bool = False,
) -> dict:
    return queue.enqueue(
        project_id,
        run_id,
        _task(
            project_root=project_root,
            input_bundle=input_bundle,
            output_dir=output_dir,
            overwrite=overwrite,
        ),
    )


def test_bounded_loop_consumes_one_queued_oled_local_demo_job_then_idles(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    queued = _enqueue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
    )

    result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=3,
        now="2026-01-01T00:00:00Z",
    )

    assert result == {
        "actions": ["completed", "idle"],
        "adapters_executed": False,
        "cancelled_jobs": [],
        "completed_jobs": [queued["job_id"]],
        "executable": True,
        "executed_tasks": ["execute_oled_local_demo"],
        "failed_jobs": [],
        "failure_messages": {},
        "idle": True,
        "iterations": 2,
        "max_iterations": 3,
        "ok": True,
        "queue_root": str(queue_root),
        "recovered_jobs": [],
        "worker_id": "local-worker-1",
    }
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "succeeded"
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"
    storage = ProjectStorage(project_root)
    state = storage.read_stage_state("demo-project", "oled-local-demo")
    registry = storage.read_artifact_registry("demo-project", "oled-local-demo")
    assert state is not None
    assert state.status == RunStatus.SUCCEEDED
    assert list(registry) == EXPECTED_ARTIFACTS
    for artifact_id in EXPECTED_ARTIFACTS:
        assert Path(registry[artifact_id]).exists()
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_worker_loop_cli_executes_bounded_loop_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    _enqueue_job(
        queue=queue,
        project_root=tmp_path / "projects",
        input_bundle=_write_template(tmp_path / "oled_demo_bundle.json"),
        output_dir=tmp_path / "out",
    )

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--worker-id",
            "local-worker-1",
            "--max-iterations",
            "3",
            "--now",
            "2026-01-01T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["actions"] == ["completed", "idle"]
    assert payload["completed_jobs"] == ["job-demo-project-oled-local-demo"]
    assert payload["executed_tasks"] == ["execute_oled_local_demo"]
    assert payload["idle"] is True
    assert payload["executable"] is True
    assert payload["adapters_executed"] is False
    assert "scenarios" not in payload


def test_max_iterations_one_processes_one_job_without_requiring_idle(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    queued = _enqueue_job(
        queue=queue,
        project_root=tmp_path / "projects",
        input_bundle=_write_template(tmp_path / "oled_demo_bundle.json"),
        output_dir=tmp_path / "out",
    )

    result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=1,
    )

    assert result["actions"] == ["completed"]
    assert result["iterations"] == 1
    assert result["idle"] is False
    assert result["completed_jobs"] == [queued["job_id"]]


def test_max_iterations_must_be_positive(tmp_path: Path) -> None:
    try:
        run_oled_local_demo_worker_loop(
            queue_root=tmp_path / "queue",
            worker_id="local-worker-1",
            max_iterations=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_iterations must be positive"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected ValueError for non-positive max_iterations")


def test_target_selectors_choose_existing_queued_jobs(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")

    queue_root_by_job = tmp_path / "queue-by-job"
    queue_by_job = WorkerQueue(JsonWorkerQueueStore(queue_root_by_job))
    first = _enqueue_job(
        queue=queue_by_job,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "job-first",
        run_id="run-first",
    )
    second = _enqueue_job(
        queue=queue_by_job,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "job-second",
        run_id="run-second",
    )
    by_job = run_oled_local_demo_worker_loop(
        queue_root=queue_root_by_job,
        worker_id="worker-job",
        max_iterations=2,
        target_job_id=second["job_id"],
    )
    assert by_job["completed_jobs"] == [second["job_id"]]
    assert queue_by_job.status(first["job_id"])["status"] == "queued"  # type: ignore[index]
    assert queue_by_job.status(second["job_id"])["status"] == "succeeded"  # type: ignore[index]

    queue_root_by_project = tmp_path / "queue-by-project"
    queue_by_project = WorkerQueue(JsonWorkerQueueStore(queue_root_by_project))
    project_a = _enqueue_job(
        queue=queue_by_project,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "project-a",
        project_id="project-a",
        run_id="run-a",
    )
    project_b = _enqueue_job(
        queue=queue_by_project,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "project-b",
        project_id="project-b",
        run_id="run-b",
    )
    by_project = run_oled_local_demo_worker_loop(
        queue_root=queue_root_by_project,
        worker_id="worker-project",
        max_iterations=2,
        target_project_id="project-b",
    )
    assert by_project["completed_jobs"] == [project_b["job_id"]]
    assert queue_by_project.status(project_a["job_id"])["status"] == "queued"  # type: ignore[index]
    assert queue_by_project.status(project_b["job_id"])["status"] == "succeeded"  # type: ignore[index]

    queue_root_by_run = tmp_path / "queue-by-run"
    queue_by_run = WorkerQueue(JsonWorkerQueueStore(queue_root_by_run))
    run_a = _enqueue_job(
        queue=queue_by_run,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "run-a",
        run_id="run-a",
    )
    run_b = _enqueue_job(
        queue=queue_by_run,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=tmp_path / "run-b",
        run_id="run-b",
    )
    by_run = run_oled_local_demo_worker_loop(
        queue_root=queue_root_by_run,
        worker_id="worker-run",
        max_iterations=2,
        target_run_id="run-b",
    )
    assert by_run["completed_jobs"] == [run_b["job_id"]]
    assert queue_by_run.status(run_a["job_id"])["status"] == "queued"  # type: ignore[index]
    assert queue_by_run.status(run_b["job_id"])["status"] == "succeeded"  # type: ignore[index]


def test_existing_outputs_without_overwrite_fail_and_are_reported(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _enqueue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="first-demo",
    )
    run_oled_local_demo_worker_loop(queue_root=queue_root, worker_id="worker-a", max_iterations=2)
    second = _enqueue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="second-demo",
    )

    result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="worker-b",
        max_iterations=2,
    )

    assert result["actions"] == ["failed", "idle"]
    assert result["failed_jobs"] == [second["job_id"]]
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in result["failure_messages"][second["job_id"]]
    status = queue.status(second["job_id"])
    assert status is not None
    assert status["status"] == "failed"


def test_existing_outputs_with_overwrite_succeeds(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _enqueue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="first-demo",
    )
    run_oled_local_demo_worker_loop(queue_root=queue_root, worker_id="worker-a", max_iterations=2)
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    second = _enqueue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="second-demo",
        overwrite=True,
    )

    result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="worker-b",
        max_iterations=2,
    )

    assert result["actions"] == ["completed", "idle"]
    assert result["completed_jobs"] == [second["job_id"]]
    assert json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))["source"] == "local_input_bundle"


def test_worker_loop_does_not_open_artifact_labels(monkeypatch, tmp_path: Path) -> None:
    bundle = local_input_bundle_template()
    forbidden_dataset_label = str(tmp_path / "do-not-open-dataset.jsonl")
    forbidden_training_label = str(tmp_path / "do-not-open-training.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_dataset_label}
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": forbidden_training_label
    }
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    queue_root = tmp_path / "queue"
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    _enqueue_job(
        queue=queue,
        project_root=tmp_path / "projects",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
    )
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = run_oled_local_demo_worker_loop(
        queue_root=queue_root,
        worker_id="local-worker-1",
        max_iterations=2,
    )

    assert result["actions"] == ["completed", "idle"]
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_worker_loop_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_worker_loop)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "ai4s_agent.adapters.phase1",
        "ai4s_agent.adapters.phase3",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
    )

    assert "resume_after_gate" not in source
    assert ".enqueue(" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_local_demo_worker_loop.__name__
    assert "receipt" not in oled_local_demo_worker_loop.__name__
    assert "preflight" not in oled_local_demo_worker_loop.__name__
    assert "writer" not in oled_local_demo_worker_loop.__name__
