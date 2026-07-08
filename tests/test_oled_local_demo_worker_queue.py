from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
import ai4s_agent.agents.oled_local_demo_worker as oled_local_demo_worker
from ai4s_agent.agents.oled_local_demo_worker import (
    OLEDLocalDemoRunPlanWorkerTaskRunner,
    execute_oled_local_demo_worker_queue,
    main,
)
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import TaskRunResult


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


def _task(*, project_root: Path, input_bundle: Path, output_dir: Path, overwrite: bool = False) -> dict:
    return {
        "task_id": "execute_oled_local_demo_runplan",
        "project_root": str(project_root),
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "goal": "Find OLED emitters with high PLQY and red-shifted emission",
        "overwrite": overwrite,
    }


def _job(*, tmp_path: Path, run_id: str = "oled-local-demo", overwrite: bool = False) -> tuple[dict, Path, Path, Path]:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "oled-agent-demo"
    job = {
        "job_id": f"job-{run_id}",
        "project_id": "demo-project",
        "run_id": run_id,
        "task": _task(
            project_root=project_root,
            input_bundle=bundle_path,
            output_dir=output_dir,
            overwrite=overwrite,
        ),
    }
    return job, project_root, bundle_path, output_dir


def _queue_job(
    *,
    queue: WorkerQueue,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    run_id: str = "oled-local-demo",
    overwrite: bool = False,
) -> dict:
    return queue.enqueue(
        "demo-project",
        run_id,
        _task(
            project_root=project_root,
            input_bundle=input_bundle,
            output_dir=output_dir,
            overwrite=overwrite,
        ),
    )


def test_worker_runner_start_executes_valid_job(tmp_path: Path) -> None:
    job, _project_root, _bundle_path, _output_dir = _job(tmp_path=tmp_path)

    result = OLEDLocalDemoRunPlanWorkerTaskRunner().start(job)

    assert result == TaskRunResult(
        state="succeeded",
        message="oled local demo runplan succeeded",
        output={
            "adapters_executed": False,
            "adapter": "execute_oled_local_demo_adapter",
            "artifacts": EXPECTED_ARTIFACTS,
            "executable": True,
            "executed_tasks": ["execute_oled_local_demo"],
            "input_bundle": "oled_demo_bundle.json",
            "ok": True,
            "output_dir": str(tmp_path / "oled-agent-demo"),
            "project_id": "demo-project",
            "project_root": str(tmp_path / "projects"),
            "run_id": "oled-local-demo",
            "status": "succeeded",
            "task": "execute_oled_local_demo",
        },
    )


def test_queue_poller_completes_oled_local_demo_job_in_one_poll(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    queued = _queue_job(queue=queue, project_root=project_root, input_bundle=bundle_path, output_dir=output_dir)
    poller = WorkerQueuePoller(queue, worker_id="local-worker-1", runner=OLEDLocalDemoRunPlanWorkerTaskRunner())

    result = poller.poll_once(now="2026-01-01T00:00:00Z")

    assert result.action == "completed"
    assert result.runner_result is not None
    assert result.runner_result.state == "succeeded"
    assert result.runner_result.output is not None
    assert result.runner_result.output["executed_tasks"] == ["execute_oled_local_demo"]
    status = queue.status(queued["job_id"])
    lease = queue.lease_status(queued["lease_id"]) if queued.get("lease_id") else None
    assert status is not None
    assert status["status"] == "succeeded"
    assert status["result"]["executed_tasks"] == ["execute_oled_local_demo"]
    completed_lease = queue.list_leases()[0]
    assert completed_lease["status"] == "completed"
    assert lease is None


def test_execute_oled_local_demo_worker_queue_enqueues_and_completes_one_job(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    result = execute_oled_local_demo_worker_queue(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        worker_id="local-worker-1",
        goal="Find OLED emitters with high PLQY and red-shifted emission",
        overwrite=False,
        now="2026-01-01T00:00:00Z",
    )

    assert result == {
        "adapters_executed": False,
        "artifacts": EXPECTED_ARTIFACTS,
        "executable": True,
        "executed_tasks": ["execute_oled_local_demo"],
        "job_id": "job-demo-project-oled-local-demo",
        "job_status": "succeeded",
        "lease_status": "completed",
        "ok": True,
        "output_dir": str(output_dir),
        "poll_action": "completed",
        "project_id": "demo-project",
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "run_id": "oled-local-demo",
        "worker_id": "local-worker-1",
    }
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    status = queue.status(result["job_id"])
    assert status is not None
    assert status["status"] == "succeeded"
    assert queue.list_leases()[0]["status"] == "completed"


def test_worker_queue_cli_executes_successfully_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--queue-root",
            str(tmp_path / "queue"),
            "--project-root",
            str(tmp_path / "projects"),
            "--project-id",
            "demo-project",
            "--run-id",
            "oled-local-demo",
            "--input-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--worker-id",
            "local-worker-1",
            "--goal",
            "Find OLED emitters with high PLQY and red-shifted emission",
            "--overwrite",
            "--now",
            "2026-01-01T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["job_status"] == "succeeded"
    assert payload["poll_action"] == "completed"
    assert payload["lease_status"] == "completed"
    assert payload["executed_tasks"] == ["execute_oled_local_demo"]
    assert payload["artifacts"] == EXPECTED_ARTIFACTS
    assert payload["executable"] is True
    assert payload["adapters_executed"] is False
    assert "scenarios" not in payload
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_task_payload_cannot_override_queue_project_or_run(tmp_path: Path) -> None:
    job, _project_root, _bundle_path, output_dir = _job(tmp_path=tmp_path, run_id="queue-run")
    job["task"]["project_id"] = "payload-project"
    job["task"]["run_id"] = "payload-run"

    result = OLEDLocalDemoRunPlanWorkerTaskRunner().start(job)

    assert result.state == "succeeded"
    assert result.output is not None
    assert result.output["project_id"] == "demo-project"
    assert result.output["run_id"] == "queue-run"
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["project_id"] == "demo-project"
    assert report["run_id"] == "queue-run"


def test_poll_returns_cached_synchronous_result(tmp_path: Path) -> None:
    job, _project_root, _bundle_path, _output_dir = _job(tmp_path=tmp_path)
    runner = OLEDLocalDemoRunPlanWorkerTaskRunner()
    started = runner.start(job)

    polled = runner.poll(job)

    assert started.state == "succeeded"
    assert polled == started


def test_project_storage_artifacts_and_outputs_are_written(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _queue_job(queue=queue, project_root=project_root, input_bundle=bundle_path, output_dir=output_dir)

    WorkerQueuePoller(
        queue,
        worker_id="local-worker-1",
        runner=OLEDLocalDemoRunPlanWorkerTaskRunner(),
    ).poll_once()

    storage = ProjectStorage(project_root)
    state = storage.read_stage_state("demo-project", "oled-local-demo")
    registry = storage.read_artifact_registry("demo-project", "oled-local-demo")
    assert state is not None
    assert state.stage == "execute_oled_local_demo"
    assert state.status == RunStatus.SUCCEEDED
    assert list(registry) == EXPECTED_ARTIFACTS
    for artifact_id in EXPECTED_ARTIFACTS:
        assert Path(registry[artifact_id]).exists()
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()
    adapter_result = (
        storage.run_dir("demo-project", "oled-local-demo")
        / "execute_oled_local_demo"
        / "adapter_result.json"
    )
    assert adapter_result.exists()
    assert storage.read_gate_decisions("demo-project", "oled-local-demo") == []
    assert not (storage.run_dir("demo-project", "oled-local-demo") / "asset_promotion_records.json").exists()


def test_invalid_task_id_fails_clearly() -> None:
    result = OLEDLocalDemoRunPlanWorkerTaskRunner().start(
        {
            "job_id": "job-bad",
            "project_id": "demo-project",
            "run_id": "oled-local-demo",
            "task": {"task_id": "wrong_task"},
        }
    )

    assert result.state == "failed"
    assert "unsupported_oled_local_demo_worker_task:wrong_task" in result.message
    assert result.output == {"task_id": "wrong_task"}


def test_missing_input_bundle_fails_clearly(tmp_path: Path) -> None:
    job, _project_root, _bundle_path, _output_dir = _job(tmp_path=tmp_path)
    job["task"]["input_bundle"] = str(tmp_path / "missing.json")

    result = OLEDLocalDemoRunPlanWorkerTaskRunner().start(job)

    assert result.state == "failed"
    assert "missing_local_input_bundle:missing.json" in result.message
    assert result.output == {"task_id": "execute_oled_local_demo_runplan"}


def test_queue_marks_job_failed_for_existing_outputs_without_overwrite(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _queue_job(queue=queue, project_root=project_root, input_bundle=bundle_path, output_dir=output_dir)
    WorkerQueuePoller(queue, worker_id="worker-a", runner=OLEDLocalDemoRunPlanWorkerTaskRunner()).poll_once()
    second = _queue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="second-demo",
    )

    result = WorkerQueuePoller(queue, worker_id="worker-b", runner=OLEDLocalDemoRunPlanWorkerTaskRunner()).poll_once()

    assert result.action == "failed"
    assert result.runner_result is not None
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in result.runner_result.message
    status = queue.status(second["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in status["error"]["reason"]


def test_worker_queue_cli_helper_raises_for_existing_outputs_without_overwrite(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    execute_oled_local_demo_worker_queue(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id="first-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        worker_id="worker-a",
        overwrite=False,
    )

    try:
        execute_oled_local_demo_worker_queue(
            queue_root=queue_root,
            project_root=project_root,
            project_id="demo-project",
            run_id="second-demo",
            input_bundle=bundle_path,
            output_dir=output_dir,
            worker_id="worker-b",
            overwrite=False,
        )
    except ValueError as exc:
        assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected ValueError for existing output")


def test_existing_outputs_with_overwrite_succeeds(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _queue_job(queue=queue, project_root=project_root, input_bundle=bundle_path, output_dir=output_dir)
    WorkerQueuePoller(queue, worker_id="worker-a", runner=OLEDLocalDemoRunPlanWorkerTaskRunner()).poll_once()
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    second = _queue_job(
        queue=queue,
        project_root=project_root,
        input_bundle=bundle_path,
        output_dir=output_dir,
        run_id="overwrite-demo",
        overwrite=True,
    )

    result = WorkerQueuePoller(queue, worker_id="worker-b", runner=OLEDLocalDemoRunPlanWorkerTaskRunner()).poll_once()

    assert result.action == "completed"
    assert queue.status(second["job_id"])["status"] == "succeeded"  # type: ignore[index]
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["source"] == "local_input_bundle"


def test_worker_queue_cli_helper_overwrite_succeeds_and_goal_override_is_written(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    execute_oled_local_demo_worker_queue(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id="first-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        worker_id="worker-a",
        overwrite=False,
    )
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    result = execute_oled_local_demo_worker_queue(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id="second-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        worker_id="worker-b",
        goal="Override OLED demo goal",
        overwrite=True,
    )

    assert result["job_status"] == "succeeded"
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["goal"] == "Override OLED demo goal"


def test_worker_queue_cli_helper_raises_for_missing_input_bundle(tmp_path: Path) -> None:
    try:
        execute_oled_local_demo_worker_queue(
            queue_root=tmp_path / "queue",
            project_root=tmp_path / "projects",
            project_id="demo-project",
            run_id="oled-local-demo",
            input_bundle=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
            worker_id="local-worker-1",
            overwrite=False,
        )
    except ValueError as exc:
        assert "missing_local_input_bundle:missing.json" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected ValueError for missing input bundle")


def test_only_input_bundle_is_opened_for_reading(monkeypatch, tmp_path: Path) -> None:
    bundle = local_input_bundle_template()
    forbidden_dataset_label = str(tmp_path / "do-not-open-dataset.jsonl")
    forbidden_training_label = str(tmp_path / "do-not-open-training.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_dataset_label}
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": forbidden_training_label
    }
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    _queue_job(queue=queue, project_root=project_root, input_bundle=bundle_path, output_dir=output_dir)
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = WorkerQueuePoller(
        queue,
        worker_id="local-worker-1",
        runner=OLEDLocalDemoRunPlanWorkerTaskRunner(),
    ).poll_once()

    assert result.action == "completed"
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_worker_queue_cli_helper_does_not_open_artifact_labels(monkeypatch, tmp_path: Path) -> None:
    bundle = local_input_bundle_template()
    forbidden_dataset_label = str(tmp_path / "do-not-open-dataset.jsonl")
    forbidden_training_label = str(tmp_path / "do-not-open-training.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_dataset_label}
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": forbidden_training_label
    }
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = execute_oled_local_demo_worker_queue(
        queue_root=tmp_path / "queue",
        project_root=tmp_path / "projects",
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
        worker_id="local-worker-1",
        overwrite=False,
    )

    assert result["job_status"] == "succeeded"
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_cancel_returns_cancelled_without_subprocesses() -> None:
    result = OLEDLocalDemoRunPlanWorkerTaskRunner().cancel(
        {
            "job_id": "job-cancel",
            "project_id": "demo-project",
            "run_id": "oled-local-demo",
            "task": {"task_id": "execute_oled_local_demo_runplan"},
        }
    )

    assert result == TaskRunResult(
        state="cancelled",
        message="oled local demo runplan cancellation acknowledged",
        output={"task_id": "execute_oled_local_demo_runplan"},
    )


def test_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_worker)
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
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_local_demo_worker.__name__
    assert "receipt" not in oled_local_demo_worker.__name__
    assert "preflight" not in oled_local_demo_worker.__name__
    assert "writer" not in oled_local_demo_worker.__name__
