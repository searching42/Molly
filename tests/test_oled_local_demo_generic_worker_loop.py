from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_worker_loop as generic_worker_loop
from ai4s_agent.agents.oled_local_demo_generic_worker_loop import (
    ALLOWLISTED_RUN_PLAN_TASKS,
    AllowlistedOLEDRunPlanExecutorTaskRunner,
    main,
    run_oled_local_demo_generic_worker_loop,
)
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
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


def _queue(queue_root: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(queue_root))


def _enqueue_oled_job(
    *,
    queue_root: Path,
    project_id: str = "demo-project",
    run_id: str = "oled-generic-loop-demo",
    input_bundle: Path,
    output_dir: Path,
    goal: str | None = None,
    overwrite: bool = False,
) -> dict:
    return enqueue_run_plan_execute_job(
        _queue(queue_root),
        project_id=project_id,
        run_plan=expand_run_plan(run_id=run_id, requested_tasks=["execute_oled_local_demo"]),
        input_artifacts={},
        task_options={
            "execute_oled_local_demo": {
                "input_bundle": str(input_bundle),
                "output_dir": str(output_dir),
                "overwrite": overwrite,
                "goal": goal,
                "project_id": project_id,
            }
        },
    )


def _run_loop(
    tmp_path: Path,
    *,
    queue_root: Path | None = None,
    project_root: Path | None = None,
    worker_id: str = "generic-worker-1",
    max_iterations: int = 3,
    target_job_id: str | None = None,
    target_project_id: str | None = None,
    target_run_id: str | None = None,
) -> dict:
    return run_oled_local_demo_generic_worker_loop(
        queue_root=queue_root or (tmp_path / "queue"),
        project_root=project_root or (tmp_path / "projects"),
        worker_id=worker_id,
        max_iterations=max_iterations,
        target_job_id=target_job_id,
        target_project_id=target_project_id,
        target_run_id=target_run_id,
        now="2026-01-01T00:00:00Z",
    )


def test_existing_generic_run_plan_job_is_consumed_successfully(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    job = _enqueue_oled_job(queue_root=queue_root, input_bundle=bundle_path, output_dir=output_dir)

    before_jobs = _queue(queue_root).list_jobs()
    result = _run_loop(tmp_path, queue_root=queue_root, project_root=project_root)

    assert result == {
        "actions": ["completed", "idle"],
        "allowlist": ["execute_oled_local_demo"],
        "cancelled_jobs": [],
        "completed_jobs": [job["job_id"]],
        "executable": True,
        "executed_tasks": ["execute_oled_local_demo"],
        "failed_jobs": [],
        "generic_run_plan_queue": True,
        "iterations": 2,
        "max_iterations": 3,
        "ok": True,
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "scientific_adapters_executed": False,
        "worker_id": "generic-worker-1",
    }
    after_jobs = _queue(queue_root).list_jobs()
    assert len(after_jobs) == len(before_jobs) == 1
    final_job = _queue(queue_root).status(job["job_id"])
    assert final_job is not None
    assert final_job["status"] == "succeeded"
    assert final_job["task"]["task_id"] == "run_plan_execute"
    assert final_job["task"]["kind"] == "run_plan_execute"
    assert final_job["task"]["task_id"] != "execute_oled_local_demo_runplan"
    assert final_job["task"]["run_plan"]["requested_tasks"] == ["execute_oled_local_demo"]
    lease = _queue(queue_root).lease_status(str(final_job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"

    storage = ProjectStorage(project_root)
    state = storage.read_stage_state("demo-project", "oled-generic-loop-demo")
    registry = storage.read_artifact_registry("demo-project", "oled-generic-loop-demo")
    assert state is not None
    assert state.stage == "execute_oled_local_demo"
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == ["execute_oled_local_demo"]
    assert list(registry) == EXPECTED_ARTIFACTS
    for artifact_id in EXPECTED_ARTIFACTS:
        assert Path(registry[artifact_id]).exists()
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_cli_runs_worker_loop_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    _enqueue_oled_job(queue_root=queue_root, input_bundle=bundle_path, output_dir=tmp_path / "out")

    exit_code = main(
        [
            "--queue-root",
            str(queue_root),
            "--project-root",
            str(tmp_path / "projects"),
            "--worker-id",
            "generic-worker-1",
            "--max-iterations",
            "3",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["actions"] == ["completed", "idle"]
    assert payload["executed_tasks"] == ["execute_oled_local_demo"]
    assert payload["allowlist"] == ["execute_oled_local_demo"]
    assert payload["scientific_adapters_executed"] is False
    assert "scenarios" not in payload


@pytest.mark.parametrize("selector", ["target_job_id", "target_project_id", "target_run_id"])
def test_target_selectors_consume_matching_job(selector: str, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    first_bundle = _write_template(tmp_path / "first.json")
    second_bundle = _write_template(tmp_path / "second.json")
    _enqueue_oled_job(
        queue_root=queue_root,
        project_id="other-project",
        run_id="other-run",
        input_bundle=first_bundle,
        output_dir=tmp_path / "other-out",
    )
    target = _enqueue_oled_job(
        queue_root=queue_root,
        project_id="demo-project",
        run_id="target-run",
        input_bundle=second_bundle,
        output_dir=tmp_path / "target-out",
    )

    kwargs = {
        "target_job_id": target["job_id"] if selector == "target_job_id" else None,
        "target_project_id": "demo-project" if selector == "target_project_id" else None,
        "target_run_id": "target-run" if selector == "target_run_id" else None,
    }
    result = _run_loop(tmp_path, queue_root=queue_root, **kwargs)

    assert target["job_id"] in result["completed_jobs"]
    assert _queue(queue_root).status(target["job_id"])["status"] == "succeeded"
    assert _queue(queue_root).status("job-other-project-other-run")["status"] == "queued"


def test_disallowed_generic_run_plan_fails_before_execution(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    job = enqueue_run_plan_execute_job(
        _queue(queue_root),
        project_id="demo-project",
        run_plan=expand_run_plan(run_id="disallowed-run", requested_tasks=["run_baseline"]),
        input_artifacts={},
        task_options={},
    )

    result = _run_loop(tmp_path, queue_root=queue_root, max_iterations=2)

    assert result["actions"] == ["failed", "idle"]
    assert result["failed_jobs"] == [job["job_id"]]
    failed = _queue(queue_root).status(job["job_id"])
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == {"reason": "generic_run_plan_not_allowlisted_for_oled_local_demo"}
    assert ProjectStorage(tmp_path / "projects").read_stage_state("demo-project", "disallowed-run") is None
    assert not (tmp_path / "out").exists()


def test_allowlist_runner_poll_and_cancel_delegate_to_run_plan_runner(tmp_path: Path) -> None:
    runner = AllowlistedOLEDRunPlanExecutorTaskRunner(storage=ProjectStorage(tmp_path / "projects"))

    poll_result = runner.poll({"task": {"task_id": "run_plan_execute"}})
    cancel_result = runner.cancel({"task": {"task_id": "run_plan_execute"}})

    assert isinstance(poll_result, TaskRunResult)
    assert poll_result.state == "failed"
    assert "poll is unsupported" in poll_result.message
    assert cancel_result.state == "cancelled"
    assert "cancellation is not supported" in cancel_result.message


def test_existing_outputs_without_overwrite_fail_clearly(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    _enqueue_oled_job(
        queue_root=queue_root,
        run_id="first-run",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=True,
    )
    _run_loop(tmp_path, queue_root=queue_root)
    _enqueue_oled_job(
        queue_root=queue_root,
        run_id="second-run",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=False,
    )

    result = _run_loop(tmp_path, queue_root=queue_root)

    assert result["actions"] == ["failed", "idle"]
    assert result["failed_jobs"] == ["job-demo-project-second-run"]
    failed = _queue(queue_root).status("job-demo-project-second-run")
    assert failed["error"] == {"reason": "local_demo_output_exists:oled_agent_mvp_demo_bundle.json"}


def test_existing_outputs_with_overwrite_succeeds(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_dir = tmp_path / "out"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    _enqueue_oled_job(queue_root=queue_root, run_id="first-run", input_bundle=bundle_path, output_dir=output_dir)
    _run_loop(tmp_path, queue_root=queue_root)
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")
    _enqueue_oled_job(
        queue_root=queue_root,
        run_id="overwrite-run",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=True,
    )

    result = _run_loop(tmp_path, queue_root=queue_root)

    assert result["completed_jobs"] == ["job-demo-project-overwrite-run"]
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["source"] == "local_input_bundle"


def test_missing_input_bundle_fails_during_execution(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _enqueue_oled_job(
        queue_root=queue_root,
        run_id="missing-bundle-run",
        input_bundle=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )

    result = _run_loop(tmp_path, queue_root=queue_root)

    assert result["actions"] == ["failed", "idle"]
    failed = _queue(queue_root).status("job-demo-project-missing-bundle-run")
    assert failed is not None
    assert failed["error"] == {"reason": "missing_local_input_bundle:missing.json"}


def test_only_input_bundle_is_opened_for_reading(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    bundle = local_input_bundle_template()
    forbidden_dataset_label = str(tmp_path / "do-not-open-dataset.jsonl")
    forbidden_training_label = str(tmp_path / "do-not-open-training.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_dataset_label}
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": forbidden_training_label
    }
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _enqueue_oled_job(queue_root=queue_root, input_bundle=bundle_path, output_dir=tmp_path / "out")
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = _run_loop(tmp_path, queue_root=queue_root)

    assert result["ok"] is True
    assert str(bundle_path) in opened_for_read
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_max_iterations_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        _run_loop(tmp_path, max_iterations=0)


def test_generic_worker_loop_module_safety_guards() -> None:
    source = inspect.getsource(generic_worker_loop)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "ai4s_agent.agents.oled_local_demo_worker",
        "ai4s_agent.adapters",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
    )

    assert list(ALLOWLISTED_RUN_PLAN_TASKS) == ["execute_oled_local_demo"]
    assert "execute_oled_local_demo_runplan" not in source
    assert "resume_after_gate" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in generic_worker_loop.__name__
    assert "receipt" not in generic_worker_loop.__name__
    assert "preflight" not in generic_worker_loop.__name__
    assert "writer" not in generic_worker_loop.__name__
