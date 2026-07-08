from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_generic_queue as oled_local_demo_generic_queue
from ai4s_agent.agents.oled_local_demo_generic_queue import (
    execute_oled_local_demo_via_generic_run_plan_queue,
    main,
)
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


def _execute(
    tmp_path: Path,
    *,
    queue_root: Path | None = None,
    project_root: Path | None = None,
    run_id: str = "oled-generic-queue-demo",
    input_bundle: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
    max_iterations: int = 3,
    goal: str | None = None,
) -> dict:
    return execute_oled_local_demo_via_generic_run_plan_queue(
        queue_root=queue_root or (tmp_path / "queue"),
        project_root=project_root or (tmp_path / "projects"),
        project_id="demo-project",
        run_id=run_id,
        input_bundle=input_bundle or _write_template(tmp_path / f"{run_id}.json"),
        output_dir=output_dir or (tmp_path / f"{run_id}-out"),
        worker_id="generic-worker-1",
        max_iterations=max_iterations,
        goal=goal,
        overwrite=overwrite,
        now="2026-01-01T00:00:00Z",
    )


def test_helper_enqueues_generic_run_plan_execute_job_and_executes_it(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"

    result = _execute(tmp_path, queue_root=queue_root, project_root=project_root, output_dir=output_dir)

    assert result == {
        "actions": ["completed", "idle"],
        "artifacts": EXPECTED_ARTIFACTS,
        "completed_jobs": ["job-demo-project-oled-generic-queue-demo"],
        "executable": True,
        "executed_tasks": ["execute_oled_local_demo"],
        "failed_jobs": [],
        "generic_run_plan_queue": True,
        "job_id": "job-demo-project-oled-generic-queue-demo",
        "ok": True,
        "output_dir": str(output_dir),
        "project_id": "demo-project",
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "queue_task_id": "run_plan_execute",
        "run_id": "oled-generic-queue-demo",
        "run_plan_tasks": ["execute_oled_local_demo"],
        "scientific_adapters_executed": False,
        "status": "succeeded",
        "worker_id": "generic-worker-1",
    }
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    jobs = queue.list_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["task"]["task_id"] == "run_plan_execute"
    assert job["task"]["kind"] == "run_plan_execute"
    assert job["task"]["run_plan"]["requested_tasks"] == ["execute_oled_local_demo"]
    assert [task["task_id"] for task in job["task"]["run_plan"]["tasks"]] == ["execute_oled_local_demo"]
    assert job["task"]["task_options"] == {
        "execute_oled_local_demo": {
            "goal": None,
            "input_bundle": str(tmp_path / "oled-generic-queue-demo.json"),
            "output_dir": str(output_dir),
            "overwrite": False,
            "project_id": "demo-project",
        }
    }
    assert job["task"]["task_id"] != "execute_oled_local_demo_runplan"
    lease = queue.lease_status(str(job["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"
    assert job["result"]["executed_tasks"] == ["execute_oled_local_demo"]


def test_generic_queue_execution_writes_project_storage_and_outputs(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    output_dir = tmp_path / "out"

    _execute(tmp_path, project_root=project_root, output_dir=output_dir)

    storage = ProjectStorage(project_root)
    state = storage.read_stage_state("demo-project", "oled-generic-queue-demo")
    registry = storage.read_artifact_registry("demo-project", "oled-generic-queue-demo")
    assert state is not None
    assert state.stage == "execute_oled_local_demo"
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == ["execute_oled_local_demo"]
    assert list(registry) == EXPECTED_ARTIFACTS
    for artifact_id in EXPECTED_ARTIFACTS:
        assert Path(registry[artifact_id]).exists()
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_cli_executes_generic_queue_path_and_prints_compact_json(capsys, tmp_path: Path) -> None:
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
            "cli-demo",
            "--input-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
            "--worker-id",
            "generic-worker-1",
            "--max-iterations",
            "3",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["queue_task_id"] == "run_plan_execute"
    assert payload["run_plan_tasks"] == ["execute_oled_local_demo"]
    assert payload["actions"] == ["completed", "idle"]
    assert payload["executed_tasks"] == ["execute_oled_local_demo"]
    assert payload["generic_run_plan_queue"] is True
    assert payload["scientific_adapters_executed"] is False
    assert "scenarios" not in payload


def test_existing_outputs_without_overwrite_fail_during_generic_execution(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    _execute(tmp_path, output_dir=output_dir, run_id="first-demo")

    with pytest.raises(ValueError, match="local_demo_output_exists:oled_agent_mvp_demo_bundle.json"):
        _execute(tmp_path, output_dir=output_dir, run_id="second-demo", overwrite=False)

    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    failed = queue.status("job-demo-project-second-demo")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["task"]["task_id"] == "run_plan_execute"


def test_existing_outputs_with_overwrite_succeeds(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    _execute(tmp_path, output_dir=output_dir, run_id="first-demo")
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    result = _execute(tmp_path, output_dir=output_dir, run_id="overwrite-demo", overwrite=True)

    assert result["ok"] is True
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["source"] == "local_input_bundle"


def test_missing_input_bundle_fails_during_execution_not_enqueue(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_local_input_bundle:missing.json"):
        _execute(tmp_path, input_bundle=tmp_path / "missing.json", run_id="missing-demo")

    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))
    job = queue.status("job-demo-project-missing-demo")
    assert job is not None
    assert job["task"]["task_id"] == "run_plan_execute"
    assert job["status"] == "failed"


def test_goal_override_is_respected_in_written_bundle_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    _execute(tmp_path, output_dir=output_dir, goal="Override OLED goal")

    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["goal"] == "Override OLED goal"


def test_only_input_bundle_is_opened_for_reading_during_execution(monkeypatch, tmp_path: Path) -> None:
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

    result = _execute(tmp_path, input_bundle=bundle_path)

    assert result["ok"] is True
    assert str(bundle_path) in opened_for_read
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_max_iterations_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        _execute(tmp_path, max_iterations=0)


def test_generic_queue_module_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_generic_queue)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "ai4s_agent.agents.oled_local_demo_worker",
        "ai4s_agent.agents.oled_local_demo_runplan",
        "ai4s_agent.adapters",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
        "subprocess",
    )

    assert "execute_oled_local_demo_runplan" not in source
    assert "resume_after_gate" not in source
    assert "Popen" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_local_demo_generic_queue.__name__
    assert "receipt" not in oled_local_demo_generic_queue.__name__
    assert "preflight" not in oled_local_demo_generic_queue.__name__
    assert "writer" not in oled_local_demo_generic_queue.__name__
