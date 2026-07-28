from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import ai4s_agent.agents.oled_local_demo_health as oled_local_demo_health
from ai4s_agent.agents.oled_local_demo_enqueue import enqueue_oled_local_demo_worker_job
from ai4s_agent.agents.oled_local_demo_health import check_oled_local_demo_worker_health, main
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def _check(result: dict, name: str) -> dict:
    matches = [item for item in result["checks"] if item["name"] == name]
    assert len(matches) == 1
    return matches[0]


def _enqueue_sample_job(
    *,
    queue_root: Path,
    project_root: Path,
    input_bundle: Path,
    output_dir: Path,
    run_id: str = "oled-local-demo",
) -> dict:
    return enqueue_oled_local_demo_worker_job(
        queue_root=queue_root,
        project_root=project_root,
        project_id="demo-project",
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        created_at="2026-01-01T00:00:00Z",
    )


def test_health_command_returns_ok_for_empty_queue_root(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"

    result = check_oled_local_demo_worker_health(queue_root=queue_root)

    assert result["ok"] is True
    assert result["queue_root"] == str(queue_root)
    assert result["project_root"] == ""
    assert result["executed"] is False
    assert result["executable"] is False
    assert result["summary"] == {"fail": 0, "pass": 4, "warn": 0}
    assert _check(result, "queue_store_readable") == {
        "name": "queue_store_readable",
        "status": "pass",
        "details": {"job_count": 0, "lease_count": 0},
    }
    entrypoints = _check(result, "entrypoints_importable")
    assert entrypoints["status"] == "pass"
    assert entrypoints["details"] == {
        "cancel": True,
        "enqueue": True,
        "retry": True,
        "status": True,
        "worker_loop": True,
        "worker_runner": True,
    }
    task_id = _check(result, "task_id_consistency")
    assert task_id == {
        "name": "task_id_consistency",
        "status": "pass",
        "details": {"expected": TASK_ID},
    }
    status_check = _check(result, "status_read_only_check")
    assert status_check["status"] == "pass"
    assert status_check["details"]["executed"] is False
    assert status_check["details"]["executable"] is False
    assert status_check["details"]["job_count"] == 0


def test_health_cli_prints_compact_json(capsys, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"

    exit_code = main(["--queue-root", str(queue_root)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.endswith("\n")
    assert "\n" not in captured.out.strip()
    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["executable"] is False
    assert "actions" not in payload
    assert "executed_tasks" not in payload


def test_health_includes_queue_job_and_lease_counts(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    enqueued = _enqueue_sample_job(
        queue_root=queue_root,
        project_root=tmp_path / "projects",
        input_bundle=tmp_path / "missing_bundle.json",
        output_dir=tmp_path / "out",
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    acquired = queue.acquire("worker-a", target_job_id=str(enqueued["job_id"]))
    assert acquired is not None

    result = check_oled_local_demo_worker_health(queue_root=queue_root)

    queue_check = _check(result, "queue_store_readable")
    assert queue_check["status"] == "pass"
    assert queue_check["details"] == {"job_count": 1, "lease_count": 1}
    status_check = _check(result, "status_read_only_check")
    assert status_check["details"]["job_count"] == 1
    assert status_check["details"]["status_counts"] == {"running": 1}


def test_health_with_project_root_reports_existence_without_creating_project_storage_dirs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects-root"

    missing_result = check_oled_local_demo_worker_health(queue_root=queue_root, project_root=project_root)

    project_check = _check(missing_result, "project_root_readable")
    assert project_check == {
        "name": "project_root_readable",
        "status": "warn",
        "details": {
            "exists": False,
            "projects_dir_exists": False,
            "project_root": str(project_root),
        },
    }
    assert not project_root.exists()

    project_root.mkdir()
    existing_result = check_oled_local_demo_worker_health(queue_root=queue_root, project_root=project_root)
    existing_project_check = _check(existing_result, "project_root_readable")
    assert existing_project_check["status"] == "pass"
    assert existing_project_check["details"] == {
        "exists": True,
        "projects_dir_exists": False,
        "project_root": str(project_root),
    }
    assert not (project_root / "projects").exists()


def test_health_does_not_read_bundle_open_artifacts_or_create_outputs(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    input_bundle = tmp_path / "do-not-open-bundle.json"
    output_dir = tmp_path / "do-not-create-output"
    _enqueue_sample_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=input_bundle,
        output_dir=output_dir,
    )
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = check_oled_local_demo_worker_health(queue_root=queue_root, project_root=project_root)

    assert result["ok"] is True
    assert str(input_bundle) not in opened_for_read
    assert not output_dir.exists()
    assert not project_root.exists()


def test_health_does_not_mutate_existing_queue_jobs_leases_or_project_files(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    project_root = tmp_path / "projects"
    stage_path = project_root / "projects" / "demo-project" / "runs" / "oled-local-demo" / "stage.json"
    registry_path = project_root / "projects" / "demo-project" / "runs" / "oled-local-demo" / "artifact_registry.json"
    stage_path.parent.mkdir(parents=True)
    stage_path.write_text('{"stage":"execute_oled_local_demo","status":"SUCCEEDED","updated_at":"2026-01-01T00:00:00Z"}\n', encoding="utf-8")
    registry_path.write_text('{"artifacts":{"oled_demo_bundle_report":"report.json"}}\n', encoding="utf-8")
    _enqueue_sample_job(
        queue_root=queue_root,
        project_root=project_root,
        input_bundle=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    before_jobs = queue.list_jobs()
    before_leases = queue.list_leases()
    before_stage = stage_path.read_text(encoding="utf-8")
    before_registry = registry_path.read_text(encoding="utf-8")

    result = check_oled_local_demo_worker_health(queue_root=queue_root, project_root=project_root)

    assert result["ok"] is True
    assert queue.list_jobs() == before_jobs
    assert queue.list_leases() == before_leases
    assert stage_path.read_text(encoding="utf-8") == before_stage
    assert registry_path.read_text(encoding="utf-8") == before_registry


def test_health_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_health)
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
    assert "admission" not in oled_local_demo_health.__name__
    assert "receipt" not in oled_local_demo_health.__name__
    assert "preflight" not in oled_local_demo_health.__name__
    assert "writer" not in oled_local_demo_health.__name__
