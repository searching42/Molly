from pathlib import Path

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.remote_worker import RemoteWorkerRegistry
from ai4s_agent.schemas import RemoteWorkerConfig, RemoteWorkerRequest


def test_remote_worker_registry_plans_safe_non_executable_assignment(tmp_path: Path) -> None:
    registry = RemoteWorkerRegistry(workspace_dir=tmp_path)
    registry.save_worker(
        RemoteWorkerConfig(
            worker_id="workstation2-mineru",
            transport="ssh",
            host="workstation2",
            capabilities=["gpu", "mineru_parse", "unimol_train"],
            environment="MinerU",
            work_dir="/remote/work/ai4s",
        )
    )

    assignment = registry.plan_assignment(
        RemoteWorkerRequest(
            project_id="proj-remote",
            run_id="run-remote",
            task_id="parse_document",
            required_capabilities=["mineru_parse", "gpu"],
            preferred_worker_id="workstation2-mineru",
            budget_limit_sec=1200,
        )
    )

    assert assignment.status == "needs_confirmation"
    assert assignment.executable is False
    assert assignment.requires_confirmation is True
    assert assignment.worker_id == "workstation2-mineru"
    assert assignment.matched_capabilities == ["gpu", "mineru_parse"]
    assert assignment.missing_capabilities == []
    assert assignment.required_permissions == ["remote_worker:workstation2-mineru", "external_network:ssh"]


def test_remote_worker_registry_reports_missing_capabilities_without_execution(tmp_path: Path) -> None:
    registry = RemoteWorkerRegistry(workspace_dir=tmp_path)
    registry.save_worker(
        RemoteWorkerConfig(
            worker_id="workstation2-mineru",
            transport="ssh",
            host="workstation2",
            capabilities=["mineru_parse"],
        )
    )

    assignment = registry.plan_assignment(
        RemoteWorkerRequest(
            run_id="run-remote",
            task_id="train_model",
            required_capabilities=["unimol_train", "gpu"],
        )
    )

    assert assignment.status == "no_worker"
    assert assignment.worker_id == ""
    assert assignment.executable is False
    assert assignment.requires_confirmation is True
    assert assignment.missing_capabilities == ["gpu", "unimol_train"]


def test_remote_worker_request_requires_capabilities() -> None:
    with pytest.raises(ValueError, match="required_capabilities"):
        RemoteWorkerRequest(run_id="run-remote", task_id="parse_document")


def test_remote_worker_config_rejects_bool_numeric_limits() -> None:
    with pytest.raises(ValueError, match="numeric limits"):
        RemoteWorkerConfig(
            worker_id="workstation2-mineru",
            transport="ssh",
            host="workstation2",
            capabilities=["mineru_parse"],
            max_concurrent_jobs=True,
        )
    with pytest.raises(ValueError, match="numeric limits"):
        RemoteWorkerConfig(
            worker_id="workstation2-mineru",
            transport="ssh",
            host="workstation2",
            capabilities=["mineru_parse"],
            default_timeout_sec=True,
        )


def test_remote_worker_registry_rejects_path_traversal_worker_id(tmp_path: Path) -> None:
    registry = RemoteWorkerRegistry(workspace_dir=tmp_path)

    with pytest.raises(ValueError):
        registry.save_worker(
            RemoteWorkerConfig(
                worker_id="../escape",
                transport="ssh",
                host="workstation2",
                capabilities=["mineru_parse"],
            )
        )

    assert not (tmp_path.parent / "escape").exists()


def test_remote_worker_api_registers_worker_and_returns_assignment(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    created = client.post(
        "/api/workers",
        json={
            "worker_id": "workstation2-mineru",
            "transport": "ssh",
            "host": "workstation2",
            "display_name": "Workstation2 MinerU",
            "capabilities": ["gpu", "mineru_parse"],
            "environment": "MinerU",
        },
    )
    assert created.status_code == 200
    assert created.json["worker"]["worker_id"] == "workstation2-mineru"

    listed = client.get("/api/workers")
    assert listed.status_code == 200
    assert listed.json["workers"][0]["worker_id"] == "workstation2-mineru"

    planned = client.post(
        "/api/workers/assignment",
        json={
            "project_id": "proj-remote-api",
            "run_id": "run-remote-api",
            "task_id": "parse_document",
            "required_capabilities": ["mineru_parse", "gpu"],
            "preferred_worker_id": "workstation2-mineru",
            "budget_limit_sec": 900,
        },
    )
    assert planned.status_code == 200
    assignment = planned.json["assignment"]
    assert assignment["status"] == "needs_confirmation"
    assert assignment["executable"] is False
    assert assignment["required_permissions"] == ["remote_worker:workstation2-mineru", "external_network:ssh"]


def test_remote_worker_api_rejects_sensitive_metadata(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/workers",
        json={
            "worker_id": "bad-worker",
            "transport": "ssh",
            "host": "workstation2",
            "capabilities": ["mineru_parse"],
            "metadata": {"password": "secret"},
        },
    )

    assert resp.status_code == 400
    assert "sensitive credential" in resp.json["error"]


def test_remote_worker_api_rejects_assignment_without_capabilities(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/api/workers",
        json={
            "worker_id": "workstation2-mineru",
            "transport": "ssh",
            "host": "workstation2",
            "capabilities": ["gpu", "mineru_parse"],
        },
    )

    resp = client.post(
        "/api/workers/assignment",
        json={
            "run_id": "run-remote-api",
            "task_id": "parse_document",
        },
    )

    assert resp.status_code == 400
    assert "required_capabilities" in resp.json["error"]
