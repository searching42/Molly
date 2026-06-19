from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.execution_policy import ExecutionPolicyRegistry
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.schemas import GateName
from ai4s_agent.storage import ProjectStorage


def test_execution_policy_registry_resolves_adapter_aliases_and_gates() -> None:
    registry = ExecutionPolicyRegistry()

    mineru = registry.adapter_policy("parse_document_mineru_adapter")
    folder = registry.adapter_policy("parse_pdf_folder_mineru_adapter")
    grobid = registry.adapter_policy("parse_document_grobid_adapter")

    assert mineru is not None
    assert mineru.task_id == "parse_document"
    assert mineru.required_gates == (GateName.DATA_MINING.value,)
    assert mineru.validate_execute_boolean is True
    assert folder is not None
    assert folder.task_id == "parse_document"
    assert folder.required_gates == (GateName.DATA_MINING.value,)
    assert grobid is not None
    assert grobid.task_id == "parse_document"
    assert grobid.required_gates == (GateName.DATA_MINING.value,)


def test_execution_policy_registry_handles_dynamic_generation_action() -> None:
    registry = ExecutionPolicyRegistry()

    assert registry.adapter_execution_policy("generate_candidates_stub_adapter", {"backend": "deterministic_stub", "count": 32}) == (
        "generate_candidates",
        [GateName.FINAL_THRESHOLD.value],
    )
    assert registry.adapter_execution_policy("generate_candidates_stub_adapter", {"backend": "reinvent4", "count": 32}) == (
        "generate_candidates_expensive",
        [GateName.FINAL_THRESHOLD.value],
    )
    with pytest.raises(ValueError, match="generation count must be a positive integer"):
        registry.adapter_execution_policy("generate_candidates_stub_adapter", {"count": 0})


def test_run_plan_executor_uses_policy_registry_for_adapter_overrides(tmp_path: Path) -> None:
    executor = RunPlanExecutor(storage=ProjectStorage(tmp_path))

    assert executor._adapter_name_for(
        "parse_document",
        "parse_document_mineru_adapter",
        {"adapter": "parse_document_grobid_adapter"},
    ) == "parse_document_grobid_adapter"
    with pytest.raises(ValueError, match="adapter override not allowed"):
        executor._adapter_name_for(
            "render_report",
            "render_report_adapter",
            {"adapter": "parse_document_grobid_adapter"},
        )


def test_direct_adapter_snapshot_guard_comes_from_execution_policy_registry(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "run-domain-model-direct",
            "adapter": "predict_candidates_domain_model_adapter",
            "project_approved": True,
            "payload": {"execute": True, "project_approved": True},
        },
    )

    assert response.status_code == 400
    assert response.json["error"] == "this adapter requires run-plan snapshot approval when execute=true"
    assert response.json["permission"]["action"] == "predict_candidates"


def test_direct_adapter_execute_flag_validation_comes_from_execution_policy_registry() -> None:
    registry = ExecutionPolicyRegistry()
    result = registry.strict_execute_error({"execute": "true"}, adapter="parse_document_mineru")

    assert result is not None
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_execute_flag"
