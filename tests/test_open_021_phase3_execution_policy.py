from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.app import create_app
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import GateName


@pytest.mark.parametrize(
    ("adapter", "adapter_name"),
    [
        (adapters.parse_document_mineru_adapter, "parse_document_mineru"),
        (adapters.parse_pdf_folder_mineru_adapter, "parse_pdf_folder_mineru"),
        (adapters.parse_document_grobid_adapter, "parse_document_grobid"),
    ],
)
def test_phase3_remote_adapters_reject_non_boolean_execute(adapter, adapter_name: str) -> None:
    result = adapter({"execute": "true"})

    assert result["status"] == "failed"
    assert result["adapter"] == adapter_name
    assert result["error"]["code"] == "invalid_execute_flag"
    assert "execute must be a boolean" in result["error"]["message"]


def test_remote_document_tasks_require_data_mining_gate() -> None:
    registry = AtomicTaskRegistry()

    assert registry.get("parse_document").gates == [GateName.DATA_MINING.value]
    assert registry.get("parse_document_grobid").gates == [GateName.DATA_MINING.value]
    assert registry.get("parse_document_pdfplumber").gates == []
    assert registry.get("parse_document_pymupdf").gates == []


@pytest.mark.parametrize(
    "adapter_name",
    [
        "parse_document_mineru_adapter",
        "parse_pdf_folder_mineru_adapter",
        "parse_document_grobid_adapter",
    ],
)
def test_direct_remote_document_execute_requires_run_plan_snapshot(
    tmp_path: Path,
    adapter_name: str,
) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/adapters/execute",
        json={
            "run_id": f"run-{adapter_name}",
            "adapter": adapter_name,
            "confirmed": True,
            "actor": "test-user",
            "payload": {"execute": True},
        },
    )

    assert response.status_code == 400
    assert response.json["error"] == "gated adapter execution requires run-plan snapshot approval"
    assert response.json["required_gates"] == [GateName.DATA_MINING.value]
