from __future__ import annotations

import json
from pathlib import Path

import httpx

from ai4s_agent.document_parse_mineru import MinerUApiDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseRequest
from ai4s_agent.mineru_api_client import MinerUApiClient
from document_parse_test_helpers import write_synthetic_pdf


def test_mineru_provider_returns_stable_failure_result_with_task_audit(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    status_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "protocol_version": "v1"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path == "/tasks/task-123":
            status_calls["count"] += 1
            if status_calls["count"] == 1:
                return httpx.Response(200, json={"task_id": "task-123", "state": "pending", "queued_ahead": 4})
            return httpx.Response(200, json={"task_id": "task-123", "state": "completed", "queued_ahead": 0})
        if request.url.path == "/tasks/task-123/result":
            return httpx.Response(500, json={"error": "download failed"})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        poll_interval_sec=0.01,
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-provider-failure",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    result = MinerUApiDocumentParseProvider(client=client).parse(request)

    assert result.ok is False
    assert result.status == "failed"
    assert result.remote_task_id == "task-123"
    assert result.error is not None
    assert result.error.code == "result_download_failure"
    assert result.audit.task_status_history == ["pending", "completed"]
    assert result.audit.queued_ahead_history == [4, 0]
    audit_path = Path(result.outputs.parser_audit_json)
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["remote_task_id"] == "task-123"
    assert audit["task_status_history"] == ["pending", "completed"]
    assert audit["error"]["code"] == "result_download_failure"
    assert "Authorization" not in json.dumps(audit)
