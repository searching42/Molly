from __future__ import annotations

import io
import json
from pathlib import Path

import httpx

from ai4s_agent.document_parse_cli import main
from document_parse_test_helpers import build_zip_from_dir, fixture_mineru_output_dir, write_synthetic_pdf


def test_document_parse_cli_pdfplumber_succeeds_locally(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--provider",
            "pdfplumber",
            "--input",
            str(pdf),
            "--output",
            str(tmp_path / "out"),
            "--run-id",
            "cli-pdfplumber",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["ok"] is True
    assert payload["provider"] == "pdfplumber"
    assert payload["parser_backend"] == "pdfplumber_local"
    assert stderr.getvalue() == ""


def test_document_parse_cli_mineru_uses_injected_transport(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    zip_payload = build_zip_from_dir(fixture_mineru_output_dir())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path == "/tasks/task-123":
            return httpx.Response(200, json={"task_id": "task-123", "state": "completed", "_backend": "hybrid-engine"})
        if request.url.path == "/tasks/task-123/result":
            return httpx.Response(200, stream=httpx.ByteStream(zip_payload))
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "--provider",
            "mineru-api",
            "--input",
            str(pdf),
            "--output",
            str(tmp_path / "out"),
            "--run-id",
            "cli-mineru",
            "--api-url",
            "http://127.0.0.1:8000",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=httpx.MockTransport(handler),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["ok"] is True
    assert payload["provider"] == "mineru_api"
    assert payload["remote_task_id"] == "task-123"
    assert payload["audit"]["request_provider"] == "mineru_api"
    assert stderr.getvalue() == ""


def test_document_parse_cli_requires_remote_upload_flag_for_non_loopback(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--provider",
            "mineru-api",
            "--input",
            str(pdf),
            "--output",
            str(tmp_path / "out"),
            "--run-id",
            "cli-mineru-remote",
            "--api-url",
            "https://mineru.example.com",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert "allow_remote_upload" in payload["error"]["message"]
    assert "allow_remote_upload" in stderr.getvalue()
