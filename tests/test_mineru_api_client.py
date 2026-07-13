from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path

import httpx
import pytest

from ai4s_agent.document_parse_provider import DocumentParseRequest
from ai4s_agent.mineru_api_client import MinerUApiClient, MinerUApiError, safe_extract_result_archive
from document_parse_test_helpers import build_zip_from_dir, fixture_mineru_output_dir, write_synthetic_pdf


def _stream_response(content: bytes, *, headers: dict[str, str] | None = None, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, stream=httpx.ByteStream(content), headers=headers)


def test_mineru_api_client_parse_pdf_uses_task_api_and_downloads_zip(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    requests_seen: list[tuple[str, str, bytes]] = []
    task_status_calls = {"count": 0}
    zip_payload = build_zip_from_dir(fixture_mineru_output_dir())

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        requests_seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "_version_name": "mineru-live"})
        if request.method == "POST" and request.url.path == "/tasks":
            text = body.decode("utf-8", errors="ignore")
            assert "backend" in text
            assert "hybrid-engine" in text
            assert "parse_method" in text
            assert "return_md" in text
            assert "paper.pdf" in text
            return httpx.Response(
                202,
                json={
                    "task_id": "task-123",
                    "status_url": "/tasks/task-123",
                    "result_url": "/tasks/task-123/result",
                    "queued_ahead": 3,
                },
            )
        if request.method == "GET" and request.url.path == "/tasks/task-123":
            task_status_calls["count"] += 1
            if task_status_calls["count"] == 1:
                return httpx.Response(200, json={"task_id": "task-123", "state": "pending", "queued_ahead": 2})
            return httpx.Response(
                200,
                json={
                    "task_id": "task-123",
                    "state": "completed",
                    "queued_ahead": 0,
                    "_backend": "hybrid-engine",
                    "_version_name": "mineru-live",
                    "protocol_version": "v1",
                },
            )
        if request.method == "GET" and request.url.path == "/tasks/task-123/result":
            return _stream_response(zip_payload, headers={"content-type": "application/zip"})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
        backend="hybrid-engine",
    )

    health = client.health()
    outcome = client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")

    assert health["_version_name"] == "mineru-live"
    assert outcome.remote_task_id == "task-123"
    assert outcome.source_pdf_sha256 == f"sha256:{hashlib.sha256(pdf.read_bytes()).hexdigest()}"
    assert outcome.task_status_history == ["pending", "completed"]
    assert outcome.queued_ahead_history == [2, 0]
    assert "synthetic_content_list.json" in outcome.extracted_relative_paths
    assert (tmp_path / "bundle" / "synthetic.md").exists()
    assert any(method == "POST" and path == "/tasks" for method, path, _ in requests_seen)
    assert any(method == "GET" and path == "/tasks/task-123/result" for method, path, _ in requests_seen)


def test_mineru_api_client_rejects_source_hash_mismatch_before_upload(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request.url.path)
        return httpx.Response(500)

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
        expected_source_pdf_sha256="sha256:" + "0" * 64,
    )

    with pytest.raises(MinerUApiError) as exc_info:
        client.submit_pdf(request=request, input_pdf=pdf)

    assert exc_info.value.code == "source_hash_mismatch"
    assert requests_seen == []


def test_mineru_api_client_downloads_result_zip_as_raw_bytes(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    zip_payload = build_zip_from_dir(fixture_mineru_output_dir())
    result_accept_encoding = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal result_accept_encoding
        if request.method == "POST" and request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-raw"})
        if request.method == "GET" and request.url.path == "/tasks/task-raw":
            return httpx.Response(200, json={"task_id": "task-raw", "state": "completed"})
        if request.method == "GET" and request.url.path == "/tasks/task-raw/result":
            result_accept_encoding = request.headers.get("accept-encoding", "")
            return _stream_response(zip_payload, headers={"content-type": "application/zip", "content-encoding": "deflate"})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    outcome = client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")

    assert result_accept_encoding == "identity"
    assert "synthetic_content_list.json" in outcome.extracted_relative_paths
    assert (tmp_path / "bundle" / "synthetic.md").exists()


def test_mineru_api_client_polls_with_interval_and_submits_once(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    calls = {"submit": 0, "status": 0}
    sleeps: list[float] = []
    now = {"value": 100.0}

    def fake_monotonic() -> float:
        return now["value"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["value"] += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks":
            calls["submit"] += 1
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path == "/tasks/task-123":
            calls["status"] += 1
            if calls["status"] < 3:
                return httpx.Response(200, json={"task_id": "task-123", "state": "processing"})
            return httpx.Response(200, json={"task_id": "task-123", "state": "completed"})
        if request.url.path == "/tasks/task-123/result":
            return _stream_response(build_zip_from_dir(fixture_mineru_output_dir()))
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        http_timeout_sec=5,
        task_timeout_sec=20,
        poll_interval_sec=2,
        max_poll_attempts=5,
        monotonic=fake_monotonic,
        sleep=fake_sleep,
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    outcome = client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")

    assert outcome.task_status_history == ["processing", "processing", "completed"]
    assert calls == {"submit": 1, "status": 3}
    assert sleeps == [2, 2]


def test_mineru_api_client_rejects_invalid_timing_configuration() -> None:
    with pytest.raises(ValueError, match="http_timeout_sec must be positive"):
        MinerUApiClient(base_url="http://127.0.0.1:8000", http_timeout_sec=0)
    with pytest.raises(ValueError, match="task_timeout_sec must be positive"):
        MinerUApiClient(base_url="http://127.0.0.1:8000", task_timeout_sec=0)
    with pytest.raises(ValueError, match="poll_interval_sec must be positive"):
        MinerUApiClient(base_url="http://127.0.0.1:8000", poll_interval_sec=0)
    with pytest.raises(ValueError, match="max_poll_attempts must be positive"):
        MinerUApiClient(base_url="http://127.0.0.1:8000", max_poll_attempts=0)


def test_mineru_api_client_wraps_transport_errors_and_redacts_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cannot connect with secret-token", request=request)

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        api_token="secret-token",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(MinerUApiError) as exc_info:
        client.health()

    error = exc_info.value.to_error_dict()
    assert error["code"] == "api_unavailable"
    assert "secret-token" not in str(error)


def test_mineru_api_client_rejects_base_url_userinfo_query_and_fragment() -> None:
    for value in [
        "https://token@example.com",
        "https://mineru.example.com?token=secret",
        "https://mineru.example.com/#secret",
    ]:
        with pytest.raises(ValueError, match="must not include userinfo, query, or fragment"):
            MinerUApiClient(base_url=value)


def test_mineru_api_client_rejects_non_loopback_without_allow_remote_upload(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    client = MinerUApiClient(
        base_url="https://mineru.example.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    with pytest.raises(MinerUApiError, match="allow_remote_upload"):
        client.submit_pdf(request=request, input_pdf=pdf)


def test_mineru_api_client_rejects_remote_http_without_insecure_override(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    client = MinerUApiClient(
        base_url="http://mineru.example.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
        allow_remote_upload=True,
    )

    with pytest.raises(MinerUApiError, match="explicit insecure development override"):
        client.submit_pdf(request=request, input_pdf=pdf)


def test_mineru_api_client_rejects_oversized_result_and_does_not_leak_token(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks":
            assert request.headers["Authorization"] == "Bearer secret-token"
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path == "/tasks/task-123":
            return httpx.Response(200, json={"task_id": "task-123", "state": "completed"})
        if request.url.path == "/tasks/task-123/result":
            return _stream_response(b"x" * 32)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        api_token="secret-token",
        max_result_bytes=8,
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    with pytest.raises(MinerUApiError) as exc_info:
        client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")

    error = exc_info.value.to_error_dict()
    assert error["code"] == "oversized_result"
    assert "secret-token" not in str(error)


def test_mineru_api_client_result_download_failure_preserves_response_payload(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-500"})
        if request.url.path == "/tasks/task-500":
            return httpx.Response(200, json={"task_id": "task-500", "state": "completed"})
        if request.url.path == "/tasks/task-500/result":
            return _stream_response(
                b'{"error":"download failed","detail":"bundle expired"}',
                headers={"content-type": "application/json"},
                status_code=500,
            )
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    with pytest.raises(MinerUApiError) as exc_info:
        client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")

    error = exc_info.value.to_error_dict()
    assert error["code"] == "result_download_failure"
    assert error["details"]["task_id"] == "task-500"
    assert error["details"]["status_code"] == 500
    assert error["details"]["response_json"] == {"error": "download failed", "detail": "bundle expired"}


def test_safe_extract_result_archive_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")

    bad_zip = io.BytesIO()
    with zipfile.ZipFile(bad_zip, "w") as archive:
        archive.writestr("../escape.txt", "nope")
    with pytest.raises(MinerUApiError, match="path traversal"):
        safe_extract_result_archive(
            archive_bytes=bad_zip.getvalue(),
            destination_dir=tmp_path / "out",
            original_pdf=pdf,
        )

    symlink_zip = io.BytesIO()
    info = zipfile.ZipInfo("symlink")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_zip, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(MinerUApiError, match="symlinks"):
        safe_extract_result_archive(
            archive_bytes=symlink_zip.getvalue(),
            destination_dir=tmp_path / "out2",
            original_pdf=pdf,
        )


def test_safe_extract_result_archive_rejects_member_count_and_uncompressed_limits_before_writing(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    many_members = io.BytesIO()
    with zipfile.ZipFile(many_members, "w") as archive:
        archive.writestr("a.txt", "a")
        archive.writestr("b.txt", "b")
    with pytest.raises(MinerUApiError, match="too many files"):
        safe_extract_result_archive(
            archive_bytes=many_members.getvalue(),
            destination_dir=tmp_path / "too-many",
            original_pdf=pdf,
            max_member_count=1,
        )
    assert not (tmp_path / "too-many" / "a.txt").exists()

    large_member = io.BytesIO()
    with zipfile.ZipFile(large_member, "w") as archive:
        archive.writestr("large.txt", "x" * 32)
    with pytest.raises(MinerUApiError, match="member exceeds"):
        safe_extract_result_archive(
            archive_bytes=large_member.getvalue(),
            destination_dir=tmp_path / "large",
            original_pdf=pdf,
            max_member_bytes=8,
            max_total_uncompressed_bytes=64,
        )
    assert not (tmp_path / "large" / "large.txt").exists()


def test_mineru_api_client_times_out_without_resubmission(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    calls = {"submit": 0, "status": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks":
            calls["submit"] += 1
            return httpx.Response(202, json={"task_id": "task-123"})
        if request.url.path == "/tasks/task-123":
            calls["status"] += 1
            return httpx.Response(200, json={"task_id": "task-123", "state": "processing", "queued_ahead": 1})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = MinerUApiClient(
        base_url="http://127.0.0.1:8000",
        max_poll_attempts=2,
        transport=httpx.MockTransport(handler),
    )
    request = DocumentParseRequest(
        run_id="mineru-api",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="mineru_api",
    )

    with pytest.raises(MinerUApiError, match="polling exceeded"):
        client.parse_pdf(request=request, input_pdf=pdf, output_dir=tmp_path / "bundle")
    assert calls == {"submit": 1, "status": 2}
