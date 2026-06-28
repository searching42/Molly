from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai4s_agent.document_parse_live_acceptance import (
    DocumentParseAcceptanceThresholds,
    DocumentParseLiveAcceptanceReport,
    main,
    run_document_parse_live_acceptance,
)
from document_parse_test_helpers import fixture_mineru_output_dir


def _mineru_zip_payload(*, nested: bool = True, quality_miss: bool = False) -> bytes:
    bundle = fixture_mineru_output_dir()
    content_list = json.loads((bundle / "synthetic_content_list.json").read_text(encoding="utf-8"))
    if quality_miss:
        for item in content_list:
            if item.get("type") == "table":
                item["table_body"] = "<table><tr><th>SMILES</th><th>PLQY</th><th>lambda_em</th></tr><tr><td>CCO</td><td>0.10</td><td>999</td></tr></table>"
    prefix = Path("paper") / "hybrid_auto" if nested else Path("")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            arcname = prefix / path.relative_to(bundle)
            if path.name == "synthetic_content_list.json":
                archive.writestr(str(arcname), json.dumps(content_list))
            else:
                archive.write(path, arcname=str(arcname))
    return payload.getvalue()


def _stream_response(content: bytes, *, headers: dict[str, str] | None = None, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, stream=httpx.ByteStream(content), headers=headers)


def _success_transport(*, token_seen: list[str] | None = None, calls: dict[str, int] | None = None) -> httpx.MockTransport:
    zip_payload = _mineru_zip_payload()
    counters = calls if calls is not None else {"submit": 0, "status": 0, "result": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if token_seen is not None:
            token_seen.append(str(request.headers.get("authorization") or ""))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version_name": "mineru-live-test", "protocol_version": 2})
        if request.url.path == "/tasks":
            counters["submit"] = counters.get("submit", 0) + 1
            return httpx.Response(202, json={"task_id": "task-live-123"})
        if request.url.path == "/tasks/task-live-123":
            counters["status"] = counters.get("status", 0) + 1
            return httpx.Response(
                200,
                json={
                    "task_id": "task-live-123",
                    "state": "completed",
                    "_backend": "hybrid-engine",
                    "_version_name": "mineru-live-test",
                    "protocol_version": 2,
                    "queued_ahead": 0,
                },
            )
        if request.url.path == "/tasks/task-live-123/result":
            counters["result"] = counters.get("result", 0) + 1
            return _stream_response(zip_payload)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    return httpx.MockTransport(handler)


def _quality_miss_transport() -> httpx.MockTransport:
    zip_payload = _mineru_zip_payload(quality_miss=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "protocol_version": 2})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-quality"})
        if request.url.path == "/tasks/task-quality":
            return httpx.Response(200, json={"task_id": "task-quality", "state": "completed", "protocol_version": 2})
        if request.url.path == "/tasks/task-quality/result":
            return _stream_response(zip_payload)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    return httpx.MockTransport(handler)


def _protocol_transport(*, health_protocol: Any = 2, status_protocol: Any = 2) -> httpx.MockTransport:
    zip_payload = _mineru_zip_payload()

    def with_protocol(payload: dict[str, Any], value: Any) -> dict[str, Any]:
        if value is not None:
            payload["protocol_version"] = value
        return payload

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=with_protocol({"status": "ok"}, health_protocol))
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-protocol"})
        if request.url.path == "/tasks/task-protocol":
            return httpx.Response(200, json=with_protocol({"task_id": "task-protocol", "state": "completed"}, status_protocol))
        if request.url.path == "/tasks/task-protocol/result":
            return _stream_response(zip_payload)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    return httpx.MockTransport(handler)


def test_live_acceptance_passes_with_mock_mineru_and_pdfplumber_baseline(tmp_path: Path) -> None:
    calls = {"submit": 0, "status": 0, "result": 0}

    report = run_document_parse_live_acceptance(
        run_id="mineru-live-smoke",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        compare_pdfplumber=True,
        transport=_success_transport(calls=calls),
    )

    assert report.decision == "passed"
    assert report.mineru.ok is True
    assert report.mineru.remote_task_id == "task-live-123"
    assert report.mineru.task_status_history == ["completed"]
    assert report.mineru.protocol_version == "2"
    assert report.mineru.markdown_path == "mineru/mineru_bundle/paper/hybrid_auto/synthetic.md"
    assert report.pdfplumber is not None
    assert report.pdfplumber.ok is True
    assert report.comparison is not None
    assert report.comparison.mineru_better_fields == []
    assert calls == {"submit": 1, "status": 1, "result": 1}
    assert (tmp_path / "acceptance" / "mineru-live-smoke" / "acceptance_report.json").exists()
    assert (tmp_path / "acceptance" / "mineru-live-smoke" / "acceptance_summary.md").exists()
    assert DocumentParseLiveAcceptanceReport.model_validate_json(
        (tmp_path / "acceptance" / "mineru-live-smoke" / "acceptance_report.json").read_text(encoding="utf-8")
    ).decision == "passed"
    assert report.outputs["source_pdf"].endswith("synthetic_source.pdf")
    assert report.outputs["acceptance_report"] == "acceptance_report.json"
    run_root = tmp_path / "acceptance" / "mineru-live-smoke"
    for rel_path in [
        report.outputs["source_pdf"],
        report.outputs["acceptance_report"],
        report.outputs["acceptance_summary"],
        report.mineru.markdown_path,
        report.mineru.content_list_path,
        report.mineru.middle_json_path,
        report.mineru.parsed_document_path,
        report.mineru.parser_audit_path,
        report.pdfplumber.parsed_document_path,
        report.pdfplumber.parser_audit_path,
    ]:
        resolved = (run_root / rel_path).resolve()
        assert run_root.resolve() in resolved.parents or resolved == run_root.resolve()
        assert resolved.exists(), rel_path
    assert report.mineru.markdown_path == "mineru/mineru_bundle/paper/hybrid_auto/synthetic.md"
    mineru_audit = json.loads((run_root / report.mineru.parser_audit_path).read_text(encoding="utf-8"))
    assert mineru_audit["source_pdf_sha256"] == report.source_pdf_sha256


@pytest.mark.parametrize(
    ("health_protocol", "status_protocol", "expected_code"),
    [
        (None, None, "missing_protocol_version"),
        (2, 1, "unsupported_protocol_version"),
    ],
)
def test_live_acceptance_requires_mineru_protocol_v2(
    tmp_path: Path,
    health_protocol: Any,
    status_protocol: Any,
    expected_code: str,
) -> None:
    report = run_document_parse_live_acceptance(
        run_id=f"mineru-protocol-{expected_code}",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=_protocol_transport(health_protocol=health_protocol, status_protocol=status_protocol),
    )

    assert report.decision == "failed"
    assert any(error.code == expected_code for error in report.errors)


def test_live_acceptance_needs_review_when_threshold_misses(tmp_path: Path) -> None:
    report = run_document_parse_live_acceptance(
        run_id="mineru-needs-review",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        thresholds=DocumentParseAcceptanceThresholds(simple_cell_exact_match_rate=0.90),
        transport=_quality_miss_transport(),
    )

    assert report.decision == "needs_review"
    assert any(error.code == "threshold_miss" for error in report.errors)


def test_live_acceptance_failed_health_persists_report(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503, json={"status": "down"})
        raise AssertionError("provider must stop after failed health")

    report = run_document_parse_live_acceptance(
        run_id="mineru-health-failed",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=httpx.MockTransport(handler),
    )

    assert report.decision == "failed"
    assert report.mineru.ok is False
    assert report.mineru.error is not None
    assert report.mineru.error.code == "health_check_failure"
    assert (tmp_path / "acceptance" / "mineru-health-failed" / "acceptance_report.json").exists()


def test_live_acceptance_failed_submission_timeout_download_and_unsafe_zip(tmp_path: Path) -> None:
    scenarios: list[tuple[str, httpx.MockTransport, str]] = []

    def submission_failure(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(500, json={"error": "submit failed"})
        raise AssertionError(f"unexpected {request.url.path}")

    scenarios.append(("submission-failure", httpx.MockTransport(submission_failure), "submission_failure"))

    def polling_timeout(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-timeout"})
        if request.url.path == "/tasks/task-timeout":
            return httpx.Response(200, json={"task_id": "task-timeout", "state": "processing"})
        raise AssertionError(f"unexpected {request.url.path}")

    scenarios.append(("polling-timeout", httpx.MockTransport(polling_timeout), "task_timeout"))

    def result_download_failure(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-download"})
        if request.url.path == "/tasks/task-download":
            return httpx.Response(200, json={"task_id": "task-download", "state": "completed"})
        if request.url.path == "/tasks/task-download/result":
            return _stream_response(b'{"error":"download failed"}', headers={"content-type": "application/json"}, status_code=500)
        raise AssertionError(f"unexpected {request.url.path}")

    scenarios.append(("result-download", httpx.MockTransport(result_download_failure), "result_download_failure"))

    malicious = io.BytesIO()
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    def unsafe_zip(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-zip"})
        if request.url.path == "/tasks/task-zip":
            return httpx.Response(200, json={"task_id": "task-zip", "state": "completed"})
        if request.url.path == "/tasks/task-zip/result":
            return _stream_response(malicious.getvalue())
        raise AssertionError(f"unexpected {request.url.path}")

    scenarios.append(("unsafe-zip", httpx.MockTransport(unsafe_zip), "unsafe_result_archive"))

    for run_suffix, transport, expected_code in scenarios:
        report = run_document_parse_live_acceptance(
            run_id=f"mineru-{run_suffix}",
            output_dir=tmp_path / "acceptance",
            api_url="http://127.0.0.1:8000",
            endpoint_kind="mineru_api",
            task_timeout_sec=0.02,
            poll_interval_sec=0.01,
            max_poll_attempts=2,
            transport=transport,
        )
        assert report.decision == "failed"
        assert report.mineru.error is not None
        assert report.mineru.error.code == expected_code
        assert (tmp_path / "acceptance" / f"mineru-{run_suffix}" / "acceptance_report.json").exists()


def test_live_acceptance_invalid_output_bundle_persists_failure_audit(tmp_path: Path) -> None:
    invalid_zip = io.BytesIO()
    with zipfile.ZipFile(invalid_zip, "w") as archive:
        archive.writestr("only.txt", "not a MinerU bundle")

    def invalid_bundle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-invalid"})
        if request.url.path == "/tasks/task-invalid":
            return httpx.Response(200, json={"task_id": "task-invalid", "state": "completed"})
        if request.url.path == "/tasks/task-invalid/result":
            return _stream_response(invalid_zip.getvalue())
        raise AssertionError(f"unexpected {request.url.path}")

    report = run_document_parse_live_acceptance(
        run_id="mineru-invalid-bundle",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=httpx.MockTransport(invalid_bundle),
    )

    assert report.decision == "failed"
    assert report.mineru.error is not None
    assert report.mineru.error.code == "mineru_parse_failed"
    assert report.mineru.parser_audit_path
    assert (tmp_path / "acceptance" / "mineru-invalid-bundle" / report.mineru.parser_audit_path).exists()


def test_live_acceptance_security_redacts_token_and_rejects_unsafe_inputs(tmp_path: Path, monkeypatch: Any) -> None:
    token_seen: list[str] = []
    monkeypatch.setenv("MINERU_API_TOKEN", "secret-token")
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "--api-url",
            "http://127.0.0.1:8000",
            "--endpoint-kind",
            "mineru-api",
            "--output",
            str(tmp_path / "acceptance"),
            "--run-id",
            "mineru-cli",
            "--compare-pdfplumber",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=_success_transport(token_seen=token_seen),
    )

    payload = json.loads(stdout.getvalue())
    report_text = (tmp_path / "acceptance" / "mineru-cli" / "acceptance_report.json").read_text(encoding="utf-8")
    summary_text = (tmp_path / "acceptance" / "mineru-cli" / "acceptance_summary.md").read_text(encoding="utf-8")
    assert code == 0
    assert payload["decision"] == "passed"
    assert any("secret-token" in header for header in token_seen)
    assert "secret-token" not in stdout.getvalue()
    assert "secret-token" not in stderr.getvalue()
    assert "secret-token" not in report_text
    assert "secret-token" not in summary_text
    assert payload["redacted_api_origin"] == "http://127.0.0.1:8000"

    bad = run_document_parse_live_acceptance(
        run_id="bad-url",
        output_dir=tmp_path / "bad",
        api_url="https://token@example.com/mineru?secret=yes#frag",
        endpoint_kind="compatible_endpoint",
        transport=_success_transport(),
    )
    assert bad.decision == "failed"
    assert any(error.code == "invalid_api_url" for error in bad.errors)

    remote = run_document_parse_live_acceptance(
        run_id="remote-upload-denied",
        output_dir=tmp_path / "remote",
        api_url="https://mineru.example.com",
        endpoint_kind="compatible_endpoint",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"})),
    )
    assert remote.decision == "failed"
    assert remote.mineru.error is not None
    assert "allow_remote_upload" in remote.mineru.error.message


def test_live_acceptance_cli_uses_endpoint_profile_without_leaking_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "mineru-profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "mineru_endpoint_profiles.v1",
                "profiles": [
                    {
                        "name": "node45-loopback",
                        "api_url": "http://127.0.0.1:8000",
                        "endpoint_kind": "mineru-api",
                        "backend": "hybrid-engine",
                        "effort": "medium",
                        "allow_remote_upload": True,
                        "compare_pdfplumber": True,
                    }
                ],
                "routing_policies": [
                    {
                        "name": "manual-primary",
                        "default_profile": "node45-loopback",
                        "fallback_profiles": [],
                        "mode": "manual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    token_seen: list[str] = []
    monkeypatch.setenv("MINERU_API_TOKEN", "profile-secret-token")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--endpoint-profile-file",
            str(profile_path),
            "--routing-policy",
            "manual-primary",
            "--output",
            str(tmp_path / "acceptance"),
            "--run-id",
            "mineru-profile-cli",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=_success_transport(token_seen=token_seen),
    )

    payload = json.loads(stdout.getvalue())
    report_text = (tmp_path / "acceptance" / "mineru-profile-cli" / "acceptance_report.json").read_text(
        encoding="utf-8"
    )
    assert code == 0
    assert payload["decision"] == "passed"
    assert payload["endpoint_profile"]["endpoint_profile_name"] == "node45-loopback"
    assert payload["endpoint_profile"]["routing_policy_name"] == "manual-primary"
    assert payload["endpoint_profile"]["redacted_api_origin"] == "http://127.0.0.1:8000"
    assert payload["requested_backend"] == "hybrid-engine"
    assert payload["mineru"]["ok"] is True
    assert any("profile-secret-token" in header for header in token_seen)
    assert "profile-secret-token" not in stdout.getvalue()
    assert "profile-secret-token" not in stderr.getvalue()
    assert "profile-secret-token" not in report_text


def test_live_acceptance_cli_profile_errors_are_structured_and_redacted(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad-profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "mineru_endpoint_profiles.v1",
                "profiles": [
                    {
                        "name": "bad",
                        "api_url": "http://127.0.0.1:8000?token=super-secret",
                        "endpoint_kind": "mineru-api",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [
            "--endpoint-profile-file",
            str(profile_path),
            "--endpoint-profile",
            "bad",
            "--output",
            str(tmp_path / "acceptance"),
            "--run-id",
            "bad-profile",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=_success_transport(),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 1
    assert payload["decision"] == "failed"
    assert payload["errors"][0]["code"] == "endpoint_profile_error"
    assert "super-secret" not in stdout.getvalue()
    assert "super-secret" not in stderr.getvalue()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "nested\\run", ".", ".."])
def test_live_acceptance_rejects_unsafe_run_id_without_path_escape(tmp_path: Path, run_id: str) -> None:
    report = run_document_parse_live_acceptance(
        run_id=run_id,
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=_success_transport(),
    )

    assert report.decision == "failed"
    assert any(error.code == "invalid_run_id" for error in report.errors)
    assert not (tmp_path / "escape").exists()


def test_live_acceptance_rejects_absolute_run_id_without_path_escape(tmp_path: Path) -> None:
    escaped_root = tmp_path / "absolute-escape"
    report = run_document_parse_live_acceptance(
        run_id=str(escaped_root),
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=_success_transport(),
    )

    assert report.decision == "failed"
    assert any(error.code == "invalid_run_id" for error in report.errors)
    assert not escaped_root.exists()


def test_live_acceptance_invalid_configuration_returns_failed_report(tmp_path: Path) -> None:
    timing = run_document_parse_live_acceptance(
        run_id="bad-timing",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        task_timeout_sec=0,
        transport=_success_transport(),
    )
    assert timing.decision == "failed"
    assert any(error.code == "configuration_error" for error in timing.errors)
    assert (tmp_path / "acceptance" / "bad-timing" / "acceptance_report.json").exists()

    threshold = run_document_parse_live_acceptance(
        run_id="bad-threshold",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        thresholds=DocumentParseAcceptanceThresholds(normalized_text_token_recall=1.1),
        transport=_success_transport(),
    )
    assert threshold.decision == "failed"
    assert any(error.code == "invalid_threshold" for error in threshold.errors)


def test_live_acceptance_non_empty_run_dir_is_rejected_before_invalid_config_overwrite(tmp_path: Path) -> None:
    run_root = tmp_path / "acceptance" / "existing-run"
    first = run_document_parse_live_acceptance(
        run_id="existing-run",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=_success_transport(),
    )
    original_report = (run_root / "acceptance_report.json").read_text(encoding="utf-8")
    original_summary = (run_root / "acceptance_summary.md").read_text(encoding="utf-8")

    second = run_document_parse_live_acceptance(
        run_id="existing-run",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        task_timeout_sec=0,
        transport=_success_transport(),
    )

    assert first.decision == "passed"
    assert second.decision == "failed"
    assert any(error.code == "output_directory_not_empty" for error in second.errors)
    assert not any(error.code == "configuration_error" for error in second.errors)
    assert (run_root / "acceptance_report.json").read_text(encoding="utf-8") == original_report
    assert (run_root / "acceptance_summary.md").read_text(encoding="utf-8") == original_summary


def test_live_acceptance_cli_exit_codes_and_rejects_reused_output_dir(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    needs_review_code = main(
        [
            "--api-url",
            "http://127.0.0.1:8000",
            "--output",
            str(tmp_path / "needs-review"),
            "--run-id",
            "needs-review",
            "--min-cell-match",
            "0.90",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=_quality_miss_transport(),
    )
    assert needs_review_code == 2
    assert json.loads(stdout.getvalue())["decision"] == "needs_review"

    stdout = io.StringIO()
    stderr = io.StringIO()
    failed_code = main(
        [
            "--api-url",
            "http://127.0.0.1:8000",
            "--output",
            str(tmp_path / "failed"),
            "--run-id",
            "failed",
        ],
        stdout=stdout,
        stderr=stderr,
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"status": "down"})),
    )
    assert failed_code == 1
    assert json.loads(stdout.getvalue())["decision"] == "failed"

    reused_root = tmp_path / "reuse"
    reused_run = reused_root / "same-run"
    reused_run.mkdir(parents=True)
    (reused_run / "old.txt").write_text("old", encoding="utf-8")
    reused = run_document_parse_live_acceptance(
        run_id="same-run",
        output_dir=reused_root,
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_api",
        transport=_success_transport(),
    )
    assert reused.decision == "failed"
    assert any(error.code == "output_directory_not_empty" for error in reused.errors)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with pytest.raises(SystemExit):
        main(
            [
                "--api-url",
                "http://127.0.0.1:8000",
                "--output",
                str(tmp_path / "low-effort"),
                "--run-id",
                "low-effort",
                "--effort",
                "low",
            ],
            stdout=stdout,
            stderr=stderr,
            transport=_success_transport(),
        )
    assert "invalid choice" in stderr.getvalue()


def test_live_acceptance_comparison_uses_same_pdf_and_no_fallback(tmp_path: Path) -> None:
    calls = {"submit": 0, "status": 0, "result": 0}
    report = run_document_parse_live_acceptance(
        run_id="mineru-comparison",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:8000",
        endpoint_kind="mineru_router",
        compare_pdfplumber=True,
        transport=_success_transport(calls=calls),
    )

    assert report.endpoint_kind == "mineru_router"
    assert report.mineru.source_pdf_sha256 == report.pdfplumber.source_pdf_sha256
    assert report.comparison is not None
    assert report.comparison.provider_success == {"mineru_api": True, "pdfplumber": True}
    assert report.comparison.mineru_better_fields == []
    assert report.comparison.pdfplumber_better_fields == []
    assert calls["submit"] == 1
    assert report.mineru.provider == "mineru_api"
    assert report.pdfplumber.provider == "pdfplumber"
