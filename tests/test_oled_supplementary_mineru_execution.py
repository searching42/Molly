from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from reportlab.pdfgen.canvas import Canvas

from ai4s_agent._utils import write_json
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.domains.oled_supplementary_mineru_execution import (
    OledSupplementaryMineruExecutionArtifact,
    OledSupplementaryMineruExecutionManifest,
    OledSupplementaryMineruExecutionStatus,
)
from ai4s_agent.domains.oled_supplementary_parser_preflight import (
    OledSupplementaryParserPreflightPlan,
)
from ai4s_agent.mineru_endpoint_preflight import (
    MinerUEndpointEnvironmentDiagnostics,
    MinerUEndpointHealthSummary,
    MinerUEndpointPreflightReport,
)
from ai4s_agent.mineru_endpoint_profiles import MinerUEndpointProfileReportSummary
from ai4s_agent.oled_supplementary_mineru_execution import (
    execute_oled_supplementary_mineru_from_files,
    main,
)
from ai4s_agent.oled_supplementary_parser_preflight import OledSupplementaryParserPreflightArtifact
from ai4s_agent.schemas import ParsedDocument
from document_parse_test_helpers import build_zip_from_dir, fixture_mineru_output_dir


_GENERATED_AT = "2026-07-13T10:00:00Z"


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_pdf(path: Path, *, page_count: int = 2, text: str = "supplementary") -> Path:
    canvas = Canvas(str(path))
    for page_number in range(1, page_count + 1):
        canvas.drawString(72, 720, f"{text} page {page_number}")
        canvas.showPage()
    canvas.save()
    return path


def _preflight_artifact(pdf_paths: list[Path]) -> OledSupplementaryParserPreflightArtifact:
    source_envelopes = []
    items = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        source_id = f"supp-source-{index:03d}"
        source_sha256 = _sha256_file(pdf_path)
        source_envelopes.append(
            {
                "source_id": source_id,
                "pdf_sha256": source_sha256,
                "byte_size": pdf_path.stat().st_size,
                "page_count": 2,
                "content_type": "application/pdf",
                "pdf_header_valid": True,
                "pdf_eof_marker_valid": True,
                "page_count_validated": True,
            }
        )
        items.append(
            {
                "recovery_item_id": f"supplementary-recovery:item-{index:03d}",
                "source_id": source_id,
                "source_pdf_sha256": source_sha256,
                "target_kind": "table",
                "target_locator": f"S{index}",
                "parse_scope": "full_source_then_locator_review",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "oled_supplementary_parser_preflight_plan.v1",
        "paper_id": "paper016",
        "source_request_digest": "sha256:" + "1" * 64,
        "source_mapping_result_digest": "sha256:" + "2" * 64,
        "source_context_digest": "sha256:" + "3" * 64,
        "recovery_plan_digest": "sha256:" + "4" * 64,
        "intake_plan_digest": "sha256:" + "5" * 64,
        "parse_confirmed": True,
        "reviewed_by": "reviewer-02",
        "reviewed_at": _GENERATED_AT,
        "source_envelopes": source_envelopes,
        "items": items,
        "source_count": len(source_envelopes),
        "item_count": len(items),
        "preflight_plan_digest": "",
        "review_only": True,
        "executable": False,
        "offline_only": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "pdf_content_parsed": False,
        "pdf_page_count_validated": True,
        "supplementary_downloaded": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["preflight_plan_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "preflight_plan_digest"}
    )
    plan = OledSupplementaryParserPreflightPlan.model_validate(payload)
    return OledSupplementaryParserPreflightArtifact(
        run_id="run-paper016",
        paper_id=plan.paper_id,
        generated_at=_GENERATED_AT,
        source_request_digest=plan.source_request_digest,
        source_mapping_result_digest=plan.source_mapping_result_digest,
        source_context_digest=plan.source_context_digest,
        recovery_plan_digest=plan.recovery_plan_digest,
        intake_plan_digest=plan.intake_plan_digest,
        preflight_plan_digest=plan.preflight_plan_digest,
        preflight_plan=plan,
    )


def _write_endpoint_inputs(
    tmp_path: Path,
    *,
    health_path: str = "/health",
    health_evidence_path: str | None = None,
) -> tuple[Path, Path, str]:
    profile_path = tmp_path / "mineru-profile.json"
    write_json(
        profile_path,
        {
            "schema_version": "mineru_endpoint_profiles.v1",
            "profiles": [
                {
                    "name": "node45-loopback",
                    "api_url": "http://127.0.0.1:18000",
                    "endpoint_kind": "mineru-api",
                    "backend": "hybrid-engine",
                    "effort": "medium",
                    "parse_method": "auto",
                    "allow_remote_upload": True,
                    "compare_pdfplumber": True,
                    "http_timeout_sec": 60,
                    "task_timeout_sec": 900,
                    "poll_interval_sec": 1,
                    "max_poll_attempts": 600,
                    "expected_protocol_version": "2",
                    "health_path": health_path,
                    "notes": [],
                }
            ],
            "routing_policies": [],
        },
    )
    report = MinerUEndpointPreflightReport(
        run_id="endpoint-preflight-001",
        generated_at=_GENERATED_AT,
        decision="passed",
        profile=MinerUEndpointProfileReportSummary(
            endpoint_profile_name="node45-loopback",
            profile_source_path="mineru-profile.json",
            redacted_api_origin="http://127.0.0.1:18000",
            endpoint_kind="mineru_api",
            backend="hybrid-engine",
            effort="medium",
            parse_method="auto",
            allow_remote_upload=True,
            compare_pdfplumber=True,
            http_timeout_sec=60,
            task_timeout_sec=900,
            poll_interval_sec=1,
            max_poll_attempts=600,
            expected_protocol_version="2",
            health_path=health_path,
            routing_fallback_profile_names=[],
        ),
        health=MinerUEndpointHealthSummary(
            ok=True,
            http_status_code=200,
            status="healthy",
            mineru_version="mineru-2.7",
            protocol_version="2",
            response_schema_valid=True,
            health_path=health_evidence_path or health_path,
        ),
        environment=MinerUEndpointEnvironmentDiagnostics(),
    )
    report_path = tmp_path / "endpoint-preflight-report.json"
    write_json(report_path, report.model_dump(mode="json"))
    return profile_path, report_path, _sha256_file(report_path)


def _write_execution_inputs(
    tmp_path: Path,
    pdf_paths: list[Path],
    *,
    endpoint_report_sha256: str | None = None,
    run_id: str = "supp-mineru-run-001",
    health_path: str = "/health",
    health_evidence_path: str | None = None,
) -> tuple[Path, Path, Path, Path, OledSupplementaryParserPreflightArtifact]:
    preflight = _preflight_artifact(pdf_paths)
    profile_path, endpoint_report_path, observed_report_sha256 = _write_endpoint_inputs(
        tmp_path,
        health_path=health_path,
        health_evidence_path=health_evidence_path,
    )
    manifest = OledSupplementaryMineruExecutionManifest(
        run_id=run_id,
        paper_id=preflight.paper_id,
        preflight_plan_digest=preflight.preflight_plan_digest,
        execution_confirmed=True,
        reviewed_by="reviewer-03",
        reviewed_at=_GENERATED_AT,
        endpoint_profile_name="node45-loopback",
        endpoint_preflight_sha256=endpoint_report_sha256 or observed_report_sha256,
        sources=[
            {
                "source_id": f"supp-source-{index:03d}",
                "local_pdf_path": str(pdf_path),
            }
            for index, pdf_path in enumerate(pdf_paths, start=1)
        ],
    )
    preflight_path = tmp_path / "parser-preflight.json"
    manifest_path = tmp_path / "execution-manifest.json"
    write_json(preflight_path, preflight.model_dump(mode="json"))
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return preflight_path, manifest_path, profile_path, endpoint_report_path, preflight


class _FakeHealthClient:
    def __init__(self, *, protocol_version: str = "2", health_path: str = "/health") -> None:
        self.protocol_version = protocol_version
        self.health_path = health_path
        self.calls = 0

    def health(self) -> dict[str, str]:
        self.calls += 1
        return {
            "status": "healthy",
            "protocol_version": self.protocol_version,
            "version_name": "mineru-2.7",
        }


class _FakeMineruService:
    def __init__(
        self,
        *,
        protocol_version: str = "2",
        fail_call: int | None = None,
        observed_provider: str = "mineru_api",
        audit_hash_override: str = "",
        observed_backend: str = "hybrid-engine",
        escape_output: bool = False,
        symlink_output: bool = False,
    ) -> None:
        self.client = _FakeHealthClient(protocol_version=protocol_version)
        self.mineru_provider = SimpleNamespace(client=self.client)
        self.fail_call = fail_call
        self.observed_provider = observed_provider
        self.audit_hash_override = audit_hash_override
        self.observed_backend = observed_backend
        self.escape_output = escape_output
        self.symlink_output = symlink_output
        self.requests: list[DocumentParseRequest] = []

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        self.requests.append(request)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        parsed = ParsedDocument(
            paper_id="paper016",
            source_path=request.input_pdf,
            parser_backend=f"mineru_api:{self.observed_backend}",
            pages=[{"page": 1}, {"page": 2}],
            elements=[],
            tables=[],
        )
        parsed_json = write_json(output_dir / "parsed_document.json", parsed.model_dump(mode="json"))
        parsed_md = output_dir / "parsed_document.md"
        parsed_md.write_text("# Parsed supplementary source\n", encoding="utf-8")
        audit_json = write_json(
            output_dir / "parser_audit.json",
            {"source_pdf_sha256": request.expected_source_pdf_sha256, "provider": self.observed_provider},
        )
        content_list = write_json(output_dir / "content_list.json", {"items": []})
        if self.escape_output:
            parsed_json = (output_dir.parent / "escaped.json")
            parsed_json.write_text("{}", encoding="utf-8")
        elif self.symlink_output:
            target_json = output_dir / "parsed_document_target.json"
            parsed_json.replace(target_json)
            parsed_json.symlink_to(target_json)
        should_fail = self.fail_call == len(self.requests)
        return DocumentParseResult(
            ok=not should_fail,
            status="failed" if should_fail else "success",
            provider=self.observed_provider,
            parser_backend=(
                "pdfplumber_local"
                if self.observed_provider != "mineru_api"
                else f"mineru_api:{self.observed_backend}"
            ),
            run_id=request.run_id,
            input_pdf=request.input_pdf,
            parsed_document=None if should_fail else parsed,
            outputs=DocumentParseOutputRefs(
                output_dir=str(output_dir),
                parsed_document_json=str(parsed_json),
                parsed_document_markdown=str(parsed_md),
                parser_audit_json=str(audit_json),
                content_list_json=str(content_list),
            ),
            remote_task_id="task-001",
            warnings=[],
            error=(
                DocumentParseError(code="task_failed", message="failed", details={})
                if should_fail
                else None
            ),
            audit=DocumentParseAudit(
                source_pdf_sha256=self.audit_hash_override or request.expected_source_pdf_sha256,
                request_provider=request.provider,
                selected_provider=self.observed_provider,
                selection_reason="explicit_mineru_api_provider",
                parser_backend=(
                    "pdfplumber_local"
                    if self.observed_provider != "mineru_api"
                    else f"mineru_api:{self.observed_backend}"
                ),
                task_status_history=["failed" if should_fail else "completed"],
                queued_ahead_history=[],
                extracted_relative_paths=[],
                warnings=[],
                mineru_version="mineru-2.7",
                protocol_version="2",
            ),
        )


def _execute(
    tmp_path: Path,
    pdf_paths: list[Path],
    service: _FakeMineruService,
    *,
    endpoint_report_sha256: str | None = None,
    run_id: str = "supp-mineru-run-001",
):
    preflight_path, manifest_path, profile_path, endpoint_report_path, preflight = _write_execution_inputs(
        tmp_path,
        pdf_paths,
        endpoint_report_sha256=endpoint_report_sha256,
        run_id=run_id,
    )
    artifact = execute_oled_supplementary_mineru_from_files(
        preflight_artifact_json=preflight_path,
        execution_manifest_json=manifest_path,
        endpoint_profile_config_json=profile_path,
        endpoint_preflight_report_json=endpoint_report_path,
        output_root=tmp_path / "runs",
        service=service,
        generated_at=_GENERATED_AT,
    )
    return artifact, preflight, (preflight_path, manifest_path, profile_path, endpoint_report_path)


def test_bound_execution_uses_explicit_mineru_and_writes_redacted_hash_audit(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()

    artifact, preflight, _ = _execute(tmp_path, [pdf_path], service)

    assert artifact.status == OledSupplementaryMineruExecutionStatus.SUCCESS
    assert artifact.preflight_plan_digest == preflight.preflight_plan_digest
    assert artifact.source_count == 1
    assert artifact.successful_source_count == 1
    assert artifact.mineru_called is True
    assert artifact.pdf_content_parsed is True
    assert artifact.locator_resolved is False
    assert artifact.candidate_regenerated is False
    assert artifact.device_only_admitted is False
    assert artifact.dataset_written is False
    request = service.requests[0]
    assert request.provider == "mineru_api"
    assert request.expected_source_pdf_sha256 == _sha256_file(pdf_path)
    assert request.start_page is None and request.end_page is None
    assert request.table_enabled is True
    assert request.image_analysis_enabled is False
    snapshot = tmp_path / "runs" / artifact.run_id / "sources" / "supp-source-001" / "approved_source.pdf"
    assert snapshot.read_bytes() == pdf_path.read_bytes()
    stored_path = tmp_path / "runs" / artifact.run_id / "supplementary_mineru_execution.json"
    stored_text = stored_path.read_text(encoding="utf-8")
    assert str(pdf_path) not in stored_text
    assert "local_pdf_path" not in stored_text
    output_kinds = {item.output_kind.value for item in artifact.source_results[0].output_hashes}
    assert {"parsed_document_json", "parsed_document_markdown", "parser_audit_json"}.issubset(output_kinds)


def test_execution_rejects_changed_pdf_before_health_or_parse(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf", text="approved")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(tmp_path, [pdf_path])
    _write_pdf(pdf_path, text="replaced")

    with pytest.raises(ValueError, match="hash no longer matches|byte size no longer matches"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert service.client.calls == 0
    assert service.requests == []


def test_execution_rejects_symlinked_source_rebinding_before_health(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    link_path = tmp_path / "paper016_si_link.pdf"
    try:
        link_path.symlink_to(pdf_path)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(tmp_path, [pdf_path])
    manifest_payload = json.loads(inputs[1].read_text(encoding="utf-8"))
    manifest_payload["sources"][0]["local_pdf_path"] = str(link_path)
    write_json(inputs[1], manifest_payload)

    with pytest.raises(ValueError, match="could not be snapshotted safely"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert service.client.calls == 0
    assert service.requests == []


def test_execution_rejects_endpoint_report_hash_mismatch_before_output_creation(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(
        tmp_path,
        [pdf_path],
        endpoint_report_sha256="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="report hash does not match"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert not (tmp_path / "runs").exists()
    assert service.requests == []


def test_execution_rejects_health_evidence_path_mismatch_before_output_creation(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(
        tmp_path,
        [pdf_path],
        health_path="/api/health",
        health_evidence_path="/health",
    )

    with pytest.raises(ValueError, match="health evidence path"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert not (tmp_path / "runs").exists()
    assert service.client.calls == 0
    assert service.requests == []


def test_execution_manifest_must_exactly_cover_preflight_sources(tmp_path: Path) -> None:
    first_pdf = _write_pdf(tmp_path / "paper016_si_a.pdf", text="first")
    second_pdf = _write_pdf(tmp_path / "paper016_si_b.pdf", text="second")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(tmp_path, [first_pdf, second_pdf])
    manifest_payload = json.loads(inputs[1].read_text(encoding="utf-8"))
    manifest_payload["sources"] = manifest_payload["sources"][:1]
    write_json(inputs[1], manifest_payload)

    with pytest.raises(ValueError, match="exactly cover parser preflight sources"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert not (tmp_path / "runs").exists()
    assert service.requests == []


def test_execution_requires_fresh_live_protocol_before_parse(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService(protocol_version="3")
    inputs = _write_execution_inputs(tmp_path, [pdf_path])

    with pytest.raises(ValueError, match="live MinerU protocol"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert service.client.calls == 1
    assert service.requests == []


def test_execution_rejects_live_client_health_path_mismatch_before_health_call(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()
    service.client.health_path = "/different-health"
    inputs = _write_execution_inputs(tmp_path, [pdf_path])

    with pytest.raises(ValueError, match="live MinerU health path"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert service.client.calls == 0
    assert service.requests == []


def test_execution_live_health_uses_bound_custom_health_path(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    inputs = _write_execution_inputs(tmp_path, [pdf_path], health_path="/api/health")
    bundle = build_zip_from_dir(fixture_mineru_output_dir())
    paths_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_seen.append(request.url.path)
        if request.method == "GET" and request.url.path == "/api/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "protocol_version": "2",
                    "version_name": "mineru-2.7",
                },
            )
        if request.method == "POST" and request.url.path == "/tasks":
            return httpx.Response(202, json={"task_id": "task-custom-health"})
        if request.method == "GET" and request.url.path == "/tasks/task-custom-health":
            return httpx.Response(
                200,
                json={
                    "task_id": "task-custom-health",
                    "state": "completed",
                    "backend": "hybrid-engine",
                    "protocol_version": "2",
                    "version_name": "mineru-2.7",
                },
            )
        if request.method == "GET" and request.url.path == "/tasks/task-custom-health/result":
            return httpx.Response(
                200,
                stream=httpx.ByteStream(bundle),
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    artifact = execute_oled_supplementary_mineru_from_files(
        preflight_artifact_json=inputs[0],
        execution_manifest_json=inputs[1],
        endpoint_profile_config_json=inputs[2],
        endpoint_preflight_report_json=inputs[3],
        output_root=tmp_path / "runs",
        transport=httpx.MockTransport(handler),
        generated_at=_GENERATED_AT,
    )

    assert artifact.status == OledSupplementaryMineruExecutionStatus.SUCCESS
    assert paths_seen[0] == "/api/health"
    assert paths_seen.count("/api/health") == 1
    assert "/health" not in paths_seen


@pytest.mark.parametrize(
    ("service", "expected_error_code"),
    [
        (_FakeMineruService(observed_provider="pdfplumber"), "parser_result_binding_failed"),
        (_FakeMineruService(audit_hash_override="sha256:" + "0" * 64), "parser_result_binding_failed"),
        (_FakeMineruService(escape_output=True), "parser_result_binding_failed"),
        (_FakeMineruService(symlink_output=True), "parser_result_binding_failed"),
    ],
)
def test_execution_fails_closed_on_provider_hash_or_output_binding_mismatch(
    tmp_path: Path,
    service: _FakeMineruService,
    expected_error_code: str,
) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")

    artifact, _, _ = _execute(tmp_path, [pdf_path], service)

    assert artifact.status == OledSupplementaryMineruExecutionStatus.FAILED
    assert artifact.failed_source_count == 1
    assert artifact.source_results[0].error_code == expected_error_code
    assert artifact.locator_resolved is False
    assert artifact.candidate_regenerated is False
    assert artifact.dataset_written is False


def test_execution_fails_closed_when_actual_backend_differs_from_bound_profile(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService(observed_backend="pipeline")

    artifact, _, _ = _execute(tmp_path, [pdf_path], service)

    assert artifact.backend == "hybrid-engine"
    assert artifact.status == OledSupplementaryMineruExecutionStatus.FAILED
    assert artifact.failed_source_count == 1
    assert artifact.source_results[0].parser_backend == "mineru_api:pipeline"
    assert artifact.source_results[0].error_code == "parser_result_binding_failed"


def test_execution_stops_after_first_source_failure(tmp_path: Path) -> None:
    first_pdf = _write_pdf(tmp_path / "paper016_si_a.pdf", text="first")
    second_pdf = _write_pdf(tmp_path / "paper016_si_b.pdf", text="second")
    service = _FakeMineruService(fail_call=1)

    artifact, _, _ = _execute(tmp_path, [first_pdf, second_pdf], service)

    assert artifact.status == OledSupplementaryMineruExecutionStatus.FAILED
    assert artifact.failed_source_count == 1
    assert artifact.skipped_source_count == 1
    assert len(service.requests) == 1
    assert artifact.source_results[1].status == OledSupplementaryMineruExecutionStatus.SKIPPED


def test_execution_rejects_existing_run_directory_without_mutating_inputs(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(tmp_path, [pdf_path])
    run_root = tmp_path / "runs" / "supp-mineru-run-001"
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_pdf = pdf_path.read_bytes()

    with pytest.raises(ValueError, match="run directory must be fresh"):
        execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=inputs[0],
            execution_manifest_json=inputs[1],
            endpoint_profile_config_json=inputs[2],
            endpoint_preflight_report_json=inputs[3],
            output_root=tmp_path / "runs",
            service=service,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert pdf_path.read_bytes() == original_pdf
    assert service.requests == []


def test_cli_reports_only_redacted_execution_summary(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    service = _FakeMineruService()
    inputs = _write_execution_inputs(tmp_path, [pdf_path])
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "--preflight-artifact",
            str(inputs[0]),
            "--execution-manifest",
            str(inputs[1]),
            "--endpoint-profile-config",
            str(inputs[2]),
            "--endpoint-preflight-report",
            str(inputs[3]),
            "--output-root",
            str(tmp_path / "runs"),
        ],
        stdout=stdout,
        stderr=stderr,
        service=service,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    assert str(pdf_path) not in stdout.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    summary = json.loads(stdout.getvalue())
    assert summary["status"] == "success"
    assert summary["artifact"] == "supplementary_mineru_execution.json"
    assert summary["locator_resolved"] is False
    assert summary["dataset_written"] is False


def test_execution_artifact_rejects_downstream_side_effect_tampering(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    artifact, _, _ = _execute(tmp_path, [pdf_path], _FakeMineruService())
    payload = artifact.model_dump(mode="json")
    payload["candidate_regenerated"] = True

    with pytest.raises(ValueError, match="downstream admission boundary"):
        OledSupplementaryMineruExecutionArtifact.model_validate(payload)


def test_execution_artifact_rejects_success_backend_inconsistent_with_top_level(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "paper016_si.pdf")
    artifact, _, _ = _execute(tmp_path, [pdf_path], _FakeMineruService())
    payload = artifact.model_dump(mode="json")
    payload["source_results"][0]["parser_backend"] = "mineru_api:pipeline"

    with pytest.raises(ValueError, match="source backend does not match artifact backend"):
        OledSupplementaryMineruExecutionArtifact.model_validate(payload)
