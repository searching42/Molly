from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

import httpx

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.document_parse_mineru import MinerUApiDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseRequest, DocumentParseResult
from ai4s_agent.document_parse_service import DocumentParseService
from ai4s_agent.domains.oled_supplementary_mineru_execution import (
    OledSupplementaryMineruExecutionArtifact,
    OledSupplementaryMineruExecutionManifest,
    OledSupplementaryMineruExecutionStatus,
    OledSupplementaryMineruExecutionTarget,
    OledSupplementaryMineruOutputHash,
    OledSupplementaryMineruOutputKind,
    OledSupplementaryMineruSourceExecutionResult,
    build_oled_supplementary_mineru_execution_artifact,
    validate_oled_supplementary_mineru_execution_binding,
)
from ai4s_agent.domains.oled_supplementary_source_intake import DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES
from ai4s_agent.mineru_api_client import MinerUApiClient
from ai4s_agent.mineru_endpoint_preflight import MinerUEndpointPreflightReport
from ai4s_agent.mineru_endpoint_profiles import (
    MinerUEndpointProfileReportSummary,
    ResolvedMinerUEndpointProfile,
    load_mineru_endpoint_profile_config,
    resolve_mineru_endpoint_profile,
)
from ai4s_agent.oled_supplementary_parser_preflight import OledSupplementaryParserPreflightArtifact


_ARTIFACT_BASENAME = "supplementary_mineru_execution.json"
_SNAPSHOT_BASENAME = "approved_source.pdf"
_PDF_TRAILER_SCAN_BYTES = 8192
_SAFE_ERROR_CODE_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_OUTPUT_KIND_TO_FIELD = {
    OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_JSON: "parsed_document_json",
    OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_MARKDOWN: "parsed_document_markdown",
    OledSupplementaryMineruOutputKind.PARSER_AUDIT_JSON: "parser_audit_json",
    OledSupplementaryMineruOutputKind.CONTENT_LIST_JSON: "content_list_json",
    OledSupplementaryMineruOutputKind.CONTENT_LIST_V2_JSON: "content_list_v2_json",
    OledSupplementaryMineruOutputKind.MIDDLE_JSON: "middle_json",
}
_REQUIRED_OUTPUT_KINDS = {
    OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_JSON,
    OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_MARKDOWN,
    OledSupplementaryMineruOutputKind.PARSER_AUDIT_JSON,
}


def execute_oled_supplementary_mineru_from_files(
    *,
    preflight_artifact_json: str | Path,
    execution_manifest_json: str | Path,
    endpoint_profile_config_json: str | Path,
    endpoint_preflight_report_json: str | Path,
    output_root: str | Path,
    service: Any | None = None,
    transport: httpx.BaseTransport | None = None,
    generated_at: str | None = None,
) -> OledSupplementaryMineruExecutionArtifact:
    """Run only the sources approved by a bound parser preflight and endpoint gate."""

    preflight_path = _resolve_local_path(preflight_artifact_json)
    manifest_path = _resolve_local_path(execution_manifest_json)
    profile_path = _resolve_local_path(endpoint_profile_config_json)
    endpoint_report_path = _resolve_local_path(endpoint_preflight_report_json)
    root = _resolve_local_path(output_root)

    preflight_payload, _ = _read_bound_json(preflight_path, "supplementary parser preflight artifact")
    manifest_payload, _ = _read_bound_json(manifest_path, "supplementary MinerU execution manifest")
    endpoint_payload, endpoint_report_sha256 = _read_bound_json(
        endpoint_report_path,
        "MinerU endpoint preflight report",
    )
    preflight_artifact = OledSupplementaryParserPreflightArtifact.model_validate(preflight_payload)
    execution_manifest = OledSupplementaryMineruExecutionManifest.model_validate(manifest_payload)
    endpoint_report = MinerUEndpointPreflightReport.model_validate(endpoint_payload)
    preflight_artifact.validate_artifact_binding()
    validate_oled_supplementary_mineru_execution_binding(
        preflight_artifact.preflight_plan,
        execution_manifest,
    )
    if endpoint_report_sha256 != execution_manifest.endpoint_preflight_sha256:
        raise ValueError("MinerU endpoint preflight report hash does not match execution manifest")

    resolved_profile = _load_and_validate_endpoint_binding(
        profile_path=profile_path,
        report=endpoint_report,
        manifest=execution_manifest,
    )
    protected_paths = {
        preflight_path,
        manifest_path,
        profile_path,
        endpoint_report_path,
        *(_resolve_local_path(source.local_pdf_path) for source in execution_manifest.sources),
    }
    run_root = _create_fresh_run_root(root, execution_manifest.run_id, protected_paths=protected_paths)

    plan = preflight_artifact.preflight_plan
    source_envelopes = {source.source_id: source for source in plan.source_envelopes}
    manifest_sources = {source.source_id: source for source in execution_manifest.sources}
    targets_by_source: dict[str, list[OledSupplementaryMineruExecutionTarget]] = {
        source_id: [] for source_id in source_envelopes
    }
    for item in plan.items:
        targets_by_source[item.source_id].append(
            OledSupplementaryMineruExecutionTarget(
                recovery_item_id=item.recovery_item_id,
                target_kind=item.target_kind,
                target_locator=item.target_locator,
            )
        )
    for targets in targets_by_source.values():
        targets.sort(key=lambda target: target.recovery_item_id)

    snapshots: dict[str, Path] = {}
    for source_id in sorted(source_envelopes):
        envelope = source_envelopes[source_id]
        source_root = run_root / "sources" / source_id
        source_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        snapshot = source_root / _SNAPSHOT_BASENAME
        _snapshot_approved_pdf(
            _expand_local_path(manifest_sources[source_id].local_pdf_path),
            snapshot,
            expected_sha256=envelope.pdf_sha256,
            expected_byte_size=envelope.byte_size,
        )
        snapshots[source_id] = snapshot

    parse_service = service or _build_service(resolved_profile, transport=transport)
    live_mineru_version, live_protocol_version = _validate_live_endpoint(
        parse_service,
        expected_protocol_version=resolved_profile.profile.expected_protocol_version,
        expected_health_path=resolved_profile.profile.health_path,
    )

    source_results: list[OledSupplementaryMineruSourceExecutionResult] = []
    prior_failure = False
    for source_id in sorted(source_envelopes):
        envelope = source_envelopes[source_id]
        targets = targets_by_source[source_id]
        if prior_failure:
            source_results.append(
                OledSupplementaryMineruSourceExecutionResult(
                    source_id=source_id,
                    source_pdf_sha256=envelope.pdf_sha256,
                    byte_size=envelope.byte_size,
                    page_count=envelope.page_count,
                    targets=targets,
                    status=OledSupplementaryMineruExecutionStatus.SKIPPED,
                    mineru_called=False,
                    error_code="skipped_after_prior_failure",
                    error_message="Skipped because an earlier approved source failed MinerU execution.",
                )
            )
            continue
        source_output_dir = run_root / "sources" / source_id / "mineru"
        request = DocumentParseRequest(
            run_id=f"{execution_manifest.run_id}-{source_id}",
            input_pdf=str(snapshots[source_id]),
            output_dir=str(source_output_dir),
            provider="mineru_api",
            parse_method=resolved_profile.profile.parse_method,
            backend=resolved_profile.profile.backend,
            effort=resolved_profile.profile.effort,
            formula_enabled=True,
            table_enabled=True,
            image_analysis_enabled=False,
            start_page=None,
            end_page=None,
            allow_remote_upload=resolved_profile.profile.allow_remote_upload,
            expected_source_pdf_sha256=envelope.pdf_sha256,
        )
        try:
            parse_result = parse_service.parse(request)
            source_result = _source_result_from_parse(
                result=parse_result,
                request=request,
                source_id=source_id,
                source_pdf_sha256=envelope.pdf_sha256,
                byte_size=envelope.byte_size,
                page_count=envelope.page_count,
                targets=targets,
                source_output_dir=source_output_dir,
                live_mineru_version=live_mineru_version,
                expected_protocol_version=live_protocol_version,
            )
        except Exception:
            source_result = OledSupplementaryMineruSourceExecutionResult(
                source_id=source_id,
                source_pdf_sha256=envelope.pdf_sha256,
                byte_size=envelope.byte_size,
                page_count=envelope.page_count,
                targets=targets,
                status=OledSupplementaryMineruExecutionStatus.FAILED,
                mineru_called=True,
                provider="mineru_api",
                parser_backend=f"mineru_api:{resolved_profile.profile.backend}",
                mineru_version=live_mineru_version or "not_reported",
                protocol_version=live_protocol_version,
                error_code="mineru_service_exception",
                error_message="MinerU execution raised an exception; inspect local parser evidence.",
            )
        source_results.append(source_result)
        prior_failure = source_result.status != OledSupplementaryMineruExecutionStatus.SUCCESS

    profile_summary = MinerUEndpointProfileReportSummary.model_validate(
        resolved_profile.redacted_summary(base_dir=profile_path.parent)
    )
    artifact = build_oled_supplementary_mineru_execution_artifact(
        manifest=execution_manifest,
        generated_at=generated_at or now_iso(),
        redacted_api_origin=profile_summary.redacted_api_origin,
        backend=resolved_profile.profile.backend,
        effort=resolved_profile.profile.effort,
        parse_method=resolved_profile.profile.parse_method,
        source_results=source_results,
    )
    write_json(run_root / _ARTIFACT_BASENAME, artifact.model_dump(mode="json"))
    return artifact


def _load_and_validate_endpoint_binding(
    *,
    profile_path: Path,
    report: MinerUEndpointPreflightReport,
    manifest: OledSupplementaryMineruExecutionManifest,
) -> ResolvedMinerUEndpointProfile:
    config = load_mineru_endpoint_profile_config(profile_path)
    if config.schema_version != "mineru_endpoint_profiles.v1":
        raise ValueError("unexpected MinerU endpoint profile schema_version")
    resolved = resolve_mineru_endpoint_profile(
        config,
        profile_name=manifest.endpoint_profile_name,
        policy_name=None,
        profile_source_path=profile_path,
    )
    expected = MinerUEndpointProfileReportSummary.model_validate(
        resolved.redacted_summary(base_dir=profile_path.parent)
    )
    observed = report.profile
    if report.schema_version != "mineru_endpoint_preflight.v1":
        raise ValueError("unexpected MinerU endpoint preflight report schema_version")
    if report.decision != "passed" or not report.health.ok:
        raise ValueError("MinerU endpoint preflight report must have a passed decision")
    if report.errors or report.health.http_status_code != 200 or not report.health.response_schema_valid:
        raise ValueError("MinerU endpoint preflight report contains failed health evidence")
    if str(report.health.status or "").strip().lower() not in {"healthy", "ok"}:
        raise ValueError("MinerU endpoint preflight report health status is not acceptable")
    if report.health.protocol_version != resolved.profile.expected_protocol_version:
        raise ValueError("MinerU endpoint preflight protocol does not match endpoint profile")
    if report.health.health_path != resolved.profile.health_path:
        raise ValueError("MinerU endpoint preflight health evidence path does not match endpoint profile")
    if resolved.profile.api_url.rstrip("/") != expected.redacted_api_origin:
        raise ValueError("supplementary MinerU execution requires an origin-only endpoint api_url")
    compared_fields = (
        "endpoint_profile_name",
        "redacted_api_origin",
        "endpoint_kind",
        "backend",
        "effort",
        "parse_method",
        "allow_remote_upload",
        "http_timeout_sec",
        "task_timeout_sec",
        "poll_interval_sec",
        "max_poll_attempts",
        "expected_protocol_version",
        "health_path",
    )
    mismatches = [
        field_name
        for field_name in compared_fields
        if getattr(observed, field_name) != getattr(expected, field_name)
    ]
    if mismatches:
        raise ValueError("MinerU endpoint profile does not match the bound endpoint preflight report")
    return resolved


def _build_service(
    resolved: ResolvedMinerUEndpointProfile,
    *,
    transport: httpx.BaseTransport | None,
) -> DocumentParseService:
    profile = resolved.profile
    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
    client = MinerUApiClient(
        base_url=profile.api_url,
        health_path=profile.health_path,
        api_token=token,
        http_timeout_sec=profile.http_timeout_sec,
        task_timeout_sec=profile.task_timeout_sec,
        poll_interval_sec=profile.poll_interval_sec,
        max_poll_attempts=profile.max_poll_attempts,
        transport=transport,
    )
    return DocumentParseService(
        mineru_provider=MinerUApiDocumentParseProvider(client=client, check_health=False)
    )


def _validate_live_endpoint(
    service: Any,
    *,
    expected_protocol_version: str,
    expected_health_path: str,
) -> tuple[str, str]:
    provider = getattr(service, "mineru_provider", None)
    client = getattr(provider, "client", None)
    if client is None or not callable(getattr(client, "health", None)):
        raise ValueError("supplementary MinerU execution requires a health-checkable MinerU provider")
    if str(getattr(client, "health_path", "") or "").strip() != expected_health_path:
        raise ValueError("live MinerU health path does not match the bound endpoint profile")
    payload = client.health()
    if not isinstance(payload, dict):
        raise ValueError("live MinerU health response must be a JSON object")
    status = str(payload.get("status") or "").strip().lower()
    protocol = str(payload.get("protocol_version") or "").strip()
    if status not in {"healthy", "ok"}:
        raise ValueError("live MinerU endpoint is not healthy")
    if protocol != expected_protocol_version:
        raise ValueError("live MinerU protocol does not match the bound endpoint profile")
    version = str(payload.get("version") or payload.get("version_name") or payload.get("_version_name") or "").strip()
    return version or "not_reported", protocol


def _source_result_from_parse(
    *,
    result: DocumentParseResult,
    request: DocumentParseRequest,
    source_id: str,
    source_pdf_sha256: str,
    byte_size: int,
    page_count: int,
    targets: list[OledSupplementaryMineruExecutionTarget],
    source_output_dir: Path,
    live_mineru_version: str,
    expected_protocol_version: str,
) -> OledSupplementaryMineruSourceExecutionResult:
    mineru_version = str(result.audit.mineru_version or live_mineru_version or "not_reported").strip()
    warning_codes = ["parser_warning_present"] if result.warnings else []
    if mineru_version == "not_reported":
        warning_codes.append("mineru_version_not_reported")
    warning_codes.sort()
    base = {
        "source_id": source_id,
        "source_pdf_sha256": source_pdf_sha256,
        "byte_size": byte_size,
        "page_count": page_count,
        "targets": targets,
        "mineru_called": True,
        "provider": str(result.provider or "").strip(),
        "parser_backend": str(result.parser_backend or "mineru_api:unknown").strip(),
        "mineru_version": mineru_version,
        "protocol_version": str(result.audit.protocol_version or expected_protocol_version).strip(),
        "warning_codes": warning_codes,
    }
    if not result.ok:
        return OledSupplementaryMineruSourceExecutionResult(
            **base,
            status=OledSupplementaryMineruExecutionStatus.FAILED,
            output_hashes=_collect_output_hashes_if_safe(result, source_output_dir),
            error_code=_safe_error_code(result.error.code if result.error is not None else "mineru_parse_failed"),
            error_message="MinerU returned a failed parse result; inspect local parser evidence.",
        )
    try:
        _validate_successful_parse_result(
            result=result,
            request=request,
            source_pdf_sha256=source_pdf_sha256,
            source_output_dir=source_output_dir,
            expected_protocol_version=expected_protocol_version,
        )
        output_hashes = _collect_output_hashes(result, source_output_dir)
    except ValueError:
        return OledSupplementaryMineruSourceExecutionResult(
            **base,
            status=OledSupplementaryMineruExecutionStatus.FAILED,
            error_code="parser_result_binding_failed",
            error_message="MinerU output did not match the bound execution request.",
        )
    return OledSupplementaryMineruSourceExecutionResult(
        **base,
        status=OledSupplementaryMineruExecutionStatus.SUCCESS,
        output_hashes=output_hashes,
    )


def _validate_successful_parse_result(
    *,
    result: DocumentParseResult,
    request: DocumentParseRequest,
    source_pdf_sha256: str,
    source_output_dir: Path,
    expected_protocol_version: str,
) -> None:
    if result.provider != "mineru_api" or result.audit.selected_provider != "mineru_api":
        raise ValueError("supplementary execution forbids parser provider fallback")
    if result.audit.request_provider != "mineru_api" or request.provider != "mineru_api":
        raise ValueError("supplementary execution requires explicit MinerU provider selection")
    expected_parser_backend = f"mineru_api:{request.backend}"
    if result.parser_backend != expected_parser_backend:
        raise ValueError("MinerU result backend does not match the bound endpoint profile")
    if result.audit.parser_backend != expected_parser_backend:
        raise ValueError("MinerU audit backend does not match the bound endpoint profile")
    if result.audit.source_pdf_sha256 != source_pdf_sha256:
        raise ValueError("MinerU audit hash does not match the approved source hash")
    if str(result.audit.protocol_version or expected_protocol_version).strip() != expected_protocol_version:
        raise ValueError("MinerU result protocol does not match the bound endpoint profile")
    if result.parsed_document is None:
        raise ValueError("MinerU success result is missing ParsedDocument")
    if result.parsed_document.parser_backend != expected_parser_backend:
        raise ValueError("MinerU parsed document backend does not match the bound endpoint profile")
    if _resolve_local_path(result.input_pdf) != _resolve_local_path(request.input_pdf):
        raise ValueError("MinerU result input does not match the approved source snapshot")
    if _resolve_local_path(result.outputs.output_dir) != source_output_dir.resolve():
        raise ValueError("MinerU result output directory does not match the isolated run directory")
    current_sha256, _ = _hash_regular_file_bound(_resolve_local_path(request.input_pdf))
    if current_sha256 != source_pdf_sha256:
        raise ValueError("approved source snapshot changed during MinerU execution")


def _collect_output_hashes_if_safe(
    result: DocumentParseResult,
    source_output_dir: Path,
) -> list[OledSupplementaryMineruOutputHash]:
    try:
        return _collect_output_hashes(result, source_output_dir, require_success_outputs=False)
    except ValueError:
        return []


def _collect_output_hashes(
    result: DocumentParseResult,
    source_output_dir: Path,
    *,
    require_success_outputs: bool = True,
) -> list[OledSupplementaryMineruOutputHash]:
    root = source_output_dir.resolve()
    if _resolve_local_path(result.outputs.output_dir) != root:
        raise ValueError("MinerU output root is outside the isolated execution directory")
    hashes: list[OledSupplementaryMineruOutputHash] = []
    for output_kind, field_name in _OUTPUT_KIND_TO_FIELD.items():
        value = str(getattr(result.outputs, field_name) or "").strip()
        if not value:
            continue
        unresolved_path = _expand_local_path(value)
        if unresolved_path.is_symlink():
            raise ValueError("MinerU output reference must not be a symlink")
        path = _resolve_local_path(unresolved_path)
        if root not in path.parents:
            raise ValueError("MinerU output reference escapes the isolated execution directory")
        sha256, byte_size = _hash_regular_file_bound(path)
        hashes.append(
            OledSupplementaryMineruOutputHash(
                output_kind=output_kind,
                sha256=sha256,
                byte_size=byte_size,
            )
        )
    hashes.sort(key=lambda item: item.output_kind.value)
    if require_success_outputs and not _REQUIRED_OUTPUT_KINDS.issubset({item.output_kind for item in hashes}):
        raise ValueError("MinerU result is missing required output files")
    return hashes


def _snapshot_approved_pdf(
    source_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_byte_size: int,
) -> None:
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("approved supplementary source must use a .pdf filename")
    if expected_byte_size > DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES:
        raise ValueError("approved supplementary source exceeds the execution size limit")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary MinerU execution requires O_NOFOLLOW support")
    source_descriptor = -1
    destination_descriptor = -1
    temp_path = destination.with_name(f".{destination.name}.tmp")
    try:
        source_descriptor = os.open(source_path, os.O_RDONLY | no_follow)
        destination_descriptor = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o400,
        )
        with (
            os.fdopen(source_descriptor, "rb", closefd=True) as source,
            os.fdopen(destination_descriptor, "wb", closefd=True) as sink,
        ):
            source_descriptor = -1
            destination_descriptor = -1
            initial_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError("approved supplementary source must be a regular file")
            if int(initial_stat.st_size) != int(expected_byte_size):
                raise ValueError("approved supplementary source byte size no longer matches preflight")
            digest = hashlib.sha256()
            first_bytes = b""
            trailer = b""
            bytes_read = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                if not first_bytes:
                    first_bytes = chunk[:8]
                trailer = (trailer + chunk)[-_PDF_TRAILER_SCAN_BYTES:]
                digest.update(chunk)
                sink.write(chunk)
                bytes_read += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
            final_stat = os.fstat(source.fileno())
            if (
                final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
                or bytes_read != initial_stat.st_size
            ):
                raise ValueError("approved supplementary source changed while creating execution snapshot")
        if not first_bytes.startswith(b"%PDF-") or b"%%EOF" not in trailer:
            raise ValueError("approved supplementary source no longer has a valid PDF envelope")
        observed_sha256 = f"sha256:{digest.hexdigest()}"
        if observed_sha256 != expected_sha256:
            raise ValueError("approved supplementary source hash no longer matches parser preflight")
        os.replace(temp_path, destination)
        destination.chmod(0o400)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("approved supplementary source could not be snapshotted safely") from exc
    finally:
        if source_descriptor != -1:
            os.close(source_descriptor)
        if destination_descriptor != -1:
            os.close(destination_descriptor)
        if temp_path.exists():
            temp_path.unlink()


def _create_fresh_run_root(root: Path, run_id: str, *, protected_paths: set[Path]) -> Path:
    if root.exists() and not root.is_dir():
        raise ValueError("supplementary MinerU output root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    run_root = (root / run_id).resolve()
    if run_root.parent != root.resolve():
        raise ValueError("supplementary MinerU run_id must stay directly under output root")
    if run_root in protected_paths:
        raise ValueError("supplementary MinerU run directory must not overwrite an input")
    if run_root.exists() or run_root.is_symlink():
        raise ValueError("supplementary MinerU run directory must be fresh")
    run_root.mkdir(mode=0o700, exist_ok=False)
    return run_root


def _read_bound_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload_bytes, sha256 = _read_regular_file_bound(path)
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload, sha256


def _read_regular_file_bound(path: Path) -> tuple[bytes, str]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary MinerU execution requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError("supplementary MinerU input must be a regular file")
            payload = handle.read()
            final_stat = os.fstat(handle.fileno())
            if (
                final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
                or len(payload) != initial_stat.st_size
            ):
                raise ValueError("supplementary MinerU input changed while being read")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("supplementary MinerU input is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return payload, f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _hash_regular_file_bound(path: Path) -> tuple[str, int]:
    payload, sha256 = _read_regular_file_bound(path)
    if not payload:
        raise ValueError("supplementary MinerU output file is empty")
    return sha256, len(payload)


def _resolve_local_path(path_like: str | Path) -> Path:
    try:
        return Path(path_like).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("supplementary MinerU execution path could not be resolved safely") from exc


def _expand_local_path(path_like: str | Path) -> Path:
    try:
        return Path(path_like).expanduser()
    except (OSError, RuntimeError) as exc:
        raise ValueError("supplementary MinerU execution path could not be expanded safely") from exc


def _safe_error_code(value: str) -> str:
    clean = str(value or "").strip()
    credential_markers = ("token", "secret", "authorization", "password", "bearer")
    if not _SAFE_ERROR_CODE_RE.fullmatch(clean) or any(marker in clean.lower() for marker in credential_markers):
        return "mineru_parse_failed"
    return clean


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    if not clean:
        return "supplementary MinerU execution failed"
    lowered = clean.lower()
    if "/" in clean or "\\" in clean or "input_value=" in clean or any(
        marker in lowered for marker in ("token", "secret", "authorization", "password", "bearer")
    ):
        return "supplementary MinerU execution failed; inspect trusted local evidence"
    return clean


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    service: Any | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute explicitly approved supplementary PDFs through MinerU while preserving "
            "preflight, source-hash, endpoint, and output-hash bindings."
        )
    )
    parser.add_argument("--preflight-artifact", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--endpoint-profile-config", required=True)
    parser.add_argument("--endpoint-preflight-report", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = execute_oled_supplementary_mineru_from_files(
            preflight_artifact_json=args.preflight_artifact,
            execution_manifest_json=args.execution_manifest,
            endpoint_profile_config_json=args.endpoint_profile_config,
            endpoint_preflight_report_json=args.endpoint_preflight_report,
            output_root=args.output_root,
            service=service,
            transport=transport,
        )
    except Exception as exc:
        err.write(f"{_safe_error_message(str(exc))}\n")
        return 1
    output.write(
        json.dumps(
            {
                "status": artifact.status.value,
                "run_id": artifact.run_id,
                "paper_id": artifact.paper_id,
                "endpoint_profile_name": artifact.endpoint_profile_name,
                "source_count": artifact.source_count,
                "successful_source_count": artifact.successful_source_count,
                "failed_source_count": artifact.failed_source_count,
                "skipped_source_count": artifact.skipped_source_count,
                "artifact": _ARTIFACT_BASENAME,
                "locator_resolved": False,
                "candidate_regenerated": False,
                "dataset_written": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if artifact.status == OledSupplementaryMineruExecutionStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "execute_oled_supplementary_mineru_from_files",
    "main",
]
