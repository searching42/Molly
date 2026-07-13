from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent.document_parse_provider import DocumentParseRequest


class MinerUApiError(RuntimeError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_error_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class MinerUApiTaskSubmission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    status_url: str = ""
    result_url: str = ""
    file_names: list[str] = Field(default_factory=list)
    queued_ahead: int | None = None
    source_pdf_sha256: str = ""


class MinerUApiTaskStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str = ""
    state: str
    queued_ahead: int | None = None
    message: str = ""
    mineru_version: str = ""
    protocol_version: str = ""
    backend: str = ""


@dataclass(frozen=True)
class MinerUDownloadedResult:
    output_dir: Path
    extracted_relative_paths: list[str]


@dataclass(frozen=True)
class MinerUApiParseOutcome:
    remote_task_id: str
    output_dir: Path
    extracted_relative_paths: list[str]
    task_status_history: list[str]
    queued_ahead_history: list[int]
    mineru_version: str
    protocol_version: str
    backend: str
    source_pdf_sha256: str


def _is_loopback_host(host: str) -> bool:
    clean = str(host or "").strip().lower()
    return clean in {"127.0.0.1", "localhost", "::1"}


def _normalize_base_url(base_url: str) -> str:
    clean = str(base_url or "").strip().rstrip("/")
    if not clean:
        raise ValueError("MinerU API base_url required")
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("MinerU API base_url must use http or https")
    if not parsed.netloc:
        raise ValueError("MinerU API base_url must include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("MinerU API base_url must not include userinfo, query, or fragment")
    return clean


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _read_source_pdf_for_upload(
    input_pdf: Path,
    *,
    expected_source_pdf_sha256: str = "",
) -> tuple[bytes, str]:
    """Read exactly the bytes that will be uploaded and bind them to an expected hash."""

    path = input_pdf.expanduser()
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if expected_source_pdf_sha256 and no_follow is None:
        raise MinerUApiError(
            code="source_binding_unavailable",
            message="content-bound MinerU upload requires O_NOFOLLOW support",
        )
    flags = os.O_RDONLY | (no_follow or 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise MinerUApiError(
                    code="source_read_failed",
                    message="MinerU upload source must be a regular file",
                )
            source_bytes = handle.read()
            final_stat = os.fstat(handle.fileno())
            if (
                final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
                or len(source_bytes) != initial_stat.st_size
            ):
                raise MinerUApiError(
                    code="source_changed_during_read",
                    message="MinerU upload source changed while its bytes were being read",
                )
    except MinerUApiError:
        raise
    except OSError as exc:
        raise MinerUApiError(
            code="source_read_failed",
            message="MinerU upload source could not be read safely",
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    source_pdf_sha256 = f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"
    if expected_source_pdf_sha256 and source_pdf_sha256 != expected_source_pdf_sha256:
        raise MinerUApiError(
            code="source_hash_mismatch",
            message="MinerU upload source does not match expected_source_pdf_sha256",
        )
    return source_bytes, source_pdf_sha256


def _safe_member_path(name: str) -> Path:
    member = Path(str(name or ""))
    if member.is_absolute():
        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains absolute paths")
    if any(part in {"..", ""} for part in member.parts):
        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains path traversal")
    return member


def _validate_archive_infos(
    *,
    infos: list[zipfile.ZipInfo],
    destination_root: Path,
    original_pdf: Path,
    max_member_count: int,
    max_member_bytes: int,
    max_total_uncompressed_bytes: int,
    max_compression_ratio: float,
) -> None:
    if max_member_count <= 0:
        raise ValueError("max_member_count must be positive")
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    if max_total_uncompressed_bytes <= 0:
        raise ValueError("max_total_uncompressed_bytes must be positive")
    if max_compression_ratio <= 0:
        raise ValueError("max_compression_ratio must be positive")
    if len(infos) > max_member_count:
        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains too many files")

    seen: set[str] = set()
    total_uncompressed = 0
    for info in infos:
        member = _safe_member_path(info.filename)
        relative = str(member)
        if relative in seen:
            raise MinerUApiError(code="unsafe_result_archive", message="result archive contains duplicate paths")
        seen.add(relative)
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type:
            if stat.S_ISLNK(mode):
                raise MinerUApiError(code="unsafe_result_archive", message="result archive contains symlinks")
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise MinerUApiError(code="unsafe_result_archive", message="result archive contains special files")
        target = (destination_root / member).resolve()
        if target == original_pdf.resolve():
            raise MinerUApiError(code="unsafe_result_archive", message="result archive attempts to overwrite the source PDF")
        if destination_root not in target.parents and target != destination_root:
            raise MinerUApiError(code="unsafe_result_archive", message="result archive escapes destination root")
        if info.is_dir():
            continue
        if info.file_size > max_member_bytes:
            raise MinerUApiError(code="unsafe_result_archive", message="result archive member exceeds configured size limit")
        total_uncompressed += int(info.file_size)
        if total_uncompressed > max_total_uncompressed_bytes:
            raise MinerUApiError(code="unsafe_result_archive", message="result archive exceeds configured uncompressed size limit")
        compressed_size = max(int(info.compress_size), 1)
        if info.file_size > 0 and (float(info.file_size) / float(compressed_size)) > max_compression_ratio:
            raise MinerUApiError(code="unsafe_result_archive", message="result archive compression ratio exceeds configured limit")


def safe_extract_result_archive(
    *,
    archive_bytes: bytes,
    destination_dir: Path,
    original_pdf: Path,
    max_member_count: int = 1000,
    max_member_bytes: int = 64 * 1024 * 1024,
    max_total_uncompressed_bytes: int = 512 * 1024 * 1024,
    max_compression_ratio: float = 100.0,
) -> list[str]:
    destination_root = destination_dir.expanduser().resolve()
    extracted_paths: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            infos = archive.infolist()
            _validate_archive_infos(
                infos=infos,
                destination_root=destination_root,
                original_pdf=original_pdf,
                max_member_count=max_member_count,
                max_member_bytes=max_member_bytes,
                max_total_uncompressed_bytes=max_total_uncompressed_bytes,
                max_compression_ratio=max_compression_ratio,
            )
            if destination_root.exists() and any(destination_root.iterdir()):
                raise MinerUApiError(code="unsafe_result_archive", message="result destination directory must be empty")
            destination_root.mkdir(parents=True, exist_ok=True)
            for info in infos:
                member = _safe_member_path(info.filename)
                target = (destination_root / member).resolve()
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(info, "r") as source, target.open("wb") as sink:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > info.file_size or written > max_member_bytes:
                            raise MinerUApiError(code="unsafe_result_archive", message="result archive member exceeds configured size limit")
                        sink.write(chunk)
                extracted_paths.append(str(member))
    except zipfile.BadZipFile as exc:
        raise MinerUApiError(code="invalid_output_bundle", message="result archive is not a valid ZIP bundle") from exc
    return extracted_paths


class MinerUApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str = "",
        token_header: str = "Authorization",
        allow_insecure_remote_http: bool = False,
        timeout_sec: float | None = None,
        http_timeout_sec: float | None = None,
        task_timeout_sec: float = 300.0,
        poll_interval_sec: float = 1.0,
        max_poll_attempts: int = 120,
        max_result_bytes: int = 100 * 1024 * 1024,
        max_result_member_count: int = 1000,
        max_result_member_bytes: int = 64 * 1024 * 1024,
        max_result_uncompressed_bytes: int = 512 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        parsed = urlparse(self.base_url)
        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._api_token = str(api_token or "").strip()
        self._token_header = str(token_header or "Authorization").strip() or "Authorization"
        self._allow_insecure_remote_http = bool(allow_insecure_remote_http)
        if http_timeout_sec is None:
            http_timeout_sec = timeout_sec if timeout_sec is not None else 30.0
        self._http_timeout_sec = _positive_float(http_timeout_sec, "http_timeout_sec")
        self._task_timeout_sec = _positive_float(task_timeout_sec, "task_timeout_sec")
        self._poll_interval_sec = _positive_float(poll_interval_sec, "poll_interval_sec")
        self._max_poll_attempts = _positive_int(max_poll_attempts, "max_poll_attempts")
        self._max_result_bytes = _positive_int(max_result_bytes, "max_result_bytes")
        self._max_result_member_count = _positive_int(max_result_member_count, "max_result_member_count")
        self._max_result_member_bytes = _positive_int(max_result_member_bytes, "max_result_member_bytes")
        self._max_result_uncompressed_bytes = _positive_int(max_result_uncompressed_bytes, "max_result_uncompressed_bytes")
        self._transport = transport
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep

    def configured(self) -> bool:
        return bool(self.base_url)

    def health(self) -> dict[str, Any]:
        response = self._request("GET", self._url("/health"), error_code="api_unavailable")
        if response.status_code != 200:
            raise MinerUApiError(
                code="health_check_failure",
                message=f"MinerU health check failed with HTTP {response.status_code}",
            )
        payload = self._json_object(response, code="health_check_failure")
        status = str(payload.get("status") or "").strip().lower()
        if status and status not in {"ok", "healthy", "ready"}:
            raise MinerUApiError(code="health_check_failure", message=f"MinerU health check reported status {status!r}")
        return payload

    def submit_pdf(self, *, request: DocumentParseRequest, input_pdf: Path) -> MinerUApiTaskSubmission:
        self.validate_upload_policy(request)
        source_bytes, source_pdf_sha256 = _read_source_pdf_for_upload(
            input_pdf,
            expected_source_pdf_sha256=request.expected_source_pdf_sha256,
        )
        files = [("files", (input_pdf.name, source_bytes, "application/pdf"))]
        response = self._request(
            "POST",
            self._url("/tasks"),
            error_code="api_unavailable",
            data=self._submission_form_data(request),
            files=files,
        )
        if response.status_code not in {200, 202}:
            raise MinerUApiError(
                code="submission_failure",
                message=f"MinerU task submission failed with HTTP {response.status_code}",
            )
        payload = self._json_object(response, code="invalid_submission_response")
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            raise MinerUApiError(code="invalid_submission_response", message="MinerU submission response missing task_id")
        return MinerUApiTaskSubmission.model_validate(payload).model_copy(
            update={"source_pdf_sha256": source_pdf_sha256}
        )

    def get_task_status(self, task_id: str) -> MinerUApiTaskStatus:
        response = self._request("GET", self._url(f"/tasks/{task_id}"), error_code="api_unavailable")
        if response.status_code != 200:
            raise MinerUApiError(
                code="task_failed",
                message=f"MinerU task status request failed with HTTP {response.status_code}",
                details={"task_id": task_id},
            )
        payload = self._json_object(response, code="task_failed")
        state = str(payload.get("state") or payload.get("status") or "").strip().lower()
        if not state:
            raise MinerUApiError(code="task_failed", message="MinerU task status response missing state", details={"task_id": task_id})
        return MinerUApiTaskStatus(
            task_id=str(payload.get("task_id") or task_id),
            state=state,
            queued_ahead=self._as_int_or_none(payload.get("queued_ahead")),
            message=str(payload.get("message") or payload.get("detail") or "").strip(),
            mineru_version=str(payload.get("version_name") or payload.get("_version_name") or "").strip(),
            protocol_version=str(payload.get("protocol_version") or "").strip(),
            backend=str(payload.get("backend") or payload.get("_backend") or "").strip(),
        )

    def wait_for_task(self, task_id: str) -> tuple[MinerUApiTaskStatus, list[str], list[int]]:
        history: list[str] = []
        queued_history: list[int] = []
        deadline = self._monotonic() + self._task_timeout_sec
        for _ in range(self._max_poll_attempts):
            if self._monotonic() >= deadline:
                raise self._task_error(
                    code="task_timeout",
                    message="MinerU task polling exceeded the configured deadline",
                    task_id=task_id,
                    history=history,
                    queued_history=queued_history,
                )
            try:
                status = self.get_task_status(task_id)
            except MinerUApiError as exc:
                raise self._enrich_task_error(exc, task_id=task_id, history=history, queued_history=queued_history) from exc
            history.append(status.state)
            if status.queued_ahead is not None:
                queued_history.append(int(status.queued_ahead))
            if status.state == "completed":
                return status, history, queued_history
            if status.state in {"pending", "processing", "queued", "running"}:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise self._task_error(
                        code="task_timeout",
                        message="MinerU task polling exceeded the configured deadline",
                        task_id=task_id,
                        history=history,
                        queued_history=queued_history,
                    )
                self._sleep(min(self._poll_interval_sec, remaining))
                continue
            if status.state == "failed":
                raise self._task_error(
                    code="task_failed",
                    message=status.message or "MinerU task failed",
                    task_id=task_id,
                    history=history,
                    queued_history=queued_history,
                )
            if status.state == "cancelled":
                raise self._task_error(
                    code="task_cancelled",
                    message=status.message or "MinerU task cancelled",
                    task_id=task_id,
                    history=history,
                    queued_history=queued_history,
                )
            raise self._task_error(
                code="task_failed",
                message=f"MinerU task entered unknown state {status.state!r}",
                task_id=task_id,
                history=history,
                queued_history=queued_history,
            )
        raise self._task_error(
            code="task_timeout",
            message="MinerU task polling exceeded the configured limit",
            task_id=task_id,
            history=history,
            queued_history=queued_history,
        )

    def download_result(self, *, task_id: str, output_dir: Path, original_pdf: Path) -> MinerUDownloadedResult:
        response = self._stream_request_bytes(
            "GET",
            self._url(f"/tasks/{task_id}/result"),
            error_code="api_unavailable",
            task_id=task_id,
        )
        if response.status_code != 200:
            raise MinerUApiError(
                code="result_download_failure",
                message=f"MinerU result download failed with HTTP {response.status_code}",
                details=self._response_error_details(response=response, task_id=task_id),
            )
        extracted = safe_extract_result_archive(
            archive_bytes=response.content,
            destination_dir=output_dir,
            original_pdf=original_pdf,
            max_member_count=self._max_result_member_count,
            max_member_bytes=self._max_result_member_bytes,
            max_total_uncompressed_bytes=self._max_result_uncompressed_bytes,
        )
        return MinerUDownloadedResult(output_dir=output_dir, extracted_relative_paths=extracted)

    def parse_pdf(self, *, request: DocumentParseRequest, input_pdf: Path, output_dir: Path) -> MinerUApiParseOutcome:
        self.validate_upload_policy(request)
        submission = self.submit_pdf(request=request, input_pdf=input_pdf)
        history: list[str] = []
        queued_history: list[int] = []
        try:
            status, history, queued_history = self.wait_for_task(submission.task_id)
            downloaded = self.download_result(
                task_id=submission.task_id,
                output_dir=output_dir,
                original_pdf=input_pdf,
            )
        except MinerUApiError as exc:
            details = dict(exc.details)
            details.setdefault("task_id", submission.task_id)
            details.setdefault("source_pdf_sha256", submission.source_pdf_sha256)
            details.setdefault("task_status_history", history or list(details.get("task_status_history") or []))
            details.setdefault("queued_ahead_history", queued_history or list(details.get("queued_ahead_history") or []))
            raise MinerUApiError(code=exc.code, message=exc.message, details=details) from exc
        return MinerUApiParseOutcome(
            remote_task_id=submission.task_id,
            output_dir=downloaded.output_dir,
            extracted_relative_paths=downloaded.extracted_relative_paths,
            task_status_history=history,
            queued_ahead_history=queued_history,
            mineru_version=status.mineru_version,
            protocol_version=status.protocol_version,
            backend=status.backend or request.backend,
            source_pdf_sha256=submission.source_pdf_sha256,
        )

    def _submission_form_data(self, request: DocumentParseRequest) -> dict[str, Any]:
        data: dict[str, Any] = {
            "backend": request.backend,
            "effort": request.effort,
            "parse_method": request.parse_method,
            "formula_enable": self._bool_text(request.formula_enabled),
            "table_enable": self._bool_text(request.table_enabled),
            "image_analysis": self._bool_text(request.image_analysis_enabled),
            "return_md": "true",
            "return_middle_json": "true",
            "return_content_list": "true",
            "return_images": "false",
            "response_format_zip": "true",
            "return_original_file": "false",
        }
        if request.start_page is not None:
            data["start_page_id"] = str(request.start_page - 1)
        if request.end_page is not None:
            data["end_page_id"] = str(request.end_page - 1)
        return data

    def validate_upload_policy(self, request: DocumentParseRequest) -> None:
        if self._scheme == "http" and not _is_loopback_host(self._host) and not self._allow_insecure_remote_http:
            raise MinerUApiError(
                code="submission_failure",
                message="non-loopback MinerU HTTP endpoints require an explicit insecure development override",
            )
        if not _is_loopback_host(self._host) and not request.allow_remote_upload:
            raise MinerUApiError(
                code="submission_failure",
                message="allow_remote_upload must be true for non-loopback MinerU endpoints",
            )

    def _json_object(self, response: httpx.Response, *, code: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MinerUApiError(code=code, message="MinerU response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MinerUApiError(code=code, message="MinerU response JSON root must be an object")
        return payload

    def _request(self, method: str, url: str, *, error_code: str, **kwargs: Any) -> httpx.Response:
        client = self._client()
        try:
            return client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise MinerUApiError(code=error_code, message=self._redacted_network_message(exc)) from exc
        finally:
            if self._transport is None:
                client.close()

    def _stream_request_bytes(self, method: str, url: str, *, error_code: str, task_id: str) -> httpx.Response:
        client = self._client()
        try:
            with client.stream(method, url, headers={"Accept-Encoding": "identity"}) as response:
                payload = bytearray()
                for chunk in response.iter_raw():
                    payload.extend(chunk)
                    if len(payload) > self._max_result_bytes:
                        raise MinerUApiError(
                            code="oversized_result",
                            message="MinerU result archive exceeds the configured size limit",
                            details={"task_id": task_id},
                        )
                headers = dict(response.headers)
                headers.pop("content-encoding", None)
                return httpx.Response(
                    response.status_code,
                    content=bytes(payload),
                    headers=headers,
                    request=response.request,
                )
        except MinerUApiError:
            raise
        except httpx.RequestError as exc:
            raise MinerUApiError(code=error_code, message=self._redacted_network_message(exc), details={"task_id": task_id}) from exc
        finally:
            if self._transport is None:
                client.close()

    @staticmethod
    def _response_error_details(*, response: httpx.Response, task_id: str) -> dict[str, Any]:
        details: dict[str, Any] = {
            "task_id": task_id,
            "status_code": response.status_code,
        }
        body = response.content
        if not body:
            return details
        content_type = str(response.headers.get("content-type") or "").lower()
        if "json" in content_type:
            try:
                details["response_json"] = json.loads(body.decode("utf-8"))
                return details
            except (UnicodeDecodeError, ValueError):
                pass
        preview_bytes = body[:4096]
        details["response_body_preview"] = preview_bytes.decode("utf-8", errors="replace")
        details["response_body_truncated"] = len(body) > len(preview_bytes)
        return details

    def _client(self) -> httpx.Client:
        headers = {}
        if self._api_token:
            headers[self._token_header] = (
                f"Bearer {self._api_token}"
                if self._token_header.lower() == "authorization" and not self._api_token.lower().startswith("bearer ")
                else self._api_token
            )
        return httpx.Client(
            headers=headers,
            timeout=self._http_timeout_sec,
            transport=self._transport,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _task_error(
        self,
        *,
        code: str,
        message: str,
        task_id: str,
        history: list[str],
        queued_history: list[int],
    ) -> MinerUApiError:
        return MinerUApiError(
            code=code,
            message=message,
            details={
                "task_id": task_id,
                "task_status_history": list(history),
                "queued_ahead_history": list(queued_history),
            },
        )

    @staticmethod
    def _enrich_task_error(
        exc: MinerUApiError,
        *,
        task_id: str,
        history: list[str],
        queued_history: list[int],
    ) -> MinerUApiError:
        details = dict(exc.details)
        details.setdefault("task_id", task_id)
        details.setdefault("task_status_history", list(history))
        details.setdefault("queued_ahead_history", list(queued_history))
        return MinerUApiError(code=exc.code, message=exc.message, details=details)

    def _redacted_network_message(self, exc: httpx.RequestError) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        if self._api_token:
            message = message.replace(self._api_token, "[redacted]")
        return message

    @staticmethod
    def _bool_text(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _as_int_or_none(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
