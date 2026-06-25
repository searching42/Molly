from __future__ import annotations

import io
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
    return clean


def _safe_member_path(name: str) -> Path:
    member = Path(str(name or ""))
    if member.is_absolute():
        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains absolute paths")
    if any(part in {"..", ""} for part in member.parts):
        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains path traversal")
    return member


def safe_extract_result_archive(
    *,
    archive_bytes: bytes,
    destination_dir: Path,
    original_pdf: Path,
) -> list[str]:
    destination_root = destination_dir.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[str] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for info in archive.infolist():
                member = _safe_member_path(info.filename)
                relative = str(member)
                if relative in seen:
                    raise MinerUApiError(code="unsafe_result_archive", message="result archive contains duplicate paths")
                seen.add(relative)
                mode = (info.external_attr >> 16) & 0xFFFF
                is_dir = info.is_dir()
                if mode:
                    if stat.S_ISLNK(mode):
                        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains symlinks")
                    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                        raise MinerUApiError(code="unsafe_result_archive", message="result archive contains special files")
                target = (destination_root / member).resolve()
                if target == original_pdf.resolve():
                    raise MinerUApiError(code="unsafe_result_archive", message="result archive attempts to overwrite the source PDF")
                if destination_root not in target.parents and target != destination_root:
                    raise MinerUApiError(code="unsafe_result_archive", message="result archive escapes destination root")
                if is_dir:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as sink:
                    sink.write(source.read())
                extracted_paths.append(relative)
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
        timeout_sec: float = 300.0,
        poll_interval_sec: float = 1.0,
        max_poll_attempts: int = 120,
        max_result_bytes: int = 100 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        parsed = urlparse(self.base_url)
        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._api_token = str(api_token or "").strip()
        self._token_header = str(token_header or "Authorization").strip() or "Authorization"
        self._allow_insecure_remote_http = bool(allow_insecure_remote_http)
        self._timeout_sec = float(timeout_sec)
        self._poll_interval_sec = float(poll_interval_sec)
        self._max_poll_attempts = int(max_poll_attempts)
        self._max_result_bytes = int(max_result_bytes)
        self._transport = transport

    def configured(self) -> bool:
        return bool(self.base_url)

    def health(self) -> dict[str, Any]:
        response = self._client().get(self._url("/health"))
        if response.status_code != 200:
            raise MinerUApiError(
                code="health_check_failure",
                message=f"MinerU health check failed with HTTP {response.status_code}",
            )
        payload = self._json_object(response, code="health_check_failure")
        return payload

    def submit_pdf(self, *, request: DocumentParseRequest, input_pdf: Path) -> MinerUApiTaskSubmission:
        self.validate_upload_policy(request)
        files = [("files", (input_pdf.name, input_pdf.read_bytes(), "application/pdf"))]
        response = self._client().post(
            self._url("/tasks"),
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
        return MinerUApiTaskSubmission.model_validate(payload)

    def get_task_status(self, task_id: str) -> MinerUApiTaskStatus:
        response = self._client().get(self._url(f"/tasks/{task_id}"))
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
        for _ in range(self._max_poll_attempts):
            status = self.get_task_status(task_id)
            history.append(status.state)
            if status.queued_ahead is not None:
                queued_history.append(int(status.queued_ahead))
            if status.state == "completed":
                return status, history, queued_history
            if status.state in {"pending", "processing", "queued", "running"}:
                continue
            if status.state == "failed":
                raise MinerUApiError(code="task_failed", message=status.message or "MinerU task failed", details={"task_id": task_id})
            if status.state == "cancelled":
                raise MinerUApiError(code="task_cancelled", message=status.message or "MinerU task cancelled", details={"task_id": task_id})
            raise MinerUApiError(
                code="task_failed",
                message=f"MinerU task entered unknown state {status.state!r}",
                details={"task_id": task_id},
            )
        raise MinerUApiError(code="task_timeout", message="MinerU task polling exceeded the configured limit", details={"task_id": task_id})

    def download_result(self, *, task_id: str, output_dir: Path, original_pdf: Path) -> MinerUDownloadedResult:
        response = self._client().get(self._url(f"/tasks/{task_id}/result"))
        if response.status_code != 200:
            raise MinerUApiError(
                code="result_download_failure",
                message=f"MinerU result download failed with HTTP {response.status_code}",
                details={"task_id": task_id},
            )
        archive_bytes = bytes(response.content)
        if len(archive_bytes) > self._max_result_bytes:
            raise MinerUApiError(
                code="oversized_result",
                message="MinerU result archive exceeds the configured size limit",
                details={"task_id": task_id},
            )
        extracted = safe_extract_result_archive(
            archive_bytes=archive_bytes,
            destination_dir=output_dir,
            original_pdf=original_pdf,
        )
        return MinerUDownloadedResult(output_dir=output_dir, extracted_relative_paths=extracted)

    def parse_pdf(self, *, request: DocumentParseRequest, input_pdf: Path, output_dir: Path) -> MinerUApiParseOutcome:
        self.validate_upload_policy(request)
        submission = self.submit_pdf(request=request, input_pdf=input_pdf)
        status, history, queued_history = self.wait_for_task(submission.task_id)
        downloaded = self.download_result(
            task_id=submission.task_id,
            output_dir=output_dir,
            original_pdf=input_pdf,
        )
        return MinerUApiParseOutcome(
            remote_task_id=submission.task_id,
            output_dir=downloaded.output_dir,
            extracted_relative_paths=downloaded.extracted_relative_paths,
            task_status_history=history,
            queued_ahead_history=queued_history,
            mineru_version=status.mineru_version,
            protocol_version=status.protocol_version,
            backend=status.backend or request.backend,
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
        except json.JSONDecodeError as exc:
            raise MinerUApiError(code=code, message="MinerU response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MinerUApiError(code=code, message="MinerU response JSON root must be an object")
        return payload

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
            timeout=self._timeout_sec,
            transport=self._transport,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

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
