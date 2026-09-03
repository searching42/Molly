"""Small dependency-free local HTTP surface for Molly Core."""

from __future__ import annotations

from base64 import b64decode
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
import hmac
import html
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import secrets
import threading
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from molly.core import ApprovalDecision, ReviewDecision
from molly.core.errors import (
    ArtifactError,
    ApprovalError,
    CoreContractError,
    InspectionError,
    ReviewError,
    RunError,
    ToolError,
)
from molly.core.ids import validate_identifier
from molly.observability import (
    ExporterUnavailableError,
    JsonTraceExporter,
    LangSmithExporter,
    ObserverIntegrityError,
    OpenTelemetryExporter,
)
from molly.runtime import (
    RuntimeBindingError,
    RuntimeProfileRegistry,
    RuntimeProfileUnavailable,
    RuntimeService,
    RuntimeStateError,
)

from .providers import ProviderConfigError, ProviderConfigStore
from .runtime_profiles import configured_br1_profiles


# OE62 ``df_5k`` split-JSON files are roughly 50 MB.  The current dependency-
# free browser transport is JSON/base64, so the request limit includes its
# expansion.  A later multipart transport can reduce peak memory without
# changing the artifact contract.
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_REQUEST_BYTES = 192 * 1024 * 1024
LOCAL_SESSION_TOKEN_HEADER = "X-Local-Session-Token"
LOCAL_SESSION_TOKEN_META = "local-session-token"
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})

STATUS_LABELS = {
    "NEW": "未开始",
    "ACTIVE": "执行中",
    "WAITING_APPROVAL": "等待确认",
    "WAITING_REVIEW": "等待审阅",
    "INTERRUPTED": "已中断",
    "STOPPED": "已完成",
    "FAILED": "执行失败",
}


class WebRequestError(Exception):
    """An expected request error with a safe browser-facing message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _status_payload(status: str) -> dict[str, str]:
    return {"code": status, "label": STATUS_LABELS.get(status, status)}


def _profile_label(profile: Any) -> tuple[str, str]:
    config = profile.config
    name = config.get("display_name") if isinstance(config, Mapping) else None
    description = config.get("description") if isinstance(config, Mapping) else None
    safe_name = name.strip() if isinstance(name, str) and name.strip() else profile.profile_id
    safe_description = (
        description.strip()
        if isinstance(description, str) and description.strip()
        else "由服务器端注册的运行配置"
    )
    return safe_name, safe_description


def _runtime_profile_view(profile: Any) -> dict[str, Any]:
    name, description = _profile_label(profile)
    config = profile.config if isinstance(profile.config, Mapping) else {}
    resource_constraints = config.get("resource_constraints", {})
    return {
        "profile_id": profile.profile_id,
        "name": name,
        "description": description,
        "available": profile.decision_provider_factory is not None,
        "workflow": config.get("workflow", "core"),
        "backend_kind": config.get("backend_kind", "local"),
        "host_identity": config.get("host_identity"),
        "resource_constraints": resource_constraints if isinstance(resource_constraints, Mapping) else {},
    }


def _run_summary(inspection: Any) -> dict[str, Any]:
    return {
        "run_id": inspection.run_id,
        "goal": inspection.goal,
        "status": inspection.status,
        "status_label": STATUS_LABELS.get(inspection.status, inspection.status),
        "step_count": inspection.step_count,
        "tool_call_count": inspection.tool_call_count,
        "artifact_count": len(inspection.referenced_artifact_ids),
        "needs_action": inspection.status
        in {"WAITING_APPROVAL", "WAITING_REVIEW", "INTERRUPTED"},
    }


def _safe_exception_response(exc: BaseException) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, WebRequestError):
        return exc.status, {"error_type": exc.code, "message": exc.message}
    if isinstance(exc, ProviderConfigError):
        return 400, {
            "error_type": "PROVIDER_CONFIG_INVALID",
            "message": "模型服务配置无效",
        }
    if isinstance(exc, RuntimeStateError):
        return 503, {
            "error_type": "RUNTIME_STATE_UNAVAILABLE",
            "message": "本地任务状态暂不可用",
        }
    if isinstance(exc, RuntimeProfileUnavailable):
        return 409, {
            "error_type": "RUNTIME_PROFILE_UNAVAILABLE",
            "message": "当前运行配置不可用",
        }
    if isinstance(exc, (RuntimeBindingError, ApprovalError, ReviewError, RunError, ToolError)):
        return 409, {
            "error_type": "CORE_OPERATION_REJECTED",
            "message": "Core 拒绝了这次操作",
        }
    if isinstance(exc, ExporterUnavailableError):
        return 503, {
            "error_type": "OBSERVER_UNAVAILABLE",
            "message": "监控导出配置或依赖不可用",
        }
    if isinstance(exc, ObserverIntegrityError):
        return 409, {
            "error_type": "OBSERVER_INTEGRITY_ERROR",
            "message": "监控导出未通过 Core 完整性检查",
        }
    if isinstance(exc, (InspectionError, ArtifactError)):
        return 404, {
            "error_type": "RESOURCE_UNAVAILABLE",
            "message": "请求的任务或数据文件不可用",
        }
    if isinstance(exc, CoreContractError):
        return 400, {
            "error_type": "INVALID_REQUEST",
            "message": "请求内容不符合要求",
        }
    return 500, {
        "error_type": "WEB_SERVER_ERROR",
        "message": "本地服务遇到未预期的问题",
    }


class MollyWebApplication:
    """Translate small browser requests into existing Core operations."""

    def __init__(
        self,
        *,
        service: RuntimeService,
        provider_store: ProviderConfigStore,
        static_root: Path | None = None,
    ) -> None:
        if not isinstance(service, RuntimeService):
            raise TypeError("service must be a RuntimeService")
        if not isinstance(provider_store, ProviderConfigStore):
            raise TypeError("provider_store must be a ProviderConfigStore")
        self.service = service
        self.provider_store = provider_store
        self.static_root = static_root or Path(__file__).with_name("static")
        self._local_session_token = secrets.token_urlsafe(32)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="molly-run")
        self._futures: dict[str, Future[Any]] = {}
        self._future_lock = threading.Lock()

    @property
    def local_session_token(self) -> str:
        """Return the process-local token used by the browser write surface."""

        return self._local_session_token

    def dispatch(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Handle one request without exposing exception details to clients."""

        try:
            return self._dispatch(method.upper(), path, payload)
        except Exception as exc:  # the HTTP surface must never leak raw exceptions
            return _safe_exception_response(exc)

    def static_file(self, path: str) -> tuple[int, bytes, str] | None:
        parsed = urlsplit(path)
        requested = unquote(parsed.path)
        if requested == "/":
            name = "index.html"
        elif requested.startswith("/static/"):
            name = requested.removeprefix("/static/")
        else:
            return None
        if name not in {"index.html", "app.js", "styles.css"}:
            return None
        candidate = (self.static_root / name).resolve()
        if not candidate.is_relative_to(self.static_root.resolve()) or not candidate.is_file():
            return None
        media_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }[name]
        content = candidate.read_bytes()
        if name == "index.html":
            content = content.replace(
                b"__LOCAL_SESSION_TOKEN__",
                html.escape(self.local_session_token, quote=True).encode("ascii"),
            )
        return 200, content, media_type

    def _dispatch(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
    ) -> tuple[int, dict[str, Any]]:
        parsed = urlsplit(path)
        route = [unquote(part) for part in parsed.path.split("/") if part]
        body = {} if payload is None else payload
        if not isinstance(body, Mapping):
            raise WebRequestError(400, "INVALID_JSON", "请求内容必须是 JSON 对象")

        if route == ["api", "health"] and method == "GET":
            return 200, {"status": "ok", "service": "local-scientific-workbench"}
        if route == ["api", "bootstrap"] and method == "GET":
            return 200, self._bootstrap()
        if route == ["api", "runs"]:
            if method == "GET":
                return 200, {"runs": self._run_summaries()}
            if method == "POST":
                return 201, self._start_run(body)
        if len(route) == 3 and route[:2] == ["api", "runs"]:
            run_id = route[2]
            if method == "GET":
                return 200, self._run_detail(run_id)
        if len(route) == 4 and route[:2] == ["api", "runs"]:
            run_id, action = route[2], route[3]
            if method == "POST" and action == "resume":
                return 200, self._resume_run(run_id)
            if method == "POST" and action == "approval":
                return 200, self._approve_run(run_id, body)
            if method == "POST" and action == "observe":
                return 200, self._observe_run(run_id, body)
        if route == ["api", "artifacts"] and method == "POST":
            return 201, self._upload_artifact(body)
        if len(route) == 3 and route[:2] == ["api", "artifacts"]:
            artifact_id = route[2]
            if method == "GET":
                return 200, self.service.inspect_artifact(artifact_id).to_dict()
        if len(route) == 4 and route[:2] == ["api", "artifacts"]:
            artifact_id, action = route[2], route[3]
            if method == "POST" and action == "review":
                return 201, self._review_artifact(artifact_id, body)
            if method == "GET" and action == "content":
                record = self.service.inspect_artifact(artifact_id)
                return 200, {
                    "artifact_id": record.artifact_id,
                    "media_type": record.media_type,
                    "size_bytes": record.size_bytes,
                    "download": True,
                }
        if route == ["api", "model-profiles"]:
            if method == "GET":
                return 200, {"profiles": self._provider_profiles()}
            if method == "POST":
                return 201, self._save_provider_profile(body)
        if len(route) == 4 and route[:2] == ["api", "model-profiles"]:
            profile_ref, action = route[2], route[3]
            if method == "POST" and action == "credential":
                return 200, self._save_provider_credential(profile_ref, body)
            if method == "POST" and action == "check":
                return 200, self._check_provider_profile(profile_ref)
        if route and route[0] == "api":
            raise WebRequestError(404, "NOT_FOUND", "页面或接口不存在")
        raise WebRequestError(404, "NOT_FOUND", "页面不存在")

    def _bootstrap(self) -> dict[str, Any]:
        return {
            "app_subtitle": "科学任务工作台",
            "runtime_profiles": [
                _runtime_profile_view(profile) for profile in self.service.profiles.profiles
            ],
            "model_profiles": self._provider_profiles(),
            "runs": self._run_summaries(),
            "compute_profiles": [
                {
                    "profile_id": item["profile_id"],
                    "name": item["name"],
                    "workflow": item["workflow"],
                    "backend_kind": item["backend_kind"],
                    "host_identity": item["host_identity"],
                    "resource_constraints": item["resource_constraints"],
                }
                for item in (_runtime_profile_view(profile) for profile in self.service.profiles.profiles)
                if item["workflow"] == "br1"
            ],
            "observability": {
                "json": {"available": True},
                "otel": {"available": bool(os.environ.get("MOLLY_OTEL_ENDPOINT", "").strip())},
                "langsmith": {
                    "available": bool(
                        os.environ.get("LANGSMITH_API_KEY", "").strip()
                        or os.environ.get("LANGCHAIN_API_KEY", "").strip()
                    )
                },
            },
        }

    def _run_summaries(self) -> list[dict[str, Any]]:
        try:
            inspections = self.service.list_runs()
        except RuntimeStateError:
            return []
        return [_run_summary(inspection) for inspection in reversed(inspections)]

    def _run_detail(self, run_id: str) -> dict[str, Any]:
        with self._future_lock:
            future = self._futures.get(run_id)
        background_error_type: str | None = None
        if future is not None and future.done():
            try:
                # Join a completed worker before projecting the ledger.  This
                # closes the small race where the future is marked done just
                # after inspection read a transient INTERRUPTED state.
                future.result()
            except Exception as exc:
                error_type = type(exc).__name__
                background_error_type = (
                    error_type if error_type.isidentifier() else "BackgroundRunError"
                )
        inspection = self.service.inspect_run(run_id)
        value = inspection.to_dict()
        value["status_label"] = STATUS_LABELS.get(inspection.status, inspection.status)
        value["step_count"] = inspection.step_count
        value["tool_call_count"] = inspection.tool_call_count
        value["artifact_count"] = len(inspection.referenced_artifact_ids)
        try:
            profile = self.service.profiles.resolve(
                inspection.runtime_profile_ref or "",
                expected_digest=inspection.runtime_profile_digest,
            )
            value["workflow"] = (
                profile.config.get("workflow", "core")
                if isinstance(profile.config, Mapping)
                else "core"
            )
            value["runtime_profile"] = _runtime_profile_view(profile)
        except Exception:
            # The authoritative inspection remains useful even if a profile
            # was removed from the currently running server.  The runtime
            # operation itself will fail closed on resume.
            value["workflow"] = "unknown"
            value["runtime_profile"] = None
        background_pending = future is not None and not future.done()
        value["background_pending"] = background_pending
        if background_error_type is not None:
            value["background_error_type"] = background_error_type
        return value

    def _is_br1_run(self, run_id: str) -> bool:
        inspection = self.service.inspect_run(run_id)
        if not inspection.runtime_profile_ref or not inspection.runtime_profile_digest:
            return False
        profile = self.service.profiles.resolve(
            inspection.runtime_profile_ref,
            expected_digest=inspection.runtime_profile_digest,
        )
        return isinstance(profile.config, Mapping) and profile.config.get("workflow") == "br1"

    def _submit_background(self, run_id: str, operation: Callable[[], Any]) -> bool:
        with self._future_lock:
            current = self._futures.get(run_id)
            if current is not None and not current.done():
                return False
            self._futures[run_id] = self._executor.submit(operation)
            return True

    def _background_response(self, run_id: str, message: str) -> dict[str, Any]:
        value = self._run_detail(run_id)
        value["message"] = message
        value["background_pending"] = True
        return {"run_id": run_id, "status": value["status"], "inspection": value, "message": message}

    @staticmethod
    def _artifact_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
        value = payload.get("input_artifact_ids", ())
        if not isinstance(value, (list, tuple)):
            raise WebRequestError(400, "INVALID_FILES", "数据文件列表格式不正确")
        if len(value) > 32 or any(not isinstance(item, str) for item in value):
            raise WebRequestError(400, "INVALID_FILES", "数据文件列表格式不正确")
        return tuple(value)

    @staticmethod
    def _result_payload(service: RuntimeService, result: Any) -> dict[str, Any]:
        inspection = service.inspect_run(result.run_id)
        value = result.to_dict()
        value["status_label"] = STATUS_LABELS.get(result.status, result.status)
        value["inspection"] = inspection.to_dict()
        value["inspection"]["status_label"] = STATUS_LABELS.get(
            inspection.status, inspection.status
        )
        return value

    def _start_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = payload.get("profile_id")
        goal = payload.get("goal")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise WebRequestError(400, "PROFILE_REQUIRED", "请选择运行配置")
        if not isinstance(goal, str) or not goal.strip():
            raise WebRequestError(400, "GOAL_REQUIRED", "请先写下任务目标")
        llm_profile_ref = payload.get("llm_profile_ref")
        metadata: dict[str, Any] = {}
        if llm_profile_ref is not None:
            if not isinstance(llm_profile_ref, str) or not llm_profile_ref.strip():
                raise WebRequestError(400, "INVALID_LLM_PROFILE", "模型服务配置不正确")
            validate_identifier(llm_profile_ref.strip(), field="llm_profile_ref")
            llm_profile = self.provider_store.get_profile(llm_profile_ref.strip())
            if not llm_profile.credential_configured:
                raise WebRequestError(
                    409,
                    "LLM_CREDENTIAL_REQUIRED",
                    "请先为所选模型服务保存 API Key",
                )
            metadata["llm_profile_ref"] = llm_profile_ref.strip()
            metadata["llm_profile_digest"] = llm_profile.profile.digest
        try:
            runtime_profile = self.service.profiles.resolve(profile_id.strip())
        except Exception:
            runtime_profile = None
        workflow = (
            runtime_profile.config.get("workflow")
            if runtime_profile is not None and isinstance(runtime_profile.config, Mapping)
            else None
        )
        if workflow == "br1" and "llm_profile_ref" not in metadata:
            raise WebRequestError(
                400,
                "LLM_PROFILE_REQUIRED",
                "BR1 任务必须选择自然语言解析模型服务",
            )
        result = self.service.start_run(
            profile_id=profile_id.strip(),
            goal=goal.strip(),
            input_artifact_ids=self._artifact_ids(payload),
            metadata=metadata,
        )
        return self._result_payload(self.service, result)

    def _resume_run(self, run_id: str) -> dict[str, Any]:
        if self._is_br1_run(run_id):
            if not self._submit_background(run_id, lambda: self.service.resume_run(run_id)):
                return self._background_response(run_id, "任务正在后台执行")
            return self._background_response(run_id, "任务已提交后台执行")
        return self._result_payload(self.service, self.service.resume_run(run_id))

    def _approve_run(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        decision = payload.get("decision")
        if decision not in {item.value for item in ApprovalDecision}:
            raise WebRequestError(400, "INVALID_DECISION", "确认结果不正确")
        reviewer_ref = payload.get("reviewer_ref", "local-user")
        if not isinstance(reviewer_ref, str) or not reviewer_ref.strip():
            raise WebRequestError(400, "REVIEWER_REQUIRED", "确认人不能为空")
        call_id = payload.get("call_id")
        if call_id is not None and not isinstance(call_id, str):
            raise WebRequestError(400, "INVALID_CALL", "操作标识不正确")
        if self._is_br1_run(run_id):
            if not self._submit_background(
                run_id,
                lambda: self.service.record_approval(
                    run_id,
                    decision=decision,
                    reviewer_ref=reviewer_ref.strip(),
                    call_id=call_id,
                ),
            ):
                return self._background_response(run_id, "任务正在后台执行")
            return self._background_response(run_id, "确认已提交，任务正在后台执行")
        outcome = self.service.record_approval(
            run_id,
            decision=decision,
            reviewer_ref=reviewer_ref.strip(),
            call_id=call_id,
        )
        value = outcome.to_dict()
        value["result"] = self._result_payload(self.service, outcome.result)
        return value

    def _upload_artifact(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        encoded = payload.get("content_base64")
        if not isinstance(encoded, str) or not encoded:
            raise WebRequestError(400, "FILE_REQUIRED", "请选择一个数据文件")
        if len(encoded) > (MAX_UPLOAD_BYTES * 4 // 3) + 16:
            raise WebRequestError(413, "FILE_TOO_LARGE", "数据文件不能超过 128 MB")
        try:
            content = b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise WebRequestError(400, "INVALID_FILE", "数据文件内容无法读取") from None
        if not content:
            raise WebRequestError(400, "EMPTY_FILE", "不能上传空文件")
        if len(content) > MAX_UPLOAD_BYTES:
            raise WebRequestError(413, "FILE_TOO_LARGE", "数据文件不能超过 128 MB")
        file_name = payload.get("file_name", "")
        if not isinstance(file_name, str) or len(file_name) > 200 or any(
            char in file_name for char in "\r\n\x00"
        ):
            raise WebRequestError(400, "INVALID_FILE_NAME", "文件名不正确")
        media_type = payload.get("media_type")
        if media_type is None:
            media_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        if not isinstance(media_type, str) or not media_type or any(
            char in media_type for char in "\r\n\x00"
        ):
            raise WebRequestError(400, "INVALID_MEDIA_TYPE", "文件类型不正确")
        record = self.service.publish_artifact(content, media_type=media_type)
        return {
            "artifact_id": record.artifact_id,
            "name": file_name or "未命名文件",
            "media_type": record.media_type,
            "size_bytes": record.size_bytes,
            "download_path": f"/api/artifacts/{record.artifact_id}/content",
        }

    def artifact_content(self, artifact_id: str) -> tuple[int, bytes, str, str]:
        """Return one verified artifact for the HTTP download adapter."""

        record, content = self.service.read_artifact(artifact_id)
        return 200, content, record.media_type, f"artifact-{record.sha256}.bin"

    def close(self) -> None:
        """Stop accepting new background work when the local server exits."""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def _review_artifact(self, artifact_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        decision = payload.get("decision")
        if decision not in {item.value for item in ReviewDecision}:
            raise WebRequestError(400, "INVALID_REVIEW", "审阅结果不正确")
        reviewer_ref = payload.get("reviewer_ref", "local-user")
        reason = payload.get("reason", "")
        if not isinstance(reviewer_ref, str) or not reviewer_ref.strip():
            raise WebRequestError(400, "REVIEWER_REQUIRED", "审阅人不能为空")
        if not isinstance(reason, str) or len(reason) > 2_000:
            raise WebRequestError(400, "INVALID_REVIEW", "审阅说明过长")
        return self.service.create_review(
            artifact_id,
            decision=decision,
            reviewer_ref=reviewer_ref.strip(),
            reason=reason,
        ).to_dict()

    def _observe_run(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        exporter_name = payload.get("exporter", "json")
        if exporter_name == "json":
            exporter = JsonTraceExporter()
        elif exporter_name == "otel":
            endpoint = os.environ.get("MOLLY_OTEL_ENDPOINT", "").strip() or None
            exporter = OpenTelemetryExporter(endpoint=endpoint)
        elif exporter_name == "langsmith":
            api_url = os.environ.get("MOLLY_LANGSMITH_API_URL", "").strip() or None
            exporter = LangSmithExporter(api_url=api_url)
        else:
            raise WebRequestError(400, "INVALID_EXPORTER", "不支持的监控导出类型")
        return self.service.observe_run(run_id, exporter).to_dict()

    def _provider_profiles(self) -> list[dict[str, Any]]:
        return [view.to_public_dict() for view in self.provider_store.list_profiles()]

    def _save_provider_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        view = self.provider_store.upsert_profile(payload)
        return {
            "profile": view.to_public_dict(),
            "message": "已保存非敏感配置；密钥仍只保存在本机服务端",
        }

    def _check_provider_profile(self, profile_ref: str) -> dict[str, Any]:
        validate_identifier(profile_ref, field="provider profile_ref")
        view = self.provider_store.get_profile(profile_ref)
        if view.credential_configured:
            return {
                "ready": True,
                "credential_status": "已配置",
                "message": "本机服务端已找到密钥；此次仅检查本地配置，没有向外部服务发起请求",
            }
        return {
            "ready": False,
            "credential_status": "未配置",
            "message": "请在本页面输入 API Key，密钥只会保存到本机服务端",
        }

    def _save_provider_credential(
        self, profile_ref: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        validate_identifier(profile_ref, field="provider profile_ref")
        if set(payload) != {"api_key"}:
            raise WebRequestError(400, "INVALID_CREDENTIAL", "请求只允许包含 API Key")
        api_key = payload.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise WebRequestError(400, "INVALID_CREDENTIAL", "API Key 不能为空")
        self.provider_store.set_secret(profile_ref, api_key)
        return {
            "profile_ref": profile_ref,
            "credential_configured": True,
            "credential_status": "已配置",
            "message": "API Key 已安全保存到本机服务端，不会显示或写入任务数据",
        }


class MollyHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter kept separate from the application for direct testing."""

    application: MollyWebApplication
    server_version = "LocalScientificWorkbench/0.1"

    def _headers(self, *, content_type: str, length: int, cache: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")

    def _send_json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._headers(content_type="application/json; charset=utf-8", length=len(payload), cache="no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, status: int, content: bytes, media_type: str) -> None:
        self.send_response(status)
        self._headers(content_type=media_type, length=len(content), cache="no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_binary(
        self,
        status: int,
        content: bytes,
        media_type: str,
        download_name: str,
    ) -> None:
        self.send_response(status)
        self._headers(content_type=media_type, length=len(content), cache="no-store")
        self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(content)

    def _read_json(self) -> Mapping[str, Any]:
        content_types = self.headers.get_all("Content-Type") or []
        if len(content_types) != 1:
            raise WebRequestError(415, "JSON_CONTENT_TYPE_REQUIRED", "写请求必须使用 application/json")
        media_type = content_types[0].split(";", 1)[0].strip().casefold()
        if media_type != "application/json":
            raise WebRequestError(415, "JSON_CONTENT_TYPE_REQUIRED", "写请求必须使用 application/json")
        header = self.headers.get("Content-Length")
        try:
            length = int(header) if header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            raise WebRequestError(411, "LENGTH_REQUIRED", "请求大小未知")
        if length > MAX_REQUEST_BYTES:
            raise WebRequestError(413, "REQUEST_TOO_LARGE", "请求内容过大")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise WebRequestError(400, "INVALID_REQUEST", "请求内容不完整")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise WebRequestError(400, "INVALID_JSON", "请求内容不是有效 JSON") from None
        if not isinstance(value, Mapping):
            raise WebRequestError(400, "INVALID_JSON", "请求内容必须是 JSON 对象")
        return value

    @staticmethod
    def _loopback_hostname(hostname: str) -> bool:
        clean = str(hostname or "").strip().casefold()
        if clean == "localhost":
            return True
        try:
            return ipaddress.ip_address(clean).is_loopback
        except ValueError:
            return False

    def _validate_host(self) -> None:
        values = self.headers.get_all("Host") or []
        if len(values) != 1 or not values[0].strip():
            raise WebRequestError(403, "LOCAL_HOST_REQUIRED", "只允许通过本机 Host 访问")
        authority = values[0].strip()
        parsed = None
        try:
            parsed = urlsplit("//" + authority)
            hostname = parsed.hostname or ""
            _ = parsed.port
        except ValueError:
            hostname = ""
        if (
            parsed is None
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or not self._loopback_hostname(hostname)
        ):
            raise WebRequestError(403, "LOCAL_HOST_REQUIRED", "只允许通过本机 Host 访问")

    def _validate_origin(self) -> None:
        values = self.headers.get_all("Origin") or []
        if len(values) != 1 or not values[0].strip():
            raise WebRequestError(403, "LOCAL_ORIGIN_REQUIRED", "写请求必须来自本机 Origin")
        try:
            parsed = urlsplit(values[0].strip())
            _ = parsed.port
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.scheme.casefold() not in {"http", "https"}
            or not self._loopback_hostname(parsed.hostname or "")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise WebRequestError(403, "LOCAL_ORIGIN_REQUIRED", "写请求必须来自本机 Origin")

    def _authorize_request(self, *, mutation: bool) -> None:
        remote_address = self.client_address[0] if self.client_address else ""
        if not self._loopback_hostname(remote_address):
            raise WebRequestError(403, "LOCAL_CLIENT_REQUIRED", "只允许本机客户端访问")
        self._validate_host()
        if not mutation:
            return
        self._validate_origin()
        fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().casefold()
        if fetch_site and fetch_site not in _ALLOWED_FETCH_SITES:
            raise WebRequestError(403, "LOCAL_ORIGIN_REQUIRED", "跨站写请求已拒绝")
        tokens = self.headers.get_all(LOCAL_SESSION_TOKEN_HEADER) or []
        if len(tokens) != 1 or not hmac.compare_digest(
            tokens[0], self.application.local_session_token
        ):
            raise WebRequestError(403, "LOCAL_SESSION_REQUIRED", "需要本机会话令牌")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            self._authorize_request(mutation=False)
        except Exception as exc:
            status, value = _safe_exception_response(exc)
            self._send_json(status, value)
            return
        if self.path.startswith("/api/"):
            parsed = urlsplit(self.path)
            route = [unquote(part) for part in parsed.path.split("/") if part]
            if len(route) == 4 and route[:2] == ["api", "artifacts"] and route[3] == "content":
                try:
                    status, content, media_type, download_name = self.application.artifact_content(route[2])
                except Exception as exc:
                    status, value = _safe_exception_response(exc)
                    self._send_json(status, value)
                    return
                self._send_binary(status, content, media_type, download_name)
                return
            status, value = self.application.dispatch("GET", self.path)
            self._send_json(status, value)
            return
        static = self.application.static_file(self.path)
        if static is None:
            self._send_json(404, {"error_type": "NOT_FOUND", "message": "页面不存在"})
            return
        status, content, media_type = static
        self._send_static(status, content, media_type)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            self._authorize_request(mutation=True)
            payload = self._read_json()
        except Exception as exc:
            status, value = _safe_exception_response(exc)
            self._send_json(status, value)
            return
        status, value = self.application.dispatch("POST", self.path, payload)
        self._send_json(status, value)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the default useful access log, but never print request bodies.
        super().log_message(format, *args)


def create_application(
    root: Path | str,
    *,
    service: RuntimeService | None = None,
    provider_store: ProviderConfigStore | None = None,
) -> MollyWebApplication:
    """Create the local web app from server-owned runtime profiles."""

    configured_root = Path(root)
    configured_provider_store = provider_store or ProviderConfigStore(configured_root)
    if service is None:
        profiles = configured_br1_profiles(
            configured_root,
            intent_provider_resolver=configured_provider_store.create_intent_provider,
        )
        service = RuntimeService(
            configured_root,
            profiles=RuntimeProfileRegistry(profiles),
        )
    return MollyWebApplication(
        service=service,
        provider_store=configured_provider_store,
    )


def serve(
    application: MollyWebApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> int:
    """Serve the UI on loopback until the operator presses Ctrl-C."""

    if not isinstance(application, MollyWebApplication):
        raise TypeError("application must be a MollyWebApplication")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be localhost or a loopback address")
    host = host.strip()
    if host.casefold() != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("host must be a loopback address")
        except ValueError as exc:
            if str(exc) == "host must be a loopback address":
                raise
            raise ValueError("host must be localhost or a loopback address") from exc
    handler = type(
        "BoundMollyHTTPRequestHandler",
        (MollyHTTPRequestHandler,),
        {"application": application},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"科学任务工作台: http://{host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        application.close()
    return 0


__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_UPLOAD_BYTES",
    "LOCAL_SESSION_TOKEN_HEADER",
    "LOCAL_SESSION_TOKEN_META",
    "MollyHTTPRequestHandler",
    "MollyWebApplication",
    "STATUS_LABELS",
    "create_application",
    "serve",
]
