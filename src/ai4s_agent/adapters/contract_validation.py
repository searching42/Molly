from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.adapters.runtime import AdapterRuntimeError, run_json_adapter_cmd


class ContractValidationError(RuntimeError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _expect_dict(obj: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ContractValidationError(
            code="invalid_type",
            message=f"{label} must be a JSON object",
            details={"label": label, "json_type": type(obj).__name__},
        )
    return obj


def validate_adapter_output_shape(
    output: dict[str, Any],
    *,
    required_top_level_keys: list[str] | None = None,
) -> dict[str, Any]:
    required = required_top_level_keys or ["status"]
    for key in required:
        if key not in output:
            raise ContractValidationError(
                code="missing_required_key",
                message=f"adapter output missing required key: {key}",
                details={"required": required, "output_keys": sorted(output.keys())},
            )

    status = output.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ContractValidationError(
            code="invalid_status",
            message="adapter output status must be a non-empty string",
            details={"status": status},
        )

    if status == "failed" and "error" not in output:
        raise ContractValidationError(
            code="missing_error",
            message="adapter output with status=failed must include error",
            details={"output_keys": sorted(output.keys())},
        )

    if "error" in output:
        err = _expect_dict(output.get("error"), label="error")
        code = err.get("code")
        msg = err.get("message")
        if code is not None and (not isinstance(code, str) or not code.strip()):
            raise ContractValidationError(
                code="invalid_error_code",
                message="error.code must be a non-empty string when present",
                details={"error": err},
            )
        if msg is not None and (not isinstance(msg, str) or not msg.strip()):
            raise ContractValidationError(
                code="invalid_error_message",
                message="error.message must be a non-empty string when present",
                details={"error": err},
            )
    return output


def validate_adapter_command_contract(
    *,
    cmd: str,
    payload: dict[str, Any],
    workspace_root: Path,
    timeout_sec: int = 120,
    required_top_level_keys: list[str] | None = None,
) -> dict[str, Any]:
    try:
        output = run_json_adapter_cmd(cmd=cmd, payload=payload, cwd=workspace_root, timeout_sec=timeout_sec)
    except AdapterRuntimeError as exc:
        raise ContractValidationError(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ) from exc
    return validate_adapter_output_shape(output, required_top_level_keys=required_top_level_keys)
