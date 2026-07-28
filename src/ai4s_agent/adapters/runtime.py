from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


class AdapterRuntimeError(RuntimeError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def parse_json_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "")
    if not text.strip():
        raise AdapterRuntimeError(code="empty_payload", message="stdin payload is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterRuntimeError(
            code="invalid_json_payload",
            message=f"payload is not valid JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise AdapterRuntimeError(
            code="invalid_payload_type",
            message="payload JSON must be an object",
            details={"json_type": type(payload).__name__},
        )
    return payload


def parse_cmd(cmd: str) -> list[str]:
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        raise AdapterRuntimeError(
            code="invalid_cmd",
            message=f"failed to parse command: {exc}",
            details={"cmd": cmd},
        ) from exc
    if not argv:
        raise AdapterRuntimeError(code="invalid_cmd", message="empty command")
    return argv


def run_json_adapter_cmd(
    *,
    cmd: str,
    payload: dict[str, Any],
    cwd: Path,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    argv = parse_cmd(cmd)
    return run_json_adapter_argv(argv=argv, payload=payload, cwd=cwd, timeout_sec=timeout_sec)


def run_json_adapter_argv(
    *,
    argv: list[str],
    payload: dict[str, Any],
    cwd: Path,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    if not argv:
        raise AdapterRuntimeError(code="invalid_cmd", message="empty argv")
    try:
        cp = subprocess.run(
            argv,
            cwd=str(cwd),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterRuntimeError(
            code="adapter_timeout",
            message=f"adapter command timed out after {timeout_sec}s",
            details={"argv": argv},
        ) from exc

    if cp.returncode != 0:
        raise AdapterRuntimeError(
            code="adapter_nonzero_exit",
            message=f"adapter command exited with code {cp.returncode}",
            details={
                "argv": argv,
                "returncode": cp.returncode,
                "stderr_tail": (cp.stderr or "")[-1000:],
                "stdout_tail": (cp.stdout or "")[-1000:],
            },
        )

    stdout = (cp.stdout or "").strip()
    if not stdout:
        raise AdapterRuntimeError(
            code="empty_stdout",
            message="adapter stdout is empty; expected one JSON object",
            details={"argv": argv},
        )

    try:
        out = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AdapterRuntimeError(
            code="invalid_json_stdout",
            message=f"adapter stdout is not valid JSON: {exc}",
            details={"argv": argv, "stdout_tail": stdout[-1000:]},
        ) from exc
    if not isinstance(out, dict):
        raise AdapterRuntimeError(
            code="invalid_json_stdout_type",
            message="adapter stdout JSON must be an object",
            details={"json_type": type(out).__name__},
        )
    return out


def run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int = 120) -> dict[str, Any]:
    if not argv:
        raise AdapterRuntimeError(code="invalid_cmd", message="empty argv")
    try:
        cp = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterRuntimeError(
            code="adapter_timeout",
            message=f"command timed out after {timeout_sec}s",
            details={"argv": argv},
        ) from exc

    return {
        "argv": argv,
        "returncode": cp.returncode,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
    }


def run_argv_cmd_with_env(
    *,
    argv: list[str],
    cwd: Path,
    timeout_sec: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not argv:
        raise AdapterRuntimeError(code="invalid_cmd", message="empty argv")
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    try:
        cp = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterRuntimeError(
            code="adapter_timeout",
            message=f"command timed out after {timeout_sec}s",
            details={"argv": argv},
        ) from exc

    return {
        "argv": argv,
        "returncode": cp.returncode,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
    }
