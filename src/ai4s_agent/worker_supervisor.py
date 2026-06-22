from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

WorkerStatus = Literal["pending", "running", "stopped", "failed"]


@dataclass
class WorkerHeartbeat:
    run_id: str
    pid: int
    status: WorkerStatus = "pending"
    started_at: str = ""
    command: list[str] = field(default_factory=list)
    cwd: str = ""
    exit_code: int | None = None
    updated_at: str = ""


class WorkerSupervisor:
    """Lightweight local-process worker lifecycle manager.

    Manages worker processes by project-scoped run_id.  Each worker writes a
    heartbeat JSON file into its project run directory.  The supervisor is
    intentionally decoupled from RunPlanExecutor and remote workers — it
    provides start / status / stop primitives that a future durable-job layer
    can build on.
    """

    def __init__(self, projects_root: Path) -> None:
        self._projects_root = Path(projects_root).resolve()
        self._workers: dict[tuple[str, str], subprocess.Popen[Any]] = {}

    # ------------------------------------------------------------------ public

    def start(
        self,
        *,
        project_id: str,
        run_id: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> WorkerHeartbeat:
        project, run = _normalize_ids(project_id, run_id)
        key = _worker_key(project, run)
        if key in self._workers:
            existing = self.status(project, run)
            if existing.status == "running":
                raise ValueError(f"worker already running for {project}/{run}")
            del self._workers[key]

        work_dir = Path(cwd or self._projects_root).resolve()
        run_cwd = str(work_dir)
        heartbeat = WorkerHeartbeat(
            run_id=run,
            pid=-1,
            status="pending",
            started_at=_now_iso(),
            command=list(command),
            cwd=run_cwd,
        )
        self._write_heartbeat(project, run, heartbeat)

        proc = subprocess.Popen(
            command,
            cwd=run_cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        heartbeat.pid = proc.pid
        heartbeat.status = "running"
        heartbeat.updated_at = _now_iso()
        self._workers[key] = proc
        self._write_heartbeat(project, run, heartbeat)
        return heartbeat

    def status(self, project_id: str, run_id: str) -> WorkerHeartbeat:
        project, run = _normalize_ids(project_id, run_id)
        key = _worker_key(project, run)
        cached = self._read_heartbeat(project, run)
        proc = self._workers.get(key)
        if proc is None:
            if cached.status in ("running", "pending") and cached.pid > 0:
                if not _process_alive(cached.pid):
                    cached.status = "failed"
                    cached.exit_code = -1
                    cached.updated_at = _now_iso()
                    self._write_heartbeat(project, run, cached)
            return cached

        returncode = proc.poll()
        if returncode is not None:
            cached.status = "stopped" if returncode == 0 else "failed"
            cached.exit_code = returncode
            cached.updated_at = _now_iso()
            self._write_heartbeat(project, run, cached)
        return cached

    def stop(self, project_id: str, run_id: str, *, timeout_sec: int = 10) -> WorkerHeartbeat:
        project, run = _normalize_ids(project_id, run_id)
        key = _worker_key(project, run)
        proc = self._workers.get(key)
        if proc is None:
            return self.status(project, run)

        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        return self.status(project, run)

    def list_workers(self, project_id: str) -> list[WorkerHeartbeat]:
        project = _safe_component(project_id, "project_id")
        run_dir = self._projects_root / project / "runs"
        heartbeats: list[WorkerHeartbeat] = []
        if not run_dir.exists():
            return heartbeats
        for child in sorted(run_dir.iterdir()):
            hb_path = child / "worker_heartbeat.json"
            if hb_path.exists():
                heartbeats.append(self._parse_heartbeat(hb_path))
        return heartbeats

    # ---------------------------------------------------------------- private

    def _heartbeat_path(self, project_id: str, run_id: str, *, create: bool) -> Path:
        run_dir = self._projects_root / project_id / "runs" / run_id
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "worker_heartbeat.json"

    def _write_heartbeat(self, project_id: str, run_id: str, heartbeat: WorkerHeartbeat) -> Path:
        path = self._heartbeat_path(project_id, run_id, create=True)
        path.write_text(
            json.dumps(
                {
                    "run_id": heartbeat.run_id,
                    "pid": heartbeat.pid,
                    "status": heartbeat.status,
                    "started_at": heartbeat.started_at,
                    "command": heartbeat.command,
                    "cwd": heartbeat.cwd,
                    "exit_code": heartbeat.exit_code,
                    "updated_at": heartbeat.updated_at,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return path

    def _read_heartbeat(self, project_id: str, run_id: str) -> WorkerHeartbeat:
        path = self._heartbeat_path(project_id, run_id, create=False)
        return self._parse_heartbeat(path)

    @staticmethod
    def _parse_heartbeat(path: Path) -> WorkerHeartbeat:
        if not path.exists():
            return WorkerHeartbeat(run_id="", pid=-1, status="pending")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return WorkerHeartbeat(run_id="", pid=-1, status="pending")
        return WorkerHeartbeat(
            run_id=str(data.get("run_id") or ""),
            pid=int(data.get("pid", -1)),
            status=str(data.get("status") or "pending"),
            started_at=str(data.get("started_at") or ""),
            command=[str(item) for item in data.get("command", [])],
            cwd=str(data.get("cwd") or ""),
            exit_code=int(data["exit_code"]) if data.get("exit_code") is not None else None,
            updated_at=str(data.get("updated_at") or ""),
        )


def _worker_key(project_id: str, run_id: str) -> tuple[str, str]:
    return (_safe_component(project_id, "project_id"), _safe_component(run_id, "run_id"))


def _normalize_ids(project_id: str, run_id: str) -> tuple[str, str]:
    return _worker_key(project_id, run_id)


def _safe_component(value: str, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} must not be empty")
    if clean in {".", ".."}:
        raise ValueError(f"{field_name} must not be a reserved name: {clean}")
    if any(ch in clean for ch in "/\\"):
        raise ValueError(f"{field_name} must not contain path separators: {clean}")
    return clean


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
