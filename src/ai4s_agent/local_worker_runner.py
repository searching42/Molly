from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import RunStatus


WorkerTask = Callable[["LocalWorkerContext"], dict[str, Any] | None]


@dataclass
class LocalWorkerResult:
    project_id: str
    run_id: str
    worker_id: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    should_stop_reason: str = ""


class LocalWorkerRunner:
    """Synchronous local worker loop backed by durable JobManager state.

    The runner is intentionally small: it does not start processes or threads by
    itself. It owns the worker lifecycle around an already-active project job so
    callers can plug it into a thread pool, process pool, CLI loop, or test.
    """

    def __init__(self, jobs: JobManager, *, worker_id: str = "local-worker", lease_ttl_sec: int = 300) -> None:
        self.jobs = jobs
        self.worker_id = _clean_segment(worker_id, "worker_id")
        self.lease_ttl_sec = _positive_int(lease_ttl_sec, "lease_ttl_sec")

    def run_project_job(self, project_id: str, run_id: str, task: WorkerTask) -> LocalWorkerResult:
        project = _clean_segment(project_id, "project_id")
        run = _clean_segment(run_id, "run_id")
        lease = acquire_project_worker_lease(self.jobs, project, run, worker_id=self.worker_id, lease_ttl_sec=self.lease_ttl_sec)
        lease_id = str(lease.get("worker_lease", {}).get("lease_id") or "")
        context = LocalWorkerContext(
            jobs=self.jobs,
            project_id=project,
            run_id=run,
            worker_id=self.worker_id,
            lease_id=lease_id,
            lease_ttl_sec=self.lease_ttl_sec,
        )
        self.jobs.add_project_log(project, run, "INFO", "local_worker", f"Local worker {self.worker_id} started")
        try:
            stop = context.should_stop()
            if stop["should_stop"]:
                final = _finish_cancelled_or_current(self.jobs, project, run)
                return LocalWorkerResult(project, run, self.worker_id, str(final.get("status") or RunStatus.CANCELLED.value), should_stop_reason=str(stop.get("reason") or ""))
            context.heartbeat()
            result = task(context) or {}
            stop = context.should_stop()
            if stop["should_stop"]:
                final = _finish_cancelled_or_current(self.jobs, project, run)
                self.jobs.add_project_log(project, run, "WARN", "local_worker", f"Local worker stopped: {stop.get('reason')}")
                return LocalWorkerResult(project, run, self.worker_id, str(final.get("status") or RunStatus.CANCELLED.value), result=result, should_stop_reason=str(stop.get("reason") or ""))
            final = self.jobs.complete_project_job(project, run, status=RunStatus.SUCCEEDED)
            self.jobs.add_project_log(project, run, "INFO", "local_worker", f"Local worker {self.worker_id} completed")
            return LocalWorkerResult(project, run, self.worker_id, str(final.get("status") or RunStatus.SUCCEEDED.value), result=result)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            _fail_project_job(self.jobs, project, run, worker_id=self.worker_id, error=message, traceback_text=traceback.format_exc())
            return LocalWorkerResult(project, run, self.worker_id, RunStatus.FAILED.value, error=message)


@dataclass
class LocalWorkerContext:
    jobs: JobManager
    project_id: str
    run_id: str
    worker_id: str
    lease_id: str
    lease_ttl_sec: int

    def heartbeat(self) -> dict[str, Any]:
        return record_project_worker_heartbeat(
            self.jobs,
            self.project_id,
            self.run_id,
            worker_id=self.worker_id,
            lease_id=self.lease_id,
            lease_ttl_sec=self.lease_ttl_sec,
        )

    def should_stop(self) -> dict[str, Any]:
        return project_worker_should_stop(
            self.jobs,
            self.project_id,
            self.run_id,
            worker_id=self.worker_id,
            lease_id=self.lease_id,
        )

    def log(self, level: str, source: str, message: str) -> None:
        self.jobs.add_project_log(self.project_id, self.run_id, level, source, message)


def acquire_project_worker_lease(
    jobs: JobManager,
    project_id: str,
    run_id: str,
    *,
    worker_id: str,
    lease_ttl_sec: int = 300,
    external_task_id: str = "",
    force: bool = False,
) -> dict[str, Any]:
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    worker = _clean_segment(worker_id, "worker_id")
    ttl = _positive_int(lease_ttl_sec, "lease_ttl_sec")
    job = jobs.read_project_job_state(project, run)
    if not job or str(job.get("status") or "") not in _active_statuses():
        raise KeyError(f"no active project job: {project}/{run}")
    now = _utc_now()
    lease = _lease(job)
    if lease and not _lease_expired(lease, now=now) and not force:
        current_worker = str(lease.get("worker_id") or "")
        if current_worker and current_worker != worker:
            raise ValueError(f"project job lease already held by active worker: {current_worker}")
    lease_id = f"lease-{project}-{run}-{worker}-{int(now.timestamp())}"
    expires_at = _iso(now + timedelta(seconds=ttl))
    updated = _ensure_project_worker_control(dict(job))
    updated["worker_lease"] = {
        "lease_id": lease_id,
        "worker_id": worker,
        "external_task_id": str(external_task_id or "").strip(),
        "acquired_at": _iso(now),
        "heartbeat_at": _iso(now),
        "expires_at": expires_at,
        "ttl_sec": ttl,
        "stale": False,
    }
    updated["external_task_id"] = str(external_task_id or "").strip()
    updated["heartbeat_at"] = _iso(now)
    updated["lease_expires_at"] = expires_at
    updated["updated_at"] = _iso(now)
    _append_history(updated, event="worker_lease_acquired", note=f"worker={worker}")
    _write_project_job_state(jobs, project, run, updated)
    jobs.add_project_log(project, run, "INFO", "worker_lease", f"Worker lease acquired by {worker}")
    return dict(updated)


def record_project_worker_heartbeat(
    jobs: JobManager,
    project_id: str,
    run_id: str,
    *,
    worker_id: str,
    lease_id: str = "",
    lease_ttl_sec: int = 300,
) -> dict[str, Any]:
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    worker = _clean_segment(worker_id, "worker_id")
    job = jobs.read_project_job_state(project, run)
    if not job or str(job.get("status") or "") not in _active_statuses():
        raise KeyError(f"no active project job: {project}/{run}")
    lease = _lease(job)
    if not lease:
        raise ValueError(f"project job has no worker lease: {project}/{run}")
    if str(lease.get("worker_id") or "") != worker:
        raise ValueError("heartbeat worker does not hold the lease")
    clean_lease_id = str(lease_id or "").strip()
    if clean_lease_id and clean_lease_id != str(lease.get("lease_id") or ""):
        raise ValueError("heartbeat lease_id does not match current lease")
    ttl = _positive_int(lease_ttl_sec, "lease_ttl_sec")
    now = _utc_now()
    expires_at = _iso(now + timedelta(seconds=ttl))
    updated = _ensure_project_worker_control(dict(job))
    updated["worker_lease"] = {**lease, "heartbeat_at": _iso(now), "expires_at": expires_at, "ttl_sec": ttl, "stale": False}
    updated["heartbeat_at"] = _iso(now)
    updated["lease_expires_at"] = expires_at
    updated["updated_at"] = _iso(now)
    _append_history(updated, event="worker_heartbeat", note=f"worker={worker}")
    _write_project_job_state(jobs, project, run, updated)
    jobs.add_project_log(project, run, "INFO", "worker_heartbeat", f"Heartbeat from {worker}")
    return dict(updated)


def project_worker_should_stop(
    jobs: JobManager,
    project_id: str,
    run_id: str,
    *,
    worker_id: str = "",
    lease_id: str = "",
) -> dict[str, Any]:
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    job = jobs.read_project_job_state(project, run)
    if not job or str(job.get("status") or "") not in _active_statuses():
        return {"project_id": project, "run_id": run, "should_stop": True, "reason": "job_not_active"}
    if bool(job.get("cancel_requested")):
        return {"project_id": project, "run_id": run, "should_stop": True, "reason": "cancel_requested"}
    lease = _lease(job)
    if lease:
        clean_worker = str(worker_id or "").strip()
        clean_lease_id = str(lease_id or "").strip()
        if clean_worker and clean_worker != str(lease.get("worker_id") or ""):
            return {"project_id": project, "run_id": run, "should_stop": True, "reason": "lease_lost"}
        if clean_lease_id and clean_lease_id != str(lease.get("lease_id") or ""):
            return {"project_id": project, "run_id": run, "should_stop": True, "reason": "lease_lost"}
        if _lease_expired(lease, now=_utc_now()):
            return {"project_id": project, "run_id": run, "should_stop": True, "reason": "lease_expired"}
    return {"project_id": project, "run_id": run, "should_stop": False, "reason": ""}


def _finish_cancelled_or_current(jobs: JobManager, project_id: str, run_id: str) -> dict[str, Any]:
    try:
        return jobs.complete_project_job(project_id, run_id, status=RunStatus.CANCELLED)
    except KeyError:
        current = jobs.read_project_job_state(project_id, run_id)
        if current:
            return current
        return {"project_id": project_id, "run_id": run_id, "status": RunStatus.CANCELLED.value}


def _fail_project_job(jobs: JobManager, project_id: str, run_id: str, *, worker_id: str, error: str, traceback_text: str = "") -> dict[str, Any]:
    job = jobs.read_project_job_state(project_id, run_id)
    if not job:
        raise KeyError(f"no project job: {project_id}/{run_id}")
    now = now_iso()
    updated = _ensure_project_worker_control(dict(job))
    updated["status"] = RunStatus.FAILED.value
    updated["updated_at"] = now
    updated["ended_at"] = now
    updated["error"] = {"message": error, "worker_id": worker_id, "traceback": traceback_text}
    _append_history(updated, event="worker_failed", note=error)
    _write_project_job_state(jobs, project_id, run_id, updated)
    jobs.add_project_log(project_id, run_id, "ERROR", "local_worker", error)
    return dict(updated)


def _ensure_project_worker_control(job: dict[str, Any]) -> dict[str, Any]:
    updated = dict(job)
    updated.setdefault("worker_control", {"durable": True, "supports_lease": True, "supports_heartbeat": True, "supports_cancel_request": True})
    updated.setdefault("worker_lease", {})
    updated.setdefault("external_task_id", "")
    updated.setdefault("heartbeat_at", "")
    updated.setdefault("lease_expires_at", "")
    updated.setdefault("cancel_requested", False)
    updated.setdefault("cancellation", {})
    updated["durable_worker_control"] = True
    updated["local_worker_runner"] = True
    return updated


def _write_project_job_state(jobs: JobManager, project_id: str, run_id: str, job: dict[str, Any]) -> None:
    write_json(jobs.project_run_dir(project_id, run_id) / "job_state.json", job)


def _append_history(job: dict[str, Any], *, event: str, note: str = "") -> None:
    history = job.get("history") if isinstance(job.get("history"), list) else []
    history.append({"status": str(job.get("status") or RunStatus.RUNNING.value), "updated_at": str(job.get("updated_at") or now_iso()), "event": event, "attempt": int(job.get("attempt") or 1), "note": note})
    job["history"] = history


def _active_statuses() -> set[str]:
    return {RunStatus.PENDING.value, RunStatus.RUNNING.value, RunStatus.PAUSED_BY_USER.value}


def _lease(job: dict[str, Any]) -> dict[str, Any]:
    lease = job.get("worker_lease")
    return dict(lease) if isinstance(lease, dict) and lease else {}


def _lease_expired(lease: dict[str, Any], *, now: datetime) -> bool:
    expires_at = _parse_iso(str(lease.get("expires_at") or ""))
    return bool(expires_at and expires_at <= now)


def _parse_iso(value: str) -> datetime | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean in {".", ".."} or any(ch in clean for ch in "/\\"):
        raise ValueError(f"{label} must be a single safe path segment")
    return clean


def _positive_int(value: int, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed
