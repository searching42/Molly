from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ai4s_agent._utils import now_iso
from ai4s_agent.schemas import RunStatus


_ACTIVE_JOB_STATUSES = {
    RunStatus.PENDING.value,
    RunStatus.RUNNING.value,
    RunStatus.PAUSED_BY_USER.value,
}


def install_durable_job_control() -> None:
    """Add durable worker lease, heartbeat, and cancellation control to JobManager."""

    from ai4s_agent.job_manager import JobManager

    if getattr(JobManager, "_durable_worker_control_installed", False):
        return

    original_start_job = JobManager.start_job

    def start_job_with_worker_control(self: Any, run_id: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
        job = original_start_job(self, run_id, details=details)
        job = _ensure_worker_control(job)
        self._write_job_state(run_id, job)
        return dict(job)

    JobManager.start_job = start_job_with_worker_control  # type: ignore[method-assign]
    JobManager.acquire_worker_lease = acquire_worker_lease  # type: ignore[attr-defined]
    JobManager.record_worker_heartbeat = record_worker_heartbeat  # type: ignore[attr-defined]
    JobManager.request_cancel = request_cancel  # type: ignore[attr-defined]
    JobManager.worker_should_stop = worker_should_stop  # type: ignore[attr-defined]
    JobManager.release_worker_lease = release_worker_lease  # type: ignore[attr-defined]
    JobManager.list_stale_worker_leases = list_stale_worker_leases  # type: ignore[attr-defined]
    JobManager._durable_worker_control_installed = True  # type: ignore[attr-defined]


def acquire_worker_lease(
    self: Any,
    run_id: str,
    *,
    worker_id: str,
    lease_ttl_sec: int = 300,
    external_task_id: str = "",
    force: bool = False,
) -> dict[str, Any]:
    job = _require_active_job(self, run_id)
    worker = _clean_worker_id(worker_id)
    ttl = _positive_ttl(lease_ttl_sec)
    now = _utc_now()
    lease = _lease(job)
    if lease and not _lease_expired(lease, now=now) and not force:
        current_worker = str(lease.get("worker_id") or "")
        if current_worker and current_worker != worker:
            raise ValueError(f"job lease already held by active worker: {current_worker}")
    lease_id = f"lease-{run_id}-{worker}-{int(now.timestamp())}"
    expires_at = _iso(now + timedelta(seconds=ttl))
    updated = _ensure_worker_control(dict(job))
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
    self._write_job_state(run_id, updated)
    self._emit_log(run_id, "INFO", "worker_lease", f"Worker lease acquired by {worker}")
    return dict(updated)


def record_worker_heartbeat(
    self: Any,
    run_id: str,
    *,
    worker_id: str,
    lease_id: str = "",
    lease_ttl_sec: int | None = None,
) -> dict[str, Any]:
    job = _require_active_job(self, run_id)
    worker = _clean_worker_id(worker_id)
    lease = _lease(job)
    if not lease:
        raise ValueError(f"job has no worker lease: {run_id}")
    if str(lease.get("worker_id") or "") != worker:
        raise ValueError("heartbeat worker does not hold the lease")
    clean_lease_id = str(lease_id or "").strip()
    if clean_lease_id and clean_lease_id != str(lease.get("lease_id") or ""):
        raise ValueError("heartbeat lease_id does not match current lease")
    ttl = _positive_ttl(lease_ttl_sec if lease_ttl_sec is not None else int(lease.get("ttl_sec") or 300))
    now = _utc_now()
    expires_at = _iso(now + timedelta(seconds=ttl))
    updated = _ensure_worker_control(dict(job))
    updated["worker_lease"] = {
        **lease,
        "heartbeat_at": _iso(now),
        "expires_at": expires_at,
        "ttl_sec": ttl,
        "stale": False,
    }
    updated["heartbeat_at"] = _iso(now)
    updated["lease_expires_at"] = expires_at
    updated["updated_at"] = _iso(now)
    _append_history(updated, event="worker_heartbeat", note=f"worker={worker}")
    self._write_job_state(run_id, updated)
    self._emit_log(run_id, "INFO", "worker_heartbeat", f"Heartbeat from {worker}")
    return dict(updated)


def request_cancel(self: Any, run_id: str, *, actor: str = "", reason: str = "") -> dict[str, Any]:
    job = _require_active_job(self, run_id)
    now = now_iso()
    updated = _ensure_worker_control(dict(job))
    updated["cancel_requested"] = True
    updated["cancellation"] = {
        "requested_at": now,
        "requested_by": str(actor or "").strip(),
        "reason": str(reason or "").strip(),
    }
    updated["updated_at"] = now
    _append_history(updated, event="cancel_requested", note=str(reason or "").strip())
    self._write_job_state(run_id, updated)
    self._emit_log(run_id, "WARN", "cancel_requested", f"Cancellation requested for {run_id}")
    return dict(updated)


def worker_should_stop(self: Any, run_id: str, *, worker_id: str = "", lease_id: str = "") -> dict[str, Any]:
    job = self.read_job_state(run_id)
    if not job or str(job.get("status") or "") not in _ACTIVE_JOB_STATUSES:
        return {"run_id": run_id, "should_stop": True, "reason": "job_not_active"}
    if bool(job.get("cancel_requested")):
        return {"run_id": run_id, "should_stop": True, "reason": "cancel_requested"}
    lease = _lease(job)
    if lease:
        clean_worker = str(worker_id or "").strip()
        clean_lease_id = str(lease_id or "").strip()
        if clean_worker and clean_worker != str(lease.get("worker_id") or ""):
            return {"run_id": run_id, "should_stop": True, "reason": "lease_lost"}
        if clean_lease_id and clean_lease_id != str(lease.get("lease_id") or ""):
            return {"run_id": run_id, "should_stop": True, "reason": "lease_lost"}
        if _lease_expired(lease, now=_utc_now()):
            return {"run_id": run_id, "should_stop": True, "reason": "lease_expired"}
    return {"run_id": run_id, "should_stop": False, "reason": ""}


def release_worker_lease(self: Any, run_id: str, *, worker_id: str, lease_id: str = "") -> dict[str, Any]:
    job = _require_active_job(self, run_id)
    worker = _clean_worker_id(worker_id)
    lease = _lease(job)
    if not lease:
        return dict(job)
    if str(lease.get("worker_id") or "") != worker:
        raise ValueError("worker does not hold the lease")
    clean_lease_id = str(lease_id or "").strip()
    if clean_lease_id and clean_lease_id != str(lease.get("lease_id") or ""):
        raise ValueError("lease_id does not match current lease")
    updated = _ensure_worker_control(dict(job))
    updated["worker_lease"] = {}
    updated["lease_expires_at"] = ""
    updated["updated_at"] = now_iso()
    _append_history(updated, event="worker_lease_released", note=f"worker={worker}")
    self._write_job_state(run_id, updated)
    self._emit_log(run_id, "INFO", "worker_lease", f"Worker lease released by {worker}")
    return dict(updated)


def list_stale_worker_leases(self: Any) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    now = _utc_now()
    for job in self.list_jobs():
        lease = _lease(job)
        if lease and _lease_expired(lease, now=now):
            stale.append(
                {
                    "run_id": job.get("run_id"),
                    "worker_id": lease.get("worker_id"),
                    "lease_id": lease.get("lease_id"),
                    "expires_at": lease.get("expires_at"),
                    "external_task_id": lease.get("external_task_id", ""),
                }
            )
    return stale


def _require_active_job(manager: Any, run_id: str) -> dict[str, Any]:
    job = manager.read_job_state(run_id)
    if not job or str(job.get("status") or "") not in _ACTIVE_JOB_STATUSES:
        raise KeyError(f"no active job: {run_id}")
    return _ensure_worker_control(job)


def _ensure_worker_control(job: dict[str, Any]) -> dict[str, Any]:
    updated = dict(job)
    updated.setdefault("worker_control", {"durable": True, "supports_lease": True, "supports_heartbeat": True, "supports_cancel_request": True})
    updated.setdefault("worker_lease", {})
    updated.setdefault("external_task_id", "")
    updated.setdefault("heartbeat_at", "")
    updated.setdefault("lease_expires_at", "")
    updated.setdefault("cancel_requested", False)
    updated.setdefault("cancellation", {})
    updated["durable_worker_control"] = True
    return updated


def _append_history(job: dict[str, Any], *, event: str, note: str = "") -> None:
    history = job.get("history") if isinstance(job.get("history"), list) else []
    history.append(
        {
            "status": str(job.get("status") or RunStatus.RUNNING.value),
            "updated_at": str(job.get("updated_at") or now_iso()),
            "event": event,
            "attempt": int(job.get("attempt") or 1),
            "note": note,
        }
    )
    job["history"] = history


def _lease(job: dict[str, Any]) -> dict[str, Any]:
    lease = job.get("worker_lease")
    return dict(lease) if isinstance(lease, dict) and lease else {}


def _lease_expired(lease: dict[str, Any], *, now: datetime) -> bool:
    expires_at = _parse_iso(str(lease.get("expires_at") or ""))
    if expires_at is None:
        return False
    return expires_at <= now


def _clean_worker_id(worker_id: str) -> str:
    clean = str(worker_id or "").strip()
    if not clean:
        raise ValueError("worker_id required")
    if any(ch in clean for ch in "/\\"):
        raise ValueError("worker_id must not contain path separators")
    return clean


def _positive_ttl(value: int) -> int:
    try:
        ttl = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("lease_ttl_sec must be a positive integer") from exc
    if ttl <= 0:
        raise ValueError("lease_ttl_sec must be a positive integer")
    return ttl


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
