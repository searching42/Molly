from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json

try:  # pragma: no cover - POSIX path is covered in normal CI.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class JsonWorkerQueueStore:
    root_dir: Path

    def __init__(self, root_dir: str | Path) -> None:
        object.__setattr__(self, "root_dir", Path(root_dir).expanduser().resolve())
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def queue_path(self) -> Path:
        return self.root_dir / "worker_queue.json"

    @property
    def leases_path(self) -> Path:
        return self.root_dir / "worker_leases.json"

    @property
    def lock_path(self) -> Path:
        return self.root_dir / ".worker_queue.lock"

    def read_queue(self) -> dict[str, Any]:
        return _read_json_object(self.queue_path, default={"jobs": []})

    def read_leases(self) -> dict[str, Any]:
        return _read_json_object(self.leases_path, default={"leases": []})

    def write_queue(self, payload: dict[str, Any]) -> Path:
        return write_json(self.queue_path, payload)

    def write_leases(self, payload: dict[str, Any]) -> Path:
        return write_json(self.leases_path, payload)


class WorkerQueue:
    """Durable control-plane queue for future supervised workers.

    This skeleton intentionally does not execute tasks. It only persists queued
    jobs and worker leases with locked JSON read-modify-write updates.
    """

    def __init__(
        self,
        store: JsonWorkerQueueStore,
        *,
        lease_ttl_sec: int = 300,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.lease_ttl_sec = _positive_int(lease_ttl_sec, "lease_ttl_sec")
        self._clock = clock or _utc_now
        self._thread_lock = threading.RLock()

    def enqueue(self, project_id: str, run_id: str, task: dict[str, Any], *, created_at: str = "") -> dict[str, Any]:
        project = _clean_segment(project_id, "project_id")
        run = _clean_segment(run_id, "run_id")
        if not isinstance(task, dict) or not task:
            raise ValueError("task must be a non-empty object")
        created = _normalize_time(created_at, fallback=self._now())
        with self._locked_state() as state:
            jobs = state.jobs
            job_id = _next_job_id(project, run, jobs)
            job = {
                "job_id": job_id,
                "project_id": project,
                "run_id": run,
                "task": {str(key): value for key, value in task.items()},
                "status": "queued",
                "created_at": created,
                "updated_at": created,
                "lease_id": "",
                "worker_id": "",
                "heartbeat_at": "",
                "cancellation_requested": False,
                "attempts": 0,
                "error": {},
            }
            jobs.append(job)
            state.save()
            return dict(job)

    def acquire(
        self,
        worker_id: str,
        *,
        now: str = "",
        target_job_id: str | None = None,
        target_project_id: str | None = None,
        target_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        worker = _clean_segment(worker_id, "worker_id")
        clean_target_job_id = _clean_optional_segment(target_job_id, "target_job_id")
        clean_target_project_id = _clean_optional_segment(target_project_id, "target_project_id")
        clean_target_run_id = _clean_optional_segment(target_run_id, "target_run_id")
        current = _normalize_time(now, fallback=self._now())
        with self._locked_state() as state:
            self._recover_stale_locked(state, now=current)
            candidates = [
                job for job in state.jobs
                if (
                    str(job.get("status") or "") == "queued"
                    and not bool(job.get("cancellation_requested"))
                    and _matches_target(
                        job,
                        target_job_id=clean_target_job_id,
                        target_project_id=clean_target_project_id,
                        target_run_id=clean_target_run_id,
                    )
                )
            ]
            if not candidates:
                state.save()
                return None
            job = sorted(candidates, key=lambda item: (str(item.get("created_at") or ""), str(item.get("job_id") or "")))[0]
            lease_id = f"lease-{uuid.uuid4().hex[:16]}"
            expires_at = _add_seconds(current, self.lease_ttl_sec)
            job.update(
                {
                    "status": "running",
                    "lease_id": lease_id,
                    "worker_id": worker,
                    "heartbeat_at": current,
                    "updated_at": current,
                    "attempts": int(job.get("attempts") or 0) + 1,
                }
            )
            state.leases.append(
                {
                    "lease_id": lease_id,
                    "job_id": str(job.get("job_id") or ""),
                    "project_id": str(job.get("project_id") or ""),
                    "run_id": str(job.get("run_id") or ""),
                    "worker_id": worker,
                    "status": "active",
                    "acquired_at": current,
                    "heartbeat_at": current,
                    "expires_at": expires_at,
                    "ttl_sec": self.lease_ttl_sec,
                }
            )
            state.save()
            return dict(job)

    def heartbeat(self, lease_id: str, *, now: str = "") -> dict[str, Any]:
        clean_lease = _clean_required(lease_id, "lease_id")
        current = _normalize_time(now, fallback=self._now())
        with self._locked_state() as state:
            lease = _find_by_id(state.leases, "lease_id", clean_lease)
            if lease is None or str(lease.get("status") or "") != "active":
                raise KeyError(f"active lease not found: {clean_lease}")
            job = _find_by_id(state.jobs, "job_id", str(lease.get("job_id") or ""))
            if job is None or str(job.get("status") or "") != "running":
                raise KeyError(f"running job not found for lease: {clean_lease}")
            expires_at = _add_seconds(current, int(lease.get("ttl_sec") or self.lease_ttl_sec))
            lease["heartbeat_at"] = current
            lease["expires_at"] = expires_at
            job["heartbeat_at"] = current
            job["updated_at"] = current
            state.save()
            return dict(job)

    def complete(self, lease_id: str, *, now: str = "", result: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._finish(lease_id, status="succeeded", lease_status="completed", now=now, result=result)

    def fail(self, lease_id: str, *, reason: str = "", now: str = "") -> dict[str, Any]:
        return self._finish(lease_id, status="failed", lease_status="failed", reason=reason, now=now)

    def cancel(self, job_id: str, *, now: str = "") -> dict[str, Any]:
        clean_job = _clean_required(job_id, "job_id")
        current = _normalize_time(now, fallback=self._now())
        with self._locked_state() as state:
            job = _find_by_id(state.jobs, "job_id", clean_job)
            if job is None:
                raise KeyError(f"job not found: {clean_job}")
            job["cancellation_requested"] = True
            if str(job.get("status") or "") == "queued":
                job["status"] = "cancelled"
            job["updated_at"] = current
            state.save()
            return dict(job)

    def recover_stale_leases(self, *, now: str = "") -> list[str]:
        current = _normalize_time(now, fallback=self._now())
        with self._locked_state() as state:
            recovered = self._recover_stale_locked(state, now=current)
            state.save()
            return recovered

    def status(self, job_id: str) -> dict[str, Any] | None:
        clean_job = _clean_required(job_id, "job_id")
        queue = self.store.read_queue()
        jobs = _records(queue, "jobs")
        job = _find_by_id(jobs, "job_id", clean_job)
        return dict(job) if job is not None else None

    def lease_status(self, lease_id: str) -> dict[str, Any] | None:
        clean_lease = _clean_required(lease_id, "lease_id")
        leases = _records(self.store.read_leases(), "leases")
        lease = _find_by_id(leases, "lease_id", clean_lease)
        return dict(lease) if lease is not None else None

    def list_leases(self) -> list[dict[str, Any]]:
        return [dict(lease) for lease in _records(self.store.read_leases(), "leases")]

    def list_jobs(self) -> list[dict[str, Any]]:
        return [dict(job) for job in _records(self.store.read_queue(), "jobs")]

    def _finish(
        self,
        lease_id: str,
        *,
        status: str,
        lease_status: str,
        reason: str = "",
        now: str = "",
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_lease = _clean_required(lease_id, "lease_id")
        current = _normalize_time(now, fallback=self._now())
        with self._locked_state() as state:
            lease = _find_by_id(state.leases, "lease_id", clean_lease)
            if lease is None or str(lease.get("status") or "") != "active":
                raise KeyError(f"active lease not found: {clean_lease}")
            job = _find_by_id(state.jobs, "job_id", str(lease.get("job_id") or ""))
            if job is None:
                raise KeyError(f"job not found for lease: {clean_lease}")
            job["status"] = status
            job["updated_at"] = current
            if result is not None:
                job["result"] = dict(result)
            if reason:
                job["error"] = {"reason": str(reason)}
            lease["status"] = lease_status
            lease["completed_at"] = current
            state.save()
            return dict(job)

    def _recover_stale_locked(self, state: "_WorkerQueueState", *, now: str) -> list[str]:
        recovered: list[str] = []
        now_dt = _parse_iso(now)
        for lease in state.leases:
            if str(lease.get("status") or "") != "active":
                continue
            expires_at = _parse_iso(str(lease.get("expires_at") or ""))
            if expires_at is None or expires_at > now_dt:
                continue
            job = _find_by_id(state.jobs, "job_id", str(lease.get("job_id") or ""))
            lease["status"] = "stale"
            lease["stale_at"] = now
            if job is not None and str(job.get("status") or "") == "running":
                job["status"] = "queued"
                job["lease_id"] = ""
                job["worker_id"] = ""
                job["heartbeat_at"] = ""
                job["updated_at"] = now
                job["stale_recovered_at"] = now
                recovered.append(str(job.get("job_id") or ""))
        return recovered

    def _locked_state(self) -> "_LockedWorkerQueueState":
        return _LockedWorkerQueueState(self)

    def _now(self) -> str:
        return _iso(self._clock())


@dataclass
class _WorkerQueueState:
    queue: dict[str, Any]
    leases_payload: dict[str, Any]
    store: JsonWorkerQueueStore

    @property
    def jobs(self) -> list[dict[str, Any]]:
        return _records(self.queue, "jobs")

    @property
    def leases(self) -> list[dict[str, Any]]:
        return _records(self.leases_payload, "leases")

    def save(self) -> None:
        self.store.write_queue({"jobs": self.jobs})
        self.store.write_leases({"leases": self.leases})


class _LockedWorkerQueueState:
    def __init__(self, queue: WorkerQueue) -> None:
        self.queue = queue
        self.lock_file: Any | None = None
        self.state: _WorkerQueueState | None = None

    def __enter__(self) -> _WorkerQueueState:
        self.queue.store.root_dir.mkdir(parents=True, exist_ok=True)
        self.queue._thread_lock.acquire()
        self.lock_file = self.queue.store.lock_path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        self.state = _WorkerQueueState(
            queue=self.queue.store.read_queue(),
            leases_payload=self.queue.store.read_leases(),
            store=self.queue.store,
        )
        return self.state

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.lock_file is not None and fcntl is not None:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            if self.lock_file is not None:
                self.lock_file.close()
        finally:
            self.queue._thread_lock.release()


def _read_json_object(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} JSON root must be an object")
    return loaded


def _records(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = payload.get(key)
    if not isinstance(records, list):
        raise ValueError(f"{key} must be a list")
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{index}] must be an object")
    return records


def _find_by_id(records: list[dict[str, Any]], field: str, value: str) -> dict[str, Any] | None:
    for record in records:
        if str(record.get(field) or "") == value:
            return record
    return None


def _next_job_id(project_id: str, run_id: str, jobs: list[dict[str, Any]]) -> str:
    prefix = f"job-{project_id}-{run_id}"
    existing = {str(job.get("job_id") or "") for job in jobs}
    if prefix not in existing:
        return prefix
    index = 2
    while f"{prefix}-{index}" in existing:
        index += 1
    return f"{prefix}-{index}"


def _clean_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean in {".", ".."} or "/" in clean or "\\" in clean or Path(clean).name != clean:
        raise ValueError(f"{label} must be a single safe path segment")
    return clean


def _clean_optional_segment(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    clean = str(value or "").strip()
    if not clean:
        return None
    return _clean_segment(clean, label)


def _matches_target(
    job: dict[str, Any],
    *,
    target_job_id: str | None,
    target_project_id: str | None,
    target_run_id: str | None,
) -> bool:
    if target_job_id is not None and str(job.get("job_id") or "") != target_job_id:
        return False
    if target_project_id is not None and str(job.get("project_id") or "") != target_project_id:
        return False
    if target_run_id is not None and str(job.get("run_id") or "") != target_run_id:
        return False
    return True


def _clean_required(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    return clean


def _positive_int(value: int, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _normalize_time(value: str, *, fallback: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return fallback
    parsed = _parse_iso(clean)
    return _iso(parsed)


def _parse_iso(value: str) -> datetime:
    clean = str(value or "").strip()
    if clean.endswith("Z"):
        clean = f"{clean[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_seconds(value: str, seconds: int) -> str:
    return _iso(_parse_iso(value) + timedelta(seconds=seconds))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
