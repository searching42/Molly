from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.worker_queue import WorkerQueue


TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_LEASE_STATUSES = {"completed", "failed", "cancelled", "stale"}


def internal_run_plan_queue_dir(workspace: str | Path, project_id: str, run_id: str) -> Path:
    project = _safe_path_component(project_id, "project_id")
    run = _safe_path_component(run_id, "run_id")
    base = (Path(workspace).expanduser().resolve() / ".ai4s_internal" / "run_plan_queues").resolve()
    queue_dir = (base / project / run).resolve()
    if queue_dir != base and not queue_dir.is_relative_to(base):
        raise ValueError("internal run-plan queue path must stay under workspace")
    return queue_dir


def read_run_plan_queue_status(queue: WorkerQueue) -> dict[str, Any]:
    jobs = queue.list_jobs()
    leases = queue.list_leases()
    counts = _counts(jobs, leases)
    return {
        "jobs": jobs,
        "leases": leases,
        "counts": counts,
        "has_active_jobs": bool(counts["queued"] or counts["running"] or counts["active_leases"]),
        "has_terminal_jobs": bool(counts["terminal_jobs"]),
        "waiting_user_jobs": _waiting_user_jobs(jobs),
    }


def recover_stale_run_plan_queue(queue: WorkerQueue, *, now: str = "") -> dict[str, Any]:
    recovered = queue.recover_stale_leases(now=now)
    return {
        "ok": True,
        "recovered_job_ids": recovered,
        "recovered_count": len(recovered),
        "error": None,
    }


def cleanup_terminal_run_plan_queue(queue: WorkerQueue, *, workspace: str | Path | None = None) -> dict[str, Any]:
    if workspace is not None:
        _ensure_internal_queue_root(Path(workspace), queue.store.root_dir)
    with queue._locked_state() as state:
        jobs = state.jobs
        leases = state.leases
        removed_job_ids = [
            str(job.get("job_id") or "")
            for job in jobs
            if str(job.get("status") or "") in TERMINAL_JOB_STATUSES
        ]
        terminal_job_ids = set(removed_job_ids)
        kept_jobs = [job for job in jobs if str(job.get("job_id") or "") not in terminal_job_ids]
        removed_lease_ids = [
            str(lease.get("lease_id") or "")
            for lease in leases
            if str(lease.get("status") or "") in TERMINAL_LEASE_STATUSES
            and str(lease.get("job_id") or "") in terminal_job_ids
        ]
        removed_lease_id_set = set(removed_lease_ids)
        kept_leases = [lease for lease in leases if str(lease.get("lease_id") or "") not in removed_lease_id_set]
        removed_jobs = len(removed_job_ids)
        removed_leases = len(removed_lease_ids)
        has_active_jobs = any(str(job.get("status") or "") in {"queued", "running"} for job in kept_jobs) or any(
            str(lease.get("status") or "") == "active" for lease in kept_leases
        )
        deleted_files = False
        if not kept_jobs and not kept_leases and (removed_jobs or removed_leases):
            queue.store.queue_path.unlink(missing_ok=True)
            queue.store.leases_path.unlink(missing_ok=True)
            deleted_files = True
        elif removed_jobs or removed_leases:
            state.queue["jobs"] = kept_jobs
            state.leases_payload["leases"] = kept_leases
            state.save()
    return {
        "ok": True,
        "removed_job_ids": removed_job_ids,
        "removed_lease_ids": removed_lease_ids,
        "removed_jobs": removed_jobs,
        "removed_leases": removed_leases,
        "deleted_files": deleted_files,
        "has_active_jobs": has_active_jobs,
        "error": None,
    }


def _counts(jobs: list[dict[str, Any]], leases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "jobs_total": len(jobs),
        "leases_total": len(leases),
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
        "active_leases": 0,
        "terminal_jobs": 0,
        "terminal_leases": 0,
        "waiting_user": 0,
    }
    for job in jobs:
        status = str(job.get("status") or "")
        if status in counts:
            counts[status] += 1
        if status in TERMINAL_JOB_STATUSES:
            counts["terminal_jobs"] += 1
        if _job_waiting_user(job):
            counts["waiting_user"] += 1
    for lease in leases:
        status = str(lease.get("status") or "")
        if status == "active":
            counts["active_leases"] += 1
        if status in TERMINAL_LEASE_STATUSES:
            counts["terminal_leases"] += 1
    return counts


def _waiting_user_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for job in jobs:
        if not _job_waiting_user(job):
            continue
        job_result = job.get("result") if isinstance(job.get("result"), dict) else {}
        result.append(
            {
                "job_id": str(job.get("job_id") or ""),
                "run_id": str(job.get("run_id") or ""),
                "status": str(job.get("status") or ""),
                "waiting_task": str(job_result.get("waiting_task") or ""),
                "required_gates": [
                    str(item).strip()
                    for item in job_result.get("required_gates", [])
                    if str(item).strip()
                ] if isinstance(job_result.get("required_gates"), list) else [],
            }
        )
    return result


def _job_waiting_user(job: dict[str, Any]) -> bool:
    result = job.get("result")
    if not isinstance(result, dict):
        return False
    return bool(result.get("waiting_user")) or str(result.get("status") or "").strip().upper() == "WAITING_USER"


def _safe_path_component(value: object, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    if clean in {".", ".."} or "/" in clean or "\\" in clean:
        raise ValueError(f"{label} must be a safe path component")
    return clean


def _ensure_internal_queue_root(workspace: Path, queue_root: Path) -> None:
    base = (workspace.expanduser().resolve() / ".ai4s_internal" / "run_plan_queues").resolve()
    resolved_root = queue_root.expanduser().resolve()
    if resolved_root != base and not resolved_root.is_relative_to(base):
        raise ValueError("queue root must stay under internal run-plan queue root")
