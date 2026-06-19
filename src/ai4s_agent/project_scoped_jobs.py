from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import BackgroundJobBudget, BackgroundJobState, RunStatus


_ACTIVE_JOB_STATUSES = {
    RunStatus.PENDING.value,
    RunStatus.RUNNING.value,
    RunStatus.PAUSED_BY_USER.value,
}


def install_project_scoped_jobs() -> None:
    from ai4s_agent.job_manager import JobManager

    if getattr(JobManager, "_project_scoped_jobs_installed", False):
        return
    JobManager.project_run_dir = project_run_dir  # type: ignore[attr-defined]
    JobManager.start_project_job = start_project_job  # type: ignore[attr-defined]
    JobManager.read_project_job_state = read_project_job_state  # type: ignore[attr-defined]
    JobManager.get_project_job = get_project_job  # type: ignore[attr-defined]
    JobManager.list_project_jobs = list_project_jobs  # type: ignore[attr-defined]
    JobManager.complete_project_job = complete_project_job  # type: ignore[attr-defined]
    JobManager.start_project_background_job = start_project_background_job  # type: ignore[attr-defined]
    JobManager.get_project_background_job = get_project_background_job  # type: ignore[attr-defined]
    JobManager._project_scoped_jobs_installed = True  # type: ignore[attr-defined]


def project_run_dir(self: Any, project_id: str, run_id: str) -> Path:
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    base = (self.runs_dir / "projects" / project / "runs").resolve()
    path = (base / run).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_project_job(self: Any, project_id: str, run_id: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = self.read_project_job_state(project_id, run_id)
    if existing and str(existing.get("status") or "") in _ACTIVE_JOB_STATUSES:
        raise ValueError(f"job already active: {project_id}/{run_id}")
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    now = now_iso()
    attempt = int(existing.get("attempt") or 0) + 1 if existing else 1
    history = _job_history(existing)
    history.append({"status": RunStatus.RUNNING.value, "updated_at": now, "event": "started" if attempt == 1 else "restarted", "attempt": attempt})
    job = {
        "schema_version": 2,
        "project_id": project,
        "run_id": run,
        "job_key": {"project_id": project, "run_id": run},
        "status": RunStatus.RUNNING.value,
        "attempt": attempt,
        "created_at": str(existing.get("created_at") or now) if existing else now,
        "started_at": now,
        "updated_at": now,
        "details": details or {},
        "history": history,
        "durable_state": True,
        "project_scoped": True,
        "executable": False,
    }
    _write_project_job_state(self, project_id, run_id, job)
    _append_project_log(self, project_id, run_id, "INFO", "job_started", f"Job {project}/{run} started")
    return dict(job)


def read_project_job_state(self: Any, project_id: str, run_id: str) -> dict[str, Any] | None:
    path = self.project_run_dir(project_id, run_id) / "job_state.json"
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    if not isinstance(loaded, dict):
        return None
    if str(loaded.get("project_id") or "") != project or str(loaded.get("run_id") or "") != run:
        return None
    return {str(key): value for key, value in loaded.items()}


def get_project_job(self: Any, project_id: str, run_id: str) -> dict[str, Any] | None:
    job = self.read_project_job_state(project_id, run_id)
    if not job or str(job.get("status") or "") not in _ACTIVE_JOB_STATUSES:
        return None
    return dict(job)


def list_project_jobs(self: Any, project_id: str | None = None) -> list[dict[str, Any]]:
    root = self.runs_dir / "projects"
    if not root.exists():
        return []
    projects = [_clean_segment(project_id, "project_id")] if project_id else [child.name for child in sorted(root.iterdir()) if child.is_dir()]
    jobs: list[dict[str, Any]] = []
    for project in projects:
        runs_base = root / project / "runs"
        if not runs_base.exists():
            continue
        for run_dir in sorted(runs_base.iterdir()):
            if not run_dir.is_dir():
                continue
            job = self.read_project_job_state(project, run_dir.name)
            if job and str(job.get("status") or "") in _ACTIVE_JOB_STATUSES:
                jobs.append(job)
    return jobs


def complete_project_job(self: Any, project_id: str, run_id: str, status: RunStatus = RunStatus.SUCCEEDED) -> dict[str, Any]:
    job = self.read_project_job_state(project_id, run_id)
    if not job or str(job.get("status") or "") not in _ACTIVE_JOB_STATUSES:
        raise KeyError(f"no active job: {project_id}/{run_id}")
    now = now_iso()
    updated = dict(job)
    updated["status"] = status.value
    updated["updated_at"] = now
    history = _job_history(updated)
    history.append({"status": status.value, "updated_at": now, "event": "completed", "attempt": int(updated.get("attempt") or 1)})
    updated["history"] = history
    _write_project_job_state(self, project_id, run_id, updated)
    return updated


def start_project_background_job(self: Any, project_id: str, run_id: str, *, task_id: str, budget: BackgroundJobBudget | dict[str, Any] | None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if budget is None:
        raise ValueError("background job budget required")
    validated_budget = BackgroundJobBudget.model_validate(budget.model_dump(mode="json") if isinstance(budget, BackgroundJobBudget) else budget)
    existing = self.get_project_background_job(project_id, run_id)
    if existing and existing.get("status") in _ACTIVE_JOB_STATUSES:
        raise ValueError(f"background job already active: {project_id}/{run_id}")
    project = _clean_segment(project_id, "project_id")
    run = _clean_segment(run_id, "run_id")
    now = now_iso()
    state = BackgroundJobState(
        job_id=f"bg-{project}-{run}",
        project_id=project,
        run_id=run,
        task_id=str(task_id or "").strip(),
        status=RunStatus.RUNNING,
        created_at=now,
        updated_at=now,
        budget=validated_budget,
        details={**(details or {}), "job_key": {"project_id": project, "run_id": run}},
        executable=False,
    )
    write_json(self.project_run_dir(project_id, run_id) / "background_job_state.json", state.model_dump(mode="json"))
    return state.model_dump(mode="json")


def get_project_background_job(self: Any, project_id: str, run_id: str) -> dict[str, Any] | None:
    path = self.project_run_dir(project_id, run_id) / "background_job_state.json"
    if not path.exists():
        return None
    try:
        state = BackgroundJobState.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    return state.model_dump(mode="json")


def _write_project_job_state(self: Any, project_id: str, run_id: str, job: dict[str, Any]) -> Path:
    return write_json(self.project_run_dir(project_id, run_id) / "job_state.json", job)


def _append_project_log(self: Any, project_id: str, run_id: str, level: str, source: str, message: str) -> None:
    entry = {"ts": now_iso(), "level": level, "source": source, "message": message}
    log_path = self.project_run_dir(project_id, run_id) / "job_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _job_history(job: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not job or not isinstance(job.get("history"), list):
        return []
    return [dict(item) for item in job["history"] if isinstance(item, dict)]


def _clean_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or Path(clean).name != clean:
        raise ValueError(f"{label} must be a single path segment")
    return clean
