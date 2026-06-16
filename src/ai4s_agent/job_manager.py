from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import BackgroundJobBudget, BackgroundJobCheckpoint, BackgroundJobState, RunStatus


class JobManager:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir.resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, dict[str, Any]] = {}
        self._logs: dict[str, list[dict[str, str]]] = defaultdict(list)

    def start_job(self, run_id: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
        self._safe_run_dir(run_id)
        if run_id in self._active:
            raise ValueError(f"job already active: {run_id}")
        now = now_iso()
        job = {
            "run_id": run_id,
            "status": RunStatus.RUNNING.value,
            "started_at": now,
            "updated_at": now,
            "details": details or {},
        }
        self._active[run_id] = job
        self._emit_log(run_id, "INFO", "job_started", f"Job {run_id} started")
        return dict(job)

    def pause_job(self, run_id: str) -> dict[str, Any]:
        job = self._require_active(run_id)
        job["status"] = RunStatus.PAUSED_BY_USER.value
        job["updated_at"] = now_iso()
        self._emit_log(run_id, "INFO", "job_paused", f"Job {run_id} paused by user")
        return dict(job)

    def resume_job(self, run_id: str) -> dict[str, Any]:
        job = self._require_active(run_id)
        job["status"] = RunStatus.RUNNING.value
        job["updated_at"] = now_iso()
        self._emit_log(run_id, "INFO", "job_resumed", f"Job {run_id} resumed")
        return dict(job)

    def stop_job(self, run_id: str) -> dict[str, Any]:
        job = self._require_active(run_id)
        job["status"] = RunStatus.CANCELLED.value
        job["updated_at"] = now_iso()
        self._emit_log(run_id, "INFO", "job_cancelled", f"Job {run_id} cancelled")
        self.save_job_log(run_id)
        del self._active[run_id]
        return dict(job)

    def complete_job(self, run_id: str, status: RunStatus = RunStatus.SUCCEEDED) -> dict[str, Any]:
        job = self._require_active(run_id)
        job["status"] = status.value
        job["updated_at"] = now_iso()
        self._emit_log(run_id, "INFO", "job_completed", f"Job {run_id} completed: {status.value}")
        self.save_job_log(run_id)
        self._active.pop(run_id, None)
        return dict(job)

    def get_job(self, run_id: str) -> dict[str, Any] | None:
        return dict(self._active.get(run_id, {}))

    def list_jobs(self) -> list[dict[str, Any]]:
        return [dict(job) for job in self._active.values()]

    def start_background_job(
        self,
        run_id: str,
        *,
        project_id: str = "",
        task_id: str,
        budget: BackgroundJobBudget | dict[str, Any] | None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._safe_run_dir(run_id)
        if budget is None:
            raise ValueError("background job budget required")
        validated_budget = BackgroundJobBudget.model_validate(
            budget.model_dump(mode="json") if isinstance(budget, BackgroundJobBudget) else budget
        )
        existing = self.get_background_job(run_id)
        if existing and existing.get("status") in {
            RunStatus.PENDING.value,
            RunStatus.RUNNING.value,
            RunStatus.PAUSED_BY_USER.value,
        }:
            raise ValueError(f"background job already active: {run_id}")
        now = now_iso()
        state = BackgroundJobState(
            job_id=f"bg-{run_id}",
            project_id=str(project_id or "").strip(),
            run_id=run_id,
            task_id=str(task_id or "").strip(),
            status=RunStatus.RUNNING,
            created_at=now,
            updated_at=now,
            budget=validated_budget,
            details=details or {},
            executable=False,
        )
        self._write_background_job_state(state)
        self._emit_log(run_id, "INFO", "background_job", f"Background job {run_id} registered")
        return state.model_dump(mode="json")

    def get_background_job(self, run_id: str) -> dict[str, Any] | None:
        path = self._background_job_path(run_id)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            state = BackgroundJobState.model_validate(loaded)
        except Exception:
            return None
        return state.model_dump(mode="json")

    def record_background_checkpoint(
        self,
        run_id: str,
        *,
        stage: str,
        cursor: dict[str, Any] | None = None,
        completed_units: int = 0,
        runtime_sec: int = 0,
        cost_usd: float = 0.0,
        artifact_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        state = self._read_background_job_state(run_id)
        checkpoint = BackgroundJobCheckpoint(
            checkpoint_id=f"ckpt-{run_id}-{len(state.checkpoints) + 1:03d}",
            stage=stage,
            cursor=cursor or {},
            completed_units=completed_units,
            runtime_sec=runtime_sec,
            cost_usd=cost_usd,
            artifact_refs=artifact_refs or [],
        )
        state.checkpoints.append(checkpoint)
        state.resume_from_checkpoint_id = checkpoint.checkpoint_id
        state.consumed_steps = max(state.consumed_steps, len(state.checkpoints))
        state.consumed_records = max(state.consumed_records, checkpoint.completed_units)
        state.consumed_runtime_sec = max(state.consumed_runtime_sec, checkpoint.runtime_sec)
        state.consumed_cost_usd = max(state.consumed_cost_usd, checkpoint.cost_usd)
        state.budget_exhausted = self._is_background_budget_exhausted(state)
        state.updated_at = now_iso()
        self._write_background_job_state(state)
        self._emit_log(run_id, "INFO", "background_checkpoint", f"Checkpoint recorded: {checkpoint.checkpoint_id}")
        return checkpoint.model_dump(mode="json")

    def background_resume_plan(self, run_id: str) -> dict[str, Any]:
        state = self._read_background_job_state(run_id)
        latest = state.checkpoints[-1] if state.checkpoints else None
        return {
            "run_id": state.run_id,
            "project_id": state.project_id,
            "task_id": state.task_id,
            "status": state.status.value,
            "resumable": bool(latest and state.resumable and not state.budget_exhausted),
            "resume_from_checkpoint_id": latest.checkpoint_id if latest else "",
            "latest_checkpoint": latest.model_dump(mode="json") if latest else None,
            "budget": state.budget.model_dump(mode="json"),
            "consumed": {
                "runtime_sec": state.consumed_runtime_sec,
                "steps": state.consumed_steps,
                "records": state.consumed_records,
                "cost_usd": state.consumed_cost_usd,
            },
            "budget_exhausted": state.budget_exhausted,
            "requires_confirmation": True,
            "executable": False,
        }

    def get_logs(self, run_id: str, *, limit: int = 50) -> list[dict[str, str]]:
        all_entries = self._read_log_from_disk(run_id)
        return all_entries[-limit:] if limit > 0 else all_entries

    def add_log(self, run_id: str, level: str, source: str, message: str) -> None:
        self._emit_log(run_id, level, source, message)

    def save_job_log(self, run_id: str) -> Path:
        log_path = self._safe_run_dir(run_id) / "job_log.jsonl"
        existing = self._read_log_from_disk(run_id)
        new_entries = [
            entry for entry in self._logs.get(run_id, [])
            if entry not in existing
        ]
        all_entries = existing + new_entries
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._logs.pop(run_id, None)
        return log_path

    def _read_background_job_state(self, run_id: str) -> BackgroundJobState:
        path = self._background_job_path(run_id)
        if not path.exists():
            raise KeyError(f"no background job: {run_id}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return BackgroundJobState.model_validate(loaded)

    def _write_background_job_state(self, state: BackgroundJobState) -> Path:
        return write_json(self._background_job_path(state.run_id), state.model_dump(mode="json"))

    def _require_active(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._active:
            raise KeyError(f"no active job: {run_id}")
        return self._active[run_id]

    def _emit_log(self, run_id: str, level: str, source: str, message: str) -> None:
        entry = {"ts": now_iso(), "level": level, "source": source, "message": message}
        self._logs[run_id].append(entry)
        log_path = self._safe_run_dir(run_id) / "job_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _safe_run_dir(self, run_id: str) -> Path:
        run_path = (self.runs_dir / run_id).resolve()
        if not run_path.is_relative_to(self.runs_dir):
            raise ValueError("run_id escapes runs_dir")
        return run_path

    def _background_job_path(self, run_id: str) -> Path:
        return self._safe_run_dir(run_id) / "background_job_state.json"

    @staticmethod
    def _is_background_budget_exhausted(state: BackgroundJobState) -> bool:
        budget = state.budget
        limits = (
            (budget.max_runtime_sec, state.consumed_runtime_sec),
            (budget.max_steps, state.consumed_steps),
            (budget.max_records, state.consumed_records),
            (budget.max_cost_usd, state.consumed_cost_usd),
        )
        return any(limit is not None and consumed >= limit for limit, consumed in limits)

    def _read_log_from_disk(self, run_id: str) -> list[dict[str, str]]:
        log_path = self._safe_run_dir(run_id) / "job_log.jsonl"
        if not log_path.exists():
            return []
        entries: list[dict[str, str]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                import json as _json
                entry = _json.loads(line)
                if isinstance(entry, dict):
                    entries.append(
                        {
                            "ts": str(entry.get("ts", "")),
                            "level": str(entry.get("level", "")),
                            "source": str(entry.get("source", "")),
                            "message": str(entry.get("message", "")),
                        }
                    )
            except Exception:
                pass
        return entries
