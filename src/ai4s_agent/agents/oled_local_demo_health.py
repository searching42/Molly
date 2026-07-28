from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.agents.oled_local_demo_status import inspect_oled_local_demo_worker_jobs
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def check_oled_local_demo_worker_health(
    *,
    queue_root: Path | str,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run read-only health checks for the OLED local demo worker stack."""
    checks = [
        _queue_store_readable(queue_root),
        _entrypoints_importable(),
        _task_id_consistency(),
        _status_read_only_check(queue_root=queue_root, project_root=project_root),
    ]
    if project_root is not None:
        checks.append(_project_root_readable(project_root))
    summary = _summary(checks)
    return {
        "ok": summary["fail"] == 0,
        "queue_root": str(queue_root),
        "project_root": str(project_root) if project_root is not None else "",
        "checks": checks,
        "summary": summary,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OLED local demo worker stack health without executing jobs.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root")
    args = parser.parse_args(argv)

    result = check_oled_local_demo_worker_health(
        queue_root=args.queue_root,
        project_root=args.project_root,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _queue_store_readable(queue_root: Path | str) -> dict[str, Any]:
    try:
        queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
        jobs = queue.list_jobs()
        leases = queue.list_leases()
    except Exception as exc:
        return _check("queue_store_readable", "fail", {"error": _error_message(exc)})
    return _check("queue_store_readable", "pass", {"job_count": len(jobs), "lease_count": len(leases)})


def _entrypoints_importable() -> dict[str, Any]:
    details = {
        "cancel": _has_imported_attr("ai4s_agent.agents.oled_local_demo_cancel", "cancel_oled_local_demo_worker_job"),
        "enqueue": _has_imported_attr("ai4s_agent.agents.oled_local_demo_enqueue", "enqueue_oled_local_demo_worker_job"),
        "retry": _has_imported_attr("ai4s_agent.agents.oled_local_demo_retry", "retry_failed_oled_local_demo_worker_job"),
        "status": _has_imported_attr("ai4s_agent.agents.oled_local_demo_status", "inspect_oled_local_demo_worker_jobs"),
        "worker_loop": _source_has_attr("ai4s_agent.agents.oled_local_demo_worker_loop", "run_oled_local_demo_worker_loop"),
        "worker_runner": _source_has_attr("ai4s_agent.agents.oled_local_demo_worker", "OLEDLocalDemoRunPlanWorkerTaskRunner"),
    }
    return _check("entrypoints_importable", "pass" if all(details.values()) else "fail", details)


def _task_id_consistency() -> dict[str, Any]:
    details: dict[str, Any] = {"expected": TASK_ID}
    for label, module_name in (
        ("cancel", "ai4s_agent.agents.oled_local_demo_cancel"),
        ("enqueue", "ai4s_agent.agents.oled_local_demo_enqueue"),
        ("retry", "ai4s_agent.agents.oled_local_demo_retry"),
        ("status", "ai4s_agent.agents.oled_local_demo_status"),
    ):
        module = importlib.import_module(module_name)
        details[label] = str(getattr(module, "TASK_ID", ""))
    details["worker_runner"] = _source_contains("ai4s_agent.agents.oled_local_demo_worker", TASK_ID)
    consistent = all(value == TASK_ID for key, value in details.items() if key not in {"expected", "worker_runner"})
    consistent = consistent and bool(details["worker_runner"])
    return _check("task_id_consistency", "pass" if consistent else "fail", details if not consistent else {"expected": TASK_ID})


def _status_read_only_check(queue_root: Path | str, project_root: Path | str | None) -> dict[str, Any]:
    try:
        status = inspect_oled_local_demo_worker_jobs(queue_root=queue_root, project_root=project_root)
    except Exception as exc:
        return _check("status_read_only_check", "fail", {"error": _error_message(exc)})
    details = {
        "executed": bool(status.get("executed")),
        "executable": bool(status.get("executable")),
        "job_count": int(status.get("job_count") or 0),
        "status_counts": status.get("status_counts") if isinstance(status.get("status_counts"), dict) else {},
    }
    passes = details["executed"] is False and details["executable"] is False
    return _check("status_read_only_check", "pass" if passes else "fail", details)


def _project_root_readable(project_root: Path | str) -> dict[str, Any]:
    root = Path(project_root)
    exists = root.exists()
    details = {
        "exists": exists,
        "projects_dir_exists": (root / "projects").exists(),
        "project_root": str(project_root),
    }
    return _check("project_root_readable", "pass" if exists else "warn", details)


def _has_imported_attr(module_name: str, attr_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return callable(getattr(module, attr_name, None))


def _source_has_attr(module_name: str, attr_name: str) -> bool:
    source = _module_source(module_name)
    return bool(source and attr_name in source)


def _source_contains(module_name: str, token: str) -> bool:
    source = _module_source(module_name)
    return bool(source and token in source)


def _module_source(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return ""
    path = Path(spec.origin)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _check(name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _summary(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"fail": 0, "pass": 0, "warn": 0}
    for item in checks:
        status = str(item.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
