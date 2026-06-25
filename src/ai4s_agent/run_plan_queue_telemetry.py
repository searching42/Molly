from __future__ import annotations

import json
from typing import Any


def build_run_plan_queued_canary_telemetry(
    *,
    project_id: str,
    run_id: str,
    queue_summary: dict[str, Any],
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_job = queue_summary.get("final_job") if isinstance(queue_summary.get("final_job"), dict) else {}
    final_lease = queue_summary.get("final_lease") if isinstance(queue_summary.get("final_lease"), dict) else {}
    execution_dict = execution if isinstance(execution, dict) else {}
    execution_error = execution_dict.get("error") if isinstance(execution_dict.get("error"), dict) else {}
    summary_error = queue_summary.get("error") if isinstance(queue_summary.get("error"), dict) else {}
    final_job_error = final_job.get("error") if isinstance(final_job.get("error"), dict) else {}
    raw_required_gates = queue_summary.get("required_gates")
    raw_loop_results = queue_summary.get("loop_results")

    error_type = (
        str(summary_error.get("type") or "").strip()
        or str(execution_error.get("type") or "").strip()
        or str(execution_error.get("code") or "").strip()
    )
    error_message_present = any(
        str(value or "").strip()
        for value in (
            summary_error.get("message"),
            execution_error.get("message"),
            execution_dict.get("message"),
            final_job_error.get("reason"),
        )
    )

    return {
        "project_id": str(project_id or "").strip(),
        "run_id": str(run_id or "").strip(),
        "execution_backend": "queued_canary",
        "queued_job_id": str(queue_summary.get("queued_job_id") or ""),
        "job_id": str(final_job.get("job_id") or ""),
        "lease_id": str(final_lease.get("lease_id") or final_job.get("lease_id") or ""),
        "worker_id": str(final_lease.get("worker_id") or ""),
        "ok": bool(queue_summary.get("ok")),
        "terminal": bool(queue_summary.get("terminal")),
        "final_job_status": str(final_job.get("status") or ""),
        "final_lease_status": str(final_lease.get("status") or ""),
        "waiting_user": bool(queue_summary.get("waiting_user")),
        "waiting_task": str(queue_summary.get("waiting_task") or "").strip(),
        "required_gates": [str(item).strip() for item in raw_required_gates if str(item).strip()] if isinstance(raw_required_gates, list) else [],
        "failed_task": str(execution_dict.get("failed_task") or "").strip(),
        "error_type": error_type,
        "error_message_present": error_message_present,
        "loop_results": [str(item) for item in raw_loop_results] if isinstance(raw_loop_results, list) else [],
    }


def format_run_plan_queued_canary_telemetry_log(telemetry: dict[str, Any]) -> str:
    return f"RunPlan queued canary telemetry: {json.dumps(telemetry, sort_keys=True, separators=(',', ':'))}"
