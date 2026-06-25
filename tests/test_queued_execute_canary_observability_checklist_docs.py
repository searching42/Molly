from __future__ import annotations

from pathlib import Path


POLICY_DOC = Path("docs/queued-execute-canary-rollout-policy.md")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_queued_canary_observability_checklist_documents_required_sections() -> None:
    text = _policy_text()

    for required in [
        "Telemetry and Observability Checklist",
        "Required backend markers",
        "Required identity fields",
        "Required execution state fields",
        "Required safety evidence",
        "Still missing / not production-grade",
        "Current decision",
    ]:
        assert required in text


def test_queued_canary_observability_checklist_names_backend_markers() -> None:
    text = _policy_text()

    for required in [
        "RunPlan execution backend: sync",
        "RunPlan execution backend: queued_canary",
        "RunPlan execution backend: sync_fallback_not_allowlisted",
        'execution_backend="queued_canary"',
        "queue_summary",
    ]:
        assert required in text


def test_queued_canary_observability_checklist_names_identity_and_state_fields() -> None:
    text = _policy_text()

    for required in [
        "project_id",
        "run_id",
        "job_id",
        "lease_id",
        "worker_id",
        "queued_job_id",
        "final_job.status",
        "final_lease.status",
        "failed task",
        "waiting task",
        "required gates",
        "cancellation status",
        "stale lease",
    ]:
        assert required in text


def test_queued_canary_observability_checklist_blocks_production_claims() -> None:
    text = _policy_text().lower()

    for forbidden in [
        "telemetry is production-grade",
        "observability is production-ready",
        "dashboard is implemented",
        "alerting is implemented",
        "queued execution is now default",
        "default migration completed",
    ]:
        assert forbidden not in text
