from __future__ import annotations

from pathlib import Path


RUNBOOK = Path("docs/queued-canary-operational-rollback-runbook.md")


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_operational_rollback_runbook_exists_and_documents_required_steps() -> None:
    assert RUNBOOK.exists()
    text = _text().lower()

    for required in [
        "ai4s_enable_run_plan_execute_queued_canary",
        "rollback owner",
        "rollback trigger conditions",
        "set ai4s_enable_run_plan_execute_queued_canary=false",
        "sync-compatible response",
        "execution_backend",
        "queue_summary",
        "existing jobs and leases remain unchanged",
        "do not delete queue files as part of rollback.",
        "do not modify `worker_queue.json` manually.",
        "do not cancel all queued jobs.",
        "do not retry failed jobs automatically.",
        "re-enable",
        "continued sync-only operation",
        "/api/run-plan/resume",
        "remote workers",
        "sqlite",
        "default migration remains blocked",
    ]:
        assert required in text


def test_operational_rollback_runbook_forbids_destructive_or_expansive_steps() -> None:
    text = _text().lower()

    for forbidden in [
        "rollback deletes existing queue jobs",
        "rollback retries failed jobs",
        "rollback cancels all active work",
        "rollback proves production readiness",
        "default migration is complete",
    ]:
        assert forbidden not in text
