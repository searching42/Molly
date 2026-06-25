from __future__ import annotations

from pathlib import Path


POLICY_DOC = Path("docs/queued-execute-canary-rollout-policy.md")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_default_migration_readiness_checklist_documents_green_and_blocking_items() -> None:
    text = _policy_text()

    for required in [
        "Default Migration Readiness Checklist",
        "Current Green Coverage",
        "Still Blocking Default Migration",
        "Required Before Default Migration",
        "Current decision: do not make queued execution default",
        "AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY",
        "train_model",
        "remote worker",
        "SQLite",
        "/api/run-plan/resume",
    ]:
        assert required in text


def test_default_migration_readiness_checklist_names_existing_coverage() -> None:
    text = _policy_text()

    for required in [
        "artifact registry parity",
        "second allowlisted chain parity",
        "failure classification parity",
        "repeated-run stability",
        "cancellation coverage",
        "stale lease",
        "queue recovery",
        "target-job safety",
        "sync fallback",
        "retry",
    ]:
        assert required in text


def test_default_migration_readiness_checklist_blocks_default_migration_claims() -> None:
    text = _policy_text().lower()

    for forbidden in [
        "queued execution is now default",
        "default migration completed",
        "remote workers are enabled",
        "sqlite migration completed",
        "train_model is allowlisted",
    ]:
        assert forbidden not in text
