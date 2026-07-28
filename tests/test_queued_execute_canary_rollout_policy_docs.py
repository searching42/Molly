from __future__ import annotations

from pathlib import Path


POLICY_DOC = Path("docs/queued-execute-canary-rollout-policy.md")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_queued_execute_canary_rollout_policy_documents_required_sections() -> None:
    text = _policy_text()

    for required in [
        "Decision Matrix",
        "Green Criteria Before Expanding Allowlist",
        "Red Conditions",
        "Allowlist Expansion Rules",
        "Exit Criteria Before Default Migration",
        "AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY",
        "sync_fallback_not_allowlisted",
        "queued_canary",
        "train_model",
        "remote worker",
        "SQLite",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_names_current_allowlist() -> None:
    text = _policy_text()

    for task_id in [
        "inspect_dataset",
        "clean_dataset",
        "check_trainability",
        "run_baseline",
        "render_report",
    ]:
        assert task_id in text

    assert "train_model remains excluded" in text
    assert "generation remains excluded" in text
    assert "literature/mining remains excluded" in text


def test_queued_execute_canary_docs_do_not_claim_default_migration() -> None:
    text = _policy_text().lower()

    for forbidden in [
        "queued execution is now default",
        "remote workers are enabled",
        "sqlite migration completed",
    ]:
        assert forbidden not in text


def test_queued_execute_canary_rollout_policy_documents_artifact_parity_fixture() -> None:
    text = _policy_text()

    for required in [
        "PR #125 adds first artifact registry parity fixture",
        "logical artifact ids",
        "file existence",
        "Exact paths/hashes are not required",
        "This does not expand the allowlist",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_documents_failure_parity_fixture() -> None:
    text = _policy_text()

    for required in [
        "PR #126 adds failure classification parity fixture",
        "failed status",
        "failed task",
        "useful error message fields",
        "Exact error strings are not required",
        "Queued executor failed dict must not be treated as success",
        "This does not expand the allowlist",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_documents_repeated_run_stability() -> None:
    text = _policy_text()

    for required in [
        "PR #127 adds repeated-run stability coverage",
        "isolate queue state by project_id/run_id",
        "stable response shape",
        "logical artifact ids",
        "Rollback to sync must not touch existing queued jobs",
        "This does not expand the allowlist",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_documents_queue_recovery_and_stale_lease_coverage() -> None:
    text = _policy_text()

    for required in [
        "PR #128 adds queue recovery and stale lease coverage",
        "Stale running jobs must not be mistaken for the target job",
        "Target-job selection must remain valid after stale lease recovery",
        "Sync fallback must not process or mutate queued jobs",
        "This does not enable remote workers or SQLite",
        "This does not expand the allowlist",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_documents_second_allowlisted_chain_parity_fixture() -> None:
    text = _policy_text()

    for required in [
        "PR #130 adds a second allowlisted chain parity fixture",
        "second allowlisted chain",
        "render_report",
        "This still does not expand the allowlist",
        "This still does not justify default migration by itself",
    ]:
        assert required in text


def test_queued_execute_canary_rollout_policy_documents_cancellation_and_retry_boundary() -> None:
    text = _policy_text()

    for required in [
        "PR #131 adds cancellation coverage",
        "Cancelled queued jobs must not be mistaken for the target job",
        "Sync fallback must not process or mutate cancelled queued jobs",
        "retry",
        "This does not expand the allowlist",
        "This does not enable remote workers or SQLite",
    ]:
        assert required in text

    assert "This still does not implement a public retry/requeue API, automatic retry," in text
    assert "route changes, allowlist expansion, or default migration." in text
    assert "This still does not implement retry/requeue operations, automatic retry, API" not in text
