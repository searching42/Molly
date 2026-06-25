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
