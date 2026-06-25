from __future__ import annotations

from pathlib import Path


POLICY_DOC = Path("docs/queued-execute-canary-rollout-policy.md")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_nightly_production_sized_lane_design_documents_required_sections() -> None:
    text = _policy_text()

    for required in [
        "Optional Nightly Production-Sized Fixture Lane Design",
        "Purpose",
        "Dataset profile",
        "Execution scope",
        "Required parity checks",
        "Runtime and CI budget",
        "Storage and observability requirements",
        "Current decision",
    ]:
        assert required in text


def test_nightly_lane_design_keeps_scope_conservative() -> None:
    text = _policy_text()

    for required in [
        "not enabled by this PR",
        "not part of the default presubmit suite",
        "default CI must remain lightweight and deterministic",
        "No allowlist expansion",
        "train_model remains excluded",
        "generation remains excluded",
        "literature/mining remains excluded",
        "Default migration remains blocked",
    ]:
        assert required in text


def test_nightly_lane_design_names_required_checks() -> None:
    text = _policy_text()

    for required in [
        "logical artifact ids match",
        "artifact files exist",
        "failure classification remains comparable",
        "old queued jobs are not consumed",
        "stale/cancelled jobs are not consumed",
        "rollback to sync remains safe",
        "telemetry markers",
    ]:
        assert required in text


def test_nightly_lane_design_blocks_false_claims() -> None:
    text = _policy_text().lower()

    for forbidden in [
        "the nightly workflow is enabled",
        "production-sized coverage is complete",
        "queued execution is production-ready",
        "queued execution is now default",
        "default migration completed",
        "train_model is allowlisted",
    ]:
        assert forbidden not in text
