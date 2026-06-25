from __future__ import annotations

from pathlib import Path


POLICY_DOC = Path("docs/queued-execute-canary-rollout-policy.md")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_production_sized_fixture_boundary_is_documented() -> None:
    text = _policy_text()

    for required in [
        "Production-Sized Fixture Boundary",
        "small deterministic datasets",
        "not enough for production-sized scientific workload confidence",
        "runtime budget",
        "CI constraints",
        "nightly",
        "artifact file existence",
        "target-job safety",
        "flag off returns sync-compatible response",
        "Default migration remains blocked",
    ]:
        assert required in text


def test_production_sized_boundary_does_not_claim_production_ready() -> None:
    text = _policy_text().lower()

    for forbidden in [
        "is production-ready",
        "production-ready queued execution",
        "production migration completed",
        "queued execution is now default",
        "default migration completed",
        "full production workload is covered",
    ]:
        assert forbidden not in text
