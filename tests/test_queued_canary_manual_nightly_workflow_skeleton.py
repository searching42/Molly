from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/queued-canary-manual-nightly.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_manual_nightly_workflow_skeleton_exists_and_uses_manual_dispatch_only() -> None:
    assert WORKFLOW.exists()
    text = _workflow_text()

    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "\npush:" not in text
    assert "schedule:" not in text


def test_manual_nightly_workflow_skeleton_uses_conservative_inputs_and_bounded_evidence() -> None:
    text = _workflow_text()

    for required in [
        "fixture_profile:",
        "default: control_plane_small",
        "run_extended_canary:",
        "default: false",
        "evidence_retention_days:",
        'default: "14"',
        "python -m compileall -q src tests",
        "--junitxml=queued-canary-junit.xml",
        "queued-canary-pytest.log",
        "queued-canary-summary.txt",
        "actions/upload-artifact@v4",
        "tests/test_queued_execute_canary_minimal_telemetry.py",
        "tests/test_queued_execute_canary_observability_checklist_docs.py",
        "tests/test_queued_execute_canary_nightly_fixture_lane_docs.py",
        "tests/test_queued_execute_canary_production_sized_boundary.py",
    ]:
        assert required in text


def test_manual_nightly_workflow_skeleton_stays_bounded_and_does_not_expand_scope() -> None:
    text = _workflow_text().lower()

    for forbidden in [
        "tests/test_queued_execute_canary_*",
        "train_model",
        "generation",
        "literature/mining",
        "remote worker",
        "sqlite",
        "curl ",
        "wget ",
        "gh auth",
        "secrets.",
        "actions/download-artifact",
        "queued execution is now default",
        "default migration completed",
    ]:
        assert forbidden not in text
