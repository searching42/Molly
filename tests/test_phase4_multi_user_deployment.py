from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent._utils import write_json
from ai4s_agent.deployment import assess_multi_user_deployment
from ai4s_agent.storage import ProjectStorage


def _check_by_name(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_multi_user_readiness_checks_permission_memory_and_audit_boundaries(tmp_path: Path) -> None:
    report = assess_multi_user_deployment(workspace_dir=tmp_path, runs_dir=tmp_path / "runs")

    assert report.status == "ready"
    assert report.executable is False
    assert _check_by_name(report, "permission_actor_boundary").status == "pass"
    assert _check_by_name(report, "project_memory_boundary").status == "pass"
    assert _check_by_name(report, "audit_actor_boundary").status == "pass"


def test_multi_user_readiness_blocks_approved_audit_records_without_actor(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("proj-audit", "run-audit")
    write_json(
        run_dir / "gate_decisions.json",
        {
            "decisions": [
                {
                    "gate": "gate_3_train_config",
                    "approved": True,
                    "approved_at": "2026-06-05T10:00:00Z",
                }
            ]
        },
    )

    report = assess_multi_user_deployment(workspace_dir=tmp_path, runs_dir=tmp_path / "runs")
    audit_check = _check_by_name(report, "audit_actor_boundary")

    assert report.status == "blocked"
    assert audit_check.status == "fail"
    assert audit_check.evidence["missing_actor_count"] == 1
    assert "gate_decisions.json" in json.dumps(audit_check.evidence)
