from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent._utils import write_json
from ai4s_agent.storage_consistency import check_workspace_storage, main

INVALID_JSON_TEXT = "not" + "_json"


def _issue_codes(report: object) -> set[str]:
    return {issue.code for issue in report.errors + report.warnings}


def test_storage_consistency_reports_clean_minimal_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "projects" / "proj-a" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "artifact_registry.json",
        {"artifacts": {"report": "reports/report.json"}},
    )
    write_json(run_dir / "reports" / "report.json", {"ok": True})
    write_json(
        run_dir / "gate_decisions.json",
        {
            "run_id": "run-1",
            "decisions": [
                {
                    "gate": "gate_1_task_parse",
                    "approved": True,
                    "actor": "alice",
                    "approved_at": "2026-01-01T00:00:00Z",
                }
            ],
        },
    )

    report = check_workspace_storage(workspace)

    assert report.ok is True
    assert report.errors == []
    assert str(run_dir / "artifact_registry.json") in report.checked_files
    assert report.summary["checked_files"] >= 2


def test_storage_consistency_reports_parse_and_reference_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "projects" / "proj-a" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage.json").write_text(INVALID_JSON_TEXT, encoding="utf-8")
    write_json(
        run_dir / "artifact_registry.json",
        {"artifacts": {"missing": "reports/missing.json", "escape": "../outside.json"}},
    )

    report = check_workspace_storage(workspace)

    assert report.ok is False
    codes = _issue_codes(report)
    assert "invalid_json" in codes
    assert "artifact_missing" in codes
    assert "artifact_path_escape" in codes


def test_storage_consistency_checks_gate_permission_and_audit_semantics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "projects" / "proj-a" / "runs" / "run-1"
    perm_dir = workspace / "projects" / "proj-a" / "permissions"
    run_dir.mkdir(parents=True)
    perm_dir.mkdir(parents=True)
    write_json(
        run_dir / "gate_decisions.json",
        {"run_id": "run-1", "decisions": [{"gate": "gate_2_data_mining", "approved": True}]},
    )
    write_json(
        perm_dir / "permission_grants.json",
        {
            "project_id": "proj-a",
            "grants": [
                {
                    "grant_id": "grant-upload-1",
                    "project_id": "proj-a",
                    "action": "upload_dataset",
                    "actor": "alice",
                    "active": False,
                    "expires_at": "not-a-date",
                }
            ],
        },
    )
    audit = perm_dir / "permission_audit.jsonl"
    audit.write_text('{"action": "upload_dataset"}\n' + INVALID_JSON_TEXT + "\n", encoding="utf-8")

    report = check_workspace_storage(workspace)

    assert report.ok is False
    codes = _issue_codes(report)
    assert "gate_decision_missing_field" in codes
    assert "permission_grant_invalid_expires_at" in codes
    assert "permission_grant_revoked_missing_field" in codes
    assert "permission_audit_missing_field" in codes
    assert "invalid_jsonl" in codes


def test_storage_consistency_rejects_permission_grant_expires_at_without_timezone(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    perm_dir = workspace / "projects" / "proj-a" / "permissions"
    perm_dir.mkdir(parents=True)
    write_json(
        perm_dir / "permission_grants.json",
        {
            "project_id": "proj-a",
            "grants": [
                {
                    "grant_id": "grant-upload-1",
                    "project_id": "proj-a",
                    "action": "upload_dataset",
                    "actor": "alice",
                    "active": True,
                    "expires_at": "2026-01-01T00:00:00",
                }
            ],
        },
    )

    report = check_workspace_storage(workspace)

    assert report.ok is False
    codes = _issue_codes(report)
    assert "permission_grant_invalid_expires_at" in codes


def test_storage_consistency_checks_memory_stage_job_and_background_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "projects" / "proj-a" / "runs" / "run-1"
    memory_dir = workspace / "memory" / "proj-a"
    run_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    write_json(run_dir / "stage.json", {"stage": "", "status": "not-a-status"})
    write_json(run_dir / "job_state.json", {"run_id": "other-run", "status": "unknown"})
    write_json(
        run_dir / "background_job_state.json",
        {"job_id": "", "run_id": "run-1", "task_id": "", "budget": {}},
    )
    write_json(
        memory_dir / "project_memory_records.json",
        {"project_id": "proj-a", "records": [{"record_id": "", "category": ""}]},
    )
    write_json(
        memory_dir / "memory_manifest.json",
        {"project_id": "proj-a", "artifacts": [{"run_id": "run-1", "artifact_path": str(run_dir / "missing.json")}]},
    )

    report = check_workspace_storage(workspace)

    assert report.ok is False
    codes = _issue_codes(report)
    assert "invalid_stage_state" in codes
    assert "job_state_run_mismatch" in codes
    assert "job_state_invalid_status" in codes
    assert "invalid_background_job_state" in codes
    assert "invalid_project_memory_record" in codes
    assert "memory_artifact_missing" in codes


def test_storage_consistency_report_is_json_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "projects").mkdir(parents=True)

    payload = check_workspace_storage(workspace).to_dict()

    assert payload["ok"] is True
    assert payload["summary"]["checked_files"] == 0
    json.dumps(payload)


def test_storage_consistency_cli_outputs_json_report(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "projects" / "proj-a" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage.json").write_text(INVALID_JSON_TEXT, encoding="utf-8")

    exit_code = main([str(workspace)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "invalid_json"
