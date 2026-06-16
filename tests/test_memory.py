from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

import ai4s_agent.memory as memory_module
from ai4s_agent._utils import write_json
from ai4s_agent.memory import (
    PermissionLevel,
    PermissionPolicy,
    ProjectMemory,
    ProjectMemoryManifest,
    RunMemoryEntry,
)
from ai4s_agent.schemas import ProjectMemoryRecord


# ---- ProjectMemory tests ----

def test_collect_run_artifacts_from_registry(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)

    # write a fake artifact registry
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"clean": "02_data/cleaned.csv", "report": "05_report/report.md"}})
    )
    # create one artifact file so path exists
    (run_dir / "02_data").mkdir(parents=True)
    (run_dir / "02_data" / "cleaned.csv").write_text("ok")

    entries = pm.collect_run_artifacts(project_id, run_id)
    assert len(entries) == 2

    clean_entry = next(e for e in entries if e.artifact_id == "clean")
    assert clean_entry.run_id == run_id
    assert clean_entry.stage == "clean"
    assert clean_entry.artifact_path.endswith("cleaned.csv")

    report_entry = next(e for e in entries if e.artifact_id == "report")
    assert report_entry.artifact_path == ""  # file doesn't exist


def test_collect_run_artifacts_raises_on_missing_run(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        pm.collect_run_artifacts("proj-1", "nonexistent")


def test_project_memory_rejects_project_and_run_path_traversal(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    with pytest.raises(ValueError):
        pm.collect_run_artifacts("../escape", "run-1")
    with pytest.raises(ValueError):
        pm.collect_run_artifacts("proj-1", "../escape")
    with pytest.raises(ValueError):
        pm.get_project_memory("../escape")
    with pytest.raises(ValueError):
        pm.confirm_memory_entry("../escape", "clean", run_id="run-1")
    assert not (tmp_path.parent / "escape" / "memory_manifest.json").exists()


def test_collect_run_artifacts_writes_memory_json(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"clean": "02_data/cleaned.csv"}})
    )
    (run_dir / "02_data").mkdir(parents=True)
    (run_dir / "02_data" / "cleaned.csv").write_text("ok")

    pm.collect_run_artifacts(project_id, run_id)
    memory_path = pm.memory_dir / project_id / f"{run_id}_memory.json"
    assert memory_path.exists()
    data = json.loads(memory_path.read_text())
    assert data["run_id"] == run_id
    assert len(data["entries"]) == 1


def test_collect_run_artifacts_filter_by_ids(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"clean": "02_data/cleaned.csv", "report": "05_report/report.md"}})
    )
    (run_dir / "02_data").mkdir(parents=True)
    (run_dir / "02_data" / "cleaned.csv").write_text("ok")
    (run_dir / "05_report").mkdir(parents=True)
    (run_dir / "05_report" / "report.md").write_text("ok")

    entries = pm.collect_run_artifacts(project_id, run_id, artifact_ids=["clean"])
    assert len(entries) == 1
    assert entries[0].artifact_id == "clean"


def test_get_project_memory_returns_entries(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"clean": "02_data/cleaned.csv"}})
    )
    (run_dir / "02_data").mkdir(parents=True)
    (run_dir / "02_data" / "cleaned.csv").write_text("ok")
    pm.collect_run_artifacts(project_id, run_id)

    entries = pm.get_project_memory(project_id)
    assert len(entries) == 1
    assert entries[0].run_id == run_id


def test_get_project_memory_returns_empty_for_unknown_project(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    assert pm.get_project_memory("no-such-project") == []


def test_confirm_memory_entry_updates_confirmed_flag(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"clean": "02_data/cleaned.csv"}})
    )
    (run_dir / "02_data").mkdir(parents=True)
    (run_dir / "02_data" / "cleaned.csv").write_text("ok")
    pm.collect_run_artifacts(project_id, run_id)

    result = pm.confirm_memory_entry(project_id, "clean", run_id=run_id)
    assert result is True

    entries = pm.get_project_memory(project_id)
    clean_entry = next(e for e in entries if e.artifact_id == "clean")
    assert clean_entry.metadata.get("confirmed") is True


def test_confirm_memory_entry_returns_false_on_unknown_entry(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    result = pm.confirm_memory_entry("proj-1", "nonexistent", run_id="run-1")
    assert result is False


def test_collect_with_require_confirmation_sets_unconfirmed(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"
    run_id = "run-1"
    run_dir = pm.projects_dir / project_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_registry.json").write_text(
        json.dumps({"artifacts": {"model": "03_training/model.pkl"}})
    )
    (run_dir / "03_training").mkdir(parents=True)
    (run_dir / "03_training" / "model.pkl").write_text("model")

    entries = pm.collect_run_artifacts(project_id, run_id, require_confirmation=True)
    assert entries[0].metadata["confirmed"] is False


def test_manifest_updates_total_runs(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    project_id = "proj-1"

    for i, run_id in enumerate(["run-1", "run-2"], 1):
        run_dir = pm.projects_dir / project_id / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "artifact_registry.json").write_text(
            json.dumps({"artifacts": {f"stage-{i}": f"dir/file-{i}.txt"}})
        )
        (run_dir / "dir").mkdir(parents=True, exist_ok=True)
        (run_dir / "dir" / f"file-{i}.txt").write_text("ok")

    pm.collect_run_artifacts(project_id, "run-1")
    pm.collect_run_artifacts(project_id, "run-2")

    manifest_path = pm.memory_dir / project_id / "memory_manifest.json"
    data = json.loads(manifest_path.read_text())
    assert data["total_runs"] == 2
    assert sorted(data["runs"]) == ["run-1", "run-2"]


def test_project_memory_records_store_decisions_not_raw_data(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    record = ProjectMemoryRecord(
        record_id="backend-rf",
        category="backend_choice",
        summary="Use random forest for small OLED datasets.",
        value={"backend": "random_forest", "feature_type": "ecfp"},
        source_refs=["run:run-1:model_metrics"],
        source_hashes=["sha256:abc"],
        decision="confirmed_backend_choice",
        confirmed_by="user",
    )

    saved = pm.save_project_record("proj-memory", record)
    records = pm.list_project_records("proj-memory")

    assert saved.record_id == "backend-rf"
    assert len(records) == 1
    assert records[0].record_id == record.record_id
    assert records[0].value == record.value
    assert records[0].source_refs == record.source_refs
    raw = json.loads((pm.memory_dir / "proj-memory" / "project_memory_records.json").read_text())
    assert "random_forest" in raw["records"][0]["value"]["backend"]
    assert "raw_data" not in json.dumps(raw)


def test_project_memory_record_rejects_sensitive_or_raw_payloads() -> None:
    with pytest.raises(ValueError):
        ProjectMemoryRecord(
            record_id="bad-secret",
            category="remote_host",
            summary="bad",
            value={"api_key": "secret-value"},
            decision="do_not_store",
        )
    with pytest.raises(ValueError):
        ProjectMemoryRecord(
            record_id="bad-raw",
            category="user_preference",
            summary="bad",
            value={"raw_data": [{"smiles": "CCO", "plqy": 0.8}]},
            decision="do_not_store",
        )
    with pytest.raises(ValueError):
        ProjectMemoryRecord(
            record_id="bad-secret-string",
            category="user_preference",
            summary="Store api_key=sk-test in project memory.",
            value={},
            decision="do_not_store",
        )
    with pytest.raises(ValueError):
        ProjectMemoryRecord(
            record_id="bad-token-string",
            category="user_preference",
            summary="bad",
            value={"note": "Use token=abc123 for remote parsing."},
            decision="do_not_store",
        )
    with pytest.raises(ValueError):
        ProjectMemoryRecord(
            record_id="bad-source-ref",
            category="remote_host",
            summary="Store remote host decision.",
            value={"host": "workstation2"},
            source_refs=["https://example.test/run?api_key=secret-value"],
            decision="confirmed_remote_host",
        )


def test_project_memory_governance_edit_delete_export_and_disable(tmp_path: Path) -> None:
    pm = ProjectMemory(workspace_dir=tmp_path)
    record = ProjectMemoryRecord(
        record_id="parser-choice",
        category="parser_choice",
        summary="Prefer MinerU unless table extraction is poor.",
        value={"preferred_parser": "mineru", "fallback": "pdfplumber"},
        decision="confirmed_parser_policy",
        confirmed_by="user",
    )
    pm.save_project_record("proj-memory", record)

    updated = pm.update_project_record(
        "proj-memory",
        "parser-choice",
        {"summary": "Prefer MinerU; use pdfplumber for poor tables."},
    )
    assert updated is not None
    assert "poor tables" in updated.summary

    exported = pm.export_project_records("proj-memory")
    assert exported["project_id"] == "proj-memory"
    assert exported["records"][0]["record_id"] == "parser-choice"

    pm.set_project_memory_enabled("proj-memory", False)
    assert pm.project_memory_enabled("proj-memory") is False
    assert pm.list_project_records("proj-memory") == []
    assert len(pm.list_project_records("proj-memory", include_disabled_project=True)) == 1

    assert pm.delete_project_record("proj-memory", "parser-choice") is True
    assert pm.delete_project_record("proj-memory", "missing") is False


# ---- PermissionPolicy tests ----

def test_memory_module_keeps_imports_at_top_level() -> None:
    source = inspect.getsource(memory_module)
    enum_import = source.index("from enum import Enum")
    first_class = source.index("class RunMemoryEntry")
    assert enum_import < first_class


def test_permission_resolve_signature_is_action_only() -> None:
    signature = inspect.signature(PermissionPolicy.resolve)
    assert list(signature.parameters) == ["self", "action"]


def test_permission_defaults() -> None:
    pp = PermissionPolicy()
    assert pp.resolve("train_model") == PermissionLevel.CONFIRM_EACH_TIME
    assert pp.resolve("predict_candidates") == PermissionLevel.PROJECT_APPROVED
    assert pp.resolve("filter_rank") == PermissionLevel.AUTO
    assert pp.resolve("render_report") == PermissionLevel.AUTO
    assert pp.resolve("upload_dataset") == PermissionLevel.PROJECT_APPROVED
    assert pp.resolve("promote_asset") == PermissionLevel.CONFIRM_EACH_TIME
    assert pp.resolve("generate_candidates_expensive") == PermissionLevel.CONFIRM_EACH_TIME
    assert pp.resolve("inspect_dataset") == PermissionLevel.AUTO
    assert pp.resolve("clean_dataset") == PermissionLevel.AUTO


def test_permission_unknown_action_defaults_to_confirm() -> None:
    pp = PermissionPolicy()
    assert pp.resolve("unknown_action") == PermissionLevel.CONFIRM_EACH_TIME


def test_permission_policy_decision_requires_project_or_per_action_confirmation() -> None:
    pp = PermissionPolicy()

    auto = pp.decide("filter_rank")
    assert auto.allowed is True
    assert auto.level == PermissionLevel.AUTO

    project_gate = pp.decide("predict_candidates")
    assert project_gate.allowed is False
    assert project_gate.reason == "PROJECT_APPROVAL_REQUIRED"

    project_allowed = pp.decide("predict_candidates", project_approved=True)
    assert project_allowed.allowed is True

    confirm_missing_actor = pp.decide("train_model", confirmed=True)
    assert confirm_missing_actor.allowed is False
    assert confirm_missing_actor.reason == "CONFIRMATION_ACTOR_REQUIRED"

    confirmed = pp.decide("train_model", confirmed=True, actor="user")
    assert confirmed.allowed is True

    expensive_missing_confirmation = pp.decide("generate_candidates_expensive")
    assert expensive_missing_confirmation.allowed is False
    assert expensive_missing_confirmation.reason == "CONFIRMATION_REQUIRED"

    expensive_confirmed = pp.decide(
        "generate_candidates_expensive",
        confirmed=True,
        actor="user",
    )
    assert expensive_confirmed.allowed is True


def test_permission_policy_strict_project_approval_requires_actor() -> None:
    pp = PermissionPolicy(require_actor_for_project_approved=True)

    missing_actor = pp.decide("predict_candidates", project_approved=True)
    assert missing_actor.allowed is False
    assert missing_actor.reason == "PROJECT_APPROVAL_ACTOR_REQUIRED"

    with_actor = pp.decide("predict_candidates", project_approved=True, actor="user-a")
    assert with_actor.allowed is True
    assert with_actor.actor == "user-a"


def test_set_policy_overrides_default() -> None:
    pp = PermissionPolicy()
    pp.set_policy("train_model", PermissionLevel.AUTO)
    assert pp.resolve("train_model") == PermissionLevel.AUTO


def test_dataset_public_flag() -> None:
    pp = PermissionPolicy()
    assert pp.is_dataset_public("data/private.csv") is False
    pp.set_dataset_public("data/public.csv", public=True)
    assert pp.is_dataset_public("data/public.csv") is True
    assert pp.is_dataset_public("data/private.csv") is False


def test_external_llm_context_allowed_respects_dataset_public() -> None:
    pp = PermissionPolicy()
    assert pp.external_llm_context_allowed("data/private.csv") is False
    pp.set_dataset_public("data/open.csv", public=True)
    assert pp.external_llm_context_allowed("data/open.csv") is True


# ---- RunMemoryEntry / ProjectMemoryManifest models ----

def test_run_memory_entry_defaults() -> None:
    entry = RunMemoryEntry(
        run_id="r1",
        stage="clean",
        artifact_id="clean",
        artifact_path="/tmp/clean.csv",
        collected_at="2026-01-01T00:00:00Z",
    )
    assert entry.metadata == {}


def test_project_memory_manifest_defaults() -> None:
    m = ProjectMemoryManifest(project_id="p1", updated_at="2026-01-01T00:00:00Z")
    assert m.runs == []
    assert m.artifacts == []
    assert m.total_runs == 0
