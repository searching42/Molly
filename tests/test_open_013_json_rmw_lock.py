from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent.schemas import AssetPromotionRecord, GateDecision, GateName
from ai4s_agent.storage import ProjectStorage


def test_artifact_registry_concurrent_updates_do_not_drop_entries(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)

    def write_artifact(index: int) -> None:
        storage.register_artifact_path("project", "run", f"artifact_{index:03d}", f"outputs/{index:03d}.json")

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write_artifact, range(60)))

    registry = storage.read_artifact_registry("project", "run")
    assert len(registry) == 60
    assert registry["artifact_000"] == "outputs/000.json"
    assert registry["artifact_059"] == "outputs/059.json"


def test_gate_decisions_concurrent_appends_do_not_drop_entries(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)

    def append_decision(index: int) -> None:
        storage.append_gate_decision(
            "project",
            "run",
            GateDecision(
                gate=GateName.DATA_MINING,
                approved=True,
                actor=f"reviewer-{index:03d}",
                note="concurrent append",
                approved_at=now_iso(),
            ),
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(append_decision, range(50)))

    decisions = storage.read_gate_decisions("project", "run")
    assert len(decisions) == 50
    assert {item["actor"] for item in decisions} == {f"reviewer-{index:03d}" for index in range(50)}


def test_asset_promotion_records_concurrent_appends_do_not_drop_entries(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)

    def append_record(index: int) -> None:
        storage.append_asset_promotion_record(
            "project",
            "run",
            AssetPromotionRecord(
                run_id="run",
                asset_id=f"model/oled/plqy/v{index:03d}",
                asset_type="promoted_model_asset",
                version=f"v{index:03d}",
                source_artifacts=[f"artifact_{index:03d}"],
                approved_by=f"reviewer-{index:03d}",
                approved_at=now_iso(),
            ),
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(append_record, range(40)))

    payload = json.loads((storage.run_dir("project", "run") / "asset_promotion_records.json").read_text(encoding="utf-8"))
    records = payload["records"]
    assert len(records) == 40
    assert {item["version"] for item in records} == {f"v{index:03d}" for index in range(40)}


def test_locked_rmw_preserves_storage_containment_for_symlink_json(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    run_path = storage.run_dir("project", "run")
    outside = tmp_path / "outside_registry.json"
    outside.write_text(json.dumps({"artifacts": {"outside": "do-not-touch"}}), encoding="utf-8")
    symlink = run_path / "artifact_registry.json"
    symlink.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes base directory"):
        storage.register_artifact_path("project", "run", "new_artifact", "outputs/new.json")

    assert json.loads(outside.read_text(encoding="utf-8")) == {"artifacts": {"outside": "do-not-touch"}}
    assert symlink.is_symlink()
