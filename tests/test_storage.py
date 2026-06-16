from pathlib import Path

import pytest

from ai4s_agent.schemas import (
    AssetManifest,
    AssetPromotionRecord,
    AssetStatus,
    GateName,
    PlanModel,
    RunStatus,
    StageState,
)
from ai4s_agent.storage import ArtifactStore, ProjectStorage
from ai4s_agent._utils import write_json


def test_plan_has_five_required_gates() -> None:
    plan = PlanModel(run_id="r1", steps=[], gates=[g.value for g in GateName])
    assert len(plan.gates) == 5


def test_artifact_store_writes_json(tmp_path: Path) -> None:
    store = ArtifactStore(base_dir=tmp_path)
    store.write_json("r1", "plan.json", {"ok": True})
    payload = store.read_json("r1", "plan.json")
    assert payload["ok"] is True


def test_artifact_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(base_dir=tmp_path)
    with pytest.raises(ValueError):
        store.write_json("../outside", "plan.json", {"ok": True})
    with pytest.raises(ValueError):
        store.write_json("r1", "../outside.json", {"ok": True})


def test_missing_read_does_not_create_run_dir(tmp_path: Path) -> None:
    store = ArtifactStore(base_dir=tmp_path)
    assert store.read_json("missing", "plan.json") == {}
    assert not (tmp_path / "missing").exists()


def test_read_json_returns_empty_for_non_object_or_invalid_json(tmp_path: Path) -> None:
    store = ArtifactStore(base_dir=tmp_path)
    path = store.run_dir("r1") / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store.read_json("r1", "list.json") == {}

    invalid_path = store.run_dir("r1") / "invalid.json"
    invalid_path.write_text("{bad json", encoding="utf-8")
    assert store.read_json("r1", "invalid.json") == {}


def test_project_storage_uses_project_run_paths(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    path = storage.run_dir("proj-a", "run-1")
    assert path == tmp_path / "projects" / "proj-a" / "runs" / "run-1"


def test_project_storage_rejects_project_and_run_path_traversal(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    with pytest.raises(ValueError):
        storage.project_dir("../escape")
    with pytest.raises(ValueError):
        storage.run_dir("proj-a", "../escape")
    assert not (tmp_path / "projects" / "proj-a" / "escape").exists()


def test_project_storage_stage_roundtrip(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    stage = StageState(
        stage="clean_execute",
        next_stage="trainability",
        status=RunStatus.RUNNING,
        started_at="2026-05-28T10:00:00Z",
        updated_at="2026-05-28T10:01:00Z",
    )
    storage.write_stage_state("proj-a", "run-1", stage)
    loaded = storage.read_stage_state("proj-a", "run-1")
    assert loaded is not None
    assert loaded.stage == "clean_execute"
    assert loaded.status == RunStatus.RUNNING


def test_project_storage_allocates_v001_style_asset_versions(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    first = storage.allocate_asset_version("proj-a", ["datasets", "cleaned", "train"])
    assert first == "v001"
    created = storage.create_asset_version_dir("proj-a", ["datasets", "cleaned", "train"])
    assert created.name == "v001"
    second = storage.allocate_asset_version("proj-a", ["datasets", "cleaned", "train"])
    assert second == "v002"


def test_project_storage_writes_manifest_and_promotion_records(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    manifest = AssetManifest(
        asset_id="asset-1",
        asset_type="cleaned_dataset",
        version="v001",
        status=AssetStatus.CONFIRMED,
        created_from_run_id="run-1",
        source_artifacts=["02_data/cleaned_train.csv"],
        content_hash="sha256:abc",
    )
    manifest_path = storage.write_asset_manifest(
        "proj-a",
        ["datasets", "cleaned", "train"],
        "v001",
        manifest,
    )
    assert manifest_path.exists()

    record = AssetPromotionRecord(
        run_id="run-1",
        asset_id="asset-1",
        asset_type="cleaned_dataset",
        version="v001",
        source_artifacts=["02_data/cleaned_train.csv"],
        approved_by="user",
        approved_at="2026-05-28T10:03:00Z",
    )
    path = storage.append_asset_promotion_record("proj-a", "run-1", record)
    assert path.exists()


def test_project_storage_artifact_registry_roundtrip(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    storage.register_artifact_path(
        "proj-a",
        "run-1",
        artifact_id="trainability_report",
        relative_path="03_training/trainability_report.json",
    )
    registry = storage.read_artifact_registry("proj-a", "run-1")
    assert registry["trainability_report"] == "03_training/trainability_report.json"


def test_register_model_asset_creates_versioned_manifest(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(model_dir / "model_metadata.json", {"property_id": "plqy", "train_size": 100})
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    manifest, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="random_forest",
        content_hash="sha256:abc123",
        approved_by="user",
        approval_note="confirmed training output",
    )

    assert manifest.asset_id == "model/random_forest/plqy"
    assert manifest.status == AssetStatus.CANDIDATE
    assert manifest.version == "v001"
    assert manifest.created_from_run_id == "run-1"
    assert version_dir.name == "v001"
    assert (version_dir / "model" / "model.pkl").exists()
    assert (version_dir / "asset_manifest.json").exists()
    assert (version_dir / "model_registration_record.json").exists()


def test_register_model_asset_copies_nested_model_directories(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch_1.pt").write_bytes(b"weights")
    config_dir = model_dir / "configs"
    config_dir.mkdir()
    (config_dir / "model.json").write_text("{}", encoding="utf-8")

    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="random_forest",
        content_hash="sha256:abc123",
        approved_by="user",
    )

    assert (version_dir / "model" / "checkpoints" / "epoch_1.pt").exists()
    assert (version_dir / "model" / "configs" / "model.json").exists()


def test_register_model_asset_requires_user_confirmation(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    with pytest.raises(ValueError, match="confirmation"):
        storage.register_model_asset(
            "proj-a",
            "run-1",
            model_dir,
            property_id="plqy",
            backend="random_forest",
            content_hash="sha256:abc123",
        )

    assert not (tmp_path / "projects" / "proj-a" / "assets" / "models").exists()


def test_register_model_asset_rejects_sources_outside_run_dir(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = tmp_path / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    with pytest.raises(ValueError, match="model_dir escapes"):
        storage.register_model_asset(
            "proj-a",
            "run-1",
            model_dir,
            property_id="plqy",
            backend="random_forest",
            content_hash="sha256:abc123",
            approved_by="user",
        )


def test_register_model_asset_rejects_symlinked_files(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "safe.txt").write_text("ok", encoding="utf-8")
    (model_dir / "passwd_link").symlink_to("/etc/passwd")

    with pytest.raises(ValueError, match="symlink"):
        storage.register_model_asset(
            "proj-a",
            "run-1",
            model_dir,
            property_id="plqy",
            backend="random_forest",
            content_hash="sha256:abc123",
            approved_by="user",
        )


def test_register_model_asset_prevents_overwrite_via_versioning(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir_1 = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir_1.mkdir(parents=True)
    (model_dir_1 / "model.pkl").write_bytes(b"v1")

    manifest1, _ = storage.register_model_asset(
        "proj-a", "run-1", model_dir_1, property_id="plqy",
        backend="baseline", content_hash="sha256:abc", approved_by="user",
    )
    assert manifest1.version == "v001"

    model_dir_2 = storage.run_dir("proj-a", "run-2") / "03_training" / "source_model"
    model_dir_2.mkdir(parents=True)
    (model_dir_2 / "model.pkl").write_bytes(b"v2")

    manifest2, _ = storage.register_model_asset(
        "proj-a", "run-2", model_dir_2, property_id="plqy",
        backend="baseline", content_hash="sha256:def", approved_by="user",
    )
    assert manifest2.version == "v002"


def test_project_storage_allocates_versions_after_v999(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    scope_path = storage.assets_dir("proj-a") / "datasets" / "cleaned" / "train"
    (scope_path / "v1000").mkdir(parents=True)

    assert storage.allocate_asset_version("proj-a", ["datasets", "cleaned", "train"]) == "v1001"


def test_register_model_asset_uses_reserved_version_dir_for_manifest(tmp_path: Path) -> None:
    class RacingStorage(ProjectStorage):
        def create_asset_version_dir(self, project_id: str, scope: list[str]) -> Path:
            scope_path = self._asset_scope_dir(project_id, scope)
            (scope_path / "v001").mkdir(parents=True, exist_ok=True)
            version_dir = scope_path / "v002"
            version_dir.mkdir(parents=True, exist_ok=False)
            return version_dir

    storage = RacingStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    manifest, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="baseline",
        content_hash="sha256:abc",
        approved_by="user",
    )

    assert manifest.version == version_dir.name == "v002"
    assert (version_dir / "asset_manifest.json").exists()


def test_model_registration_api_requires_confirmation_and_registers_asset(tmp_path: Path) -> None:
    from ai4s_agent.app import create_app

    model_dir = tmp_path / "projects" / "proj-a" / "runs" / "run-1" / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    app = create_app(base_runs_dir=tmp_path)
    with app.test_client() as client:
        resp = client.post(
            "/api/projects/proj-a/runs/run-1/models/register",
            json={
                "model_dir": str(model_dir),
                "property_id": "plqy",
                "backend": "baseline",
                "content_hash": "sha256:abc",
            },
        )
        assert resp.status_code == 403
        assert resp.json["permission"]["level"] == "confirm-each-time"

        resp = client.post(
            "/api/projects/proj-a/runs/run-1/models/register",
            json={
                "model_dir": str(model_dir),
                "property_id": "plqy",
                "backend": "baseline",
                "content_hash": "sha256:abc",
                "confirmed": True,
                "approved_by": "user",
                "approval_note": "confirmed after review",
            },
        )
        assert resp.status_code == 200
        assert resp.json["manifest"]["asset_id"] == "model/baseline/plqy"
        assert Path(resp.json["version_dir"]).name == "v001"


def test_model_registration_api_rejects_model_dir_outside_run(tmp_path: Path) -> None:
    from ai4s_agent.app import create_app

    model_dir = tmp_path / "source_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"fake-model")

    app = create_app(base_runs_dir=tmp_path)
    with app.test_client() as client:
        resp = client.post(
            "/api/projects/proj-a/runs/run-1/models/register",
            json={
                "model_dir": str(model_dir),
                "property_id": "plqy",
                "backend": "baseline",
                "content_hash": "sha256:abc",
                "confirmed": True,
                "approved_by": "user",
            },
        )
        assert resp.status_code == 400
        assert "model_dir escapes" in resp.json["error"]
