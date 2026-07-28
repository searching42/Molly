from pathlib import Path

import pytest

import ai4s_agent._utils as utils
from ai4s_agent.schemas import (
    AssetManifest,
    AssetPromotionRecord,
    AssetStatus,
    GateName,
    PlanModel,
    PromotedModelAsset,
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


def test_write_json_replaces_from_same_directory_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "payload.json"
    calls: list[tuple[Path, Path]] = []
    original_replace = utils.os.replace

    def recording_replace(src: object, dst: object) -> None:
        calls.append((Path(src), Path(dst)))
        original_replace(src, dst)

    monkeypatch.setattr(utils.os, "replace", recording_replace)

    write_json(path, {"ok": True})

    assert path.read_text(encoding="utf-8").strip().startswith("{")
    assert calls
    temp_path, dest_path = calls[0]
    assert temp_path.parent == path.parent
    assert temp_path != path
    assert dest_path == path
    assert not temp_path.exists()


def test_project_storage_stage_writes_use_atomic_json_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    original_replace = utils.os.replace

    def recording_replace(src: object, dst: object) -> None:
        calls.append((Path(src), Path(dst)))
        original_replace(src, dst)

    monkeypatch.setattr(utils.os, "replace", recording_replace)
    storage = ProjectStorage(workspace_dir=tmp_path)

    storage.write_stage_state(
        "proj-a",
        "run-1",
        StageState(
            stage="inspect_dataset",
            status=RunStatus.SUCCEEDED,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
    )

    assert calls
    assert calls[0][1].name == "stage.json"


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


def test_immutable_artifact_registry_group_preserves_concurrent_entries_and_cleanup_is_cas(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    group = {"inverse_receipt": "inverse/receipt.json", "inverse_csv": "inverse/candidates.csv"}
    storage.register_artifact_path("proj-a", "run-1", "unrelated", "other.json")
    storage.register_new_artifact_registry_paths("proj-a", "run-1", group)
    with pytest.raises(ValueError, match="already immutable"):
        storage.register_new_artifact_registry_paths(
            "proj-a", "run-1", {"inverse_receipt": "replacement.json"}
        )
    storage.register_artifact_path(
        "proj-a", "run-1", "inverse_receipt", "concurrently-replaced.json"
    )
    storage.remove_artifact_registry_paths_if_all_equal("proj-a", "run-1", group)
    assert storage.read_artifact_registry("proj-a", "run-1") == {
        "unrelated": "other.json",
        "inverse_receipt": "concurrently-replaced.json",
        "inverse_csv": "inverse/candidates.csv",
    }

    clean_group = {"other_receipt": "other/receipt.json", "other_csv": "other/candidates.csv"}
    storage.register_new_artifact_registry_paths("proj-a", "run-1", clean_group)
    storage.remove_artifact_registry_paths_if_all_equal("proj-a", "run-1", clean_group)
    assert storage.read_artifact_registry("proj-a", "run-1") == {
        "unrelated": "other.json",
        "inverse_receipt": "concurrently-replaced.json",
        "inverse_csv": "inverse/candidates.csv",
    }


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


def test_promote_registered_model_asset_writes_confirmed_promoted_asset(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "domain_model_manifest.json",
        {"model_id": "plqy_promoted_v001", "model_backend": "unimol_with_solvent_pca64"},
    )
    (model_dir / "weights.pt").write_bytes(b"fake-weights")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="unimol_with_solvent_pca64",
        content_hash="sha256:model",
        approved_by="user",
    )

    promoted, promoted_path = storage.promote_registered_model_asset(
        "proj-a",
        "run-1",
        version_dir,
        model_id="plqy_promoted_v001",
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        backend="unimol_with_solvent_pca64",
        approved_by="user",
        metrics={"mae": 0.171, "r2": 0.41},
        applicability={"split": "scaffold", "dataset": "chromophore solvent-conditioned"},
        feature_requirements=["canonical_smiles", "solvent"],
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        limitations=["high PLQY compression remains monitored"],
        rollback_asset_id="model/unimol_with_solvent_pca64/plqy/v000",
    )

    assert promoted.asset_id == "model/unimol_with_solvent_pca64/plqy/v001"
    assert promoted.status == AssetStatus.CONFIRMED
    assert promoted.model_dir == str(version_dir / "model")
    assert promoted.created_from_run_id == "run-1"
    assert promoted.source_artifacts == [str(model_dir)]
    assert promoted_path == version_dir / "promoted_model_asset.json"
    restored = PromotedModelAsset.model_validate_json(promoted_path.read_text(encoding="utf-8"))
    assert restored.model_dump(mode="json") == promoted.model_dump(mode="json")

    manifest_payload = (version_dir / "asset_manifest.json").read_text(encoding="utf-8")
    manifest = AssetManifest.model_validate_json(manifest_payload)
    assert manifest.status == AssetStatus.CONFIRMED

    listed = storage.list_promoted_model_assets("proj-a", domain="oled", property_id="plqy")
    assert [asset.asset_id for asset in listed] == [promoted.asset_id]
    record_path = tmp_path / "projects" / "proj-a" / "runs" / "run-1" / "asset_promotion_records.json"
    assert record_path.exists()


def test_promoted_model_asset_draft_infers_fields_from_model_metadata(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "domain_model_manifest.json",
        {
            "model_id": "plqy_promoted_v001",
            "model_backend": "unimol_with_solvent_pca64",
            "domain": "oled",
            "property_id": "plqy",
            "use_case": "scalar_prediction",
            "metrics": {"mae": 0.171, "r2": 0.41},
            "applicability": {"split": "scaffold", "dataset": "chromophore solvent-conditioned"},
            "feature_requirements": ["canonical_smiles", "solvent"],
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
            "limitations": ["high PLQY compression remains monitored"],
        },
    )
    write_json(model_dir / "model_metadata.json", {"train_size": 13049, "metrics": {"pearson": 0.64}})
    (model_dir / "weights.pt").write_bytes(b"fake-weights")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="unimol_with_solvent_pca64",
        content_hash="sha256:model",
        approved_by="user",
    )

    draft = storage.build_promoted_model_asset_draft("proj-a", version_dir)

    assert draft["version_dir"] == str(version_dir)
    assert draft["model_id"] == "plqy_promoted_v001"
    assert draft["domain"] == "oled"
    assert draft["property_id"] == "plqy"
    assert draft["use_case"] == "scalar_prediction"
    assert draft["backend"] == "unimol_with_solvent_pca64"
    assert draft["metrics"] == {"mae": 0.171, "r2": 0.41, "pearson": 0.64}
    assert draft["applicability"]["split"] == "scaffold"
    assert draft["applicability"]["train_size"] == 13049
    assert draft["feature_requirements"] == ["canonical_smiles", "solvent"]
    assert draft["input_columns"] == {"canonical_smiles": "SMILES", "solvent": "solvent"}
    assert draft["limitations"] == ["high PLQY compression remains monitored"]
    assert draft["warnings"] == []


def test_promoted_model_asset_draft_allows_explicit_overrides(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(model_dir / "model_metadata.json", {"model_id": "raw_model", "metrics": {"r2": 0.2}})
    (model_dir / "model.pkl").write_bytes(b"fake-model")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="baseline",
        content_hash="sha256:model",
        approved_by="user",
    )

    draft = storage.build_promoted_model_asset_draft(
        "proj-a",
        version_dir,
        overrides={
            "model_id": "reviewed_model",
            "domain": "oled",
            "property_id": "emission_max_nm",
            "use_case": "high_plqy_screening",
            "backend": "other_backend",
            "metrics": {"r2": 0.31, "bad": True},
            "feature_requirements": ["canonical_smiles"],
        },
    )

    assert draft["model_id"] == "reviewed_model"
    assert draft["domain"] == "oled"
    assert draft["property_id"] == "plqy"
    assert draft["backend"] == "baseline"
    assert draft["use_case"] == "high_plqy_screening"
    assert draft["metrics"] == {"r2": 0.31}
    assert draft["feature_requirements"] == ["canonical_smiles"]


def test_promoted_model_asset_draft_skips_nonfinite_metrics(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "model_metadata.json",
        {"model_id": "raw_model", "metrics": {"r2": float("nan"), "mae": 0.17}},
    )
    (model_dir / "model.pkl").write_bytes(b"fake-model")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="baseline",
        content_hash="sha256:model",
        approved_by="user",
    )

    draft = storage.build_promoted_model_asset_draft("proj-a", version_dir)

    assert draft["metrics"] == {"mae": 0.17}


def test_promote_registered_model_asset_rejects_non_model_manifest(tmp_path: Path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    manifest = AssetManifest(
        asset_id="dataset/cleaned/train",
        asset_type="cleaned_dataset",
        version="v001",
        status=AssetStatus.CANDIDATE,
        created_from_run_id="run-1",
        source_artifacts=["02_data/cleaned.csv"],
        content_hash="sha256:data",
    )
    storage.write_asset_manifest("proj-a", ["datasets", "cleaned", "train"], "v001", manifest)
    version_dir = tmp_path / "projects" / "proj-a" / "assets" / "datasets" / "cleaned" / "train" / "v001"
    (version_dir / "model").mkdir()

    with pytest.raises(ValueError, match="trained_model"):
        storage.promote_registered_model_asset(
            "proj-a",
            "run-1",
            version_dir,
            model_id="bad_model",
            domain="oled",
            property_id="plqy",
            use_case="scalar_prediction",
            backend="baseline",
            approved_by="user",
        )


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
