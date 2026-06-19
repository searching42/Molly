from __future__ import annotations

import json
from io import BytesIO

from ai4s_agent.app import create_app


def test_upload_creates_immutable_versioned_asset_and_legacy_compat_copy(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    response = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    asset = response.json["asset"]
    assert asset["asset_id"] == "upload/dataset"
    assert asset["version"] == "v001"
    assert asset["filename"] == "dataset.csv"
    assert asset["original_filename"] == "dataset.csv"
    assert asset["size_bytes"] == len(b"SMILES,value\nCCO,1.0\n")
    assert asset["sha256"]
    asset_file = tmp_path / "projects" / "proj-a" / "assets" / "uploads" / "dataset" / "v001" / "dataset.csv"
    assert asset_file.exists()
    assert asset_file.read_bytes() == b"SMILES,value\nCCO,1.0\n"
    manifest = json.loads((asset_file.parent / "asset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["asset_id"] == "upload/dataset"
    assert manifest["asset_type"] == "uploaded_dataset"
    assert manifest["version"] == "v001"
    assert manifest["content_hash"] == asset["content_hash"]
    legacy = tmp_path / "projects" / "proj-a" / "uploads" / "dataset.csv"
    assert legacy.exists()
    assert response.json["path"] == str(legacy)


def test_reupload_same_filename_allocates_new_asset_version_without_overwriting_legacy_copy(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    first = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )
    second = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCN,2.0\n"), "dataset.csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json["asset"]["version"] == "v001"
    assert second.json["asset"]["version"] == "v002"
    first_asset = tmp_path / "projects" / "proj-a" / "assets" / "uploads" / "dataset" / "v001" / "dataset.csv"
    second_asset = tmp_path / "projects" / "proj-a" / "assets" / "uploads" / "dataset" / "v002" / "dataset.csv"
    assert first_asset.read_bytes() == b"SMILES,value\nCCO,1.0\n"
    assert second_asset.read_bytes() == b"SMILES,value\nCCN,2.0\n"
    assert first.json["asset"]["content_hash"] != second.json["asset"]["content_hash"]
    legacy = tmp_path / "projects" / "proj-a" / "uploads" / "dataset.csv"
    assert legacy.read_bytes() == b"SMILES,value\nCCO,1.0\n"
