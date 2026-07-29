from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.schemas import GateName, RunStatus


@pytest.fixture
def dataset_app_client(tmp_path: Path):
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    return app, app.test_client()


def _dataset_bytes(row_count: int = 48) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["Chromophore", "Quantum yield", "Reference"],
    )
    writer.writeheader()
    for index in range(1, row_count + 1):
        writer.writerow(
            {
                "Chromophore": "C" * index,
                "Quantum yield": round(0.05 + index / (row_count + 5), 6),
                "Reference": f"source-{index}",
            }
        )
    return stream.getvalue().encode("utf-8")


def _upload_inspect_confirm(client, project_id: str = "dataset-workflow") -> dict:
    conversation = client.post(
        f"/api/projects/{project_id}/conversations",
        json={"title": "Dataset workflow"},
    )
    assert conversation.status_code == 201
    conversation_id = conversation.get_json()["conversation"]["conversation_id"]
    upload = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/attachments",
        data={
            "files": (
                io.BytesIO(_dataset_bytes()),
                "chromophores.csv",
                "text/csv",
            )
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201
    attachment = upload.get_json()["attachments"][0]

    inspected = client.post(
        f"/api/projects/{project_id}/datasets/inspect-attachment",
        json={"artifact_id": attachment["artifact_id"]},
    )
    assert inspected.status_code == 201
    raw = inspected.get_json()["dataset"]
    assert raw["dataset_id"] == f'dataset_{attachment["sha256"]}'
    assert raw["status"] == "raw"
    assert raw["dataset_profile"]["row_count"] == 48
    assert "Chromophore" in raw["available_inputs"]
    assert "input_csv" not in raw["dataset_profile"]
    assert raw["source_attachment"]["original_name"] == "chromophores.csv"

    confirmed = client.post(
        f"/api/projects/{project_id}/datasets/{raw['dataset_id']}/confirm",
        json={
            "smiles_column": "Chromophore",
            "target_column": "Quantum yield",
            "property_id": "plqy",
            "confirmed_by": "dataset-reviewer",
            "strict_smiles_cleaning": False,
            "drop_empty_target_rows": True,
        },
    )
    assert confirmed.status_code == 200
    return confirmed.get_json()["confirmed_dataset"]


def test_dataset_routes_bind_raw_attachment_and_publish_confirmed_dataset(
    dataset_app_client,
) -> None:
    _app, client = dataset_app_client

    confirmed = _upload_inspect_confirm(client)

    assert confirmed["status"] == "confirmed"
    assert confirmed["mapping"] == {
        "smiles_column": "Chromophore",
        "target_column": "Quantum yield",
        "property_id": "plqy",
    }
    assert confirmed["confirmation"]["confirmed_by"] == "dataset-reviewer"
    assert confirmed["summary"]["numeric_label_count"] == 48
    assert confirmed["summary"]["source_row_count"] == 48
    assert confirmed["summary"]["confirmed_row_count"] == 48
    assert set(confirmed["artifacts"]) == {
        "cleaned_train_dataset",
        "confirmed_training_dataset",
        "property_catalog",
        "confirmed_dataset_manifest",
    }

    confirmed_csv = Path(confirmed["artifacts"]["cleaned_train_dataset"])
    manifest_path = Path(confirmed["artifacts"]["confirmed_dataset_manifest"])
    assert confirmed_csv.is_file()
    assert manifest_path.is_file()
    with confirmed_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert rows[0]["SMILES"] == "C"
    assert rows[0]["plqy"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hashes"]["source_sha256"] == confirmed["source_attachment"]["sha256"]
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    listing = client.get("/api/projects/dataset-workflow/datasets")
    assert listing.status_code == 200
    assert listing.get_json()["datasets"][0]["confirmations"][0]["confirmation_id"] == confirmed["confirmation_id"]


def test_confirmed_dataset_runs_model_package_generation_publication_and_topn(
    tmp_path: Path,
    dataset_app_client,
) -> None:
    _app, client = dataset_app_client
    project_id = "dataset-chain"
    confirmed = _upload_inspect_confirm(client, project_id)
    run_id = "dataset-chain-local-001"

    expanded = client.post(
        "/api/run-plan/expand",
        json={
            "run_id": run_id,
            "requested_tasks": ["run_baseline", "render_report"],
            "available_artifacts": ["cleaned_train_dataset", "property_catalog"],
        },
    )
    assert expanded.status_code == 200
    run_plan = expanded.get_json()["run_plan"]
    task_options = {
        "train_model": {"property_id": "plqy", "smiles_col": "SMILES"},
        "generate_candidates": {"backend": "deterministic_stub", "count": 12, "seed": 7},
        "filter_rank": {"topn": 3},
    }
    start = client.post(
        "/api/run-plan/execute",
        json={
            "project_id": project_id,
            "run_plan": run_plan,
            "input_artifacts": confirmed["artifacts"],
            "task_options": task_options,
        },
    )
    assert start.status_code == 200
    assert start.get_json()["execution"]["status"] == RunStatus.WAITING_USER.value
    assert start.get_json()["execution"]["waiting_task"] == "train_model"

    trained = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": project_id,
            "run_plan": run_plan,
            "approved_gates": [GateName.TRAIN_CONFIG.value],
            "actor": "model-reviewer",
            "note": "Approve deterministic acceptance training.",
            "input_artifacts": confirmed["artifacts"],
            "task_options": task_options,
        },
    )
    assert trained.status_code == 200
    assert trained.get_json()["execution"]["status"] == RunStatus.WAITING_USER.value
    assert trained.get_json()["execution"]["waiting_task"] == "generate_candidates"

    completed = client.post(
        "/api/run-plan/resume",
        json={
            "project_id": project_id,
            "run_plan": run_plan,
            "approved_gates": [GateName.FINAL_THRESHOLD.value],
            "actor": "generation-reviewer",
            "note": "Approve deterministic candidate generation.",
            "input_artifacts": confirmed["artifacts"],
            "task_options": task_options,
        },
    )
    assert completed.status_code == 200
    assert completed.get_json()["execution"]["status"] == RunStatus.SUCCEEDED.value

    status = client.get(f"/api/runs/{run_id}?project_id={project_id}")
    artifacts = status.get_json()["artifacts"]
    assert {
        "trained_model",
        "model_manifest",
        "generation_publication",
        "topn_export",
    }.issubset(artifacts)
    run_dir = tmp_path / "workspace" / "projects" / project_id / "runs" / run_id
    publication = json.loads((run_dir / artifacts["generation_publication"]).read_text(encoding="utf-8"))
    assert publication["publication_kind"] == "deterministic_local_smoke"
    assert publication["claim_boundary"] == "recommendation_only_not_experimental_validation"
    with (run_dir / artifacts["topn_export"]).open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3

    candidate_path = run_dir / artifacts["candidate_dataset"]
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "tampered,CC,generated,stub,99\n",
        encoding="utf-8",
    )
    rejected = client.get(f"/api/runs/{run_id}?project_id={project_id}")
    assert rejected.status_code == 400
    assert rejected.get_json()["error"] == (
        "generation publication failed integrity verification"
    )


@pytest.mark.parametrize(
    "artifact_id",
    ["cleaned_train_dataset", "property_catalog"],
)
def test_confirmed_dataset_rejects_replaced_artifact_bytes(
    tmp_path: Path,
    artifact_id: str,
    dataset_app_client,
) -> None:
    _app, client = dataset_app_client
    project_id = f"tamper-{artifact_id.replace('_', '-')}"
    confirmed = _upload_inspect_confirm(client, project_id)
    artifact_path = Path(confirmed["artifacts"][artifact_id])
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\nreplaced\n")

    listing = client.get(f"/api/projects/{project_id}/datasets")
    assert listing.status_code == 409
    body = listing.get_json()
    assert body["error_code"] == "dataset_verification_failed"
    assert str(tmp_path) not in json.dumps(body)

    replay = client.post(
        f"/api/projects/{project_id}/datasets/{confirmed['dataset_id']}/confirm",
        json={
            **confirmed["mapping"],
            "confirmed_by": "dataset-reviewer",
            "strict_smiles_cleaning": False,
            "drop_empty_target_rows": True,
        },
    )
    assert replay.status_code == 400
    assert replay.get_json()["error_code"] == "dataset_confirmation_failed"
    assert str(tmp_path) not in json.dumps(replay.get_json())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "molly_confirmed_dataset.invalid"),
        ("project_id", "another-project"),
        ("source_attachment", {"artifact_id": "artifact_forged"}),
        (
            "artifacts",
            {
                "cleaned_train_dataset": "../../outside.csv",
                "confirmed_training_dataset": "../../outside.csv",
                "property_catalog": "property_catalog.json",
            },
        ),
    ],
)
def test_confirmed_dataset_rejects_manifest_identity_and_path_tampering(
    field: str,
    replacement: object,
    dataset_app_client,
) -> None:
    _app, client = dataset_app_client
    project_id = f"manifest-tamper-{field.replace('_', '-')}"
    confirmed = _upload_inspect_confirm(client, project_id)
    manifest_path = Path(confirmed["artifacts"]["confirmed_dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rejected = client.get(f"/api/projects/{project_id}/datasets")
    assert rejected.status_code == 409
    assert rejected.get_json()["error_code"] == "dataset_verification_failed"


def test_dataset_routes_do_not_echo_internal_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    dataset_app_client,
) -> None:
    app, client = dataset_app_client
    service = app.extensions["dataset_workflow_service"]
    secret = "/private/internal/datasets/secret.csv"

    def fail(*args, **kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(service, "list_datasets", fail)
    listed = client.get("/api/projects/safe-errors/datasets")
    assert listed.status_code == 409
    assert secret not in json.dumps(listed.get_json())

    monkeypatch.setattr(service, "inspect_attachment", fail)
    inspected = client.post(
        "/api/projects/safe-errors/datasets/inspect-attachment",
        json={"artifact_id": "artifact_test"},
    )
    assert inspected.status_code == 400
    assert secret not in json.dumps(inspected.get_json())

    monkeypatch.setattr(service, "confirm_dataset", fail)
    confirmed = client.post(
        "/api/projects/safe-errors/datasets/dataset_"
        + "a" * 64
        + "/confirm",
        json={
            "smiles_column": "SMILES",
            "target_column": "target",
            "property_id": "target",
            "confirmed_by": "reviewer",
        },
    )
    assert confirmed.status_code == 400
    assert secret not in json.dumps(confirmed.get_json())
