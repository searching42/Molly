from __future__ import annotations

import io
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import ai4s_agent.adapters.phase3 as phase3_module
import ai4s_agent.literature_intake as literature_intake_module
from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.schemas import RunStatus
from tests.document_parse_test_helpers import write_synthetic_pdf


def _create_conversation(client, project_id: str) -> str:
    response = client.post(
        f"/api/projects/{project_id}/conversations",
        json={"title": "Literature intake"},
    )
    assert response.status_code == 201
    return str(response.get_json()["conversation"]["conversation_id"])


def _upload_pdfs(
    client,
    tmp_path: Path,
    project_id: str,
    conversation_id: str,
    count: int,
) -> list[dict]:
    files = []
    for index in range(count):
        pdf = write_synthetic_pdf(tmp_path / f"paper-{index + 1}.pdf")
        files.append((io.BytesIO(pdf.read_bytes()), pdf.name, "application/pdf"))
    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/attachments",
        data={"files": files},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    return list(response.get_json()["attachments"])


def _freeze_request(
    client,
    project_id: str,
    conversation_id: str,
    attachments: list[dict],
) -> dict:
    message = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/messages",
        json={
            "role": "user",
            "content": "Parse the attached literature.",
            "attachment_ids": [item["artifact_id"] for item in attachments],
        },
    )
    assert message.status_code == 201
    message_id = message.get_json()["message"]["message_id"]
    frozen = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/execution-requests",
        json={
            "selected_message_ids": [message_id],
            "task_type": "literature_parse",
            "model_profile_id": "not_applicable",
            "user_parameters": {"parser_profile": "pdfplumber_local"},
            "client_request_id": "literature-intake-test",
        },
    )
    assert frozen.status_code == 201
    return dict(frozen.get_json()["execution_request"])


def _intake_id(service, project_id: str, conversation_id: str, frozen: dict, attachments: list[dict]) -> str:
    identity = {
        "project_id": project_id,
        "conversation_id": conversation_id,
        "conversation_request_id": frozen["request_id"],
        "conversation_request_sha256": frozen["request_sha256"],
        "parser_profile": "pdfplumber_local",
        "attachment_sha256": sorted(item["sha256"] for item in attachments),
    }
    return f"literature_intake_{service._sha256_json(identity)[:24]}"


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_single_local_pdf_click_registers_corpus_and_executes_existing_task(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "single-literature-intake"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)

    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={
            "request_id": frozen["request_id"],
            "parser_profile": "pdfplumber_local",
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert payload["intake"]["authorization_mode"] == "click"
    assert payload["intake"]["required_gates"] == []
    assert payload["intake"]["task_id"] == "parse_document_pdfplumber"
    assert payload["intake"]["sources"][0]["artifact_id"] == attachments[0]["artifact_id"]
    assert payload["intake"]["sources"][0]["corpus_relative_path"].startswith("inputs/")
    serialized = json.dumps(payload, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(payload["intake"], ensure_ascii=False)
    assert "parsed_document" in serialized

    storage = app.extensions["literature_intake_service"].projects
    registry = storage.read_artifact_registry(project_id, payload["intake"]["run_id"])
    assert set(registry) >= {"parsed_document", "parsed_tables", "parser_audit"}
    assert set(registry) >= {"parsed_document_markdown", "literature_parse_publication"}
    stage = storage.read_stage_state(project_id, payload["intake"]["run_id"])
    assert stage is not None and stage.status == RunStatus.SUCCEEDED

    repeated = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert repeated.status_code == 200
    assert repeated.get_json()["idempotent"] is True
    assert repeated.get_json()["execution"]["status"] == RunStatus.SUCCEEDED.value


def test_batch_local_pdf_intake_waits_for_exact_existing_gate_before_parsing(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "batch-literature-intake"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)

    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert submitted.status_code == 201
    payload = submitted.get_json()
    intake = payload["intake"]
    assert intake["authorization_mode"] == "gate"
    assert intake["required_gates"] == ["gate_2_data_mining"]
    assert intake["task_id"] == "parse_pdf_corpus_pdfplumber"
    assert payload["execution"]["status"] == RunStatus.WAITING_USER.value
    listing = client.get(
        f"/api/projects/{project_id}/literature-intakes",
        query_string={"conversation_id": conversation_id},
    )
    assert listing.status_code == 200
    listed = listing.get_json()["literature_intakes"]
    assert len(listed) == 1
    assert listed[0]["intake"]["intake_id"] == intake["intake_id"]
    assert listed[0]["execution"]["status"] == RunStatus.WAITING_USER.value

    storage = app.extensions["literature_intake_service"].projects
    run_dir = storage.run_dir(project_id, intake["run_id"])
    assert not (run_dir / "parsed_corpus").exists()
    stage = storage.read_stage_state(project_id, intake["run_id"])
    assert stage is not None
    snapshot = dict(stage.details["execution_snapshot"])

    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
        json={"actor": "local-user", "note": "Approved exact two-PDF corpus."},
    )
    assert approved.status_code == 200
    approved_payload = approved.get_json()
    assert approved_payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry(project_id, intake["run_id"])
    assert set(registry) >= {
        "parsed_corpus_manifest",
        "parsed_document_001",
        "parsed_document_002",
        "parser_audit_001",
        "parser_audit_002",
        "parser_audit",
    }
    assert registry["parser_audit"] != registry["parsed_corpus_manifest"]
    corpus_manifest = json.loads(
        (run_dir / registry["parsed_corpus_manifest"]).read_text(encoding="utf-8")
    )
    serialized_manifest = json.dumps(corpus_manifest, ensure_ascii=False)
    assert str(tmp_path) not in serialized_manifest
    for document in corpus_manifest["documents"]:
        for descriptor in document["outputs"].values():
            assert not Path(descriptor["relative_path"]).is_absolute()
            assert descriptor["size_bytes"] > 0
            assert len(descriptor["sha256"]) == 64
    decisions = storage.read_gate_decisions(project_id, intake["run_id"])
    assert len(decisions) == 1
    assert decisions[0]["approved_snapshot_id"] == snapshot["snapshot_id"]
    assert decisions[0]["approved_snapshot_hash"] == snapshot["snapshot_hash"]


def test_intake_rejects_non_pdf_frozen_attachment(tmp_path: Path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "invalid-literature-intake"
    conversation_id = _create_conversation(client, project_id)
    upload = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/attachments",
        data={"files": [(io.BytesIO(b"not a pdf"), "notes.txt", "text/plain")]},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201
    frozen = _freeze_request(
        client,
        project_id,
        conversation_id,
        list(upload.get_json()["attachments"]),
    )

    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert response.status_code == 400
    assert "PDF attachments only" in response.get_json()["error"]


def test_literature_intake_fails_closed_when_owned_corpus_roster_changes(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "tampered-literature-intake"
    conversation_id = _create_conversation(client, project_id)
    frozen = _freeze_request(
        client,
        project_id,
        conversation_id,
        _upload_pdfs(client, tmp_path, project_id, conversation_id, 2),
    )
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    service = app.extensions["literature_intake_service"]
    intake_dir = (
        service.projects.project_dir(project_id)
        / "literature-intakes"
        / intake["intake_id"]
    )
    (intake_dir / "inputs" / "unexpected.pdf").write_bytes(b"%PDF-unregistered")

    fetched = client.get(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}"
    )
    assert fetched.status_code == 400
    assert "input roster changed" in fetched.get_json()["error"]
    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
        json={"actor": "local-user"},
    )
    assert approved.status_code == 400
    stage = service.projects.read_stage_state(project_id, intake["run_id"])
    assert stage is not None and stage.status == RunStatus.WAITING_USER


def test_main_ui_exposes_literature_atomic_task_and_gate_action(tmp_path: Path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'id="parse-literature-button"' in html
    assert 'id="approve-literature-button"' in html
    assert "literature-intakes" in html
    assert "intakeListing.literature_intakes" in html
    assert "task_type: \"literature_parse\"" in html


def test_single_pdf_above_click_threshold_requires_existing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(literature_intake_module, "LOCAL_CLICK_MAX_PDF_BYTES", 1)
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "large-single-literature-intake"
    conversation_id = _create_conversation(client, project_id)
    frozen = _freeze_request(
        client,
        project_id,
        conversation_id,
        _upload_pdfs(client, tmp_path, project_id, conversation_id, 1),
    )

    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["intake"]["authorization_mode"] == "gate"
    assert payload["intake"]["click_authorization_limit_bytes"] == 1
    assert payload["execution"]["status"] == RunStatus.WAITING_USER.value


def test_recomputed_manifest_cannot_change_click_authorization_policy(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "manifest-policy-tamper"
    conversation_id = _create_conversation(client, project_id)
    frozen = _freeze_request(
        client,
        project_id,
        conversation_id,
        _upload_pdfs(client, tmp_path, project_id, conversation_id, 1),
    )
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    service = app.extensions["literature_intake_service"]
    manifest_path = (
        service.projects.project_dir(project_id)
        / "literature-intakes"
        / intake["intake_id"]
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["click_authorization_limit_bytes"] += 1
    material = dict(manifest)
    material.pop("intake_sha256")
    manifest["intake_sha256"] = service._sha256_json(material)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fetched = client.get(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}"
    )
    assert fetched.status_code == 400
    assert "authorization policy changed" in fetched.get_json()["error"]


@pytest.mark.parametrize("symlink_component", ["intake", "inputs"])
def test_registration_rejects_preexisting_symlink_components_without_external_writes(
    tmp_path: Path,
    symlink_component: str,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = f"symlink-{symlink_component}"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    service = app.extensions["literature_intake_service"]
    intake_id = _intake_id(service, project_id, conversation_id, frozen, attachments)
    root = service.projects.project_dir(project_id) / "literature-intakes"
    root.mkdir(mode=0o700, exist_ok=True)
    external = tmp_path / f"external-{symlink_component}"
    external.mkdir()
    (external / "sentinel.bin").write_bytes(b"unchanged")
    if symlink_component == "intake":
        (root / intake_id).symlink_to(external, target_is_directory=True)
    else:
        intake_dir = root / intake_id
        intake_dir.mkdir(mode=0o700)
        (intake_dir / "inputs").symlink_to(external, target_is_directory=True)
    before = _directory_bytes(external)

    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )

    assert response.status_code == 400
    assert _directory_bytes(external) == before


def test_single_parser_fails_closed_when_approved_inode_is_replaced_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "single-pdf-inode-race"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)

    def replace_named_inode(path: Path) -> None:
        path.rename(path.with_name(path.name + ".approved-inode"))
        path.write_bytes(b"%PDF-replacement-not-approved")

    monkeypatch.setattr(
        phase3_module,
        "_PINNED_PDF_AFTER_PRECHECK_HOOK",
        replace_named_inode,
    )
    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["execution"]["status"] == RunStatus.FAILED.value
    registry = app.extensions["literature_intake_service"].projects.read_artifact_registry(
        project_id,
        payload["intake"]["run_id"],
    )
    assert "parsed_document" not in registry
    assert "parser_audit" not in registry


def test_batch_parser_binds_each_approved_member_against_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "batch-pdf-inode-race"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    fired = False

    def replace_first_named_inode(path: Path) -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        path.rename(path.with_name(path.name + ".approved-inode"))
        path.write_bytes(b"%PDF-replacement-not-approved")

    monkeypatch.setattr(
        phase3_module,
        "_PINNED_PDF_AFTER_PRECHECK_HOOK",
        replace_first_named_inode,
    )
    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{submitted['intake']['intake_id']}/approve",
        json={"actor": "local-user"},
    )

    assert approved.status_code == 200
    assert approved.get_json()["execution"]["status"] == RunStatus.FAILED.value
    registry = app.extensions["literature_intake_service"].projects.read_artifact_registry(
        project_id,
        submitted["intake"]["run_id"],
    )
    assert "parsed_corpus_manifest" not in registry
    assert "parser_audit" not in registry


@pytest.mark.parametrize(
    ("artifact_id", "mutation"),
    [
        ("parsed_document_001", "replace"),
        ("parser_audit_001", "delete"),
        ("parsed_document_002", "symlink"),
    ],
)
def test_completed_batch_publication_fails_closed_after_member_output_tampering(
    tmp_path: Path,
    artifact_id: str,
    mutation: str,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = f"batch-output-{mutation}"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
        json={"actor": "local-user"},
    )
    assert approved.status_code == 200
    assert approved.get_json()["execution"]["status"] == RunStatus.SUCCEEDED.value
    storage = app.extensions["literature_intake_service"].projects
    registry = storage.read_artifact_registry(project_id, intake["run_id"])
    assert registry["parser_audit"] != registry["parsed_corpus_manifest"]
    assert set(registry) >= {
        "parsed_document_001",
        "parsed_document_002",
        "parser_audit_001",
        "parser_audit_002",
        "parser_audit",
    }
    target = storage.run_dir(project_id, intake["run_id"]) / registry[artifact_id]
    if mutation == "replace":
        target.write_bytes(b"replaced after successful publication")
    elif mutation == "delete":
        target.unlink()
    else:
        external = tmp_path / "external-output.json"
        external.write_text("{}", encoding="utf-8")
        target.unlink()
        target.symlink_to(external)

    fetched = client.get(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}"
    )
    assert fetched.status_code == 400


@pytest.mark.parametrize(
    ("artifact_id", "mutation"),
    [
        ("parsed_document", "replace"),
        ("parser_audit", "delete"),
        ("parsed_tables", "symlink"),
    ],
)
def test_completed_single_publication_fails_closed_after_output_tampering(
    tmp_path: Path,
    artifact_id: str,
    mutation: str,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = f"single-output-{mutation}"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert submitted.status_code == 201
    intake = submitted.get_json()["intake"]
    storage = app.extensions["literature_intake_service"].projects
    registry = storage.read_artifact_registry(project_id, intake["run_id"])
    target = storage.run_dir(project_id, intake["run_id"]) / registry[artifact_id]
    if mutation == "replace":
        target.write_bytes(b"replaced after successful publication")
    elif mutation == "delete":
        target.unlink()
    else:
        external = tmp_path / "external-single-output.json"
        external.write_text("{}", encoding="utf-8")
        target.unlink()
        target.symlink_to(external)

    fetched = client.get(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}"
    )
    repeated = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )
    assert fetched.status_code == 400
    assert repeated.status_code == 400


def test_batch_replay_rejects_member_and_manifest_resigning_without_stage_anchor(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "batch-manifest-resign"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
        json={"actor": "local-user"},
    )
    assert approved.status_code == 200
    storage = app.extensions["literature_intake_service"].projects
    run_dir = storage.run_dir(project_id, intake["run_id"])
    registry = storage.read_artifact_registry(project_id, intake["run_id"])
    manifest_path = run_dir / registry["parsed_corpus_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member_path = run_dir / registry["parsed_document_001"]
    member_path.write_bytes(b'{"resigned":"member"}\n')
    member_descriptor = manifest["documents"][0]["outputs"]["parsed_document"]
    member_descriptor["size_bytes"] = member_path.stat().st_size
    member_descriptor["sha256"] = hashlib.sha256(member_path.read_bytes()).hexdigest()
    audit_path = run_dir / registry["parser_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["document_outputs"] = manifest["documents"]
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    manifest["corpus_audit"]["size_bytes"] = audit_path.stat().st_size
    manifest["corpus_audit"]["sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    fetched = client.get(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}"
    )
    assert fetched.status_code == 400
    assert "StageState" in fetched.get_json()["error"] or "artifact content changed" in fetched.get_json()["error"]


def test_concurrent_duplicate_registration_executes_once_and_returns_same_intake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    setup_client = app.test_client()
    project_id = "concurrent-literature-registration"
    conversation_id = _create_conversation(setup_client, project_id)
    attachments = _upload_pdfs(setup_client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(setup_client, project_id, conversation_id, attachments)
    original_execute = RunPlanExecutor.execute
    count_lock = threading.Lock()
    execute_count = 0

    def counted_execute(self, **kwargs):
        nonlocal execute_count
        with count_lock:
            execute_count += 1
        time.sleep(0.1)
        return original_execute(self, **kwargs)

    monkeypatch.setattr(RunPlanExecutor, "execute", counted_execute)
    barrier = threading.Barrier(2)

    def submit() -> tuple[int, dict]:
        with app.test_client() as client:
            barrier.wait(timeout=5)
            response = client.post(
                f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
                json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
            )
            return response.status_code, response.get_json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))

    assert {status for status, _ in results} == {200, 201}
    assert execute_count == 1
    payloads = [payload for _, payload in results]
    assert {payload["intake"]["intake_id"] for payload in payloads} == {
        payloads[0]["intake"]["intake_id"]
    }
    assert {payload["intake"]["registered_at"] for payload in payloads} == {
        payloads[0]["intake"]["registered_at"]
    }
    assert {payload["idempotent"] for payload in payloads} == {False, True}


def test_concurrent_gate_approval_resumes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    setup_client = app.test_client()
    project_id = "concurrent-literature-approval"
    conversation_id = _create_conversation(setup_client, project_id)
    attachments = _upload_pdfs(setup_client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(setup_client, project_id, conversation_id, attachments)
    submitted = setup_client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    original_resume = RunPlanExecutor.resume_after_gate
    count_lock = threading.Lock()
    resume_count = 0

    def counted_resume(self, **kwargs):
        nonlocal resume_count
        with count_lock:
            resume_count += 1
        time.sleep(0.1)
        return original_resume(self, **kwargs)

    monkeypatch.setattr(RunPlanExecutor, "resume_after_gate", counted_resume)
    barrier = threading.Barrier(2)

    def approve() -> tuple[int, dict]:
        with app.test_client() as client:
            barrier.wait(timeout=5)
            response = client.post(
                f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
                json={"actor": "local-user"},
            )
            return response.status_code, response.get_json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: approve(), range(2)))

    assert [status for status, _ in results] == [200, 200]
    assert resume_count == 1
    assert {payload["idempotent"] for _, payload in results} == {False, True}
    storage = app.extensions["literature_intake_service"].projects
    assert len(storage.read_gate_decisions(project_id, intake["run_id"])) == 1


def test_single_parser_rejects_preexisting_output_symlink_without_external_writes(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "single-output-symlink"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    service = app.extensions["literature_intake_service"]
    intake_id = _intake_id(service, project_id, conversation_id, frozen, attachments)
    run_id = f"literature-parse-{intake_id.removeprefix('literature_intake_')}"
    run_dir = service.projects.run_dir(project_id, run_id)
    external = tmp_path / "external-single-parser"
    external.mkdir()
    (external / "sentinel.bin").write_bytes(b"unchanged")
    (run_dir / "parsed_document").symlink_to(external, target_is_directory=True)
    before = _directory_bytes(external)

    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )

    assert response.status_code == 201
    assert response.get_json()["execution"]["status"] == RunStatus.FAILED.value
    assert _directory_bytes(external) == before
    registry = service.projects.read_artifact_registry(project_id, run_id)
    assert "literature_parse_publication" not in registry
    assert "parsed_document" not in registry
    assert not (run_dir / "literature_parse_publication.json").exists()
    stage = service.projects.read_stage_state(project_id, run_id)
    assert stage is not None and stage.status == RunStatus.FAILED


def test_batch_parser_rejects_preexisting_output_symlink_without_external_writes(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "batch-output-symlink"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 2)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    submitted = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    ).get_json()
    intake = submitted["intake"]
    service = app.extensions["literature_intake_service"]
    run_dir = service.projects.run_dir(project_id, intake["run_id"])
    external = tmp_path / "external-batch-parser"
    external.mkdir()
    (external / "sentinel.bin").write_bytes(b"unchanged")
    (run_dir / "parsed_corpus").symlink_to(external, target_is_directory=True)
    before = _directory_bytes(external)

    approved = client.post(
        f"/api/projects/{project_id}/literature-intakes/{intake['intake_id']}/approve",
        json={"actor": "local-user"},
    )

    assert approved.status_code == 200
    assert approved.get_json()["execution"]["status"] == RunStatus.FAILED.value
    assert _directory_bytes(external) == before
    registry = service.projects.read_artifact_registry(project_id, intake["run_id"])
    assert "literature_parse_publication" not in registry
    assert "parsed_corpus_manifest" not in registry
    assert not (run_dir / "literature_parse_publication.json").exists()
    stage = service.projects.read_stage_state(project_id, intake["run_id"])
    assert stage is not None and stage.status == RunStatus.FAILED


def test_parser_output_directory_replacement_after_precheck_cannot_write_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    project_id = "single-output-directory-race"
    conversation_id = _create_conversation(client, project_id)
    attachments = _upload_pdfs(client, tmp_path, project_id, conversation_id, 1)
    frozen = _freeze_request(client, project_id, conversation_id, attachments)
    external = tmp_path / "external-output-race"
    external.mkdir()
    (external / "sentinel.bin").write_bytes(b"unchanged")
    before = _directory_bytes(external)
    fired = False

    def replace_named_output_directory(path: Path) -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        path.rename(path.with_name(path.name + ".pinned-inode"))
        path.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(
        phase3_module,
        "_PINNED_OUTPUT_AFTER_PRECHECK_HOOK",
        replace_named_output_directory,
    )
    response = client.post(
        f"/api/projects/{project_id}/conversations/{conversation_id}/literature-intakes",
        json={"request_id": frozen["request_id"], "parser_profile": "pdfplumber_local"},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["execution"]["status"] == RunStatus.FAILED.value
    assert _directory_bytes(external) == before
    service = app.extensions["literature_intake_service"]
    registry = service.projects.read_artifact_registry(project_id, payload["intake"]["run_id"])
    assert "literature_parse_publication" not in registry
    assert "parsed_document" not in registry
