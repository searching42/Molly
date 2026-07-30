from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import ai4s_agent.conversation_store as conversation_store_module
from ai4s_agent.app import create_app
from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.storage import ProjectStorage


def _store(tmp_path) -> ConversationStore:
    return ConversationStore(projects=ProjectStorage(workspace_dir=tmp_path))


def test_messages_jsonl_is_versioned_append_only_and_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    metadata, created = store.create_conversation(
        "project-a",
        title="OLED discussion",
        conversation_id="conversation-a",
    )
    assert created is True
    assert metadata.schema_version == "conversation.v1"

    first, idempotent, recovered = store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="Train a PLQY model.",
        client_message_id="browser-message-1",
    )
    repeated, repeated_idempotent, _ = store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="Train a PLQY model.",
        client_message_id="browser-message-1",
    )

    assert idempotent is False
    assert recovered is False
    assert repeated_idempotent is True
    assert repeated.message_id == first.message_id
    messages_path = (
        tmp_path / "projects" / "project-a" / "conversations" / "conversation-a" / "messages.jsonl"
    )
    lines = messages_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema_version"] == "conversation_message.v1"
    if os.name == "posix":
        assert stat.S_IMODE(messages_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(messages_path.parent.stat().st_mode) == 0o700

    with pytest.raises(ValueError, match="different content"):
        store.append_message(
            "project-a",
            "conversation-a",
            role="user",
            content="Changed content.",
            client_message_id="browser-message-1",
        )


def test_messages_jsonl_recovers_only_a_crashed_tail(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="complete",
    )
    path = tmp_path / "projects" / "project-a" / "conversations" / "conversation-a" / "messages.jsonl"
    valid_size = path.stat().st_size
    with path.open("ab") as output:
        output.write(b'{"schema_version":"conversation_message.v1","message_id":')

    messages, recovered = store.list_messages("project-a", "conversation-a")
    assert recovered is True
    assert [item.content for item in messages] == ["complete"]
    assert path.stat().st_size == valid_size

    valid_line = path.read_bytes()
    path.write_bytes(b"{not-json}\n" + valid_line)
    with pytest.raises(ValueError, match="corrupt complete record"):
        store.list_messages("project-a", "conversation-a")


def test_messages_jsonl_rejects_complete_corrupt_schema_and_sequence_records(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="complete",
    )
    path = tmp_path / "projects" / "project-a" / "conversations" / "conversation-a" / "messages.jsonl"
    original = path.read_bytes()

    path.write_bytes(original + b"{not-json}\n")
    with pytest.raises(ValueError, match="corrupt complete record"):
        store.list_messages("project-a", "conversation-a")
    assert path.read_bytes() == original + b"{not-json}\n"

    path.write_bytes(original + b"{not-json}")
    with pytest.raises(ValueError, match="corrupt complete record"):
        store.list_messages("project-a", "conversation-a")
    assert path.read_bytes() == original + b"{not-json}"

    first = json.loads(original.decode("utf-8"))
    invalid_schema = dict(first)
    invalid_schema.pop("role")
    invalid_schema["sequence"] = 2
    path.write_bytes(
        original + (json.dumps(invalid_schema, separators=(",", ":")) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError):
        store.list_messages("project-a", "conversation-a")

    invalid_sequence = dict(first)
    invalid_sequence["message_id"] = "msg_sequence_mismatch"
    invalid_sequence["sequence"] = 7
    path.write_bytes(
        original + (json.dumps(invalid_sequence, separators=(",", ":")) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match="sequence is not contiguous"):
        store.list_messages("project-a", "conversation-a")


def test_messages_jsonl_repairs_valid_record_missing_final_newline_before_append(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="first",
    )
    path = tmp_path / "projects" / "project-a" / "conversations" / "conversation-a" / "messages.jsonl"
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    first_read, recovered = store.list_messages("project-a", "conversation-a")
    assert recovered is True
    assert [item.content for item in first_read] == ["first"]
    assert path.read_bytes().endswith(b"\n")

    store.append_message(
        "project-a",
        "conversation-a",
        role="assistant",
        content="second",
    )
    messages, recovered_again = store.list_messages("project-a", "conversation-a")

    assert recovered_again is False
    assert [item.content for item in messages] == ["first", "second"]
    assert [item.sequence for item in messages] == [1, 2]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_conversation_files_are_bound_to_their_path_identity(tmp_path) -> None:
    store = _store(tmp_path)
    for conversation_id in ("conversation-a", "conversation-b"):
        store.create_conversation("project-a", conversation_id=conversation_id)
        store.append_message(
            "project-a",
            conversation_id,
            role="user",
            content=conversation_id,
        )

    root = tmp_path / "projects" / "project-a" / "conversations"
    metadata_a = root / "conversation-a" / "metadata.json"
    metadata_b = root / "conversation-b" / "metadata.json"
    original_metadata_a = metadata_a.read_bytes()
    metadata_a.write_bytes(metadata_b.read_bytes())
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        store.get_conversation("project-a", "conversation-a")
    metadata_a.write_bytes(original_metadata_a)

    messages_a = root / "conversation-a" / "messages.jsonl"
    messages_b = root / "conversation-b" / "messages.jsonl"
    messages_a.write_bytes(messages_b.read_bytes())
    with pytest.raises(ValueError, match="message identity mismatch"):
        store.list_messages("project-a", "conversation-a")


def test_concurrent_message_appends_have_unique_contiguous_sequences(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")

    def append(index: int) -> None:
        store.append_message(
            "project-a",
            "conversation-a",
            role="user",
            content=f"message-{index}",
            client_message_id=f"client-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))

    messages, recovered = store.list_messages("project-a", "conversation-a")
    assert recovered is False
    assert [item.sequence for item in messages] == list(range(1, 41))
    assert len({item.message_id for item in messages}) == 40


def test_delete_waits_for_inflight_append_and_archives_the_appended_message(
    tmp_path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    reached_append = threading.Barrier(2)
    release_append = threading.Event()
    original_append = store._append_json_line

    def delayed_append(path, payload) -> None:
        reached_append.wait(timeout=5)
        assert release_append.wait(timeout=5)
        original_append(path, payload)

    monkeypatch.setattr(store, "_append_json_line", delayed_append)
    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(
            store.append_message,
            "project-a",
            "conversation-a",
            role="user",
            content="must remain in the archived generation",
        )
        reached_append.wait(timeout=5)
        delete_future = executor.submit(store.delete_conversation, "project-a", "conversation-a")
        assert delete_future.done() is False
        release_append.set()
        append_future.result(timeout=5)
        delete_future.result(timeout=5)

    archived = list(
        (tmp_path / "projects" / "project-a" / "conversations" / ".deleted").glob(
            "conversation-a.*"
        )
    )
    assert len(archived) == 1
    assert b"must remain in the archived generation" in (archived[0] / "messages.jsonl").read_bytes()


def test_delete_serializes_recreate_and_rejects_stale_append(
    tmp_path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    reached_rename = threading.Barrier(2)
    release_rename = threading.Event()
    original_rename = conversation_store_module.os.rename

    def delayed_rename(source, target) -> None:
        if str(source).endswith("/conversation-a"):
            reached_rename.wait(timeout=5)
            assert release_rename.wait(timeout=5)
        original_rename(source, target)

    monkeypatch.setattr(conversation_store_module.os, "rename", delayed_rename)
    with ThreadPoolExecutor(max_workers=2) as executor:
        delete_future = executor.submit(store.delete_conversation, "project-a", "conversation-a")
        reached_rename.wait(timeout=5)
        recreate_future = executor.submit(
            store.create_conversation,
            "project-a",
            conversation_id="conversation-a",
        )
        assert recreate_future.done() is False
        release_rename.set()
        delete_future.result(timeout=5)
        with pytest.raises(ValueError, match="deleted conversation_id cannot be reused"):
            recreate_future.result(timeout=5)

    with pytest.raises(FileNotFoundError, match="conversation not found"):
        store.append_message(
            "project-a",
            "conversation-a",
            role="user",
            content="stale request",
        )
    active = tmp_path / "projects" / "project-a" / "conversations" / "conversation-a"
    assert not active.exists()


def test_attachment_artifacts_are_content_addressed_and_message_refs_have_no_paths(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    first = store.register_attachment(
        "project-a",
        stream=io.BytesIO(b"paper bytes"),
        original_name="paper.pdf",
        media_type="application/pdf",
        max_bytes=1024,
    )
    duplicate = store.register_attachment(
        "project-a",
        stream=io.BytesIO(b"paper bytes"),
        original_name="copy.pdf",
        media_type="application/pdf",
        max_bytes=1024,
    )
    assert duplicate.artifact_id == first.artifact_id

    message, _, _ = store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="Read the attached paper.",
        attachment_ids=[first.artifact_id],
    )
    serialized = message.model_dump(mode="json")
    assert serialized["attachments"][0]["sha256"] == first.sha256
    assert "path" not in serialized["attachments"][0]
    content_path = store.resolve_attachment_path("project-a", first.artifact_id)
    assert content_path.read_bytes() == b"paper bytes"
    assert first.sha256 in str(content_path)


def test_attachment_size_and_hash_are_verified(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="size limit"):
        store.register_attachment(
            "project-a",
            stream=io.BytesIO(b"too large"),
            original_name="large.bin",
            max_bytes=3,
        )
    assert list((tmp_path / "projects" / "project-a" / "artifacts" / "conversation_attachments" / ".staging").iterdir()) == []

    attachment = store.register_attachment(
        "project-a",
        stream=io.BytesIO(b"verified"),
        original_name="verified.txt",
        max_bytes=100,
    )
    store.resolve_attachment_path("project-a", attachment.artifact_id).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.resolve_attachment("project-a", attachment.artifact_id)


def test_local_storage_import_is_idempotent_and_content_bound(tmp_path) -> None:
    store = _store(tmp_path)
    payload = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    first = store.import_local_storage(
        "project-a",
        import_id="localstorage:stable-digest",
        messages=payload,
        title="Legacy chat",
    )
    second = store.import_local_storage(
        "project-a",
        import_id="localstorage:stable-digest",
        messages=payload,
        title="Legacy chat",
    )
    messages, _ = store.list_messages("project-a", first["conversation_id"])

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["message_ids"] == first["message_ids"]
    assert len(messages) == 2
    assert all(item.import_id == "localstorage:stable-digest" for item in messages)
    with pytest.raises(ValueError, match="different localStorage content"):
        store.import_local_storage(
            "project-a",
            import_id="localstorage:stable-digest",
            messages=[{"role": "user", "content": "changed"}],
            title="Legacy chat",
        )


def test_local_storage_import_resumes_after_partial_failure_without_duplicates(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    payload = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    original_append = store.append_message
    calls = 0

    def crash_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated process failure")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(store, "append_message", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated process failure"):
        store.import_local_storage(
            "project-a",
            import_id="localstorage:resumable",
            messages=payload,
        )
    monkeypatch.setattr(store, "append_message", original_append)
    resumed = store.import_local_storage(
        "project-a",
        import_id="localstorage:resumable",
        messages=payload,
    )
    repeated = store.import_local_storage(
        "project-a",
        import_id="localstorage:resumable",
        messages=payload,
    )
    messages, _ = store.list_messages("project-a", resumed["conversation_id"])

    assert resumed["status"] == "complete"
    assert resumed["imported_count"] == 2
    assert repeated["idempotent"] is True
    assert [item.content for item in messages] == ["first", "second"]


def test_execution_request_freezes_messages_and_verified_attachment_refs(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_conversation("project-a", conversation_id="conversation-a")
    attachment = store.register_attachment(
        "project-a",
        stream=io.BytesIO(b"dataset"),
        original_name="dataset.csv",
        media_type="text/csv",
        max_bytes=1024,
    )
    first, _, _ = store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="Train from this dataset.",
        attachment_ids=[attachment.artifact_id],
    )
    frozen = store.freeze_execution_request(
        "project-a",
        "conversation-a",
        selected_message_ids=[first.message_id],
        task_type="train_model",
        model_profile_id="default",
        user_parameters={"target": "plqy"},
        client_request_id="submit-1",
    )
    store.append_message(
        "project-a",
        "conversation-a",
        role="user",
        content="This later message must not alter the request.",
    )
    replay = store.get_frozen_execution_request(
        "project-a",
        "conversation-a",
        frozen.request_id,
    )
    repeated = store.freeze_execution_request(
        "project-a",
        "conversation-a",
        selected_message_ids=[first.message_id],
        task_type="train_model",
        model_profile_id="default",
        user_parameters={"target": "plqy"},
        client_request_id="submit-1",
    )

    assert replay.request_sha256 == frozen.request_sha256 == repeated.request_sha256
    assert [item.message_id for item in replay.selected_messages] == [first.message_id]
    assert replay.attachments[0].artifact_id == attachment.artifact_id
    assert replay.executable is False
    with pytest.raises(ValueError, match="different execution request"):
        store.freeze_execution_request(
            "project-a",
            "conversation-a",
            selected_message_ids=[first.message_id],
            task_type="train_model",
            model_profile_id="default",
            user_parameters={"target": "different"},
            client_request_id="submit-1",
        )
    with pytest.raises(ValueError, match="sensitive credential material"):
        store.freeze_execution_request(
            "project-a",
            "conversation-a",
            selected_message_ids=[first.message_id],
            task_type="train_model",
            model_profile_id="default",
            user_parameters={"api_key": "must-not-freeze"},
        )
    token_budget_request = store.freeze_execution_request(
        "project-a",
        "conversation-a",
        selected_message_ids=[first.message_id],
        task_type="train_model",
        model_profile_id="default",
        user_parameters={"token_budget": 2048, "dataset_id": attachment.artifact_id},
    )
    assert token_budget_request.user_parameters["token_budget"] == 2048
    with pytest.raises(ValueError, match="attachment artifact reference"):
        store.freeze_execution_request(
            "project-a",
            "conversation-a",
            selected_message_ids=[first.message_id],
            task_type="train_model",
            model_profile_id="default",
            user_parameters={"raw_dataset": [{"smiles": "CCO"}]},
        )


def test_frozen_execution_request_is_bound_to_its_path_identity(tmp_path) -> None:
    store = _store(tmp_path)
    frozen_by_conversation = {}
    for conversation_id in ("conversation-a", "conversation-b"):
        store.create_conversation("project-a", conversation_id=conversation_id)
        message, _, _ = store.append_message(
            "project-a",
            conversation_id,
            role="user",
            content=conversation_id,
        )
        frozen_by_conversation[conversation_id] = store.freeze_execution_request(
            "project-a",
            conversation_id,
            selected_message_ids=[message.message_id],
            task_type="literature_parse",
            model_profile_id="default",
        )

    first = frozen_by_conversation["conversation-a"]
    second = frozen_by_conversation["conversation-b"]
    root = tmp_path / "projects" / "project-a" / "conversations"
    first_path = root / "conversation-a" / "execution_requests" / f"{first.request_id}.json"
    second_path = root / "conversation-b" / "execution_requests" / f"{second.request_id}.json"
    first_path.write_bytes(second_path.read_bytes())

    with pytest.raises(ValueError, match="execution request identity mismatch"):
        store.get_frozen_execution_request(
            "project-a",
            "conversation-a",
            first.request_id,
        )


def test_conversation_api_supports_batch_attachments_import_and_freeze(tmp_path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()
    created = client.post(
        "/api/projects/project-a/conversations",
        json={"conversation_id": "conversation-a", "title": "API chat"},
    )
    assert created.status_code == 201
    uploaded = client.post(
        "/api/projects/project-a/conversations/conversation-a/attachments",
        data={
            "files": [
                (io.BytesIO(b"pdf"), "paper.pdf"),
                (io.BytesIO(b"csv"), "dataset.csv"),
            ]
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    attachments = uploaded.json["attachments"]
    assert len(attachments) == 2
    assert all("path" not in item for item in attachments)

    appended = client.post(
        "/api/projects/project-a/conversations/conversation-a/messages",
        json={
            "role": "user",
            "content": "Use both files.",
            "attachment_ids": [item["artifact_id"] for item in attachments],
            "client_message_id": "api-message-1",
        },
    )
    assert appended.status_code == 201
    message_id = appended.json["message"]["message_id"]
    frozen = client.post(
        "/api/projects/project-a/conversations/conversation-a/execution-requests",
        json={
            "selected_message_ids": [message_id],
            "task_type": "literature_parse",
            "model_profile_id": "default",
            "user_parameters": {"mode": "review_only"},
            "client_request_id": "api-freeze-1",
        },
    )
    assert frozen.status_code == 201
    assert frozen.json["execution_request"]["status"] == "frozen"
    assert frozen.json["execution_request"]["executable"] is False
    assert frozen.headers["Cache-Control"] == "no-store"

    imported = client.post(
        "/api/projects/project-a/conversations/import-local-storage",
        json={
            "import_id": "localstorage:api-import",
            "title": "Imported via API",
            "messages": [{"role": "user", "content": "legacy"}],
        },
    )
    repeated = client.post(
        "/api/projects/project-a/conversations/import-local-storage",
        json={
            "import_id": "localstorage:api-import",
            "title": "Imported via API",
            "messages": [{"role": "user", "content": "legacy"}],
        },
    )
    assert imported.status_code == 200
    assert imported.json["idempotent"] is False
    assert repeated.json["idempotent"] is True


def test_conversation_delete_archives_record_and_api_removes_it_from_active_list(
    tmp_path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()
    created = client.post(
        "/api/projects/project-a/conversations",
        json={"conversation_id": "conversation-delete", "title": "Delete me"},
    )
    assert created.status_code == 201
    appended = client.post(
        "/api/projects/project-a/conversations/conversation-delete/messages",
        json={"role": "user", "content": "retained in recoverable archive"},
    )
    assert appended.status_code == 201

    deleted = client.delete(
        "/api/projects/project-a/conversations/conversation-delete"
    )

    assert deleted.status_code == 200
    assert deleted.json == {
        "ok": True,
        "deleted": True,
        "conversation_id": "conversation-delete",
    }
    assert deleted.headers["Cache-Control"] == "no-store"
    listing = client.get("/api/projects/project-a/conversations")
    assert listing.json["conversations"] == []
    assert client.get(
        "/api/projects/project-a/conversations/conversation-delete"
    ).status_code == 404
    archived = list(
        (tmp_path / "projects" / "project-a" / "conversations" / ".deleted").glob(
            "conversation-delete.*"
        )
    )
    assert len(archived) == 1
    assert b"retained in recoverable archive" in (
        archived[0] / "messages.jsonl"
    ).read_bytes()

    repeated = client.delete(
        "/api/projects/project-a/conversations/conversation-delete"
    )
    assert repeated.status_code == 404
    assert repeated.json == {"ok": False, "error": "conversation not found"}


def test_batch_attachment_limit_rejects_file_after_budget_is_exhausted(tmp_path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )
    app.config["AI4S_MAX_CONVERSATION_ATTACHMENT_BYTES"] = 3
    client = app.test_client()
    created = client.post(
        "/api/projects/project-a/conversations",
        json={"conversation_id": "conversation-a"},
    )
    assert created.status_code == 201

    uploaded = client.post(
        "/api/projects/project-a/conversations/conversation-a/attachments",
        data={
            "files": [
                (io.BytesIO(b"abc"), "exact.bin"),
                (io.BytesIO(b"x"), "overflow.bin"),
            ]
        },
        content_type="multipart/form-data",
    )

    assert uploaded.status_code == 413
    assert "size limit: 0 bytes" in uploaded.json["error"]


def test_browser_ui_uses_server_conversations_and_idempotent_local_storage_import(tmp_path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'id="conversation-file-input"' in html
    assert 'id="conversation-upload-button"' in html
    assert "/conversations/import-local-storage" in html
    assert "localStorage.removeItem(key)" in html
    assert "/conversations/${encodeURIComponent(conversationId)}/messages" in html
    assert "attachment_ids: attachmentIds" in html
    assert "async function deleteConversation(projectId, conversationId, title)" in html
    assert 'method: "DELETE"' in html


@pytest.mark.pr_fast
def test_browser_delete_decision_uses_post_response_conversation_and_pending_load(tmp_path) -> None:
    node_binary = shutil.which("node")
    if node_binary is None:
        pytest.skip("Node.js is unavailable for the executable UI contract test")
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )
    html = app.test_client().get("/").get_data(as_text=True)
    function_start = html.index("function conversationDeletionDecision(")
    function_end = html.index("function renderProjectList(", function_start)
    function_source = html[function_start:function_end]
    script = f"""
{function_source}
const results = {{
  currentToOther: conversationDeletionDecision(
    "project-a", "conversation-a", "project-a", "conversation-c",
    {{ projectId: "project-a", conversationId: "conversation-c", generation: 9 }},
  ),
  otherToDeleted: conversationDeletionDecision(
    "project-a", "conversation-a", "project-a", "conversation-a",
    {{ projectId: "project-a", conversationId: "conversation-a", generation: 10 }},
  ),
  pendingDeletedWhileOtherRemainsCurrent: conversationDeletionDecision(
    "project-a", "conversation-a", "project-a", "conversation-b",
    {{ projectId: "project-a", conversationId: "conversation-a", generation: 11 }},
  ),
}};
process.stdout.write(JSON.stringify(results));
"""
    completed = subprocess.run(
        [node_binary],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "currentToOther": {
            "sameProject": True,
            "deletedIsCurrent": False,
            "cancelPendingLoad": False,
        },
        "otherToDeleted": {
            "sameProject": True,
            "deletedIsCurrent": True,
            "cancelPendingLoad": True,
        },
        "pendingDeletedWhileOtherRemainsCurrent": {
            "sameProject": True,
            "deletedIsCurrent": False,
            "cancelPendingLoad": True,
        },
    }
    assert "deletedConversationKeys.has(conversationIdentityKey(projectId, nextConversationId))" in html


@pytest.mark.parametrize("project_id,conversation_id", [("../escape", "safe"), ("safe", "../escape")])
def test_conversation_paths_reject_traversal(tmp_path, project_id: str, conversation_id: str) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create_conversation(project_id, conversation_id=conversation_id)
