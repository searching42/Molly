from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

try:  # pragma: no cover - POSIX CI exercises the primary branch.
    import fcntl
except ImportError:  # pragma: no cover - process-local locks preserve portability.
    fcntl = None  # type: ignore[assignment]

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import (
    ConversationAttachmentManifest,
    ConversationAttachmentRef,
    ConversationMessage,
    ConversationMetadata,
    FrozenConversationExecutionRequest,
)
from ai4s_agent.storage import ProjectStorage


_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)
_COPY_CHUNK_BYTES = 1024 * 1024


class ConversationStore:
    """Append-only UI conversation storage, separate from scientific run state."""

    def __init__(self, *, projects: ProjectStorage) -> None:
        self.projects = projects

    def create_conversation(
        self,
        project_id: str,
        *,
        title: str = "New conversation",
        conversation_id: str = "",
    ) -> tuple[ConversationMetadata, bool]:
        clean_id = self._clean_id(conversation_id or f"conv_{uuid.uuid4().hex}", "conversation_id")
        directory = self._conversation_dir(project_id, clean_id)
        self._ensure_private_directory(directory)
        with self._directory_lock(directory):
            metadata_path = directory / "metadata.json"
            if metadata_path.exists():
                metadata = self._read_metadata(
                    metadata_path,
                    expected_project_id=project_id,
                    expected_conversation_id=clean_id,
                )
                return metadata, False
            timestamp = now_iso()
            metadata = ConversationMetadata(
                project_id=project_id,
                conversation_id=clean_id,
                title=self._clean_title(title),
                created_at=timestamp,
                updated_at=timestamp,
            )
            write_json(metadata_path, metadata.model_dump(mode="json"))
            messages_path = self._messages_path(directory)
            messages_path.touch(mode=0o600, exist_ok=True)
            os.chmod(messages_path, 0o600)
            return metadata, True

    def list_conversations(self, project_id: str) -> list[ConversationMetadata]:
        root = self._conversations_root(project_id)
        if not root.exists():
            return []
        result: list[ConversationMetadata] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            metadata_path = child / "metadata.json"
            if metadata_path.exists():
                result.append(
                    self._read_metadata(
                        metadata_path,
                        expected_project_id=project_id,
                        expected_conversation_id=child.name,
                    )
                )
        return sorted(result, key=lambda item: (item.updated_at, item.conversation_id), reverse=True)

    def get_conversation(self, project_id: str, conversation_id: str) -> ConversationMetadata:
        metadata_path = self._conversation_dir(project_id, conversation_id) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("conversation not found")
        return self._read_metadata(
            metadata_path,
            expected_project_id=project_id,
            expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
        )

    def list_messages(
        self,
        project_id: str,
        conversation_id: str,
    ) -> tuple[list[ConversationMessage], bool]:
        directory = self._existing_conversation_dir(project_id, conversation_id)
        with self._directory_lock(directory):
            return self._read_messages_locked(
                self._messages_path(directory),
                expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
            )

    def append_message(
        self,
        project_id: str,
        conversation_id: str,
        *,
        role: str,
        content: str = "",
        attachment_ids: list[str] | None = None,
        client_message_id: str = "",
        import_id: str = "",
        created_at: str = "",
    ) -> tuple[ConversationMessage, bool, bool]:
        directory = self._existing_conversation_dir(project_id, conversation_id)
        attachments = [
            self.resolve_attachment(project_id, artifact_id)
            for artifact_id in self._dedupe_strings(attachment_ids or [])
        ]
        clean_client_id = str(client_message_id or "").strip()
        with self._directory_lock(directory):
            messages, recovered_tail = self._read_messages_locked(
                self._messages_path(directory),
                expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
            )
            if clean_client_id:
                existing = next(
                    (item for item in messages if item.client_message_id == clean_client_id),
                    None,
                )
                if existing is not None:
                    expected_ids = [item.artifact_id for item in attachments]
                    actual_ids = [item.artifact_id for item in existing.attachments]
                    if (
                        existing.role != role
                        or existing.content != str(content or "")
                        or actual_ids != expected_ids
                    ):
                        raise ValueError("client_message_id already belongs to different content")
                    return existing, True, recovered_tail
            message = ConversationMessage(
                message_id=f"msg_{uuid.uuid4().hex}",
                conversation_id=conversation_id,
                sequence=len(messages) + 1,
                role=role,
                content=str(content or ""),
                attachments=attachments,
                created_at=str(created_at or "").strip() or now_iso(),
                client_message_id=clean_client_id,
                import_id=str(import_id or "").strip(),
            )
            self._append_json_line(
                self._messages_path(directory),
                message.model_dump(mode="json"),
            )
            metadata = self._read_metadata(
                directory / "metadata.json",
                expected_project_id=project_id,
                expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
            )
            updated = metadata.model_copy(update={"updated_at": now_iso()})
            write_json(directory / "metadata.json", updated.model_dump(mode="json"))
            return message, False, recovered_tail

    def register_attachment(
        self,
        project_id: str,
        *,
        stream: BinaryIO,
        original_name: str,
        media_type: str = "application/octet-stream",
        max_bytes: int,
    ) -> ConversationAttachmentRef:
        project = self.projects.project_dir(project_id)
        artifacts_root = self._artifacts_root(project_id)
        staging = (artifacts_root / ".staging").resolve()
        self._ensure_relative(artifacts_root, staging, "attachment staging")
        self._ensure_private_directory(staging)
        fd, temp_name = tempfile.mkstemp(prefix="upload-", suffix=".tmp", dir=staging)
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if max_bytes >= 0 and size > max_bytes:
                        raise ValueError(f"attachment exceeds size limit: {max_bytes} bytes")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha256 = digest.hexdigest()
            artifact_id = f"attachment_{sha256}"
            target_dir = (artifacts_root / sha256[:2] / sha256).resolve()
            self._ensure_relative(artifacts_root, target_dir, "attachment artifact")
            self._ensure_private_directory(target_dir)
            content_path = (target_dir / "content").resolve()
            self._ensure_relative(target_dir, content_path, "attachment content")
            try:
                os.link(temp_path, content_path)
                self._fsync_directory(target_dir)
            except FileExistsError:
                self._verify_file(content_path, expected_sha256=sha256, expected_size=size)
            manifest_path = target_dir / "manifest.json"
            if manifest_path.exists():
                manifest = self._read_attachment_manifest(manifest_path)
                if manifest.sha256 != sha256 or manifest.size_bytes != size:
                    raise ValueError("attachment artifact manifest conflicts with stored content")
            else:
                manifest = ConversationAttachmentManifest(
                    artifact_id=artifact_id,
                    original_name=self._clean_original_name(original_name),
                    sha256=sha256,
                    media_type=self._clean_media_type(media_type),
                    size_bytes=size,
                    created_at=now_iso(),
                )
                try:
                    self._write_json_exclusive(
                        manifest_path,
                        manifest.model_dump(mode="json"),
                    )
                except FileExistsError:
                    manifest = self._read_attachment_manifest(manifest_path)
            self._ensure_relative(project, manifest_path.resolve(), "attachment manifest")
            return manifest.as_reference()
        finally:
            temp_path.unlink(missing_ok=True)

    def resolve_attachment(
        self,
        project_id: str,
        artifact_id: str,
    ) -> ConversationAttachmentRef:
        clean_id = str(artifact_id or "").strip()
        if not clean_id.startswith("attachment_"):
            raise ValueError("invalid conversation attachment artifact_id")
        sha256 = clean_id.removeprefix("attachment_")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("invalid conversation attachment artifact_id")
        directory = (self._artifacts_root(project_id) / sha256[:2] / sha256).resolve()
        self._ensure_relative(self._artifacts_root(project_id), directory, "attachment artifact")
        manifest = self._read_attachment_manifest(directory / "manifest.json")
        if manifest.artifact_id != clean_id or manifest.sha256 != sha256:
            raise ValueError("attachment artifact identity mismatch")
        self._verify_file(
            directory / "content",
            expected_sha256=manifest.sha256,
            expected_size=manifest.size_bytes,
        )
        return manifest.as_reference()

    def resolve_attachment_path(self, project_id: str, artifact_id: str) -> Path:
        reference = self.resolve_attachment(project_id, artifact_id)
        path = (
            self._artifacts_root(project_id)
            / reference.sha256[:2]
            / reference.sha256
            / "content"
        ).resolve()
        self._ensure_relative(self._artifacts_root(project_id), path, "attachment content")
        return path

    def import_local_storage(
        self,
        project_id: str,
        *,
        import_id: str,
        messages: list[dict[str, Any]],
        title: str = "Imported conversation",
        conversation_id: str = "",
    ) -> dict[str, Any]:
        clean_import_id = str(import_id or "").strip()
        if not clean_import_id or len(clean_import_id) > 256:
            raise ValueError("import_id is required and must be at most 256 characters")
        if not isinstance(messages, list) or len(messages) > 5000:
            raise ValueError("localStorage messages must be a list of at most 5000 entries")
        source_payload = {
            "import_id": clean_import_id,
            "messages": messages,
            "title": str(title or ""),
            "conversation_id": str(conversation_id or ""),
        }
        source_digest = self._sha256_json(source_payload)
        import_key = hashlib.sha256(clean_import_id.encode("utf-8")).hexdigest()
        imports_dir = self._imports_root(project_id)
        self._ensure_private_directory(imports_dir)
        receipt_path = imports_dir / f"{import_key}.json"
        with self._directory_lock(imports_dir, lock_name=".imports.lock"):
            if receipt_path.exists():
                receipt = self._read_json_object(receipt_path)
                if receipt.get("source_sha256") != source_digest:
                    raise ValueError("import_id already belongs to different localStorage content")
                if receipt.get("status") == "complete":
                    return {**receipt, "idempotent": True}
                selected_conversation_id = str(receipt.get("conversation_id") or "")
            else:
                derived_id = f"conv_import_{import_key[:24]}"
                selected_conversation_id = conversation_id or derived_id
                write_json(
                    receipt_path,
                    {
                        "schema_version": "local_storage_conversation_import_receipt.v1",
                        "status": "importing",
                        "import_id": clean_import_id,
                        "source_sha256": source_digest,
                        "project_id": project_id,
                        "conversation_id": selected_conversation_id,
                        "message_ids": [],
                        "imported_count": 0,
                        "started_at": now_iso(),
                    },
                )
            metadata, _ = self.create_conversation(
                project_id,
                title=title,
                conversation_id=selected_conversation_id,
            )
            imported: list[ConversationMessage] = []
            for index, raw in enumerate(messages):
                if not isinstance(raw, dict):
                    raise ValueError("localStorage message entries must be objects")
                attachment_ids = self._attachment_ids_from_import(raw)
                message, _, _ = self.append_message(
                    project_id,
                    metadata.conversation_id,
                    role=str(raw.get("role") or ""),
                    content=str(raw.get("content") or ""),
                    attachment_ids=attachment_ids,
                    client_message_id=f"import:{import_key[:20]}:{index}",
                    import_id=clean_import_id,
                    created_at=str(raw.get("created_at") or ""),
                )
                imported.append(message)
            receipt = {
                "schema_version": "local_storage_conversation_import_receipt.v1",
                "status": "complete",
                "import_id": clean_import_id,
                "source_sha256": source_digest,
                "project_id": project_id,
                "conversation_id": metadata.conversation_id,
                "message_ids": [item.message_id for item in imported],
                "imported_count": len(imported),
                "imported_at": now_iso(),
                "idempotent": False,
            }
            write_json(receipt_path, receipt)
            return receipt

    def freeze_execution_request(
        self,
        project_id: str,
        conversation_id: str,
        *,
        selected_message_ids: list[str],
        task_type: str,
        model_profile_id: str,
        user_parameters: dict[str, Any] | None = None,
        client_request_id: str = "",
    ) -> FrozenConversationExecutionRequest:
        directory = self._existing_conversation_dir(project_id, conversation_id)
        clean_message_ids = self._dedupe_strings(selected_message_ids)
        if not clean_message_ids:
            raise ValueError("selected_message_ids are required")
        clean_task_type = str(task_type or "").strip()
        clean_profile = str(model_profile_id or "").strip()
        if not clean_task_type or not clean_profile:
            raise ValueError("task_type and model_profile_id are required")
        parameters = dict(user_parameters or {})
        with self._directory_lock(directory):
            messages, _ = self._read_messages_locked(
                self._messages_path(directory),
                expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
            )
            by_id = {item.message_id: item for item in messages}
            missing = [message_id for message_id in clean_message_ids if message_id not in by_id]
            if missing:
                raise ValueError("selected conversation messages not found: " + ", ".join(missing))
            selected = [by_id[message_id] for message_id in clean_message_ids]
            attachments_by_id: dict[str, ConversationAttachmentRef] = {}
            for message in selected:
                for attachment in message.attachments:
                    verified = self.resolve_attachment(project_id, attachment.artifact_id)
                    if verified.sha256 != attachment.sha256:
                        raise ValueError("conversation attachment digest changed before freeze")
                    attachments_by_id[verified.artifact_id] = verified
            clean_client_request_id = str(client_request_id or "").strip()
            if clean_client_request_id:
                request_key = hashlib.sha256(
                    f"{project_id}\0{conversation_id}\0{clean_client_request_id}".encode("utf-8")
                ).hexdigest()
                request_id = f"conversation_request_{request_key[:24]}"
            else:
                request_id = f"conversation_request_{uuid.uuid4().hex}"
            requests_dir = (directory / "execution_requests").resolve()
            self._ensure_relative(directory, requests_dir, "conversation execution requests")
            self._ensure_private_directory(requests_dir)
            request_path = (requests_dir / f"{request_id}.json").resolve()
            self._ensure_relative(requests_dir, request_path, "conversation execution request")
            intent = {
                "project_id": project_id,
                "conversation_id": conversation_id,
                "task_type": clean_task_type,
                "model_profile_id": clean_profile,
                "selected_message_ids": clean_message_ids,
                "user_parameters": parameters,
            }
            if request_path.exists():
                existing = FrozenConversationExecutionRequest.model_validate(
                    self._read_json_object(request_path)
                )
                self._verify_frozen_request(
                    existing,
                    expected_project_id=project_id,
                    expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
                    expected_request_id=request_id,
                )
                existing_intent = {
                    "project_id": existing.project_id,
                    "conversation_id": existing.conversation_id,
                    "task_type": existing.task_type,
                    "model_profile_id": existing.model_profile_id,
                    "selected_message_ids": existing.selected_message_ids,
                    "user_parameters": existing.user_parameters,
                }
                if existing_intent != intent:
                    raise ValueError("client_request_id already belongs to a different execution request")
                return existing
            request_payload = {
                "schema_version": "conversation_execution_request.v1",
                "request_id": request_id,
                **intent,
                "selected_messages": [item.model_dump(mode="json") for item in selected],
                "attachments": [
                    attachments_by_id[key].model_dump(mode="json")
                    for key in sorted(attachments_by_id)
                ],
                "frozen_at": now_iso(),
                "status": "frozen",
                "executable": False,
            }
            request_payload["request_sha256"] = self._sha256_json(request_payload)
            frozen = FrozenConversationExecutionRequest.model_validate(request_payload)
            self._write_json_exclusive(request_path, frozen.model_dump(mode="json"))
            return frozen

    def get_frozen_execution_request(
        self,
        project_id: str,
        conversation_id: str,
        request_id: str,
    ) -> FrozenConversationExecutionRequest:
        directory = self._existing_conversation_dir(project_id, conversation_id)
        clean_request_id = self._clean_id(request_id, "request_id")
        requests_dir = (directory / "execution_requests").resolve()
        path = (requests_dir / f"{clean_request_id}.json").resolve()
        self._ensure_relative(requests_dir, path, "conversation execution request")
        if not path.exists():
            raise FileNotFoundError("conversation execution request not found")
        frozen = FrozenConversationExecutionRequest.model_validate(self._read_json_object(path))
        self._verify_frozen_request(
            frozen,
            expected_project_id=project_id,
            expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
            expected_request_id=clean_request_id,
        )
        return frozen

    def _verify_frozen_request(
        self,
        request: FrozenConversationExecutionRequest,
        *,
        expected_project_id: str,
        expected_conversation_id: str,
        expected_request_id: str,
    ) -> None:
        if (
            request.project_id != expected_project_id
            or request.conversation_id != expected_conversation_id
            or request.request_id != expected_request_id
        ):
            raise ValueError("conversation execution request identity mismatch")
        payload = request.model_dump(mode="json")
        claimed = str(payload.pop("request_sha256"))
        if self._sha256_json(payload) != claimed:
            raise ValueError("conversation execution request digest mismatch")
        for attachment in request.attachments:
            verified = self.resolve_attachment(request.project_id, attachment.artifact_id)
            if verified.sha256 != attachment.sha256 or verified.size_bytes != attachment.size_bytes:
                raise ValueError("frozen conversation attachment no longer matches artifact")

    def _read_messages_locked(
        self,
        path: Path,
        *,
        expected_conversation_id: str,
    ) -> tuple[list[ConversationMessage], bool]:
        if not path.exists():
            path.touch(mode=0o600, exist_ok=True)
            os.chmod(path, 0o600)
            return [], False
        os.chmod(path, 0o600)
        data = path.read_bytes()
        if not data:
            return [], False
        lines = data.splitlines(keepends=True)
        messages: list[ConversationMessage] = []
        offset = 0
        recovered = False
        for index, raw in enumerate(lines):
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                is_unterminated_tail = (
                    index == len(lines) - 1 and not raw.endswith((b"\n", b"\r"))
                )
                if not is_unterminated_tail or exc.reason != "unexpected end of data":
                    raise ValueError(
                        f"messages.jsonl contains a corrupt complete record at line {index + 1}"
                    ) from exc
                self._truncate_crashed_tail(path, offset)
                recovered = True
                break
            try:
                loaded = json.loads(decoded)
            except json.JSONDecodeError as exc:
                is_unterminated_tail = (
                    index == len(lines) - 1 and not raw.endswith((b"\n", b"\r"))
                )
                if not is_unterminated_tail or not self._json_error_is_incomplete(decoded, exc):
                    raise ValueError(
                        f"messages.jsonl contains a corrupt complete record at line {index + 1}"
                    ) from exc
                self._truncate_crashed_tail(path, offset)
                recovered = True
                break
            message = ConversationMessage.model_validate(loaded)
            if message.conversation_id != expected_conversation_id:
                raise ValueError("conversation message identity mismatch")
            if message.sequence != len(messages) + 1:
                raise ValueError("conversation message sequence is not contiguous")
            messages.append(message)
            offset += len(raw)
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                self._append_record_terminator(path)
                recovered = True
        return messages, recovered

    @staticmethod
    def _json_error_is_incomplete(decoded: str, error: json.JSONDecodeError) -> bool:
        stripped_length = len(decoded.rstrip())
        return error.msg.startswith("Unterminated string") or error.pos >= stripped_length

    @staticmethod
    def _truncate_crashed_tail(path: Path, offset: int) -> None:
        with path.open("r+b") as output:
            output.truncate(offset)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _append_record_terminator(path: Path) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            written = os.write(fd, b"\n")
            if written != 1:
                raise OSError("could not repair conversation record terminator")
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:  # pragma: no cover - Windows compatibility.
                os.chmod(path, 0o600)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("could not append conversation message")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.link(temp_path, path)
            ConversationStore._fsync_directory(path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    @contextmanager
    def _directory_lock(self, directory: Path, *, lock_name: str = ".conversation.lock") -> Iterator[None]:
        self._ensure_private_directory(directory)
        lock_path = (directory / lock_name).resolve()
        self._ensure_relative(directory, lock_path, "conversation lock")
        with _LOCKS[str(lock_path)]:
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _conversation_dir(self, project_id: str, conversation_id: str) -> Path:
        root = self._conversations_root(project_id)
        clean_id = self._clean_id(conversation_id, "conversation_id")
        directory = (root / clean_id).resolve()
        self._ensure_relative(root, directory, "conversation_id")
        return directory

    def _existing_conversation_dir(self, project_id: str, conversation_id: str) -> Path:
        directory = self._conversation_dir(project_id, conversation_id)
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("conversation not found")
        self._read_metadata(
            metadata_path,
            expected_project_id=project_id,
            expected_conversation_id=self._clean_id(conversation_id, "conversation_id"),
        )
        return directory

    def _conversations_root(self, project_id: str) -> Path:
        project = self.projects.project_dir(project_id)
        root = (project / "conversations").resolve()
        self._ensure_relative(project, root, "conversations")
        return root

    def _artifacts_root(self, project_id: str) -> Path:
        project = self.projects.project_dir(project_id)
        root = (project / "artifacts" / "conversation_attachments").resolve()
        self._ensure_relative(project, root, "conversation attachments")
        self._ensure_private_directory(root)
        return root

    def _imports_root(self, project_id: str) -> Path:
        project = self.projects.project_dir(project_id)
        root = (project / "conversation_imports").resolve()
        self._ensure_relative(project, root, "conversation imports")
        return root

    @staticmethod
    def _messages_path(directory: Path) -> Path:
        path = (directory / "messages.jsonl").resolve()
        ConversationStore._ensure_relative(directory, path, "messages.jsonl")
        return path

    @staticmethod
    def _read_metadata(
        path: Path,
        *,
        expected_project_id: str,
        expected_conversation_id: str,
    ) -> ConversationMetadata:
        metadata = ConversationMetadata.model_validate(ConversationStore._read_json_object(path))
        if (
            metadata.project_id != expected_project_id
            or metadata.conversation_id != expected_conversation_id
        ):
            raise ValueError("conversation metadata identity mismatch")
        return metadata

    @staticmethod
    def _read_attachment_manifest(path: Path) -> ConversationAttachmentManifest:
        if not path.exists():
            raise FileNotFoundError("conversation attachment artifact not found")
        return ConversationAttachmentManifest.model_validate(
            ConversationStore._read_json_object(path)
        )

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {path.name}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"JSON object required: {path.name}")
        return loaded

    @staticmethod
    def _verify_file(path: Path, *, expected_sha256: str, expected_size: int) -> None:
        if not path.exists() or not path.is_file() or path.is_symlink():
            raise ValueError("attachment artifact content is unavailable or unsafe")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
                size += len(chunk)
                digest.update(chunk)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError("attachment artifact content hash mismatch")

    @staticmethod
    def _sha256_json(payload: Any) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _attachment_ids_from_import(payload: dict[str, Any]) -> list[str]:
        raw = payload.get("attachment_ids")
        if isinstance(raw, list):
            return [str(item or "").strip() for item in raw if str(item or "").strip()]
        attachments = payload.get("attachments")
        if not isinstance(attachments, list):
            return []
        result: list[str] = []
        for item in attachments:
            artifact_id = item.get("artifact_id") if isinstance(item, dict) else item
            clean = str(artifact_id or "").strip()
            if clean:
                result.append(clean)
        return result

    @staticmethod
    def _clean_id(value: str, label: str) -> str:
        clean = str(value or "").strip()
        if not clean or len(clean) > 128:
            raise ValueError(f"{label} is required and must be at most 128 characters")
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in clean):
            raise ValueError(f"{label} contains unsupported characters")
        if clean in {".", ".."}:
            raise ValueError(f"{label} contains unsupported characters")
        return clean

    @staticmethod
    def _clean_title(value: str) -> str:
        clean = str(value or "").strip() or "New conversation"
        return clean[:200]

    @staticmethod
    def _clean_original_name(value: str) -> str:
        clean = str(value or "").replace("\\", "/").split("/")[-1].strip()
        clean = clean.replace("\x00", "")[:255]
        return clean or "attachment"

    @staticmethod
    def _clean_media_type(value: str) -> str:
        clean = str(value or "").strip()[:255]
        return clean or "application/octet-stream"

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _ensure_relative(parent: Path, child: Path, label: str) -> None:
        if not child.is_relative_to(parent):
            raise ValueError(f"{label} escapes base directory")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform/filesystem dependent.
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _ensure_private_directory(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:  # pragma: no cover - platform/filesystem dependent.
            pass
