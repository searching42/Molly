from __future__ import annotations

from flask import Flask, after_this_request, jsonify, request
from pydantic import ValidationError

from ai4s_agent.conversation_store import ConversationStore


def register_conversation_routes(
    app: Flask,
    *,
    conversations: ConversationStore,
    max_attachment_bytes_default: int,
) -> None:
    def no_store() -> None:
        @after_this_request
        def add_no_store(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

    @app.post("/api/projects/<project_id>/conversations")
    def create_conversation(project_id: str):
        no_store()
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        try:
            metadata, created = conversations.create_conversation(
                project_id,
                title=str(payload.get("title") or "New conversation"),
                conversation_id=str(payload.get("conversation_id") or ""),
            )
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "created": created,
                "conversation": metadata.model_dump(mode="json"),
            }
        ), 201 if created else 200

    @app.get("/api/projects/<project_id>/conversations")
    def list_conversations(project_id: str):
        no_store()
        try:
            items = conversations.list_conversations(project_id)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "conversations": [item.model_dump(mode="json") for item in items],
            }
        )

    @app.get("/api/projects/<project_id>/conversations/<conversation_id>")
    def get_conversation(project_id: str, conversation_id: str):
        no_store()
        try:
            metadata = conversations.get_conversation(project_id, conversation_id)
            messages, recovered_tail = conversations.list_messages(project_id, conversation_id)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "conversation": metadata.model_dump(mode="json"),
                "messages": [item.model_dump(mode="json") for item in messages],
                "recovered_tail": recovered_tail,
            }
        )

    @app.delete("/api/projects/<project_id>/conversations/<conversation_id>")
    def delete_conversation(project_id: str, conversation_id: str):
        no_store()
        try:
            metadata = conversations.delete_conversation(project_id, conversation_id)
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except (ValidationError, ValueError):
            return jsonify({"ok": False, "error": "invalid conversation identifier"}), 400
        return jsonify(
            {
                "ok": True,
                "deleted": True,
                "conversation_id": metadata.conversation_id,
            }
        )

    @app.post("/api/projects/<project_id>/conversations/<conversation_id>/messages")
    def append_conversation_message(project_id: str, conversation_id: str):
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        attachment_ids = payload.get("attachment_ids", [])
        if not isinstance(attachment_ids, list):
            return jsonify({"ok": False, "error": "attachment_ids must be a list"}), 400
        try:
            message, idempotent, recovered_tail = conversations.append_message(
                project_id,
                conversation_id,
                role=str(payload.get("role") or ""),
                content=str(payload.get("content") or ""),
                attachment_ids=[str(item or "") for item in attachment_ids],
                client_message_id=str(payload.get("client_message_id") or ""),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "idempotent": idempotent,
                "recovered_tail": recovered_tail,
                "message": message.model_dump(mode="json"),
            }
        ), 200 if idempotent else 201

    @app.post("/api/projects/<project_id>/conversations/<conversation_id>/attachments")
    def upload_conversation_attachments(project_id: str, conversation_id: str):
        no_store()
        try:
            conversations.get_conversation(project_id, conversation_id)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files:
            return jsonify({"ok": False, "error": "at least one attachment file is required"}), 400
        if len(files) > 20:
            return jsonify({"ok": False, "error": "at most 20 files may be uploaded at once"}), 400
        max_bytes = int(
            app.config.get(
                "AI4S_MAX_CONVERSATION_ATTACHMENT_BYTES",
                max_attachment_bytes_default,
            )
            or max_attachment_bytes_default
        )
        uploaded = []
        remaining_bytes = max_bytes
        try:
            for item in files:
                if not item.filename:
                    raise ValueError("attachment filename is required")
                attachment = conversations.register_attachment(
                    project_id,
                    stream=item.stream,
                    original_name=item.filename,
                    media_type=item.mimetype or "application/octet-stream",
                    max_bytes=remaining_bytes,
                )
                uploaded.append(attachment)
                remaining_bytes -= attachment.size_bytes
        except (ValidationError, ValueError) as exc:
            status = 413 if "size limit" in str(exc) else 400
            return jsonify({"ok": False, "error": str(exc)}), status
        return jsonify(
            {
                "ok": True,
                "attachments": [item.model_dump(mode="json") for item in uploaded],
            }
        ), 201

    @app.get("/api/projects/<project_id>/conversation-attachments/<artifact_id>")
    def get_conversation_attachment(project_id: str, artifact_id: str):
        no_store()
        try:
            attachment = conversations.resolve_attachment(project_id, artifact_id)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "attachment": attachment.model_dump(mode="json")})

    @app.post("/api/projects/<project_id>/conversations/import-local-storage")
    def import_local_storage_conversation(project_id: str):
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return jsonify({"ok": False, "error": "messages must be a list"}), 400
        try:
            receipt = conversations.import_local_storage(
                project_id,
                import_id=str(payload.get("import_id") or ""),
                messages=messages,
                title=str(payload.get("title") or "Imported conversation"),
                conversation_id=str(payload.get("conversation_id") or ""),
            )
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **receipt})

    @app.post(
        "/api/projects/<project_id>/conversations/<conversation_id>/execution-requests"
    )
    def freeze_conversation_execution_request(project_id: str, conversation_id: str):
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        selected_message_ids = payload.get("selected_message_ids")
        user_parameters = payload.get("user_parameters", {})
        if not isinstance(selected_message_ids, list):
            return jsonify({"ok": False, "error": "selected_message_ids must be a list"}), 400
        if not isinstance(user_parameters, dict):
            return jsonify({"ok": False, "error": "user_parameters must be an object"}), 400
        try:
            frozen = conversations.freeze_execution_request(
                project_id,
                conversation_id,
                selected_message_ids=[str(item or "") for item in selected_message_ids],
                task_type=str(payload.get("task_type") or ""),
                model_profile_id=str(payload.get("model_profile_id") or ""),
                user_parameters=user_parameters,
                client_request_id=str(payload.get("client_request_id") or ""),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution_request": frozen.model_dump(mode="json")}), 201

    @app.get(
        "/api/projects/<project_id>/conversations/<conversation_id>/execution-requests/<request_id>"
    )
    def get_conversation_execution_request(
        project_id: str,
        conversation_id: str,
        request_id: str,
    ):
        no_store()
        try:
            frozen = conversations.get_frozen_execution_request(
                project_id,
                conversation_id,
                request_id,
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution_request": frozen.model_dump(mode="json")})
