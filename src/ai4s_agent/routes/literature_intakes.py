from __future__ import annotations

from flask import Flask, after_this_request, jsonify, request
from pydantic import ValidationError

from ai4s_agent.literature_intake import LiteratureIntakeService


def register_literature_intake_routes(
    app: Flask,
    *,
    intakes: LiteratureIntakeService,
) -> None:
    def no_store() -> None:
        @after_this_request
        def add_no_store(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

    @app.post(
        "/api/projects/<project_id>/conversations/<conversation_id>/literature-intakes"
    )
    def register_and_submit_literature_intake(
        project_id: str,
        conversation_id: str,
    ):
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        try:
            result = intakes.register_and_submit(
                project_id=project_id,
                conversation_id=conversation_id,
                request_id=str(payload.get("request_id") or ""),
                parser_profile=str(payload.get("parser_profile") or "pdfplumber_local"),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **result}), 200 if result["idempotent"] else 201

    @app.get("/api/projects/<project_id>/literature-intakes/<intake_id>")
    def get_literature_intake(project_id: str, intake_id: str):
        no_store()
        try:
            result = intakes.get(project_id=project_id, intake_id=intake_id)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **result})

    @app.get("/api/projects/<project_id>/literature-intakes")
    def list_literature_intakes(project_id: str):
        no_store()
        try:
            items = intakes.list(
                project_id=project_id,
                conversation_id=str(request.args.get("conversation_id") or ""),
            )
        except (OSError, ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "literature_intakes": items})

    @app.post("/api/projects/<project_id>/literature-intakes/<intake_id>/approve")
    def approve_literature_intake(project_id: str, intake_id: str):
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        actor = str(payload.get("actor") or "").strip()
        if not actor:
            return jsonify({"ok": False, "error": "actor required"}), 400
        try:
            result = intakes.approve(
                project_id=project_id,
                intake_id=intake_id,
                actor=actor,
                note=str(payload.get("note") or ""),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **result})


__all__ = ["register_literature_intake_routes"]
