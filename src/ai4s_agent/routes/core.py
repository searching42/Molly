from __future__ import annotations

from flask import Flask, after_this_request, jsonify, render_template

from ai4s_agent.schemas import GateName


def register_core_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        @after_this_request
        def prevent_local_token_cache(response):
            response.headers["Cache-Control"] = "no-store"
            return response

        return render_template(
            "index.html",
            gate_names=[gate.value for gate in GateName],
            local_session_token=app.config["MOLLY_LOCAL_SESSION_TOKEN"],
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})
