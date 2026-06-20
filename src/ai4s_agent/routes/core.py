from __future__ import annotations

from flask import Flask, jsonify, render_template

from ai4s_agent.schemas import GateName


def register_core_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        return render_template("index.html", gate_names=[gate.value for gate in GateName])

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})
