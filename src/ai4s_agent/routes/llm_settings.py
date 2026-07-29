from __future__ import annotations

from collections.abc import Callable

from flask import Flask, after_this_request, jsonify, request

from ai4s_agent.llm_provider import LLMProviderManager
from ai4s_agent.llm_settings import LLMSettingsStore


def register_llm_settings_routes(
    app: Flask,
    *,
    settings: LLMSettingsStore,
    providers: LLMProviderManager,
    on_change: Callable[[], None] | None = None,
) -> None:
    def no_store() -> None:
        @after_this_request
        def add_no_store(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

    @app.get("/api/settings/llm")
    def get_llm_settings():
        no_store()
        return jsonify({"ok": True, **settings.public_state()})

    @app.patch("/api/settings/llm")
    def update_llm_settings():
        no_store()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        try:
            config = settings.patch(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if on_change is not None:
            on_change()
        return jsonify(
            {
                "ok": True,
                **settings.public_state(),
                "settings_scope": "user",
                "settings_file": settings.path.name,
                "model": config.model,
            }
        )

    @app.delete("/api/settings/llm/api-key")
    def delete_llm_api_key():
        no_store()
        try:
            settings.delete_api_key()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if on_change is not None:
            on_change()
        return jsonify({"ok": True, **settings.public_state()})

    @app.post("/api/settings/llm/probe")
    def probe_llm_settings():
        """Send one minimal OpenAI-compatible request without exposing secrets."""

        no_store()
        config = settings.read()
        if config is None:
            return jsonify(
                {
                    "ok": False,
                    "error_code": "llm_settings_unavailable",
                    "error": "LLM settings or the selected API key source are unavailable.",
                }
            ), 400
        try:
            with providers.lease(config) as provider:
                provider.complete_text(
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply only with OK.",
                        }
                    ],
                    prompt_version="llm-settings-connection-probe.v1",
                )
        except Exception:
            app.logger.warning("LLM settings connection probe failed", exc_info=True)
            return jsonify(
                {
                    "ok": False,
                    "error_code": "llm_connection_failed",
                    "error": "The configured LLM endpoint, model, or API key could not be verified.",
                }
            ), 409
        return jsonify(
            {
                "ok": True,
                "probe": {
                    "status": "available",
                    "provider": config.provider,
                    "model": config.model,
                    "request_kind": "minimal_chat_completion",
                },
            }
        )
