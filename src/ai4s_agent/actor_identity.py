from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import current_app, g, has_app_context, has_request_context


@dataclass(frozen=True)
class ActorContext:
    actor: str
    source: str
    required: bool


def resolve_actor(request: Any, *, required: bool = False) -> ActorContext:
    header_actor = str(request.headers.get("X-Actor") or "").strip()
    if header_actor:
        return ActorContext(actor=header_actor, source="header:X-Actor", required=required)

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        for key in ("actor", "approved_by", "revoked_by", "confirmed_by"):
            value = str(payload.get(key) or "").strip()
            if value:
                return ActorContext(actor=value, source=f"json:{key}", required=required)

    form = getattr(request, "form", None)
    if form is not None:
        for key in ("actor", "approved_by", "revoked_by", "confirmed_by"):
            value = str(form.get(key) or "").strip()
            if value:
                return ActorContext(actor=value, source=f"form:{key}", required=required)

    query_actor = str(request.args.get("actor") or "").strip()
    if query_actor:
        return ActorContext(actor=query_actor, source="query:actor", required=required)

    return ActorContext(actor="", source="missing", required=required)


def resolve_authenticated_actor(request: Any, *, required: bool = False) -> ActorContext:
    """Resolve only a server-owned authenticated principal.

    ``X-Actor`` and all body/query/form values are deliberately excluded.  A
    deployment may install authenticated middleware that writes the principal
    to ``flask.g.ai4s_authenticated_principal`` or the private WSGI environ key
    ``ai4s.authenticated_principal``.  Local single-user deployments may set a
    fixed owner in ``AI4S_AGENT_AUTHORIZATION_OWNER``.  With none configured,
    authorization is unavailable by default.
    """

    if has_request_context():
        middleware_actor = str(
            getattr(g, "ai4s_authenticated_principal", "") or ""
        ).strip()
        if middleware_actor:
            return ActorContext(
                actor=middleware_actor,
                source="flask.g:ai4s_authenticated_principal",
                required=required,
            )

    environ = getattr(request, "environ", None)
    if isinstance(environ, dict):
        environ_actor = str(
            environ.get("ai4s.authenticated_principal") or ""
        ).strip()
        if environ_actor:
            return ActorContext(
                actor=environ_actor,
                source="wsgi.environ:ai4s.authenticated_principal",
                required=required,
            )

    if has_app_context():
        configured_actor = str(
            current_app.config.get("AI4S_AGENT_AUTHORIZATION_OWNER") or ""
        ).strip()
        if configured_actor:
            return ActorContext(
                actor=configured_actor,
                source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
                required=required,
            )

    return ActorContext(actor="", source="missing", required=required)
