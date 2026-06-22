from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        for key in ("actor", "approved_by", "revoked_by"):
            value = str(payload.get(key) or "").strip()
            if value:
                return ActorContext(actor=value, source=f"json:{key}", required=required)

    form = getattr(request, "form", None)
    if form is not None:
        for key in ("actor", "approved_by", "revoked_by"):
            value = str(form.get(key) or "").strip()
            if value:
                return ActorContext(actor=value, source=f"form:{key}", required=required)

    query_actor = str(request.args.get("actor") or "").strip()
    if query_actor:
        return ActorContext(actor=query_actor, source="query:actor", required=required)

    return ActorContext(actor="", source="missing", required=required)
